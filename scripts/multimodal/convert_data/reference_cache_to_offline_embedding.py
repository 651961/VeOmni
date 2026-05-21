# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Convert a reference SFT pth cache directory into VeOmni offline_embedding
parquet shards.

Use case: substitute VeOmni's own offline_embedding output with a cache
produced by an external reference training stack (one ``<i>.pth`` per
sample, numbered 0..N-1). Lets us drive ``offline_training`` from a cache
whose text/vision encoder path is known-good, isolating the DiT training
loop from upstream-of-cache differences.

Input pth schema (per file, ``torch.load`` returns a 3-tuple):
    tuple[0]: dict with VAE artifacts. Required keys:
        ``input_latents``  Tensor[(1, 16, H/8, W/8)]   target image clean
                                                       VAE latent
        ``edit_latents``   list[Tensor[(1, 16, He/8, We/8)]]  per ref-image
                                                              clean VAE latent
    tuple[1]: dict with positive-prompt encoder output. Required keys:
        ``prompt_emb``      Tensor[(1, L, 3584)]
        ``prompt_emb_mask`` Tensor[(1, L)]
    tuple[2]: (negative prompt — ignored)

Output: a directory containing ``rank_0_shard_0.parquet`` (or sharded
across multiple files) with the four-field cache contract enforced by
:class:`OfflineEmbeddingSaver` (see
``veomni/data/diffusion/offline_embedding_saver.py``):
    latents          : Tensor[(1, 16, H/8, W/8)]    bf16
    edit_latents     : list[Tensor[(1, 16, He/8, We/8)]]  bf16
    prompt_emb       : Tensor[(1, L, 3584)]  bf16
    prompt_emb_mask  : Tensor[(1, L)]        int64

Row order in the output matches the pth file's numeric stem (0.pth -> row
0, 1.pth -> row 1, ...). Files are processed in numeric order, not the
default ``sorted()`` lexical order, so ``10.pth`` does not slot in before
``2.pth``.
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import torch
from tqdm import tqdm

# Make ``veomni`` importable when this script is invoked directly.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from veomni.data.diffusion.offline_embedding_saver import OfflineEmbeddingSaver


def _numeric_sorted_pth(input_dir: str) -> List[str]:
    """Return ``*.pth`` filenames sorted by their integer stem (0,1,2,...,10,...).

    A plain ``sorted()`` would lexically place ``10.pth`` before ``2.pth``.
    """
    files = [f for f in os.listdir(input_dir) if f.endswith(".pth")]
    try:
        files.sort(key=lambda f: int(os.path.splitext(f)[0]))
    except ValueError as exc:
        raise ValueError(
            f"Non-integer pth stem under {input_dir}; refusing to guess an order. "
            "Expected files named '<i>.pth' with integer i."
        ) from exc
    return files


def _extract_one(pth_path: str) -> Dict[str, Any]:
    """Load one pth and project it into the VeOmni cache contract."""
    sample = torch.load(pth_path, map_location="cpu", weights_only=False)
    if not (isinstance(sample, tuple) and len(sample) >= 2):
        raise ValueError(
            f"{pth_path}: expected a tuple of length >= 2, got {type(sample).__name__} "
            f"len={len(sample) if hasattr(sample, '__len__') else 'n/a'}"
        )

    vae_dict, prompt_dict = sample[0], sample[1]
    if "input_latents" not in vae_dict:
        raise KeyError(f"{pth_path}: tuple[0] missing 'input_latents'")
    if "edit_latents" not in vae_dict:
        raise KeyError(f"{pth_path}: tuple[0] missing 'edit_latents'")
    if "prompt_emb" not in prompt_dict or "prompt_emb_mask" not in prompt_dict:
        raise KeyError(f"{pth_path}: tuple[1] missing 'prompt_emb' or 'prompt_emb_mask'")

    latents = vae_dict["input_latents"].detach().cpu().to(torch.bfloat16)
    edit_latents = [t.detach().cpu().to(torch.bfloat16) for t in vae_dict["edit_latents"]]
    prompt_emb = prompt_dict["prompt_emb"].detach().cpu().to(torch.bfloat16)
    prompt_emb_mask = prompt_dict["prompt_emb_mask"].detach().cpu().to(torch.int64)

    # Shape sanity. Surface a clear error here rather than letting it fail
    # downstream inside the trainer.
    if latents.ndim != 4 or latents.shape[0] != 1 or latents.shape[1] != 16:
        raise ValueError(f"{pth_path}: latents must be (1, 16, H/8, W/8), got {tuple(latents.shape)}")
    if not isinstance(edit_latents, list) or len(edit_latents) == 0:
        raise ValueError(f"{pth_path}: edit_latents must be a non-empty list")
    for i, e in enumerate(edit_latents):
        if e.ndim != 4 or e.shape[0] != 1 or e.shape[1] != 16:
            raise ValueError(
                f"{pth_path}: edit_latents[{i}] must be (1, 16, He/8, We/8), got {tuple(e.shape)}"
            )
    if prompt_emb.ndim != 3 or prompt_emb.shape[0] != 1 or prompt_emb.shape[2] != 3584:
        raise ValueError(f"{pth_path}: prompt_emb must be (1, L, 3584), got {tuple(prompt_emb.shape)}")
    if prompt_emb_mask.ndim != 2 or prompt_emb_mask.shape[0] != 1 or prompt_emb_mask.shape[1] != prompt_emb.shape[1]:
        raise ValueError(
            f"{pth_path}: prompt_emb_mask must be (1, L) with L matching prompt_emb, "
            f"got {tuple(prompt_emb_mask.shape)} vs {tuple(prompt_emb.shape)}"
        )

    return {
        "latents": latents,
        "edit_latents": edit_latents,
        "prompt_emb": prompt_emb,
        "prompt_emb_mask": prompt_emb_mask,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a reference SFT pth cache (one '<i>.pth' per sample) into "
            "VeOmni offline_embedding parquet shards."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the numbered '.pth' cache files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Destination directory for the parquet shards (created if missing).",
    )
    parser.add_argument(
        "--shard_num",
        type=int,
        default=1,
        help="Number of parquet shards to split the output into (default: 1).",
    )
    args = parser.parse_args()

    files = _numeric_sorted_pth(args.input_dir)
    if not files:
        raise SystemExit(f"No '.pth' files under {args.input_dir}")
    print(f"Found {len(files)} pth files under {args.input_dir}; writing to {args.output_dir}")

    saver = OfflineEmbeddingSaver(
        save_path=args.output_dir,
        dataset_length=len(files),
        shard_num=args.shard_num,
        dp_rank=0,
        dp_size=1,
    )

    for fname in tqdm(files, desc="convert pth -> parquet"):
        item = _extract_one(os.path.join(args.input_dir, fname))
        saver.save(item)
    saver.save_last()

    print(f"Done. Output: {args.output_dir}")


if __name__ == "__main__":
    main()

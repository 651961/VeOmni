# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Build the offline embedding cache for Qwen-Image-Edit-2511 SFT.

Reads the raw-bytes parquet emitted by
``scripts/multimodal/convert_data/qwen_image_edit.py``, runs the condition
model (VAE + Qwen2.5-VL text encoder) per sample, and writes the
``OfflineEmbeddingSaver`` parquet that the offline-training path consumes.

For multi-rank execution wrap this with the usual launcher (``torchrun`` /
``train.sh``) and pass ``--dp_rank`` / ``--dp_size``. The saver writes one
shard per rank under ``rank_{r}_shard_{i}.parquet``.

Single-rank example (cache the 731-sample dev set on one GPU)::

    CUDA_VISIBLE_DEVICES=0 python scripts/multimodal/build_offline_embedding_cache.py \\
        --model_path /models/Qwen-Image-Edit-2511 \\
        --input_parquet /datasets/.../train_parquet \\
        --output_dir /datasets/.../train_offline_embedding_veomni \\
        --source_name Qwen-Image-Edit-2511

The output format matches the cache contract in
``veomni/data/diffusion/offline_embedding_saver.py``:

    latents          (1, 16, H/8, W/8) bf16
    edit_latents     list[(1, 16, He/8, We/8)] bf16
    prompt_emb       (1, L_actual, 3584)       bf16
    prompt_emb_mask  (1, L_actual)             int64

(All un-padded; the offline-training collator handles batch padding.)
"""

from __future__ import annotations

import argparse
import glob
import io
import os
from typing import List, Sequence

import torch
from PIL import Image
from tqdm import tqdm


def _load_parquet_rows(input_parquet: str) -> List[dict]:
    """Read one or more parquet files into an in-memory list of dicts.

    Each row carries ``prompt`` (str), ``image_bytes`` (bytes),
    ``edit_image_bytes`` (list[bytes]), ``source`` (str).
    """
    import pyarrow.parquet as pq

    if os.path.isdir(input_parquet):
        # Numeric sort, not lexical: "10.parquet" must come AFTER "9.parquet"
        # (otherwise it lands right after "1.parquet" and the sample order
        # in the output cache no longer matches metadata.jsonl, which
        # silently misaligns ours-vs-reference comparisons starting at the
        # boundary between the first shard and "10.parquet").
        def _shard_index(path: str) -> tuple:
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                return (0, int(stem))
            except ValueError:
                return (1, stem)

        files = sorted(glob.glob(os.path.join(input_parquet, "*.parquet")), key=_shard_index)
    else:
        files = [input_parquet]
    if not files:
        raise RuntimeError(f"No parquet files found at {input_parquet}")
    rows: List[dict] = []
    for f in files:
        tbl = pq.read_table(f)
        for row in tbl.to_pylist():
            rows.append(row)
    return rows


def _bytes_to_pil(b) -> Image.Image:
    """Decode bytes (or list-of-bytes single-element) to a PIL.Image (RGB)."""
    if isinstance(b, (list, tuple)):
        assert len(b) == 1
        b = b[0]
    return Image.open(io.BytesIO(b)).convert("RGB")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", type=str, required=True, help="Root of the released QIE-2511 model dir.")
    parser.add_argument(
        "--input_parquet",
        type=str,
        required=True,
        help="Path to a parquet file or a directory of parquet shards built by qwen_image_edit.py.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Where to write the embedding-cache parquet.")
    parser.add_argument("--source_name", type=str, default="Qwen-Image-Edit-2511")
    parser.add_argument("--dp_rank", type=int, default=0)
    parser.add_argument("--dp_size", type=int, default=1)
    parser.add_argument("--shard_num", type=int, default=1, help="Shards written per rank.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of samples (smoke-test).")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but not available.")

    from veomni.data.diffusion.offline_embedding_saver import OfflineEmbeddingSaver, pack_condition_for_save
    from veomni.models.diffusers.qwen_image_edit_2511.qwen_image_edit_condition.configuration_qwen_image_edit_condition import (  # noqa: E501
        QwenImageEditConditionModelConfig,
    )
    from veomni.models.diffusers.qwen_image_edit_2511.qwen_image_edit_condition.modeling_qwen_image_edit_condition import (  # noqa: E501
        QwenImageEditConditionModel,
    )

    print(f"[build_offline_embedding_cache] loading parquet rows from {args.input_parquet}", flush=True)
    rows = _load_parquet_rows(args.input_parquet)
    if args.limit is not None:
        rows = rows[: args.limit]
    total = len(rows)
    # Round-robin shard the dataset across ranks (matches the convention used
    # by the older saver: sample i lands on rank i % dp_size at row i //
    # dp_size).
    my_rows = [r for i, r in enumerate(rows) if i % args.dp_size == args.dp_rank]
    print(f"[rank {args.dp_rank}/{args.dp_size}] {total} total rows, {len(my_rows)} on this rank", flush=True)

    print(f"[rank {args.dp_rank}] loading condition model from {args.model_path}", flush=True)
    cfg = QwenImageEditConditionModelConfig(base_model_path=args.model_path)
    cm = QwenImageEditConditionModel(cfg, meta_init=False)
    cm.vae = cm.vae.to(args.device)
    cm.text_encoder.model = cm.text_encoder.model.to(args.device)

    saver = OfflineEmbeddingSaver(
        save_path=args.output_dir,
        dataset_length=len(my_rows),
        shard_num=args.shard_num,
        dp_rank=args.dp_rank,
        dp_size=args.dp_size,
    )

    for row in tqdm(my_rows, desc=f"rank {args.dp_rank}"):
        prompt = row["prompt"]
        target_pil = _bytes_to_pil(row["image_bytes"])
        edit_pils = [_bytes_to_pil(b) for b in row["edit_image_bytes"]]
        sample_images = {"image": [target_pil], "edit_image": edit_pils}

        with torch.no_grad():
            cond = cm.get_condition(inputs=[prompt], images=[sample_images])
        saver.save(pack_condition_for_save(cond, sample_idx=0))

    saver.save_last()
    print(f"[rank {args.dp_rank}] wrote shards to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

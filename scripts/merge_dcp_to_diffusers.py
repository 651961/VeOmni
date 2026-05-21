import argparse
import gc
import json
import os
import shutil
from collections import OrderedDict
from typing import Optional, Union

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import Metadata

from veomni.checkpoint.dcp_checkpointer import (
    _get_sharding_plan,
    _normalize_key,
    _process_shard,
    get_dtype_size,
)
from veomni.utils import helper


# Diffusers filename conventions (mirrors diffusers.utils.constants — duplicated
# here so this script doesn't pull in the diffusers package).
SAFE_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
WEIGHTS_NAME = "diffusion_pytorch_model.bin"
SAFE_WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.bin.index.json"


logger = helper.create_logger(__name__)


def _get_sharding_plan_from_index(
    checkpoint_path: Union[str, os.PathLike],
    reference_index_path: Union[str, os.PathLike],
    save_dtype: Optional[Union[str, torch.dtype]] = None,
):
    """
    Build a sharding plan that mirrors the per-key assignment in
    ``reference_index_path`` (a diffusers ``*.safetensors.index.json``). Every
    tensor lands in the same shard file as the untuned release, so the produced
    checkpoint is layout-identical, not just functionally equivalent.

    Returns:
        shards: List of OrderedDict[hf_key, dcp_key], one per output shard.
        shard_filenames: Parallel list of output filenames (taken verbatim from
            the reference index, e.g. ``diffusion_pytorch_model-00001-of-00005.safetensors``).
        total_size: Sum of tensor sizes in bytes at ``save_dtype``.
        all_dcp_keys: All DCP keys that map to a model tensor.
    """
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    if not isinstance(metadata, Metadata):
        raise ValueError(f"Invalid metadata format in {checkpoint_path}")

    # Walk DCP metadata once, build hf_key -> (dcp_key, byte_size).
    hf_to_dcp: "OrderedDict[str, str]" = OrderedDict()
    hf_to_size: "dict[str, int]" = {}
    all_dcp_keys = []

    for key, tensor_meta in metadata.state_dict_metadata.items():
        hf_key = _normalize_key(key)
        if hf_key is None:
            continue

        if save_dtype:
            dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype
        else:
            if not hasattr(tensor_meta.properties, "dtype"):
                raise ValueError(
                    f"Cannot determine dtype for tensor '{key}': metadata does not contain dtype information"
                )
            dtype = tensor_meta.properties.dtype

        numel = 1
        for dim in tensor_meta.size:
            numel *= dim
        byte_size = numel * get_dtype_size(dtype)

        hf_to_dcp[hf_key] = key
        hf_to_size[hf_key] = byte_size
        all_dcp_keys.append(key)

    # Load the reference weight_map.
    with open(reference_index_path, "r", encoding="utf-8") as f:
        ref_index = json.load(f)
    ref_weight_map = ref_index["weight_map"]

    # Schema check: reference must be a subset of DCP. Surface drift loudly
    # — a silent mismatch here would mean the released loader rejects the
    # produced checkpoint at runtime.
    missing = [k for k in ref_weight_map if k not in hf_to_dcp]
    if missing:
        raise ValueError(
            f"Reference index references {len(missing)} key(s) not found in DCP "
            f"checkpoint. First 5: {missing[:5]}. The fine-tuned model schema "
            f"diverges from the untuned release; cannot preserve shard layout."
        )
    extra = [k for k in hf_to_dcp if k not in ref_weight_map]
    if extra:
        logger.warning(
            f"DCP checkpoint has {len(extra)} key(s) absent from the reference index; "
            f"first 5: {extra[:5]}. These will be dropped to keep layout identical."
        )

    # Group keys by shard filename. Sort shard filenames to fix output order
    # (the reference uses zero-padded ``-NNNNN-of-NNNNN-`` so lexicographic
    # sort is also numeric).
    shard_filename_to_keys: "OrderedDict[str, OrderedDict[str, str]]" = OrderedDict()
    for filename in sorted(set(ref_weight_map.values())):
        shard_filename_to_keys[filename] = OrderedDict()

    total_size = 0
    # Iterate ref_weight_map in its own order so within-shard key order matches
    # the reference (the reference json is sort_keys=True, so this is alphabetical).
    for hf_key, filename in ref_weight_map.items():
        shard_filename_to_keys[filename][hf_key] = hf_to_dcp[hf_key]
        total_size += hf_to_size[hf_key]

    shards = list(shard_filename_to_keys.values())
    shard_filenames = list(shard_filename_to_keys.keys())
    return shards, shard_filenames, total_size, all_dcp_keys


@torch.no_grad()
def save_model_weights(
    output_dir: Union[str, os.PathLike],
    checkpoint_path: Union[str, os.PathLike],
    save_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    shard_size: int = 10_000_000_000,
    safe_serialization: bool = True,
    config_path: Optional[str] = None,
    reference_index_path: Optional[str] = None,
) -> None:
    """Convert DCP checkpoint to diffusers format with shard-by-shard processing (memory-efficient).

    If ``reference_index_path`` is provided (or auto-discoverable under
    ``config_path``), tensors are routed to shards using the reference index as
    a per-key routing table so the output layout matches the untuned release
    byte-for-byte at the shard-assignment level. Otherwise falls back to
    size-based greedy packing of alphabetically sorted keys (does NOT preserve
    the diffusers ``save_pretrained`` traversal order — output is loadable but
    keys may land in different shards than the untuned release).
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving model weights to {output_dir}")
    logger.info(
        f"Format: {'safetensors' if safe_serialization else 'pytorch'}, dtype={save_dtype}, shard_size={shard_size}"
    )

    # Auto-discover reference index next to config_path.
    if reference_index_path is None and config_path is not None:
        candidate = os.path.join(config_path, SAFE_WEIGHTS_INDEX_NAME)
        if os.path.isfile(candidate):
            reference_index_path = candidate
            logger.info(f"Using reference index for shard routing: {reference_index_path}")

    if reference_index_path is not None:
        logger.info("Analyzing DCP metadata and routing by reference index...")
        shards, shard_filenames, total_size, all_dcp_keys = _get_sharding_plan_from_index(
            checkpoint_path, reference_index_path, save_dtype
        )
    else:
        logger.info("Analyzing DCP metadata and planning shards by size...")
        shards, total_size, all_dcp_keys = _get_sharding_plan(checkpoint_path, shard_size, save_dtype)
        shard_filenames = None  # generated below from shard_idx

    logger.info(f"Found {len(all_dcp_keys)} model tensors, total size: ~{total_size / 1e9:.2f}GB")
    logger.info(f"Split into {len(shards)} shards")

    if len(shards) == 0:
        logger.warning("No model weights found! Check if checkpoint path is correct and contains 'model.' keys.")
        return

    # Process each shard
    weight_map = OrderedDict()
    num_shards = len(shards)

    for shard_idx, shard_keys in enumerate(shards):
        if shard_filenames is not None:
            filename = shard_filenames[shard_idx]
        else:
            weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
            if num_shards == 1:
                filename = weights_name
            else:
                prefix, extension = weights_name.rsplit(".", maxsplit=1)
                filename = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"

        save_path = os.path.join(output_dir, filename)
        logger.info(f"Processing shard {shard_idx + 1}/{num_shards}: {filename} ({len(shard_keys)} tensors)")

        processed_dict = _process_shard(shard_keys, checkpoint_path, save_dtype)

        # Save shard
        if safe_serialization:
            save_file(processed_dict, save_path, metadata={"format": "pt"})
        else:
            torch.save(processed_dict, save_path)

        del processed_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for hf_key in shard_keys.keys():
            weight_map[hf_key] = filename

    # Save index file for multi-shard checkpoints
    if num_shards > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
        logger.info(f"Saved index file to {index_file}")

    logger.info("Weight conversion complete.")

    # Copy diffusers config.json from the source transformer dir so the output
    # is a self-contained diffusers checkpoint (matches the layout of the
    # untuned /models/Qwen-Image-Edit-2511/transformer/ release).
    if config_path is not None:
        src_config = os.path.join(config_path, "config.json")
        if os.path.isfile(src_config):
            dst_config = os.path.join(output_dir, "config.json")
            shutil.copyfile(src_config, dst_config)
            logger.info(f"Copied config.json from {src_config}")
        else:
            logger.warning(f"config.json not found in {config_path}, skipping config copy")


def merge_to_diffusers(
    load_dir: str,
    save_path: str,
    config_path: Optional[str] = None,
    shard_size: int = 10_000_000_000,
    reference_index_path: Optional[str] = None,
) -> None:
    """Main conversion function: load DCP from load_dir and save diffusers format to save_path."""
    save_model_weights(
        save_path,
        load_dir,
        shard_size=shard_size,
        config_path=config_path,
        reference_index_path=reference_index_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Merge DCP checkpoint to diffusers format (streaming optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--load-dir", type=str, required=True, help="Directory containing DCP checkpoint")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory for diffusers format checkpoint (default: <load-dir>/diffusers_ckpt)",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Directory containing the original diffusers transformer config.json (e.g. "
        "/models/Qwen-Image-Edit-2511/transformer). config.json is copied verbatim into "
        "--save-dir; if it also contains diffusion_pytorch_model.safetensors.index.json, "
        "that index is used as a shard routing table so output shard assignment matches "
        "the untuned release per-key.",
    )
    parser.add_argument(
        "--reference-index",
        type=str,
        default=None,
        help="Explicit path to a diffusers safetensors index json to use as the shard "
        "routing table. Overrides auto-discovery under --config-path.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10_000_000_000,
        help="Maximum shard size in bytes (default: 10GB). Ignored when a reference index is in use.",
    )
    args = parser.parse_args()

    load_dir = args.load_dir
    save_dir = os.path.join(load_dir, "diffusers_ckpt") if args.save_dir is None else args.save_dir
    config_path = args.config_path
    shard_size = args.shard_size
    reference_index_path = args.reference_index

    merge_to_diffusers(
        load_dir,
        save_dir,
        config_path,
        shard_size=shard_size,
        reference_index_path=reference_index_path,
    )


if __name__ == "__main__":
    main()

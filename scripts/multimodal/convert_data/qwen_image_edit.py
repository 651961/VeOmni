# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Build a raw-bytes parquet dataset for Qwen-Image-Edit-2511 SFT.

Reads ``metadata.jsonl`` (one row per sample) + the on-disk images under
``train/`` and emits a sharded parquet with these columns:

    prompt           : str
    image_bytes      : bytes        (target image file contents)
    edit_image_bytes : list[bytes]  (one or more reference image file contents)
    source           : str          (fixed ``"Qwen-Image-Edit-2511"``)

This parquet is the input to the offline-embedding pass (which runs the
condition model on each row to produce pickled latents / prompt embeddings)
and to the ``online_training`` data path (which fetches PIL images at step
time and runs the condition model live).

Example
-------

    python3 scripts/multimodal/convert_data/qwen_image_edit.py \\
        --dataset_base_path /path/to/example_dataset \\
        --output_dir /path/to/example_dataset/train_parquet \\
        --num_shards 32 \\
        --num_proc 32
"""

import argparse
import json
import math
import os
from typing import List, Optional, Sequence

from datasets import Dataset


SOURCE_NAME = "Qwen-Image-Edit-2511"


def _load_metadata(metadata_path: str) -> List[dict]:
    if metadata_path.endswith(".jsonl"):
        rows = []
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if metadata_path.endswith(".json"):
        with open(metadata_path, encoding="utf-8") as f:
            return json.load(f)
    if metadata_path.endswith(".csv"):
        import pandas as pd

        df = pd.read_csv(metadata_path)
        return [df.iloc[i].to_dict() for i in range(len(df))]
    raise ValueError(f"Unsupported metadata extension: {metadata_path}")


def _resolve_paths(field, train_dir: str) -> List[str]:
    if field is None:
        return []
    if isinstance(field, str):
        field = [field]
    return [p if os.path.isabs(p) else os.path.join(train_dir, p) for p in field]


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def convert(
    dataset_base_path: str,
    metadata_path: str,
    output_dir: str,
    num_shards: int = 30,
    num_proc: int = 16,
    image_key: str = "image",
    edit_image_key: str = "edit_image",
    prompt_key: str = "prompt",
    limit: Optional[int] = None,
):
    train_dir = os.path.join(dataset_base_path, "train")
    rows = _load_metadata(metadata_path)
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError(f"No rows loaded from {metadata_path}")

    os.makedirs(output_dir, exist_ok=True)
    total = len(rows)
    batch_len = math.ceil(total / num_shards)
    print(f"[qwen_image_edit] Loaded {total} rows; sharding into {num_shards} parquet files (~{batch_len} rows each).")

    meta_ds = Dataset.from_list(rows)

    index = 0
    for start in range(0, total, batch_len):
        end = min(start + batch_len, total)
        chunk = meta_ds.select(range(start, end))
        chunk_num_proc = min(num_proc, len(chunk))
        print(f"[qwen_image_edit] Building shard {index} (rows {start}:{end})")

        def process_example(example):
            image_paths = _resolve_paths(example.get(image_key), train_dir)
            assert len(image_paths) == 1, (
                f"Each row must have exactly one target image (key '{image_key}'); got {len(image_paths)}."
            )
            edit_paths = _resolve_paths(example.get(edit_image_key), train_dir)
            return {
                "prompt": str(example.get(prompt_key, "") or ""),
                "image_bytes": _read_bytes(image_paths[0]),
                "edit_image_bytes": [_read_bytes(p) for p in edit_paths],
                "source": SOURCE_NAME,
            }

        ds = chunk.map(
            process_example,
            num_proc=chunk_num_proc,
            remove_columns=chunk.column_names,
            keep_in_memory=True,
            desc=f"Processing shard {index}",
        )
        ds.to_parquet(os.path.join(output_dir, f"{index}.parquet"))
        index += 1

    print(f"[qwen_image_edit] Wrote {index} parquet shards to {output_dir}.")


def main(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset_base_path", type=str, required=True, help="Root containing 'metadata.jsonl' and 'train/'."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Override metadata file path (default: <dataset_base_path>/metadata.jsonl).",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_shards", type=int, default=30)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--edit_image_key", type=str, default="edit_image")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for smoke tests.")
    args = parser.parse_args(argv)

    metadata_path = args.metadata_path or os.path.join(args.dataset_base_path, "metadata.jsonl")
    convert(
        dataset_base_path=args.dataset_base_path,
        metadata_path=metadata_path,
        output_dir=args.output_dir,
        num_shards=args.num_shards,
        num_proc=args.num_proc,
        image_key=args.image_key,
        edit_image_key=args.edit_image_key,
        prompt_key=args.prompt_key,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

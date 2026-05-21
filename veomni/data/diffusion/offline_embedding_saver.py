# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Offline embedding-cache saver for diffusion training.

Writes the per-sample condition-model output (target VAE latent, edit VAE
latents, prompt embedding, prompt embedding mask) to a sharded parquet
dataset. Each row holds the four fields as pickled bytes - the offline
training path reads them back via the ``dit_offline`` data transform and
``pickle.loads``.

Per-shard file naming: ``rank_{dp_rank}_shard_{idx}.parquet``. Sharding is
controlled by ``shard_num`` (how many shards per rank); the buffer size
follows ``ceil(dataset_length / shard_num)`` so each shard ends up with a
similar row count.

Format contract (each row stores ``pickle.dumps(cpu_tensor_or_list)``):

    latents          : Tensor[(1, 16, H/8, W/8)]    bf16
    edit_latents     : list[Tensor[(1, 16, He/8, We/8)]]  bf16
    prompt_emb       : Tensor[(1, L_actual, 3584)]  bf16
    prompt_emb_mask  : Tensor[(1, L_actual)]        int64

Do not switch to a 5-D ``(1, 32, 1, H/8, W/8)`` latent shape - the cache
contract requires the 4-D 16-channel form (see ``modeling_qwen_image_edit_vae.py``
for the rationale).
"""

import math
import os
import pickle
from typing import Any, Dict

import torch
from datasets import Dataset


class OfflineEmbeddingSaver:
    """Per-rank buffered saver that emits sharded parquet files.

    Args:
        save_path: Output directory for the parquet shards.
        dataset_length: Total number of samples this rank will see. Used to
            size the per-shard buffer (``batch_len = ceil(N / shard_num)``)
            and to discard trailing dummy padding samples.
        shard_num: Number of shards per rank (default 1 = one shard per
            rank). Total shards = ``shard_num * dp_size``, capped by
            ``max_shard``.
        max_shard: Hard cap on the total shard count; ``shard_num`` is
            shrunk if ``dp_size * shard_num`` would exceed this.
        dp_rank: Data-parallel rank of this process (default 0 for
            single-rank).
        dp_size: Total data-parallel size (default 1).
    """

    def __init__(
        self,
        save_path: str,
        dataset_length: int = 0,
        shard_num: int = 1,
        max_shard: int = 1000,
        dp_rank: int = 0,
        dp_size: int = 1,
    ):
        if dp_size * shard_num > max_shard:
            shard_num = max(1, max_shard // dp_size)
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.shard_num = shard_num
        self.max_shard = max_shard
        self.save_path = save_path
        self.dataset_length = dataset_length
        self.batch_len = max(1, math.ceil(dataset_length / self.shard_num)) if dataset_length else 1
        self.rest_len = dataset_length
        self.index = 0
        self.buffer: list = []
        os.makedirs(self.save_path, exist_ok=True)

    @staticmethod
    def _cpu_recursive(obj: Any) -> Any:
        """Move tensors to CPU recursively, leave other types unchanged."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: OfflineEmbeddingSaver._cpu_recursive(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(OfflineEmbeddingSaver._cpu_recursive(v) for v in obj)
        return obj

    def _to_save_bytes(self, save_item: Dict[str, Any]) -> Dict[str, bytes]:
        out: Dict[str, bytes] = {}
        for key in list(save_item.keys()):
            out[key] = pickle.dumps(self._cpu_recursive(save_item[key]))
            del save_item[key]
        return out

    def save(self, save_item: Dict[str, Any]) -> None:
        """Buffer one sample; auto-flush a shard once the buffer fills."""
        if self.rest_len > 0:
            self.buffer.append(self._to_save_bytes(save_item))
            self.rest_len -= 1
        if len(self.buffer) >= self.batch_len:
            self._flush_shard()

    def save_last(self) -> None:
        """Flush any remaining buffered rows. Call once at end of iteration."""
        if self.buffer:
            self._flush_shard()

    def _flush_shard(self) -> None:
        ds = Dataset.from_list(self.buffer)
        path = os.path.join(self.save_path, f"rank_{self.dp_rank}_shard_{self.index}.parquet")
        ds.to_parquet(path)
        self.buffer = []
        self.index += 1


def maybe_unsqueeze_batch_dim(tensor_or_list, target_ndim: int):
    """If a tensor/list-element is one dim short of ``target_ndim``, prepend
    a singleton batch dim. Returns a copy of the input with the right ndim.

    The cache contract stores prompt embeddings as ``(1, L, 3584)`` and
    masks as ``(1, L)``. ``get_condition`` currently returns un-batched
    forms; this helper bridges the difference at save time.
    """
    if isinstance(tensor_or_list, list):
        return [maybe_unsqueeze_batch_dim(t, target_ndim) for t in tensor_or_list]
    if isinstance(tensor_or_list, torch.Tensor) and tensor_or_list.ndim == target_ndim - 1:
        return tensor_or_list.unsqueeze(0)
    return tensor_or_list


def pack_condition_for_save(condition_out: Dict[str, Any], sample_idx: int) -> Dict[str, Any]:
    """Take one sample out of a ``get_condition`` output and put it in the
    cache shape contract.

    ``get_condition`` returns per-batch lists; we index sample-by-sample
    and add the singleton batch dim to ``prompt_emb`` / ``prompt_emb_mask``
    where needed.

    Returns a fresh dict ready for ``OfflineEmbeddingSaver.save``.
    """
    latents = condition_out["latents"][sample_idx]
    edit_latents = condition_out["edit_latents"][sample_idx]
    prompt_emb = condition_out["prompt_emb"][sample_idx]
    prompt_emb_mask = condition_out["prompt_emb_mask"][sample_idx]
    return {
        "latents": latents,
        "edit_latents": edit_latents,
        "prompt_emb": maybe_unsqueeze_batch_dim(prompt_emb, target_ndim=3),
        "prompt_emb_mask": maybe_unsqueeze_batch_dim(prompt_emb_mask, target_ndim=2),
    }

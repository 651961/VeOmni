# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Selective Activation Checkpointing (SAC) policies.

Plugs into ``torch.utils.checkpoint.checkpoint`` via the ``context_fn``
hook to keep the outputs of selected expensive ops live across the
forward/backward boundary while still recomputing everything else under
an activation-checkpointed scope. Trades a small amount of memory for
skipping the most expensive recompute work.

Resolves a policy name to a ``context_fn`` callable consumable by
``torch.utils.checkpoint.checkpoint``. PyTorch rejects ``context_fn``
under reentrant mode, so SAC requires ``use_reentrant=False``.

Policies:
  ``"none"``           noop; returns ``None`` so callers fall through to
                       torch's ``noop_context_fn`` and preserve full ckpt.
  ``"attn_only"``      MUST_SAVE the output of any fused SDPA / flash-
                       attention op; PREFER_RECOMPUTE everything else.
  ``"attn_rotary"``    Same as ``attn_only`` plus MUST_SAVE the fused
                       interleaved-rotary triton kernel
                       (``veomni::rotary_interleaved_fwd``). The rotary
                       op runs twice per block (q and k), so on a 60-
                       block DiT skipping its recompute is a measurable
                       saving on top of attention.
  ``"attn_and_mlp"``   Same as ``attn_only`` plus MUST_SAVE every
                       ``aten.mm`` / ``aten.addmm`` (i.e. every
                       ``nn.Linear`` output). Skips recomputing all
                       matmuls in backward at the cost of holding their
                       outputs live across the ckpt boundary.
  ``"attn_rotary_and_mlp"``
                       Union of ``attn_rotary`` and ``attn_and_mlp``:
                       MUST_SAVE attention + rotary + every matmul.
                       Largest memory footprint, smallest backward
                       recompute.
"""
from __future__ import annotations

import os
from functools import partial
from typing import Callable, Optional, Tuple

import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


# Set VEOMNI_SAC_DEBUG=1 to print each unique MUST_SAVE op the first time the
# policy hits it (rank-0 only). Useful for confirming SAC still fires after
# torch.compile -- inductor may fuse some aten ops away, so the policy can
# silently degrade. When unset the debug path is a single bool check.
_DEBUG: bool = os.environ.get("VEOMNI_SAC_DEBUG", "") in ("1", "true", "True")
_SEEN: set = set()


def _maybe_log(func) -> None:
    if not _DEBUG:
        return
    key = str(func)
    if key in _SEEN:
        return
    _SEEN.add(key)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[SAC] MUST_SAVE first hit: {key}", flush=True)


def _collect_attn_ops() -> set:
    """Return the set of fused-attention op overloads present in this
    build. Torch ships different backends across versions (cuDNN SDPA
    in 2.4+, flash via aten path, etc.); look up each candidate
    defensively so the policy still loads if some are missing.

    Also picks up ``flash_attn_3::_flash_attn_forward`` (registered
    upstream as a ``torch.library.custom_op``), which sits outside the
    ``aten`` namespace but is what dispatcher sees when the model calls
    FA3 directly. Without this the policy never fires on FA3 paths.
    """
    candidates = (
        ("aten", "_flash_attention_forward"),
        ("aten", "_scaled_dot_product_efficient_attention"),
        ("aten", "_scaled_dot_product_flash_attention"),
        ("aten", "_scaled_dot_product_cudnn_attention"),
        ("aten", "_efficient_attention_forward"),
        ("flash_attn_3", "_flash_attn_forward"),
    )
    ops = set()
    for ns_name, name in candidates:
        namespace = getattr(torch.ops, ns_name, None)
        if namespace is None:
            continue
        op = getattr(namespace, name, None)
        if op is None:
            continue
        default = getattr(op, "default", None)
        if default is not None:
            ops.add(default)
    return ops


def _collect_rotary_ops() -> set:
    """Return the fused interleaved-rotary forward overload, if registered.

    The kernel is registered as ``veomni::rotary_interleaved_fwd`` from the
    DiT modeling module on first import; resolving lazily mirrors the FA3
    case so the policy still picks it up no matter the import order.
    """
    candidates = (("veomni", "rotary_interleaved_fwd"),)
    ops = set()
    for ns_name, name in candidates:
        namespace = getattr(torch.ops, ns_name, None)
        if namespace is None:
            continue
        op = getattr(namespace, name, None)
        if op is None:
            continue
        default = getattr(op, "default", None)
        if default is not None:
            ops.add(default)
    return ops


def _collect_mm_ops() -> set:
    """Return ``aten`` matmul overloads that back ``nn.Linear``: ``mm`` for
    bias-free Linear, ``addmm`` for biased Linear. Looked up defensively
    in case a torch build renames them.
    """
    candidates = ("mm", "addmm")
    ops = set()
    for name in candidates:
        op = getattr(torch.ops.aten, name, None)
        if op is None:
            continue
        default = getattr(op, "default", None)
        if default is not None:
            ops.add(default)
    return ops


# Resolved lazily on first policy call. ``flash_attn_3::_flash_attn_forward``
# is registered by ``import flash_attn_interface`` inside the model module,
# which runs after ``sac.py`` is imported by ``torch_parallelize``; resolving
# at import time would miss it and silently degrade ``attn_only`` to a noop.
_ATTN_OPS: Optional[set] = None
_ROTARY_OPS: Optional[set] = None
_MM_OPS: Optional[set] = None


def _attn_only_policy(ctx, func, *args, **kwargs):
    global _ATTN_OPS
    if _ATTN_OPS is None:
        _ATTN_OPS = _collect_attn_ops()
    if func in _ATTN_OPS:
        _maybe_log(func)
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _attn_rotary_policy(ctx, func, *args, **kwargs):
    global _ATTN_OPS, _ROTARY_OPS
    if _ATTN_OPS is None:
        _ATTN_OPS = _collect_attn_ops()
    if _ROTARY_OPS is None:
        _ROTARY_OPS = _collect_rotary_ops()
    if func in _ATTN_OPS or func in _ROTARY_OPS:
        _maybe_log(func)
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _attn_and_mlp_policy(ctx, func, *args, **kwargs):
    global _ATTN_OPS, _MM_OPS
    if _ATTN_OPS is None:
        _ATTN_OPS = _collect_attn_ops()
    if _MM_OPS is None:
        _MM_OPS = _collect_mm_ops()
    if func in _ATTN_OPS or func in _MM_OPS:
        _maybe_log(func)
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def _attn_rotary_and_mlp_policy(ctx, func, *args, **kwargs):
    global _ATTN_OPS, _ROTARY_OPS, _MM_OPS
    if _ATTN_OPS is None:
        _ATTN_OPS = _collect_attn_ops()
    if _ROTARY_OPS is None:
        _ROTARY_OPS = _collect_rotary_ops()
    if _MM_OPS is None:
        _MM_OPS = _collect_mm_ops()
    if func in _ATTN_OPS or func in _ROTARY_OPS or func in _MM_OPS:
        _maybe_log(func)
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


def get_sac_context_fn(policy_name: str) -> Optional[Callable[[], Tuple]]:
    """Resolve a policy name to a ``context_fn``, or ``None`` for noop.

    Returning ``None`` lets the caller fall back to torch's
    ``noop_context_fn`` exactly (bit-identical to current behaviour),
    rather than wrapping a no-op SAC scope.
    """
    if policy_name == "none":
        return None
    if policy_name == "attn_only":
        return partial(create_selective_checkpoint_contexts, _attn_only_policy)
    if policy_name == "attn_rotary":
        return partial(create_selective_checkpoint_contexts, _attn_rotary_policy)
    if policy_name == "attn_and_mlp":
        return partial(create_selective_checkpoint_contexts, _attn_and_mlp_policy)
    if policy_name == "attn_rotary_and_mlp":
        return partial(create_selective_checkpoint_contexts, _attn_rotary_and_mlp_policy)
    raise ValueError(
        f"Unknown SAC policy {policy_name!r}. "
        "Expected one of: 'none', 'attn_only', 'attn_rotary', 'attn_and_mlp', 'attn_rotary_and_mlp'."
    )

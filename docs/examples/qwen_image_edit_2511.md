# Qwen-Image-Edit-2511 Training Guide

End-to-end Qwen-Image-Edit-2511 SFT in VeOmni. The recipe is split into two
stages and driven by a single yaml (`configs/dit/qwen_image_edit_2511_sft.yaml`)
whose `train.training_task` field selects the active stage:

| `training_task`     | What it does                                                                                                  |
|---------------------|---------------------------------------------------------------------------------------------------------------|
| `offline_embedding` | One-shot. Encodes every sample with the text encoder + VAE, dumps pickled tensors to parquet shards.          |
| `offline_training`  | Trains the DiT on the cached embeddings. The text encoder and VAE are not loaded onto the training GPUs.      |

`online_training` (running the condition model live inside the train loop)
is not supported in this release.

---

## 1. Model layout

VeOmni reads the upstream Qwen-Image-Edit-2511 weights directly. Expected
layout under `model.condition_model_path` (e.g. `/models/Qwen-Image-Edit-2511`):

```
/models/Qwen-Image-Edit-2511/
├── model_index.json
├── scheduler/scheduler_config.json
├── text_encoder/                      # multimodal Qwen2.5-VL text encoder
├── tokenizer/                         # Qwen2 tokenizer
├── processor/                         # multimodal processor (image preprocessor)
├── transformer/                       # DiT
└── vae/                               # 3D causal VAE
```

`model.config_path` and `model.model_path` both point at the `transformer/`
subfolder. `model.condition_model_path` points at the root.

---

## 2. Dataset preparation

```
/path/to/example_dataset/
├── metadata.jsonl               # one JSON object per line
└── train/                       # asset files (paths in metadata.jsonl are relative to this)
```

`metadata.jsonl` rows:

| Field        | Type                 | Description                                          |
|--------------|----------------------|------------------------------------------------------|
| `prompt`     | `str`                | Edit instruction.                                    |
| `image`      | `str`                | Path (relative to `train/`) of the target image.     |
| `edit_image` | `str` or `list[str]` | One or more reference images.                        |

### 2.1 Convert to parquet

```shell
python3 scripts/multimodal/convert_data/qwen_image_edit.py \
    --dataset_base_path /path/to/example_dataset \
    --output_dir /path/to/example_dataset/train_parquet \
    --num_shards 32 \
    --num_proc 32
```

Each parquet row carries:

| Column              | Type          |
|---------------------|---------------|
| `prompt`            | `str`         |
| `image_bytes`       | `bytes`       |
| `edit_image_bytes`  | `list[bytes]` |
| `source`            | `str` (fixed `"Qwen-Image-Edit-2511"`) |

These columns are consumed by the `Qwen-Image-Edit-2511` source in
`veomni/data/multimodal/dit/preprocess.py`, which decodes the bytes into PIL
images on the fly.

---

## 3.1 Run offline embedding

```shell
NPROC_PER_NODE=8 \
bash train.sh tasks/train_dit.py \
    configs/dit/qwen_image_edit_2511_sft.yaml \
    --model.config_path /models/Qwen-Image-Edit-2511/transformer \
    --model.model_path /models/Qwen-Image-Edit-2511/transformer \
    --model.condition_model_path /models/Qwen-Image-Edit-2511 \
    --data.source_name Qwen-Image-Edit-2511 \
    --data.train_path /path/to/example_dataset/train_parquet \
    --data.offline_embedding_save_dir /path/to/example_dataset/train_offline_embedding \
    --train.training_task offline_embedding \
    --train.global_batch_size 8 \
    --train.accelerator.ulysses_size 1 \
    --train.num_train_epochs 1 \
    --train.wandb.enable false \
    --train.checkpoint.save_steps 0 \
    --train.checkpoint.output_dir /path/to/checkpoints/qwen_image_edit_2511_runs
```

Each output row stores four pickled tensors:

| Column            | Shape                                  | Description                                                              |
|-------------------|----------------------------------------|--------------------------------------------------------------------------|
| `latents`         | `(1, 16, H/8, W/8)` bf16               | Normalised target-image VAE latent.                                      |
| `edit_latents`    | `list[(1, 16, H/8, W/8)]` bf16         | One normalised VAE latent per reference image.                           |
| `prompt_emb`      | `(1, L, 3584)` bf16                    | Last hidden state of the text encoder, conditioned on the edit image, with the template prefix removed. |
| `prompt_emb_mask` | `(1, L)` int64                         | Validity mask for `prompt_emb` (currently all-ones — the un-padded split has no padding positions). |

Mode-of-Gaussian sampling and `latents_mean`/`latents_std` normalisation
are folded into the saved tensor — `process_condition` later only needs
to sample noise and a timestep.

---

## 3.2 Run offline training

Same launcher, different `training_task`. The DiT is built (FSDP2-sharded)
and the condition model is left as a meta-init shell. Every step reads
pickled embeddings from the parquet output of step 3.1.

```shell
export WANDB_API_KEY=""

NPROC_PER_NODE=8 \
bash train.sh tasks/train_dit.py \
    configs/dit/qwen_image_edit_2511_sft.yaml \
    --model.config_path /models/Qwen-Image-Edit-2511/transformer \
    --model.model_path /models/Qwen-Image-Edit-2511/transformer \
    --model.condition_model_path /models/Qwen-Image-Edit-2511 \
    --data.source_name Qwen-Image-Edit-2511 \
    --data.train_path /path/to/example_dataset/train_offline_embedding \
    --train.training_task offline_training \
    --train.global_batch_size 64 \
    --train.micro_batch_size 1 \
    --train.accelerator.ulysses_size 1 \
    --train.gradient_checkpointing.enable true \
    --train.gradient_checkpointing.sac_policy attn_rotary_and_mlp \
    --train.enable_compile true \
    --train.num_train_epochs 20 \
    --train.wandb.enable true \
    --train.wandb.project Qwen-Image-Edit-2511-SFT \
    --train.wandb.name 1node8gpu \
    --train.checkpoint.save_steps 1000 \
    --train.checkpoint.output_dir /path/to/checkpoints/qwen_image_edit_2511_runs
```

Per-step flow:

- The `dit_offline` data transform unpickles the four cached fields into
  a per-sample dict.
- `QwenImageEditConditionModel.process_condition` samples a uniform
  timestep on `[0, 1000)` and Gaussian noise from a per-rank generator,
  then builds `x_t = (1 - sigma) * x + sigma * noise`, `target = noise - x`,
  and `training_weight = bsmntw[timestep_id]`.
- The DiT patchifies the noisy target latent, concatenates the (clean)
  edit latents along the token dim, runs all 60 dual-stream blocks, slices
  the output back to the target portion, and computes the
  bsmntw-weighted flow-matching MSE against `training_target`.

### Loss weighting (always on)

The flow-matching MSE is multiplied by a **per-timestep weight** drawn
from a bell-shaped curve over the 1000-step training schedule. Plain
unweighted MSE under-trains because the signal at the extremes
(t → 0 and t → 1) is degenerate.

```
weight(t) = exp(-2 · ((t − 500) / 1000)²)             # Gaussian bell, peaks at t=500
weight    = (weight − weight.min()) · 1000 / sum       # normalised so mean ≈ 1
```

`FlowMatchScheduler._set_training_weight` computes this curve once and
caches it; `process_condition` indexes into it for the sample's
timestep. The DiT multiplies the per-sample MSE by the scalar before
returning. No user-facing flag — this is structural to the
Qwen-Image-Edit-2511 SFT recipe.

### Loss outlier mask (optional)

An elementwise mask on the flow-matching MSE: any element where
`|prediction − target| > threshold` is zeroed out before the spatial
mean, so a single exploding element cannot dominate the per-sample
loss. Useful as a stability guard under bf16.

Off by default. Enable via:

```yaml
model:
  model_config:
    loss_outlier_threshold: 50.0
```

Setting `loss_outlier_threshold: null` (or removing the field) disables
the mask entirely.

### FFN learning-rate multiplier (on in the yaml)

The feed-forward modules of every dual-stream block (`img_mlp` and
`txt_mlp`, each a two-linear `QwenFeedForward`) are trained at **2x the
base learning rate**; all other parameters stay at the base lr. This
yaml turns it on:

```yaml
train:
  optimizer:
    lr: 1.0e-4
    ffn_lr_mult: 2.0
    ffn_lr_mult_modules: [img_mlp, txt_mlp]
```

- `ffn_lr_mult_modules` is matched as substrings against parameter
  names; any param whose name contains one of the entries is moved into
  a separate group with `lr = lr * ffn_lr_mult`. With the two entries
  above this is `transformer_blocks.*.img_mlp.*` and `*.txt_mlp.*`
  (4 params each, 60 blocks).
- The lr ratio survives warmup and decay untouched: the scheduler is a
  `LambdaLR` that scales every group from its own captured initial lr by
  the same factor, so the 2x is preserved without any scheduler change.
- Weight-decay grouping is reapplied per subset, so a populated
  `no_decay_modules` / `no_decay_params` keeps working alongside the
  multiplier.
- `ffn_lr_mult: 1.0` (the field default) is a no-op and falls back to
  the standard single-lr optimizer; the override lives in
  `veomni/trainer/dit_trainer.py:_build_optimizer`. Not supported with
  `optimizer.type: muon` (logs a warning and ignores the multiplier).

A startup log line confirms the split, e.g.
`FFN lr override: <N> params matching ['img_mlp', 'txt_mlp'] use
lr=0.0002 (2.0x base); <M> params use lr=0.0001.` The matched set
includes block 59's structurally-dead `txt_mlp` (see §5) — harmless,
since those params never receive gradient regardless of their lr.

---

## 3.3 Performance defaults (on in the yaml)

The yaml turns on two compute-side optimizations by default. Both are
training-only; they do not change loss numerics. Override with
`--train.gradient_checkpointing.sac_policy <name>` or
`--train.enable_compile true`.

### Selective Activation Checkpointing (SAC)

`train.gradient_checkpointing.sac_policy: attn_rotary_and_mlp`. Sits on
top of normal full ckpt: the wrapped scope still recomputes everything
in backward, *except* the outputs of selected expensive ops which are
kept live across the forward/backward boundary. For Qwen-Image-Edit-2511
the saved set is:

| Op | Origin |
|----|--------|
| `flash_attn_3::_flash_attn_forward` | FA3 joint-attention call (custom_op) |
| `veomni::rotary_interleaved_fwd`    | Fused interleaved-rotary triton kernel (custom_op) |
| `aten.addmm`                        | Every `nn.Linear` (all linears in the model are biased) |

Together these cover virtually all per-block FLOPs, so backward
recompute drops to layer-norm / activation / elementwise residual only.
Trade-off is activation memory: each MUST_SAVE op holds its output live
across all 60 blocks. Available policies (`veomni/distributed/sac.py`):

| Policy | Saves |
|---|---|
| `none` | nothing extra (vanilla full ckpt) |
| `attn_only` | attention only |
| `attn_rotary` | attention + rotary |
| `attn_and_mlp` | attention + matmul |
| `attn_rotary_and_mlp`| all three |

Requires `train.gradient_checkpointing.enable_reentrant: false` (the
default). Set `VEOMNI_SAC_DEBUG=1` in the env to log every unique
MUST_SAVE hit once on rank 0 — useful for confirming the policy still
fires after a torch upgrade or model change.

### torch.compile

`train.enable_compile: true`. After FSDP2 wrap, `DiTTrainer` wraps each
of the 60 dual-stream blocks with `torch.compile(block)` in the default
mode (`veomni/trainer/dit_trainer.py:_maybe_compile_mlps`). FA3 and the
rotary triton kernel are registered as `torch.library.custom_op`, so
dynamo traces through them without graph breaks.

Caveats:

- **First step is slow**: 60 sequential compiles take minutes. Steady-
  state is reached from step ~3 onward.
- **Steady-state speedup is small** on this model (FA3, rotary, cuBLAS
  matmul are already optimized; inductor has little left to fuse). On
  top of `attn_rotary_and_mlp` SAC the measured delta is at parity or
  marginally better. Kept on by default because it costs only the cold
  start; set `--train.enable_compile false` to skip if you restart
  often or care about first-step latency.
- **`max-autotune-no-cudagraphs`** has been measured slower than the
  default mode on this model — keep the default.

---

## 4. Convert DCP checkpoint to a single-file format

Training writes distributed `.distcp` shards to
`<output_dir>/checkpoints/global_step_<N>/`. To get a single-process
`safetensors` checkpoint that drops in at the `transformer/` subfolder
of an upstream model directory, merge with `scripts/merge_dcp_to_hf.py`:

```shell
python scripts/merge_dcp_to_hf.py \
    --load-dir /path/to/checkpoints/qwen_image_edit_2511_runs/checkpoints/global_step_1000 \
    --save-dir /path/to/checkpoints/qwen_image_edit_2511_runs/checkpoints/global_step_1000/hf_ckpt \
    --model-assets-dir /models/Qwen-Image-Edit-2511/transformer \
    --shard-size 10000000000
```

`--shard-size 10000000000` (10 GB) matches the upstream Qwen-Image-Edit-2511
shard layout (5 shards, ~10 GB each).

---

## 5. Notes and limits

- **`micro_batch_size: 1` is hard-required.** Each sample has a unique
  `(H, W)` (aspect-ratio-preserving resize to fixed area), so the
  per-sample latent / edit latents cannot be stacked along dim 0. Use
  gradient accumulation (`global_batch_size / dp_size`) to grow the
  effective batch size.
- **Resize knobs** live under `model.condition_model_cfg`:
  `target_max_pixels`, `vae_image_area`, `condition_image_area`, and the
  matching `*_alignment` fields. Defaults: 1024² for VAE inputs, 384²
  for the text-encoder condition image, 32-pixel alignment.
- **`zero_cond_t` (Qwen-Image-Edit-2511 specific)** is on by default in
  `condition_model_cfg`. Inside the DiT this duplicates the timestep
  batch `[t, 0]` so target tokens see the real timestep and
  edit-condition tokens see `t = 0`. Disable only if you are porting
  the same trainer to a non-2511 checkpoint.
- **Saved HF weights** (`save_hf_weights: true`) write a
  `diffusion_pytorch_model*.safetensors` set compatible with the
  upstream Qwen-Image-Edit-2511 layout; drop them under `transformer/`
  in a Qwen-Image-Edit model directory to use with the upstream
  inference pipeline.
- **Checkpoint cadence** is controlled by two independent knobs under
  `train.checkpoint`, both honoured at the same time:
  - `save_steps: N` — save at every `global_step` that is a multiple of
    `N`. Set to `0` to disable.
  - `save_epochs: N` — save at the end of every `N`-th epoch. Set to
    `0` to disable.

  This config sets `save_steps: 1000`, `save_epochs: 0` — checkpoints
  land strictly every 1000 optimizer steps. The matching
  `hf_save_steps` / `hf_save_epochs` pair follows the same rules and is
  only consulted when `save_hf_weights: true`.
- **Last-block text-branch parameters are structurally dead** and will
  trigger an INFO log at every DCP save:

  ```
  OptimizerState: filled default state for 6 param(s) without optimizer state
    (no gradient received or aliased): [
      'transformer_blocks.59.attn.to_add_out.bias',
      'transformer_blocks.59.attn.to_add_out.weight',
      'transformer_blocks.59.txt_mlp.net.0.proj.bias',
      'transformer_blocks.59.txt_mlp.net.0.proj.weight',
      'transformer_blocks.59.txt_mlp.net.2.bias',
      'transformer_blocks.59.txt_mlp.net.2.weight',
    ]
  ```

  This is **expected**, not a bug. The DiT main loop runs all 60
  dual-stream blocks but **discards the text stream returned by the
  last block** — only the image stream feeds `norm_out` + `proj_out`.
  Block 59's text-output projection (`attn.to_add_out`) and text-side
  MLP therefore never reach the loss, never receive gradient, and never
  accumulate optimizer state. The DCP checkpointer fills zero
  placeholders so the saved state dict's keyset matches the model's
  parameter keyset (≈150 KB total). The released
  `/models/Qwen-Image-Edit-2511/transformer` weights confirm this is
  structural: block 59's three biases are exactly zero and the matching
  weights sit at their init magnitude (~0.1, vs >1.0 for block 58's
  trained counterparts) — upstream pretraining also never updated these
  tensors. The block-59 image-side parameters (`attn.to_out.0`,
  `img_mlp.net.{0.proj, 2}`) are trained as normal.

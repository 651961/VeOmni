# Krea2 Image-Edit SFT

This guide describes the normal Krea2 image-edit SFT workflow in VeOmni:

1. Run `offline_embedding` on raw image-edit parquet.
2. Run `offline_training` on the generated embedding cache.

The training path uses the Krea2 Raw diffusers-format model only. It does not import the
external Krea2 inference repository.

## Model

Expected model layout:

```text
/path/to/Krea-2-Raw/
├── transformer/
├── text_encoder/
├── tokenizer/
└── vae/
```

The training config is:

```text
configs/dit/krea2_image_edit_sft.yaml
```

## Data

Raw training parquet must contain:

| Column | Type |
|--------|------|
| `prompt` | `str` |
| `image_bytes` | `bytes` target image |
| `edit_image_bytes` | `list[bytes]` reference/source images |
| `source` | `str` optional |

## 1. Offline Embedding

This stage loads the frozen Krea2 VAE and Qwen3-VL conditioner, then writes
cached tensors to parquet. It does not train the DiT.

```shell
NPROC_PER_NODE=8 \
bash train.sh tasks/train_dit.py \
    configs/dit/krea2_image_edit_sft.yaml \
    --model.model_path /path/to/Krea-2-Raw \
    --data.source_name Krea2-Image-Edit \
    --data.train_path /path/to/example_dataset/train_parquet \
    --data.offline_embedding_save_dir /path/to/example_dataset/krea2_offline_embedding \
    --train.training_task offline_embedding \
    --train.global_batch_size 8 \
    --train.accelerator.dp_shard_size 1 \
    --train.accelerator.ulysses_size 1 \
    --train.num_train_epochs 1 \
    --train.wandb.enable false \
    --train.checkpoint.save_steps 0 \
    --train.checkpoint.output_dir /path/to/checkpoints/krea2_image_edit_runs
```

The output cache contains `latents`, `edit_latents`, `prompt_emb`, and
`prompt_emb_mask`. Use this cache as `data.train_path` for SFT.

## 2. Offline Training

This stage trains the Krea2 Raw transformer from the cached embeddings.
The condition model is meta-initialized and does not load VAE/text-encoder
weights.

```shell
export WANDB_API_KEY=""

NPROC_PER_NODE=8 \
bash train.sh tasks/train_dit.py \
    configs/dit/krea2_image_edit_sft.yaml \
    --model.model_path /path/to/Krea-2-Raw \
    --data.source_name Krea2-Image-Edit \
    --data.train_path /path/to/example_dataset/krea2_offline_embedding \
    --train.training_task offline_training \
    --train.global_batch_size 128 \
    --train.micro_batch_size 1 \
    --train.accelerator.dp_shard_size 8 \
    --train.accelerator.ulysses_size 1 \
    --train.gradient_checkpointing.enable true \
    --train.gradient_checkpointing.sac_policy attn_rotary_and_mlp \
    --train.num_train_epochs 10 \
    --train.wandb.enable true \
    --train.wandb.project Krea2-Image-Edit-SFT \
    --train.wandb.name krea2-image-edit \
    --train.checkpoint.save_steps 1000 \
    --train.checkpoint.output_dir /path/to/checkpoints/krea2_image_edit_runs
```

To resume:

```shell
--train.checkpoint.load_path auto
```

or pass a specific checkpoint directory:

```shell
--train.checkpoint.load_path /path/to/checkpoints/krea2_image_edit_runs/checkpoints/global_step_<step>
```

## Selective Activation Checkpointing

`configs/dit/krea2_image_edit_sft.yaml` enables
`train.gradient_checkpointing.sac_policy: attn_rotary_and_mlp` for the
offline-training stage. This keeps the outputs of Krea2's SDPA attention,
shared `veomni::rotary_interleaved_fwd` rotary custom op, and Linear matmuls
live across the checkpoint boundary while recomputing the remaining cheap ops.

The policy requires non-reentrant checkpointing, which is the VeOmni default
(`train.gradient_checkpointing.enable_reentrant: false`). Set
`VEOMNI_SAC_DEBUG=1` to print each unique MUST_SAVE op once on rank 0.

## Notes

- `offline_embedding` reads raw parquet.
- `offline_training` reads the generated offline embedding cache.
- `offline_embedding` does not train or shard the DiT. Use
  `NPROC_PER_NODE` to choose preprocessing parallelism and set
  `--train.accelerator.dp_shard_size 1` to override the SFT default.
- For Krea2, set `--model.model_path` to the Raw model root
  (`/path/to/Krea-2-Raw`). VeOmni derives `transformer`, `text_encoder`,
  `tokenizer`, and `vae` paths from that root.
- Krea2 SFT samples training timesteps from the same resolution-aware
  exponential-shift grid as the released Raw sampler. By default `mu` is derived
  from the target latent token count using the `(256, 0.5)` to `(6400, 1.15)`
  interpolation; set `model.condition_model_cfg.timestep_shift_mu` to pin a
  constant value.
- `micro_batch_size` is fixed to `1` by `DiTTrainer` for DiT tasks.
- `configs/dit/krea2_image_edit_sft.yaml` defaults to `offline_training`; set
  `--train.training_task offline_embedding` explicitly when generating cache.

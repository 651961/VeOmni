# Copyright 2023 Zhongjie Duan
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rectified-flow noise scheduler shared by VeOmni's diffusion models.

The default (``template=None``) reproduces the Flux/Wan style schedule used by
the deprecated single-shift trainers (``shift * sigmas / (1 + (shift - 1) * sigmas)``).
``template="Qwen-Image"`` switches to the exponential-shift + terminal-shift
schedule required by the Qwen-Image-Edit-2511 SFT recipe (see
``configs/dit/qwen_image_edit_2511_sft.yaml``).

Floating-point order is load-bearing under bf16.

The arithmetic in the Qwen-Image branch is preserved verbatim from the
reference SFT implementation. Do **not** rewrite "equivalent" forms - bf16
rounding differs under reorderings such as ``(1 - sigmas) / sigmas`` vs
``1 / sigmas - 1``, or ``one_minus_z * (1 / scale)`` vs
``one_minus_z / scale``. The alignment unit tests in
``tests/qwen_image_edit_2511/`` enforce <1e-6 absolute deviation against the
reference; any "simplification" here breaks them.
"""

import math
from typing import Optional

import torch


class FlowMatchScheduler:
    """Rectified-flow scheduler.

    Args:
        template: Selects the sigma/timestep schedule.
            * ``None`` (default) - legacy single-shift schedule used by
              ``deprecated_task/train_flux.py`` and ``train_wan.py``.
              All other ``__init__`` kwargs (``shift``, ``sigma_max`` ...
              ``reverse_sigmas``) are honored in this mode.
            * ``"Qwen-Image"`` - exponential shift + terminal shift, used by
              the Qwen-Image-Edit-2511 SFT recipe.
        num_inference_steps: Initial schedule length built at construction.
        num_train_timesteps: Number of training timesteps; timesteps array
            equals ``sigmas * num_train_timesteps``.
        shift, sigma_max, sigma_min, inverse_timesteps, extra_one_step,
        reverse_sigmas: Legacy single-shift schedule parameters. Ignored when
            ``template == "Qwen-Image"``.
    """

    def __init__(
        self,
        template: Optional[str] = None,
        num_inference_steps: int = 100,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
        inverse_timesteps: bool = False,
        extra_one_step: bool = False,
        reverse_sigmas: bool = False,
    ):
        self.template = template
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.linear_timesteps_weights: Optional[torch.Tensor] = None
        self.training: bool = False
        self.set_timesteps(num_inference_steps)

    # ----- Schedule construction ----------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps: int = 100,
        denoising_strength: float = 1.0,
        training: bool = False,
        shift: Optional[float] = None,
        dynamic_shift_len: Optional[int] = None,
        exponential_shift_mu: Optional[float] = None,
    ) -> None:
        """Populate ``self.sigmas`` and ``self.timesteps``.

        Args:
            num_inference_steps: Length of the sigma / timestep arrays.
            denoising_strength: Upper bound of the sigma linspace.
            training: If True, additionally populate
                ``self.linear_timesteps_weights`` and set ``self.training = True``.
            shift: Legacy template override for the single-shift parameter.
                Only used when ``template is None``.
            dynamic_shift_len: Qwen-Image only. If given, ``mu`` is derived
                from this image sequence length.
            exponential_shift_mu: Qwen-Image only. If given, ``mu`` is taken
                directly. Takes priority over ``dynamic_shift_len``.
        """
        if self.template == "Qwen-Image":
            self.sigmas, self.timesteps = self._set_timesteps_qwen_image(
                num_inference_steps=num_inference_steps,
                denoising_strength=denoising_strength,
                dynamic_shift_len=dynamic_shift_len,
                exponential_shift_mu=exponential_shift_mu,
            )
        else:
            self.sigmas, self.timesteps = self._set_timesteps_legacy(
                num_inference_steps=num_inference_steps,
                denoising_strength=denoising_strength,
                shift=shift,
            )
        if training:
            self._set_training_weight(num_inference_steps=num_inference_steps)
            self.training = True
        else:
            self.linear_timesteps_weights = None
            self.training = False

    def _set_timesteps_legacy(
        self,
        num_inference_steps: int,
        denoising_strength: float,
        shift: Optional[float],
    ):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            sigmas = torch.flip(sigmas, dims=[0])
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        if self.reverse_sigmas:
            sigmas = 1 - sigmas
        timesteps = sigmas * self.num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def _set_timesteps_qwen_image(
        num_inference_steps: int,
        denoising_strength: float,
        dynamic_shift_len: Optional[int],
        exponential_shift_mu: Optional[float],
    ):
        sigma_min = 0.0
        sigma_max = 1.0
        num_train_timesteps = 1000
        shift_terminal = 0.02

        # Linspace dropping the trailing 0 so sigmas[-1] > 0 before shifting.
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]

        # Mu selection. The training default falls through to the constant 0.8.
        if exponential_shift_mu is not None:
            mu = exponential_shift_mu
        elif dynamic_shift_len is not None:
            mu = FlowMatchScheduler._empirical_mu_qwen_image(dynamic_shift_len)
        else:
            mu = 0.8

        # Exponential shift. Keep ``1 / sigmas - 1`` literally - bf16 rounding
        # differs vs ``(1 - sigmas) / sigmas``.
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / sigmas - 1))

        # Terminal shift so sigmas[-1] == shift_terminal. Keep both divisions
        # as ``/``; ``* (1 / x)`` rounds differently under bf16.
        one_minus_z = 1 - sigmas
        scale_factor = one_minus_z[-1] / (1 - shift_terminal)
        sigmas = 1 - (one_minus_z / scale_factor)

        # Timesteps. Keep as ``*``; ``/ (1 / N)`` rounds differently.
        timesteps = sigmas * num_train_timesteps
        return sigmas, timesteps

    @staticmethod
    def _empirical_mu_qwen_image(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 8192,
        base_shift: float = 0.5,
        max_shift: float = 0.9,
    ) -> float:
        """Linear ``mu`` interpolation used when ``dynamic_shift_len`` is given."""
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def _set_training_weight(self, num_inference_steps: int) -> None:
        """Populate ``self.linear_timesteps_weights`` (bsmntw bell-shaped curve)."""
        if self.template == "Qwen-Image":
            steps = 1000
            x = self.timesteps
            y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
            y_shifted = y - y.min()
            # Parenthesise as ``(steps / sum)``; broadcast the scalar last.
            bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
            if len(self.timesteps) != 1000:
                # Empirical correction kept for parity when N != 1000. The
                # SFT recipe (N = 1000) never triggers this branch.
                bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
                bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
            self.linear_timesteps_weights = bsmntw_weighing
        else:
            # Legacy: normalizer is ``num_inference_steps``. Preserved as-is
            # for backwards compatibility with the deprecated Flux/Wan
            # trainers.
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    # ----- Training-step primitives -------------------------------------------

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        """One Euler integration step (inference)."""
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def return_to_timestep(self, timestep, sample, sample_stablized):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output

    def add_noise(
        self,
        original_samples,
        noise,
        timestep,
        micro_batch_size=None,
        enable_mixed_precision=None,
    ):
        """Linearly interpolate ``original_samples`` and ``noise`` by sigma_t.

        ``micro_batch_size`` and ``enable_mixed_precision`` are accepted for
        backwards compatibility with the deprecated Flux/Wan trainers and are
        unused (the arithmetic is shape-agnostic).
        """
        del micro_batch_size, enable_mixed_precision
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        # Keep the (1 - sigma) * x + sigma * noise form. The algebraically
        # equivalent ``x + sigma * (noise - x)`` rounds differently in bf16.
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(self, sample, noise, timestep):
        """Rectified-flow velocity field target: ``noise - sample``."""
        del timestep  # signature parity only
        target = noise - sample
        return target

    def training_weight(self, timestep, micro_batch_size=None):
        """Look up the bsmntw loss weight at ``timestep``.

        ``micro_batch_size`` is accepted for backwards compatibility and unused.
        """
        del micro_batch_size
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        weights = self.linear_timesteps_weights[timestep_id]
        return weights

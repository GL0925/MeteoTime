from __future__ import annotations

from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from config_model import ModelConfig


class ResidualProjection(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.skip = nn.Linear(input_size, output_size, bias=False)
        self.input = nn.Linear(input_size, hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, output_size, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.skip(inputs) + self.output(F.silu(self.input(inputs)))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.float().square().mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps).to(inputs.dtype)
        return normalized * self.weight.to(inputs.dtype)


class PatchEmbedding(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        input_size = config.patch_size
        self.projection = ResidualProjection(input_size, config.hidden_size, config.hidden_size)
        self.mask_token = nn.Parameter(torch.empty(config.hidden_size))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(
        self,
        values: torch.Tensor,
        random_mask_ratio: float = 0.0,
    ) -> torch.Tensor:
        embeddings = self.projection(values)
        if self.training and random_mask_ratio > 0:
            random_mask = torch.rand(
                embeddings.shape[:2], device=embeddings.device
            ) < random_mask_ratio
            embeddings = torch.where(
                random_mask.unsqueeze(-1), self.mask_token, embeddings
            )
        return embeddings


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_sequence_tokens: int, theta: float) -> None:
        super().__init__()
        frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_sequence_tokens, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    @staticmethod
    def _rotate_half(inputs: torch.Tensor) -> torch.Tensor:
        first = inputs[..., ::2]
        second = inputs[..., 1::2]
        return torch.stack((-second, first), dim=-1).flatten(-2)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = queries.shape[-2]
        cos = self.cos[:sequence_length].repeat_interleave(2, dim=-1)
        sin = self.sin[:sequence_length].repeat_interleave(2, dim=-1)
        cos = cos.to(dtype=queries.dtype)[None, None, :, :]
        sin = sin.to(dtype=queries.dtype)[None, None, :, :]
        return (
            queries * cos + self._rotate_half(queries) * sin,
            keys * cos + self._rotate_half(keys) * sin,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.query_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.key_norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.rotary = rotary

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = inputs.shape
        qkv = self.qkv(inputs).view(
            batch_size, sequence_length, 3, self.num_heads, self.head_dim
        )
        queries, keys, values = qkv.unbind(dim=2)
        queries = self.query_norm(queries).transpose(1, 2)
        keys = self.key_norm(keys).transpose(1, 2)
        values = values.transpose(1, 2)
        queries, keys = self.rotary(queries, keys)
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(
            batch_size, sequence_length, hidden_size
        )
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(inputs)) * self.up(inputs))


class DecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = CausalSelfAttention(config, rotary)
        self.feed_forward_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = SwiGLU(config.hidden_size, config.intermediate_size)
        self.dropout = config.residual_dropout

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attention = self.attention(self.attention_norm(inputs))
        inputs = inputs + F.dropout(attention, p=self.dropout, training=self.training)
        feed_forward = self.feed_forward(self.feed_forward_norm(inputs))
        return inputs + F.dropout(feed_forward, p=self.dropout, training=self.training)


class MeteoTime(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.config.validate()
        self.patch_embedding = PatchEmbedding(self.config)
        rotary = RotaryEmbedding(
            self.config.head_dim,
            self.config.max_sequence_tokens,
            self.config.rope_theta,
        )
        self.blocks = nn.ModuleList(
            DecoderBlock(self.config, rotary) for _ in range(self.config.num_layers)
        )
        self.final_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        output_size = (
            self.config.forecast_horizon
            * len(self.config.quantiles)
        )
        self.quantile_head = ResidualProjection(
            self.config.hidden_size, self.config.hidden_size, output_size
        )
        self.register_buffer(
            "quantile_levels",
            torch.tensor(self.config.quantiles, dtype=torch.float32),
            persistent=True,
        )
        self.apply(self._initialize_weights)
        residual_scale = (2 * self.config.num_layers) ** -0.5
        for block in self.blocks:
            block.attention.output.weight.data.mul_(residual_scale)
            block.feed_forward.down.weight.data.mul_(residual_scale)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def median_index(self) -> int:
        return self.config.quantiles.index(0.5)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _patchify(
        self, values: torch.Tensor, observed_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim != 2 or observed_mask.shape != values.shape:
            raise ValueError("values and observed_mask must both have shape [batch, time]")
        if values.shape[1] % self.config.patch_size:
            raise ValueError("input length must be divisible by patch_size")
        token_count = values.shape[1] // self.config.patch_size
        if token_count > self.config.max_sequence_tokens:
            raise ValueError("input exceeds max_sequence_tokens")
        return (
            values.unfold(1, self.config.patch_size, self.config.patch_size),
            observed_mask.unfold(1, self.config.patch_size, self.config.patch_size),
        )

    def forward(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor | None = None,
        random_mask_ratio: float | None = None,
        target: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target is not None:
            if observed_mask is None or target_mask is None:
                raise ValueError("training forward requires context and target masks")
            return self.training_loss(
                values,
                target,
                observed_mask,
                target_mask,
                random_mask_ratio=random_mask_ratio,
            )
        if observed_mask is None:
            observed_mask = torch.isfinite(values)
        else:
            observed_mask = observed_mask.bool() & torch.isfinite(values)
        values = torch.where(observed_mask, torch.nan_to_num(values), torch.zeros_like(values))
        patches, _ = self._patchify(values, observed_mask)
        ratio = self.config.patch_mask_ratio if random_mask_ratio is None else random_mask_ratio
        hidden = self.patch_embedding(patches, ratio)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        predictions = self.quantile_head(hidden)
        return predictions.view(
            values.shape[0],
            patches.shape[1],
            self.config.forecast_horizon,
            len(self.config.quantiles),
        )

    def pinball_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        errors = targets.unsqueeze(-1) - predictions
        levels = self.quantile_levels.to(device=predictions.device, dtype=predictions.dtype)
        losses = torch.maximum(levels * errors, (levels - 1.0) * errors)
        weights = target_mask.unsqueeze(-1).to(losses.dtype)
        denominator = (weights.sum() * len(self.config.quantiles)).clamp_min(1.0)
        return (losses * weights).sum() / denominator

    def training_loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
        random_mask_ratio: float | None = None,
    ) -> torch.Tensor:
        if context.shape[1] > self.config.max_context_points:
            raise ValueError("context exceeds max_context_points")
        sequence = torch.cat((context, target), dim=1)
        sequence_mask = torch.cat((context_mask, target_mask), dim=1)
        output_reserve = (
            (self.config.forecast_horizon + self.config.patch_size - 1)
            // self.config.patch_size
            * self.config.patch_size
        )
        padding = (-sequence.shape[1]) % self.config.patch_size
        if padding:
            sequence = F.pad(sequence, (0, padding), value=0.0)
            sequence_mask = F.pad(sequence_mask, (0, padding), value=False)
        inputs = sequence[:, :-output_reserve]
        input_mask = sequence_mask[:, :-output_reserve]
        target_points = sequence[:, self.config.patch_size :]
        target_point_mask = sequence_mask[:, self.config.patch_size :]
        predictions = self(
            inputs,
            input_mask,
            random_mask_ratio=random_mask_ratio,
        )
        target_patches = target_points.unfold(
            1, self.config.forecast_horizon, self.config.patch_size
        )
        target_patch_masks = target_point_mask.unfold(
            1, self.config.forecast_horizon, self.config.patch_size
        )
        return self.pinball_loss(predictions, target_patches, target_patch_masks)

    def _left_pad_to_patch(
        self, values: torch.Tensor, observed_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        remainder = values.shape[1] % self.config.patch_size
        if remainder == 0:
            return values, observed_mask
        padding = self.config.patch_size - remainder
        return (
            F.pad(values, (padding, 0), value=0.0),
            F.pad(observed_mask, (padding, 0), value=False),
        )

    @torch.no_grad()
    def generate(
        self,
        context: torch.Tensor,
        prediction_length: int,
        context_mask: torch.Tensor | None = None,
        fix_quantile_crossing: bool = True,
    ) -> torch.Tensor:
        if prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if context_mask is None:
            context_mask = torch.isfinite(context)
        else:
            context_mask = context_mask.bool() & torch.isfinite(context)
        context = torch.where(context_mask, torch.nan_to_num(context), torch.zeros_like(context))
        if context.shape[1] > self.config.max_context_points:
            context = context[:, -self.config.max_context_points :]
            context_mask = context_mask[:, -self.config.max_context_points :]
        context, context_mask = self._left_pad_to_patch(context, context_mask)
        generated: list[torch.Tensor] = []
        output_patch_count = (
            (self.config.forecast_horizon + self.config.patch_size - 1)
            // self.config.patch_size
        )
        steps = ceil(prediction_length / self.config.forecast_horizon)
        if (
            context.shape[1] // self.config.patch_size
            + steps * output_patch_count
            > self.config.max_sequence_tokens
        ):
            raise ValueError("context and prediction exceed max_sequence_tokens")

        was_training = self.training
        self.eval()
        values = context
        observed_mask = context_mask
        for _ in range(steps):
            # A 48-point horizon is not a multiple of the 32-point input patch.
            # Re-align the growing sequence before each autoregressive step.
            values, observed_mask = self._left_pad_to_patch(values, observed_mask)
            next_quantiles = self(values, observed_mask, random_mask_ratio=0.0)[:, -1]
            if fix_quantile_crossing:
                next_quantiles = next_quantiles.sort(dim=-1).values
            generated.append(next_quantiles)
            median = next_quantiles[..., self.median_index]
            values = torch.cat((values, median), dim=1)
            observed_mask = torch.cat(
                (observed_mask, torch.ones_like(median, dtype=torch.bool)), dim=1
            )
        self.train(was_training)
        return torch.cat(generated, dim=1)[:, :prediction_length]

    @torch.no_grad()
    def forecast(
        self,
        context: torch.Tensor,
        prediction_length: int,
        context_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if context_mask is None:
            context_mask = torch.isfinite(context)
        context_mask = context_mask.bool() & torch.isfinite(context)
        clean = torch.where(context_mask, context, torch.zeros_like(context))
        counts = context_mask.sum(dim=1, keepdim=True).clamp_min(1)
        location = clean.sum(dim=1, keepdim=True) / counts
        centered = torch.where(context_mask, context - location, torch.zeros_like(context))
        # 对齐 TimesFM RevIN：使用无偏标准差，近常量序列的 scale 取 1。
        sample_count = context_mask.sum(dim=1, keepdim=True)
        scale = torch.sqrt(
            centered.square().sum(dim=1, keepdim=True)
            / (sample_count - 1).clamp_min(1)
        )
        scale = torch.where(
            (sample_count > 1)
            & (scale >= self.config.normalization_scale_floor),
            scale,
            torch.ones_like(scale),
        )
        normalized = centered / scale
        quantiles = self.generate(
            normalized,
            prediction_length=prediction_length,
            context_mask=context_mask,
        )
        quantiles = quantiles * scale.unsqueeze(-1) + location.unsqueeze(-1)
        return {
            "quantiles": quantiles,
            "median": quantiles[..., self.median_index],
            "loc": location,
            "scale": scale,
        }

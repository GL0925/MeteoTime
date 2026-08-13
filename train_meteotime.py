from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config_data import DataConfig
from config_model import ModelConfig
from config_train import TrainConfig
from models.meteotime import MeteoTime
from scripts_data.mixture_dataset import WeatherBatchDataset


def parse_args() -> argparse.Namespace:
    config = TrainConfig()
    parser = argparse.ArgumentParser(description="预训练 MeteoTime（支持 DDP）")
    parser.add_argument("--epochs", type=int, default=config.epochs)
    parser.add_argument("--batches-per-epoch", type=int, default=config.batches_per_epoch)
    parser.add_argument("--batch-size", type=int, default=config.batch_size, help="每张显卡的微批量")
    parser.add_argument("--workers", type=int, default=config.workers)
    parser.add_argument("--learning-rate", type=float, default=config.learning_rate)
    parser.add_argument("--min-learning-rate", type=float, default=config.min_learning_rate)
    parser.add_argument("--warmup-ratio", type=float, default=config.warmup_ratio)
    parser.add_argument("--weight-decay", type=float, default=config.weight_decay)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=config.gradient_accumulation,
    )
    parser.add_argument("--max-grad-norm", type=float, default=config.max_grad_norm)
    parser.add_argument("--seed", type=int, default=config.seed)
    parser.add_argument("--log-every", type=int, default=config.log_every)
    parser.add_argument("--output-dir", type=Path, default=config.output_dir)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def setup_distributed(backend: str) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("MeteoTime training requires CUDA")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, local_rank, world_size


def seed_everything(seed: int, rank: int) -> None:
    local_seed = seed + rank
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    torch.cuda.manual_seed(local_seed)


def cosine_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    base_learning_rate = optimizer.param_groups[0]["lr"]
    min_ratio = min(1.0, min_learning_rate / base_learning_rate)

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def continuation_schedule(
    optimizer: torch.optim.Optimizer,
    remaining_steps: int,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Continue from the checkpoint LR without restarting warmup or increasing it."""
    horizon = max(1, remaining_steps)
    base_learning_rate = optimizer.param_groups[0]["lr"]
    min_ratio = min(1.0, min_learning_rate / base_learning_rate)

    def multiplier(step: int) -> float:
        progress = min(1.0, step / horizon)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def two_stage_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    total_steps: int,
    warmup_ratio: float,
    restart_epoch: int,
    stage_min_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup + cosine, then restart cosine from the first stage endpoint."""
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    restart_step = max(warmup_steps + 1, restart_epoch * steps_per_epoch)
    restart_step = min(restart_step, total_steps - 1)
    first_floor = stage_min_ratio
    second_floor = stage_min_ratio * stage_min_ratio

    def cosine_between(progress: float, start: float, end: float) -> float:
        progress = min(1.0, max(0.0, progress))
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if step <= restart_step:
            progress = (step - warmup_steps) / max(1, restart_step - warmup_steps)
            return cosine_between(progress, 1.0, first_floor)
        progress = (step - restart_step) / max(1, total_steps - restart_step)
        return cosine_between(progress, first_floor, second_floor)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def move_training_tensors(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("context", "target", "context_mask", "target_mask")
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


def clip_grad_norm_fp32(
    parameters: list[torch.nn.Parameter] | tuple[torch.nn.Parameter, ...],
    max_norm: float,
) -> torch.Tensor:
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.zeros((), device="cuda", dtype=torch.float32)
    total_squared_norm = torch.zeros(
        (), device=gradients[0].device, dtype=torch.float32
    )
    for gradient in gradients:
        total_squared_norm.add_(gradient.detach().float().square().sum())
    total_norm = total_squared_norm.sqrt()
    if not torch.isfinite(total_norm):
        raise FloatingPointError("non-finite gradient detected before optimizer step")
    coefficient = max_norm / (total_norm + 1e-6)
    if coefficient < 1:
        for gradient in gradients:
            gradient.mul_(coefficient.to(dtype=gradient.dtype))
    return total_norm


@torch.inference_mode()
def run_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    world_size: int,
    target_categories: tuple[str, ...],
) -> dict:
    was_training = model.training
    model.eval()
    category_to_index = {name: index for index, name in enumerate(target_categories)}
    total_index = len(target_categories)
    loss_totals = torch.zeros(
        len(target_categories) + 1, 2, device=device, dtype=torch.float64
    )
    forecast_totals = torch.zeros(
        len(target_categories) + 1, 3, device=device, dtype=torch.float64
    )
    sample_totals = torch.zeros(
        len(target_categories) + 1, device=device, dtype=torch.float64
    )
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    try:
        for batch in loader:
            tensors = move_training_tensors(batch, device)
            sequence = torch.cat((tensors["context"], tensors["target"]), dim=1)
            sequence_mask = torch.cat((tensors["context_mask"], tensors["target_mask"]), dim=1)
            output_reserve = (
                (unwrapped.config.forecast_horizon + unwrapped.config.patch_size - 1)
                // unwrapped.config.patch_size
                * unwrapped.config.patch_size
            )
            padding = (-sequence.shape[1]) % unwrapped.config.patch_size
            if padding:
                sequence = F.pad(sequence, (0, padding), value=0.0)
                sequence_mask = F.pad(sequence_mask, (0, padding), value=False)
            inputs = sequence[:, :-output_reserve]
            input_mask = sequence_mask[:, :-output_reserve]
            target_points = sequence[:, unwrapped.config.patch_size :]
            target_point_mask = sequence_mask[:, unwrapped.config.patch_size :]
            target_windows = target_points.unfold(
                1, unwrapped.config.forecast_horizon, unwrapped.config.patch_size
            )
            target_window_masks = target_point_mask.unfold(
                1, unwrapped.config.forecast_horizon, unwrapped.config.patch_size
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions = model(
                    inputs,
                    input_mask,
                    random_mask_ratio=0.0,
                )
                quantiles = unwrapped.generate(
                    tensors["context"],
                    prediction_length=tensors["target"].shape[1],
                    context_mask=tensors["context_mask"],
                )
            median = quantiles[..., unwrapped.median_index].float()
            target = tensors["target"].float()
            target_mask = tensors["target_mask"]
            errors = torch.where(target_mask, median - target, torch.zeros_like(target))
            levels = unwrapped.quantile_levels.to(
                device=predictions.device, dtype=predictions.dtype
            )
            pinball = torch.maximum(
                levels * (target_windows.unsqueeze(-1) - predictions),
                (levels - 1.0) * (target_windows.unsqueeze(-1) - predictions),
            )
            point_weights = target_window_masks.unsqueeze(-1).to(pinball.dtype)
            category_names = batch["target_variable"]
            for category_index, category_name in enumerate(category_names):
                row_index = category_to_index[category_name]
                row_pinball = pinball[category_index]
                row_weights = point_weights[category_index]
                loss_totals[row_index, 0] += (row_pinball * row_weights).sum().double()
                loss_totals[row_index, 1] += (
                    row_weights.sum() * len(levels)
                ).double()
                forecast_totals[row_index, 0] += errors[category_index].abs().double().sum()
                forecast_totals[row_index, 1] += errors[category_index].square().double().sum()
                forecast_totals[row_index, 2] += target_mask[category_index].sum().double()
                sample_totals[row_index] += 1

            loss_totals[total_index, 0] += (pinball * point_weights).sum().double()
            loss_totals[total_index, 1] += (
                point_weights.sum() * len(levels)
            ).double()
            forecast_totals[total_index, 0] += errors.abs().double().sum()
            forecast_totals[total_index, 1] += errors.square().double().sum()
            forecast_totals[total_index, 2] += target_mask.sum().double()
            sample_totals[total_index] += len(category_names)
        if world_size > 1:
            dist.all_reduce(loss_totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(forecast_totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(sample_totals, op=dist.ReduceOp.SUM)

        def metrics_for(index: int) -> dict:
            point_count = forecast_totals[index, 2].clamp_min(1)
            return {
                "pinball_loss": (
                    loss_totals[index, 0] / loss_totals[index, 1].clamp_min(1)
                ).item(),
                "forecast_nmae": (forecast_totals[index, 0] / point_count).item(),
                "forecast_nrmse": torch.sqrt(
                    forecast_totals[index, 1] / point_count
                ).item(),
                "samples": int(sample_totals[index].item()),
            }

        by_variable = {}
        for index, category_name in enumerate(target_categories):
            by_variable[category_name] = metrics_for(index)

        total_metrics = metrics_for(total_index)
        return {
            "pinball_loss": total_metrics["pinball_loss"],
            "forecast_nmae": total_metrics["forecast_nmae"],
            "forecast_nrmse": total_metrics["forecast_nrmse"],
            "samples": total_metrics["samples"],
            "by_variable": by_variable,
        }
    finally:
        model.train(was_training)


def record_validation(
    metrics: dict,
    global_step: int,
    elapsed_seconds: float,
    batches_per_rank: int,
    writer: SummaryWriter | None,
) -> None:
    print(
        json.dumps(
            {
                "phase": "validation",
                "optimizer_step": global_step,
                "pinball_loss": metrics["pinball_loss"],
                "forecast_nmae": metrics["forecast_nmae"],
                "forecast_nrmse": metrics["forecast_nrmse"],
                "samples": metrics["samples"],
                "by_variable": metrics["by_variable"],
                "batches_per_rank": batches_per_rank,
                "elapsed_seconds": round(elapsed_seconds, 2),
            }
        )
    )
    if writer is None:
        return
    writer.add_scalar("validation/pinball_loss", metrics["pinball_loss"], global_step)
    writer.add_scalar("validation/forecast_nmae", metrics["forecast_nmae"], global_step)
    writer.add_scalar("validation/forecast_nrmse", metrics["forecast_nrmse"], global_step)
    for variable_name, variable_metrics in metrics["by_variable"].items():
        for metric_name, value in variable_metrics.items():
            if metric_name == "samples":
                continue
            writer.add_scalar(
                f"validation_{variable_name}/{metric_name}",
                value,
                global_step,
            )


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    model_config: ModelConfig,
    data_config: DataConfig,
    train_config: TrainConfig,
    best_validation_loss: float,
    tensorboard_log_dir: Path | None,
) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    checkpoint_epoch = epoch
    checkpoint_batch = batch_in_epoch
    if checkpoint_batch >= train_config.batches_per_epoch:
        checkpoint_epoch += 1
        checkpoint_batch = 0
    torch.save(
        {
            "model": unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": checkpoint_epoch,
            "batch_in_epoch": checkpoint_batch,
            "global_step": global_step,
            "best_validation_loss": best_validation_loss,
            "tensorboard_log_dir": (
                str(tensorboard_log_dir) if tensorboard_log_dir is not None else None
            ),
            "model_config": asdict(model_config),
            "data_config": asdict(data_config),
            "train_config": asdict(train_config),
        },
        temporary_path,
    )
    os.replace(temporary_path, checkpoint_path)
    return checkpoint_path


def main() -> None:
    args = parse_args()
    train_config = replace(
        TrainConfig(),
        epochs=args.epochs,
        batches_per_epoch=args.batches_per_epoch,
        batch_size=args.batch_size,
        workers=args.workers,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation=args.gradient_accumulation,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        log_every=args.log_every,
        output_dir=args.output_dir,
    )
    train_config.validate()
    rank, local_rank, world_size = setup_distributed(train_config.distributed_backend)
    seed_everything(train_config.seed, rank)
    device = torch.device("cuda", local_rank)

    data_config = DataConfig(project_root=Path.cwd())
    model_config = ModelConfig()
    data_config.validate()
    model_config.validate()
    if data_config.patch_size != model_config.patch_size:
        raise ValueError("data and model patch sizes must match")
    if data_config.max_context_length != model_config.max_context_points:
        raise ValueError("data and model maximum context lengths must match")
    if data_config.prediction_length != model_config.forecast_horizon:
        raise ValueError("data prediction length must match the model forecast horizon")
    if data_config.normalization_scale_floor != model_config.normalization_scale_floor:
        raise ValueError("data and model normalization scale floors must match")
    dataset = WeatherBatchDataset(
        data_config,
        batches_per_epoch=train_config.batches_per_epoch,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
        rank=rank,
        split="train",
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=train_config.workers,
        persistent_workers=train_config.workers > 0,
        pin_memory=train_config.pin_memory,
        prefetch_factor=train_config.prefetch_factor if train_config.workers > 0 else None,
    )
    validation_schedule = tuple(
        context_length
        for context_length in data_config.context_lengths
        for _ in range(train_config.validation_batches_per_context)
    )
    validation_dataset = WeatherBatchDataset(
        data_config,
        batches_per_epoch=len(validation_schedule),
        batch_size=train_config.validation_batch_size,
        context_schedule=validation_schedule,
        seed=train_config.validation_seed,
        rank=rank,
        split="validation",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=None,
        num_workers=train_config.validation_workers,
        persistent_workers=train_config.validation_workers > 0,
        pin_memory=train_config.pin_memory,
        prefetch_factor=(
            train_config.prefetch_factor
            if train_config.validation_workers > 0
            else None
        ),
    )

    model = MeteoTime(model_config).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.adam_beta1, train_config.adam_beta2),
        weight_decay=train_config.weight_decay,
        fused=train_config.fused_optimizer,
    )
    optimizer_steps_per_epoch = math.ceil(
        train_config.batches_per_epoch / train_config.gradient_accumulation
    )
    total_steps = train_config.epochs * optimizer_steps_per_epoch
    start_epoch = 0
    resume_batch_in_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    checkpoint_tensorboard_log_dir: Path | None = None
    resume_path = args.resume
    if resume_path is None and train_config.resume_enabled:
        resume_path = train_config.resume_path
    checkpoint = None
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        unwrapped.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])
        resume_batch_in_epoch = int(checkpoint.get("batch_in_epoch", 0))
        global_step = int(checkpoint["global_step"])
        best_validation_loss = float(
            checkpoint.get("best_validation_loss", float("inf"))
        )
        stored_log_dir = checkpoint.get("tensorboard_log_dir")
        if stored_log_dir:
            checkpoint_tensorboard_log_dir = Path(stored_log_dir)

    if resume_path is not None:
        remaining_steps = max(0, total_steps - global_step)
        # AdamW keeps the original warmup LR as ``initial_lr``. Replace it
        # with the checkpoint's current LR before constructing the scheduler.
        for param_group in optimizer.param_groups:
            param_group["initial_lr"] = param_group["lr"]
        scheduler = continuation_schedule(
            optimizer,
            remaining_steps,
            train_config.min_learning_rate,
        )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "resume_checkpoint": str(resume_path),
                        "resume_step": global_step,
                        "resume_learning_rate": optimizer.param_groups[0]["lr"],
                        "remaining_steps": remaining_steps,
                        "scheduler": "continuation_cosine",
                    }
                )
            )
    else:
        scheduler = two_stage_cosine_schedule(
            optimizer,
            optimizer_steps_per_epoch,
            total_steps,
            train_config.warmup_ratio,
            train_config.cosine_restart_epoch,
            train_config.cosine_stage_min_ratio,
        )

    writer = None
    tensorboard_log_dir: Path | None = None
    if rank == 0:
        if train_config.tensorboard_enabled:
            tensorboard_log_dir = checkpoint_tensorboard_log_dir
            if tensorboard_log_dir is None:
                run_name = f"meteotime-v1-{time.strftime('%Y%m%d-%H%M%S')}"
                tensorboard_log_dir = train_config.tensorboard_log_dir / run_name
            writer_kwargs = {
                "log_dir": str(tensorboard_log_dir),
                "flush_secs": train_config.tensorboard_flush_secs,
            }
            if resume_path is not None:
                writer_kwargs["purge_step"] = global_step
            writer = SummaryWriter(**writer_kwargs)
            writer.add_text(
                "config",
                "```json\n"
                + json.dumps(
                    {
                        "data": asdict(data_config),
                        "model": asdict(model_config),
                        "train": asdict(train_config),
                    },
                    indent=2,
                    default=str,
                )
                + "\n```",
                global_step,
            )
        print(
            json.dumps(
                {
                    "parameters": (
                        model.module.num_parameters
                        if isinstance(model, DistributedDataParallel)
                        else model.num_parameters
                    ),
                    "world_size": world_size,
                    "micro_batch_per_gpu": train_config.batch_size,
                    "effective_batch": train_config.batch_size
                    * world_size
                    * train_config.gradient_accumulation,
                    "precision": train_config.precision,
                    "configured_learning_rate": float(
                        f"{train_config.learning_rate:.2e}"
                    ),
                    "effective_learning_rate": float(
                        f"{optimizer.param_groups[0]['lr']:.2e}"
                    ),
                    "resume_checkpoint": str(resume_path) if resume_path else None,
                    "tensorboard_log_dir": str(tensorboard_log_dir) if writer else None,
                },
                indent=2,
            )
        )

    optimizer.zero_grad(set_to_none=True)
    last_validation_step = -1
    for epoch in range(start_epoch, train_config.epochs):
        dataset.set_epoch(epoch)
        epoch_start = time.perf_counter()
        last_log_time = epoch_start
        latest_grad_norm = float("nan")
        epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        epoch_batches = 0
        log_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        log_batches = 0
        for batch_idx, batch in enumerate(loader):
            if epoch == start_epoch and batch_idx < resume_batch_in_epoch:
                continue
            tensors = move_training_tensors(batch, device)
            should_step = (
                (batch_idx + 1) % train_config.gradient_accumulation == 0
                or batch_idx + 1 == train_config.batches_per_epoch
            )
            sync_context = (
                nullcontext()
                if should_step or not isinstance(model, DistributedDataParallel)
                else model.no_sync()
            )
            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(
                        tensors["context"],
                        tensors["context_mask"],
                        target=tensors["target"],
                        target_mask=tensors["target_mask"],
                    )
                    scaled_loss = loss / train_config.gradient_accumulation
                scaled_loss.backward()
            epoch_loss_sum += loss.detach().float()
            epoch_batches += 1
            log_loss_sum += loss.detach().double()
            log_batches += 1

            if should_step:
                grad_norm = clip_grad_norm_fp32(
                    tuple(model.parameters()), train_config.max_grad_norm
                )
                latest_grad_norm = float(grad_norm.detach().float().item())
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if (batch_idx + 1) % train_config.log_every == 0:
                log_totals = torch.tensor(
                    [log_loss_sum.item(), log_batches],
                    device=device,
                    dtype=torch.float64,
                )
                if world_size > 1:
                    dist.all_reduce(log_totals, op=dist.ReduceOp.SUM)
                mean_loss = (log_totals[0] / log_totals[1].clamp_min(1)).item()
                log_loss_sum.zero_()
                log_batches = 0
                torch.cuda.synchronize(device)
                now = time.perf_counter()
                elapsed = now - epoch_start
                interval = max(now - last_log_time, 1e-6)
                samples_per_second = (
                    train_config.log_every
                    * train_config.batch_size
                    * world_size
                    / interval
                )
                last_log_time = now
                if rank == 0:
                    memory_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                    learning_rate = scheduler.get_last_lr()[0]
                    display_learning_rate = float(f"{learning_rate:.2e}")
                    print(
                        json.dumps(
                            {
                                "epoch": epoch + 1,
                                "batch": batch_idx + 1,
                                "optimizer_step": global_step,
                                "loss": mean_loss,
                                "loss_window_batches": train_config.log_every,
                                "learning_rate": display_learning_rate,
                                "gradient_norm": round(latest_grad_norm, 3),
                                "context_length_last_batch": int(
                                    batch["context"].shape[1]
                                ),
                                "samples_per_second": round(samples_per_second, 2),
                                "elapsed_seconds": round(elapsed, 2),
                                "max_memory_gb": round(memory_gb, 3),
                            }
                        )
                    )
                    if writer is not None:
                        writer.add_scalar("train/loss", mean_loss, global_step)
                        writer.add_scalar("train/learning_rate", learning_rate, global_step)
                        writer.add_scalar("train/gradient_norm", latest_grad_norm, global_step)
                        writer.add_scalar(
                            "data/context_length",
                            int(batch["context"].shape[1]),
                            global_step,
                        )
                        writer.add_scalar(
                            "performance/samples_per_second",
                            samples_per_second,
                            global_step,
                        )
                        writer.add_scalar("system/max_memory_gb", memory_gb, global_step)
            if (
                should_step
                and global_step % train_config.validation_every_steps == 0
            ):
                validation_start = time.perf_counter()
                validation_metrics = run_validation(
                    model,
                    validation_loader,
                    device,
                    world_size,
                    tuple(data_config.target_variable_weights),
                )
                validation_seconds = time.perf_counter() - validation_start
                last_validation_step = global_step
                last_log_time = time.perf_counter()
                if rank == 0:
                    record_validation(
                        validation_metrics,
                        global_step,
                        validation_seconds,
                        len(validation_schedule),
                        writer,
                    )
                    current_validation_loss = validation_metrics["pinball_loss"]
                    if current_validation_loss < best_validation_loss:
                        best_validation_loss = current_validation_loss
                        path = save_checkpoint(
                            train_config.output_dir / "best.pt",
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            batch_idx + 1,
                            global_step,
                            model_config,
                            data_config,
                            train_config,
                            best_validation_loss,
                            tensorboard_log_dir,
                        )
                        print(
                            json.dumps(
                                {
                                    "checkpoint": str(path),
                                    "optimizer_step": global_step,
                                    "best_validation_loss": best_validation_loss,
                                }
                            )
                        )
                    else:
                        print(
                            json.dumps(
                                {
                                    "checkpoint": "skipped",
                                    "optimizer_step": global_step,
                                    "validation_loss": current_validation_loss,
                                    "best_validation_loss": best_validation_loss,
                                }
                            )
                        )
        epoch_mean_loss = epoch_loss_sum / max(epoch_batches, 1)
        if world_size > 1:
            dist.all_reduce(epoch_mean_loss, op=dist.ReduceOp.SUM)
            epoch_mean_loss /= world_size
        if last_validation_step != global_step:
            validation_start = time.perf_counter()
            validation_metrics = run_validation(
                model,
                validation_loader,
                device,
                world_size,
                tuple(data_config.target_variable_weights),
            )
            validation_seconds = time.perf_counter() - validation_start
            last_validation_step = global_step
            if rank == 0:
                record_validation(
                    validation_metrics,
                    global_step,
                    validation_seconds,
                    len(validation_schedule),
                    writer,
                )
                current_validation_loss = validation_metrics["pinball_loss"]
                if current_validation_loss < best_validation_loss:
                    best_validation_loss = current_validation_loss
                    path = save_checkpoint(
                        train_config.output_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        train_config.batches_per_epoch,
                        global_step,
                        model_config,
                        data_config,
                        train_config,
                        best_validation_loss,
                        tensorboard_log_dir,
                    )
                    print(
                        json.dumps(
                            {
                                "checkpoint": str(path),
                                "optimizer_step": global_step,
                                "best_validation_loss": best_validation_loss,
                            }
                        )
                    )
                else:
                    print(
                        json.dumps(
                            {
                                "checkpoint": "skipped",
                                "optimizer_step": global_step,
                                "validation_loss": current_validation_loss,
                                "best_validation_loss": best_validation_loss,
                            }
                        )
                    )
        if rank == 0:
            if writer is not None:
                writer.add_scalar(
                    "epoch/train_loss",
                    epoch_mean_loss.item(),
                    epoch + 1,
                )
                writer.add_scalar(
                    "epoch/duration_seconds",
                    time.perf_counter() - epoch_start,
                    epoch + 1,
                )
                writer.flush()
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])
        resume_batch_in_epoch = 0

    if world_size > 1:
        dist.destroy_process_group()
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()

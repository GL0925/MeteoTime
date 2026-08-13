from __future__ import annotations

import torch


def weather_collate(batch: list[dict]) -> dict:
    if not batch:
        raise ValueError("cannot collate an empty batch")
    max_context = max(sample["context"].numel() for sample in batch)
    prediction_length = batch[0]["target"].numel()
    if any(sample["target"].numel() != prediction_length for sample in batch):
        raise ValueError("all targets in a batch must have equal length")

    contexts = torch.zeros(len(batch), max_context, dtype=torch.float32)
    context_masks = torch.zeros(len(batch), max_context, dtype=torch.bool)
    for index, sample in enumerate(batch):
        length = sample["context"].numel()
        contexts[index, -length:] = sample["context"]
        context_masks[index, -length:] = sample["context_mask"]

    return {
        "context": contexts,
        "target": torch.stack([sample["target"] for sample in batch]),
        "context_mask": context_masks,
        "target_mask": torch.stack([sample["target_mask"] for sample in batch]),
        "loc": torch.stack([sample["loc"] for sample in batch]),
        "scale": torch.stack([sample["scale"] for sample in batch]),
        "source_id": torch.tensor([sample["source_id"] for sample in batch]),
        "variable_id": torch.tensor([sample["variable_id"] for sample in batch]),
        "source_name": [sample["source_name"] for sample in batch],
        "target_variable": [sample["target_variable"] for sample in batch],
        "item_id": [sample["item_id"] for sample in batch],
        "window_start": torch.tensor([sample["window_start"] for sample in batch]),
        "context_length": torch.tensor([sample["context"].numel() for sample in batch]),
    }

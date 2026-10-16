from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR
from transformers import get_scheduler


def _cosine_with_min_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float,
) -> float:
    if num_training_steps <= 0:
        return 1.0

    if num_warmup_steps > 0 and current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))

    decay_steps = max(1, num_training_steps - num_warmup_steps)
    progress = float(current_step - num_warmup_steps) / float(decay_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_lr_scheduler(args, optimizer, num_training_steps: int):
    """Build the configured learning-rate scheduler.

    transformers' cosine scheduler decays to zero.  The custom
    cosine_with_min_lr variant keeps the same warmup/cosine shape but floors
    the final LR at args.min_learning_rate.
    """
    scheduler_name = str(args.lr_scheduler_type).strip().lower()
    if scheduler_name != "cosine_with_min_lr":
        return get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=max(num_training_steps, 1),
        )

    min_lr_ratio = float(args.min_learning_rate) / float(args.learning_rate)
    return LambdaLR(
        optimizer,
        lambda step: _cosine_with_min_lr_lambda(
            step,
            num_warmup_steps=int(args.num_warmup_steps),
            num_training_steps=max(int(num_training_steps), 1),
            min_lr_ratio=min_lr_ratio,
        ),
    )

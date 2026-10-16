"""Pure tensor transforms used by specificity adapter interventions."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch


def weighted_cat_factors(
    source_factors: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    weights: Sequence[float],
    scalings: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not source_factors or not (
        len(source_factors) == len(weights) == len(scalings)
    ):
        raise ValueError("source_factors, weights, and scalings must have the same non-zero length.")
    cat_a = torch.cat(
        [a * (float(weight) * float(scaling)) for (a, _), weight, scaling in zip(source_factors, weights, scalings)],
        dim=0,
    )
    cat_b = torch.cat([b for _, b in source_factors], dim=1)
    return cat_a, cat_b


def select_module_cat_factors(
    background: Tuple[torch.Tensor, torch.Tensor],
    donor: Tuple[torch.Tensor, torch.Tensor],
    *,
    use_donor: bool,
    background_scaling: float = 1.0,
    donor_scaling: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    weights = [0.0, 1.0] if use_donor else [1.0, 0.0]
    return weighted_cat_factors(
        [background, donor],
        weights,
        [background_scaling, donor_scaling],
    )

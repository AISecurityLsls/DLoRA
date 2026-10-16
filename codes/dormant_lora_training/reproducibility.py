from __future__ import annotations

import random
from typing import Dict

import torch


def configure_reproducibility(seed: int) -> Dict[str, object]:
    """Seed all RNG sources controlled by this codebase.

    This intentionally does not change numeric backend settings such as TF32,
    cuDNN benchmark mode, CUBLAS workspace config, or deterministic kernel
    selection.  It only fixes random-number generators used for initialization,
    dropout, data sampling, and random trigger placement.
    """
    seed = int(seed)

    random.seed(seed)
    numpy_seeded = False
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
        numpy_seeded = True
    except Exception:
        numpy_seeded = False

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_seeded": numpy_seeded,
        "torch_seeded": True,
        "cuda_seeded": bool(torch.cuda.is_available()),
        "numeric_backend_unchanged": True,
    }

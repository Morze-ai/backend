"""Sets deterministic random seeds across backends."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.utils.torch_runtime import prepare_torch_import

prepare_torch_import()


def set_global_seed(seed: int) -> None:
    """Sets the random seed for all relevant libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

"""Deterministic RNG seeding for paired inference policy sweeps."""

import random

import numpy as np
import torch


def derive_sweep_batch_seed(seed: int, repeat: int, batch_idx: int) -> int:
    """Derive a stable RNG seed from the sweep coordinates."""
    return (
        int(seed) + 1_000_003 * int(repeat) + 1_009 * (int(batch_idx) + 1)
    ) % (2**63 - 1)


def seed_sweep_rngs(seed: int, repeat: int, batch_idx: int) -> int:
    """Seed Python, NumPy, Torch CPU, and all CUDA generators."""
    derived = derive_sweep_batch_seed(seed, repeat, batch_idx)
    random.seed(derived)
    np.random.seed(derived % (2**32))
    torch.manual_seed(derived)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived)
    return derived

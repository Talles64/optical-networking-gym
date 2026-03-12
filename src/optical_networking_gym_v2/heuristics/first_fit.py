from __future__ import annotations

import numpy as np


def select_first_fit_action(mask: np.ndarray) -> int:
    candidate_mask = np.asarray(mask, dtype=np.uint8)
    if candidate_mask.ndim != 1:
        raise ValueError("mask must be a one-dimensional array")
    if candidate_mask.size == 0:
        raise ValueError("mask must not be empty")

    reject_action = int(candidate_mask.size - 1)
    valid_actions = np.flatnonzero(candidate_mask[:-1])
    if valid_actions.size == 0:
        return reject_action
    return int(valid_actions[0])
__all__ = ["select_first_fit_action"]

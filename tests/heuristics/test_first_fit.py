from __future__ import annotations

import numpy as np

from optical_networking_gym_v2.heuristics import select_first_fit_action


def test_select_first_fit_action_chooses_first_valid_non_reject() -> None:
    mask = np.array([0, 0, 1, 1, 1], dtype=np.uint8)

    action = select_first_fit_action(mask)

    assert action == 2


def test_select_first_fit_action_returns_reject_when_only_reject_is_available() -> None:
    mask = np.array([0, 0, 0, 1], dtype=np.uint8)

    action = select_first_fit_action(mask)

    assert action == 3

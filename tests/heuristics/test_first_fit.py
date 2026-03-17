from __future__ import annotations

import numpy as np

from optical_networking_gym_v2 import make_env, set_topology_dir
from optical_networking_gym_v2.heuristics import select_first_fit_action
from optical_networking_gym_v2.optical.first_fit import (
    shortest_available_path_first_fit_best_modulation,
)
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = PROJECT_ROOT.parent / "examples" / "topologies"


def test_select_first_fit_action_chooses_first_valid_non_reject() -> None:
    mask = np.array([0, 0, 1, 1, 1], dtype=np.uint8)

    action = select_first_fit_action(mask)

    assert action == 2


def test_select_first_fit_action_returns_reject_when_only_reject_is_available() -> None:
    mask = np.array([0, 0, 0, 1], dtype=np.uint8)

    action = select_first_fit_action(mask)

    assert action == 3


def test_shortest_available_path_first_fit_best_modulation_matches_mask_first_fit() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="BPSK,QPSK,8QAM,16QAM",
        seed=7,
        bit_rates=(10, 40, 100, 400),
        load=10.0,
        num_spectrum_resources=24,
        episode_length=6,
        modulations_to_consider=4,
        k_paths=2,
    )
    _, _ = env.reset(seed=7)

    for _step in range(6):
        mask = env.action_masks()
        assert mask is not None

        action_from_mask = select_first_fit_action(mask)
        action_from_env, blocked_due_to_resources, blocked_due_to_qot = (
            shortest_available_path_first_fit_best_modulation(env)
        )

        assert action_from_env == action_from_mask
        assert blocked_due_to_resources is False
        assert blocked_due_to_qot is False

        _, _, terminated, truncated, _ = env.step(action_from_env)
        if terminated or truncated:
            break


def test_shortest_available_path_first_fit_best_modulation_returns_reject_when_resources_are_unavailable() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="BPSK",
        seed=11,
        bit_rates=(400,),
        load=10.0,
        num_spectrum_resources=1,
        episode_length=1,
        modulations_to_consider=1,
        k_paths=2,
    )
    _, _ = env.reset(seed=11)

    action, blocked_due_to_resources, blocked_due_to_qot = (
        shortest_available_path_first_fit_best_modulation(env)
    )

    assert action == env.action_space.n - 1
    assert blocked_due_to_resources is True
    assert blocked_due_to_qot is False

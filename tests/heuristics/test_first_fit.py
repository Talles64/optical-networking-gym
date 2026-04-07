from __future__ import annotations

import numpy as np

from optical_networking_gym_v2 import make_env, set_topology_dir
from optical_networking_gym_v2.heuristics import (
    build_runtime_heuristic_context,
    select_first_fit_action,
    select_first_fit_runtime_action,
    select_highest_snr_first_fit_runtime_action,
    select_ksp_best_mod_last_fit_runtime_action,
    select_load_balancing_runtime_action,
    select_lowest_fragmentation_runtime_action,
    select_random_action,
    select_random_runtime_action,
)
from optical_networking_gym_v2.runtime.action_codec import encode_action
from optical_networking_gym_v2.runtime.request_analysis import PATH_FEATURE_NAMES, PATH_SLOT_FEATURE_NAMES
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


def test_select_random_action_chooses_valid_non_reject_action() -> None:
    rng = np.random.default_rng(7)
    mask = np.array([0, 1, 0, 1], dtype=np.uint8)

    action = select_random_action(mask, rng=rng)

    assert action == 1


def test_select_random_action_returns_reject_when_only_reject_is_available() -> None:
    rng = np.random.default_rng(9)
    mask = np.array([0, 0, 0, 1], dtype=np.uint8)

    action = select_random_action(mask, rng=rng)

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
        action_from_runtime = select_first_fit_runtime_action(env.heuristic_context())
        action_from_env, blocked_due_to_resources, blocked_due_to_qot = (
            shortest_available_path_first_fit_best_modulation(env)
        )

        assert action_from_env == action_from_mask
        assert action_from_runtime == action_from_mask
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


def test_runtime_heuristic_context_exposes_current_analysis_without_extra_setup() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        seed=5,
        bit_rates=(40,),
        load=10.0,
        num_spectrum_resources=24,
        episode_length=2,
        modulations_to_consider=2,
        k_paths=2,
    )
    _, _ = env.reset(seed=5)

    context = build_runtime_heuristic_context(env)
    action = select_first_fit_runtime_action(context)
    metrics = context.selected_candidate_metrics(action)

    assert context.analysis is env.simulator.current_analysis
    assert context.request is env.simulator.current_request
    assert metrics is not None


def test_runtime_random_action_can_run_without_materialized_mask() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        seed=13,
        bit_rates=(40,),
        load=10.0,
        num_spectrum_resources=24,
        episode_length=2,
        modulations_to_consider=2,
        k_paths=2,
        enable_action_mask=False,
    )
    _, info = env.reset(seed=13)

    assert info["mask"] is None
    assert env.action_masks() is None

    action = select_random_runtime_action(env.heuristic_context(), rng=np.random.default_rng(13))

    assert 0 <= action < env.action_space.n


def test_runtime_load_balancing_action_matches_expected_candidate_ordering() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        seed=17,
        bit_rates=(40,),
        load=10.0,
        num_spectrum_resources=24,
        episode_length=2,
        modulations_to_consider=2,
        k_paths=2,
    )
    _, _ = env.reset(seed=17)

    context = build_runtime_heuristic_context(env)
    action = select_load_balancing_runtime_action(context)

    path_link_util_mean_index = int(PATH_FEATURE_NAMES.index("path_link_util_mean"))
    path_link_util_max_index = int(PATH_FEATURE_NAMES.index("path_link_util_max"))
    path_common_free_ratio_index = int(PATH_FEATURE_NAMES.index("path_common_free_ratio"))

    expected_candidates: list[tuple[tuple[float, float, float, float, float, int], int]] = []
    valid_flags = context.analysis.qot_valid_starts
    for path_index, _path in enumerate(context.analysis.paths):
        for modulation_offset, modulation_index in enumerate(context.analysis.modulation_indices):
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            for initial_slot in candidate_indices.tolist():
                encoded = encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                )
                metrics = context.selected_candidate_metrics(encoded)
                assert metrics is not None
                path_features = context.analysis.path_features[path_index]
                expected_candidates.append(
                    (
                        (
                            float(path_features[path_link_util_max_index]),
                            float(path_features[path_link_util_mean_index]),
                            -float(path_features[path_common_free_ratio_index]),
                            float(metrics.fragmentation_damage_num_blocks),
                            float(metrics.fragmentation_damage_largest_block),
                            int(encoded),
                        ),
                        int(encoded),
                    )
                )

    assert expected_candidates
    expected_action = min(expected_candidates, key=lambda item: item[0])[1]
    assert action == expected_action


def test_runtime_load_balancing_action_returns_reject_when_only_reject_is_available() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="BPSK",
        seed=19,
        bit_rates=(400,),
        load=10.0,
        num_spectrum_resources=1,
        episode_length=1,
        modulations_to_consider=1,
        k_paths=2,
    )
    _, _ = env.reset(seed=19)

    action = select_load_balancing_runtime_action(env.heuristic_context())

    assert action == env.action_space.n - 1


def test_runtime_highest_snr_first_fit_action_chooses_highest_osnr_margin_with_first_fit_tiebreak() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        seed=23,
        bit_rates=(40,),
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        episode_length=2,
        modulations_to_consider=2,
        k_paths=2,
    )
    _, _ = env.reset(seed=23)

    context = build_runtime_heuristic_context(env)
    action = select_highest_snr_first_fit_runtime_action(context)

    expected_candidates: list[tuple[tuple[float, int, int, int, int], int]] = []
    valid_flags = context.analysis.qot_valid_starts
    for path_index, _path in enumerate(context.analysis.paths):
        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            for initial_slot in candidate_indices.tolist():
                encoded = encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                )
                metrics = context.selected_candidate_metrics(encoded)
                assert metrics is not None
                expected_candidates.append(
                    (
                        (
                            float(metrics.osnr_margin),
                            -int(path_index),
                            -int(modulation_offset),
                            -int(initial_slot),
                            -int(encoded),
                        ),
                        int(encoded),
                    )
                )

    assert expected_candidates
    expected_action = max(expected_candidates, key=lambda item: item[0])[1]
    assert action == expected_action


def test_runtime_highest_snr_first_fit_returns_reject_when_only_reject_is_available() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="BPSK",
        seed=29,
        bit_rates=(400,),
        load=10.0,
        num_spectrum_resources=1,
        episode_length=1,
        modulations_to_consider=1,
        k_paths=2,
    )
    _, _ = env.reset(seed=29)

    action = select_highest_snr_first_fit_runtime_action(env.heuristic_context())

    assert action == env.action_space.n - 1


def test_runtime_highest_snr_first_fit_can_run_without_materialized_mask() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        seed=31,
        bit_rates=(40,),
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        episode_length=2,
        modulations_to_consider=2,
        k_paths=2,
        enable_action_mask=False,
    )
    _, info = env.reset(seed=31)

    assert info["mask"] is None
    assert env.action_masks() is None

    action = select_highest_snr_first_fit_runtime_action(env.heuristic_context())

    assert 0 <= action < env.action_space.n


def test_runtime_ksp_best_mod_last_fit_chooses_first_ksp_path_best_modulation_last_slot() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        seed=37,
        bit_rates=(40,),
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        episode_length=2,
        modulations_to_consider=2,
        k_paths=2,
    )
    _, _ = env.reset(seed=37)

    context = build_runtime_heuristic_context(env)
    action = select_ksp_best_mod_last_fit_runtime_action(context)

    valid_flags = context.analysis.qot_valid_starts
    expected_action = context.reject_action
    for path_index, _path in enumerate(context.analysis.paths):
        offsets = sorted(
            range(len(context.analysis.modulation_indices)),
            key=lambda offset: (
                -int(context.config.modulations[context.analysis.modulation_indices[offset]].spectral_efficiency),
                int(offset),
            ),
        )
        for modulation_offset in offsets:
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            if candidate_indices.size == 0:
                continue
            expected_action = encode_action(
                context.config,
                path_index=path_index,
                modulation_offset=modulation_offset,
                initial_slot=int(candidate_indices[-1]),
            )
            break
        if expected_action != context.reject_action:
            break

    assert action == expected_action


def test_runtime_ksp_best_mod_last_fit_returns_reject_when_only_reject_is_available() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="BPSK",
        seed=41,
        bit_rates=(400,),
        load=10.0,
        num_spectrum_resources=1,
        episode_length=1,
        modulations_to_consider=1,
        k_paths=2,
    )
    _, _ = env.reset(seed=41)

    action = select_ksp_best_mod_last_fit_runtime_action(env.heuristic_context())

    assert action == env.action_space.n - 1


def test_runtime_lowest_fragmentation_chooses_smallest_fragmentation_tuple() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        seed=43,
        bit_rates=(40,),
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        episode_length=2,
        modulations_to_consider=2,
        k_paths=2,
    )
    _, _ = env.reset(seed=43)

    context = build_runtime_heuristic_context(env)
    action = select_lowest_fragmentation_runtime_action(context)

    local_fragmentation_index = int(PATH_SLOT_FEATURE_NAMES.index("local_fragmentation"))
    valid_flags = context.analysis.qot_valid_starts
    expected_candidates: list[tuple[tuple[float, ...], int]] = []
    for path_index, _path in enumerate(context.analysis.paths):
        path_features = context.analysis.path_features[path_index]
        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            for initial_slot in candidate_indices.tolist():
                encoded = encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                )
                metrics = context.selected_candidate_metrics(encoded)
                assert metrics is not None
                slot_features = context.analysis.path_slot_features[path_index, int(initial_slot)]
                expected_candidates.append(
                    (
                        (
                            float(metrics.fragmentation_damage_num_blocks),
                            float(metrics.fragmentation_damage_largest_block),
                            float(slot_features[local_fragmentation_index]),
                            float(path_features[int(PATH_FEATURE_NAMES.index("path_common_num_blocks_norm"))]),
                            float(path_features[int(PATH_FEATURE_NAMES.index("path_route_cuts_norm"))]),
                            -float(path_features[int(PATH_FEATURE_NAMES.index("path_route_rss"))]),
                            float(
                                context.analysis.required_slots_by_path_mod[path_index, modulation_offset]
                            ),
                            float(path_features[int(PATH_FEATURE_NAMES.index("path_link_util_max"))]),
                            -float(np.clip(metrics.osnr_margin, 0.0, 3.0)),
                            float(encoded),
                        ),
                        int(encoded),
                    )
                )

    assert expected_candidates
    expected_action = min(expected_candidates, key=lambda item: item[0])[1]
    assert action == expected_action


def test_runtime_lowest_fragmentation_returns_reject_when_only_reject_is_available() -> None:
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="BPSK",
        seed=47,
        bit_rates=(400,),
        load=10.0,
        num_spectrum_resources=1,
        episode_length=1,
        modulations_to_consider=1,
        k_paths=2,
    )
    _, _ = env.reset(seed=47)

    action = select_lowest_fragmentation_runtime_action(env.heuristic_context())

    assert action == env.action_space.n - 1

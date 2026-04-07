from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import runpy

from optical_networking_gym_v2.judge.heuristic_judge import (
    CandidateMetricsPayload,
    DecodedActionPayload,
    JudgeCandidate,
    JudgeVerdict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "llm" / "online_heuristic_judge.py"


def _load_module() -> dict[str, object]:
    return runpy.run_path(str(EXAMPLE_PATH))


class FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def judge(self, payload):
        self.calls += 1
        winner = payload.candidates[0].candidate_id
        return JudgeVerdict(
            winner_candidate_id=winner,
            confidence=0.91,
            decision_basis="balanced_tie_break",
        )


class CountingEnv:
    def __init__(self, env, counter: dict[str, int]) -> None:
        self._env = env
        self._counter = counter

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def step(self, action: int):
        self._counter["step_calls"] += 1
        return self._env.step(action)


class BaselineTripEnv:
    def __init__(
        self,
        env,
        counter: dict[str, int],
        *,
        trip_after_steps: int,
        blocking_rate: float,
    ) -> None:
        self._env = env
        self._counter = counter
        self._trip_after_steps = trip_after_steps
        self._blocking_rate = blocking_rate

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def step(self, action: int):
        observation, reward, terminated, truncated, info = self._env.step(action)
        self._counter["step_calls"] += 1
        mutated_info = dict(info)
        if self._counter["step_calls"] >= self._trip_after_steps:
            mutated_info["episode_service_blocking_rate"] = self._blocking_rate
        return observation, reward, terminated, truncated, mutated_info


def _manual_candidate(
    *,
    heuristic_name: str,
    raw_action: int,
    required_slots: int,
    route_pressure_score: float,
    local_damage_score: float,
    plausibility_score: float,
    path_common_free_ratio: float,
    qot_margin_clipped_db: float = 1.0,
    qot_safe_now: bool = True,
    is_pareto_dominated: bool = False,
    fragmentation_added_blocks: int = 0,
    largest_block_loss_slots: int = 0,
    path_common_num_blocks_norm: float = 0.2,
    path_route_cuts_norm: float = 0.2,
    path_route_rss: float | None = None,
    path_link_util_mean: float | None = None,
    path_link_util_max: float | None = None,
    common_block_length_norm: float = 0.4,
    local_fragmentation: float | None = None,
    left_free_span_norm: float = 0.1,
    right_free_span_norm: float = 0.2,
    future_risk_band: str = "high",
    proposed_by: tuple[str, ...] | None = None,
    path_index: int = 0,
    modulation_name: str = "QPSK",
    initial_slot: int | None = None,
) -> JudgeCandidate:
    resolved_path_link_util_max = route_pressure_score if path_link_util_max is None else path_link_util_max
    resolved_path_link_util_mean = max(route_pressure_score - 0.05, 0.0) if path_link_util_mean is None else path_link_util_mean
    resolved_path_route_rss = max(0.0, 1.0 - route_pressure_score) if path_route_rss is None else path_route_rss
    resolved_local_fragmentation = local_damage_score if local_fragmentation is None else local_fragmentation
    resolved_initial_slot = raw_action if initial_slot is None else initial_slot
    return JudgeCandidate(
        heuristic_name=heuristic_name,
        proposed_by=proposed_by or (heuristic_name,),
        raw_action=raw_action,
        is_reject=not qot_safe_now,
        decoded_action=DecodedActionPayload(
            path_index=path_index,
            path_rank_k=0,
            path_node_names=("A", "B"),
            path_hops=1,
            path_length_km=100.0,
            source_name="A",
            destination_name="B",
            modulation_index=0,
            modulation_name=modulation_name,
            modulation_spectral_efficiency=2,
            initial_slot=resolved_initial_slot,
            required_slots=required_slots,
            slot_end_exclusive=resolved_initial_slot + required_slots,
        ),
        metrics=CandidateMetricsPayload(
            required_slots=required_slots,
            path_link_util_mean=resolved_path_link_util_mean,
            path_link_util_max=resolved_path_link_util_max,
            path_common_free_ratio=path_common_free_ratio,
            path_common_largest_block_ratio=0.2,
            path_common_num_blocks_norm=path_common_num_blocks_norm,
            path_route_cuts_norm=path_route_cuts_norm,
            path_route_rss=resolved_path_route_rss,
            osnr_margin_db=qot_margin_clipped_db,
            nli_share=0.2,
            worst_link_nli_share=0.2,
            common_block_length_norm=common_block_length_norm,
            left_free_span_norm=left_free_span_norm,
            right_free_span_norm=right_free_span_norm,
            local_fragmentation=resolved_local_fragmentation,
            fragmentation_damage_num_blocks=0.0,
            fragmentation_damage_largest_block=0.0,
            fragmentation_added_blocks=fragmentation_added_blocks,
            largest_block_loss_slots=largest_block_loss_slots,
            qot_safe_now=qot_safe_now,
            qot_band="moderate" if qot_safe_now else "unsafe",
            qot_margin_clipped_db=qot_margin_clipped_db,
            qot_excess_db_over_floor=max(qot_margin_clipped_db - 0.5, 0.0),
            slot_cost_vs_best=required_slots - 1,
            slot_ratio_vs_best=float(required_slots),
            local_damage_score=local_damage_score,
            route_pressure_score=route_pressure_score,
            future_risk_band=future_risk_band,
            is_pareto_dominated=is_pareto_dominated,
            plausibility_score=plausibility_score,
            blocking_proxy_score=plausibility_score,
        ),
    )


def test_plausible_candidates_keep_axis_leaders_even_when_one_is_below_prune_gap() -> None:
    module = _load_module()
    plausible_candidates = module["_plausible_candidates"]

    candidates = (
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=10,
            required_slots=2,
            route_pressure_score=0.29,
            local_damage_score=0.03,
            plausibility_score=0.91,
            path_common_free_ratio=0.50,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=11,
            required_slots=2,
            route_pressure_score=0.16,
            local_damage_score=0.10,
            plausibility_score=0.79,
            path_common_free_ratio=0.68,
        ),
        _manual_candidate(
            heuristic_name="ksp_best_mod_last_fit",
            raw_action=12,
            required_slots=2,
            route_pressure_score=0.22,
            local_damage_score=0.02,
            plausibility_score=0.80,
            path_common_free_ratio=0.58,
        ),
    )

    selected = plausible_candidates(candidates, prune_gap=0.10)

    assert {candidate.heuristic_name for candidate in selected} == {
        "lowest_fragmentation",
        "load_balancing",
        "ksp_best_mod_last_fit",
    }


def test_select_prompt_candidates_preserves_route_and_local_tradeoff_candidates() -> None:
    module = _load_module()
    select_prompt_candidates = module["_select_prompt_candidates"]

    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=10,
            required_slots=1,
            route_pressure_score=0.24,
            local_damage_score=0.05,
            plausibility_score=0.90,
            path_common_free_ratio=0.60,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=11,
            required_slots=1,
            route_pressure_score=0.16,
            local_damage_score=0.08,
            plausibility_score=0.84,
            path_common_free_ratio=0.67,
        ),
        _manual_candidate(
            heuristic_name="ksp_best_mod_last_fit",
            raw_action=12,
            required_slots=1,
            route_pressure_score=0.25,
            local_damage_score=0.03,
            plausibility_score=0.88,
            path_common_free_ratio=0.61,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=13,
            required_slots=2,
            route_pressure_score=0.18,
            local_damage_score=0.00,
            plausibility_score=0.82,
            path_common_free_ratio=0.71,
        ),
    )

    selected = select_prompt_candidates(
        candidates,
        min_prompt_candidates=3,
        max_prompt_candidates=4,
        third_candidate_gap=0.03,
        same_slot_route_advantage_material=0.015,
        same_slot_local_advantage_material=0.08,
        same_slot_common_free_penalty_material=0.05,
        extra_slot_route_advantage_material=0.03,
        extra_slot_local_advantage_material=0.10,
        extra_slot_common_free_advantage_material=0.08,
    )

    selected_names = tuple(candidate.heuristic_name for candidate in selected)

    assert selected_names[:3] == (
        "first_fit",
        "ksp_best_mod_last_fit",
        "load_balancing",
    )
    assert set(selected_names) >= {"ksp_best_mod_last_fit", "load_balancing", "first_fit"}


def test_select_prompt_candidates_skips_extra_slot_candidate_without_material_gain() -> None:
    module = _load_module()
    select_prompt_candidates = module["_select_prompt_candidates"]

    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=20,
            required_slots=1,
            route_pressure_score=0.20,
            local_damage_score=0.05,
            plausibility_score=0.92,
            path_common_free_ratio=0.60,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=21,
            required_slots=1,
            route_pressure_score=0.19,
            local_damage_score=0.06,
            plausibility_score=0.89,
            path_common_free_ratio=0.61,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=22,
            required_slots=2,
            route_pressure_score=0.20,
            local_damage_score=0.03,
            plausibility_score=0.84,
            path_common_free_ratio=0.64,
        ),
        _manual_candidate(
            heuristic_name="ksp_best_mod_last_fit",
            raw_action=23,
            required_slots=1,
            route_pressure_score=0.22,
            local_damage_score=0.04,
            plausibility_score=0.86,
            path_common_free_ratio=0.59,
        ),
    )

    selected = select_prompt_candidates(
        candidates,
        min_prompt_candidates=3,
        max_prompt_candidates=4,
        third_candidate_gap=0.03,
        same_slot_route_advantage_material=0.015,
        same_slot_local_advantage_material=0.08,
        same_slot_common_free_penalty_material=0.05,
        extra_slot_route_advantage_material=0.03,
        extra_slot_local_advantage_material=0.10,
        extra_slot_common_free_advantage_material=0.08,
    )

    assert tuple(candidate.heuristic_name for candidate in selected) == (
        "first_fit",
        "ksp_best_mod_last_fit",
        "load_balancing",
    )


def test_select_prompt_candidates_skips_extra_slot_candidate_with_route_only_gain() -> None:
    module = _load_module()
    select_prompt_candidates = module["_select_prompt_candidates"]

    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=24,
            required_slots=1,
            route_pressure_score=0.24,
            local_damage_score=0.05,
            plausibility_score=0.91,
            path_common_free_ratio=0.62,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=25,
            required_slots=1,
            route_pressure_score=0.20,
            local_damage_score=0.06,
            plausibility_score=0.89,
            path_common_free_ratio=0.63,
        ),
        _manual_candidate(
            heuristic_name="ksp_best_mod_last_fit",
            raw_action=26,
            required_slots=1,
            route_pressure_score=0.22,
            local_damage_score=0.04,
            plausibility_score=0.87,
            path_common_free_ratio=0.61,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=27,
            required_slots=2,
            route_pressure_score=0.12,
            local_damage_score=0.035,
            plausibility_score=0.85,
            path_common_free_ratio=0.64,
        ),
    )

    selected = select_prompt_candidates(
        candidates,
        min_prompt_candidates=3,
        max_prompt_candidates=4,
        third_candidate_gap=0.03,
        same_slot_route_advantage_material=0.015,
        same_slot_local_advantage_material=0.08,
        same_slot_common_free_penalty_material=0.05,
        extra_slot_route_advantage_material=0.03,
        extra_slot_local_advantage_material=0.10,
        extra_slot_common_free_advantage_material=0.08,
    )

    assert tuple(candidate.heuristic_name for candidate in selected) == (
        "first_fit",
        "ksp_best_mod_last_fit",
        "load_balancing",
    )


def test_select_prompt_candidate_entries_include_balanced_anchor_when_distinct() -> None:
    module = _load_module()
    select_prompt_candidate_entries = module["_select_prompt_candidate_entries"]

    selected_entries = select_prompt_candidate_entries(
        (
            _manual_candidate(
                heuristic_name="first_fit",
                raw_action=95,
                required_slots=1,
                route_pressure_score=0.22,
                local_damage_score=0.05,
                plausibility_score=0.93,
                path_common_free_ratio=0.60,
                fragmentation_added_blocks=1,
                largest_block_loss_slots=1,
            ),
            _manual_candidate(
                heuristic_name="ksp_best_mod_last_fit",
                raw_action=96,
                required_slots=1,
                route_pressure_score=0.24,
                local_damage_score=0.02,
                plausibility_score=0.89,
                path_common_free_ratio=0.58,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
            ),
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=97,
                required_slots=1,
                route_pressure_score=0.16,
                local_damage_score=0.08,
                plausibility_score=0.87,
                path_common_free_ratio=0.67,
                fragmentation_added_blocks=1,
                largest_block_loss_slots=1,
            ),
        ),
        min_prompt_candidates=3,
        max_prompt_candidates=3,
        third_candidate_gap=0.03,
        same_slot_route_advantage_material=0.015,
        same_slot_local_advantage_material=0.08,
        same_slot_common_free_penalty_material=0.05,
        extra_slot_route_advantage_material=0.03,
        extra_slot_local_advantage_material=0.10,
        extra_slot_common_free_advantage_material=0.08,
    )

    assert tuple(role for role, _candidate in selected_entries) == (
        "balanced_anchor",
        "same_slot_preservation_anchor",
        "same_slot_route_challenger",
    )
    assert tuple(candidate.heuristic_name for _role, candidate in selected_entries) == (
        "first_fit",
        "ksp_best_mod_last_fit",
        "load_balancing",
    )


def test_select_prompt_candidate_entries_include_same_path_slot_variant_challenger() -> None:
    module = _load_module()
    select_prompt_candidate_entries = module["_select_prompt_candidate_entries"]

    selected_entries = select_prompt_candidate_entries(
        (
            _manual_candidate(
                heuristic_name="ksp_best_mod_last_fit",
                raw_action=110,
                initial_slot=20,
                required_slots=1,
                route_pressure_score=0.22,
                local_damage_score=0.02,
                local_fragmentation=0.02,
                path_common_free_ratio=0.58,
                largest_block_loss_slots=0,
                left_free_span_norm=0.12,
                right_free_span_norm=0.10,
                plausibility_score=0.90,
                path_index=0,
            ),
            _manual_candidate(
                heuristic_name="first_fit",
                raw_action=111,
                initial_slot=2,
                required_slots=1,
                route_pressure_score=0.22,
                local_damage_score=0.08,
                local_fragmentation=0.08,
                path_common_free_ratio=0.58,
                largest_block_loss_slots=4,
                left_free_span_norm=0.00,
                right_free_span_norm=0.22,
                plausibility_score=0.88,
                path_index=0,
            ),
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=112,
                required_slots=1,
                route_pressure_score=0.16,
                local_damage_score=0.09,
                path_common_free_ratio=0.63,
                fragmentation_added_blocks=1,
                largest_block_loss_slots=2,
                plausibility_score=0.89,
                path_index=1,
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=113,
                required_slots=1,
                route_pressure_score=0.18,
                local_damage_score=0.03,
                path_common_free_ratio=0.66,
                fragmentation_added_blocks=1,
                largest_block_loss_slots=1,
                plausibility_score=0.87,
                path_index=2,
            ),
        ),
        min_prompt_candidates=3,
        max_prompt_candidates=4,
        third_candidate_gap=0.03,
        same_slot_route_advantage_material=0.015,
        same_slot_local_advantage_material=0.08,
        same_slot_common_free_penalty_material=0.05,
        extra_slot_route_advantage_material=0.03,
        extra_slot_local_advantage_material=0.10,
        extra_slot_common_free_advantage_material=0.08,
    )

    selected_roles = {role for role, _candidate in selected_entries}
    selected_heuristics = {candidate.heuristic_name for _role, candidate in selected_entries}

    assert "same_path_slot_variant_challenger" in selected_roles
    assert "first_fit" in selected_heuristics
    assert "ksp_best_mod_last_fit" in selected_heuristics
    assert len(selected_entries) == 4


def test_select_prompt_candidate_entries_deduplicate_identical_actions() -> None:
    module = _load_module()
    select_prompt_candidate_entries = module["_select_prompt_candidate_entries"]

    selected_entries = select_prompt_candidate_entries(
        (
            _manual_candidate(
                heuristic_name="first_fit",
                raw_action=0,
                initial_slot=0,
                required_slots=1,
                route_pressure_score=0.21,
                local_damage_score=0.05,
                local_fragmentation=0.10,
                path_common_free_ratio=0.72,
                plausibility_score=0.91,
                path_index=0,
                modulation_name="64QAM",
                proposed_by=("first_fit", "load_balancing"),
            ),
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=0,
                initial_slot=0,
                required_slots=1,
                route_pressure_score=0.21,
                local_damage_score=0.05,
                local_fragmentation=0.10,
                path_common_free_ratio=0.72,
                plausibility_score=0.90,
                path_index=0,
                modulation_name="64QAM",
            ),
            _manual_candidate(
                heuristic_name="ksp_best_mod_last_fit",
                raw_action=319,
                initial_slot=319,
                required_slots=1,
                route_pressure_score=0.21,
                local_damage_score=0.05,
                local_fragmentation=0.10,
                path_common_free_ratio=0.72,
                plausibility_score=0.89,
                path_index=0,
                modulation_name="64QAM",
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=3401,
                initial_slot=161,
                required_slots=1,
                route_pressure_score=0.43,
                local_damage_score=0.03,
                local_fragmentation=0.20,
                path_common_free_ratio=0.11,
                plausibility_score=0.70,
                path_index=3,
                modulation_name="32QAM",
            ),
        ),
        min_prompt_candidates=3,
        max_prompt_candidates=3,
        third_candidate_gap=0.03,
        same_slot_route_advantage_material=0.015,
        same_slot_local_advantage_material=0.08,
        same_slot_common_free_penalty_material=0.05,
        extra_slot_route_advantage_material=0.03,
        extra_slot_local_advantage_material=0.10,
        extra_slot_common_free_advantage_material=0.08,
    )

    selected_actions = tuple(candidate.raw_action for _role, candidate in selected_entries)
    selected_heuristics = {candidate.heuristic_name for _role, candidate in selected_entries}

    assert len(selected_actions) == len(set(selected_actions))
    assert selected_heuristics >= {"first_fit", "ksp_best_mod_last_fit", "lowest_fragmentation"}


def test_select_prompt_candidate_entries_caps_same_slot_candidates_at_three() -> None:
    module = _load_module()
    select_prompt_candidate_entries = module["_select_prompt_candidate_entries"]

    selected_entries = select_prompt_candidate_entries(
        (
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=120,
                initial_slot=180,
                required_slots=2,
                route_pressure_score=0.50,
                local_damage_score=0.01,
                path_common_free_ratio=0.16,
                plausibility_score=0.90,
                path_index=3,
                modulation_name="16QAM",
            ),
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=121,
                initial_slot=40,
                required_slots=2,
                route_pressure_score=0.38,
                local_damage_score=0.07,
                path_common_free_ratio=0.22,
                plausibility_score=0.88,
                path_index=4,
                modulation_name="32QAM",
            ),
            _manual_candidate(
                heuristic_name="first_fit",
                raw_action=122,
                initial_slot=24,
                required_slots=2,
                route_pressure_score=0.43,
                local_damage_score=0.07,
                path_common_free_ratio=0.18,
                plausibility_score=0.87,
                path_index=0,
                modulation_name="64QAM",
            ),
            _manual_candidate(
                heuristic_name="ksp_best_mod_last_fit",
                raw_action=123,
                initial_slot=260,
                required_slots=2,
                route_pressure_score=0.43,
                local_damage_score=0.04,
                path_common_free_ratio=0.18,
                plausibility_score=0.875,
                path_index=0,
                modulation_name="64QAM",
            ),
        ),
        min_prompt_candidates=3,
        max_prompt_candidates=4,
        third_candidate_gap=0.03,
        same_slot_route_advantage_material=0.015,
        same_slot_local_advantage_material=0.08,
        same_slot_common_free_penalty_material=0.05,
        extra_slot_route_advantage_material=0.03,
        extra_slot_local_advantage_material=0.10,
        extra_slot_common_free_advantage_material=0.08,
    )

    assert len(selected_entries) == 3
    assert {candidate.heuristic_name for _role, candidate in selected_entries} == {
        "lowest_fragmentation",
        "load_balancing",
        "first_fit",
    }


def test_collect_semantic_warning_flags_marks_incoherent_same_slot_route_basis() -> None:
    module = _load_module()
    collect_semantic_warning_flags = module["_collect_semantic_warning_flags"]

    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=30,
            required_slots=1,
            route_pressure_score=0.21,
            local_damage_score=0.04,
            plausibility_score=0.92,
            path_common_free_ratio=0.58,
            proposed_by=("first_fit", "ksp_best_mod_last_fit"),
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=31,
            required_slots=1,
            route_pressure_score=0.17,
            local_damage_score=0.05,
            plausibility_score=0.88,
            path_common_free_ratio=0.60,
        ),
    )
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type("Ctx", (), {"request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(), "topology": type("Topo", (), {"node_names": ("A", "B")})()})(),
        topology_profile=None,
        operational_state=type("State", (), {
            "services_processed": 10,
            "episode_service_blocking_rate": 0.0,
            "network_util_mean": 0.1,
            "network_util_max": 0.2,
            "free_slots_ratio": 0.8,
        })(),
        global_regimes=None,
        candidates=candidates,
        candidate_ids=("GOOD", "BAD"),
    )
    verdict = JudgeVerdict(
        winner_candidate_id="GOOD",
        confidence=0.9,
        decision_basis="same_slot_route_advantage",
    )

    flags = collect_semantic_warning_flags(payload, verdict)

    assert "same_slot_route_basis_mismatch" in flags


def test_collect_semantic_warning_flags_marks_local_basis_without_fragmentation_support() -> None:
    module = _load_module()
    collect_semantic_warning_flags = module["_collect_semantic_warning_flags"]

    candidates = (
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=40,
            required_slots=1,
            route_pressure_score=0.18,
            local_damage_score=0.10,
            plausibility_score=0.89,
            path_common_free_ratio=0.60,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
            local_fragmentation=0.50,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=41,
            required_slots=1,
            route_pressure_score=0.19,
            local_damage_score=0.01,
            plausibility_score=0.87,
            path_common_free_ratio=0.59,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
            local_fragmentation=0.01,
        ),
    )
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type("Ctx", (), {"request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(), "topology": type("Topo", (), {"node_names": ("A", "B")})()})(),
        topology_profile=None,
        operational_state=type("State", (), {
            "services_processed": 10,
            "episode_service_blocking_rate": 0.0,
            "network_util_mean": 0.1,
            "network_util_max": 0.2,
            "free_slots_ratio": 0.8,
        })(),
        global_regimes=None,
        candidates=candidates,
        candidate_ids=("BASE", "LOCAL"),
    )
    verdict = JudgeVerdict(
        winner_candidate_id="LOCAL",
        confidence=0.9,
        decision_basis="same_slot_local_advantage",
    )

    flags = collect_semantic_warning_flags(payload, verdict)

    assert "same_slot_local_basis_mismatch" in flags


def test_build_judge_payload_exposes_non_prescriptive_prompt_context_bands() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 70,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.50,
                "network_util_max": 0.65,
                "free_slots_ratio": 0.50,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "high"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=50,
                required_slots=1,
                route_pressure_score=0.15,
                local_damage_score=0.10,
                plausibility_score=0.90,
                path_common_free_ratio=0.55,
                qot_margin_clipped_db=1.00,
            ),
            _manual_candidate(
                heuristic_name="ksp_best_mod_last_fit",
                raw_action=51,
                required_slots=1,
                route_pressure_score=0.17,
                local_damage_score=0.01,
                plausibility_score=0.89,
                path_common_free_ratio=0.62,
                qot_margin_clipped_db=0.95,
            ),
        ),
        candidate_ids=("ROUTE", "SAFE"),
    )
    context = payload.prompt_context.to_mapping()

    assert context["progress_ratio"] == 0.7
    assert context["congestion_band"] == "tight"
    assert context["same_slot_near_tie_band"] == "soft_tradeoff"
    assert context["same_slot_route_common_free_alignment"] == "split"
    assert context["route_gain_band_vs_same_slot_best"] == "small"
    assert context["local_gain_band_vs_same_slot_best"] == "material"
    assert context["same_slot_local_support_band"] == "partial"
    assert context["common_free_penalty_band_vs_same_slot_best"] == "material"
    assert context["future_feasibility_risk_band"] == "high"


def test_build_judge_payload_marks_missing_same_slot_local_support() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 70,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.50,
                "network_util_max": 0.65,
                "free_slots_ratio": 0.50,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "moderate"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=52,
                required_slots=1,
                route_pressure_score=0.15,
                local_damage_score=0.10,
                plausibility_score=0.90,
                path_common_free_ratio=0.60,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=53,
                required_slots=1,
                route_pressure_score=0.17,
                local_damage_score=0.01,
                plausibility_score=0.89,
                path_common_free_ratio=0.59,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
            ),
        ),
        candidate_ids=("ROUTE", "LOCAL"),
    )
    context = payload.prompt_context.to_mapping()

    assert context["route_gain_band_vs_same_slot_best"] == "small"
    assert context["local_gain_band_vs_same_slot_best"] == "material"
    assert context["same_slot_local_support_band"] == "none"
    assert context["same_slot_damage_axes_tie_band"] == "tied"
    assert context["same_slot_route_common_free_alignment"] == "aligned"


def test_build_judge_payload_treats_local_fragmentation_only_support_as_partial() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 71,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.50,
                "network_util_max": 0.65,
                "free_slots_ratio": 0.50,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "moderate"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=57,
                required_slots=1,
                route_pressure_score=0.15,
                local_damage_score=0.10,
                local_fragmentation=0.60,
                plausibility_score=0.90,
                path_common_free_ratio=0.70,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=58,
                required_slots=1,
                route_pressure_score=0.19,
                local_damage_score=0.04,
                local_fragmentation=0.20,
                plausibility_score=0.89,
                path_common_free_ratio=0.61,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
            ),
        ),
        candidate_ids=("ROUTE", "SAFE"),
        candidate_roles=("same_slot_route_challenger", "same_slot_preservation_anchor"),
    )
    prompt_mapping = payload.to_prompt_mapping()
    context = prompt_mapping["prompt_context"]
    candidates = {candidate["candidate_id"]: candidate for candidate in prompt_mapping["candidates"]}

    assert context["same_slot_local_support_band"] == "partial"
    assert context["same_slot_damage_axes_tie_band"] == "tied"
    assert context["same_slot_route_common_free_alignment"] == "aligned"
    assert "same_slot_preservation_leader" not in candidates["SAFE"]["candidate_roles"]
    assert "same_slot_local_damage_leader" not in candidates["SAFE"]["candidate_roles"]
    assert "same_slot_preservation_leader" not in candidates["ROUTE"]["candidate_roles"]


def test_build_judge_payload_does_not_promote_local_support_from_dominated_rival_only() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 100})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 57,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.42,
                "network_util_max": 0.58,
                "free_slots_ratio": 0.53,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "high"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=95,
                required_slots=2,
                route_pressure_score=0.49,
                local_damage_score=0.09,
                local_fragmentation=0.60,
                path_common_free_ratio=0.07,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
                plausibility_score=0.91,
            ),
            _manual_candidate(
                heuristic_name="first_fit",
                raw_action=96,
                required_slots=2,
                route_pressure_score=0.51,
                local_damage_score=0.39,
                local_fragmentation=0.20,
                path_common_free_ratio=0.07,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=2,
                plausibility_score=0.87,
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=97,
                required_slots=2,
                route_pressure_score=0.505,
                local_damage_score=0.03,
                local_fragmentation=0.20,
                path_common_free_ratio=0.04,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
                plausibility_score=0.89,
            ),
        ),
        candidate_ids=("ROUTE", "DOM", "LOCAL"),
        candidate_roles=(
            "same_slot_route_challenger",
            "same_slot_backfill_challenger",
            "same_slot_preservation_anchor",
        ),
    )
    prompt_mapping = payload.to_prompt_mapping()
    context = prompt_mapping["prompt_context"]
    candidates = {candidate["candidate_id"]: candidate for candidate in prompt_mapping["candidates"]}

    assert context["same_slot_local_support_band"] == "partial"
    assert "same_slot_preservation_leader" not in candidates["LOCAL"]["candidate_roles"]
    assert "same_slot_local_damage_leader" not in candidates["LOCAL"]["candidate_roles"]


def test_build_judge_payload_downgrades_early_building_high_risk_to_medium() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 35,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.50,
                "network_util_max": 0.65,
                "free_slots_ratio": 0.50,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "moderate"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=59,
                required_slots=1,
                route_pressure_score=0.15,
                local_damage_score=0.10,
                local_fragmentation=0.60,
                plausibility_score=0.90,
                path_common_free_ratio=0.70,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
                future_risk_band="high",
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=60,
                required_slots=1,
                route_pressure_score=0.19,
                local_damage_score=0.04,
                local_fragmentation=0.20,
                plausibility_score=0.89,
                path_common_free_ratio=0.61,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
                future_risk_band="high",
            ),
        ),
        candidate_ids=("ROUTE", "SAFE"),
    )
    context = payload.prompt_context.to_mapping()

    assert context["congestion_band"] == "building"
    assert context["progress_ratio"] == 0.35
    assert context["future_feasibility_risk_band"] == "medium"


def test_build_judge_payload_keeps_building_submid_high_risk_at_medium() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 49,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.50,
                "network_util_max": 0.65,
                "free_slots_ratio": 0.50,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "moderate"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=61,
                required_slots=1,
                route_pressure_score=0.15,
                local_damage_score=0.10,
                local_fragmentation=0.60,
                plausibility_score=0.90,
                path_common_free_ratio=0.70,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
                future_risk_band="high",
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=62,
                required_slots=1,
                route_pressure_score=0.19,
                local_damage_score=0.04,
                local_fragmentation=0.20,
                plausibility_score=0.89,
                path_common_free_ratio=0.61,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
                future_risk_band="high",
            ),
        ),
        candidate_ids=("ROUTE", "SAFE"),
    )
    context = payload.prompt_context.to_mapping()

    assert context["progress_ratio"] == 0.49
    assert context["future_feasibility_risk_band"] == "medium"


def test_build_judge_payload_promotes_building_high_risk_after_new_midpoint() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 55,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.50,
                "network_util_max": 0.65,
                "free_slots_ratio": 0.50,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "moderate"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=63,
                required_slots=1,
                route_pressure_score=0.15,
                local_damage_score=0.10,
                local_fragmentation=0.60,
                plausibility_score=0.90,
                path_common_free_ratio=0.70,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
                future_risk_band="high",
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=64,
                required_slots=1,
                route_pressure_score=0.19,
                local_damage_score=0.04,
                local_fragmentation=0.20,
                plausibility_score=0.89,
                path_common_free_ratio=0.61,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
                future_risk_band="high",
            ),
        ),
        candidate_ids=("ROUTE", "SAFE"),
    )
    context = payload.prompt_context.to_mapping()

    assert context["progress_ratio"] == 0.55
    assert context["future_feasibility_risk_band"] == "high"


def test_build_judge_payload_exposes_same_path_slot_variant_band_and_roles() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 70,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.50,
                "network_util_max": 0.65,
                "free_slots_ratio": 0.50,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "high"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="ksp_best_mod_last_fit",
                raw_action=54,
                initial_slot=20,
                required_slots=1,
                route_pressure_score=0.22,
                local_damage_score=0.02,
                local_fragmentation=0.02,
                path_common_free_ratio=0.58,
                largest_block_loss_slots=0,
                left_free_span_norm=0.12,
                right_free_span_norm=0.10,
                plausibility_score=0.90,
                path_index=0,
            ),
            _manual_candidate(
                heuristic_name="first_fit",
                raw_action=55,
                initial_slot=2,
                required_slots=1,
                route_pressure_score=0.22,
                local_damage_score=0.08,
                local_fragmentation=0.08,
                path_common_free_ratio=0.58,
                largest_block_loss_slots=4,
                left_free_span_norm=0.00,
                right_free_span_norm=0.22,
                plausibility_score=0.88,
                path_index=0,
            ),
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=56,
                required_slots=1,
                route_pressure_score=0.16,
                local_damage_score=0.09,
                path_common_free_ratio=0.63,
                plausibility_score=0.89,
                path_index=1,
            ),
        ),
        candidate_ids=("SAFE", "EDGE", "ROUTE"),
        candidate_roles=(
            "same_slot_preservation_anchor",
            "same_path_slot_variant_challenger",
            "same_slot_route_challenger",
        ),
    )
    prompt_mapping = payload.to_prompt_mapping()
    candidates = {candidate["candidate_id"]: candidate for candidate in prompt_mapping["candidates"]}
    pairwise_same_path = next(
        pair for pair in prompt_mapping["pairwise_deltas"] if pair["candidate_id"] == "SAFE" and pair["vs_candidate_id"] == "EDGE"
    )

    assert prompt_mapping["prompt_context"]["same_path_slot_variant_band"] == "material"
    assert "same_path_slot_variant_candidate" in candidates["SAFE"]["candidate_roles"]
    assert "same_path_slot_variant_candidate" in candidates["EDGE"]["candidate_roles"]
    assert "same_path_slot_variant_preservation_leader" in candidates["SAFE"]["candidate_roles"]
    assert "same_path_slot_variant_preservation_leader" not in candidates["EDGE"]["candidate_roles"]
    assert "left_free_span_norm" in candidates["SAFE"]["metrics"]
    assert "right_free_span_norm" in candidates["SAFE"]["metrics"]
    assert candidates["SAFE"]["metrics"]["slot_span_total_norm"] == 0.22
    assert pairwise_same_path["same_path_same_modulation"] is True
    assert pairwise_same_path["delta_largest_block_loss_slots"] == -4
    assert pairwise_same_path["delta_local_fragmentation"] == -0.06
    assert pairwise_same_path["delta_slot_span_total_norm"] == 0.0


def test_build_judge_payload_breaks_exact_same_path_slot_variant_ties_by_lower_initial_slot() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 52,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.48,
                "network_util_max": 0.63,
                "free_slots_ratio": 0.49,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "high"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="ksp_best_mod_last_fit",
                raw_action=60,
                initial_slot=232,
                required_slots=1,
                route_pressure_score=0.21,
                local_damage_score=0.04,
                local_fragmentation=0.40,
                path_common_free_ratio=0.22,
                largest_block_loss_slots=0,
                common_block_length_norm=0.02,
                left_free_span_norm=0.00,
                right_free_span_norm=0.01,
                qot_margin_clipped_db=1.0,
                plausibility_score=0.90,
                path_index=0,
                modulation_name="32QAM",
            ),
            _manual_candidate(
                heuristic_name="first_fit",
                raw_action=61,
                initial_slot=265,
                required_slots=1,
                route_pressure_score=0.21,
                local_damage_score=0.04,
                local_fragmentation=0.40,
                path_common_free_ratio=0.22,
                largest_block_loss_slots=0,
                common_block_length_norm=0.02,
                left_free_span_norm=0.00,
                right_free_span_norm=0.01,
                qot_margin_clipped_db=1.0,
                plausibility_score=0.89,
                path_index=0,
                modulation_name="32QAM",
            ),
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=62,
                initial_slot=12,
                required_slots=1,
                route_pressure_score=0.17,
                local_damage_score=0.06,
                local_fragmentation=0.40,
                path_common_free_ratio=0.18,
                largest_block_loss_slots=0,
                common_block_length_norm=0.02,
                left_free_span_norm=0.00,
                right_free_span_norm=0.01,
                qot_margin_clipped_db=1.0,
                plausibility_score=0.88,
                path_index=1,
                modulation_name="32QAM",
            ),
        ),
        candidate_ids=("LOW", "HIGH", "ROUTE"),
        candidate_roles=(
            "same_slot_preservation_anchor",
            "same_slot_backfill_challenger",
            "same_slot_route_challenger",
        ),
    )
    prompt_mapping = payload.to_prompt_mapping()

    assert "same_path_slot_variant_preservation_leader" in prompt_mapping["candidates"][0]["candidate_roles"]
    assert "same_path_slot_variant_preservation_leader" not in prompt_mapping["candidates"][1]["candidate_roles"]


def test_build_judge_payload_elevates_future_risk_when_same_slot_candidates_are_high_risk() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 55,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.20,
                "network_util_max": 0.35,
                "free_slots_ratio": 0.78,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "light"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=60,
                required_slots=1,
                route_pressure_score=0.41,
                local_damage_score=0.10,
                plausibility_score=0.90,
                path_common_free_ratio=0.22,
                qot_margin_clipped_db=1.90,
                future_risk_band="high",
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=61,
                required_slots=1,
                route_pressure_score=0.44,
                local_damage_score=0.04,
                plausibility_score=0.88,
                path_common_free_ratio=0.18,
                qot_margin_clipped_db=3.00,
                future_risk_band="high",
            ),
        ),
        candidate_ids=("ROUTE", "LOCAL"),
    )
    context = payload.prompt_context.to_mapping()

    assert context["congestion_band"] == "open"
    assert context["future_feasibility_risk_band"] == "medium"


def test_build_judge_payload_treats_qot_gap_as_tie_breaker_for_same_slot_band() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 50,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.35,
                "network_util_max": 0.45,
                "free_slots_ratio": 0.60,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "moderate"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=70,
                required_slots=1,
                route_pressure_score=0.40,
                local_damage_score=0.09,
                plausibility_score=0.91,
                path_common_free_ratio=0.30,
                qot_margin_clipped_db=1.10,
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=71,
                required_slots=1,
                route_pressure_score=0.415,
                local_damage_score=0.03,
                plausibility_score=0.90,
                path_common_free_ratio=0.27,
                qot_margin_clipped_db=3.00,
            ),
        ),
        candidate_ids=("ROUTE", "LOCAL"),
    )
    context = payload.prompt_context.to_mapping()

    assert context["same_slot_near_tie_band"] == "near_tie"


def test_build_judge_payload_exposes_candidate_roles_without_heuristic_names() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 80,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.40,
                "network_util_max": 0.60,
                "free_slots_ratio": 0.52,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "high"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=90,
                required_slots=1,
                route_pressure_score=0.15,
                local_damage_score=0.10,
                plausibility_score=0.90,
                path_common_free_ratio=0.70,
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=91,
                required_slots=1,
                route_pressure_score=0.19,
                local_damage_score=0.01,
                plausibility_score=0.89,
                path_common_free_ratio=0.61,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
            ),
        ),
        candidate_ids=("ROUTE", "SAFE"),
        candidate_roles=("same_slot_route_challenger", "same_slot_preservation_anchor"),
    )
    candidate_payload = payload.to_prompt_mapping()["candidates"]

    assert candidate_payload[0]["candidate_id"] == "ROUTE"
    assert "same_slot_route_challenger" not in candidate_payload[0]["candidate_roles"]
    assert "same_slot_route_leader" in candidate_payload[0]["candidate_roles"]
    assert "same_slot_common_free_leader" in candidate_payload[0]["candidate_roles"]
    assert "same_slot_preservation_leader" not in candidate_payload[0]["candidate_roles"]
    assert "same_slot_candidate" in candidate_payload[0]["candidate_roles"]
    assert "heuristic_name" not in candidate_payload[0]
    assert "same_slot_preservation_anchor" not in candidate_payload[1]["candidate_roles"]
    assert "same_slot_preservation_leader" not in candidate_payload[1]["candidate_roles"]
    assert "same_slot_local_damage_leader" not in candidate_payload[1]["candidate_roles"]


def test_build_judge_payload_marks_tied_best_route_and_common_free_roles_on_all_tied_candidates() -> None:
    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 100})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 82,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.42,
                "network_util_max": 0.58,
                "free_slots_ratio": 0.50,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "high"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="first_fit",
                raw_action=92,
                required_slots=1,
                route_pressure_score=0.20,
                local_damage_score=0.04,
                plausibility_score=0.90,
                path_common_free_ratio=0.70,
                qot_margin_clipped_db=1.2,
                proposed_by=("first_fit", "load_balancing"),
            ),
            _manual_candidate(
                heuristic_name="ksp_best_mod_last_fit",
                raw_action=93,
                required_slots=1,
                route_pressure_score=0.20,
                local_damage_score=0.04,
                plausibility_score=0.89,
                path_common_free_ratio=0.70,
                qot_margin_clipped_db=1.1,
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=94,
                required_slots=1,
                route_pressure_score=0.24,
                local_damage_score=0.02,
                plausibility_score=0.88,
                path_common_free_ratio=0.64,
                qot_margin_clipped_db=2.0,
            ),
        ),
        candidate_ids=("A1", "B2", "C3"),
        candidate_roles=(
            "same_slot_preservation_anchor",
            "same_slot_backfill_challenger",
            "same_slot_backfill_challenger",
        ),
    )
    prompt_mapping = payload.to_prompt_mapping()
    candidates = {candidate["candidate_id"]: candidate for candidate in prompt_mapping["candidates"]}

    assert prompt_mapping["prompt_context"]["same_slot_route_common_free_alignment"] == "aligned"
    assert "same_slot_route_leader" in candidates["A1"]["candidate_roles"]
    assert "same_slot_common_free_leader" in candidates["A1"]["candidate_roles"]
    assert "same_slot_route_leader" in candidates["B2"]["candidate_roles"]
    assert "same_slot_common_free_leader" in candidates["B2"]["candidate_roles"]
    assert "same_slot_route_leader" not in candidates["C3"]["candidate_roles"]
    assert "same_slot_common_free_leader" not in candidates["C3"]["candidate_roles"]
    assert "same_slot_local_damage_leader" not in candidates["C3"]["candidate_roles"]
    assert "same_slot_local_damage_leader" not in candidates["A1"]["candidate_roles"]
    assert "same_slot_qot_leader" not in candidates["C3"]["candidate_roles"]
    assert "same_slot_qot_leader" not in candidates["A1"]["candidate_roles"]


def test_prompt_candidate_target_size_prefers_three_or_more_options() -> None:
    module = _load_module()
    select_prompt_candidates = module["_select_prompt_candidates"]

    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=20,
            required_slots=1,
            route_pressure_score=0.20,
            local_damage_score=0.05,
            plausibility_score=0.92,
            path_common_free_ratio=0.60,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=21,
            required_slots=1,
            route_pressure_score=0.18,
            local_damage_score=0.06,
            plausibility_score=0.89,
            path_common_free_ratio=0.61,
        ),
        _manual_candidate(
            heuristic_name="ksp_best_mod_last_fit",
            raw_action=22,
            required_slots=1,
            route_pressure_score=0.22,
            local_damage_score=0.04,
            plausibility_score=0.87,
            path_common_free_ratio=0.58,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=23,
            required_slots=2,
            route_pressure_score=0.17,
            local_damage_score=0.01,
            plausibility_score=0.84,
            path_common_free_ratio=0.63,
        ),
    )

    selected = select_prompt_candidates(
        candidates,
        min_prompt_candidates=3,
        max_prompt_candidates=4,
        third_candidate_gap=0.03,
        same_slot_route_advantage_material=0.015,
        same_slot_local_advantage_material=0.08,
        same_slot_common_free_penalty_material=0.05,
        extra_slot_route_advantage_material=0.03,
        extra_slot_local_advantage_material=0.10,
        extra_slot_common_free_advantage_material=0.08,
    )

    assert 3 <= len(selected) <= 4


def test_select_prompt_candidates_caps_same_slot_options_at_three_without_structural_extra_slot() -> None:
    module = _load_module()
    select_prompt_candidates = module["_select_prompt_candidates"]

    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=80,
            required_slots=1,
            route_pressure_score=0.24,
            local_damage_score=0.11,
            plausibility_score=0.82,
            path_common_free_ratio=0.56,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=81,
            required_slots=1,
            route_pressure_score=0.16,
            local_damage_score=0.10,
            plausibility_score=0.90,
            path_common_free_ratio=0.66,
        ),
        _manual_candidate(
            heuristic_name="ksp_best_mod_last_fit",
            raw_action=82,
            required_slots=1,
            route_pressure_score=0.20,
            local_damage_score=0.03,
            plausibility_score=0.84,
            path_common_free_ratio=0.60,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=83,
            required_slots=1,
            route_pressure_score=0.21,
            local_damage_score=0.02,
            plausibility_score=0.87,
            path_common_free_ratio=0.59,
        ),
    )

    selected = select_prompt_candidates(
        candidates,
        min_prompt_candidates=3,
        max_prompt_candidates=4,
        third_candidate_gap=0.03,
        same_slot_route_advantage_material=0.015,
        same_slot_local_advantage_material=0.08,
        same_slot_common_free_penalty_material=0.05,
        extra_slot_route_advantage_material=0.03,
        extra_slot_local_advantage_material=0.10,
        extra_slot_common_free_advantage_material=0.08,
    )

    assert len(selected) == 3


def test_select_prompt_candidates_prefers_structural_preservation_anchor_over_raw_local_damage() -> None:
    module = _load_module()
    select_prompt_candidates = module["_select_prompt_candidates"]

    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=84,
            required_slots=1,
            route_pressure_score=0.18,
            local_damage_score=0.01,
            plausibility_score=0.89,
            path_common_free_ratio=0.42,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=9,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=85,
            required_slots=1,
            route_pressure_score=0.20,
            local_damage_score=0.03,
            plausibility_score=0.90,
            path_common_free_ratio=0.72,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=86,
            required_slots=1,
            route_pressure_score=0.16,
            local_damage_score=0.08,
            plausibility_score=0.91,
            path_common_free_ratio=0.66,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
        ),
    )

    selected = select_prompt_candidates(
        candidates,
        min_prompt_candidates=3,
        max_prompt_candidates=4,
        third_candidate_gap=0.03,
        same_slot_route_advantage_material=0.015,
        same_slot_local_advantage_material=0.08,
        same_slot_common_free_penalty_material=0.05,
        extra_slot_route_advantage_material=0.03,
        extra_slot_local_advantage_material=0.10,
        extra_slot_common_free_advantage_material=0.08,
    )

    assert tuple(candidate.heuristic_name for candidate in selected) == (
        "load_balancing",
        "lowest_fragmentation",
        "first_fit",
    )


def test_shuffle_prompt_entries_is_reproducible_and_tracks_permutation() -> None:
    module = _load_module()
    shuffle_prompt_entries = module["_shuffle_prompt_entries"]
    rng = module["np"].random.default_rng(7)

    entries = (
        ("balanced_anchor", _manual_candidate(
            heuristic_name="first_fit",
            raw_action=10,
            required_slots=1,
            route_pressure_score=0.20,
            local_damage_score=0.05,
            plausibility_score=0.92,
            path_common_free_ratio=0.60,
        )),
        ("same_slot_route_challenger", _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=11,
            required_slots=1,
            route_pressure_score=0.18,
            local_damage_score=0.06,
            plausibility_score=0.89,
            path_common_free_ratio=0.61,
        )),
        ("same_slot_local_challenger", _manual_candidate(
            heuristic_name="ksp_best_mod_last_fit",
            raw_action=12,
            required_slots=1,
            route_pressure_score=0.22,
            local_damage_score=0.04,
            plausibility_score=0.87,
            path_common_free_ratio=0.58,
        )),
    )

    shuffled_entries, permutation = shuffle_prompt_entries(entries, rng=rng)

    assert permutation == (0, 2, 1)
    assert tuple(candidate.raw_action for _role, candidate in shuffled_entries) == (10, 12, 11)


def test_load_best_heuristic_blocking_threshold_reads_requested_load(tmp_path: Path) -> None:
    module = _load_module()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "loads": [
                    {"load": 315.0, "best_service_blocking_rate_mean": 0.012},
                    {"load": 320.0, "best_service_blocking_rate_mean": 0.021},
                ]
            }
        ),
        encoding="utf-8",
    )

    threshold = module["_load_best_heuristic_blocking_threshold"](
        baseline_path=baseline_path,
        load=315.0,
    )

    assert threshold == 0.012


def test_online_llm_judge_can_stop_early_when_blocking_exceeds_baseline(
    tmp_path: Path,
) -> None:
    module = _load_module()
    fake_judge = FakeJudge()
    counter = {"step_calls": 0}
    original_build_env = module["build_env"]
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"loads": [{"load": 10.0, "best_service_blocking_rate_mean": 0.0}]}),
        encoding="utf-8",
    )

    def baseline_trip_build_env(*, scenario):
        return BaselineTripEnv(
            original_build_env(scenario=scenario),
            counter,
            trip_after_steps=3,
            blocking_rate=0.50,
        )

    module["run_experiment"].__globals__["build_env"] = baseline_trip_build_env
    try:
        experiment = module["LLMJudgeExperiment"](
            topology_id="ring_4",
            episode_count=1,
            episode_length=10,
            scenario_profile="ofc_v1",
            seed=7,
            load=10.0,
            mean_holding_time=8.0,
            num_spectrum_resources=24,
            k_paths=2,
            modulations_to_consider=2,
            output_dir=tmp_path,
            baseline_path=baseline_path,
            stop_when_blocking_exceeds_baseline=True,
            min_steps_before_stop=2,
        )

        outputs = module["run_experiment"](
            experiment=experiment,
            judge=fake_judge,
            now=datetime(2026, 3, 19),
        )
    finally:
        module["run_experiment"].__globals__["build_env"] = original_build_env

    with outputs.summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    episode_row = next(row for row in summary_rows if row["scope"] == "episode")
    run_row = next(row for row in summary_rows if row["scope"] == "run")

    assert int(episode_row["steps"]) == 3
    assert episode_row["stop_reason"] == "baseline_blocking_exceeded"
    assert episode_row["stopped_early"] == "True"
    assert episode_row["baseline_blocking_threshold"] == "0.0"
    assert run_row["stopped_early_episodes"] == "1"
    assert run_row["stop_reason"] == "baseline_blocking_exceeded"
    assert counter["step_calls"] == 3


def test_online_llm_judge_smoke_generates_outputs_and_uses_single_step_per_decision(
    tmp_path: Path,
) -> None:
    module = _load_module()
    fake_judge = FakeJudge()
    counter = {"step_calls": 0}
    original_build_env = module["build_env"]

    def counting_build_env(*, scenario):
        return CountingEnv(original_build_env(scenario=scenario), counter)

    module["run_experiment"].__globals__["build_env"] = counting_build_env

    experiment = module["LLMJudgeExperiment"](
        topology_id="ring_4",
        episode_count=1,
        episode_length=6,
        scenario_profile="ofc_v1",
        seed=7,
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        k_paths=2,
        modulations_to_consider=2,
        output_dir=tmp_path,
    )

    outputs = module["run_experiment"](
        experiment=experiment,
        judge=fake_judge,
        now=datetime(2026, 3, 19),
    )

    assert outputs.steps_csv == tmp_path / "19-03-00h00-llm-judge-steps.csv"
    assert outputs.summary_csv == tmp_path / "19-03-00h00-llm-judge-summary.csv"
    assert outputs.calls_jsonl == tmp_path / "19-03-00h00-llm-judge-calls.jsonl"
    assert outputs.steps_csv.exists()
    assert outputs.summary_csv.exists()
    assert outputs.calls_jsonl.exists()

    with outputs.steps_csv.open("r", encoding="utf-8", newline="") as handle:
        step_rows = list(csv.DictReader(handle))
    with outputs.summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    with outputs.calls_jsonl.open("r", encoding="utf-8") as handle:
        call_records = [json.loads(line) for line in handle if line.strip()]

    assert step_rows
    assert call_records
    assert counter["step_calls"] == len(step_rows)
    assert fake_judge.calls <= len(step_rows)
    assert "winner_decoded_path_nodes" in step_rows[0]
    assert "winner_modulation_name" in step_rows[0]
    assert "winner_confidence" in step_rows[0]
    assert "winner_decision_basis" in step_rows[0]
    assert "decisive_signals_summary" in step_rows[0]
    assert "controller_decision_source" in step_rows[0]
    assert "raw_candidate_count" in step_rows[0]
    assert "surviving_candidate_count" in step_rows[0]
    assert "prompt_candidate_count" in step_rows[0]
    assert "fallback_reason" in step_rows[0]
    assert "judge_error_message" in step_rows[0]
    assert "semantic_warning_flags" in step_rows[0]
    assert "basis_vs_payload_mismatch" in step_rows[0]
    assert "pre_shuffle_shortlist_actions" in step_rows[0]
    assert "post_shuffle_shortlist_actions" in step_rows[0]
    assert "prompt_permutation" in step_rows[0]
    assert "hidden_balanced_candidate_id" in step_rows[0]
    assert "hidden_balanced_candidate_action" in step_rows[0]
    assert "hidden_balanced_candidate_heuristic" in step_rows[0]
    assert "hidden_balanced_position" in step_rows[0]
    assert "winner_prompt_position" in step_rows[0]
    assert "winner_matches_hidden_balanced" in step_rows[0]
    assert not any(
        row["controller_decision_source"] == "late_episode_conservative_override"
        for row in step_rows
    )
    assert any(row["scope"] == "run" for row in summary_rows)
    run_row = next(row for row in summary_rows if row["scope"] == "run")
    assert "mean_episode_service_blocking_rate" in run_row
    assert "llm_only_agreement_rate" in run_row
    assert "hidden_balanced_agreement_rate" in run_row
    assert "llm_non_trivial_usage_rate" in run_row
    assert "shortlist_collapse_rate" in run_row
    assert "hidden_top1_match_rate_multi_option" in run_row
    assert "decision_change_rate_vs_hidden_top1" in run_row
    assert "decorative_failed_check_count_observable" in run_row
    assert "decorative_judge_alert_observable" in run_row
    assert "decorative_judge_regression_observable" in run_row

    first_record = call_records[0]
    llm_record = next(
        (record for record in call_records if record["audit"]["controller_decision_source"] == "llm"),
        first_record,
    )
    assert "audit" in first_record
    assert "decision_payload" in first_record
    assert "prompt" in first_record
    assert "model_io" in first_record
    assert "baseline_winner" in first_record["audit"]
    assert "prompt_version" in first_record["audit"]
    assert "raw_action" in first_record["audit"]["candidate_audit"][0]
    assert "system_prompt" in first_record["prompt"]
    assert "user_prompt" in first_record["prompt"]
    assert "raw_model_response" in first_record["model_io"]
    assert "parsed_response" in first_record["model_io"]
    assert "reference_winner" in first_record["audit"]
    assert "controller_decision_source" in first_record["audit"]
    assert "raw_candidate_count" in first_record["audit"]
    assert "surviving_candidate_count" in first_record["audit"]
    assert "prompt_candidate_count" in first_record["audit"]
    assert "winner_proposed_by" in first_record["audit"]
    assert "pre_shuffle_shortlist_actions" in first_record["audit"]
    assert "post_shuffle_shortlist_actions" in first_record["audit"]
    assert "prompt_permutation" in first_record["audit"]
    assert "hidden_balanced_candidate_id" in first_record["audit"]
    assert "hidden_balanced_candidate_action" in first_record["audit"]
    assert "hidden_balanced_candidate_heuristic" in first_record["audit"]
    assert "semantic_warning_flags" in first_record["model_io"]
    assert "basis_vs_payload_mismatch" in first_record["model_io"]

    candidate_payload = llm_record["decision_payload"]["candidates"][0]
    assert candidate_payload["candidate_id"]
    assert candidate_payload["candidate_id"] not in {"A", "B", "C"}
    assert "heuristic_name" not in candidate_payload
    assert "metrics" in candidate_payload
    assert "route" in candidate_payload
    assert "pairwise_deltas" in llm_record["decision_payload"]
    assert "prompt_context" in llm_record["decision_payload"]
    assert len(llm_record["decision_payload"]["candidates"]) <= 4
    assert "future_risk_band" in candidate_payload["metrics"]
    assert "support_count" in candidate_payload["metrics"]
    assert "has_multi_heuristic_support" in candidate_payload["metrics"]
    assert "left_free_span_norm" in candidate_payload["metrics"]
    assert "right_free_span_norm" in candidate_payload["metrics"]
    assert "slot_span_total_norm" in candidate_payload["metrics"]
    assert "candidate_roles" in candidate_payload
    assert candidate_payload["candidate_roles"]
    assert "plausibility_score" not in candidate_payload["metrics"]
    assert "plausibility_rank" not in candidate_payload["metrics"]
    assert "candidate_roles" not in llm_record["decision_payload"]["prompt_context"]
    assert "balanced_candidate_id" not in llm_record["decision_payload"]["prompt_context"]
    assert "progress_ratio" in llm_record["decision_payload"]["prompt_context"]
    assert "congestion_band" in llm_record["decision_payload"]["prompt_context"]
    assert "same_slot_near_tie_band" in llm_record["decision_payload"]["prompt_context"]
    assert "same_slot_damage_axes_tie_band" in llm_record["decision_payload"]["prompt_context"]
    assert "same_slot_route_common_free_alignment" in llm_record["decision_payload"]["prompt_context"]
    assert "same_path_slot_variant_band" in llm_record["decision_payload"]["prompt_context"]
    assert "route_gain_band_vs_same_slot_best" in llm_record["decision_payload"]["prompt_context"]
    assert "local_gain_band_vs_same_slot_best" in llm_record["decision_payload"]["prompt_context"]
    assert "same_slot_local_support_band" in llm_record["decision_payload"]["prompt_context"]
    assert "common_free_penalty_band_vs_same_slot_best" in llm_record["decision_payload"]["prompt_context"]
    assert "future_feasibility_risk_band" in llm_record["decision_payload"]["prompt_context"]
    assert "same_path_same_modulation" in llm_record["decision_payload"]["pairwise_deltas"][0]
    assert "delta_largest_block_loss_slots" in llm_record["decision_payload"]["pairwise_deltas"][0]
    assert "delta_slot_span_total_norm" in llm_record["decision_payload"]["pairwise_deltas"][0]
    if llm_record["audit"]["controller_decision_source"] == "llm":
        assert llm_record["model_io"]["parsed_response"] is not None
        assert "decision_basis" in llm_record["model_io"]["parsed_response"]
        assert 2 <= len(llm_record["decision_payload"]["candidates"]) <= 4


def test_online_llm_judge_defaults_use_shared_legacy_benchmark_profile() -> None:
    module = _load_module()
    experiment = module["LLMJudgeExperiment"]()
    scenario = module["build_base_scenario"](experiment)

    assert experiment.scenario_profile == "legacy_benchmark"
    assert scenario.load == module["scenario_utils"].LEGACY_BENCHMARK_LOAD
    assert scenario.k_paths == module["scenario_utils"].LEGACY_BENCHMARK_K_PATHS
    assert scenario.launch_power_dbm == module["scenario_utils"].LEGACY_BENCHMARK_LAUNCH_POWER_DBM
    assert scenario.modulations_to_consider == module["scenario_utils"].LEGACY_BENCHMARK_MODULATIONS_TO_CONSIDER
    assert not hasattr(experiment, "enable_late_episode_conservative_override")
    assert tuple(modulation.name for modulation in scenario.modulations) == tuple(
        modulation.name
        for modulation in module["scenario_utils"].build_legacy_benchmark_scenario().modulations
    )


def test_build_decorative_judge_audit_flags_observable_regression_when_llm_is_decorative() -> None:
    module = _load_module()
    build_audit = module["_build_decorative_judge_audit"]

    step_rows = [
        {
            "raw_candidate_count": 4,
            "surviving_candidate_count": 3,
            "prompt_candidate_count": 1,
            "judge_called": False,
            "fallback_reason": "none",
            "winner_matches_hidden_balanced": True,
            "winner_heuristic": "lowest_fragmentation",
            "reference_winner": "load_balancing",
            "hidden_balanced_candidate_heuristic": "lowest_fragmentation",
        },
        {
            "raw_candidate_count": 4,
            "surviving_candidate_count": 2,
            "prompt_candidate_count": 1,
            "judge_called": False,
            "fallback_reason": "none",
            "winner_matches_hidden_balanced": True,
            "winner_heuristic": "lowest_fragmentation",
            "reference_winner": "load_balancing",
            "hidden_balanced_candidate_heuristic": "lowest_fragmentation",
        },
        {
            "raw_candidate_count": 3,
            "surviving_candidate_count": 2,
            "prompt_candidate_count": 1,
            "judge_called": False,
            "fallback_reason": "none",
            "winner_matches_hidden_balanced": True,
            "winner_heuristic": "lowest_fragmentation",
            "reference_winner": "load_balancing",
            "hidden_balanced_candidate_heuristic": "lowest_fragmentation",
        },
        {
            "raw_candidate_count": 4,
            "surviving_candidate_count": 3,
            "prompt_candidate_count": 3,
            "judge_called": True,
            "fallback_reason": "unsafe_winner_candidate_id:BAD1",
            "winner_matches_hidden_balanced": True,
            "winner_heuristic": "lowest_fragmentation",
            "reference_winner": "load_balancing",
            "hidden_balanced_candidate_heuristic": "lowest_fragmentation",
        },
    ]

    audit = build_audit(step_rows=step_rows)

    assert audit["decorative_failed_check_count_observable"] >= 4
    assert audit["decorative_judge_alert_observable"] is True
    assert audit["decorative_judge_regression_observable"] is True
    assert "llm_not_called_with_multiple_plausible_options" in audit["decorative_failed_checks_observable"]
    assert "shortlist_or_controller_collapses_multi_option_cases" in audit["decorative_failed_checks_observable"]
    assert "fallback_used_for_normal_decision_cases" in audit["decorative_failed_checks_observable"]
    assert "llm_excessively_mirrors_hidden_top1" in audit["decorative_failed_checks_observable"]


def test_build_decorative_judge_audit_passes_when_llm_changes_and_sometimes_improves() -> None:
    module = _load_module()
    build_audit = module["_build_decorative_judge_audit"]

    step_rows = [
        {
            "raw_candidate_count": 4,
            "surviving_candidate_count": 3,
            "prompt_candidate_count": 3,
            "judge_called": True,
            "fallback_reason": "none",
            "winner_matches_hidden_balanced": False,
            "winner_heuristic": "load_balancing",
            "reference_winner": "load_balancing",
            "hidden_balanced_candidate_heuristic": "lowest_fragmentation",
        },
        {
            "raw_candidate_count": 4,
            "surviving_candidate_count": 3,
            "prompt_candidate_count": 3,
            "judge_called": True,
            "fallback_reason": "none",
            "winner_matches_hidden_balanced": True,
            "winner_heuristic": "lowest_fragmentation",
            "reference_winner": "lowest_fragmentation",
            "hidden_balanced_candidate_heuristic": "lowest_fragmentation",
        },
    ]

    audit = build_audit(step_rows=step_rows)

    assert audit["llm_non_trivial_usage_rate"] == 1.0
    assert audit["shortlist_collapse_rate"] == 0.0
    assert audit["hidden_top1_match_rate_multi_option"] == 0.5
    assert audit["decision_change_rate_vs_hidden_top1"] == 0.5
    assert audit["beneficial_llm_disagreement_count"] == 1
    assert audit["decorative_check_llm_disagreement_sometimes_helps_pass"] is True
    assert audit["decorative_judge_alert_observable"] is False
    assert "strong_candidates_reach_llm" in audit["decorative_manual_review_pending"]
    assert "prompt_changes_move_hard_cases" in audit["decorative_manual_review_pending"]


def test_split_tradeoff_override_prefers_common_free_leader_when_route_gap_is_small() -> None:
    module = _load_module()
    resolve_override = module["_resolve_split_tradeoff_override"]
    build_payload = module["build_judge_payload"]

    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=10,
            required_slots=1,
            route_pressure_score=0.3765,
            local_damage_score=0.1050,
            path_common_free_ratio=0.5312,
            plausibility_score=0.7150,
            path_index=0,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=20,
            required_slots=1,
            route_pressure_score=0.4415,
            local_damage_score=0.0394,
            path_common_free_ratio=0.1406,
            plausibility_score=0.6905,
            path_index=1,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=30,
            required_slots=1,
            route_pressure_score=0.3725,
            local_damage_score=0.1031,
            path_common_free_ratio=0.3469,
            plausibility_score=0.7434,
            path_index=2,
        ),
    )
    payload = build_payload(
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 1000})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 660,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.31,
                "network_util_max": 0.62,
                "free_slots_ratio": 0.68,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "high"})(),
        candidates=candidates,
        candidate_ids=("FF", "LF", "LB"),
        candidate_roles=("same_slot_common_free_challenger", "same_slot_candidate", "same_slot_route_challenger"),
    )
    candidate_by_id = {
        candidate_id: candidate
        for candidate_id, candidate in zip(("FF", "LF", "LB"), candidates, strict=True)
    }

    assert payload.prompt_context.to_mapping()["same_slot_route_common_free_alignment"] == "split"
    assert payload.prompt_context.to_mapping()["same_slot_local_support_band"] == "none"
    assert resolve_override(
        payload=payload,
        candidate_by_id=candidate_by_id,
        winner_candidate_id="LF",
        semantic_warning_flags=("same_slot_local_basis_mismatch",),
    ) == "FF"


def test_split_tradeoff_override_prefers_route_leader_when_common_free_route_gap_is_large() -> None:
    module = _load_module()
    resolve_override = module["_resolve_split_tradeoff_override"]
    build_payload = module["build_judge_payload"]

    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=11,
            required_slots=2,
            route_pressure_score=0.5432,
            local_damage_score=0.0750,
            path_common_free_ratio=0.2531,
            plausibility_score=0.5119,
            path_index=0,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=21,
            required_slots=2,
            route_pressure_score=0.4851,
            local_damage_score=0.0700,
            path_common_free_ratio=0.0875,
            plausibility_score=0.6580,
            path_index=1,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=31,
            required_slots=2,
            route_pressure_score=0.4912,
            local_damage_score=0.0100,
            path_common_free_ratio=0.1594,
            plausibility_score=0.6805,
            path_index=2,
        ),
    )
    payload = build_payload(
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 1000})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 688,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.31,
                "network_util_max": 0.62,
                "free_slots_ratio": 0.68,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "high"})(),
        candidates=candidates,
        candidate_ids=("FF", "LB", "LF"),
        candidate_roles=("same_slot_common_free_challenger", "same_slot_route_challenger", "same_slot_candidate"),
    )
    candidate_by_id = {
        candidate_id: candidate
        for candidate_id, candidate in zip(("FF", "LB", "LF"), candidates, strict=True)
    }

    assert payload.prompt_context.to_mapping()["route_gain_band_vs_same_slot_best"] == "none"
    assert resolve_override(
        payload=payload,
        candidate_by_id=candidate_by_id,
        winner_candidate_id="LF",
        semantic_warning_flags=("same_slot_route_basis_mismatch",),
    ) == "LB"


def test_split_tradeoff_override_skips_material_route_gain_with_partial_local_support() -> None:
    module = _load_module()
    resolve_override = module["_resolve_split_tradeoff_override"]
    build_payload = module["build_judge_payload"]

    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=12,
            required_slots=1,
            route_pressure_score=0.4497,
            local_damage_score=0.1063,
            local_fragmentation=0.60,
            path_common_free_ratio=0.2188,
            plausibility_score=0.5337,
            path_index=0,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=22,
            required_slots=1,
            route_pressure_score=0.4238,
            local_damage_score=0.0394,
            local_fragmentation=0.20,
            path_common_free_ratio=0.1063,
            plausibility_score=0.6903,
            path_index=1,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=32,
            required_slots=1,
            route_pressure_score=0.3930,
            local_damage_score=0.1025,
            local_fragmentation=0.60,
            path_common_free_ratio=0.1781,
            plausibility_score=0.7061,
            path_index=2,
        ),
    )
    payload = build_payload(
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 1000})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 684,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.31,
                "network_util_max": 0.62,
                "free_slots_ratio": 0.68,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "high"})(),
        candidates=candidates,
        candidate_ids=("FF", "LF", "LB"),
        candidate_roles=("same_slot_common_free_challenger", "same_slot_candidate", "same_slot_route_challenger"),
    )
    candidate_by_id = {
        candidate_id: candidate
        for candidate_id, candidate in zip(("FF", "LF", "LB"), candidates, strict=True)
    }

    assert payload.prompt_context.to_mapping()["route_gain_band_vs_same_slot_best"] == "material"
    assert payload.prompt_context.to_mapping()["same_slot_local_support_band"] == "partial"
    assert resolve_override(
        payload=payload,
        candidate_by_id=candidate_by_id,
        winner_candidate_id="LF",
        semantic_warning_flags=("same_slot_local_basis_mismatch",),
    ) is None


def test_cross_path_preservation_override_prefers_lowest_fragmentation_for_material_outlier() -> None:
    module = _load_module()
    resolve_override = module["_resolve_cross_path_preservation_override"]
    build_payload = module["build_judge_payload"]

    candidates = (
        _manual_candidate(
            heuristic_name="ksp_best_mod_last_fit",
            raw_action=101,
            initial_slot=319,
            required_slots=1,
            route_pressure_score=0.3674,
            local_damage_score=0.1325,
            local_fragmentation=0.8,
            path_common_free_ratio=0.6594,
            largest_block_loss_slots=0,
            left_free_span_norm=0.9969,
            right_free_span_norm=0.0,
            plausibility_score=0.88,
            path_index=0,
            modulation_name="64QAM",
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=102,
            initial_slot=183,
            required_slots=1,
            route_pressure_score=0.4774,
            local_damage_score=0.0369,
            local_fragmentation=0.2,
            path_common_free_ratio=0.1125,
            largest_block_loss_slots=0,
            left_free_span_norm=0.5719,
            right_free_span_norm=0.0,
            plausibility_score=0.87,
            path_index=3,
            modulation_name="32QAM",
        ),
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=103,
            initial_slot=10,
            required_slots=1,
            route_pressure_score=0.3674,
            local_damage_score=0.0725,
            local_fragmentation=0.4,
            path_common_free_ratio=0.6594,
            largest_block_loss_slots=0,
            left_free_span_norm=0.0,
            right_free_span_norm=0.9688,
            plausibility_score=0.89,
            path_index=0,
            modulation_name="64QAM",
        ),
    )
    payload = build_payload(
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 1000})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 652,
                "episode_service_blocking_rate": 0.0100,
                "network_util_mean": 0.34,
                "network_util_max": 0.63,
                "free_slots_ratio": 0.66,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "high"})(),
        candidates=candidates,
        candidate_ids=("KSP", "LF", "FF"),
        candidate_roles=(
            "same_slot_backfill_challenger",
            "same_slot_preservation_anchor",
            "same_slot_backfill_challenger",
        ),
    )
    candidate_by_id = {
        candidate_id: candidate
        for candidate_id, candidate in zip(("KSP", "LF", "FF"), candidates, strict=True)
    }

    assert payload.prompt_context is not None
    assert payload.prompt_context.route_gain_band_vs_same_slot_best == "strong"
    assert payload.prompt_context.same_path_slot_variant_band == "material"
    assert resolve_override(
        payload=payload,
        candidate_by_id=candidate_by_id,
        winner_candidate_id="FF",
    ) == "LF"


def test_cross_path_preservation_override_prefers_lowest_fragmentation_for_partial_outlier() -> None:
    module = _load_module()
    resolve_override = module["_resolve_cross_path_preservation_override"]
    build_payload = module["build_judge_payload"]

    candidates = (
        _manual_candidate(
            heuristic_name="ksp_best_mod_last_fit",
            raw_action=111,
            initial_slot=319,
            required_slots=1,
            route_pressure_score=0.3612,
            local_damage_score=0.1019,
            local_fragmentation=0.6,
            path_common_free_ratio=0.6719,
            largest_block_loss_slots=0,
            left_free_span_norm=0.9969,
            right_free_span_norm=0.0,
            plausibility_score=0.88,
            path_index=0,
            modulation_name="64QAM",
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=112,
            initial_slot=57,
            required_slots=1,
            route_pressure_score=0.4298,
            local_damage_score=0.0381,
            local_fragmentation=0.2,
            path_common_free_ratio=0.0906,
            largest_block_loss_slots=0,
            left_free_span_norm=0.1781,
            right_free_span_norm=0.0,
            plausibility_score=0.87,
            path_index=4,
            modulation_name="32QAM",
        ),
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=113,
            initial_slot=0,
            required_slots=1,
            route_pressure_score=0.3612,
            local_damage_score=0.0719,
            local_fragmentation=0.4,
            path_common_free_ratio=0.6719,
            largest_block_loss_slots=0,
            left_free_span_norm=0.0,
            right_free_span_norm=0.9969,
            plausibility_score=0.89,
            path_index=0,
            modulation_name="64QAM",
        ),
    )
    payload = build_payload(
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 1000})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 663,
                "episode_service_blocking_rate": 0.0100,
                "network_util_mean": 0.34,
                "network_util_max": 0.63,
                "free_slots_ratio": 0.66,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "high"})(),
        candidates=candidates,
        candidate_ids=("KSP", "LF", "FF"),
        candidate_roles=(
            "same_slot_backfill_challenger",
            "same_slot_preservation_anchor",
            "same_slot_backfill_challenger",
        ),
    )
    candidate_by_id = {
        candidate_id: candidate
        for candidate_id, candidate in zip(("KSP", "LF", "FF"), candidates, strict=True)
    }

    assert payload.prompt_context is not None
    assert payload.prompt_context.same_slot_local_support_band == "partial"
    assert resolve_override(
        payload=payload,
        candidate_by_id=candidate_by_id,
        winner_candidate_id="FF",
    ) == "LF"


def test_cross_path_preservation_override_skips_outside_target_window() -> None:
    module = _load_module()
    resolve_override = module["_resolve_cross_path_preservation_override"]
    build_payload = module["build_judge_payload"]

    candidates = (
        _manual_candidate(
            heuristic_name="ksp_best_mod_last_fit",
            raw_action=121,
            initial_slot=319,
            required_slots=1,
            route_pressure_score=0.3674,
            local_damage_score=0.1325,
            local_fragmentation=0.8,
            path_common_free_ratio=0.6594,
            largest_block_loss_slots=0,
            left_free_span_norm=0.9969,
            right_free_span_norm=0.0,
            plausibility_score=0.88,
            path_index=0,
            modulation_name="64QAM",
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=122,
            initial_slot=183,
            required_slots=1,
            route_pressure_score=0.4774,
            local_damage_score=0.0369,
            local_fragmentation=0.2,
            path_common_free_ratio=0.1125,
            largest_block_loss_slots=0,
            left_free_span_norm=0.5719,
            right_free_span_norm=0.0,
            plausibility_score=0.87,
            path_index=3,
            modulation_name="32QAM",
        ),
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=123,
            initial_slot=10,
            required_slots=1,
            route_pressure_score=0.3674,
            local_damage_score=0.0725,
            local_fragmentation=0.4,
            path_common_free_ratio=0.6594,
            largest_block_loss_slots=0,
            left_free_span_norm=0.0,
            right_free_span_norm=0.9688,
            plausibility_score=0.89,
            path_index=0,
            modulation_name="64QAM",
        ),
    )
    payload = build_payload(
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 1000})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 40})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 709,
                "episode_service_blocking_rate": 0.0100,
                "network_util_mean": 0.34,
                "network_util_max": 0.63,
                "free_slots_ratio": 0.66,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "high"})(),
        candidates=candidates,
        candidate_ids=("KSP", "LF", "FF"),
        candidate_roles=(
            "same_slot_backfill_challenger",
            "same_slot_preservation_anchor",
            "same_slot_backfill_challenger",
        ),
    )
    candidate_by_id = {
        candidate_id: candidate
        for candidate_id, candidate in zip(("KSP", "LF", "FF"), candidates, strict=True)
    }

    assert payload.prompt_context is not None
    assert payload.prompt_context.progress_ratio > 0.68
    assert resolve_override(
        payload=payload,
        candidate_by_id=candidate_by_id,
        winner_candidate_id="FF",
    ) is None
def test_ollama_system_prompt_mentions_same_path_slot_variant_guidance() -> None:
    from optical_networking_gym_v2.judge.ollama import _build_system_prompt

    prompt = _build_system_prompt()

    assert "do not imitate any hidden heuristic or ranking" in prompt
    assert "<decision_frame>" in prompt
    assert "<context>" in prompt
    assert "<guardrails>" in prompt
    assert "same_slot_near_tie_band" in prompt
    assert "same_slot_damage_axes_tie_band" in prompt
    assert "same_slot_local_support_band" in prompt
    assert "same_slot_route_common_free_alignment" in prompt
    assert "same_path_slot_variant_band" in prompt
    assert "candidate_roles" in prompt
    assert "shared roles mean a tied-best set" in prompt
    assert "same_slot_qot_leader" not in prompt
    assert "same_path_same_modulation=true" in prompt
    assert "delta_slot_span_total_norm" in prompt
    assert "smaller route.initial_slot" in prompt
    assert "last tie-breaker within the same-path family" in prompt
    assert "local_fragmentation and local_damage_score lower are better" in prompt
    assert "extra_slot_override=true only when an extra-slot candidate has a clearly material structural gain" in prompt
    assert "same_slot_winner_candidate_id must come from the minimum required_slots group" in prompt
    assert "not an answer key" in prompt
    assert "do not promote a winner from raw local_damage_score alone" in prompt
    assert "use balanced_tie_break and judge the whole trade-off" in prompt
    assert "Use same_slot_route_advantage only when a minimum-slot route/common-free story is genuinely stronger" in prompt
    assert "Smaller route.initial_slot alone is not enough for same_slot_route_advantage" in prompt
    assert "Use same_slot_local_advantage only when structural preservation clearly separates the winner" in prompt
    assert "support_count is a weak tie-breaker" in prompt
    assert "<examples>" in prompt
    assert "<self_check>" not in prompt
    assert "<internal_rubric>" not in prompt
    assert "Check 1" not in prompt
    assert "Check 2" not in prompt


def test_build_ollama_prompt_record_can_encode_repair_request() -> None:
    from optical_networking_gym_v2.judge.heuristic_judge import JudgeVerdict
    from optical_networking_gym_v2.judge.ollama import build_ollama_prompt_record

    module = _load_module()
    payload = module["build_judge_payload"](
        prompt_version="v5_contextual_judge",
        context=type(
            "Ctx",
            (),
            {
                "config": type("Cfg", (), {"episode_length": 100})(),
                "request": type("Req", (), {"source_id": 0, "destination_id": 1, "bit_rate": 100})(),
                "topology": type("Topo", (), {"node_names": ("A", "B")})(),
            },
        )(),
        topology_profile=None,
        operational_state=type(
            "State",
            (),
            {
                "services_processed": 82,
                "episode_service_blocking_rate": 0.0,
                "network_util_mean": 0.42,
                "network_util_max": 0.58,
                "free_slots_ratio": 0.50,
            },
        )(),
        global_regimes=type("Regimes", (), {"load_regime": "moderate"})(),
        candidates=(
            _manual_candidate(
                heuristic_name="load_balancing",
                raw_action=90,
                required_slots=1,
                route_pressure_score=0.15,
                local_damage_score=0.10,
                plausibility_score=0.90,
                path_common_free_ratio=0.70,
            ),
            _manual_candidate(
                heuristic_name="lowest_fragmentation",
                raw_action=91,
                required_slots=1,
                route_pressure_score=0.19,
                local_damage_score=0.01,
                plausibility_score=0.89,
                path_common_free_ratio=0.61,
            ),
        ),
        candidate_ids=("ROUTE", "SAFE"),
    )

    prompt_record = build_ollama_prompt_record(
        payload,
        repair_issue="same_slot_route_basis_mismatch",
        previous_verdict=JudgeVerdict(
            winner_candidate_id="SAFE",
            confidence=0.6,
            decision_basis="same_slot_route_advantage",
            ranking=("SAFE", "ROUTE"),
        ),
    )

    assert "repair_note" in prompt_record.user_prompt
    assert "same_slot_route_basis_mismatch" in prompt_record.user_prompt
    assert "previous_verdict" in prompt_record.user_prompt
    assert "payload" in prompt_record.user_prompt

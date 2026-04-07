from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import numpy as np

from optical_networking_gym_v2 import ScenarioConfig, make_env
from optical_networking_gym_v2.defaults import (
    DEFAULT_MEAN_HOLDING_TIME,
    DEFAULT_NUM_SPECTRUM_RESOURCES,
    DEFAULT_SEED,
)
from optical_networking_gym_v2.heuristics import (
    select_first_fit_runtime_action,
    select_highest_snr_first_fit_runtime_action,
    select_ksp_best_mod_last_fit_runtime_action,
    select_load_balancing_runtime_action,
    select_lowest_fragmentation_runtime_action,
)
from optical_networking_gym_v2.judge import (
    HeuristicJudge,
    JudgeCandidate,
    JudgeDecisionPayload,
    JudgePromptRecord,
    JudgeVerdict,
    OllamaHeuristicJudge,
    OllamaJudgeConfig,
    build_global_regimes,
    build_judge_audit_record,
    build_judge_candidate,
    build_judge_payload,
    build_ollama_prompt_record,
    build_operational_state,
    build_topology_profile,
    score_candidates,
)
from optical_networking_gym_v2.utils import experiment_scenarios as scenario_utils
from optical_networking_gym_v2.utils import sweep_reporting as report_utils


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent

_OLLAMA_BASE_URL = "https://assistant-charging-price-sega.trycloudflare.com"
_OLLAMA_MODEL = "qwen3.5:4b"#"qwen3.5:4b"
_OLLAMA_TEMPERATURE = 0.0
_OLLAMA_TIMEOUT_S = 60.0
_OLLAMA_MAX_RETRIES = 3
_OLLAMA_SKIP_EXPLANATION = True


def _resolve_ollama_think() -> bool | str:
    raw = os.environ.get("LLM_JUDGE_OLLAMA_THINK", "").strip().lower()
    if not raw or raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return raw


_OLLAMA_THINK = _resolve_ollama_think()

HEURISTIC_ORDER = (
    "first_fit",
    "load_balancing",
    "highest_snr_first_fit",
    "ksp_best_mod_last_fit",
    "lowest_fragmentation",
)

DECORATIVE_LLM_USAGE_MIN = 0.30
DECORATIVE_SHORTLIST_COLLAPSE_MAX = 0.70
DECORATIVE_HIDDEN_TOP1_MATCH_MAX = 0.85
DECORATIVE_MIN_DECISION_CHANGE_RATE = 0.10
DECORATIVE_ALERT_FAIL_COUNT = 2
DECORATIVE_REGRESSION_FAIL_COUNT = 4
DECORATIVE_MANUAL_REVIEW_CHECKS = (
    "strong_candidates_reach_llm",
    "prompt_changes_move_hard_cases",
)


@dataclass(frozen=True, slots=True)
class LLMJudgeExperiment:
    topology_id: str = "nobel-eu"
    scenario_profile: Literal["legacy_benchmark", "ofc_v1", "graph_load"] = "legacy_benchmark"
    episode_count: int = 5
    episode_length: int = 1000
    seed: int = DEFAULT_SEED
    load: float = scenario_utils.LEGACY_BENCHMARK_LOAD
    mean_holding_time: float = DEFAULT_MEAN_HOLDING_TIME
    num_spectrum_resources: int = DEFAULT_NUM_SPECTRUM_RESOURCES
    k_paths: int = scenario_utils.LEGACY_BENCHMARK_K_PATHS
    launch_power_dbm: float = scenario_utils.LEGACY_BENCHMARK_LAUNCH_POWER_DBM
    modulations_to_consider: int = scenario_utils.LEGACY_BENCHMARK_MODULATIONS_TO_CONSIDER
    measure_disruptions: bool = False
    drop_on_disruption: bool = False
    controller_version: str = "v5_contextual_judge"
    prompt_version: str = "v5_contextual_judge"
    min_prompt_candidates: int = 3
    max_prompt_candidates: int = 3
    plausibility_prune_gap: float = 0.10
    third_candidate_gap: float = 0.03
    same_slot_route_advantage_material: float = 0.015
    same_slot_local_advantage_material: float = 0.08
    same_slot_common_free_penalty_material: float = 0.05
    extra_slot_route_advantage_material: float = 0.03
    extra_slot_local_advantage_material: float = 0.10
    extra_slot_common_free_advantage_material: float = 0.08
    qot_tiebreak_cap: float = 1.0
    enable_semantic_repair: bool = False
    stop_when_blocking_exceeds_baseline: bool = False
    min_steps_before_stop: int = 0
    baseline_path: Path | None = None
    env_path: Path = REPO_ROOT / ".env"
    output_dir: Path = SCRIPT_DIR

    def __post_init__(self) -> None:
        object.__setattr__(self, "env_path", Path(self.env_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.baseline_path is not None:
            object.__setattr__(self, "baseline_path", Path(self.baseline_path))
        if self.scenario_profile not in {"legacy_benchmark", "ofc_v1", "graph_load"}:
            raise ValueError("scenario_profile must be 'legacy_benchmark', 'ofc_v1' or 'graph_load'")
        for name in (
            "episode_count",
            "episode_length",
            "num_spectrum_resources",
            "k_paths",
            "modulations_to_consider",
            "min_prompt_candidates",
            "max_prompt_candidates",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.load <= 0 or self.mean_holding_time <= 0:
            raise ValueError("load and mean_holding_time must be positive")
        if self.min_steps_before_stop < 0:
            raise ValueError("min_steps_before_stop must be non-negative")
        if self.min_prompt_candidates not in {3, 4}:
            raise ValueError("min_prompt_candidates must be 3 or 4")
        if self.max_prompt_candidates not in {3, 4}:
            raise ValueError("max_prompt_candidates must be 3 or 4")
        if self.min_prompt_candidates > self.max_prompt_candidates:
            raise ValueError("min_prompt_candidates must be <= max_prompt_candidates")


@dataclass(frozen=True, slots=True)
class LLMJudgeOutputs:
    steps_csv: Path
    summary_csv: Path
    calls_jsonl: Path


def _load_best_heuristic_blocking_threshold(*, baseline_path: Path, load: float) -> float:
    baseline_payload = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    for load_entry in baseline_payload.get("loads", []):
        if math.isclose(float(load_entry["load"]), float(load), rel_tol=0.0, abs_tol=1e-9):
            return float(load_entry["best_service_blocking_rate_mean"])
    raise ValueError(f"load {load} not found in baseline file {baseline_path}")


def build_base_scenario(experiment: LLMJudgeExperiment) -> ScenarioConfig:
    if experiment.scenario_profile == "legacy_benchmark":
        return scenario_utils.build_legacy_benchmark_scenario(
            topology_id=experiment.topology_id,
            episode_length=experiment.episode_length,
            seed=experiment.seed,
            load=experiment.load,
            mean_holding_time=experiment.mean_holding_time,
            num_spectrum_resources=experiment.num_spectrum_resources,
            k_paths=experiment.k_paths,
            launch_power_dbm=experiment.launch_power_dbm,
            modulations_to_consider=experiment.modulations_to_consider,
            measure_disruptions=experiment.measure_disruptions,
            drop_on_disruption=experiment.drop_on_disruption,
        )
    if experiment.scenario_profile == "ofc_v1":
        return scenario_utils.build_nobel_eu_ofc_v1_scenario(
            topology_id=experiment.topology_id,
            episode_length=experiment.episode_length,
            seed=experiment.seed,
            load=experiment.load,
            mean_holding_time=experiment.mean_holding_time,
            num_spectrum_resources=experiment.num_spectrum_resources,
            k_paths=experiment.k_paths,
            launch_power_dbm=experiment.launch_power_dbm,
            modulations_to_consider=experiment.modulations_to_consider,
            measure_disruptions=experiment.measure_disruptions,
            drop_on_disruption=experiment.drop_on_disruption,
        )
    return scenario_utils.build_nobel_eu_graph_load_scenario(
        topology_id=experiment.topology_id,
        episode_length=experiment.episode_length,
        seed=experiment.seed,
        load=experiment.load,
        mean_holding_time=experiment.mean_holding_time,
        num_spectrum_resources=experiment.num_spectrum_resources,
        k_paths=experiment.k_paths,
        launch_power_dbm=experiment.launch_power_dbm,
        modulations_to_consider=experiment.modulations_to_consider,
        measure_disruptions=experiment.measure_disruptions,
        drop_on_disruption=experiment.drop_on_disruption,
    )


def build_episode_scenario(
    *,
    experiment: LLMJudgeExperiment,
    base_scenario: ScenarioConfig,
    episode_index: int,
) -> ScenarioConfig:
    from dataclasses import replace

    return replace(
        base_scenario,
        scenario_id=f"{experiment.topology_id}_llm_judge_seed{experiment.seed + episode_index}",
        seed=int(experiment.seed + episode_index),
    )


def build_env(*, scenario: ScenarioConfig):
    return make_env(config=scenario)


def _date_prefix(now: datetime | None = None) -> str:
    current = datetime.now() if now is None else now
    return current.strftime("%d-%m-%Hh%M")


def _build_output_paths(*, output_dir: Path, now: datetime | None = None) -> LLMJudgeOutputs:
    prefix = _date_prefix(now)
    return LLMJudgeOutputs(
        steps_csv=Path(output_dir) / f"{prefix}-llm-judge-steps.csv",
        summary_csv=Path(output_dir) / f"{prefix}-llm-judge-summary.csv",
        calls_jsonl=Path(output_dir) / f"{prefix}-llm-judge-calls.jsonl",
    )


def _dedupe_actions(candidate_actions: Mapping[str, int]) -> list[tuple[str, tuple[str, ...], int]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for heuristic_name in HEURISTIC_ORDER:
        grouped[int(candidate_actions[heuristic_name])].append(heuristic_name)
    return [(heuristic_names[0], tuple(heuristic_names), int(action)) for action, heuristic_names in grouped.items()]


def _build_candidates(*, context, candidate_actions: Mapping[str, int]) -> tuple[JudgeCandidate, ...]:
    return tuple(
        build_judge_candidate(
            context=context,
            heuristic_name=canonical_name,
            action=action,
            proposed_by=proposed_by,
        )
        for canonical_name, proposed_by, action in _dedupe_actions(candidate_actions)
    )


def _select_candidate_actions(*, context) -> dict[str, int]:
    return {
        "first_fit": int(select_first_fit_runtime_action(context)),
        "load_balancing": int(select_load_balancing_runtime_action(context)),
        "highest_snr_first_fit": int(select_highest_snr_first_fit_runtime_action(context)),
        "ksp_best_mod_last_fit": int(select_ksp_best_mod_last_fit_runtime_action(context)),
        "lowest_fragmentation": int(select_lowest_fragmentation_runtime_action(context)),
    }


def _candidate_sort_key(candidate: JudgeCandidate) -> tuple[float, int, float, float, float, int]:
    return (
        float(candidate.metrics.plausibility_score),
        -int(candidate.metrics.required_slots),
        -float(candidate.metrics.route_pressure_score),
        -float(candidate.metrics.local_damage_score),
        float(candidate.metrics.qot_margin_clipped_db),
        -int(candidate.raw_action),
    )


def _candidate_identity(candidate: JudgeCandidate) -> tuple[int, str]:
    return int(candidate.raw_action), str(candidate.heuristic_name)


def _prompt_action_identity(candidate: JudgeCandidate) -> int:
    return int(candidate.raw_action)


def _append_unique_candidates(
    selected: list[JudgeCandidate],
    candidates: Sequence[JudgeCandidate],
    *,
    limit: int,
) -> None:
    seen = {_candidate_identity(candidate) for candidate in selected}
    for candidate in candidates:
        identity = _candidate_identity(candidate)
        if identity in seen:
            continue
        selected.append(candidate)
        seen.add(identity)
        if len(selected) >= limit:
            return


def _best_route_candidate(candidates: Sequence[JudgeCandidate]) -> JudgeCandidate:
    return min(
        candidates,
        key=lambda candidate: (
            float(candidate.metrics.route_pressure_score),
            -float(candidate.metrics.path_common_free_ratio),
            float(candidate.metrics.local_damage_score),
            int(candidate.metrics.required_slots),
            -float(candidate.metrics.qot_margin_clipped_db),
            int(candidate.raw_action),
        ),
    )


def _best_local_candidate(candidates: Sequence[JudgeCandidate]) -> JudgeCandidate:
    return min(
        candidates,
        key=lambda candidate: (
            float(candidate.metrics.local_damage_score),
            float(candidate.metrics.route_pressure_score),
            int(candidate.metrics.required_slots),
            -float(candidate.metrics.path_common_free_ratio),
            -float(candidate.metrics.qot_margin_clipped_db),
            int(candidate.raw_action),
        ),
    )


def _best_preservation_candidate(candidates: Sequence[JudgeCandidate]) -> JudgeCandidate:
    return min(
        candidates,
        key=lambda candidate: (
            int(candidate.metrics.fragmentation_added_blocks),
            int(candidate.metrics.largest_block_loss_slots),
            float(candidate.metrics.local_fragmentation),
            float(candidate.metrics.local_damage_score),
            -(float(candidate.metrics.left_free_span_norm) + float(candidate.metrics.right_free_span_norm)),
            -float(candidate.metrics.common_block_length_norm),
            -float(candidate.metrics.path_common_free_ratio),
            float(candidate.metrics.route_pressure_score),
            -float(candidate.metrics.qot_margin_clipped_db),
            -_candidate_support_count(candidate),
            int(candidate.raw_action),
        ),
    )


def _best_common_free_candidate(candidates: Sequence[JudgeCandidate]) -> JudgeCandidate:
    return min(
        candidates,
        key=lambda candidate: (
            -float(candidate.metrics.path_common_free_ratio),
            int(candidate.metrics.fragmentation_added_blocks),
            int(candidate.metrics.largest_block_loss_slots),
            float(candidate.metrics.local_damage_score),
            float(candidate.metrics.route_pressure_score),
            -float(candidate.metrics.qot_margin_clipped_db),
            -_candidate_support_count(candidate),
            int(candidate.raw_action),
        ),
    )


def _shares_same_path_and_modulation(lhs: JudgeCandidate, rhs: JudgeCandidate) -> bool:
    lhs_action = lhs.decoded_action
    rhs_action = rhs.decoded_action
    if lhs_action is None or rhs_action is None:
        return False
    if int(lhs.metrics.required_slots) != int(rhs.metrics.required_slots):
        return False
    if int(lhs_action.path_index) != int(rhs_action.path_index):
        return False
    if str(lhs_action.modulation_name) != str(rhs_action.modulation_name):
        return False
    if int(lhs_action.initial_slot) == int(rhs_action.initial_slot):
        return False
    return bool(
        math.isclose(
            float(lhs.metrics.route_pressure_score),
            float(rhs.metrics.route_pressure_score),
            abs_tol=1e-9,
        )
        and math.isclose(
            float(lhs.metrics.path_common_free_ratio),
            float(rhs.metrics.path_common_free_ratio),
            abs_tol=1e-9,
        )
    )


def _same_path_slot_variant_priority(
    candidate: JudgeCandidate,
    *,
    anchor: JudgeCandidate,
) -> tuple[int, int, float, float, float, float, float, int]:
    structural_gap = abs(
        int(candidate.metrics.fragmentation_added_blocks) - int(anchor.metrics.fragmentation_added_blocks)
    ) + abs(int(candidate.metrics.largest_block_loss_slots) - int(anchor.metrics.largest_block_loss_slots))
    span_gap = abs(float(candidate.metrics.left_free_span_norm) - float(anchor.metrics.left_free_span_norm)) + abs(
        float(candidate.metrics.right_free_span_norm) - float(anchor.metrics.right_free_span_norm)
    )
    candidate_span_total = float(candidate.metrics.left_free_span_norm) + float(candidate.metrics.right_free_span_norm)
    return (
        int(structural_gap > 0),
        int(round(span_gap * 1000.0)),
        abs(float(candidate.metrics.local_fragmentation) - float(anchor.metrics.local_fragmentation)),
        candidate_span_total,
        float(candidate.metrics.plausibility_score),
        float(candidate.metrics.qot_margin_clipped_db),
        float(candidate.metrics.common_block_length_norm),
        -int(candidate.raw_action),
    )


def _same_path_slot_variant_challenger(
    candidates: Sequence[JudgeCandidate],
    *,
    anchor: JudgeCandidate,
) -> JudgeCandidate | None:
    peers = [
        candidate
        for candidate in candidates
        if _candidate_identity(candidate) != _candidate_identity(anchor)
        and _shares_same_path_and_modulation(candidate, anchor)
    ]
    if not peers:
        return None
    return max(peers, key=lambda candidate: _same_path_slot_variant_priority(candidate, anchor=anchor))


def _best_slot_candidate(candidates: Sequence[JudgeCandidate]) -> JudgeCandidate:
    return min(
        candidates,
        key=lambda candidate: (
            int(candidate.metrics.required_slots),
            float(candidate.metrics.route_pressure_score),
            float(candidate.metrics.local_damage_score),
            -float(candidate.metrics.path_common_free_ratio),
            -float(candidate.metrics.qot_margin_clipped_db),
            int(candidate.raw_action),
        ),
    )


def _best_balanced_candidate(candidates: Sequence[JudgeCandidate]) -> JudgeCandidate:
    return max(candidates, key=_candidate_sort_key)


def _candidate_support_count(candidate: JudgeCandidate) -> int:
    return int(len(candidate.proposed_by))


def _same_slot_tradeoff_key(
    candidate: JudgeCandidate,
    *,
    balanced: JudgeCandidate,
    same_slot_route_advantage_material: float,
    same_slot_local_advantage_material: float,
    same_slot_common_free_penalty_material: float,
) -> tuple[int, float, float, float, float, float, float, int]:
    route_gain = float(balanced.metrics.route_pressure_score) - float(candidate.metrics.route_pressure_score)
    local_gain = float(balanced.metrics.local_damage_score) - float(candidate.metrics.local_damage_score)
    common_free_delta = (
        float(candidate.metrics.path_common_free_ratio) - float(balanced.metrics.path_common_free_ratio)
    )
    route_material = (
        route_gain >= same_slot_route_advantage_material
        and common_free_delta >= -same_slot_common_free_penalty_material
    )
    local_material = (
        local_gain >= same_slot_local_advantage_material
        and common_free_delta >= -same_slot_common_free_penalty_material
    )
    common_free_material = common_free_delta >= same_slot_common_free_penalty_material
    return (
        1 if (route_material or local_material or common_free_material) else 0,
        max(route_gain, 0.0),
        max(local_gain, 0.0),
        max(common_free_delta, 0.0),
        float(candidate.metrics.plausibility_score),
        -float(candidate.metrics.route_pressure_score),
        -float(candidate.metrics.local_damage_score),
        -int(candidate.raw_action),
    )


def _same_slot_route_priority(
    candidate: JudgeCandidate,
    *,
    anchor: JudgeCandidate,
) -> tuple[float, float, int, float, float, int]:
    return (
        float(anchor.metrics.route_pressure_score) - float(candidate.metrics.route_pressure_score),
        float(candidate.metrics.path_common_free_ratio) - float(anchor.metrics.path_common_free_ratio),
        _candidate_support_count(candidate),
        float(candidate.metrics.plausibility_score),
        float(candidate.metrics.qot_margin_clipped_db),
        -int(candidate.raw_action),
    )


def _same_slot_local_priority(
    candidate: JudgeCandidate,
    *,
    anchor: JudgeCandidate,
) -> tuple[float, float, int, float, float, int]:
    return (
        float(anchor.metrics.local_damage_score) - float(candidate.metrics.local_damage_score),
        float(candidate.metrics.qot_margin_clipped_db) - float(anchor.metrics.qot_margin_clipped_db),
        _candidate_support_count(candidate),
        float(candidate.metrics.path_common_free_ratio),
        float(candidate.metrics.plausibility_score),
        -int(candidate.raw_action),
    )


def _extra_slot_structural_priority(
    candidate: JudgeCandidate,
    *,
    anchor: JudgeCandidate,
) -> tuple[float, float, float, float, int, float, int]:
    return (
        float(anchor.metrics.route_pressure_score) - float(candidate.metrics.route_pressure_score),
        float(anchor.metrics.local_damage_score) - float(candidate.metrics.local_damage_score),
        float(candidate.metrics.path_common_free_ratio) - float(anchor.metrics.path_common_free_ratio),
        float(candidate.metrics.qot_margin_clipped_db) - float(anchor.metrics.qot_margin_clipped_db),
        _candidate_support_count(candidate),
        float(candidate.metrics.plausibility_score),
        -int(candidate.raw_action),
    )


def _contrast_priority(
    candidate: JudgeCandidate,
    *,
    selected: Sequence[JudgeCandidate],
) -> tuple[int, int, float, float, float, float, float, int, float, int]:
    if not selected:
        return (
            0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            float(candidate.metrics.plausibility_score),
            -int(candidate.raw_action),
        )
    max_slot_gap = max(
        abs(int(candidate.metrics.required_slots) - int(existing.metrics.required_slots))
        for existing in selected
    )
    max_route_gap = max(
        abs(float(candidate.metrics.route_pressure_score) - float(existing.metrics.route_pressure_score))
        for existing in selected
    )
    max_local_gap = max(
        abs(float(candidate.metrics.local_damage_score) - float(existing.metrics.local_damage_score))
        for existing in selected
    )
    max_common_free_gap = max(
        abs(float(candidate.metrics.path_common_free_ratio) - float(existing.metrics.path_common_free_ratio))
        for existing in selected
    )
    max_qot_gap = max(
        abs(float(candidate.metrics.qot_margin_clipped_db) - float(existing.metrics.qot_margin_clipped_db))
        for existing in selected
    )
    meaningful_conflict = max(
        1 if _has_meaningful_axis_conflict(candidate, existing) else 0
        for existing in selected
    )
    structural_advantage = max(
        int(candidate.metrics.fragmentation_added_blocks < existing.metrics.fragmentation_added_blocks)
        + int(candidate.metrics.largest_block_loss_slots < existing.metrics.largest_block_loss_slots)
        for existing in selected
    )
    return (
        int(meaningful_conflict),
        int(max_slot_gap > 0),
        float(max(max_route_gap, max_local_gap, max_common_free_gap)),
        float(max_common_free_gap),
        float(max_local_gap),
        float(max_route_gap),
        float(max_qot_gap),
        int(structural_advantage),
        float(candidate.metrics.plausibility_score),
        -int(candidate.raw_action),
    )


def _passes_extra_slot_gate(
    candidate: JudgeCandidate,
    *,
    anchor: JudgeCandidate,
    extra_slot_route_advantage_material: float,
    extra_slot_local_advantage_material: float,
    extra_slot_common_free_advantage_material: float,
) -> bool:
    route_gain = float(anchor.metrics.route_pressure_score) - float(candidate.metrics.route_pressure_score)
    local_gain = float(anchor.metrics.local_damage_score) - float(candidate.metrics.local_damage_score)
    common_free_gain = (
        float(candidate.metrics.path_common_free_ratio) - float(anchor.metrics.path_common_free_ratio)
    )
    material_local = local_gain >= extra_slot_local_advantage_material
    material_common_free = common_free_gain >= extra_slot_common_free_advantage_material
    material_route = route_gain >= extra_slot_route_advantage_material
    supportive_local = local_gain >= (extra_slot_local_advantage_material * 0.5)
    supportive_common_free = common_free_gain >= (extra_slot_common_free_advantage_material * 0.5)
    return bool(
        material_local
        or material_common_free
        or (material_route and (supportive_local or supportive_common_free))
    )


def _should_expand_same_slot_to_four(
    min_slot_candidates: Sequence[JudgeCandidate],
    *,
    selected_entries: Sequence[tuple[str, JudgeCandidate]],
    structural_candidates: Sequence[JudgeCandidate],
    target_count: int,
    third_candidate_gap: float,
) -> bool:
    if target_count < 4:
        return False
    if structural_candidates:
        return False
    if len(min_slot_candidates) < 4:
        return False
    selected_ids = {_candidate_identity(candidate) for _role, candidate in selected_entries}
    remaining_same_slot = [
        candidate
        for candidate in min_slot_candidates
        if _candidate_identity(candidate) not in selected_ids
    ]
    if not remaining_same_slot:
        return False
    best_same_slot_score = max(float(candidate.metrics.plausibility_score) for candidate in min_slot_candidates)
    best_remaining = max(remaining_same_slot, key=_candidate_sort_key)
    if best_same_slot_score - float(best_remaining.metrics.plausibility_score) > third_candidate_gap:
        return False
    selected_candidates = tuple(candidate for _role, candidate in selected_entries)
    contrast = _contrast_priority(best_remaining, selected=selected_candidates)
    return bool(
        contrast[0] > 0
        or not best_remaining.metrics.is_pareto_dominated
        or _candidate_support_count(best_remaining) > 1
    )


def _has_material_same_slot_challenger(
    balanced: JudgeCandidate,
    candidates: Sequence[JudgeCandidate],
    *,
    same_slot_route_advantage_material: float,
    same_slot_local_advantage_material: float,
    same_slot_common_free_penalty_material: float,
) -> bool:
    for candidate in candidates:
        if _candidate_identity(candidate) == _candidate_identity(balanced):
            continue
        if int(candidate.metrics.required_slots) != int(balanced.metrics.required_slots):
            continue
        if _same_slot_tradeoff_key(
            candidate,
            balanced=balanced,
            same_slot_route_advantage_material=same_slot_route_advantage_material,
            same_slot_local_advantage_material=same_slot_local_advantage_material,
            same_slot_common_free_penalty_material=same_slot_common_free_penalty_material,
        )[0]:
            return True
    return False


def _axis_leaders(candidates: Sequence[JudgeCandidate]) -> tuple[JudgeCandidate, ...]:
    if not candidates:
        return ()
    selected: list[JudgeCandidate] = []
    _append_unique_candidates(selected, (_best_balanced_candidate(candidates),), limit=4)
    _append_unique_candidates(selected, (_best_route_candidate(candidates),), limit=4)
    _append_unique_candidates(selected, (_best_local_candidate(candidates),), limit=4)
    _append_unique_candidates(selected, (_best_slot_candidate(candidates),), limit=4)
    return tuple(selected)


def _materially_worse_axes(candidate: JudgeCandidate, survivor: JudgeCandidate) -> int:
    worse_axes = 0
    if int(candidate.metrics.required_slots) >= int(survivor.metrics.required_slots) + 1:
        worse_axes += 1
    if float(candidate.metrics.route_pressure_score) >= float(survivor.metrics.route_pressure_score) + 0.04:
        worse_axes += 1
    if float(candidate.metrics.local_damage_score) >= float(survivor.metrics.local_damage_score) + 0.05:
        worse_axes += 1
    if (
        float(candidate.metrics.qot_margin_clipped_db) <= float(survivor.metrics.qot_margin_clipped_db) - 0.75
        and int(candidate.metrics.required_slots) >= int(survivor.metrics.required_slots)
    ):
        worse_axes += 1
    return worse_axes


def _has_meaningful_axis_conflict(lhs: JudgeCandidate, rhs: JudgeCandidate) -> bool:
    route_gap = abs(float(lhs.metrics.route_pressure_score) - float(rhs.metrics.route_pressure_score))
    local_gap = abs(float(lhs.metrics.local_damage_score) - float(rhs.metrics.local_damage_score))
    same_slots = int(lhs.metrics.required_slots) == int(rhs.metrics.required_slots)
    opposite_tradeoff = (
        float(lhs.metrics.route_pressure_score) < float(rhs.metrics.route_pressure_score)
        and float(lhs.metrics.local_damage_score) > float(rhs.metrics.local_damage_score)
    ) or (
        float(lhs.metrics.route_pressure_score) > float(rhs.metrics.route_pressure_score)
        and float(lhs.metrics.local_damage_score) < float(rhs.metrics.local_damage_score)
    )
    return bool(
        (same_slots and (route_gap >= 0.03 or local_gap >= 0.03))
        or opposite_tradeoff
        or route_gap >= 0.05
        or local_gap >= 0.05
    )


def _plausible_candidates(
    candidates: Sequence[JudgeCandidate],
    *,
    prune_gap: float,
) -> tuple[JudgeCandidate, ...]:
    safe_candidates = tuple(
        candidate for candidate in candidates if candidate.metrics.qot_safe_now and not candidate.is_reject
    )
    if not safe_candidates:
        return ()
    ranked = sorted(safe_candidates, key=_candidate_sort_key, reverse=True)
    best_score = float(ranked[0].metrics.plausibility_score)
    axis_leader_ids = {_candidate_identity(candidate) for candidate in _axis_leaders(ranked)}
    plausible = [
        candidate
        for candidate in ranked
        if (
            float(candidate.metrics.plausibility_score) >= best_score - prune_gap
            or _candidate_identity(candidate) in axis_leader_ids
        )
    ]
    if len(plausible) <= 1:
        return tuple(plausible[:1])
    filtered = [
        candidate
        for candidate in plausible
        if (
            not candidate.metrics.is_pareto_dominated
            or float(candidate.metrics.plausibility_score) >= best_score - 0.03
            or _candidate_identity(candidate) in axis_leader_ids
        )
    ]
    if not filtered:
        filtered = plausible
    return tuple(filtered)


def _build_prompt_pool(
    candidates: Sequence[JudgeCandidate],
    *,
    fallback_candidates: Sequence[JudgeCandidate] | None = None,
    min_prompt_candidates: int,
) -> tuple[JudgeCandidate, ...]:
    safe_non_reject = [
        candidate for candidate in candidates if candidate.metrics.qot_safe_now and not candidate.is_reject
    ]
    if len(safe_non_reject) >= min_prompt_candidates:
        return tuple(safe_non_reject)

    fallback_source = tuple(candidates if fallback_candidates is None else fallback_candidates)
    non_reject = [candidate for candidate in fallback_source if not candidate.is_reject]
    if len(non_reject) <= len(safe_non_reject):
        return tuple(safe_non_reject)

    selected = list(safe_non_reject)
    seen = {_candidate_identity(candidate) for candidate in selected}
    for candidate in non_reject:
        identity = _candidate_identity(candidate)
        if identity in seen:
            continue
        selected.append(candidate)
        seen.add(identity)
        if len(selected) >= min(min_prompt_candidates, len(non_reject)):
            break
    return tuple(selected if selected else non_reject)


def _select_prompt_candidates(
    candidates: Sequence[JudgeCandidate],
    *,
    min_prompt_candidates: int,
    max_prompt_candidates: int,
    third_candidate_gap: float,
    same_slot_route_advantage_material: float = 0.015,
    same_slot_local_advantage_material: float = 0.08,
    same_slot_common_free_penalty_material: float = 0.05,
    extra_slot_route_advantage_material: float = 0.03,
    extra_slot_local_advantage_material: float = 0.10,
    extra_slot_common_free_advantage_material: float = 0.08,
) -> tuple[JudgeCandidate, ...]:
    return tuple(candidate for _role, candidate in _select_prompt_candidate_entries(
        candidates,
        min_prompt_candidates=min_prompt_candidates,
        max_prompt_candidates=max_prompt_candidates,
        third_candidate_gap=third_candidate_gap,
        same_slot_route_advantage_material=same_slot_route_advantage_material,
        same_slot_local_advantage_material=same_slot_local_advantage_material,
        same_slot_common_free_penalty_material=same_slot_common_free_penalty_material,
        extra_slot_route_advantage_material=extra_slot_route_advantage_material,
        extra_slot_local_advantage_material=extra_slot_local_advantage_material,
        extra_slot_common_free_advantage_material=extra_slot_common_free_advantage_material,
    ))


def _select_prompt_candidate_entries(
    candidates: Sequence[JudgeCandidate],
    *,
    min_prompt_candidates: int,
    max_prompt_candidates: int,
    third_candidate_gap: float,
    same_slot_route_advantage_material: float = 0.015,
    same_slot_local_advantage_material: float = 0.08,
    same_slot_common_free_penalty_material: float = 0.05,
    extra_slot_route_advantage_material: float = 0.03,
    extra_slot_local_advantage_material: float = 0.10,
    extra_slot_common_free_advantage_material: float = 0.08,
) -> tuple[tuple[str, JudgeCandidate], ...]:
    ranked = sorted(candidates, key=_candidate_sort_key, reverse=True)
    if not ranked:
        return ()

    min_required_slots = min(int(candidate.metrics.required_slots) for candidate in ranked)
    min_slot_candidates = [
        candidate for candidate in ranked if int(candidate.metrics.required_slots) == min_required_slots
    ]
    balanced_anchor = _best_balanced_candidate(min_slot_candidates)
    preservation_anchor = _best_preservation_candidate(min_slot_candidates)
    selected_entries: list[tuple[str, JudgeCandidate]] = [("balanced_anchor", balanced_anchor)]
    seen_actions = {_prompt_action_identity(balanced_anchor)}
    if _prompt_action_identity(preservation_anchor) not in seen_actions:
        selected_entries.append(("same_slot_preservation_anchor", preservation_anchor))
        seen_actions.add(_prompt_action_identity(preservation_anchor))
    slot_variant_challenger = _same_path_slot_variant_challenger(
        min_slot_candidates,
        anchor=preservation_anchor,
    )
    if (
        slot_variant_challenger is not None
        and _prompt_action_identity(slot_variant_challenger) not in seen_actions
    ):
        selected_entries.append(("same_path_slot_variant_challenger", slot_variant_challenger))
        seen_actions.add(_prompt_action_identity(slot_variant_challenger))

    same_slot_leaders = (
        ("same_slot_route_challenger", _best_route_candidate(min_slot_candidates)),
        ("same_slot_common_free_challenger", _best_common_free_candidate(min_slot_candidates)),
    )
    for role, candidate in same_slot_leaders:
        action_identity = _prompt_action_identity(candidate)
        if action_identity in seen_actions:
            continue
        selected_entries.append((role, candidate))
        seen_actions.add(action_identity)

    extra_slot_candidates = [
        candidate
        for candidate in ranked
        if int(candidate.metrics.required_slots) > min_required_slots
        and _prompt_action_identity(candidate) not in seen_actions
    ]
    structural_candidates = [
        candidate
        for candidate in extra_slot_candidates
        if _passes_extra_slot_gate(
            candidate,
            anchor=preservation_anchor,
            extra_slot_route_advantage_material=extra_slot_route_advantage_material,
            extra_slot_local_advantage_material=extra_slot_local_advantage_material,
            extra_slot_common_free_advantage_material=extra_slot_common_free_advantage_material,
        )
    ]

    target_count = min(max_prompt_candidates, len(ranked))
    target_count = max(min(target_count, len(ranked)), min(min_prompt_candidates, len(ranked)))
    same_slot_limit = min(3, len(min_slot_candidates), target_count)
    if third_candidate_gap < 0.0:
        raise ValueError("third_candidate_gap must be non-negative")
    while len(selected_entries) < same_slot_limit:
        remaining_same_slot = [
            candidate
            for candidate in min_slot_candidates
            if _prompt_action_identity(candidate) not in seen_actions
        ]
        if not remaining_same_slot:
            break
        selected_candidates = tuple(candidate for _role, candidate in selected_entries)
        same_slot_backfill = max(
            remaining_same_slot,
            key=lambda candidate: _contrast_priority(candidate, selected=selected_candidates),
        )
        selected_entries.append(("same_slot_backfill_challenger", same_slot_backfill))
        seen_actions.add(_prompt_action_identity(same_slot_backfill))

    if structural_candidates and len(selected_entries) < target_count:
        selected_candidates = tuple(candidate for _role, candidate in selected_entries)
        extra_slot_candidate = max(
            structural_candidates,
            key=lambda candidate: (
                _contrast_priority(candidate, selected=selected_candidates),
                _extra_slot_structural_priority(candidate, anchor=preservation_anchor),
            ),
        )
        selected_entries.append(("extra_slot_structural_challenger", extra_slot_candidate))
        seen_actions.add(_prompt_action_identity(extra_slot_candidate))

    structural_backfill_candidates = [
        candidate
        for candidate in structural_candidates
        if _prompt_action_identity(candidate) not in seen_actions
    ]
    while len(selected_entries) < target_count and structural_backfill_candidates:
        selected_candidates = tuple(candidate for _role, candidate in selected_entries)
        candidate = max(
            structural_backfill_candidates,
            key=lambda item: (
                _contrast_priority(item, selected=selected_candidates),
                _extra_slot_structural_priority(item, anchor=preservation_anchor),
            ),
        )
        selected_entries.append(("extra_slot_backfill_challenger", candidate))
        seen_actions.add(_prompt_action_identity(candidate))
        structural_backfill_candidates = [
            item for item in structural_backfill_candidates if _prompt_action_identity(item) not in seen_actions
        ]
    return tuple(selected_entries[: target_count if len(selected_entries) > same_slot_limit else same_slot_limit])


def _shuffle_prompt_entries(
    prompt_entries: Sequence[tuple[str, JudgeCandidate]],
    *,
    rng: np.random.Generator,
) -> tuple[tuple[tuple[str, JudgeCandidate], ...], tuple[int, ...]]:
    if len(prompt_entries) <= 1:
        return tuple(prompt_entries), tuple(range(len(prompt_entries)))
    order = tuple(int(index) for index in rng.permutation(len(prompt_entries)).tolist())
    return tuple(prompt_entries[index] for index in order), order


def _shuffle_candidates_for_prompt(
    candidates: Sequence[JudgeCandidate],
    *,
    rng: np.random.Generator,
) -> tuple[JudgeCandidate, ...]:
    shuffled_entries, _order = _shuffle_prompt_entries(
        tuple(("candidate", candidate) for candidate in candidates),
        rng=rng,
    )
    return tuple(candidate for _role, candidate in shuffled_entries)


def _make_candidate_ids(*, rng: np.random.Generator, count: int) -> tuple[str, ...]:
    alphabet = np.array(list("23456789BCDFGHJKLMNPQRSTVWXYZ"))
    candidate_ids: list[str] = []
    while len(candidate_ids) < count:
        token = "".join(rng.choice(alphabet, size=4).tolist())
        if token not in candidate_ids:
            candidate_ids.append(token)
    return tuple(candidate_ids)


def _serialize_decoded_path_nodes(candidate: JudgeCandidate) -> str:
    if candidate.decoded_action is None:
        return ""
    return "->".join(candidate.decoded_action.path_node_names)


def _format_decisive_signals_summary(verdict: JudgeVerdict | None) -> str:
    if verdict is None or not verdict.decisive_signals:
        return ""
    return "|".join(
        f"{signal.factor}:{signal.supports}:{signal.importance}"
        for signal in verdict.decisive_signals
    )


def _resolve_prompt_and_model_io(
    *,
    judge_client: HeuristicJudge,
    payload: JudgeDecisionPayload,
    verdict: JudgeVerdict | None,
) -> tuple[JudgePromptRecord, dict[str, object] | None, object | None]:
    prompt_record = build_ollama_prompt_record(payload)
    raw_model_response: dict[str, object] | None = None
    parsed_response: object | None = None if verdict is None else verdict.to_mapping()
    consume_trace = getattr(judge_client, "consume_last_trace", None)
    if callable(consume_trace):
        trace = consume_trace()
        if trace is not None:
            prompt_record = trace.prompt
            raw_model_response = None if trace.raw_model_response is None else dict(trace.raw_model_response)
            parsed_response = trace.parsed_response
    return prompt_record, raw_model_response, parsed_response


def _collect_semantic_warning_flags(
    payload: JudgeDecisionPayload,
    verdict: JudgeVerdict,
    *,
    same_slot_route_advantage_material: float = 0.015,
    same_slot_local_advantage_material: float = 0.08,
    same_slot_common_free_penalty_material: float = 0.05,
) -> tuple[str, ...]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in payload.candidates}
    winner = candidate_by_id.get(verdict.winner_candidate_id)
    if winner is None:
        return ("invalid_winner_candidate_id",)
    flags: list[str] = []
    same_slot_rivals = [
        candidate
        for candidate in payload.candidates
        if candidate.candidate_id != winner.candidate_id
        and int(candidate.metrics.required_slots) == int(winner.metrics.required_slots)
    ]
    min_required_slots = min(
        (int(candidate.metrics.required_slots) for candidate in payload.candidates),
        default=int(winner.metrics.required_slots),
    )
    if verdict.decision_basis == "same_slot_route_advantage":
        if not same_slot_rivals:
            flags.append("same_slot_route_basis_mismatch")
            return tuple(flags)
        best_route_candidate = min(
            (winner, *same_slot_rivals),
            key=lambda candidate: (
                float(candidate.metrics.route_pressure_score),
                -float(candidate.metrics.path_common_free_ratio),
                float(candidate.metrics.local_damage_score),
                -float(candidate.metrics.qot_margin_clipped_db),
                str(candidate.candidate_id),
            ),
        )
        if best_route_candidate.candidate_id != winner.candidate_id:
            flags.append("same_slot_route_basis_mismatch")
        for rival in same_slot_rivals:
            route_gain = float(rival.metrics.route_pressure_score) - float(winner.metrics.route_pressure_score)
            common_free_penalty = (
                float(rival.metrics.path_common_free_ratio) - float(winner.metrics.path_common_free_ratio)
            )
            if (
                route_gain < same_slot_route_advantage_material
                and common_free_penalty > same_slot_common_free_penalty_material
            ):
                flags.append("same_slot_route_basis_mismatch")
                break
    elif verdict.decision_basis == "same_slot_local_advantage":
        if not same_slot_rivals:
            flags.append("same_slot_local_basis_mismatch")
            return tuple(flags)
        best_local_candidate = min(
            (winner, *same_slot_rivals),
            key=lambda candidate: (
                float(candidate.metrics.local_damage_score),
                float(candidate.metrics.route_pressure_score),
                -float(candidate.metrics.path_common_free_ratio),
                -float(candidate.metrics.qot_margin_clipped_db),
                str(candidate.candidate_id),
            ),
        )
        if best_local_candidate.candidate_id != winner.candidate_id:
            flags.append("same_slot_local_basis_mismatch")
        supported = any(
            (
                float(rival.metrics.local_damage_score) - float(winner.metrics.local_damage_score)
            )
            >= same_slot_local_advantage_material
            and (
                int(winner.metrics.fragmentation_added_blocks) < int(rival.metrics.fragmentation_added_blocks)
                or int(winner.metrics.largest_block_loss_slots) < int(rival.metrics.largest_block_loss_slots)
            )
            and (
                float(winner.metrics.path_common_free_ratio) + same_slot_common_free_penalty_material
                >= float(rival.metrics.path_common_free_ratio)
            )
            for rival in same_slot_rivals
        )
        if not supported:
            flags.append("same_slot_local_basis_mismatch")
    elif verdict.decision_basis == "extra_slot_structural_advantage":
        if int(winner.metrics.required_slots) <= min_required_slots:
            flags.append("extra_slot_structural_basis_mismatch")
        else:
            same_slot_winner = min(
                (
                    candidate
                    for candidate in payload.candidates
                    if int(candidate.metrics.required_slots) == min_required_slots
                ),
                key=lambda candidate: (
                    float(candidate.metrics.route_pressure_score),
                    float(candidate.metrics.local_damage_score),
                    -float(candidate.metrics.path_common_free_ratio),
                    str(candidate.candidate_id),
                ),
                default=winner,
            )
            route_gain = (
                float(same_slot_winner.metrics.route_pressure_score) - float(winner.metrics.route_pressure_score)
            )
            local_gain = (
                float(same_slot_winner.metrics.local_damage_score) - float(winner.metrics.local_damage_score)
            )
            common_free_gain = (
                float(winner.metrics.path_common_free_ratio) - float(same_slot_winner.metrics.path_common_free_ratio)
            )
            if not (
                route_gain >= 0.03
                or local_gain >= 0.10
                or common_free_gain >= 0.08
            ):
                flags.append("extra_slot_structural_basis_mismatch")
    return tuple(flags)


def _build_step_row(
    *,
    date_label: str,
    prompt_version: str,
    topology_name: str,
    episode_index: int,
    step_index: int,
    payload: JudgeDecisionPayload,
    verdict: JudgeVerdict | None,
    agrees_with_reference: bool,
    reference_winner: str,
    winner_candidate_id: str,
    controller_decision_source: str,
    raw_candidate_count: int,
    surviving_candidate_count: int,
    pruned_dominated_count: int,
    prompt_candidate_count: int,
    fallback_reason: str,
    judge_error_message: str,
    semantic_warning_flags: Sequence[str],
    basis_vs_payload_mismatch: bool,
    pre_shuffle_shortlist_actions: Sequence[int],
    post_shuffle_shortlist_actions: Sequence[int],
    prompt_permutation: Sequence[int],
    hidden_balanced_candidate_id: str,
    hidden_balanced_candidate_action: int,
    hidden_balanced_candidate_heuristic: str,
    hidden_balanced_position: int,
    winner_prompt_position: int,
    winner_matches_hidden_balanced: bool,
    winner_candidate: JudgeCandidate,
    judge_called: bool,
    post_info: dict[str, object],
) -> dict[str, report_utils.Scalar]:
    return {
        "date": date_label,
        "prompt_version": prompt_version,
        "episode_index": int(episode_index),
        "step_index": int(step_index),
        "topology_name": topology_name,
        "services_processed": int(payload.request.services_processed),
        "winner_candidate_id": winner_candidate_id,
        "winner_heuristic": winner_candidate.heuristic_name,
        "winner_proposed_by": "|".join(winner_candidate.proposed_by),
        "winner_raw_action": int(winner_candidate.raw_action),
        "winner_decoded_path_nodes": _serialize_decoded_path_nodes(winner_candidate),
        "winner_modulation_name": (
            "" if winner_candidate.decoded_action is None else winner_candidate.decoded_action.modulation_name
        ),
        "winner_initial_slot": (
            -1 if winner_candidate.decoded_action is None else winner_candidate.decoded_action.initial_slot
        ),
        "winner_slot_end_exclusive": (
            -1 if winner_candidate.decoded_action is None else winner_candidate.decoded_action.slot_end_exclusive
        ),
        "winner_path_hops": 0 if winner_candidate.decoded_action is None else winner_candidate.decoded_action.path_hops,
        "winner_path_length_km": (
            0.0 if winner_candidate.decoded_action is None else winner_candidate.decoded_action.path_length_km
        ),
        "judge_called": bool(judge_called),
        "winner_confidence": 0.0 if verdict is None else float(verdict.confidence),
        "winner_decision_basis": "" if verdict is None else str(verdict.decision_basis),
        "decisive_signals_summary": _format_decisive_signals_summary(verdict),
        "reference_winner": reference_winner,
        "agrees_with_reference": bool(agrees_with_reference),
        "controller_decision_source": controller_decision_source,
        "raw_candidate_count": int(raw_candidate_count),
        "surviving_candidate_count": int(surviving_candidate_count),
        "pruned_dominated_count": int(pruned_dominated_count),
        "prompt_candidate_count": int(prompt_candidate_count),
        "pre_shuffle_shortlist_actions": "|".join(str(int(action)) for action in pre_shuffle_shortlist_actions),
        "post_shuffle_shortlist_actions": "|".join(str(int(action)) for action in post_shuffle_shortlist_actions),
        "prompt_permutation": "|".join(str(int(index)) for index in prompt_permutation),
        "hidden_balanced_candidate_id": str(hidden_balanced_candidate_id),
        "hidden_balanced_candidate_action": int(hidden_balanced_candidate_action),
        "hidden_balanced_candidate_heuristic": str(hidden_balanced_candidate_heuristic),
        "hidden_balanced_position": int(hidden_balanced_position),
        "winner_prompt_position": int(winner_prompt_position),
        "winner_matches_hidden_balanced": bool(winner_matches_hidden_balanced),
        "fallback_reason": fallback_reason,
        "judge_error_message": judge_error_message,
        "semantic_warning_flags": "|".join(str(flag) for flag in semantic_warning_flags),
        "basis_vs_payload_mismatch": bool(basis_vs_payload_mismatch),
        "post_status": str(post_info.get("status", "unknown")),
        "post_accepted": bool(post_info.get("accepted", False)),
        "post_reward": float(post_info.get("reward", 0.0)),
        "episode_service_blocking_rate": float(post_info.get("episode_service_blocking_rate", 0.0)),
        "episode_bit_rate_blocking_rate": float(post_info.get("episode_bit_rate_blocking_rate", 0.0)),
        "episode_disrupted_services_rate": float(post_info.get("episode_disrupted_services", 0.0)),
    }


def _print_progress(
    *,
    step: int,
    total_steps: int,
    episode: int,
    total_episodes: int,
    blocking_rate: float,
    load: float,
    elapsed_s: float,
) -> None:
    pct = step / total_steps if total_steps > 0 else 0.0
    bar_width = 24
    filled = int(bar_width * pct)
    bar = "#" * filled + "-" * (bar_width - filled)
    s_per_step = elapsed_s / step if step > 0 else 0.0
    steps_per_s = step / elapsed_s if elapsed_s > 0 else 0.0
    line = (
        f"\rllm-judge [{bar}] {step}/{total_steps} {pct * 100:.1f}%"
        f" | {s_per_step:.1f}s/step ; {steps_per_s:.2f} steps/s"
        f" | ep {episode}/{total_episodes}"
        f" | block {blocking_rate:.3f}"
        f" | load {load:.0f}"
    )
    sys.stdout.write(line)
    sys.stdout.flush()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True))
        handle.write("\n")


def _format_exception_message(exc: Exception) -> str:
    message = str(exc).strip().replace("\n", " ")
    if not message:
        return type(exc).__name__
    if len(message) > 240:
        return message[:237] + "..."
    return message


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _is_non_operational_fallback_reason(reason: str) -> bool:
    normalized = str(reason or "").strip()
    if normalized in {"", "none"}:
        return False
    return not normalized.startswith(("judge_error:", "invalid_winner_candidate_id:"))


def _build_decorative_judge_audit(
    *,
    step_rows: Sequence[Mapping[str, report_utils.Scalar]],
) -> dict[str, report_utils.Scalar]:
    multi_option_rows = [row for row in step_rows if int(row["raw_candidate_count"]) >= 2]
    non_trivial_rows = [row for row in multi_option_rows if int(row["surviving_candidate_count"]) >= 2]
    llm_multi_option_rows = [
        row
        for row in non_trivial_rows
        if bool(row["judge_called"]) and int(row["prompt_candidate_count"]) >= 2
    ]
    collapsed_rows = [
        row
        for row in non_trivial_rows
        if int(row["prompt_candidate_count"]) <= 1 or not bool(row["judge_called"])
    ]
    non_operational_fallback_rows = [
        row for row in step_rows if _is_non_operational_fallback_reason(str(row["fallback_reason"]))
    ]
    hidden_top1_match_rows = [
        row for row in llm_multi_option_rows if bool(row["winner_matches_hidden_balanced"])
    ]
    disagreement_rows = [
        row for row in llm_multi_option_rows if not bool(row["winner_matches_hidden_balanced"])
    ]
    beneficial_disagreement_rows = [
        row
        for row in disagreement_rows
        if str(row["winner_heuristic"]) == str(row["reference_winner"])
        and str(row["hidden_balanced_candidate_heuristic"]) != str(row["reference_winner"])
    ]
    harmful_disagreement_rows = [
        row
        for row in disagreement_rows
        if str(row["winner_heuristic"]) != str(row["reference_winner"])
        and str(row["hidden_balanced_candidate_heuristic"]) == str(row["reference_winner"])
    ]

    llm_non_trivial_usage_rate = _safe_ratio(len(llm_multi_option_rows), len(non_trivial_rows))
    shortlist_collapse_rate = _safe_ratio(len(collapsed_rows), len(non_trivial_rows))
    non_operational_fallback_rate = _safe_ratio(
        len(non_operational_fallback_rows),
        max(1, len(llm_multi_option_rows)),
    )
    hidden_top1_match_rate = _safe_ratio(len(hidden_top1_match_rows), len(llm_multi_option_rows))
    decision_change_rate = _safe_ratio(len(disagreement_rows), len(llm_multi_option_rows))
    beneficial_disagreement_rate = _safe_ratio(
        len(beneficial_disagreement_rows),
        len(disagreement_rows),
    )

    failed_checks: list[str] = []
    if non_trivial_rows and llm_non_trivial_usage_rate < DECORATIVE_LLM_USAGE_MIN:
        failed_checks.append("llm_not_called_with_multiple_plausible_options")
    if non_trivial_rows and shortlist_collapse_rate > DECORATIVE_SHORTLIST_COLLAPSE_MAX:
        failed_checks.append("shortlist_or_controller_collapses_multi_option_cases")
    if non_operational_fallback_rows:
        failed_checks.append("fallback_used_for_normal_decision_cases")
    if llm_multi_option_rows and hidden_top1_match_rate > DECORATIVE_HIDDEN_TOP1_MATCH_MAX:
        failed_checks.append("llm_excessively_mirrors_hidden_top1")
    if llm_multi_option_rows and decision_change_rate < DECORATIVE_MIN_DECISION_CHANGE_RATE:
        failed_checks.append("replacing_llm_with_hidden_top1_changes_too_little")
    if not beneficial_disagreement_rows:
        failed_checks.append("llm_disagreements_never_show_benefit")

    failed_check_count = len(failed_checks)
    return {
        "multi_option_case_count": int(len(multi_option_rows)),
        "non_trivial_case_count": int(len(non_trivial_rows)),
        "llm_multi_option_case_count": int(len(llm_multi_option_rows)),
        "llm_non_trivial_usage_rate": llm_non_trivial_usage_rate,
        "shortlist_collapse_rate": shortlist_collapse_rate,
        "non_operational_fallback_count": int(len(non_operational_fallback_rows)),
        "non_operational_fallback_rate": non_operational_fallback_rate,
        "hidden_top1_match_rate_multi_option": hidden_top1_match_rate,
        "decision_change_rate_vs_hidden_top1": decision_change_rate,
        "llm_disagreement_count": int(len(disagreement_rows)),
        "beneficial_llm_disagreement_count": int(len(beneficial_disagreement_rows)),
        "harmful_llm_disagreement_count": int(len(harmful_disagreement_rows)),
        "beneficial_llm_disagreement_rate": beneficial_disagreement_rate,
        "decorative_check_non_trivial_llm_usage_pass": not (
            non_trivial_rows and llm_non_trivial_usage_rate < DECORATIVE_LLM_USAGE_MIN
        ),
        "decorative_check_shortlist_collapse_pass": not (
            non_trivial_rows and shortlist_collapse_rate > DECORATIVE_SHORTLIST_COLLAPSE_MAX
        ),
        "decorative_check_operational_fallback_only_pass": not bool(non_operational_fallback_rows),
        "decorative_check_not_mirroring_hidden_top1_pass": not (
            llm_multi_option_rows and hidden_top1_match_rate > DECORATIVE_HIDDEN_TOP1_MATCH_MAX
        ),
        "decorative_check_llm_replacement_changes_decisions_pass": not (
            llm_multi_option_rows and decision_change_rate < DECORATIVE_MIN_DECISION_CHANGE_RATE
        ),
        "decorative_check_llm_disagreement_sometimes_helps_pass": bool(beneficial_disagreement_rows),
        "decorative_failed_check_count_observable": int(failed_check_count),
        "decorative_failed_checks_observable": "|".join(failed_checks),
        "decorative_judge_alert_observable": bool(failed_check_count >= DECORATIVE_ALERT_FAIL_COUNT),
        "decorative_judge_regression_observable": bool(
            failed_check_count >= DECORATIVE_REGRESSION_FAIL_COUNT
        ),
        "decorative_manual_review_pending": "|".join(DECORATIVE_MANUAL_REVIEW_CHECKS),
    }


def _resolve_split_tradeoff_override(
    *,
    payload: JudgeDecisionPayload,
    candidate_by_id: Mapping[str, JudgeCandidate],
    winner_candidate_id: str,
    semantic_warning_flags: Sequence[str],
) -> str | None:
    winner_candidate = candidate_by_id.get(winner_candidate_id)
    if winner_candidate is None or str(winner_candidate.heuristic_name) != "lowest_fragmentation":
        return None
    if len(payload.candidates) != 3:
        return None
    if {
        str(candidate.heuristic_name)
        for candidate in candidate_by_id.values()
    } != {"first_fit", "load_balancing", "lowest_fragmentation"}:
        return None

    flags = {str(flag) for flag in semantic_warning_flags}
    if not flags.intersection({"same_slot_route_basis_mismatch", "same_slot_local_basis_mismatch"}):
        return None

    prompt_context = payload.prompt_context
    local_support_band = str(prompt_context.same_slot_local_support_band)
    route_gain_band = str(prompt_context.route_gain_band_vs_same_slot_best)
    route_common_free_alignment = str(prompt_context.same_slot_route_common_free_alignment)
    if str(prompt_context.future_feasibility_risk_band) != "high":
        return None
    if route_common_free_alignment not in {"split", "aligned"}:
        return None
    if str(prompt_context.same_path_slot_variant_band) != "none":
        return None
    if local_support_band not in {"none", "partial"}:
        return None
    if route_gain_band not in {"none", "small"} and not (
        local_support_band == "none" and route_gain_band in {"material", "strong"}
    ):
        return None

    payload_candidates_by_id = {
        str(candidate.candidate_id): candidate
        for candidate in payload.candidates
    }
    winner_payload_candidate = payload_candidates_by_id.get(winner_candidate_id)
    if winner_payload_candidate is None:
        return None
    winner_roles = {str(role) for role in winner_payload_candidate.candidate_roles}
    if winner_roles.intersection({"same_slot_route_leader", "same_slot_common_free_leader"}):
        return None

    route_payload_candidate = next(
        (
            candidate
            for candidate in payload.candidates
            if "same_slot_route_leader" in candidate.candidate_roles
            and str(candidate_by_id[candidate.candidate_id].heuristic_name) != "lowest_fragmentation"
        ),
        None,
    )
    common_free_payload_candidate = next(
        (
            candidate
            for candidate in payload.candidates
            if "same_slot_common_free_leader" in candidate.candidate_roles
            and str(candidate_by_id[candidate.candidate_id].heuristic_name) != "lowest_fragmentation"
        ),
        None,
    )
    if route_payload_candidate is None or common_free_payload_candidate is None:
        return None

    route_candidate = candidate_by_id[route_payload_candidate.candidate_id]
    common_free_candidate = candidate_by_id[common_free_payload_candidate.candidate_id]
    if route_common_free_alignment != "split":
        return None

    route_gap = (
        float(common_free_candidate.metrics.route_pressure_score)
        - float(route_candidate.metrics.route_pressure_score)
    )
    common_free_gap = (
        float(common_free_candidate.metrics.path_common_free_ratio)
        - float(route_candidate.metrics.path_common_free_ratio)
    )
    if common_free_gap >= 0.05 and route_gap <= 0.03:
        return str(common_free_payload_candidate.candidate_id)
    return str(route_payload_candidate.candidate_id)


def _resolve_cross_path_preservation_override(
    *,
    payload: JudgeDecisionPayload,
    candidate_by_id: Mapping[str, JudgeCandidate],
    winner_candidate_id: str,
) -> str | None:
    winner_candidate = candidate_by_id.get(winner_candidate_id)
    if winner_candidate is None or str(winner_candidate.heuristic_name) != "first_fit":
        return None
    if len(payload.candidates) != 3:
        return None
    if {
        str(candidate.heuristic_name)
        for candidate in candidate_by_id.values()
    } != {"first_fit", "ksp_best_mod_last_fit", "lowest_fragmentation"}:
        return None

    prompt_context = payload.prompt_context
    if prompt_context is None:
        return None
    if str(prompt_context.future_feasibility_risk_band) != "high":
        return None
    if not 0.60 <= float(prompt_context.progress_ratio) <= 0.68:
        return None
    if str(prompt_context.same_path_slot_variant_band) != "material":
        return None
    if str(prompt_context.same_slot_route_common_free_alignment) != "aligned":
        return None
    if str(prompt_context.same_slot_damage_axes_tie_band) != "tied":
        return None
    if str(prompt_context.route_gain_band_vs_same_slot_best) != "strong":
        return None
    if str(prompt_context.local_gain_band_vs_same_slot_best) not in {"none", "small"}:
        return None
    if str(prompt_context.common_free_penalty_band_vs_same_slot_best) != "none":
        return None

    payload_candidates_by_heuristic = {
        str(candidate_by_id[candidate.candidate_id].heuristic_name): candidate
        for candidate in payload.candidates
    }
    ff_payload_candidate = payload_candidates_by_heuristic.get("first_fit")
    ksp_payload_candidate = payload_candidates_by_heuristic.get("ksp_best_mod_last_fit")
    lf_payload_candidate = payload_candidates_by_heuristic.get("lowest_fragmentation")
    if (
        ff_payload_candidate is None
        or ksp_payload_candidate is None
        or lf_payload_candidate is None
        or ff_payload_candidate.route is None
        or ksp_payload_candidate.route is None
        or lf_payload_candidate.route is None
    ):
        return None

    ff_route = ff_payload_candidate.route
    ksp_route = ksp_payload_candidate.route
    lf_route = lf_payload_candidate.route
    if (
        int(ff_route.path_index) != int(ksp_route.path_index)
        or str(ff_route.modulation_name) != str(ksp_route.modulation_name)
        or int(ff_route.required_slots) != int(ksp_route.required_slots)
    ):
        return None
    if int(lf_route.required_slots) != int(ff_route.required_slots):
        return None
    if int(ff_route.path_index) == int(lf_route.path_index):
        return None
    if int(ff_route.initial_slot) > 16:
        return None

    ff_candidate = candidate_by_id[ff_payload_candidate.candidate_id]
    ksp_candidate = candidate_by_id[ksp_payload_candidate.candidate_id]
    lf_candidate = candidate_by_id[lf_payload_candidate.candidate_id]
    ff_local_fragmentation = float(ff_candidate.metrics.local_fragmentation)
    ksp_local_fragmentation = float(ksp_candidate.metrics.local_fragmentation)
    lf_local_fragmentation = float(lf_candidate.metrics.local_fragmentation)
    if ff_local_fragmentation - lf_local_fragmentation < 0.15:
        return None
    if ksp_local_fragmentation - lf_local_fragmentation < 0.30:
        return None
    if float(ff_candidate.metrics.local_damage_score) - float(lf_candidate.metrics.local_damage_score) < 0.03:
        return None
    if float(ksp_candidate.metrics.local_damage_score) - float(lf_candidate.metrics.local_damage_score) < 0.05:
        return None
    route_penalty = float(lf_candidate.metrics.route_pressure_score) - float(ff_candidate.metrics.route_pressure_score)
    if route_penalty <= 0.0 or route_penalty > 0.12:
        return None
    common_free_penalty = (
        float(ff_candidate.metrics.path_common_free_ratio)
        - float(lf_candidate.metrics.path_common_free_ratio)
    )
    if common_free_penalty < 0.50:
        return None
    return str(lf_payload_candidate.candidate_id)


def _build_summary_rows(
    *,
    date_label: str,
    step_rows: list[dict[str, report_utils.Scalar]],
    episode_summaries: list[dict[str, report_utils.Scalar]],
) -> list[dict[str, report_utils.Scalar]]:
    summary_rows = list(episode_summaries)
    total_steps = len(step_rows)
    if total_steps == 0:
        return summary_rows
    llm_calls = sum(1 for row in step_rows if bool(row["judge_called"]))
    fallback_count = sum(1 for row in step_rows if str(row["fallback_reason"]) not in {"", "none"})
    episode_blocking_rates = [float(row["final_blocking_rate"]) for row in episode_summaries if row["scope"] == "episode"]
    llm_rows = [row for row in step_rows if bool(row["judge_called"])]
    hidden_balanced_agreement_rate = 0.0 if not llm_rows else float(
        sum(1 for row in llm_rows if bool(row["winner_matches_hidden_balanced"])) / len(llm_rows)
    )
    decorative_audit = _build_decorative_judge_audit(step_rows=step_rows)
    stopped_early_episodes = sum(1 for row in episode_summaries if bool(row.get("stopped_early", False)))
    baseline_blocking_threshold = next(
        (float(row["baseline_blocking_threshold"]) for row in episode_summaries if row.get("baseline_blocking_threshold") not in {"", None}),
        "",
    )
    stop_reasons = "|".join(
        str(row["stop_reason"])
        for row in episode_summaries
        if str(row.get("stop_reason", "")) not in {"", "env_done"}
    )
    summary_rows.append(
        {
            "date": date_label,
            "scope": "run",
            "episode_index": -1,
            "steps": total_steps,
            "llm_calls": llm_calls,
            "fallback_count": fallback_count,
            "reference_agreement_rate": float(
                sum(1 for row in step_rows if bool(row["agrees_with_reference"])) / total_steps
            ),
            "llm_only_agreement_rate": 0.0
            if not llm_rows
            else float(sum(1 for row in llm_rows if bool(row["agrees_with_reference"])) / len(llm_rows)),
            "hidden_balanced_agreement_rate": hidden_balanced_agreement_rate,
            "first_fit_choices": sum(1 for row in step_rows if row["winner_heuristic"] == "first_fit"),
            "load_balancing_choices": sum(1 for row in step_rows if row["winner_heuristic"] == "load_balancing"),
            "highest_snr_first_fit_choices": sum(
                1 for row in step_rows if row["winner_heuristic"] == "highest_snr_first_fit"
            ),
            "ksp_best_mod_last_fit_choices": sum(
                1 for row in step_rows if row["winner_heuristic"] == "ksp_best_mod_last_fit"
            ),
            "lowest_fragmentation_choices": sum(
                1 for row in step_rows if row["winner_heuristic"] == "lowest_fragmentation"
            ),
            "mean_episode_service_blocking_rate": 0.0 if not episode_blocking_rates else float(mean(episode_blocking_rates)),
            "final_blocking_rate": 0.0 if not episode_blocking_rates else float(mean(episode_blocking_rates)),
            "final_disrupted_rate": float(step_rows[-1]["episode_disrupted_services_rate"]),
            "stopped_early_episodes": stopped_early_episodes,
            "baseline_blocking_threshold": baseline_blocking_threshold,
            "stop_reason": stop_reasons,
            **decorative_audit,
        }
    )
    return summary_rows


def _fieldnames_for_rows(rows: Sequence[Mapping[str, report_utils.Scalar]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    return fieldnames


def run_experiment(
    *,
    experiment: LLMJudgeExperiment,
    judge: HeuristicJudge | None = None,
    now: datetime | None = None,
) -> LLMJudgeOutputs:
    outputs = _build_output_paths(output_dir=experiment.output_dir, now=now)
    outputs.calls_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if outputs.calls_jsonl.exists():
        outputs.calls_jsonl.unlink()

    date_label = _date_prefix(now)
    base_scenario = build_base_scenario(experiment)
    baseline_blocking_threshold = (
        _load_best_heuristic_blocking_threshold(
            baseline_path=experiment.baseline_path,
            load=experiment.load,
        )
        if experiment.stop_when_blocking_exceeds_baseline and experiment.baseline_path is not None
        else None
    )
    judge_client = judge if judge is not None else OllamaHeuristicJudge(
        OllamaJudgeConfig(
            base_url=_OLLAMA_BASE_URL,
            model=_OLLAMA_MODEL,
            temperature=_OLLAMA_TEMPERATURE,
            timeout_s=_OLLAMA_TIMEOUT_S,
            max_retries=_OLLAMA_MAX_RETRIES,
            skip_explanation=_OLLAMA_SKIP_EXPLANATION,
            think=_OLLAMA_THINK,
        )
    )

    step_rows: list[dict[str, report_utils.Scalar]] = []
    episode_summaries: list[dict[str, report_utils.Scalar]] = []

    for episode_index in range(experiment.episode_count):
        episode_scenario = build_episode_scenario(
            experiment=experiment,
            base_scenario=base_scenario,
            episode_index=episode_index,
        )
        rng = np.random.default_rng(int(episode_scenario.seed or 0))
        env = build_env(scenario=episode_scenario)
        _, current_info = env.reset(seed=int(episode_scenario.seed or 0))
        topology_profile = build_topology_profile(env.simulator.topology)

        episode_step_count = 0
        episode_llm_calls = 0
        episode_fallback_count = 0
        stopped_early = False
        stop_reason = "env_done"
        episode_start_time = time.monotonic()
        while True:
            context = env.heuristic_context()
            candidate_actions = _select_candidate_actions(context=context)
            candidates = _build_candidates(context=context, candidate_actions=candidate_actions)
            scored_candidates, reference_winner = score_candidates(candidates)
            raw_candidate_count = len(scored_candidates)
            pruned_dominated_count = sum(1 for candidate in scored_candidates if candidate.metrics.is_pareto_dominated)
            plausible_candidates = _plausible_candidates(
                scored_candidates,
                prune_gap=experiment.plausibility_prune_gap,
            )
            surviving_candidate_count = len(plausible_candidates)
            safe_candidates = tuple(
                candidate for candidate in scored_candidates if candidate.metrics.qot_safe_now and not candidate.is_reject
            )
            non_reject_candidates = tuple(candidate for candidate in scored_candidates if not candidate.is_reject)
            operational_state = build_operational_state(context=context, info=current_info)

            fallback_reason = "none"
            judge_error_message = ""
            judge_called = False
            verdict: JudgeVerdict | None = None
            semantic_warning_flags: tuple[str, ...] = ()
            basis_vs_payload_mismatch = False
            pre_shuffle_prompt_entries: tuple[tuple[str, JudgeCandidate], ...]
            post_shuffle_prompt_entries: tuple[tuple[str, JudgeCandidate], ...]
            prompt_permutation: tuple[int, ...]
            if len(non_reject_candidates) <= 1 and non_reject_candidates:
                controller_decision_source = "bypass_consensus"
                pre_shuffle_prompt_entries = (("balanced_anchor", non_reject_candidates[0]),)
                post_shuffle_prompt_entries = pre_shuffle_prompt_entries
                prompt_permutation = (0,)
            elif 0 < len(non_reject_candidates) < experiment.min_prompt_candidates:
                controller_decision_source = "bypass_small_prompt_pool"
                pre_shuffle_prompt_entries = (
                    ("balanced_anchor", max(non_reject_candidates, key=_candidate_sort_key)),
                )
                post_shuffle_prompt_entries = pre_shuffle_prompt_entries
                prompt_permutation = (0,)
            elif not safe_candidates:
                controller_decision_source = "bypass_consensus"
                pre_shuffle_prompt_entries = (
                    ("balanced_anchor", max(scored_candidates, key=_candidate_sort_key)),
                )
                post_shuffle_prompt_entries = pre_shuffle_prompt_entries
                prompt_permutation = (0,)
            else:
                controller_decision_source = "llm"
                prompt_pool = _build_prompt_pool(
                    scored_candidates,
                    fallback_candidates=scored_candidates,
                    min_prompt_candidates=experiment.min_prompt_candidates,
                )
                pre_shuffle_prompt_entries = _select_prompt_candidate_entries(
                    prompt_pool,
                    min_prompt_candidates=experiment.min_prompt_candidates,
                    max_prompt_candidates=experiment.max_prompt_candidates,
                    third_candidate_gap=experiment.third_candidate_gap,
                    same_slot_route_advantage_material=experiment.same_slot_route_advantage_material,
                    same_slot_local_advantage_material=experiment.same_slot_local_advantage_material,
                    same_slot_common_free_penalty_material=experiment.same_slot_common_free_penalty_material,
                    extra_slot_route_advantage_material=experiment.extra_slot_route_advantage_material,
                    extra_slot_local_advantage_material=experiment.extra_slot_local_advantage_material,
                    extra_slot_common_free_advantage_material=experiment.extra_slot_common_free_advantage_material,
                )
                post_shuffle_prompt_entries, prompt_permutation = _shuffle_prompt_entries(
                    pre_shuffle_prompt_entries,
                    rng=rng,
                )
            hidden_balanced_candidate = next(
                (candidate for role, candidate in pre_shuffle_prompt_entries if role == "balanced_anchor"),
                pre_shuffle_prompt_entries[0][1],
            )
            chosen_prompt_candidates = tuple(candidate for _role, candidate in post_shuffle_prompt_entries)
            candidate_roles = tuple(role for role, _candidate in post_shuffle_prompt_entries)
            pre_shuffle_shortlist_actions = tuple(
                int(candidate.raw_action) for _role, candidate in pre_shuffle_prompt_entries
            )
            post_shuffle_shortlist_actions = tuple(
                int(candidate.raw_action) for _role, candidate in post_shuffle_prompt_entries
            )
            prompt_candidate_count = len(chosen_prompt_candidates)
            payload = build_judge_payload(
                prompt_version=experiment.prompt_version,
                context=context,
                topology_profile=topology_profile,
                operational_state=operational_state,
                global_regimes=build_global_regimes(operational_state),
                candidates=chosen_prompt_candidates,
                candidate_ids=_make_candidate_ids(rng=rng, count=prompt_candidate_count),
                candidate_roles=candidate_roles,
            )
            candidate_by_id = {
                decision_candidate.candidate_id: candidate
                for decision_candidate, candidate in zip(payload.candidates, chosen_prompt_candidates, strict=True)
            }
            candidate_id_by_identity = {
                _candidate_identity(candidate): decision_candidate.candidate_id
                for decision_candidate, candidate in zip(payload.candidates, chosen_prompt_candidates, strict=True)
            }
            hidden_balanced_candidate_id = candidate_id_by_identity[_candidate_identity(hidden_balanced_candidate)]
            hidden_balanced_position = next(
                index
                for index, decision_candidate in enumerate(payload.candidates)
                if decision_candidate.candidate_id == hidden_balanced_candidate_id
            )
            winner_candidate_id = max(
                zip(payload.candidates, chosen_prompt_candidates, strict=True),
                key=lambda item: _candidate_sort_key(item[1]),
            )[0].candidate_id
            if controller_decision_source == "llm":
                judge_called = True
                episode_llm_calls += 1
                try:
                    verdict = judge_client.judge(payload)
                    winner_candidate_id = verdict.winner_candidate_id
                except Exception as exc:
                    judge_error_message = _format_exception_message(exc)
                    fallback_reason = f"judge_error:{type(exc).__name__}:{judge_error_message}"
                    controller_decision_source = "fallback_blocking_proxy"
                    verdict = None

            if winner_candidate_id not in candidate_by_id:
                fallback_reason = f"invalid_winner_candidate_id:{winner_candidate_id}"
                controller_decision_source = "fallback_blocking_proxy"
                verdict = None
            elif (
                controller_decision_source == "llm"
                and verdict is not None
                and (
                    candidate_by_id[winner_candidate_id].is_reject
                    or not candidate_by_id[winner_candidate_id].metrics.qot_safe_now
                )
            ):
                fallback_reason = f"unsafe_winner_candidate_id:{winner_candidate_id}"
                controller_decision_source = "fallback_blocking_proxy"
                verdict = None
            elif controller_decision_source == "llm" and verdict is not None:
                semantic_warning_flags = _collect_semantic_warning_flags(
                    payload,
                    verdict,
                    same_slot_route_advantage_material=experiment.same_slot_route_advantage_material,
                    same_slot_local_advantage_material=experiment.same_slot_local_advantage_material,
                    same_slot_common_free_penalty_material=experiment.same_slot_common_free_penalty_material,
                )
                if (
                    experiment.enable_semantic_repair
                    and semantic_warning_flags
                    and isinstance(judge_client, OllamaHeuristicJudge)
                ):
                    try:
                        repaired_verdict = judge_client.repair(
                            payload,
                            previous_verdict=verdict,
                            repair_issue="|".join(str(flag) for flag in semantic_warning_flags),
                        )
                    except Exception:
                        repaired_verdict = None
                    if repaired_verdict is not None:
                        repaired_flags = _collect_semantic_warning_flags(
                            payload,
                            repaired_verdict,
                            same_slot_route_advantage_material=experiment.same_slot_route_advantage_material,
                            same_slot_local_advantage_material=experiment.same_slot_local_advantage_material,
                            same_slot_common_free_penalty_material=experiment.same_slot_common_free_penalty_material,
                        )
                        if len(repaired_flags) < len(semantic_warning_flags):
                            verdict = repaired_verdict
                            winner_candidate_id = verdict.winner_candidate_id
                            semantic_warning_flags = repaired_flags
                basis_vs_payload_mismatch = bool(semantic_warning_flags)
                split_tradeoff_override = _resolve_split_tradeoff_override(
                    payload=payload,
                    candidate_by_id=candidate_by_id,
                    winner_candidate_id=winner_candidate_id,
                    semantic_warning_flags=semantic_warning_flags,
                )
                if split_tradeoff_override is not None:
                    winner_candidate_id = split_tradeoff_override
                    controller_decision_source = "llm_split_tradeoff_override"
            if fallback_reason != "none":
                episode_fallback_count += 1

            if controller_decision_source == "fallback_blocking_proxy":
                fallback_pool = chosen_prompt_candidates if chosen_prompt_candidates else scored_candidates
                fallback_candidate = max(fallback_pool, key=_candidate_sort_key)
                winner_candidate_id = next(
                    decision_candidate.candidate_id
                    for decision_candidate, candidate in zip(payload.candidates, chosen_prompt_candidates, strict=True)
                    if candidate.raw_action == fallback_candidate.raw_action
                    and candidate.heuristic_name == fallback_candidate.heuristic_name
                )

            winner_candidate = candidate_by_id[winner_candidate_id]
            winner_prompt_position = next(
                index
                for index, decision_candidate in enumerate(payload.candidates)
                if decision_candidate.candidate_id == winner_candidate_id
            )
            agrees_with_reference = winner_candidate.heuristic_name == reference_winner
            prompt_record, raw_model_response, parsed_response = _resolve_prompt_and_model_io(
                judge_client=judge_client,
                payload=payload,
                verdict=verdict,
            )
            _, reward, terminated, truncated, current_info = env.step(int(winner_candidate.raw_action))
            current_info = dict(current_info)
            current_info["reward"] = float(reward)

            step_rows.append(
                _build_step_row(
                    date_label=date_label,
                    prompt_version=experiment.prompt_version,
                    topology_name=topology_profile.friendly_name,
                    episode_index=episode_index,
                    step_index=episode_step_count,
                    payload=payload,
                    verdict=verdict,
                    agrees_with_reference=agrees_with_reference,
                    reference_winner=reference_winner,
                    winner_candidate_id=winner_candidate_id,
                    controller_decision_source=controller_decision_source,
                    raw_candidate_count=raw_candidate_count,
                    surviving_candidate_count=surviving_candidate_count,
                    pruned_dominated_count=pruned_dominated_count,
                    prompt_candidate_count=prompt_candidate_count,
                    fallback_reason=fallback_reason,
                    judge_error_message=judge_error_message,
                    semantic_warning_flags=semantic_warning_flags,
                    basis_vs_payload_mismatch=basis_vs_payload_mismatch,
                    pre_shuffle_shortlist_actions=pre_shuffle_shortlist_actions,
                    post_shuffle_shortlist_actions=post_shuffle_shortlist_actions,
                    prompt_permutation=prompt_permutation,
                    hidden_balanced_candidate_id=hidden_balanced_candidate_id,
                    hidden_balanced_candidate_action=int(hidden_balanced_candidate.raw_action),
                    hidden_balanced_candidate_heuristic=hidden_balanced_candidate.heuristic_name,
                    hidden_balanced_position=hidden_balanced_position,
                    winner_prompt_position=winner_prompt_position,
                    winner_matches_hidden_balanced=(
                        _candidate_identity(winner_candidate) == _candidate_identity(hidden_balanced_candidate)
                    ),
                    winner_candidate=winner_candidate,
                    judge_called=judge_called,
                    post_info=current_info,
                )
            )
            audit_record = build_judge_audit_record(
                date=date_label,
                prompt_version=experiment.prompt_version,
                seed=int(episode_scenario.seed or 0),
                episode_index=episode_index,
                step_index=episode_step_count,
                topology_id=episode_scenario.topology_id,
                decision_payload=payload,
                prompt=prompt_record,
                raw_model_response=raw_model_response,
                parsed_response=parsed_response,
                fallback_reason=fallback_reason,
                judge_error_message=judge_error_message,
                semantic_warning_flags=semantic_warning_flags,
                basis_vs_payload_mismatch=basis_vs_payload_mismatch,
                pre_shuffle_shortlist_actions=pre_shuffle_shortlist_actions,
                post_shuffle_shortlist_actions=post_shuffle_shortlist_actions,
                prompt_permutation=prompt_permutation,
                hidden_balanced_candidate_id=hidden_balanced_candidate_id,
                hidden_balanced_candidate_action=int(hidden_balanced_candidate.raw_action),
                hidden_balanced_candidate_heuristic=hidden_balanced_candidate.heuristic_name,
                candidates=scored_candidates,
                reference_winner=reference_winner,
                chosen_action=int(winner_candidate.raw_action),
                chosen_heuristic=winner_candidate.heuristic_name,
                winner_proposed_by=winner_candidate.proposed_by,
                controller_decision_source=controller_decision_source,
                raw_candidate_count=raw_candidate_count,
                surviving_candidate_count=surviving_candidate_count,
                pruned_dominated_count=pruned_dominated_count,
                prompt_candidate_count=prompt_candidate_count,
            )
            _append_jsonl(outputs.calls_jsonl, audit_record.to_mapping())

            episode_step_count += 1
            current_blocking_rate = float(current_info.get("episode_service_blocking_rate", 0.0))
            _print_progress(
                step=episode_step_count,
                total_steps=experiment.episode_length,
                episode=episode_index + 1,
                total_episodes=experiment.episode_count,
                blocking_rate=current_blocking_rate,
                load=experiment.load,
                elapsed_s=time.monotonic() - episode_start_time,
            )
            if (
                baseline_blocking_threshold is not None
                and episode_step_count >= experiment.min_steps_before_stop
                and current_blocking_rate > baseline_blocking_threshold
            ):
                stopped_early = True
                stop_reason = "baseline_blocking_exceeded"
                break
            if terminated or truncated:
                stop_reason = "env_done"
                break

        sys.stdout.write("\n")
        sys.stdout.flush()

        episode_rows = [row for row in step_rows if int(row["episode_index"]) == episode_index]
        episode_summaries.append(
            {
                "date": date_label,
                "scope": "episode",
                "episode_index": int(episode_index),
                "steps": episode_step_count,
                "llm_calls": episode_llm_calls,
                "fallback_count": episode_fallback_count,
                "reference_agreement_rate": float(
                    sum(1 for row in episode_rows if bool(row["agrees_with_reference"])) / max(1, len(episode_rows))
                ),
                "first_fit_choices": sum(1 for row in episode_rows if row["winner_heuristic"] == "first_fit"),
                "load_balancing_choices": sum(
                    1 for row in episode_rows if row["winner_heuristic"] == "load_balancing"
                ),
                "highest_snr_first_fit_choices": sum(
                    1 for row in episode_rows if row["winner_heuristic"] == "highest_snr_first_fit"
                ),
                "ksp_best_mod_last_fit_choices": sum(
                    1 for row in episode_rows if row["winner_heuristic"] == "ksp_best_mod_last_fit"
                ),
                "lowest_fragmentation_choices": sum(
                    1 for row in episode_rows if row["winner_heuristic"] == "lowest_fragmentation"
                ),
                "final_blocking_rate": float(episode_rows[-1]["episode_service_blocking_rate"]),
                "final_disrupted_rate": float(episode_rows[-1]["episode_disrupted_services_rate"]),
                "stopped_early": stopped_early,
                "stop_reason": stop_reason,
                "baseline_blocking_threshold": ""
                if baseline_blocking_threshold is None
                else baseline_blocking_threshold,
            }
        )
        env.close()

    step_fieldnames = _fieldnames_for_rows(step_rows) if step_rows else []
    summary_rows = _build_summary_rows(
        date_label=date_label,
        step_rows=step_rows,
        episode_summaries=episode_summaries,
    )
    summary_fieldnames = _fieldnames_for_rows(summary_rows) if summary_rows else []
    report_utils.write_csv_rows(path=outputs.steps_csv, fieldnames=step_fieldnames, rows=step_rows)
    report_utils.write_csv_rows(path=outputs.summary_csv, fieldnames=summary_fieldnames, rows=summary_rows)
    return outputs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM heuristic judge experiment")
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--episode-count", type=int, default=5)
    parser.add_argument("--load", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--baseline-path", type=Path, default=None)
    parser.add_argument("--stop-when-blocking-exceeds-baseline", action="store_true")
    parser.add_argument("--min-steps-before-stop", type=int, default=0)
    parser.add_argument(
        "--scenario-profile",
        choices=("legacy_benchmark", "ofc_v1", "graph_load"),
        default="legacy_benchmark",
    )
    args = parser.parse_args()

    experiment_kwargs: dict[str, object] = {
        "episode_length": args.episode_length,
        "episode_count": args.episode_count,
        "scenario_profile": args.scenario_profile,
    }
    if args.load is not None:
        experiment_kwargs["load"] = float(args.load)
    if args.seed is not None:
        experiment_kwargs["seed"] = int(args.seed)
    if args.output_dir is not None:
        experiment_kwargs["output_dir"] = args.output_dir
    if args.baseline_path is not None:
        experiment_kwargs["baseline_path"] = args.baseline_path
    if args.stop_when_blocking_exceeds_baseline:
        experiment_kwargs["stop_when_blocking_exceeds_baseline"] = True
    if args.min_steps_before_stop:
        experiment_kwargs["min_steps_before_stop"] = int(args.min_steps_before_stop)

    outputs = run_experiment(
        experiment=LLMJudgeExperiment(**experiment_kwargs),
    )
    print(asdict(outputs))


if __name__ == "__main__":
    main()

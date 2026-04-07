from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import runpy
from typing import Any

from optical_networking_gym_v2.heuristics import select_load_balancing_runtime_action
from optical_networking_gym_v2.utils import sweep_reporting as report_utils


SCRIPT_DIR = Path(__file__).resolve().parent
ONLINE_JUDGE_SCRIPT = SCRIPT_DIR / "online_heuristic_judge.py"


@dataclass(frozen=True, slots=True)
class FutureRegretConfig:
    calls_jsonl: Path
    output_dir: Path
    topology_id: str = "nobel-eu"
    scenario_profile: str = "legacy_benchmark"
    episode_length: int = 1000
    seed: int = 10
    horizons: tuple[int, ...] = (25, 50, 100)
    max_states: int | None = None
    min_step_index: int = 0
    max_step_index: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "calls_jsonl", Path(self.calls_jsonl))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.calls_jsonl.exists():
            raise FileNotFoundError(f"calls_jsonl not found: {self.calls_jsonl}")
        if not self.horizons:
            raise ValueError("at least one horizon is required")


def _load_online_judge_module() -> dict[str, Any]:
    return runpy.run_path(str(ONLINE_JUDGE_SCRIPT))


def _load_call_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _group_records_by_episode(records: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        audit = record["audit"]
        grouped[int(audit["episode_index"])].append(dict(record))
    for episode_records in grouped.values():
        episode_records.sort(key=lambda item: int(item["audit"]["step_index"]))
    return dict(sorted(grouped.items()))


def _info_counts(info: Mapping[str, object]) -> tuple[int, int, int]:
    processed = int(info.get("episode_services_processed", info.get("services_processed", 0)))
    accepted = int(info.get("episode_services_accepted", info.get("services_accepted", 0)))
    blocked = max(0, processed - accepted)
    return processed, accepted, blocked


def _candidate_metrics_match(
    prompt_metrics: Mapping[str, object],
    audit_metrics: Mapping[str, object],
    *,
    tolerance: float = 1e-4,
) -> bool:
    for key, prompt_value in prompt_metrics.items():
        if key not in audit_metrics:
            continue
        audit_value = audit_metrics[key]
        if isinstance(prompt_value, float):
            if abs(float(prompt_value) - float(audit_value)) > tolerance:
                return False
        else:
            if prompt_value != audit_value:
                return False
    return True


def _candidate_route_match(
    prompt_route: Mapping[str, object] | None,
    audit_candidate: Mapping[str, Any],
    *,
    tolerance: float = 1e-4,
) -> bool:
    decoded = audit_candidate.get("decoded_action")
    if prompt_route is None:
        return decoded is None
    if decoded is None:
        return False
    return bool(
        int(prompt_route["path_index"]) == int(decoded["path_index"])
        and int(prompt_route["path_hops"]) == int(decoded["path_hops"])
        and abs(float(prompt_route["path_length_km"]) - float(decoded["path_length_km"])) <= tolerance
        and str(prompt_route["modulation_name"]) == str(decoded["modulation_name"])
        and int(prompt_route["initial_slot"]) == int(decoded["initial_slot"])
        and int(prompt_route["required_slots"]) == int(decoded["required_slots"])
    )


def _resolve_prompt_candidates(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    prompt_candidates = record["decision_payload"]["candidates"]
    audit_candidates = list(record["audit"]["candidate_audit"])
    unmatched = list(range(len(audit_candidates)))
    prompt_actions = [
        int(action) for action in record["audit"].get("post_shuffle_shortlist_actions", [])
    ]
    resolved: list[dict[str, Any]] = []
    for prompt_index, prompt_candidate in enumerate(prompt_candidates):
        matches = [
            index
            for index in unmatched
            if bool(prompt_candidate["is_reject"]) == bool(audit_candidates[index]["is_reject"])
            and _candidate_metrics_match(prompt_candidate["metrics"], audit_candidates[index]["metrics"])
            and _candidate_route_match(prompt_candidate.get("route"), audit_candidates[index])
        ]
        match_index: int | None = matches[0] if len(matches) == 1 else None
        if match_index is None and prompt_index < len(prompt_actions):
            hinted_action = int(prompt_actions[prompt_index])
            action_matches = [
                index
                for index, audit_candidate in enumerate(audit_candidates)
                if int(audit_candidate["raw_action"]) == hinted_action
                and bool(prompt_candidate["is_reject"]) == bool(audit_candidate["is_reject"])
                and _candidate_route_match(prompt_candidate.get("route"), audit_candidate)
            ]
            metric_matches = [
                index
                for index in action_matches
                if _candidate_metrics_match(prompt_candidate["metrics"], audit_candidates[index]["metrics"])
            ]
            if len(metric_matches) == 1:
                match_index = metric_matches[0]
            elif len(action_matches) == 1:
                match_index = action_matches[0]
        if match_index is None:
            episode_index = int(record["audit"]["episode_index"])
            step_index = int(record["audit"]["step_index"])
            raise ValueError(
                f"could not uniquely resolve prompt candidate {prompt_candidate['candidate_id']} "
                f"at episode {episode_index}, step {step_index}; matches={matches}"
            )
        if match_index in unmatched:
            unmatched.remove(match_index)
        matched_audit = audit_candidates[match_index]
        resolved.append(
            {
                "candidate_id": str(prompt_candidate["candidate_id"]),
                "heuristic_name": str(matched_audit["heuristic_name"]),
                "proposed_by": tuple(matched_audit["proposed_by"]),
                "raw_action": int(matched_audit["raw_action"]),
                "required_slots": int(matched_audit["metrics"]["required_slots"]),
                "prompt_metrics": dict(prompt_candidate["metrics"]),
                "audit_metrics": dict(matched_audit["metrics"]),
            }
        )
    return resolved


def _replay_actions(env, actions: Sequence[int]) -> tuple[dict[str, object], bool]:
    last_info: dict[str, object] = {}
    ended = False
    for action in actions:
        _observation, _reward, terminated, truncated, info = env.step(int(action))
        last_info = dict(info)
        ended = bool(terminated or truncated)
        if ended:
            break
    return last_info, ended


def _clone_env(env):
    return copy.deepcopy(env)


def _collect_horizon_metrics(
    *,
    env,
    prefix_info: Mapping[str, object],
    candidate_action: int,
    horizons: Sequence[int],
) -> dict[int, tuple[int, int]]:
    max_horizon = max(int(horizon) for horizon in horizons)
    prefix_processed, prefix_accepted, prefix_blocked = _info_counts(prefix_info)
    horizon_metrics: dict[int, tuple[int, int]] = {}

    _observation, _reward, terminated, truncated, current_info = env.step(int(candidate_action))
    current_info = dict(current_info)
    steps_taken = 1
    if steps_taken in horizons:
        processed, accepted, blocked = _info_counts(current_info)
        horizon_metrics[steps_taken] = (blocked - prefix_blocked, accepted - prefix_accepted)

    while steps_taken < max_horizon and not (terminated or truncated):
        action = int(select_load_balancing_runtime_action(env.heuristic_context()))
        _observation, _reward, terminated, truncated, current_info = env.step(action)
        current_info = dict(current_info)
        steps_taken += 1
        if steps_taken in horizons:
            processed, accepted, blocked = _info_counts(current_info)
            del processed
            horizon_metrics[steps_taken] = (blocked - prefix_blocked, accepted - prefix_accepted)

    processed, accepted, blocked = _info_counts(current_info)
    del processed
    final_metrics = (blocked - prefix_blocked, accepted - prefix_accepted)
    for horizon in horizons:
        horizon_metrics.setdefault(int(horizon), final_metrics)
    return horizon_metrics


def _collect_horizon_metrics_from_fallback_replay(
    *,
    build_env,
    scenario,
    prefix_actions: Sequence[int],
    candidate_action: int,
    horizons: Sequence[int],
) -> dict[int, tuple[int, int]]:
    env = build_env(scenario=scenario)
    _observation, reset_info = env.reset(seed=int(scenario.seed or 0))
    prefix_info = dict(reset_info)
    if prefix_actions:
        prefix_info, ended = _replay_actions(env, prefix_actions)
        if ended:
            env.close()
            final_metrics = {int(horizon): (0, 0) for horizon in horizons}
            return final_metrics
    horizon_metrics = _collect_horizon_metrics(
        env=env,
        prefix_info=prefix_info,
        candidate_action=candidate_action,
        horizons=horizons,
    )
    env.close()
    return horizon_metrics


def select_future_best_candidate(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    horizons: Sequence[int] = (25, 50, 100),
) -> str:
    ordered_horizons = sorted(int(horizon) for horizon in horizons)
    largest_horizon = ordered_horizons[-1]
    medium_horizon = ordered_horizons[-2] if len(ordered_horizons) >= 2 else largest_horizon
    smallest_horizon = ordered_horizons[0]

    def sort_key(outcome: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
        return (
            int(outcome[f"blocking_delta_{largest_horizon}"]),
            int(outcome[f"blocking_delta_{medium_horizon}"]),
            int(outcome[f"blocking_delta_{smallest_horizon}"]),
            -int(outcome[f"accepted_delta_{largest_horizon}"]),
            int(outcome["required_slots"]),
            int(outcome["raw_action"]),
        )

    return min(outcomes, key=sort_key)["candidate_id"]


def _strategy_summary_row(
    *,
    strategy: str,
    rows: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
) -> dict[str, report_utils.Scalar]:
    if not rows:
        return {
            "scope": "strategy",
            "strategy": strategy,
            "coverage_count": 0,
            "future_best_hit_rate": 0.0,
            **{f"future_regret_mean_{horizon}": 0.0 for horizon in horizons},
        }
    summary: dict[str, report_utils.Scalar] = {
        "scope": "strategy",
        "strategy": strategy,
        "coverage_count": len(rows),
        "future_best_hit_rate": float(sum(1 for row in rows if bool(row["is_future_best"])) / len(rows)),
    }
    for horizon in horizons:
        summary[f"future_regret_mean_{horizon}"] = float(
            sum(float(row[f"blocking_delta_{horizon}"]) for row in rows) / len(rows)
        )
        summary[f"accepted_delta_mean_{horizon}"] = float(
            sum(float(row[f"accepted_delta_{horizon}"]) for row in rows) / len(rows)
        )
    return summary


def _fieldnames_for_rows(rows: Sequence[Mapping[str, report_utils.Scalar]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    return fieldnames


def _sorted_family(*names: str) -> str:
    ordered = sorted({str(name) for name in names if str(name)})
    return "|".join(ordered)


def _episode_phase(*, step_index: int, episode_length: int) -> str:
    if episode_length <= 0:
        return "unknown"
    return "first_half" if int(step_index) < max(1, int(episode_length) // 2) else "second_half"


def _prompt_position_by_candidate_id(prompt_candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        str(candidate["candidate_id"]): index
        for index, candidate in enumerate(prompt_candidates)
    }


def _build_outcome_row(
    *,
    record: Mapping[str, Any],
    prompt_candidate: Mapping[str, Any],
    horizon_metrics: Mapping[int, tuple[int, int]],
    horizons: Sequence[int],
    prompt_candidate_count: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "date": str(record["audit"]["date"]),
        "episode_index": int(record["audit"]["episode_index"]),
        "step_index": int(record["audit"]["step_index"]),
        "candidate_id": str(prompt_candidate["candidate_id"]),
        "heuristic_name": str(prompt_candidate["heuristic_name"]),
        "raw_action": int(prompt_candidate["raw_action"]),
        "required_slots": int(prompt_candidate["required_slots"]),
        "controller_decision_source": str(record["audit"]["controller_decision_source"]),
        "chosen_by_judge": bool(
            int(record["audit"]["chosen_action"]) == int(prompt_candidate["raw_action"])
            and str(record["audit"]["chosen_heuristic"]) == str(prompt_candidate["heuristic_name"])
        ),
        "prompt_candidate_count": int(prompt_candidate_count),
        "prompt_metrics": dict(prompt_candidate["prompt_metrics"]),
        "audit_metrics": dict(prompt_candidate["audit_metrics"]),
    }
    for horizon in horizons:
        blocking_delta, accepted_delta = horizon_metrics[int(horizon)]
        row[f"blocking_delta_{horizon}"] = int(blocking_delta)
        row[f"accepted_delta_{horizon}"] = int(accepted_delta)
    return row


def _outcome_sort_key(outcome: Mapping[str, Any]) -> tuple[float, int, float, float, float, int]:
    audit_metrics = outcome.get("audit_metrics")
    if not isinstance(audit_metrics, Mapping):
        audit_metrics = outcome.get("prompt_metrics")
    if not isinstance(audit_metrics, Mapping):
        return (
            0.0,
            -int(outcome["required_slots"]),
            0.0,
            0.0,
            0.0,
            -int(outcome["raw_action"]),
        )
    return (
        float(audit_metrics["plausibility_score"]),
        -int(outcome["required_slots"]),
        -float(audit_metrics["route_pressure_score"]),
        -float(audit_metrics["local_damage_score"]),
        float(audit_metrics["qot_margin_clipped_db"]),
        -int(outcome["raw_action"]),
    )


def _principal_rival_row(
    outcomes: Sequence[Mapping[str, Any]],
    *,
    chosen_candidate_id: str,
) -> Mapping[str, Any] | None:
    rivals = [row for row in outcomes if str(row["candidate_id"]) != str(chosen_candidate_id)]
    if not rivals:
        return None
    return max(rivals, key=_outcome_sort_key)


def _group_state_summary_rows(
    *,
    scope: str,
    key_name: str,
    state_rows: Sequence[Mapping[str, report_utils.Scalar]],
) -> list[dict[str, report_utils.Scalar]]:
    grouped: dict[str, list[Mapping[str, report_utils.Scalar]]] = defaultdict(list)
    for row in state_rows:
        grouped[str(row[key_name])].append(row)
    summary_rows: list[dict[str, report_utils.Scalar]] = []
    for group_value, rows in sorted(grouped.items(), key=lambda item: item[0]):
        count = len(rows)
        summary_rows.append(
            {
                "scope": scope,
                key_name: group_value,
                "coverage_count": count,
                "judge_future_best_hit_rate": float(
                    sum(1 for row in rows if bool(row["judge_matches_future_best"])) / max(1, count)
                ),
                "judge_reference_agreement_rate": float(
                    sum(1 for row in rows if bool(row["judge_matches_reference"])) / max(1, count)
                ),
                "judge_hidden_balanced_agreement_rate": float(
                    sum(1 for row in rows if bool(row["judge_matches_hidden_balanced"])) / max(1, count)
                ),
                "judge_position_0_rate": float(
                    sum(1 for row in rows if int(row["judge_winner_position"]) == 0) / max(1, count)
                ),
                "judge_position_1_rate": float(
                    sum(1 for row in rows if int(row["judge_winner_position"]) == 1) / max(1, count)
                ),
                "judge_position_2_rate": float(
                    sum(1 for row in rows if int(row["judge_winner_position"]) == 2) / max(1, count)
                ),
                "judge_position_3_rate": float(
                    sum(1 for row in rows if int(row["judge_winner_position"]) == 3) / max(1, count)
                ),
                "reference_position_0_rate": float(
                    sum(1 for row in rows if int(row["reference_winner_position"]) == 0) / max(1, count)
                ),
                "hidden_balanced_position_0_rate": float(
                    sum(1 for row in rows if int(row["hidden_balanced_position"]) == 0) / max(1, count)
                ),
            }
        )
    return summary_rows


def analyze_future_regret(*, config: FutureRegretConfig) -> tuple[Path, Path]:
    online_module = _load_online_judge_module()
    records = _load_call_records(config.calls_jsonl)
    grouped_records = _group_records_by_episode(records)
    experiment_cls = online_module["LLMJudgeExperiment"]
    build_base_scenario = online_module["build_base_scenario"]
    build_episode_scenario = online_module["build_episode_scenario"]
    build_env = online_module["build_env"]

    experiment = experiment_cls(
        topology_id=config.topology_id,
        scenario_profile=config.scenario_profile,
        episode_count=len(grouped_records),
        episode_length=config.episode_length,
        seed=config.seed,
        output_dir=config.output_dir,
    )
    base_scenario = build_base_scenario(experiment)

    detail_rows: list[dict[str, report_utils.Scalar]] = []
    state_rows: list[dict[str, report_utils.Scalar]] = []
    strategy_rows: dict[str, list[dict[str, report_utils.Scalar]]] = defaultdict(list)
    analyzed_states = 0

    for episode_index, episode_records in grouped_records.items():
        scenario = build_episode_scenario(
            experiment=experiment,
            base_scenario=base_scenario,
            episode_index=episode_index,
        )
        chosen_actions = [int(record["audit"]["chosen_action"]) for record in episode_records]
        episode_env = build_env(scenario=scenario)
        _observation, current_info = episode_env.reset(seed=int(scenario.seed or 0))
        current_info = dict(current_info)
        for step_offset, record in enumerate(episode_records):
            if config.max_states is not None and analyzed_states >= int(config.max_states):
                break
            step_index = int(record["audit"]["step_index"])
            if step_index < int(config.min_step_index):
                if step_offset < len(episode_records) - 1:
                    _observation, _reward, _terminated, _truncated, current_info = episode_env.step(
                        chosen_actions[step_offset]
                    )
                    current_info = dict(current_info)
                continue
            if config.max_step_index is not None and step_index > int(config.max_step_index):
                if step_offset < len(episode_records) - 1:
                    _observation, _reward, _terminated, _truncated, current_info = episode_env.step(
                        chosen_actions[step_offset]
                    )
                    current_info = dict(current_info)
                continue
            prompt_candidates = _resolve_prompt_candidates(record)
            if len(prompt_candidates) < 2:
                if step_offset < len(episode_records) - 1:
                    _observation, _reward, _terminated, _truncated, current_info = episode_env.step(
                        chosen_actions[step_offset]
                    )
                    current_info = dict(current_info)
                continue
            outcomes: list[dict[str, report_utils.Scalar]] = []
            for prompt_candidate in prompt_candidates:
                try:
                    candidate_env = _clone_env(episode_env)
                    prefix_info = dict(current_info)
                    horizon_metrics = _collect_horizon_metrics(
                        env=candidate_env,
                        prefix_info=prefix_info,
                        candidate_action=int(prompt_candidate["raw_action"]),
                        horizons=config.horizons,
                    )
                    candidate_env.close()
                except Exception:
                    horizon_metrics = _collect_horizon_metrics_from_fallback_replay(
                        build_env=build_env,
                        scenario=scenario,
                        prefix_actions=chosen_actions[:step_offset],
                        candidate_action=int(prompt_candidate["raw_action"]),
                        horizons=config.horizons,
                    )

                outcomes.append(
                    _build_outcome_row(
                        record=record,
                        prompt_candidate=prompt_candidate,
                        horizon_metrics=horizon_metrics,
                        horizons=config.horizons,
                        prompt_candidate_count=len(prompt_candidates),
                    )
                )

            if len(outcomes) < 2:
                continue

            future_best_candidate_id = str(select_future_best_candidate(outcomes, horizons=config.horizons))
            prompt_position = _prompt_position_by_candidate_id(prompt_candidates)
            prompt_set = "|".join(str(candidate["heuristic_name"]) for candidate in prompt_candidates)
            prompt_family = _sorted_family(*(str(candidate["heuristic_name"]) for candidate in prompt_candidates))
            episode_phase = _episode_phase(step_index=step_index, episode_length=int(config.episode_length))
            for row in outcomes:
                row["future_best_candidate_id"] = future_best_candidate_id
                row["is_future_best"] = bool(row["candidate_id"] == future_best_candidate_id)
                row["prompt_position"] = int(prompt_position[str(row["candidate_id"])])
                row["prompt_set"] = prompt_set
                row["prompt_family"] = prompt_family
                row["reference_winner"] = str(record["audit"]["reference_winner"])
                row["episode_phase"] = episode_phase
                detail_rows.append(row)

            chosen_rows = [row for row in outcomes if bool(row["chosen_by_judge"])]
            if chosen_rows:
                judge_row = chosen_rows[0]
                strategy_rows["judge_v5"].append(judge_row)
                future_best_row = next(
                    row for row in outcomes if str(row["candidate_id"]) == future_best_candidate_id
                )
                reference_row = next(
                    (
                        row
                        for row in outcomes
                        if str(row["heuristic_name"]) == str(record["audit"]["reference_winner"])
                    ),
                    None,
                )
                principal_rival_row = _principal_rival_row(
                    outcomes,
                    chosen_candidate_id=str(judge_row["candidate_id"]),
                )
                state_row: dict[str, report_utils.Scalar] = {
                    "date": str(record["audit"]["date"]),
                    "episode_index": int(record["audit"]["episode_index"]),
                    "step_index": int(record["audit"]["step_index"]),
                    "controller_decision_source": str(record["audit"]["controller_decision_source"]),
                    "prompt_candidate_count": len(prompt_candidates),
                    "prompt_set": prompt_set,
                    "prompt_family": prompt_family,
                    "episode_phase": episode_phase,
                    "judge_winner_candidate_id": str(judge_row["candidate_id"]),
                    "judge_winner_heuristic": str(judge_row["heuristic_name"]),
                    "judge_winner_position": int(prompt_position[str(judge_row["candidate_id"])]),
                    "reference_winner": str(record["audit"]["reference_winner"]),
                    "reference_winner_position": (
                        -1 if reference_row is None else int(prompt_position[str(reference_row["candidate_id"])])
                    ),
                    "hidden_balanced_candidate_id": str(record["audit"].get("hidden_balanced_candidate_id", "")),
                    "hidden_balanced_candidate_action": int(record["audit"].get("hidden_balanced_candidate_action", -1)),
                    "hidden_balanced_candidate_heuristic": str(
                        record["audit"].get("hidden_balanced_candidate_heuristic", "")
                    ),
                    "hidden_balanced_position": int(
                        prompt_position.get(str(record["audit"].get("hidden_balanced_candidate_id", "")), -1)
                    ),
                    "future_best_candidate_id": future_best_candidate_id,
                    "future_best_heuristic": str(future_best_row["heuristic_name"]),
                    "judge_matches_reference": bool(
                        str(judge_row["heuristic_name"]) == str(record["audit"]["reference_winner"])
                    ),
                    "judge_matches_hidden_balanced": bool(
                        str(judge_row["candidate_id"]) == str(record["audit"].get("hidden_balanced_candidate_id", ""))
                    ),
                    "judge_matches_future_best": bool(
                        str(judge_row["candidate_id"]) == future_best_candidate_id
                    ),
                    "conflict_family": _sorted_family(
                        str(judge_row["heuristic_name"]),
                        str(record["audit"]["reference_winner"]),
                    ),
                    "winner_vs_future_best_family": _sorted_family(
                        str(judge_row["heuristic_name"]),
                        str(future_best_row["heuristic_name"]),
                    ),
                }
                for horizon in config.horizons:
                    state_row[f"judge_blocking_delta_{horizon}"] = int(judge_row[f"blocking_delta_{horizon}"])
                    state_row[f"future_best_blocking_delta_{horizon}"] = int(
                        future_best_row[f"blocking_delta_{horizon}"]
                    )
                    state_row[f"judge_accepted_delta_{horizon}"] = int(judge_row[f"accepted_delta_{horizon}"])
                    state_row[f"future_best_accepted_delta_{horizon}"] = int(
                        future_best_row[f"accepted_delta_{horizon}"]
                    )
                if principal_rival_row is not None:
                    state_row["principal_rival_candidate_id"] = str(principal_rival_row["candidate_id"])
                    state_row["principal_rival_heuristic"] = str(principal_rival_row["heuristic_name"])
                    state_row["delta_required_slots_vs_rival"] = int(
                        int(judge_row["required_slots"]) - int(principal_rival_row["required_slots"])
                    )
                    state_row["delta_route_pressure_score_vs_rival"] = float(
                        float(judge_row["prompt_metrics"]["route_pressure_score"])
                        - float(principal_rival_row["prompt_metrics"]["route_pressure_score"])
                    )
                    state_row["delta_local_damage_score_vs_rival"] = float(
                        float(judge_row["prompt_metrics"]["local_damage_score"])
                        - float(principal_rival_row["prompt_metrics"]["local_damage_score"])
                    )
                    state_row["delta_path_common_free_ratio_vs_rival"] = float(
                        float(judge_row["prompt_metrics"]["path_common_free_ratio"])
                        - float(principal_rival_row["prompt_metrics"]["path_common_free_ratio"])
                    )
                    state_row["delta_qot_margin_clipped_db_vs_rival"] = float(
                        float(judge_row["prompt_metrics"]["qot_margin_clipped_db"])
                        - float(principal_rival_row["prompt_metrics"]["qot_margin_clipped_db"])
                    )
                state_rows.append(state_row)
            future_best_rows = [row for row in outcomes if bool(row["is_future_best"])]
            if future_best_rows:
                strategy_rows["oracle_future_best"].append(future_best_rows[0])
            for heuristic_name in (
                "first_fit",
                "load_balancing",
                "ksp_best_mod_last_fit",
                "lowest_fragmentation",
                "highest_snr_first_fit",
            ):
                heuristic_rows = [row for row in outcomes if str(row["heuristic_name"]) == heuristic_name]
                if heuristic_rows:
                    strategy_rows[heuristic_name].append(heuristic_rows[0])
            analyzed_states += 1
            if step_offset < len(episode_records) - 1:
                _observation, _reward, _terminated, _truncated, current_info = episode_env.step(
                    chosen_actions[step_offset]
                )
                current_info = dict(current_info)
        episode_env.close()
        if config.max_states is not None and analyzed_states >= int(config.max_states):
            break

    stem = config.calls_jsonl.stem.replace("-calls", "")
    detail_path = config.output_dir / f"{stem}-future-regret.csv"
    summary_path = config.output_dir / f"{stem}-future-regret-summary.csv"

    summary_rows: list[dict[str, report_utils.Scalar]] = [
        {
            "scope": "run",
            "strategy": "all_states",
            "coverage_count": len(state_rows),
            "candidate_rows": len(detail_rows),
            "max_states": -1 if config.max_states is None else int(config.max_states),
            "judge_position_0_rate": 0.0
            if not state_rows
            else float(sum(1 for row in state_rows if int(row["judge_winner_position"]) == 0) / len(state_rows)),
            "judge_position_1_rate": 0.0
            if not state_rows
            else float(sum(1 for row in state_rows if int(row["judge_winner_position"]) == 1) / len(state_rows)),
            "judge_position_2_rate": 0.0
            if not state_rows
            else float(sum(1 for row in state_rows if int(row["judge_winner_position"]) == 2) / len(state_rows)),
            "judge_position_3_rate": 0.0
            if not state_rows
            else float(sum(1 for row in state_rows if int(row["judge_winner_position"]) == 3) / len(state_rows)),
            "reference_position_0_rate": 0.0
            if not state_rows
            else float(sum(1 for row in state_rows if int(row["reference_winner_position"]) == 0) / len(state_rows)),
            "judge_hidden_balanced_agreement_rate": 0.0
            if not state_rows
            else float(sum(1 for row in state_rows if bool(row["judge_matches_hidden_balanced"])) / len(state_rows)),
            "hidden_balanced_position_0_rate": 0.0
            if not state_rows
            else float(sum(1 for row in state_rows if int(row["hidden_balanced_position"]) == 0) / len(state_rows)),
        }
    ]
    for strategy_name, rows in strategy_rows.items():
        summary_rows.append(
            _strategy_summary_row(
                strategy=strategy_name,
                rows=rows,
                horizons=config.horizons,
            )
        )
    summary_rows.extend(
        _group_state_summary_rows(
            scope="prompt_family",
            key_name="prompt_family",
            state_rows=state_rows,
        )
    )
    summary_rows.extend(
        _group_state_summary_rows(
            scope="conflict_family",
            key_name="conflict_family",
            state_rows=state_rows,
        )
    )
    summary_rows.extend(
        _group_state_summary_rows(
            scope="prompt_size",
            key_name="prompt_candidate_count",
            state_rows=state_rows,
        )
    )
    summary_rows.extend(
        _group_state_summary_rows(
            scope="episode_phase",
            key_name="episode_phase",
            state_rows=state_rows,
        )
    )

    report_utils.write_csv_rows(
        path=detail_path,
        fieldnames=_fieldnames_for_rows(state_rows),
        rows=state_rows,
    )
    report_utils.write_csv_rows(
        path=summary_path,
        fieldnames=_fieldnames_for_rows(summary_rows),
        rows=summary_rows,
    )
    return detail_path, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze future regret for LLM judge traces")
    parser.add_argument(
        "--calls-jsonl",
        type=Path,
        default=SCRIPT_DIR / "23-03-llm-judge-calls.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--topology-id", default="nobel-eu")
    parser.add_argument(
        "--scenario-profile",
        choices=("legacy_benchmark", "ofc_v1", "graph_load"),
        default="legacy_benchmark",
    )
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--horizons", default="25,50,100")
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--min-step-index", type=int, default=0)
    parser.add_argument("--max-step-index", type=int, default=None)
    args = parser.parse_args()

    horizons = tuple(int(token) for token in str(args.horizons).split(",") if token.strip())
    detail_path, summary_path = analyze_future_regret(
        config=FutureRegretConfig(
            calls_jsonl=args.calls_jsonl,
            output_dir=args.output_dir,
            topology_id=args.topology_id,
            scenario_profile=args.scenario_profile,
            episode_length=args.episode_length,
            seed=args.seed,
            horizons=horizons,
            max_states=args.max_states,
            min_step_index=args.min_step_index,
            max_step_index=args.max_step_index,
        )
    )
    print(json.dumps({"detail_csv": str(detail_path), "summary_csv": str(summary_path)}, ensure_ascii=True))


if __name__ == "__main__":
    main()

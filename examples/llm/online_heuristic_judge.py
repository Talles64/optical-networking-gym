from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from optical_networking_gym_v2 import ScenarioConfig, make_env
from optical_networking_gym_v2.defaults import (
    DEFAULT_K_PATHS,
    DEFAULT_LAUNCH_POWER_DBM,
    DEFAULT_LOAD,
    DEFAULT_MEAN_HOLDING_TIME,
    DEFAULT_MODULATIONS_TO_CONSIDER,
    DEFAULT_NUM_SPECTRUM_RESOURCES,
    DEFAULT_SEED,
)
from optical_networking_gym_v2.heuristics import (
    select_first_fit_runtime_action,
    select_load_balancing_runtime_action,
    select_random_runtime_action,
)
from optical_networking_gym_v2.judge import (
    HeuristicJudge,
    JudgeCandidate,
    JudgeDecisionPayload,
    JudgePromptRecord,
    JudgeVerdict,
    OllamaHeuristicJudge,
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

HEURISTIC_ORDER = ("first_fit", "load_balancing", "random")


@dataclass(frozen=True, slots=True)
class LLMJudgeExperiment:
    topology_id: str = "nobel-eu"
    episode_count: int = 1
    episode_length: int = 10
    seed: int = DEFAULT_SEED
    load: float = DEFAULT_LOAD
    mean_holding_time: float = DEFAULT_MEAN_HOLDING_TIME
    num_spectrum_resources: int = DEFAULT_NUM_SPECTRUM_RESOURCES
    k_paths: int = DEFAULT_K_PATHS
    launch_power_dbm: float = DEFAULT_LAUNCH_POWER_DBM
    modulations_to_consider: int = DEFAULT_MODULATIONS_TO_CONSIDER
    measure_disruptions: bool = True
    drop_on_disruption: bool = False
    prompt_version: str = "v2_blocking_risk_compact_payload"
    env_path: Path = REPO_ROOT / ".env"
    output_dir: Path = SCRIPT_DIR

    def __post_init__(self) -> None:
        object.__setattr__(self, "env_path", Path(self.env_path))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.episode_count <= 0:
            raise ValueError("episode_count must be positive")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.load <= 0:
            raise ValueError("load must be positive")
        if self.mean_holding_time <= 0:
            raise ValueError("mean_holding_time must be positive")
        if self.num_spectrum_resources <= 0:
            raise ValueError("num_spectrum_resources must be positive")
        if self.k_paths <= 0:
            raise ValueError("k_paths must be positive")
        if self.modulations_to_consider <= 0:
            raise ValueError("modulations_to_consider must be positive")


@dataclass(frozen=True, slots=True)
class LLMJudgeOutputs:
    steps_csv: Path
    summary_csv: Path
    calls_jsonl: Path


def build_base_scenario(experiment: LLMJudgeExperiment) -> ScenarioConfig:
    return scenario_utils.build_nobel_eu_graph_load_scenario(
        REPO_ROOT,
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
    return dataclass_replace(
        base_scenario,
        scenario_id=f"{experiment.topology_id}_llm_judge_seed{experiment.seed + episode_index}",
        seed=int(experiment.seed + episode_index),
    )


def dataclass_replace(instance, /, **changes):
    from dataclasses import replace

    return replace(instance, **changes)


def build_env(*, scenario: ScenarioConfig):
    return make_env(config=scenario)


def _date_prefix(now: datetime | None = None) -> str:
    return report_utils.date_prefix(now)


def _build_output_paths(*, output_dir: Path, now: datetime | None = None) -> LLMJudgeOutputs:
    prefix = _date_prefix(now)
    return LLMJudgeOutputs(
        steps_csv=Path(output_dir) / f"{prefix}-llm-judge-steps.csv",
        summary_csv=Path(output_dir) / f"{prefix}-llm-judge-summary.csv",
        calls_jsonl=Path(output_dir) / f"{prefix}-llm-judge-calls.jsonl",
    )


def _dedupe_actions(candidate_actions: dict[str, int]) -> list[tuple[str, tuple[str, ...], int]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for heuristic_name in HEURISTIC_ORDER:
        grouped[int(candidate_actions[heuristic_name])].append(heuristic_name)
    return [
        (heuristic_names[0], tuple(heuristic_names), int(action))
        for action, heuristic_names in grouped.items()
    ]


def _build_candidates(
    *,
    context,
    candidate_actions: dict[str, int],
) -> tuple[JudgeCandidate, ...]:
    candidates: list[JudgeCandidate] = []
    for canonical_name, proposed_by, action in _dedupe_actions(candidate_actions):
        candidates.append(
            build_judge_candidate(
                context=context,
                heuristic_name=canonical_name,
                action=action,
                proposed_by=proposed_by,
            )
        )
    return tuple(candidates)


def _select_candidate_actions(*, context, rng: np.random.Generator) -> dict[str, int]:
    return {
        "first_fit": int(select_first_fit_runtime_action(context)),
        "load_balancing": int(select_load_balancing_runtime_action(context)),
        "random": int(select_random_runtime_action(context, rng=rng)),
    }


def _serialize_decoded_path_nodes(candidate: JudgeCandidate) -> str:
    if candidate.decoded_action is None:
        return ""
    return "->".join(candidate.decoded_action.path_node_names)


def _serialize_decisive_signals(verdict: JudgeVerdict | None) -> str:
    if verdict is None or not verdict.decisive_signals:
        return ""
    return " | ".join(
        f"{signal.factor}:{signal.supports}({signal.importance})"
        for signal in verdict.decisive_signals
    )


def _resolve_prompt_and_model_io(
    *,
    judge_client: HeuristicJudge,
    payload: JudgeDecisionPayload,
    verdict: JudgeVerdict | None,
) -> tuple[JudgePromptRecord, dict[str, object] | None, dict[str, object] | None]:
    prompt_record = build_ollama_prompt_record(payload)
    raw_model_response: dict[str, object] | None = None
    parsed_response: dict[str, object] | None = None if verdict is None else verdict.to_mapping()
    consume_trace = getattr(judge_client, "consume_last_trace", None)
    if callable(consume_trace):
        trace = consume_trace()
        if trace is not None:
            prompt_record = trace.prompt
            raw_model_response = None if trace.raw_model_response is None else dict(trace.raw_model_response)
            parsed_response = None if trace.parsed_response is None else dict(trace.parsed_response)
    return prompt_record, raw_model_response, parsed_response


def _build_step_row(
    *,
    date_label: str,
    prompt_version: str,
    topology_name: str,
    episode_index: int,
    step_index: int,
    payload: JudgeDecisionPayload,
    verdict: JudgeVerdict | None,
    agrees_with_baseline: bool,
    fallback_reason: str,
    judge_error_message: str,
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
        "load_regime": payload.global_regimes.load_regime,
        "qot_pressure_regime": payload.global_regimes.qot_pressure_regime,
        "services_processed": int(payload.operational_state.services_processed),
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
        "decisive_signals_summary": _serialize_decisive_signals(verdict),
        "agrees_with_baseline": bool(agrees_with_baseline),
        "fallback_reason": fallback_reason,
        "judge_error_message": judge_error_message,
        "post_status": str(post_info.get("status", "unknown")),
        "post_accepted": bool(post_info.get("accepted", False)),
        "post_reward": float(post_info.get("reward", 0.0)),
        "episode_service_blocking_rate": float(post_info.get("episode_service_blocking_rate", 0.0)),
        "episode_bit_rate_blocking_rate": float(post_info.get("episode_bit_rate_blocking_rate", 0.0)),
        "episode_disrupted_services_rate": float(post_info.get("episode_disrupted_services", 0.0)),
    }


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
    summary_rows.append(
        {
            "date": date_label,
            "scope": "run",
            "episode_index": -1,
            "steps": total_steps,
            "llm_calls": llm_calls,
            "fallback_count": fallback_count,
            "baseline_agreement_rate": float(
                sum(1 for row in step_rows if bool(row["agrees_with_baseline"])) / total_steps
            ),
            "first_fit_choices": sum(1 for row in step_rows if row["winner_heuristic"] == "first_fit"),
            "load_balancing_choices": sum(1 for row in step_rows if row["winner_heuristic"] == "load_balancing"),
            "random_choices": sum(1 for row in step_rows if row["winner_heuristic"] == "random"),
            "final_blocking_rate": float(step_rows[-1]["episode_service_blocking_rate"]),
            "final_disrupted_rate": float(step_rows[-1]["episode_disrupted_services_rate"]),
        }
    )
    return summary_rows


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
    judge_client = judge if judge is not None else OllamaHeuristicJudge.from_env(env_path=experiment.env_path)

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
        while True:
            context = env.heuristic_context()
            candidate_actions = _select_candidate_actions(context=context, rng=rng)
            candidates = _build_candidates(context=context, candidate_actions=candidate_actions)
            scored_candidates, baseline_winner = score_candidates(candidates)
            operational_state = build_operational_state(context=context, info=current_info)
            payload = build_judge_payload(
                topology_profile=topology_profile,
                operational_state=operational_state,
                global_regimes=build_global_regimes(operational_state),
                candidates=scored_candidates,
            )

            fallback_reason = "none"
            judge_error_message = ""
            judge_called = False
            verdict: JudgeVerdict | None = None
            winner_name = baseline_winner
            if len(scored_candidates) == 1:
                fallback_reason = "single_candidate"
            elif all(candidate.is_reject for candidate in scored_candidates):
                fallback_reason = "all_reject"
            else:
                judge_called = True
                episode_llm_calls += 1
                try:
                    verdict = judge_client.judge(payload)
                    winner_name = verdict.winner
                except Exception as exc:
                    judge_error_message = _format_exception_message(exc)
                    fallback_reason = f"judge_error:{type(exc).__name__}:{judge_error_message}"
                    winner_name = baseline_winner

            if fallback_reason != "none":
                episode_fallback_count += 1

            candidate_by_name = {candidate.heuristic_name: candidate for candidate in scored_candidates}
            winner_candidate = candidate_by_name[winner_name]
            agrees_with_baseline = winner_candidate.heuristic_name == baseline_winner
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
                    agrees_with_baseline=agrees_with_baseline,
                    fallback_reason=fallback_reason,
                    judge_error_message=judge_error_message,
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
                candidates=scored_candidates,
                baseline_winner=baseline_winner,
                chosen_action=int(winner_candidate.raw_action),
                chosen_heuristic=winner_candidate.heuristic_name,
            )
            _append_jsonl(outputs.calls_jsonl, audit_record.to_mapping())

            episode_step_count += 1
            if terminated or truncated:
                break

        episode_rows = [row for row in step_rows if int(row["episode_index"]) == episode_index]
        episode_summaries.append(
            {
                "date": date_label,
                "scope": "episode",
                "episode_index": int(episode_index),
                "steps": episode_step_count,
                "llm_calls": episode_llm_calls,
                "fallback_count": episode_fallback_count,
                "baseline_agreement_rate": float(
                    sum(1 for row in episode_rows if bool(row["agrees_with_baseline"])) / max(1, len(episode_rows))
                ),
                "first_fit_choices": sum(1 for row in episode_rows if row["winner_heuristic"] == "first_fit"),
                "load_balancing_choices": sum(
                    1 for row in episode_rows if row["winner_heuristic"] == "load_balancing"
                ),
                "random_choices": sum(1 for row in episode_rows if row["winner_heuristic"] == "random"),
                "final_blocking_rate": float(episode_rows[-1]["episode_service_blocking_rate"]),
                "final_disrupted_rate": float(episode_rows[-1]["episode_disrupted_services_rate"]),
            }
        )
        env.close()

    step_fieldnames = list(step_rows[0].keys()) if step_rows else []
    summary_rows = _build_summary_rows(
        date_label=date_label,
        step_rows=step_rows,
        episode_summaries=episode_summaries,
    )
    summary_fieldnames = list(summary_rows[0].keys()) if summary_rows else []
    report_utils.write_csv_rows(path=outputs.steps_csv, fieldnames=step_fieldnames, rows=step_rows)
    report_utils.write_csv_rows(path=outputs.summary_csv, fieldnames=summary_fieldnames, rows=summary_rows)
    return outputs


def main() -> None:
    outputs = run_experiment(experiment=LLMJudgeExperiment())
    print(asdict(outputs))


if __name__ == "__main__":
    main()

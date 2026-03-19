from __future__ import annotations

from pathlib import Path

import pytest

from optical_networking_gym_v2 import make_env, set_topology_dir
from optical_networking_gym_v2.heuristics import select_first_fit_runtime_action
from optical_networking_gym_v2.judge.heuristic_judge import (
    JudgePromptRecord,
    build_global_regimes,
    build_judge_audit_record,
    build_judge_candidate,
    build_judge_payload,
    build_operational_state,
    build_topology_profile,
    score_candidates,
)
from optical_networking_gym_v2.judge.ollama import _build_verdict_from_response_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = PROJECT_ROOT.parent / "examples" / "topologies"


def _build_env():
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        seed=7,
        bit_rates=(40,),
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        episode_length=6,
        modulations_to_consider=2,
        k_paths=2,
    )
    observation, info = env.reset(seed=7)
    return env, observation, info


def test_build_judge_candidate_serializes_route_summary_fields() -> None:
    env, _observation, info = _build_env()
    context = env.heuristic_context()
    action = select_first_fit_runtime_action(context)

    candidate = build_judge_candidate(
        context=context,
        heuristic_name="first_fit",
        action=action,
    )
    decision_candidate = candidate.to_decision_candidate()

    assert candidate.raw_action == action
    assert candidate.decoded_action is not None
    assert decision_candidate.route_summary is not None
    assert decision_candidate.route_summary.endpoint_pair
    assert decision_candidate.route_summary.modulation_name in {"QPSK", "16QAM"}
    assert decision_candidate.route_summary.slot_span > 0
    assert decision_candidate.route_summary.path_hops >= 0
    assert candidate.metrics.required_slots > 0
    assert "mask" in info


def test_build_judge_candidate_uses_null_decoded_action_for_reject() -> None:
    env, _observation, _info = _build_env()
    context = env.heuristic_context()

    candidate = build_judge_candidate(
        context=context,
        heuristic_name="reject",
        action=context.reject_action,
    )

    assert candidate.raw_action == context.reject_action
    assert candidate.decoded_action is None
    assert candidate.to_decision_candidate().route_summary is None


def test_build_judge_payload_excludes_audit_only_fields() -> None:
    env, _observation, info = _build_env()
    context = env.heuristic_context()
    action = select_first_fit_runtime_action(context)
    candidate = build_judge_candidate(
        context=context,
        heuristic_name="first_fit",
        action=action,
    )
    scored_candidates, _baseline_winner = score_candidates((candidate,))
    operational_state = build_operational_state(context=context, info=info)
    payload = build_judge_payload(
        topology_profile=build_topology_profile(context.topology),
        operational_state=operational_state,
        global_regimes=build_global_regimes(operational_state),
        candidates=scored_candidates,
    )

    payload_mapping = payload.to_prompt_mapping()
    candidate_mapping = payload_mapping["candidates"][0]

    assert "topology_context" in payload_mapping
    assert "operational_state" in payload_mapping
    assert "global_regimes" in payload_mapping
    assert "decision_rules" in payload_mapping
    assert "candidates" in payload_mapping
    assert "prompt_version" not in payload_mapping
    assert "baseline_winner" not in payload_mapping
    assert "seed" not in payload_mapping
    assert "episode_index" not in payload_mapping
    assert "step_index" not in payload_mapping
    assert "topology_id" not in payload_mapping
    assert "friendly_name" not in payload_mapping["topology_context"]
    assert "decision_hint" not in payload_mapping["topology_context"]
    assert "raw_action" not in candidate_mapping
    assert "proposed_by" not in candidate_mapping
    assert "decoded_action" not in candidate_mapping
    assert "baseline_scores" not in candidate_mapping
    assert "nli_share" not in candidate_mapping["decision_metrics"]
    assert candidate_mapping["route_summary"]["endpoint_pair"]
    assert candidate_mapping["route_summary"]["modulation_name"]
    assert candidate_mapping["route_summary"]["slot_span"] > 0


def test_build_judge_audit_record_preserves_full_trace() -> None:
    env, _observation, info = _build_env()
    context = env.heuristic_context()
    action = select_first_fit_runtime_action(context)
    candidate = build_judge_candidate(
        context=context,
        heuristic_name="first_fit",
        action=action,
    )
    scored_candidates, baseline_winner = score_candidates((candidate,))
    operational_state = build_operational_state(context=context, info=info)
    payload = build_judge_payload(
        topology_profile=build_topology_profile(context.topology),
        operational_state=operational_state,
        global_regimes=build_global_regimes(operational_state),
        candidates=scored_candidates,
    )
    audit_record = build_judge_audit_record(
        date="19-03",
        prompt_version="v2",
        seed=7,
        episode_index=0,
        step_index=0,
        topology_id="ring_4",
        decision_payload=payload,
        prompt=JudgePromptRecord(system_prompt="sys", user_prompt="usr"),
        raw_model_response={"content": "{\"winner\":\"first_fit\"}"},
        parsed_response={"winner": "first_fit"},
        fallback_reason="none",
        judge_error_message="",
        candidates=scored_candidates,
        baseline_winner=baseline_winner,
        chosen_action=action,
        chosen_heuristic="first_fit",
    )

    mapping = audit_record.to_mapping()

    assert mapping["audit"]["prompt_version"] == "v2"
    assert mapping["audit"]["seed"] == 7
    assert mapping["audit"]["episode_index"] == 0
    assert mapping["audit"]["step_index"] == 0
    assert mapping["audit"]["topology_id"] == "ring_4"
    assert mapping["audit"]["chosen_action"] == action
    assert mapping["audit"]["baseline_winner"] == baseline_winner
    assert mapping["audit"]["candidate_audit"][0]["raw_action"] == action
    assert mapping["audit"]["candidate_audit"][0]["decoded_action"]["path_node_names"]
    assert mapping["prompt"]["system_prompt"] == "sys"
    assert mapping["prompt"]["user_prompt"] == "usr"
    assert mapping["model_io"]["raw_model_response"]["content"] == "{\"winner\":\"first_fit\"}"
    assert mapping["model_io"]["parsed_response"]["winner"] == "first_fit"


def test_build_verdict_from_response_mapping_validates_candidate_names() -> None:
    response = {
        "winner": "first_fit",
        "confidence": 0.87,
        "ranking": ["first_fit", "random"],
        "reason": "first_fit preserves spectrum continuity better.",
        "used_tie_break": False,
        "decisive_signals": [
            {
                "factor": "fragmentation",
                "supports": "first_fit",
                "evidence": "largest fragmentation damage is lower.",
                "importance": "critical",
            },
            {
                "factor": "load balancing",
                "supports": "first_fit would preserve flatter utilization.",
                "evidence": "link utilization stays flatter.",
                "importance": "high (under high fragmentation risk)",
            }
        ],
    }

    verdict = _build_verdict_from_response_mapping(
        response,
        candidate_names={"first_fit", "random"},
    )

    assert verdict.winner == "first_fit"
    assert verdict.decisive_signals[0].factor == "fragmentation"
    assert verdict.decisive_signals[0].importance == "high"
    assert verdict.decisive_signals[1].factor == "load_balance"
    assert verdict.decisive_signals[1].importance == "high"

    with pytest.raises(ValueError):
        _build_verdict_from_response_mapping(
            {**response, "winner": "unknown"},
            candidate_names={"first_fit", "random"},
        )

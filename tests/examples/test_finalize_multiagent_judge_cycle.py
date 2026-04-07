from __future__ import annotations

import json
from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "examples" / "llm" / "finalize_multiagent_judge_cycle.py"


def _load_module() -> dict[str, object]:
    return runpy.run_path(str(SCRIPT_PATH))


def test_finalize_multiagent_cycle_records_review_outputs_and_convergence(tmp_path: Path) -> None:
    module = _load_module()
    finalize_cycle = module["finalize_multiagent_cycle"]

    state_path = tmp_path / "state.json"
    tracker_path = tmp_path / "tracker.md"
    state_path.write_text(
        json.dumps(
            {
                "baseline": {
                    "scenario_profile": "legacy_benchmark",
                    "topology_id": "nobel-eu",
                    "seed": 10,
                    "episode_count": 1,
                    "episode_length": 1000,
                    "heuristics": ["first_fit"],
                    "loads": [],
                },
                "cycles": [
                    {
                        "cycle_id": "1",
                        "target_load": 400.0,
                        "run_dir": "results/cycle1",
                        "hypothesis": "test",
                        "analyst_findings": "pending_agent_review",
                        "reviewer_recommendation": "pending_agent_review",
                        "convergence_status": "pending_agent_review",
                        "applied_change": "none",
                        "benchmark_result": "final_blocking_rate=0.055",
                        "decision_next_step": "pending",
                        "artifacts": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    analyst_path = tmp_path / "analyst_review.json"
    analyst_path.write_text(
        json.dumps(
            {
                "primary_failure_mode": "late_same_slot_route_bias",
                "onset_window": "step 400-600",
                "dominant_error_patterns": ["lowest_fragmentation over-selected"],
                "evidence_case_ids": ["ep0-step914", "ep0-step943"],
                "why_this_is_real": "blocking rises after these mismatches",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    reviewer_path = tmp_path / "reviewer_review.json"
    reviewer_path.write_text(
        json.dumps(
            {
                "target_failure_mode": "late_same_slot_route_bias",
                "allowed_change_layer": "prompt",
                "recommended_single_change": "shorten the route-advantage rule for high-risk late steps",
                "why_this_layer": "the cases show interpretation drift",
                "why_not_other_layers": {
                    "prompt": "",
                    "payload": "signals already expose the conflict",
                    "shortlist": "plausible options are present",
                },
                "evidence_case_ids": ["ep0-step914", "ep0-step943"],
                "forbidden_side_effects": ["do not hide plausible candidates"],
                "expected_effect": "reduce late lowest_fragmentation bias",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output = finalize_cycle(
        cycle_id="1",
        analyst_review_path=analyst_path,
        reviewer_review_path=reviewer_path,
        state_path=state_path,
        tracker_path=tracker_path,
    )

    updated_state = json.loads(state_path.read_text(encoding="utf-8"))
    cycle = updated_state["cycles"][0]

    assert output["convergence_status"] == "converged"
    assert cycle["convergence_status"] == "converged"
    assert cycle["applied_change"] == "pending_implementation"
    assert cycle["artifacts"]["analyst_review"] == str(analyst_path)
    assert cycle["artifacts"]["reviewer_review"] == str(reviewer_path)

    tracker_text = tracker_path.read_text(encoding="utf-8")
    assert "late_same_slot_route_bias" in tracker_text
    assert "pending_implementation" in tracker_text


def test_finalize_multiagent_cycle_accepts_semantic_convergence(tmp_path: Path) -> None:
    module = _load_module()
    finalize_cycle = module["finalize_multiagent_cycle"]

    state_path = tmp_path / "state.json"
    tracker_path = tmp_path / "tracker.md"
    state_path.write_text(
        json.dumps(
            {
                "baseline": {
                    "scenario_profile": "legacy_benchmark",
                    "topology_id": "nobel-eu",
                    "seed": 10,
                    "episode_count": 1,
                    "episode_length": 1000,
                    "heuristics": ["first_fit"],
                    "loads": [],
                },
                "cycles": [
                    {
                        "cycle_id": "1",
                        "target_load": 400.0,
                        "run_dir": "results/cycle1",
                        "hypothesis": "test",
                        "analyst_findings": "pending_agent_review",
                        "reviewer_recommendation": "pending_agent_review",
                        "convergence_status": "pending_agent_review",
                        "applied_change": "none",
                        "benchmark_result": "final_blocking_rate=0.055",
                        "decision_next_step": "pending",
                        "artifacts": {},
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    analyst_path = tmp_path / "analyst_review.json"
    analyst_path.write_text(
        json.dumps(
            {
                "primary_failure_mode": (
                    "Late high-risk same-slot basis/payload incoherence that drifts the judge "
                    "toward the wrong heuristic family after blocking first appears."
                ),
                "onset_window": {"start_step": 367, "end_step": 500},
                "dominant_error_patterns": ["basis mismatch under high risk"],
                "evidence_case_ids": ["ep0-step867", "ep0-step943", "ep0-step970"],
                "why_this_is_real": "blocking rises after these mismatches",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    reviewer_path = tmp_path / "reviewer_review.json"
    reviewer_path.write_text(
        json.dumps(
            {
                "target_failure_mode": (
                    "late_same_slot_route_basis_mismatch_on_anonymous_same_slot_candidate_under_high_future_risk"
                ),
                "allowed_change_layer": "prompt",
                "recommended_single_change": "add one explicit prompt rule for the late high-risk regime",
                "why_this_layer": "the payload already shows the contrast",
                "why_not_other_layers": {
                    "payload": "signals already exist",
                    "shortlist": "plausible options are present",
                },
                "evidence_case_ids": ["ep0-step867", "ep0-step943", "ep0-step970"],
                "forbidden_side_effects": ["do not hide plausible candidates"],
                "expected_effect": "reduce anonymous same-slot route-basis mismatches",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    output = finalize_cycle(
        cycle_id="1",
        analyst_review_path=analyst_path,
        reviewer_review_path=reviewer_path,
        state_path=state_path,
        tracker_path=tracker_path,
    )

    updated_state = json.loads(state_path.read_text(encoding="utf-8"))
    cycle = updated_state["cycles"][0]

    assert output["convergence_status"] == "converged"
    assert cycle["convergence_status"] == "converged"
    assert cycle["applied_change"] == "pending_implementation"

from __future__ import annotations

from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "examples" / "llm" / "analyze_llm_judge_future_regret.py"


def _load_module() -> dict[str, object]:
    return runpy.run_path(str(SCRIPT_PATH))


def test_select_future_best_candidate_prioritizes_longer_horizon_blocking_then_ties() -> None:
    module = _load_module()
    select_future_best_candidate = module["select_future_best_candidate"]

    outcomes = (
        {
            "candidate_id": "C1",
            "blocking_delta_25": 1,
            "blocking_delta_50": 2,
            "blocking_delta_100": 3,
            "accepted_delta_100": 96,
            "required_slots": 2,
            "raw_action": 10,
        },
        {
            "candidate_id": "C2",
            "blocking_delta_25": 2,
            "blocking_delta_50": 1,
            "blocking_delta_100": 2,
            "accepted_delta_100": 95,
            "required_slots": 3,
            "raw_action": 11,
        },
        {
            "candidate_id": "C3",
            "blocking_delta_25": 1,
            "blocking_delta_50": 1,
            "blocking_delta_100": 2,
            "accepted_delta_100": 94,
            "required_slots": 4,
            "raw_action": 12,
        },
    )

    assert select_future_best_candidate(outcomes, horizons=(25, 50, 100)) == "C3"


def test_select_future_best_candidate_uses_accepted_and_slots_as_tie_breaks() -> None:
    module = _load_module()
    select_future_best_candidate = module["select_future_best_candidate"]

    outcomes = (
        {
            "candidate_id": "A1",
            "blocking_delta_25": 1,
            "blocking_delta_50": 1,
            "blocking_delta_100": 1,
            "accepted_delta_100": 97,
            "required_slots": 3,
            "raw_action": 21,
        },
        {
            "candidate_id": "B2",
            "blocking_delta_25": 1,
            "blocking_delta_50": 1,
            "blocking_delta_100": 1,
            "accepted_delta_100": 98,
            "required_slots": 4,
            "raw_action": 22,
        },
    )

    assert select_future_best_candidate(outcomes, horizons=(25, 50, 100)) == "B2"


def test_resolve_prompt_candidates_uses_route_to_disambiguate_same_metrics() -> None:
    module = _load_module()
    resolve_prompt_candidates = module["_resolve_prompt_candidates"]

    record = {
        "audit": {
            "episode_index": 0,
            "step_index": 5,
            "candidate_audit": [
                {
                    "heuristic_name": "first_fit",
                    "proposed_by": ["first_fit"],
                    "raw_action": 10,
                    "is_reject": False,
                    "decoded_action": {
                        "path_index": 0,
                        "path_rank_k": 0,
                        "path_node_names": ["A", "B"],
                        "path_hops": 1,
                        "path_length_km": 100.0,
                        "source_name": "A",
                        "destination_name": "B",
                        "modulation_index": 0,
                        "modulation_name": "QPSK",
                        "modulation_spectral_efficiency": 2,
                        "initial_slot": 10,
                        "required_slots": 1,
                        "slot_end_exclusive": 11,
                    },
                    "metrics": {
                        "required_slots": 1,
                        "route_pressure_score": 0.2,
                        "local_damage_score": 0.1,
                        "qot_margin_clipped_db": 1.0,
                    },
                },
                {
                    "heuristic_name": "ksp_best_mod_last_fit",
                    "proposed_by": ["ksp_best_mod_last_fit"],
                    "raw_action": 11,
                    "is_reject": False,
                    "decoded_action": {
                        "path_index": 0,
                        "path_rank_k": 0,
                        "path_node_names": ["A", "B"],
                        "path_hops": 1,
                        "path_length_km": 100.0,
                        "source_name": "A",
                        "destination_name": "B",
                        "modulation_index": 0,
                        "modulation_name": "QPSK",
                        "modulation_spectral_efficiency": 2,
                        "initial_slot": 20,
                        "required_slots": 1,
                        "slot_end_exclusive": 21,
                    },
                    "metrics": {
                        "required_slots": 1,
                        "route_pressure_score": 0.2,
                        "local_damage_score": 0.1,
                        "qot_margin_clipped_db": 1.0,
                    },
                },
            ],
        },
        "decision_payload": {
            "candidates": [
                {
                    "candidate_id": "A1",
                    "is_reject": False,
                    "route": {
                        "path_index": 0,
                        "path_hops": 1,
                        "path_length_km": 100.0,
                        "modulation_name": "QPSK",
                        "initial_slot": 20,
                        "required_slots": 1,
                    },
                    "metrics": {
                        "required_slots": 1,
                        "route_pressure_score": 0.2,
                        "local_damage_score": 0.1,
                        "qot_margin_clipped_db": 1.0,
                    },
                },
                {
                    "candidate_id": "B2",
                    "is_reject": False,
                    "route": {
                        "path_index": 0,
                        "path_hops": 1,
                        "path_length_km": 100.0,
                        "modulation_name": "QPSK",
                        "initial_slot": 10,
                        "required_slots": 1,
                    },
                    "metrics": {
                        "required_slots": 1,
                        "route_pressure_score": 0.2,
                        "local_damage_score": 0.1,
                        "qot_margin_clipped_db": 1.0,
                    },
                },
            ]
        },
    }

    resolved = resolve_prompt_candidates(record)

    assert [candidate["heuristic_name"] for candidate in resolved] == [
        "ksp_best_mod_last_fit",
        "first_fit",
    ]

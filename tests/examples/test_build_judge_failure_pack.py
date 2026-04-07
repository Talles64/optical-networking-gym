from __future__ import annotations

import csv
import json
from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "examples" / "llm" / "build_judge_failure_pack.py"


def _load_module() -> dict[str, object]:
    return runpy.run_path(str(SCRIPT_PATH))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_build_failure_pack_collects_real_evidence_without_promoting_future_regret_to_answer_key(
    tmp_path: Path,
) -> None:
    module = _load_module()
    config_cls = module["JudgeFailurePackConfig"]
    build_failure_pack = module["build_failure_pack"]

    run_dir = tmp_path / "judge_legacy_benchmark_load400_ep1_cycleX"
    run_dir.mkdir()

    summary_path = run_dir / "01-04-00h00-llm-judge-summary.csv"
    _write_csv(
        summary_path,
        [
            "scope",
            "steps",
            "llm_calls",
            "fallback_count",
            "reference_agreement_rate",
            "hidden_balanced_agreement_rate",
            "final_blocking_rate",
            "mean_episode_service_blocking_rate",
        ],
        [
            {
                "scope": "run",
                "steps": 1000,
                "llm_calls": 900,
                "fallback_count": 0,
                "reference_agreement_rate": 0.65,
                "hidden_balanced_agreement_rate": 0.80,
                "final_blocking_rate": 0.02,
                "mean_episode_service_blocking_rate": 0.02,
            }
        ],
    )

    steps_path = run_dir / "01-04-00h00-llm-judge-steps.csv"
    _write_csv(
        steps_path,
        [
            "episode_index",
            "step_index",
            "controller_decision_source",
            "winner_heuristic",
            "reference_winner",
            "winner_candidate_id",
            "winner_decision_basis",
            "basis_vs_payload_mismatch",
            "fallback_reason",
            "episode_service_blocking_rate",
            "decisive_signals_summary",
            "semantic_warning_flags",
        ],
        [
            {
                "episode_index": 0,
                "step_index": 10,
                "controller_decision_source": "llm",
                "winner_heuristic": "lowest_fragmentation",
                "reference_winner": "first_fit",
                "winner_candidate_id": "C2",
                "winner_decision_basis": "balanced_tie_break",
                "basis_vs_payload_mismatch": "True",
                "fallback_reason": "",
                "episode_service_blocking_rate": 0.00,
                "decisive_signals_summary": "near_tie",
                "semantic_warning_flags": "",
            },
            {
                "episode_index": 0,
                "step_index": 20,
                "controller_decision_source": "llm",
                "winner_heuristic": "load_balancing",
                "reference_winner": "ksp_best_mod_last_fit",
                "winner_candidate_id": "C1",
                "winner_decision_basis": "route_pressure_over_local_damage",
                "basis_vs_payload_mismatch": "False",
                "fallback_reason": "",
                "episode_service_blocking_rate": 0.01,
                "decisive_signals_summary": "route_gain",
                "semantic_warning_flags": "",
            },
        ],
    )

    calls_path = run_dir / "01-04-00h00-llm-judge-calls.jsonl"
    call_records = [
        {
            "audit": {
                "episode_index": 0,
                "step_index": 10,
            },
            "decision_payload": {
                "prompt_context": {"same_slot_local_support_band": "none"},
                "candidates": [
                    {
                        "candidate_id": "C1",
                        "candidate_roles": ["same_slot_route_leader"],
                        "is_reject": False,
                        "route": {"path_index": 0, "initial_slot": 10, "required_slots": 1},
                        "metrics": {
                            "required_slots": 1,
                            "route_pressure_score": 0.20,
                            "local_damage_score": 0.10,
                            "path_common_free_ratio": 0.55,
                            "qot_margin_clipped_db": 1.2,
                            "future_risk_band": "high",
                            "qot_safe_now": True,
                        },
                    },
                    {
                        "candidate_id": "C2",
                        "candidate_roles": ["same_slot_preservation_leader"],
                        "is_reject": False,
                        "route": {"path_index": 1, "initial_slot": 20, "required_slots": 2},
                        "metrics": {
                            "required_slots": 2,
                            "route_pressure_score": 0.18,
                            "local_damage_score": 0.03,
                            "path_common_free_ratio": 0.61,
                            "qot_margin_clipped_db": 1.6,
                            "future_risk_band": "high",
                            "qot_safe_now": True,
                        },
                    },
                ],
            },
        },
        {
            "audit": {
                "episode_index": 0,
                "step_index": 20,
            },
            "decision_payload": {
                "prompt_context": {"same_slot_route_common_free_alignment": "aligned"},
                "candidates": [
                    {
                        "candidate_id": "C1",
                        "candidate_roles": ["same_slot_route_leader"],
                        "is_reject": False,
                        "route": {"path_index": 0, "initial_slot": 11, "required_slots": 1},
                        "metrics": {
                            "required_slots": 1,
                            "route_pressure_score": 0.14,
                            "local_damage_score": 0.06,
                            "path_common_free_ratio": 0.63,
                            "qot_margin_clipped_db": 1.0,
                            "future_risk_band": "moderate",
                            "qot_safe_now": True,
                        },
                    }
                ],
            },
        },
    ]
    with calls_path.open("w", encoding="utf-8") as handle:
        for record in call_records:
            handle.write(json.dumps(record) + "\n")

    method_rules_path = tmp_path / "ONLINE_JUDGE_METHOD_RULES.md"
    method_rules_path.write_text(
        "\n".join(
            [
                "# Online Judge Method Rules",
                "## Permanent Rules",
                "- Judge LLM is the real decision maker.",
                "- Scorer and shortlist cannot become answer key.",
                "## Agent Contract",
                "- Analyst does not propose solutions.",
                "- Reviewer proposes one local change only.",
                "## Future Regret Rule",
                "- Future regret is offline evidence only.",
            ]
        ),
        encoding="utf-8",
    )

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "scenario_profile": "legacy_benchmark",
                "topology_id": "nobel-eu",
                "seed": 10,
                "episode_count": 1,
                "episode_length": 1000,
                "loads": [
                    {
                        "load": 400.0,
                        "best_heuristic": "first_fit",
                        "best_service_blocking_rate_mean": 0.015,
                        "environment": {"k_paths": 5},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    future_regret_path = tmp_path / "future_regret.csv"
    _write_csv(
        future_regret_path,
        [
            "episode_index",
            "step_index",
            "judge_matches_future_best",
            "judge_blocking_delta_25",
            "future_best_blocking_delta_25",
            "judge_blocking_delta_50",
            "future_best_blocking_delta_50",
            "judge_blocking_delta_100",
            "future_best_blocking_delta_100",
        ],
        [
            {
                "episode_index": 0,
                "step_index": 10,
                "judge_matches_future_best": "False",
                "judge_blocking_delta_25": 2,
                "future_best_blocking_delta_25": 0,
                "judge_blocking_delta_50": 3,
                "future_best_blocking_delta_50": 1,
                "judge_blocking_delta_100": 4,
                "future_best_blocking_delta_100": 1,
            }
        ],
    )

    failure_pack = build_failure_pack(
        config=config_cls(
            run_dir=run_dir,
            method_rules_path=method_rules_path,
            baseline_path=baseline_path,
            future_regret_detail_csv=future_regret_path,
            representative_case_limit=4,
        )
    )

    assert failure_pack["acceptance_target"]["best_heuristic"] == "first_fit"
    assert failure_pack["scenario_truth"]["load"] == 400.0
    assert {
        (item["chosen_heuristic"], item["reference_winner"])
        for item in failure_pack["top_mismatch_pairs"]
    } == {
        ("lowest_fragmentation", "first_fit"),
        ("load_balancing", "ksp_best_mod_last_fit"),
    }
    assert failure_pack["representative_cases"]
    case = failure_pack["representative_cases"][0]
    assert case["case_id"] == "ep0-step10"
    assert "offline_future_regret_evidence" in case
    assert "future_best_candidate_id" not in json.dumps(case)
    assert "Scorer and shortlist cannot become answer key." in failure_pack["method_rules_digest"]["sections"]["Permanent Rules"]


def test_build_failure_pack_can_filter_analysis_scope_to_a_narrow_failure_family(tmp_path: Path) -> None:
    module = _load_module()
    config_cls = module["JudgeFailurePackConfig"]
    build_failure_pack = module["build_failure_pack"]

    run_dir = tmp_path / "judge_legacy_benchmark_load400_ep1_cycleY"
    run_dir.mkdir()

    _write_csv(
        run_dir / "01-04-00h00-llm-judge-summary.csv",
        [
            "scope",
            "steps",
            "llm_calls",
            "fallback_count",
            "reference_agreement_rate",
            "hidden_balanced_agreement_rate",
            "final_blocking_rate",
            "mean_episode_service_blocking_rate",
        ],
        [
            {
                "scope": "run",
                "steps": 1000,
                "llm_calls": 900,
                "fallback_count": 0,
                "reference_agreement_rate": 0.65,
                "hidden_balanced_agreement_rate": 0.80,
                "final_blocking_rate": 0.02,
                "mean_episode_service_blocking_rate": 0.02,
            }
        ],
    )

    _write_csv(
        run_dir / "01-04-00h00-llm-judge-steps.csv",
        [
            "episode_index",
            "step_index",
            "controller_decision_source",
            "winner_heuristic",
            "reference_winner",
            "winner_candidate_id",
            "winner_decision_basis",
            "basis_vs_payload_mismatch",
            "fallback_reason",
            "episode_service_blocking_rate",
            "decisive_signals_summary",
            "semantic_warning_flags",
        ],
        [
            {
                "episode_index": 0,
                "step_index": 100,
                "controller_decision_source": "llm",
                "winner_heuristic": "lowest_fragmentation",
                "reference_winner": "lowest_fragmentation",
                "winner_candidate_id": "A1",
                "winner_decision_basis": "same_slot_local_advantage",
                "basis_vs_payload_mismatch": "True",
                "fallback_reason": "",
                "episode_service_blocking_rate": 0.00,
                "decisive_signals_summary": "late_local",
                "semantic_warning_flags": "same_slot_local_basis_mismatch",
            },
            {
                "episode_index": 0,
                "step_index": 900,
                "controller_decision_source": "llm",
                "winner_heuristic": "first_fit",
                "reference_winner": "lowest_fragmentation",
                "winner_candidate_id": "B2",
                "winner_decision_basis": "same_slot_route_advantage",
                "basis_vs_payload_mismatch": "True",
                "fallback_reason": "",
                "episode_service_blocking_rate": 0.05,
                "decisive_signals_summary": "late_route",
                "semantic_warning_flags": "same_slot_route_basis_mismatch",
            },
            {
                "episode_index": 0,
                "step_index": 910,
                "controller_decision_source": "llm",
                "winner_heuristic": "load_balancing",
                "reference_winner": "lowest_fragmentation",
                "winner_candidate_id": "C3",
                "winner_decision_basis": "balanced_tie_break",
                "basis_vs_payload_mismatch": "True",
                "fallback_reason": "",
                "episode_service_blocking_rate": 0.05,
                "decisive_signals_summary": "late_tie",
                "semantic_warning_flags": "same_slot_route_basis_mismatch",
            },
        ],
    )

    with (run_dir / "01-04-00h00-llm-judge-calls.jsonl").open("w", encoding="utf-8") as handle:
        for record in (
            {
                "audit": {"episode_index": 0, "step_index": 100},
                "decision_payload": {
                    "prompt_context": {"same_slot_local_support_band": "material"},
                    "candidates": [{"candidate_id": "A1", "candidate_roles": ["same_slot_candidate"], "is_reject": False, "metrics": {}}],
                },
            },
            {
                "audit": {"episode_index": 0, "step_index": 900},
                "decision_payload": {
                    "prompt_context": {"same_slot_local_support_band": "partial"},
                    "candidates": [{"candidate_id": "B2", "candidate_roles": ["same_slot_candidate"], "is_reject": False, "metrics": {}}],
                },
            },
            {
                "audit": {"episode_index": 0, "step_index": 910},
                "decision_payload": {
                    "prompt_context": {"same_slot_local_support_band": "partial"},
                    "candidates": [{"candidate_id": "C3", "candidate_roles": ["same_slot_candidate"], "is_reject": False, "metrics": {}}],
                },
            },
        ):
            handle.write(json.dumps(record) + "\n")

    method_rules_path = tmp_path / "ONLINE_JUDGE_METHOD_RULES.md"
    method_rules_path.write_text("# Rules\n## Permanent Rules\n- Keep judge primary.\n", encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"loads": [{"load": 400.0, "best_heuristic": "first_fit", "best_service_blocking_rate_mean": 0.02}]}), encoding="utf-8")

    failure_pack = build_failure_pack(
        config=config_cls(
            run_dir=run_dir,
            method_rules_path=method_rules_path,
            baseline_path=baseline_path,
            min_step_index=800,
            decision_basis="same_slot_route_advantage",
            require_judge_vs_reference_mismatch=True,
            require_basis_vs_payload_mismatch=True,
            representative_case_limit=5,
        )
    )

    assert failure_pack["analysis_scope"]["source_step_count"] == 3
    assert failure_pack["analysis_scope"]["analysis_step_count"] == 1
    assert failure_pack["analysis_scope"]["decision_basis"] == "same_slot_route_advantage"
    assert failure_pack["representative_cases"][0]["case_id"] == "ep0-step900"
    assert failure_pack["top_mismatch_pairs"] == [
        {"chosen_heuristic": "first_fit", "reference_winner": "lowest_fragmentation", "count": 1}
    ]

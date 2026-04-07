from __future__ import annotations

from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "examples" / "llm" / "run_multiagent_judge_cycle.py"


def _load_module() -> dict[str, object]:
    return runpy.run_path(str(SCRIPT_PATH))


def test_render_method_rules_and_tracker_include_live_baseline_and_cycle_summary() -> None:
    module = _load_module()
    config_cls = module["MultiAgentJudgeCycleConfig"]
    render_method_rules = module["_render_method_rules_markdown"]
    render_tracker = module["_render_tracker_markdown"]

    config = config_cls(cycle_id="1", target_load=400.0)
    official_snapshot = {
        "scenario_profile": "legacy_benchmark",
        "topology_id": "nobel-eu",
        "load": 400.0,
        "episode_count": 1,
        "episode_length": 1000,
        "seed": 10,
        "k_paths": 5,
        "launch_power_dbm": 0.0,
        "modulations_to_consider": 3,
        "bit_rates": [40, 100, 400],
        "num_spectrum_resources": 320,
        "mean_holding_time": 10800.0,
        "qot_constraint": "ASE+NLI",
        "measure_disruptions": False,
        "drop_on_disruption": False,
        "max_span_length_km": 80.0,
        "default_attenuation_db_per_km": 0.2,
        "default_noise_figure_db": 4.5,
        "frequency_start": 1.0,
        "frequency_slot_bandwidth": 12.5e9,
        "bandwidth": 4e12,
        "margin": 0.0,
        "scenario_id": "nobel-eu_legacy_benchmark_seed10",
        "modulations": [
            {
                "name": "BPSK",
                "maximum_length": 100000.0,
                "spectral_efficiency": 1,
                "minimum_osnr": 3.71,
                "inband_xt": -14.0,
            }
        ],
    }
    heuristics = [
        "first_fit",
        "load_balancing",
        "highest_snr_first_fit",
        "ksp_best_mod_last_fit",
        "lowest_fragmentation",
    ]

    method_rules_md = render_method_rules(
        config=config,
        official_snapshot=official_snapshot,
        heuristics=heuristics,
    )

    assert "Scorer, fallback, shortlist" in method_rules_md
    assert "`load=400`" in method_rules_md
    assert "`ksp_best_mod_last_fit`" in method_rules_md

    state = {
        "baseline": {
            "scenario_profile": "legacy_benchmark",
            "topology_id": "nobel-eu",
            "seed": 10,
            "episode_count": 1,
            "episode_length": 1000,
            "results_root": "results/heuristic_seed_baseline",
            "heuristics": heuristics,
            "loads": [
                {
                    "load": 400.0,
                    "heuristics": [
                        {"heuristic_name": "first_fit", "service_blocking_rate_mean": 0.022},
                        {"heuristic_name": "ksp_best_mod_last_fit", "service_blocking_rate_mean": 0.023},
                    ],
                    "best_heuristic": "first_fit",
                    "best_service_blocking_rate_mean": 0.022,
                    "artifacts": {
                        "run_dir": "results/heuristic_seed_baseline/load400",
                        "summary_csv": "results/heuristic_seed_baseline/load400-summary.csv",
                        "episodes_csv": "results/heuristic_seed_baseline/load400-episodes.csv",
                    },
                },
                {
                    "load": 350.0,
                    "heuristics": [
                        {"heuristic_name": "ksp_best_mod_last_fit", "service_blocking_rate_mean": 0.017},
                    ],
                    "best_heuristic": "ksp_best_mod_last_fit",
                    "best_service_blocking_rate_mean": 0.017,
                    "artifacts": {
                        "run_dir": "results/heuristic_seed_baseline/load350",
                        "summary_csv": "results/heuristic_seed_baseline/load350-summary.csv",
                        "episodes_csv": "results/heuristic_seed_baseline/load350-episodes.csv",
                    },
                },
            ],
        },
        "cycles": [
            {
                "cycle_id": "1",
                "target_load": 400.0,
                "run_dir": "results/judge_multiagent_load400_cycle1",
                "hypothesis": "bootstrap automated multi-agent cycle",
                "analyst_findings": "pending_agent_review",
                "reviewer_recommendation": "pending_agent_review",
                "convergence_status": "pending_agent_review",
                "applied_change": "none",
                "benchmark_result": "final_blocking_rate=0.055",
                "decision_next_step": "spawn Analyst + Reviewer",
                "artifacts": {
                    "steps_csv": "results/cycle1/steps.csv",
                    "summary_csv": "results/cycle1/summary.csv",
                    "calls_jsonl": "results/cycle1/calls.jsonl",
                    "failure_pack": "results/cycle1/failure_pack.json",
                    "analyst_brief": "results/cycle1/analyst_brief.json",
                    "reviewer_brief": "results/cycle1/reviewer_brief.json",
                    "analyst_review": "",
                    "reviewer_review": "",
                },
            }
        ],
    }

    tracker_md = render_tracker(state=state)

    assert "## Initial Baseline Snapshot" in tracker_md
    assert "| `first_fit` | `0.022` |" in tracker_md
    assert "Diretorio de artefatos do baseline" in tracker_md
    assert "CSV resumo do baseline" in tracker_md
    assert "## Cycle 1" in tracker_md
    assert "pending_agent_review" in tracker_md
    assert "`summary_csv=results/cycle1/summary.csv`" in tracker_md


def test_agent_review_status_requires_convergence_before_patch() -> None:
    module = _load_module()
    agent_review_status = module["_agent_review_status"]

    pending = agent_review_status(analyst_review=None, reviewer_review=None)
    assert pending["convergence_status"] == "pending_agent_review"

    diverged = agent_review_status(
        analyst_review={
            "primary_failure_mode": "late collapse due to preservation bias",
            "evidence_case_ids": ["ep0-step610"],
        },
        reviewer_review={
            "target_failure_mode": "early route pressure misread",
            "allowed_change_layer": "prompt",
            "recommended_single_change": "tighten rule",
        },
    )
    assert diverged["convergence_status"] == "no_convergence"

    converged = agent_review_status(
        analyst_review={
            "primary_failure_mode": "late collapse due to preservation bias",
            "evidence_case_ids": ["ep0-step610"],
        },
        reviewer_review={
            "target_failure_mode": "late collapse due to preservation bias",
            "allowed_change_layer": "payload",
            "recommended_single_change": "add one local signal",
        },
    )
    assert converged["convergence_status"] == "converged_ready_for_patch"


def test_agent_review_status_accepts_semantic_convergence_with_shared_evidence() -> None:
    module = _load_module()
    agent_review_status = module["_agent_review_status"]

    converged = agent_review_status(
        analyst_review={
            "primary_failure_mode": (
                "Late high-risk same-slot basis/payload incoherence that drifts the judge "
                "toward the wrong heuristic family after blocking first appears."
            ),
            "evidence_case_ids": ["ep0-step867", "ep0-step943", "ep0-step970"],
        },
        reviewer_review={
            "target_failure_mode": (
                "late_same_slot_route_basis_mismatch_on_anonymous_same_slot_candidate_under_high_future_risk"
            ),
            "allowed_change_layer": "prompt",
            "recommended_single_change": "add one explicit high-risk same-slot prompt rule",
            "evidence_case_ids": ["ep0-step867", "ep0-step943", "ep0-step970"],
        },
    )

    assert converged["convergence_status"] == "converged_ready_for_patch"

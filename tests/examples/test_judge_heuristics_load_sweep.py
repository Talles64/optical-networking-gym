from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import runpy

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = (
    PROJECT_ROOT / "examples" / "heuristics" / "judge_heuristics_load_sweep.py"
)


def _load_module() -> dict[str, object]:
    return runpy.run_path(str(EXAMPLE_PATH))


def test_judge_heuristics_load_sweep_smoke_generates_episode_and_summary_csvs(
    tmp_path: Path,
) -> None:
    module = _load_module()
    experiment = module["JudgeHeuristicLoadSweepExperiment"](
        topology_id="ring_4",
        loads=(10.0,),
        episodes_per_load=1,
        episode_length=8,
        seed=7,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        k_paths=2,
        modulations_to_consider=2,
        verbose=False,
        output_dir=tmp_path,
    )

    outputs = module["run_load_sweep"](experiment=experiment, now=datetime(2026, 3, 19))

    assert outputs.episodes_csv == tmp_path / "19-03-judge-heuristics-load-episodes.csv"
    assert outputs.summary_csv == tmp_path / "19-03-judge-heuristics-load-summary.csv"
    assert outputs.episodes_csv.exists()
    assert outputs.summary_csv.exists()

    with outputs.episodes_csv.open("r", encoding="utf-8", newline="") as handle:
        episode_rows = list(csv.DictReader(handle))
    with outputs.summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert len(episode_rows) == len(module["POLICY_NAMES"])
    assert len(summary_rows) == len(module["POLICY_NAMES"])
    assert {row["policy"] for row in episode_rows} == set(module["POLICY_NAMES"])
    assert {row["policy"] for row in summary_rows} == set(module["POLICY_NAMES"])
    assert {row["load"] for row in episode_rows} == {"10.0"}
    assert {row["requests_per_episode"] for row in episode_rows} == {"8"}
    assert {row["measure_disruptions"] for row in episode_rows} == {"False"}
    assert len({row["episode_seed"] for row in episode_rows}) == 1
    assert "total_reward" in episode_rows[0]
    assert "modulation_bpsk" in episode_rows[0]
    assert "service_blocking_rate_mean" in summary_rows[0]
    assert "total_reward_mean" in summary_rows[0]
    assert "modulation_bpsk_mean" not in summary_rows[0]


def test_judge_heuristics_load_sweep_defaults_use_shared_legacy_benchmark_profile() -> None:
    module = _load_module()
    experiment = module["JudgeHeuristicLoadSweepExperiment"](verbose=False)

    assert experiment.loads == (300.0,)
    scenario = module["build_base_scenario"](experiment)

    assert scenario.load == pytest.approx(300.0)
    assert scenario.k_paths == 5
    assert scenario.launch_power_dbm == pytest.approx(2.0)
    assert scenario.modulations_to_consider == 6
    assert tuple(modulation.name for modulation in scenario.modulations) == (
        "BPSK",
        "QPSK",
        "8QAM",
        "16QAM",
        "32QAM",
        "64QAM",
    )

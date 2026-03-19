from __future__ import annotations

from datetime import datetime
import csv
from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "SBRT26" / "osnr_margin_sweep.py"


def _load_module() -> dict[str, object]:
    return runpy.run_path(str(EXAMPLE_PATH))


def test_sbrt26_margin_sweep_build_env_forces_disruption_measurement(monkeypatch) -> None:
    module = _load_module()
    captured: dict[str, object] = {}

    def fake_make_env(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setitem(module["build_env"].__globals__, "make_env", fake_make_env)

    experiment = module["MarginSweepExperiment"](episode_length=8, seed=123)
    scenario = module["build_base_scenario"](experiment)
    episode_scenario = module["build_episode_scenario"](
        experiment=experiment,
        base_scenario=scenario,
        margin=1.5,
        episode_index=0,
    )

    module["build_env"](scenario=episode_scenario)

    assert captured["config"].measure_disruptions is True
    assert captured["config"].drop_on_disruption is False
    assert captured["config"].qot_constraint == "ASE+NLI"
    assert captured["config"].margin == 1.5


def test_sbrt26_margin_sweep_smoke_generates_episode_and_summary_csvs(tmp_path: Path) -> None:
    module = _load_module()
    experiment = module["MarginSweepExperiment"](
        topology_id="ring_4",
        margins=(0.0, 1.0),
        episodes_per_margin=1,
        episode_length=8,
        seed=7,
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        k_paths=2,
        modulations_to_consider=2,
        output_dir=tmp_path,
    )

    outputs = module["run_margin_sweep"](experiment=experiment, now=datetime(2026, 3, 19))

    assert outputs.episodes_csv == tmp_path / "19-03-margin-episodes.csv"
    assert outputs.summary_csv == tmp_path / "19-03-margin-summary.csv"
    assert outputs.episodes_csv.exists()
    assert outputs.summary_csv.exists()

    with outputs.episodes_csv.open("r", encoding="utf-8", newline="") as handle:
        episode_rows = list(csv.DictReader(handle))
    with outputs.summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))

    assert len(episode_rows) == 2
    assert len(summary_rows) == 2
    assert episode_rows[0]["policy"] == "first_fit"
    assert episode_rows[0]["measure_disruptions"] == "True"
    assert episode_rows[0]["requests_per_episode"] == "8"
    assert "blocked_due_to_resources" in episode_rows[0]
    assert "modulation_bpsk" in episode_rows[0]

    assert summary_rows[0]["policy"] == "first_fit"
    assert summary_rows[0]["measure_disruptions"] == "True"
    assert summary_rows[0]["requests_per_episode"] == "8"
    assert "services_accepted_mean" in summary_rows[0]
    assert "mean_osnr_final_mean" in summary_rows[0]
    assert "blocked_due_to_resources_mean" not in summary_rows[0]
    assert "modulation_bpsk_mean" not in summary_rows[0]

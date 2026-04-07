from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ONLINE_JUDGE_SCRIPT = SCRIPT_DIR / "online_heuristic_judge.py"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "current_judge_heuristic_seed_baseline.json"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR.parent / "results" / "heuristic_seed_baseline"


@dataclass(frozen=True, slots=True)
class JudgeHeuristicSeedBaselineConfig:
    scenario_profile: str = "legacy_benchmark"
    topology_id: str = "nobel-eu"
    loads: tuple[float, ...] = (320.0, 340.0)
    seed: int = 10
    episode_count: int = 1
    episode_length: int = 1000
    output_path: Path = DEFAULT_OUTPUT_PATH
    results_root: Path = DEFAULT_RESULTS_ROOT

    def __post_init__(self) -> None:
        object.__setattr__(self, "loads", tuple(float(load) for load in self.loads))
        object.__setattr__(self, "output_path", Path(self.output_path))
        object.__setattr__(self, "results_root", Path(self.results_root))
        if not self.loads:
            raise ValueError("loads must be non-empty")
        if any(float(load) <= 0 for load in self.loads):
            raise ValueError("loads must contain only positive values")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if int(self.episode_count) <= 0:
            raise ValueError("episode_count must be positive")
        if int(self.episode_length) <= 0:
            raise ValueError("episode_length must be positive")


def _load_online_judge_module() -> dict[str, Any]:
    return runpy.run_path(str(ONLINE_JUDGE_SCRIPT))


def _selector_functions_from_online_module(module: dict[str, Any]) -> dict[str, Any]:
    return {
        "first_fit": module["select_first_fit_runtime_action"],
        "load_balancing": module["select_load_balancing_runtime_action"],
        "highest_snr_first_fit": module["select_highest_snr_first_fit_runtime_action"],
        "ksp_best_mod_last_fit": module["select_ksp_best_mod_last_fit_runtime_action"],
        "lowest_fragmentation": module["select_lowest_fragmentation_runtime_action"],
    }


def _scenario_snapshot(scenario: Any) -> dict[str, Any]:
    return {
        "scenario_id": str(scenario.scenario_id),
        "topology_id": str(scenario.topology_id),
        "k_paths": int(scenario.k_paths),
        "num_spectrum_resources": int(scenario.num_spectrum_resources),
        "episode_length": int(scenario.episode_length),
        "max_span_length_km": float(scenario.max_span_length_km),
        "default_attenuation_db_per_km": float(scenario.default_attenuation_db_per_km),
        "default_noise_figure_db": float(scenario.default_noise_figure_db),
        "bit_rates": [int(bit_rate) for bit_rate in scenario.bit_rates],
        "load": float(scenario.load),
        "mean_holding_time": float(scenario.mean_holding_time),
        "qot_constraint": str(scenario.qot_constraint),
        "measure_disruptions": bool(scenario.measure_disruptions),
        "drop_on_disruption": bool(scenario.drop_on_disruption),
        "frequency_start": float(scenario.frequency_start),
        "frequency_slot_bandwidth": float(scenario.frequency_slot_bandwidth),
        "launch_power_dbm": float(scenario.launch_power_dbm),
        "margin": float(scenario.margin),
        "bandwidth": float(scenario.bandwidth),
        "modulations_to_consider": int(scenario.modulations_to_consider),
        "seed": int(scenario.seed),
        "modulations": [
            {
                "name": str(modulation.name),
                "maximum_length": float(modulation.maximum_length),
                "spectral_efficiency": int(modulation.spectral_efficiency),
                "minimum_osnr": float(modulation.minimum_osnr),
                "inband_xt": float(modulation.inband_xt),
            }
            for modulation in scenario.modulations
        ],
    }


def _run_single_episode(*, build_env, scenario: Any, selector: Any) -> dict[str, Any]:
    env = build_env(scenario=scenario)
    try:
        _observation, info = env.reset(seed=int(scenario.seed or 0))
        info = dict(info)
        while True:
            action = int(selector(env.heuristic_context()))
            _observation, _reward, terminated, truncated, info = env.step(action)
            info = dict(info)
            if bool(terminated or truncated):
                break
        return {
            "services_processed": int(info.get("episode_services_processed", info.get("services_processed", 0))),
            "services_accepted": int(info.get("episode_services_accepted", info.get("services_accepted", 0))),
            "service_blocking_rate": float(
                info.get("episode_service_blocking_rate", info.get("service_blocking_rate", 0.0))
            ),
            "bit_rate_blocking_rate": float(
                info.get("episode_bit_rate_blocking_rate", info.get("bit_rate_blocking_rate", 0.0))
            ),
            "disrupted_services_rate": float(
                info.get("episode_disrupted_services_rate", info.get("disrupted_services_rate", 0.0))
            ),
        }
    finally:
        env.close()


def _load_label(load: float) -> str:
    return str(int(load)) if float(load).is_integer() else str(load).replace(".", "p")


def _write_csv_rows(*, path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_heuristic_seed_baseline(
    *,
    config: JudgeHeuristicSeedBaselineConfig,
    online_module: dict[str, Any] | None = None,
) -> dict[str, Any]:
    module = _load_online_judge_module() if online_module is None else dict(online_module)
    experiment_cls = module["LLMJudgeExperiment"]
    build_base_scenario = module["build_base_scenario"]
    build_episode_scenario = module["build_episode_scenario"]
    build_env = module["build_env"]
    heuristic_order = tuple(str(name) for name in module["HEURISTIC_ORDER"])
    selectors = _selector_functions_from_online_module(module)

    generated_at = datetime.now(timezone.utc)
    timestamp_label = generated_at.strftime("%d-%m-%Hh%M")
    baseline_payload: dict[str, Any] = {
        "generated_at_utc": generated_at.isoformat(),
        "scenario_profile": str(config.scenario_profile),
        "topology_id": str(config.topology_id),
        "seed": int(config.seed),
        "episode_count": int(config.episode_count),
        "episode_length": int(config.episode_length),
        "heuristics": list(heuristic_order),
        "results_root": str(config.results_root),
        "loads": [],
    }

    for load in config.loads:
        experiment = experiment_cls(
            topology_id=config.topology_id,
            scenario_profile=config.scenario_profile,
            episode_count=config.episode_count,
            episode_length=config.episode_length,
            seed=config.seed,
            load=float(load),
            output_dir=config.output_path.parent,
        )
        base_scenario = build_base_scenario(experiment)
        heuristic_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        episode_rows_csv: list[dict[str, Any]] = []
        for heuristic_name in heuristic_order:
            episode_rows: list[dict[str, Any]] = []
            selector = selectors[heuristic_name]
            for episode_index in range(int(config.episode_count)):
                scenario = build_episode_scenario(
                    experiment=experiment,
                    base_scenario=base_scenario,
                    episode_index=episode_index,
                )
                episode_result = _run_single_episode(
                    build_env=build_env,
                    scenario=scenario,
                    selector=selector,
                )
                episode_rows.append(
                    {
                        "episode_index": episode_index,
                        "episode_seed": int(scenario.seed),
                        **episode_result,
                    }
                )
            heuristic_summary = {
                "heuristic_name": heuristic_name,
                "service_blocking_rate_mean": float(
                    mean(float(row["service_blocking_rate"]) for row in episode_rows)
                ),
                "bit_rate_blocking_rate_mean": float(
                    mean(float(row["bit_rate_blocking_rate"]) for row in episode_rows)
                ),
                "disrupted_services_rate_mean": float(
                    mean(float(row["disrupted_services_rate"]) for row in episode_rows)
                ),
                "episodes": episode_rows,
            }
            heuristic_rows.append(heuristic_summary)
            summary_rows.append(
                {
                    "load": float(load),
                    "heuristic_name": heuristic_name,
                    "service_blocking_rate_mean": heuristic_summary["service_blocking_rate_mean"],
                    "bit_rate_blocking_rate_mean": heuristic_summary["bit_rate_blocking_rate_mean"],
                    "disrupted_services_rate_mean": heuristic_summary["disrupted_services_rate_mean"],
                }
            )
            for episode_row in episode_rows:
                episode_rows_csv.append(
                    {
                        "load": float(load),
                        "heuristic_name": heuristic_name,
                        **episode_row,
                    }
                )
        best_row = min(heuristic_rows, key=lambda row: (float(row["service_blocking_rate_mean"]), str(row["heuristic_name"])))
        load_results_dir = config.results_root / f"judge_heuristic_seed_baseline_load{_load_label(float(load))}"
        summary_path = load_results_dir / f"{timestamp_label}-heuristic-baseline-summary.csv"
        episodes_path = load_results_dir / f"{timestamp_label}-heuristic-baseline-episodes.csv"
        _write_csv_rows(
            path=summary_path,
            fieldnames=[
                "load",
                "heuristic_name",
                "service_blocking_rate_mean",
                "bit_rate_blocking_rate_mean",
                "disrupted_services_rate_mean",
            ],
            rows=summary_rows,
        )
        _write_csv_rows(
            path=episodes_path,
            fieldnames=[
                "load",
                "heuristic_name",
                "episode_index",
                "episode_seed",
                "services_processed",
                "services_accepted",
                "service_blocking_rate",
                "bit_rate_blocking_rate",
                "disrupted_services_rate",
            ],
            rows=episode_rows_csv,
        )
        baseline_payload["loads"].append(
            {
                "load": float(load),
                "environment": _scenario_snapshot(base_scenario),
                "heuristics": heuristic_rows,
                "best_heuristic": str(best_row["heuristic_name"]),
                "best_service_blocking_rate_mean": float(best_row["service_blocking_rate_mean"]),
                "artifacts": {
                    "run_dir": str(load_results_dir),
                    "summary_csv": str(summary_path),
                    "episodes_csv": str(episodes_path),
                },
            }
        )
    return baseline_payload


def write_heuristic_seed_baseline(
    *,
    config: JudgeHeuristicSeedBaselineConfig,
    online_module: dict[str, Any] | None = None,
) -> Path:
    payload = build_heuristic_seed_baseline(config=config, online_module=online_module)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return config.output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 1-seed heuristic baseline for the current online judge setup")
    parser.add_argument("--scenario-profile", default="legacy_benchmark")
    parser.add_argument("--topology-id", default="nobel-eu")
    parser.add_argument("--loads", default="315.0,320.0")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--episode-count", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    args = parser.parse_args()

    loads = tuple(float(token) for token in str(args.loads).split(",") if token.strip())
    config = JudgeHeuristicSeedBaselineConfig(
        scenario_profile=args.scenario_profile,
        topology_id=args.topology_id,
        loads=loads,
        seed=args.seed,
        episode_count=args.episode_count,
        episode_length=args.episode_length,
        output_path=args.output_path,
        results_root=args.results_root,
    )
    output_path = write_heuristic_seed_baseline(config=config)
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "scenario_profile": config.scenario_profile,
                "topology_id": config.topology_id,
                "loads": list(config.loads),
                "seed": config.seed,
                "episode_count": config.episode_count,
                "episode_length": config.episode_length,
                "results_root": str(config.results_root),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

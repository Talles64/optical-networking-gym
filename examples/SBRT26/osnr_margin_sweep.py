from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import time

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
from optical_networking_gym_v2.utils import experiment_scenarios as scenario_utils
from optical_networking_gym_v2.utils import experiment_utils as sweep_utils
from optical_networking_gym_v2.utils import sweep_reporting as report_utils


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_POLICY_NAME = "first_fit"
DEFAULT_MARGINS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
DEFAULT_EPISODES_PER_MARGIN = 5
DEFAULT_EPISODE_LENGTH = 1000

MODULATION_INDEX_TO_NAME = sweep_utils.build_modulation_index_to_name(
    sweep_utils.DEFAULT_MODULATION_NAMES
)

EPISODE_BASE_FIELDS = [
    "date",
    "policy",
    "topology_id",
    "margin",
    "episode_index",
    "episode_seed",
    "episodes_per_margin",
    "requests_per_episode",
    "seed_base",
    "load",
    "mean_holding_time",
    "num_spectrum_resources",
    "k_paths",
    "launch_power_dbm",
    "measure_disruptions",
]
EPISODE_METRIC_FIELDS = [
    "services_processed",
    "services_accepted",
    "services_served",
    "service_blocking_rate",
    "service_served_rate",
    "bit_rate_blocking_rate",
    "blocked_due_to_resources",
    "blocked_due_to_osnr",
    "rejected",
    "episode_disrupted_services_count",
    "episode_disrupted_services_rate",
    "disrupted_or_dropped_services",
    "mean_osnr_accepted",
    "mean_osnr_final",
    "episode_time_s",
    *MODULATION_INDEX_TO_NAME.values(),
]
EPISODE_FIELDNAMES = EPISODE_BASE_FIELDS + EPISODE_METRIC_FIELDS

SUMMARY_BASE_FIELDS = [
    "date",
    "policy",
    "topology_id",
    "margin",
    "episodes",
    "requests_per_episode",
    "seed_base",
    "load",
    "mean_holding_time",
    "num_spectrum_resources",
    "k_paths",
    "launch_power_dbm",
    "measure_disruptions",
]
SUMMARY_METRIC_NAMES = [
    "services_accepted",
    "services_served",
    "service_blocking_rate",
    "service_served_rate",
    "bit_rate_blocking_rate",
    "mean_osnr_accepted",
    "mean_osnr_final",
    "episode_disrupted_services_count",
    "episode_disrupted_services_rate",
    "disrupted_or_dropped_services",
]
SUMMARY_FIELDNAMES = report_utils.build_summary_fieldnames(
    base_fields=SUMMARY_BASE_FIELDS,
    metric_names=SUMMARY_METRIC_NAMES,
)


@dataclass(frozen=True, slots=True)
class MarginSweepExperiment:
    topology_id: str = "nobel-eu"
    policy_name: str = DEFAULT_POLICY_NAME
    margins: tuple[float, ...] = DEFAULT_MARGINS
    episodes_per_margin: int = DEFAULT_EPISODES_PER_MARGIN
    episode_length: int = DEFAULT_EPISODE_LENGTH
    seed: int = DEFAULT_SEED
    load: float = DEFAULT_LOAD
    mean_holding_time: float = DEFAULT_MEAN_HOLDING_TIME
    num_spectrum_resources: int = DEFAULT_NUM_SPECTRUM_RESOURCES
    k_paths: int = DEFAULT_K_PATHS
    launch_power_dbm: float = DEFAULT_LAUNCH_POWER_DBM
    modulations_to_consider: int = DEFAULT_MODULATIONS_TO_CONSIDER
    measure_disruptions: bool = True
    drop_on_disruption: bool = True
    output_dir: Path = SCRIPT_DIR

    def __post_init__(self) -> None:
        object.__setattr__(self, "margins", tuple(float(margin) for margin in self.margins))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.topology_id:
            raise ValueError("topology_id must be a non-empty string")
        if self.policy_name != DEFAULT_POLICY_NAME:
            raise ValueError(f"unsupported policy {self.policy_name!r}; supported values: {DEFAULT_POLICY_NAME}")
        if not self.margins:
            raise ValueError("margins must be a non-empty sequence")
        if self.episodes_per_margin <= 0:
            raise ValueError("episodes_per_margin must be positive")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
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
class MarginSweepOutputs:
    episodes_csv: Path
    summary_csv: Path


def build_base_scenario(experiment: MarginSweepExperiment) -> ScenarioConfig:
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
    experiment: MarginSweepExperiment,
    base_scenario: ScenarioConfig,
    margin: float,
    episode_index: int,
) -> ScenarioConfig:
    episode_seed = experiment.seed + episode_index
    return replace(
        base_scenario,
        scenario_id=f"{experiment.topology_id}_margin_{margin:g}_seed{episode_seed}",
        seed=episode_seed,
        margin=float(margin),
    )


def build_env(*, scenario: ScenarioConfig):
    return make_env(config=scenario)


def run_single_episode(
    *,
    experiment: MarginSweepExperiment,
    base_scenario: ScenarioConfig,
    margin: float,
    episode_index: int,
    date_label: str,
) -> dict[str, report_utils.Scalar]:
    if experiment.policy_name != DEFAULT_POLICY_NAME:
        raise ValueError(f"unsupported policy {experiment.policy_name!r}; supported values: {DEFAULT_POLICY_NAME}")

    episode_scenario = build_episode_scenario(
        experiment=experiment,
        base_scenario=base_scenario,
        margin=margin,
        episode_index=episode_index,
    )
    env = build_env(scenario=episode_scenario)
    episode_seed = int(episode_scenario.seed or 0)
    _, info = env.reset(seed=episode_seed)

    accepted_osnrs: list[float] = []
    started_at = time.perf_counter()
    while True:
        action = sweep_utils.select_masked_first_fit_policy(env, info)
        _, _, terminated, truncated, info = env.step(action)
        if str(info.get("status", "")) == "accepted":
            accepted_osnrs.append(float(info.get("osnr", 0.0)))
        if terminated or truncated:
            break
    episode_time_s = time.perf_counter() - started_at

    simulator = env.simulator
    if simulator.statistics is None:
        raise RuntimeError("simulator statistics are not available after the episode")
    snapshot = simulator.statistics.snapshot()
    active_services = () if simulator.state is None else simulator.state.active_services_by_id.values()
    final_osnrs = [float(service.osnr) for service in active_services]

    row: dict[str, report_utils.Scalar] = {
        "date": date_label,
        "policy": experiment.policy_name,
        "topology_id": episode_scenario.topology_id,
        "margin": float(margin),
        "episode_index": int(episode_index),
        "episode_seed": episode_seed,
        "episodes_per_margin": experiment.episodes_per_margin,
        "requests_per_episode": int(episode_scenario.episode_length),
        "seed_base": int(experiment.seed),
        "load": float(episode_scenario.load),
        "mean_holding_time": float(episode_scenario.mean_holding_time),
        "num_spectrum_resources": int(episode_scenario.num_spectrum_resources),
        "k_paths": int(episode_scenario.k_paths),
        "launch_power_dbm": float(episode_scenario.launch_power_dbm),
        "measure_disruptions": bool(episode_scenario.measure_disruptions),
        "services_processed": int(snapshot.episode_services_processed),
        "services_accepted": int(snapshot.episode_services_accepted),
        "services_served": int(snapshot.episode_services_served),
        "service_blocking_rate": float(snapshot.episode_service_blocking_rate),
        "service_served_rate": float(snapshot.episode_service_served_rate),
        "bit_rate_blocking_rate": float(snapshot.episode_bit_rate_blocking_rate),
        "blocked_due_to_resources": int(snapshot.episode_services_blocked_resources),
        "blocked_due_to_osnr": int(snapshot.episode_services_blocked_qot),
        "rejected": int(snapshot.episode_services_rejected_by_agent),
        "episode_disrupted_services_count": int(snapshot.episode_disrupted_services),
        "episode_disrupted_services_rate": float(snapshot.episode_disrupted_services_rate),
        "disrupted_or_dropped_services": int(snapshot.episode_services_dropped_qot),
        "mean_osnr_accepted": sweep_utils.float_mean(accepted_osnrs),
        "mean_osnr_final": sweep_utils.float_mean(final_osnrs),
        "episode_time_s": float(episode_time_s),
    }
    row.update(
        sweep_utils.episode_modulation_counts(
            snapshot,
            modulation_index_to_name=MODULATION_INDEX_TO_NAME,
        )
    )
    return row


def _build_summary_row(
    *,
    experiment: MarginSweepExperiment,
    base_scenario: ScenarioConfig,
    margin: float,
    date_label: str,
    episode_rows: list[dict[str, report_utils.Scalar]],
) -> dict[str, report_utils.Scalar]:
    row: dict[str, report_utils.Scalar] = {
        "date": date_label,
        "policy": experiment.policy_name,
        "topology_id": base_scenario.topology_id,
        "margin": float(margin),
        "episodes": int(experiment.episodes_per_margin),
        "requests_per_episode": int(experiment.episode_length),
        "seed_base": int(experiment.seed),
        "load": float(base_scenario.load),
        "mean_holding_time": float(base_scenario.mean_holding_time),
        "num_spectrum_resources": int(base_scenario.num_spectrum_resources),
        "k_paths": int(base_scenario.k_paths),
        "launch_power_dbm": float(base_scenario.launch_power_dbm),
        "measure_disruptions": bool(base_scenario.measure_disruptions),
    }
    row.update(
        report_utils.aggregate_summary_metrics(
            episode_rows,
            metric_names=SUMMARY_METRIC_NAMES,
        )
    )
    return row


def run_margin_sweep(
    experiment: MarginSweepExperiment | None = None,
    *,
    now: datetime | None = None,
) -> MarginSweepOutputs:
    resolved_experiment = MarginSweepExperiment() if experiment is None else experiment
    base_scenario = build_base_scenario(resolved_experiment)
    date_label = report_utils.date_prefix(now)

    episode_rows: list[dict[str, report_utils.Scalar]] = []
    summary_rows: list[dict[str, report_utils.Scalar]] = []
    for margin in resolved_experiment.margins:
        margin_episode_rows = [
            run_single_episode(
                experiment=resolved_experiment,
                base_scenario=base_scenario,
                margin=float(margin),
                episode_index=episode_index,
                date_label=date_label,
            )
            for episode_index in range(resolved_experiment.episodes_per_margin)
        ]
        episode_rows.extend(margin_episode_rows)
        summary_rows.append(
            _build_summary_row(
                experiment=resolved_experiment,
                base_scenario=base_scenario,
                margin=float(margin),
                date_label=date_label,
                episode_rows=margin_episode_rows,
            )
        )

    episodes_csv, summary_csv = report_utils.build_sweep_output_paths(
        base_dir=resolved_experiment.output_dir,
        sweep_name="margin",
        now=now,
    )
    report_utils.write_csv_rows(
        path=episodes_csv,
        fieldnames=EPISODE_FIELDNAMES,
        rows=episode_rows,
    )
    report_utils.write_csv_rows(
        path=summary_csv,
        fieldnames=SUMMARY_FIELDNAMES,
        rows=summary_rows,
    )
    return MarginSweepOutputs(episodes_csv=episodes_csv, summary_csv=summary_csv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SBRT26 OSNR margin sweep with first-fit.")
    parser.add_argument("--policy", default=DEFAULT_POLICY_NAME, choices=(DEFAULT_POLICY_NAME,))
    parser.add_argument("--topology-id", default="nobel-eu")
    parser.add_argument("--episodes-per-margin", type=int, default=DEFAULT_EPISODES_PER_MARGIN)
    parser.add_argument("--request-count", type=int, default=DEFAULT_EPISODE_LENGTH)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--margins", type=float, nargs="*", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment = MarginSweepExperiment(
        topology_id=args.topology_id,
        policy_name=args.policy,
        margins=DEFAULT_MARGINS if args.margins is None else tuple(args.margins),
        episodes_per_margin=args.episodes_per_margin,
        episode_length=args.request_count,
        output_dir=args.output_dir,
    )
    outputs = run_margin_sweep(experiment=experiment)
    print(f"SBRT26 margin sweep episodes saved to: {outputs.episodes_csv}")
    print(f"SBRT26 margin sweep summary saved to: {outputs.summary_csv}")


__all__ = [
    "MarginSweepExperiment",
    "MarginSweepOutputs",
    "build_base_scenario",
    "build_env",
    "build_episode_scenario",
    "run_margin_sweep",
    "run_single_episode",
]


if __name__ == "__main__":
    main()

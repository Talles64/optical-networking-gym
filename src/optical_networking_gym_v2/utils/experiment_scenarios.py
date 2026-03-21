from __future__ import annotations

from pathlib import Path

from optical_networking_gym_v2 import BUILTIN_TOPOLOGY_DIR, ScenarioConfig, get_modulations

from .experiment_utils import DEFAULT_MODULATION_NAMES


def build_nobel_eu_graph_load_scenario(
    repo_root: Path | None = None,
    *,
    episode_length: int,
    seed: int = 50,
    load: float,
    mean_holding_time: float = 10800.0,
    num_spectrum_resources: int,
    k_paths: int,
    launch_power_dbm: float,
    modulations_to_consider: int,
    margin: float = 0.0,
    measure_disruptions: bool = True,
    drop_on_disruption: bool = False,
    topology_id: str = "nobel-eu",
) -> ScenarioConfig:
    topology_dir = BUILTIN_TOPOLOGY_DIR
    return ScenarioConfig(
        scenario_id=f"{topology_id}_graph_load_seed{seed}",
        topology_id=topology_id,
        topology_dir=topology_dir,
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        episode_length=episode_length,
        max_span_length_km=80.0,
        default_attenuation_db_per_km=0.2,
        default_noise_figure_db=4.5,
        bit_rates=(10, 40, 100, 400),
        load=load,
        mean_holding_time=mean_holding_time,
        qot_constraint="ASE+NLI",
        measure_disruptions=measure_disruptions,
        drop_on_disruption=drop_on_disruption,
        frequency_start=(3e8 / 1565e-9),
        frequency_slot_bandwidth=12.5e9,
        launch_power_dbm=launch_power_dbm,
        margin=margin,
        bandwidth=4e12,
        modulations=get_modulations(DEFAULT_MODULATION_NAMES),
        modulations_to_consider=modulations_to_consider,
        seed=seed,
    )


__all__ = ["build_nobel_eu_graph_load_scenario"]

from __future__ import annotations

from pathlib import Path

from optical_networking_gym_v2.contracts.enums import TrafficMode
from optical_networking_gym_v2.defaults import get_modulations, resolve_topology, set_topology_dir
from optical_networking_gym_v2.envs.optical_env import OpticalEnv
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.simulation.scenario import ScenarioConfig


def make_env(
    topology_name: str,
    modulation_names: str | tuple[str, ...] = "BPSK,QPSK,8QAM,16QAM,32QAM,64QAM",
    *,
    topology_dir: str | Path | None = None,
    seed: int = 42,
    bit_rates: tuple[int, ...] = (10, 40, 100, 400),
    load: float = 300.0,
    mean_holding_time: float = 100.0,
    num_spectrum_resources: int = 320,
    episode_length: int = 1_000,
    modulations_to_consider: int | None = None,
    k_paths: int = 5,
    measure_disruptions: bool = False,
    capture_traffic_table: bool = False,
) -> OpticalEnv:
    if topology_dir is not None:
        set_topology_dir(topology_dir)

    topology_path = resolve_topology(topology_name)
    topology = TopologyModel.from_file(
        topology_path,
        topology_id=topology_name,
        k_paths=k_paths,
    )
    modulations = get_modulations(modulation_names)
    config = ScenarioConfig(
        scenario_id=f"{topology_name}_seed{seed}",
        topology_id=topology_name,
        k_paths=k_paths,
        num_spectrum_resources=num_spectrum_resources,
        traffic_mode=TrafficMode.DYNAMIC,
        traffic_source={
            "bit_rates": bit_rates,
            "load": load,
            "mean_holding_time": mean_holding_time,
        },
        modulations=modulations,
        modulations_to_consider=modulations_to_consider,
        seed=seed,
        measure_disruptions=measure_disruptions,
    )
    return OpticalEnv(
        config,
        topology,
        episode_length=episode_length,
        capture_traffic_table=capture_traffic_table,
    )


__all__ = ["make_env"]

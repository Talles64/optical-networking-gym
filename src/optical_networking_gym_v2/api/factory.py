from __future__ import annotations

from pathlib import Path

from optical_networking_gym_v2.config.scenario import ScenarioConfig
from optical_networking_gym_v2.contracts.enums import MaskMode, RewardProfile, TrafficMode
from optical_networking_gym_v2.config.defaults import get_modulations, resolve_topology, set_topology_dir
from optical_networking_gym_v2.envs.optical_env import OpticalEnv
from optical_networking_gym_v2.network.topology import TopologyModel


def make_env(
    topology_name: str | None = None,
    modulation_names: str | tuple[str, ...] = "BPSK,QPSK,8QAM,16QAM,32QAM,64QAM",
    *,
    config: ScenarioConfig | None = None,
    topology_dir: str | Path | None = None,
    seed: int = 42,
    bit_rates: tuple[int, ...] = (10, 40, 100, 400),
    bit_rate_probabilities: tuple[float, ...] | None = None,
    load: float = 300.0,
    mean_holding_time: float = 100.0,
    mean_inter_arrival_time: float | None = None,
    num_spectrum_resources: int = 320,
    episode_length: int = 1_000,
    modulations_to_consider: int | None = None,
    k_paths: int = 5,
    max_span_length_km: float = 100.0,
    default_attenuation_db_per_km: float = 0.2,
    default_noise_figure_db: float = 4.5,
    traffic_mode: TrafficMode = TrafficMode.DYNAMIC,
    traffic_source: object | None = None,
    static_traffic_path: str | Path | None = None,
    mask_mode: MaskMode = MaskMode.RESOURCE_AND_QOT,
    reward_profile: RewardProfile = RewardProfile.BALANCED,
    qot_constraint: str = "ASE+NLI",
    channel_width: float = 12.5,
    frequency_start: float = (3e8 / 1565e-9),
    frequency_slot_bandwidth: float = 12.5e9,
    launch_power_dbm: float = 0.0,
    margin: float = 0.0,
    bandwidth: float | None = None,
    measure_disruptions: bool = False,
    enable_observation: bool = True,
    enable_action_mask: bool = True,
    include_mask_in_info: bool = True,
    capture_traffic_table: bool = False,
    capture_step_trace: bool = False,
) -> OpticalEnv:
    """Build an :class:`OpticalEnv` from either a complete config or a flat quick-start facade.

    Topology:
        `config`: Complete flat scenario definition. When provided, `topology_name`
        and the other flattened knobs are ignored.
        `topology_name`: Topology identifier resolved from `topology_dir` or the
        previously configured global topology directory.
        `topology_dir`: Directory containing `.xml` or `.txt` topology files.
        `k_paths`: Number of K-shortest paths to precompute.
        `max_span_length_km`, `default_attenuation_db_per_km`,
        `default_noise_figure_db`: Physical defaults applied while parsing the topology.

    Traffic:
        `traffic_mode`: `dynamic` by default; use `static` with `traffic_source`
        or `static_traffic_path` for replay.
        `bit_rates`, `bit_rate_probabilities`, `load`, `mean_holding_time`,
        `mean_inter_arrival_time`, `seed`: Dynamic-traffic controls.
        `traffic_source`: Explicit dynamic mapping or static table input.
        `static_traffic_path`: Convenience path for static trace replay.
        `episode_length`: Number of processed requests before termination.

    Physical layer:
        `num_spectrum_resources`, `channel_width`, `frequency_start`,
        `frequency_slot_bandwidth`, `launch_power_dbm`, `margin`, `bandwidth`,
        `measure_disruptions`.

    Routing / modulation:
        `modulation_names`, `modulations_to_consider`, `mask_mode`,
        `reward_profile`, `qot_constraint`.

    Outputs:
        `enable_observation`: If `False`, observations are returned as an empty
        `float32` array and the env advertises an empty Box observation space.
        `enable_action_mask`: If `False`, no action mask is materialized.
        `include_mask_in_info`: If `False`, a generated mask is exposed only
        through `env.action_masks()`.

    Instrumentation:
        `capture_traffic_table`, `capture_step_trace`: Optional capture paths
        for replay/debug artifacts. They stay disabled by default because they
        add runtime and allocation overhead.
    """
    if config is not None:
        if topology_name is not None:
            raise ValueError("topology_name cannot be combined with config")
        resolved_config = config
    else:
        if topology_name is None:
            raise ValueError("topology_name is required when config is not provided")
        resolved_config = ScenarioConfig(
            scenario_id=f"{topology_name}_seed{seed}",
            topology_id=topology_name,
            topology_dir=topology_dir,
            k_paths=k_paths,
            num_spectrum_resources=num_spectrum_resources,
            episode_length=episode_length,
            max_span_length_km=max_span_length_km,
            default_attenuation_db_per_km=default_attenuation_db_per_km,
            default_noise_figure_db=default_noise_figure_db,
            traffic_mode=traffic_mode,
            traffic_source=traffic_source,
            bit_rates=bit_rates,
            bit_rate_probabilities=bit_rate_probabilities,
            load=load,
            mean_holding_time=mean_holding_time,
            mean_inter_arrival_time=mean_inter_arrival_time,
            static_traffic_path=static_traffic_path,
            mask_mode=mask_mode,
            reward_profile=reward_profile,
            qot_constraint=qot_constraint,
            measure_disruptions=measure_disruptions,
            channel_width=channel_width,
            frequency_start=frequency_start,
            frequency_slot_bandwidth=frequency_slot_bandwidth,
            launch_power_dbm=launch_power_dbm,
            margin=margin,
            bandwidth=bandwidth,
            modulations=get_modulations(modulation_names),
            modulations_to_consider=modulations_to_consider,
            enable_observation=enable_observation,
            enable_action_mask=enable_action_mask,
            include_mask_in_info=include_mask_in_info,
            capture_traffic_table=capture_traffic_table,
            capture_step_trace=capture_step_trace,
            seed=seed,
        )

    if resolved_config.topology_dir is not None:
        set_topology_dir(resolved_config.topology_dir)

    topology_path = resolve_topology(resolved_config.topology_id)
    topology = TopologyModel.from_file(
        topology_path,
        topology_id=resolved_config.topology_id,
        k_paths=resolved_config.k_paths,
        max_span_length_km=resolved_config.max_span_length_km,
        default_attenuation_db_per_km=resolved_config.default_attenuation_db_per_km,
        default_noise_figure_db=resolved_config.default_noise_figure_db,
    )
    return OpticalEnv(
        resolved_config,
        topology,
        episode_length=resolved_config.episode_length,
        capture_traffic_table=resolved_config.capture_traffic_table,
        capture_step_trace=resolved_config.capture_step_trace,
    )


__all__ = ["make_env"]

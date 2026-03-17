from __future__ import annotations

from pathlib import Path

from optical_networking_gym_v2 import OpticalEnv, ScenarioConfig, get_modulations, make_env


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TOPOLOGY_DIR = PROJECT_ROOT / "examples" / "topologies"


def test_make_env_builds_optical_env_with_dynamic_defaults() -> None:
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        topology_dir=TOPOLOGY_DIR,
        seed=7,
        bit_rates=(40,),
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        episode_length=4,
        modulations_to_consider=2,
        k_paths=2,
    )

    assert isinstance(env, OpticalEnv)
    assert env.simulator.config.topology_id == "ring_4"
    assert env.simulator.config.seed == 7
    assert env.simulator.config.traffic_source["load"] == 10.0
    assert tuple(modulation.name for modulation in env.simulator.config.modulations) == ("QPSK", "16QAM")


def test_make_env_maps_output_flags_into_config() -> None:
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        topology_dir=TOPOLOGY_DIR,
        seed=9,
        num_spectrum_resources=24,
        episode_length=4,
        modulations_to_consider=2,
        k_paths=2,
        enable_observation=False,
        enable_action_mask=False,
        include_mask_in_info=True,
        capture_step_trace=True,
    )

    assert env.simulator.config.enable_observation is False
    assert env.simulator.config.enable_action_mask is False
    assert env.simulator.config.include_mask_in_info is False
    assert env.simulator.config.capture_step_trace is True


def test_make_env_accepts_explicit_config() -> None:
    config = ScenarioConfig(
        scenario_id="factory_config_path",
        topology_id="ring_4",
        topology_dir=TOPOLOGY_DIR,
        k_paths=2,
        num_spectrum_resources=24,
        episode_length=4,
        modulations=get_modulations("QPSK,16QAM"),
        modulations_to_consider=2,
        seed=11,
        enable_action_mask=True,
        include_mask_in_info=False,
    )

    env = make_env(config=config)

    assert isinstance(env, OpticalEnv)
    assert env.simulator.config == config
    assert env.simulator.config.include_mask_in_info is False

from __future__ import annotations

from pathlib import Path

from optical_networking_gym_v2 import OpticalEnv, make_env


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

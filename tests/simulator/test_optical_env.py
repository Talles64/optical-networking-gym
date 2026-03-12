from __future__ import annotations

from pathlib import Path

import numpy as np

from optical_networking_gym_v2 import (
    Modulation,
    OpticalEnv,
    ScenarioConfig,
    TopologyModel,
    TrafficMode,
    TrafficRecord,
    TrafficTable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RING_4_PATH = PROJECT_ROOT / "examples" / "topologies" / "ring_4.txt"


def _topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _config() -> ScenarioConfig:
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="optical_env_static",
        scenario_id="optical_env_static",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=2,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
    )
    records = (
        TrafficRecord(0, 0, 0, 2, 40, 1.0, 8.0, table_id=table.table_id, row_index=0),
        TrafficRecord(1, 1, 0, 2, 40, 2.0, 8.0, table_id=table.table_id, row_index=1),
    )
    return ScenarioConfig(
        scenario_id="optical_env_static",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source={"table": table, "records": records},
        modulations=(
            Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
            Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
        ),
        modulations_to_consider=2,
    )


def _first_valid_action(mask: np.ndarray) -> int:
    valid = np.flatnonzero(mask[:-1])
    return int(valid[0]) if valid.size > 0 else int(mask.shape[0] - 1)


def test_optical_env_exposes_spaces_and_action_masks() -> None:
    env = OpticalEnv(_config(), _topology(), episode_length=2)

    observation, info = env.reset(seed=7)

    assert env.action_space.n == 97
    assert env.observation_space.shape == observation.shape
    assert np.array_equal(env.action_masks(), info["mask"])

    action = _first_valid_action(info["mask"])
    _, _, _, _, step_info = env.step(action)

    assert env.action_masks() is not None
    assert np.array_equal(env.action_masks(), step_info["mask"])

from __future__ import annotations

from pathlib import Path

import numpy as np

from optical_networking_gym_v2 import (
    Modulation,
    Simulator,
    ScenarioConfig,
    Status,
    TopologyModel,
    TrafficMode,
    TrafficRecord,
    TrafficTable,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RING_4_PATH = PROJECT_ROOT / "examples" / "topologies" / "ring_4.txt"


def _topology() -> TopologyModel:
    return TopologyModel.from_file(RING_4_PATH, topology_id="ring_4", k_paths=2)


def _modulations() -> tuple[Modulation, ...]:
    return (
        Modulation("QPSK", 200_000.0, 2, minimum_osnr=6.72, inband_xt=-17.0),
        Modulation("16QAM", 500.0, 4, minimum_osnr=13.24, inband_xt=-23.0),
    )


def _static_source() -> dict[str, object]:
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="simulator_static",
        scenario_id="simulator_static",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=3,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
    )
    records = (
        TrafficRecord(0, 0, 0, 2, 40, 1.0, 8.0, table_id=table.table_id, row_index=0),
        TrafficRecord(1, 1, 0, 2, 40, 2.0, 8.0, table_id=table.table_id, row_index=1),
        TrafficRecord(2, 2, 0, 2, 40, 12.0, 4.0, table_id=table.table_id, row_index=2),
    )
    return {"table": table, "records": records}


def _config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="simulator_static",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source=_static_source(),
        modulations=_modulations(),
        modulations_to_consider=2,
    )


def _reverse_static_source() -> dict[str, object]:
    table = TrafficTable(
        traffic_table_version="v1",
        table_id="simulator_reverse_static",
        scenario_id="simulator_reverse_static",
        topology_id="ring_4",
        traffic_mode_source="hand_authored",
        request_count=1,
        time_unit="simulation_time",
        bit_rate_unit="Gbps",
    )
    records = (
        TrafficRecord(0, 0, 1, 0, 40, 1.0, 8.0, table_id=table.table_id, row_index=0),
    )
    return {"table": table, "records": records}


def _reverse_config() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="simulator_reverse_static",
        topology_id="ring_4",
        k_paths=2,
        num_spectrum_resources=24,
        traffic_mode=TrafficMode.STATIC,
        traffic_source=_reverse_static_source(),
        modulations=_modulations(),
        modulations_to_consider=2,
    )


def _first_valid_action(mask: np.ndarray) -> int:
    valid = np.flatnonzero(mask[:-1])
    if valid.size == 0:
        return int(mask.shape[0] - 1)
    return int(valid[0])


def test_simulator_reset_returns_initial_observation_and_mask() -> None:
    simulator = Simulator(_config(), _topology(), episode_length=3)

    observation, info = simulator.reset(seed=7)

    assert observation.dtype == np.float32
    assert observation.shape == (simulator.observation_builder.schema.total_size,)
    assert "mask" in info
    assert info["mask"].shape == (simulator.total_actions,)
    assert info["mask"][-1] == 1
    assert simulator.current_request is not None


def test_simulator_processes_static_episode_and_releases_due_services() -> None:
    simulator = Simulator(_config(), _topology(), episode_length=3)
    observation, info = simulator.reset(seed=7)
    assert observation.shape[0] > 0

    first_reward = 0.0
    final_info = {}
    for step_index in range(3):
        action = _first_valid_action(info["mask"])
        observation, reward, terminated, truncated, info = simulator.step(action)
        final_info = info
        if step_index == 0:
            first_reward = reward
            assert info["status"] == Status.ACCEPTED.value
            assert simulator.state is not None
            assert len(simulator.state.active_services_by_id) == 1
        if step_index == 1:
            assert info["status"] == Status.ACCEPTED.value
            assert simulator.state is not None
            assert len(simulator.state.active_services_by_id) == 0
        if step_index == 2:
            assert terminated is True
            assert truncated is False
            assert observation.shape == (simulator.observation_builder.schema.total_size,)
            assert np.count_nonzero(observation) == 0

    assert first_reward > 0.0
    assert final_info["episode_services_processed"] == 3
    assert final_info["episode_services_accepted"] == 3


def test_simulator_reject_action_records_rejection_and_negative_reward() -> None:
    simulator = Simulator(_config(), _topology(), episode_length=1)
    simulator.reset(seed=7)

    observation, reward, terminated, truncated, info = simulator.step(simulator.total_actions - 1)

    assert terminated is True
    assert truncated is False
    assert info["status"] == Status.REJECTED_BY_AGENT.value
    assert reward < 0.0
    assert info["reward_profile"] == "balanced"
    assert info["episode_services_accepted"] == 0
    assert info["rejected"] == 1
    assert np.count_nonzero(observation) == 0


def test_simulator_only_episode_counters_reset_preserves_current_request() -> None:
    simulator = Simulator(_config(), _topology(), episode_length=3)
    simulator.reset(seed=7)
    action = _first_valid_action(simulator.action_masks())
    simulator.step(action)

    observation, info = simulator.reset(options={"only_episode_counters": True})

    assert observation.shape == (simulator.observation_builder.schema.total_size,)
    assert info["mask"].shape == (simulator.total_actions,)
    assert simulator.current_request is not None


def test_simulator_accepts_reverse_direction_request() -> None:
    simulator = Simulator(_reverse_config(), _topology(), episode_length=1)
    _, info = simulator.reset(seed=7)

    _, reward, terminated, truncated, info = simulator.step(_first_valid_action(info["mask"]))

    assert terminated is True
    assert truncated is False
    assert reward > 0.0
    assert info["status"] == Status.ACCEPTED.value
    assert info["episode_services_accepted"] == 1

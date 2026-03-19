from __future__ import annotations

from pathlib import Path

import pytest

from optical_networking_gym_v2 import ScenarioConfig
from optical_networking_gym_v2.utils.experiment_scenarios import build_nobel_eu_graph_load_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_build_nobel_eu_graph_load_scenario_requires_episode_length() -> None:
    with pytest.raises(TypeError):
        build_nobel_eu_graph_load_scenario(
            PROJECT_ROOT.parent,
            seed=50,
            load=300.0,
            mean_holding_time=10800.0,
            num_spectrum_resources=320,
            k_paths=5,
            launch_power_dbm=1.0,
            modulations_to_consider=3,
        )


def test_build_nobel_eu_graph_load_scenario_returns_typed_config() -> None:
    scenario = build_nobel_eu_graph_load_scenario(
        PROJECT_ROOT.parent,
        topology_id="ring_4",
        episode_length=8,
        seed=7,
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        k_paths=2,
        launch_power_dbm=1.0,
        modulations_to_consider=2,
    )

    assert isinstance(scenario, ScenarioConfig)
    assert scenario.topology_id == "ring_4"
    assert scenario.episode_length == 8
    assert scenario.seed == 7
    assert scenario.topology_dir == PROJECT_ROOT.parent / "examples" / "topologies"
    assert scenario.measure_disruptions is True
    assert scenario.drop_on_disruption is False

from __future__ import annotations

from pathlib import Path

import pytest

from optical_networking_gym_v2 import BUILTIN_TOPOLOGY_DIR, ScenarioConfig
from optical_networking_gym_v2.utils.experiment_scenarios import (
    build_legacy_benchmark_scenario,
    build_nobel_eu_graph_load_scenario,
    build_nobel_eu_ofc_v1_scenario,
)


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
    assert scenario.topology_dir == BUILTIN_TOPOLOGY_DIR
    assert scenario.measure_disruptions is True
    assert scenario.drop_on_disruption is False


def test_build_nobel_eu_ofc_v1_scenario_uses_legacy_thresholds() -> None:
    scenario = build_nobel_eu_ofc_v1_scenario(seed=11)

    assert isinstance(scenario, ScenarioConfig)
    assert scenario.topology_id == "nobel-eu"
    assert scenario.episode_length == 1000
    assert scenario.seed == 11
    assert scenario.topology_dir == BUILTIN_TOPOLOGY_DIR
    assert scenario.max_span_length_km == pytest.approx(80.0)
    assert scenario.default_attenuation_db_per_km == pytest.approx(0.2)
    assert scenario.default_noise_figure_db == pytest.approx(4.5)
    assert scenario.bit_rates == (10, 40, 100, 400)
    assert scenario.measure_disruptions is False
    assert scenario.drop_on_disruption is False
    assert tuple(modulation.name for modulation in scenario.modulations) == (
        "BPSK",
        "QPSK",
        "8QAM",
        "16QAM",
        "32QAM",
        "64QAM",
    )
    assert tuple(modulation.minimum_osnr for modulation in scenario.modulations) == pytest.approx(
        (
            3.71925646843142,
            6.72955642507124,
            10.8453935345953,
            13.2406469649752,
            16.1608982942870,
            19.0134649345090,
        )
    )


def test_build_legacy_benchmark_scenario_matches_requested_profile() -> None:
    scenario = build_legacy_benchmark_scenario(seed=10)

    assert isinstance(scenario, ScenarioConfig)
    assert scenario.topology_id == "nobel-eu"
    assert scenario.episode_length == 1000
    assert scenario.seed == 10
    assert scenario.topology_dir == BUILTIN_TOPOLOGY_DIR
    assert scenario.load == 300.0
    assert scenario.mean_holding_time == 10800.0
    assert scenario.num_spectrum_resources == 320
    assert scenario.k_paths == 5
    assert scenario.launch_power_dbm == 2.0
    assert scenario.frequency_slot_bandwidth == 12.5e9
    assert scenario.frequency_start == 3e8 / 1565e-9
    assert scenario.bandwidth == 4e12
    assert scenario.bit_rates == (10, 40, 100, 400)
    assert scenario.modulations_to_consider == 6
    assert scenario.measure_disruptions is False
    assert scenario.drop_on_disruption is False
    assert tuple(modulation.name for modulation in scenario.modulations) == (
        "BPSK",
        "QPSK",
        "8QAM",
        "16QAM",
        "32QAM",
        "64QAM",
    )
    assert tuple(modulation.minimum_osnr for modulation in scenario.modulations) == pytest.approx(
        (
            3.71925646843142,
            6.72955642507124,
            10.8453935345953,
            13.2406469649752,
            16.1608982942870,
            19.0134649345090,
        )
    )

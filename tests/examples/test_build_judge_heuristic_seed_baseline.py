from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "examples" / "llm" / "build_judge_heuristic_seed_baseline.py"


def _load_module() -> dict[str, object]:
    return runpy.run_path(str(SCRIPT_PATH))


@dataclass
class _FakeScenario:
    scenario_id: str
    topology_id: str
    k_paths: int
    num_spectrum_resources: int
    episode_length: int
    max_span_length_km: float
    default_attenuation_db_per_km: float
    default_noise_figure_db: float
    bit_rates: tuple[int, ...]
    load: float
    mean_holding_time: float
    qot_constraint: str
    measure_disruptions: bool
    drop_on_disruption: bool
    frequency_start: float
    frequency_slot_bandwidth: float
    launch_power_dbm: float
    margin: float
    bandwidth: float
    modulations_to_consider: int
    seed: int
    modulations: tuple[object, ...]


class _FakeModulation:
    def __init__(self, name: str) -> None:
        self.name = name
        self.maximum_length = 1000.0
        self.spectral_efficiency = 2
        self.minimum_osnr = 10.0
        self.inband_xt = -20.0


class _FakeEnv:
    def __init__(self, scenario: _FakeScenario) -> None:
        self._scenario = scenario
        self._steps = 0

    def reset(self, *, seed: int):
        del seed
        self._steps = 0
        return None, {"episode_services_processed": 0, "episode_services_accepted": 0}

    def heuristic_context(self):
        return {}

    def step(self, action: int):
        self._steps += 1
        done = self._steps >= 2
        blocking_map = {
            1: 0.010,
            2: 0.020,
            3: 0.030,
            4: 0.015,
            5: 0.050,
        }
        blocking_rate = blocking_map[int(action)]
        info = {
            "episode_services_processed": 100,
            "episode_services_accepted": 99,
            "episode_service_blocking_rate": blocking_rate,
            "episode_bit_rate_blocking_rate": blocking_rate + 0.001,
            "episode_disrupted_services_rate": 0.0,
        }
        return None, 0.0, done, False, info

    def close(self) -> None:
        return None


@dataclass
class _FakeExperiment:
    topology_id: str
    scenario_profile: str
    episode_count: int
    episode_length: int
    seed: int
    load: float
    output_dir: Path


def test_build_heuristic_seed_baseline_uses_runner_heuristics_and_records_environment(tmp_path: Path) -> None:
    module = _load_module()
    config_cls = module["JudgeHeuristicSeedBaselineConfig"]
    build_baseline = module["build_heuristic_seed_baseline"]

    def build_base_scenario(experiment: _FakeExperiment) -> _FakeScenario:
        return _FakeScenario(
            scenario_id=f"{experiment.topology_id}_seed{experiment.seed}",
            topology_id=experiment.topology_id,
            k_paths=5,
            num_spectrum_resources=320,
            episode_length=experiment.episode_length,
            max_span_length_km=80.0,
            default_attenuation_db_per_km=0.2,
            default_noise_figure_db=4.5,
            bit_rates=(40, 100, 400),
            load=experiment.load,
            mean_holding_time=10800.0,
            qot_constraint="ASE+NLI",
            measure_disruptions=False,
            drop_on_disruption=False,
            frequency_start=1.0,
            frequency_slot_bandwidth=12.5e9,
            launch_power_dbm=0.0,
            margin=0.0,
            bandwidth=4e12,
            modulations_to_consider=3,
            seed=experiment.seed,
            modulations=(_FakeModulation("BPSK"), _FakeModulation("QPSK")),
        )

    def build_episode_scenario(*, experiment: _FakeExperiment, base_scenario: _FakeScenario, episode_index: int):
        return _FakeScenario(**{**base_scenario.__dict__, "seed": experiment.seed + episode_index})

    fake_online_module = {
        "LLMJudgeExperiment": _FakeExperiment,
        "HEURISTIC_ORDER": (
            "first_fit",
            "load_balancing",
            "highest_snr_first_fit",
            "ksp_best_mod_last_fit",
            "lowest_fragmentation",
        ),
        "build_base_scenario": build_base_scenario,
        "build_episode_scenario": build_episode_scenario,
        "build_env": lambda *, scenario: _FakeEnv(scenario),
        "select_first_fit_runtime_action": lambda _context: 1,
        "select_load_balancing_runtime_action": lambda _context: 2,
        "select_highest_snr_first_fit_runtime_action": lambda _context: 3,
        "select_ksp_best_mod_last_fit_runtime_action": lambda _context: 4,
        "select_lowest_fragmentation_runtime_action": lambda _context: 5,
    }

    payload = build_baseline(
        config=config_cls(
            loads=(400.0, 350.0),
            seed=10,
            episode_count=1,
            episode_length=1000,
            output_path=tmp_path / "baseline.json",
            results_root=tmp_path / "results",
        ),
        online_module=fake_online_module,
    )

    assert payload["heuristics"] == [
        "first_fit",
        "load_balancing",
        "highest_snr_first_fit",
        "ksp_best_mod_last_fit",
        "lowest_fragmentation",
    ]
    assert [entry["load"] for entry in payload["loads"]] == [400.0, 350.0]
    assert payload["loads"][0]["best_heuristic"] == "first_fit"
    assert payload["loads"][0]["best_service_blocking_rate_mean"] == 0.01
    assert payload["loads"][0]["environment"]["k_paths"] == 5
    assert payload["loads"][0]["environment"]["modulations_to_consider"] == 3
    assert payload["loads"][0]["environment"]["bit_rates"] == [40, 100, 400]
    assert Path(payload["loads"][0]["artifacts"]["summary_csv"]).exists()
    assert Path(payload["loads"][0]["artifacts"]["episodes_csv"]).exists()
    assert payload["loads"][0]["artifacts"]["summary_csv"].endswith("heuristic-baseline-summary.csv")
    assert payload["loads"][0]["artifacts"]["episodes_csv"].endswith("heuristic-baseline-episodes.csv")

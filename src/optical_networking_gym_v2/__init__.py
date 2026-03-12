from __future__ import annotations

from importlib import import_module

_EXPORTS: dict[str, tuple[str, str]] = {
    "ActiveService": (".envs.runtime_state", "ActiveService"),
    "ActionMask": (".rl.action_mask", "ActionMask"),
    "ActionSelection": (".contracts.action_mask", "ActionSelection"),
    "Allocation": (".contracts.allocation", "Allocation"),
    "CandidateRewardMetrics": (".contracts.reward", "CandidateRewardMetrics"),
    "Link": (".network.topology", "Link"),
    "MaskMode": (".contracts.enums", "MaskMode"),
    "Modulation": (".contracts.modulation", "Modulation"),
    "Observation": (".rl.observation", "Observation"),
    "ObservationSchema": (".contracts.observation", "ObservationSchema"),
    "ObservationSnapshot": (".contracts.observation", "ObservationSnapshot"),
    "OpticalEnv": (".envs.optical_env", "OpticalEnv"),
    "PathRecord": (".network.topology", "PathRecord"),
    "MODULATION_CATALOG": (".defaults", "MODULATION_CATALOG"),
    "QoTEngine": (".optical.qot_engine", "QoTEngine"),
    "QoTRequest": (".contracts.qot", "QoTRequest"),
    "QoTResult": (".contracts.qot", "QoTResult"),
    "RequestAnalysis": (".simulation.request_analysis", "RequestAnalysis"),
    "RequestAnalysisEngine": (".simulation.request_analysis", "RequestAnalysisEngine"),
    "RewardBreakdown": (".contracts.reward", "RewardBreakdown"),
    "RewardFunction": (".rl.reward_function", "RewardFunction"),
    "RewardInput": (".contracts.reward", "RewardInput"),
    "RewardProfile": (".contracts.enums", "RewardProfile"),
    "RuntimeState": (".envs.runtime_state", "RuntimeState"),
    "ScenarioConfig": (".simulation.scenario", "ScenarioConfig"),
    "get_modulations": (".defaults", "get_modulations"),
    "make_env": (".factory", "make_env"),
    "resolve_topology": (".defaults", "resolve_topology"),
    "select_first_fit_action": (".heuristics.first_fit", "select_first_fit_action"),
    "set_topology_dir": (".defaults", "set_topology_dir"),
    "ServiceQoTUpdate": (".contracts.qot", "ServiceQoTUpdate"),
    "ServiceRequest": (".contracts.traffic", "ServiceRequest"),
    "Simulator": (".simulation.simulator", "Simulator"),
    "Span": (".network.topology", "Span"),
    "Statistics": (".stats.statistics", "Statistics"),
    "StatisticsSnapshot": (".contracts.step", "StatisticsSnapshot"),
    "Status": (".contracts.enums", "Status"),
    "StepInfo": (".envs.step_info", "StepInfo"),
    "StepTransition": (".contracts.step", "StepTransition"),
    "TopologyModel": (".network.topology", "TopologyModel"),
    "TrafficMode": (".contracts.enums", "TrafficMode"),
    "TrafficModel": (".simulation.traffic_model", "TrafficModel"),
    "TrafficRecord": (".contracts.traffic", "TrafficRecord"),
    "TrafficTable": (".contracts.traffic", "TrafficTable"),
    "available_slots_for_path": (".network.allocation", "available_slots_for_path"),
    "benchmark_action_mask": (".bench.benchmarking", "benchmark_action_mask"),
    "benchmark_allocation": (".bench.benchmarking", "benchmark_allocation"),
    "benchmark_integrated_episode_vs_legacy": (
        ".bench.integrated_benchmarking",
        "benchmark_integrated_episode_vs_legacy",
    ),
    "benchmark_observation": (".bench.benchmarking", "benchmark_observation"),
    "benchmark_qot_engine": (".bench.benchmarking", "benchmark_qot_engine"),
    "benchmark_request_analysis": (".bench.benchmarking", "benchmark_request_analysis"),
    "benchmark_reward_function": (".bench.benchmarking", "benchmark_reward_function"),
    "benchmark_runtime_state": (".bench.benchmarking", "benchmark_runtime_state"),
    "benchmark_simulator_episode": (".bench.integrated_benchmarking", "benchmark_simulator_episode"),
    "benchmark_statistics_step_info": (".bench.benchmarking", "benchmark_statistics_step_info"),
    "build_first_fit_allocation": (".network.allocation", "build_first_fit_allocation"),
    "candidate_starts": (".network.allocation", "candidate_starts"),
    "compare_simulator_episode_with_legacy": (
        ".bench.integrated_benchmarking",
        "compare_simulator_episode_with_legacy",
    ),
    "compute_required_slots": (".network.allocation", "compute_required_slots"),
    "occupied_slot_range": (".network.allocation", "occupied_slot_range"),
    "path_is_free": (".network.allocation", "path_is_free"),
    "profile_simulator_episode": (".bench.integrated_benchmarking", "profile_simulator_episode"),
    "read_traffic_table_jsonl": (".network.traffic_table_io", "read_traffic_table_jsonl"),
    "write_traffic_table_jsonl": (".network.traffic_table_io", "write_traffic_table_jsonl"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

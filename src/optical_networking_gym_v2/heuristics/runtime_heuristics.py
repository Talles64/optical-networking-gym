from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from optical_networking_gym_v2.contracts import ActionSelection, CandidateRewardMetrics
from optical_networking_gym_v2.contracts.enums import MaskMode
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.runtime.action_codec import decode_action, encode_action, reject_action
from optical_networking_gym_v2.runtime.request_analysis import RequestAnalysis
from optical_networking_gym_v2.runtime.runtime_state import RuntimeState

if TYPE_CHECKING:
    from optical_networking_gym_v2.config.scenario import ScenarioConfig
    from optical_networking_gym_v2.contracts import ServiceRequest
    from optical_networking_gym_v2.envs.optical_env import OpticalEnv
    from optical_networking_gym_v2.runtime.simulator import Simulator


@dataclass(frozen=True, slots=True)
class RuntimeHeuristicContext:
    simulator: Simulator
    config: ScenarioConfig
    topology: TopologyModel
    state: RuntimeState
    request: ServiceRequest
    analysis: RequestAnalysis
    action_mask: np.ndarray | None

    @property
    def reject_action(self) -> int:
        return reject_action(self.config)

    def decode_action(self, action: int) -> ActionSelection | None:
        decoded = decode_action(self.config, action)
        if decoded is None:
            return None
        if decoded.path_index >= len(self.analysis.paths):
            return None
        if decoded.modulation_offset >= len(self.analysis.modulation_indices):
            return None
        return ActionSelection(
            path_index=int(decoded.path_index),
            modulation_index=int(self.analysis.modulation_indices[decoded.modulation_offset]),
            initial_slot=int(decoded.initial_slot),
        )

    def selected_candidate_metrics(self, action: int) -> CandidateRewardMetrics | None:
        selection = self.decode_action(action)
        if selection is None:
            return None
        return self.analysis.selected_candidate_metrics(
            path_index=selection.path_index,
            modulation_index=selection.modulation_index,
            initial_slot=selection.initial_slot,
        )


def _resolve_simulator(source: OpticalEnv | Simulator | RuntimeHeuristicContext):
    if isinstance(source, RuntimeHeuristicContext):
        return source.simulator

    simulator = getattr(source, "simulator", None)
    if simulator is not None and hasattr(simulator, "current_analysis") and hasattr(simulator, "topology"):
        return simulator
    if hasattr(source, "current_analysis") and hasattr(source, "topology") and hasattr(source, "config"):
        return source
    raise TypeError("runtime heuristic requires an OpticalEnv, Simulator, or RuntimeHeuristicContext")


def build_runtime_heuristic_context(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> RuntimeHeuristicContext:
    if isinstance(source, RuntimeHeuristicContext):
        return source

    simulator = _resolve_simulator(source)
    if simulator.state is None or simulator.current_request is None:
        raise RuntimeError("reset() must be called before evaluating runtime heuristics")

    analysis = simulator.current_analysis
    if analysis is None:
        analysis = simulator.analysis_engine.build(simulator.state, simulator.current_request)

    return RuntimeHeuristicContext(
        simulator=simulator,
        config=simulator.config,
        topology=simulator.topology,
        state=simulator.state,
        request=simulator.current_request,
        analysis=analysis,
        action_mask=simulator.current_mask,
    )


def _valid_start_flags(context: RuntimeHeuristicContext) -> np.ndarray:
    if context.config.mask_mode is MaskMode.RESOURCE_ONLY:
        return context.analysis.resource_valid_starts
    return context.analysis.qot_valid_starts


def select_first_fit_decision(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> tuple[int, bool, bool]:
    context = build_runtime_heuristic_context(source)
    blocked_due_to_resources = False
    blocked_due_to_qot = False
    use_resource_only = context.config.mask_mode is MaskMode.RESOURCE_ONLY

    for path_index, _path in enumerate(context.analysis.paths):
        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            resource_candidates = np.flatnonzero(
                context.analysis.resource_valid_starts[path_index, modulation_offset, :]
            )
            if resource_candidates.size == 0:
                blocked_due_to_resources = True
                continue

            if use_resource_only:
                return (
                    encode_action(
                        context.config,
                        path_index=path_index,
                        modulation_offset=modulation_offset,
                        initial_slot=int(resource_candidates[0]),
                    ),
                    False,
                    False,
                )

            qot_candidates = np.flatnonzero(context.analysis.qot_valid_starts[path_index, modulation_offset, :])
            if qot_candidates.size == 0:
                blocked_due_to_qot = True
                continue

            return (
                encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(qot_candidates[0]),
                ),
                False,
                False,
            )

    if blocked_due_to_qot:
        blocked_due_to_resources = False
    return context.reject_action, blocked_due_to_resources, blocked_due_to_qot


def select_first_fit_action(source: OpticalEnv | Simulator | RuntimeHeuristicContext) -> int:
    action, _, _ = select_first_fit_decision(source)
    return action


def select_random_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
    *,
    rng: np.random.Generator | None = None,
) -> int:
    context = build_runtime_heuristic_context(source)
    generator = rng if rng is not None else np.random.default_rng()
    valid_flags = _valid_start_flags(context)
    selected_action = context.reject_action
    valid_count = 0

    for path_index, _path in enumerate(context.analysis.paths):
        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            for initial_slot in candidate_indices:
                valid_count += 1
                if int(generator.integers(valid_count)) != 0:
                    continue
                selected_action = encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                )

    return selected_action


__all__ = [
    "RuntimeHeuristicContext",
    "build_runtime_heuristic_context",
    "select_first_fit_action",
    "select_first_fit_decision",
    "select_random_action",
]

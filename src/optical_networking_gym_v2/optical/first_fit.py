from __future__ import annotations

import numpy as np

from optical_networking_gym_v2.contracts.enums import MaskMode
from optical_networking_gym_v2.envs.optical_env import OpticalEnv
from optical_networking_gym_v2.simulation.action_codec import encode_action, reject_action
from optical_networking_gym_v2.simulation.simulator import Simulator


def _resolve_simulator(env: OpticalEnv | Simulator) -> Simulator:
    if isinstance(env, Simulator):
        return env
    simulator = getattr(env, "simulator", None)
    if isinstance(simulator, Simulator):
        return simulator
    raise TypeError("first-fit heuristic requires an OpticalEnv or Simulator instance")

def shortest_available_path_first_fit_best_modulation(
    env: OpticalEnv | Simulator,
) -> tuple[int, bool, bool]:
    simulator = _resolve_simulator(env)
    if simulator.state is None or simulator.current_request is None:
        raise RuntimeError("reset() must be called before evaluating the first-fit heuristic")

    analysis = simulator.current_analysis
    if analysis is None:
        analysis = simulator.analysis_engine.build(simulator.state, simulator.current_request)

    blocked_due_to_resources = False
    blocked_due_to_qot = False
    use_resource_only = simulator.config.mask_mode is MaskMode.RESOURCE_ONLY

    for path_index, _path in enumerate(analysis.paths):
        for modulation_offset, _modulation_index in enumerate(analysis.modulation_indices):
            resource_candidates = np.flatnonzero(
                analysis.resource_valid_starts[path_index, modulation_offset, :]
            )
            if resource_candidates.size == 0:
                blocked_due_to_resources = True
                continue

            if use_resource_only:
                initial_slot = int(resource_candidates[0])
                action = encode_action(
                    simulator.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=initial_slot,
                )
                return action, False, False

            qot_candidates = np.flatnonzero(analysis.qot_valid_starts[path_index, modulation_offset, :])
            if qot_candidates.size == 0:
                blocked_due_to_qot = True
                continue

            initial_slot = int(qot_candidates[0])
            action = encode_action(
                simulator.config,
                path_index=path_index,
                modulation_offset=modulation_offset,
                initial_slot=initial_slot,
            )
            return action, False, False

    if blocked_due_to_qot:
        blocked_due_to_resources = False
    return reject_action(simulator.config), blocked_due_to_resources, blocked_due_to_qot


def select_first_fit_action_from_env(env: OpticalEnv | Simulator) -> int:
    action, _, _ = shortest_available_path_first_fit_best_modulation(env)
    return action


__all__ = [
    "select_first_fit_action_from_env",
    "shortest_available_path_first_fit_best_modulation",
]

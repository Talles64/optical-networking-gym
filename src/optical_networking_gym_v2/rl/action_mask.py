from __future__ import annotations

import numpy as np

from optical_networking_gym_v2.contracts import ActionSelection, ServiceRequest
from optical_networking_gym_v2.envs.runtime_state import RuntimeState
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.optical.qot_engine import QoTEngine
from optical_networking_gym_v2.simulation.request_analysis import RequestAnalysisEngine
from optical_networking_gym_v2.simulation.scenario import ScenarioConfig

class ActionMask:
    def __init__(
        self,
        config: ScenarioConfig,
        topology: TopologyModel,
        qot_engine: QoTEngine,
        *,
        analysis_engine: RequestAnalysisEngine | None = None,
    ) -> None:
        if not config.modulations:
            raise ValueError("ActionMask requires ScenarioConfig.modulations")
        if config.modulations_to_consider <= 0:
            raise ValueError("ActionMask requires modulations_to_consider > 0")
        self.config = config
        self.topology = topology
        self.qot_engine = qot_engine
        self.analysis_engine = analysis_engine or RequestAnalysisEngine(config, topology, qot_engine)

    @property
    def total_actions(self) -> int:
        return (self.config.k_paths * self.config.modulations_to_consider * self.config.num_spectrum_resources) + 1

    def build(self, state: RuntimeState, request: ServiceRequest) -> np.ndarray:
        analysis = self.analysis_engine.build(state, request)
        mask = np.zeros(self.total_actions, dtype=np.uint8)
        mask[:-1] = analysis.action_mask
        mask[-1] = 1
        return mask

    def decode_action(self, action: int, state: RuntimeState, request: ServiceRequest) -> ActionSelection:
        if action < 0 or action >= self.total_actions:
            raise ValueError("action is outside the action space")
        if action == self.total_actions - 1:
            raise ValueError("reject action does not decode to a provisioning choice")

        analysis = self.analysis_engine.build(state, request)
        path_stride = self.config.modulations_to_consider * self.config.num_spectrum_resources
        path_index = action // path_stride
        modulation_and_slot = action % path_stride
        modulation_offset = modulation_and_slot // self.config.num_spectrum_resources
        initial_slot = modulation_and_slot % self.config.num_spectrum_resources
        return ActionSelection(
            path_index=int(path_index),
            modulation_index=int(analysis.modulation_indices[modulation_offset]),
            initial_slot=int(initial_slot),
        )


__all__ = ["ActionMask"]

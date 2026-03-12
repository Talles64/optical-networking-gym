from __future__ import annotations

from dataclasses import dataclass

from optical_networking_gym_v2.contracts.enums import MaskMode, RewardProfile, TrafficMode
from optical_networking_gym_v2.contracts.modulation import Modulation


_VALID_QOT_CONSTRAINTS = frozenset({"ASE+NLI", "DIST"})


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    scenario_id: str
    topology_id: str
    k_paths: int
    num_spectrum_resources: int
    traffic_mode: TrafficMode = TrafficMode.DYNAMIC
    traffic_source: object | None = None
    mask_mode: MaskMode = MaskMode.RESOURCE_AND_QOT
    reward_profile: RewardProfile = RewardProfile.BALANCED
    qot_constraint: str = "ASE+NLI"
    measure_disruptions: bool = False
    channel_width: float = 12.5
    frequency_start: float = (3e8 / 1565e-9)
    frequency_slot_bandwidth: float = 12.5e9
    launch_power_dbm: float = 0.0
    margin: float = 0.0
    bandwidth: float | None = None
    modulations: tuple[Modulation, ...] = ()
    modulations_to_consider: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id must be a non-empty string")
        if not self.topology_id:
            raise ValueError("topology_id must be a non-empty string")
        if self.k_paths <= 0:
            raise ValueError("k_paths must be positive")
        if self.num_spectrum_resources <= 0:
            raise ValueError("num_spectrum_resources must be positive")
        if self.channel_width <= 0:
            raise ValueError("channel_width must be positive")
        if self.frequency_slot_bandwidth <= 0:
            raise ValueError("frequency_slot_bandwidth must be positive")
        if self.qot_constraint not in _VALID_QOT_CONSTRAINTS:
            raise ValueError(
                "qot_constraint must be one of: " + ", ".join(sorted(_VALID_QOT_CONSTRAINTS))
            )
        if self.bandwidth is None:
            object.__setattr__(
                self,
                "bandwidth",
                self.num_spectrum_resources * self.frequency_slot_bandwidth,
            )
        elif self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if self.modulations_to_consider is None:
            object.__setattr__(self, "modulations_to_consider", len(self.modulations))
        elif self.modulations_to_consider < 0:
            raise ValueError("modulations_to_consider must be non-negative")
        elif self.modulations:
            object.__setattr__(
                self,
                "modulations_to_consider",
                min(self.modulations_to_consider, len(self.modulations)),
            )
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.traffic_mode is TrafficMode.STATIC and self.traffic_source is None:
            raise ValueError("traffic_source is required when traffic_mode is static")

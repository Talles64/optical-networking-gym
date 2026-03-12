from __future__ import annotations

from optical_networking_gym_v2 import (
    ActionMask,
    ActionSelection,
    Allocation,
    CandidateRewardMetrics,
    MaskMode,
    Modulation,
    Observation,
    ObservationSchema,
    ObservationSnapshot,
    OpticalEnv,
    QoTEngine,
    QoTResult,
    RewardBreakdown,
    RewardFunction,
    RewardInput,
    RewardProfile,
    RequestAnalysisEngine,
    ServiceRequest,
    Simulator,
    Statistics,
    StatisticsSnapshot,
    Status,
    StepInfo,
    StepTransition,
    TrafficMode,
    TrafficRecord,
    TrafficTable,
)


def test_public_api_exports_contracts() -> None:
    assert ActionMask is not None
    assert ActionSelection is not None
    assert ServiceRequest is not None
    assert TrafficRecord is not None
    assert TrafficTable is not None
    assert Allocation is not None
    assert CandidateRewardMetrics is not None
    assert Modulation is not None
    assert Observation is not None
    assert ObservationSchema is not None
    assert ObservationSnapshot is not None
    assert OpticalEnv is not None
    assert QoTEngine is not None
    assert QoTResult is not None
    assert RewardBreakdown is not None
    assert RewardFunction is not None
    assert RewardInput is not None
    assert RewardProfile.BALANCED.value == "balanced"
    assert RequestAnalysisEngine is not None
    assert Simulator is not None
    assert Statistics is not None
    assert StatisticsSnapshot is not None
    assert StepInfo is not None
    assert StepTransition is not None
    assert TrafficMode.DYNAMIC.value == "dynamic"
    assert MaskMode.RESOURCE_AND_QOT.value == "resource_and_qot"
    assert Status.BLOCKED_QOT.value == "blocked_qot"

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import json
import math
from typing import Protocol

from optical_networking_gym_v2.heuristics.runtime_heuristics import RuntimeHeuristicContext
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.runtime.request_analysis import GLOBAL_FEATURE_NAMES, PATH_FEATURE_NAMES


_GLOBAL_FEATURE_INDEX = {name: index for index, name in enumerate(GLOBAL_FEATURE_NAMES)}
_PATH_FEATURE_INDEX = {name: index for index, name in enumerate(PATH_FEATURE_NAMES)}


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(min(max(value, lower), upper))


@dataclass(frozen=True, slots=True)
class TopologyProfile:
    friendly_name: str
    route_length_regime: str
    ase_sensitivity_hint: str
    fragmentation_risk_hint: str
    decision_hint: str


@dataclass(frozen=True, slots=True)
class TopologyContextPayload:
    route_length_regime: str
    ase_sensitivity_hint: str
    fragmentation_risk_hint: str


@dataclass(frozen=True, slots=True)
class OperationalState:
    services_processed: int
    active_services_norm: float
    network_util_mean: float
    network_util_max: float
    free_slots_ratio: float
    episode_service_blocking_rate: float
    episode_bit_rate_blocking_rate: float
    episode_disrupted_services_rate: float


@dataclass(frozen=True, slots=True)
class GlobalRegimes:
    load_regime: str
    qot_pressure_regime: str


@dataclass(frozen=True, slots=True)
class DecodedActionPayload:
    path_index: int
    path_rank_k: int
    path_node_names: tuple[str, ...]
    path_hops: int
    path_length_km: float
    source_name: str
    destination_name: str
    modulation_index: int
    modulation_name: str
    modulation_spectral_efficiency: int
    initial_slot: int
    required_slots: int
    slot_end_exclusive: int


@dataclass(frozen=True, slots=True)
class RouteSummaryPayload:
    endpoint_pair: str
    path_hops: int
    path_length_km: float
    modulation_name: str
    slot_span: int


@dataclass(frozen=True, slots=True)
class CandidateMetricsPayload:
    required_slots: int
    path_link_util_mean: float
    path_link_util_max: float
    path_common_free_ratio: float
    osnr_margin: float
    nli_share: float
    worst_link_nli_share: float
    fragmentation_damage_num_blocks: float
    fragmentation_damage_largest_block: float


@dataclass(frozen=True, slots=True)
class DecisionMetricsPayload:
    required_slots: int
    path_link_util_mean: float
    path_link_util_max: float
    path_common_free_ratio: float
    osnr_margin: float
    worst_link_nli_share: float
    fragmentation_damage_num_blocks: float
    fragmentation_damage_largest_block: float


@dataclass(frozen=True, slots=True)
class CandidateCriterionScores:
    physical_safety_score: float
    fragmentation_score: float
    load_balance_score: float
    efficiency_score: float
    total_score: float


@dataclass(frozen=True, slots=True)
class JudgeDecisionCandidate:
    heuristic_name: str
    is_reject: bool
    route_summary: RouteSummaryPayload | None
    decision_metrics: DecisionMetricsPayload

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class JudgeCandidate:
    heuristic_name: str
    proposed_by: tuple[str, ...]
    raw_action: int
    is_reject: bool
    decoded_action: DecodedActionPayload | None
    metrics: CandidateMetricsPayload
    baseline_scores: CandidateCriterionScores | None = None

    def to_audit_mapping(self) -> dict[str, object]:
        return asdict(self)

    def to_decision_candidate(self) -> JudgeDecisionCandidate:
        route_summary = None
        if self.decoded_action is not None:
            route_summary = RouteSummaryPayload(
                endpoint_pair=f"{self.decoded_action.source_name}->{self.decoded_action.destination_name}",
                path_hops=int(self.decoded_action.path_hops),
                path_length_km=float(self.decoded_action.path_length_km),
                modulation_name=str(self.decoded_action.modulation_name),
                slot_span=int(self.metrics.required_slots),
            )
        return JudgeDecisionCandidate(
            heuristic_name=self.heuristic_name,
            is_reject=self.is_reject,
            route_summary=route_summary,
            decision_metrics=DecisionMetricsPayload(
                required_slots=int(self.metrics.required_slots),
                path_link_util_mean=float(self.metrics.path_link_util_mean),
                path_link_util_max=float(self.metrics.path_link_util_max),
                path_common_free_ratio=float(self.metrics.path_common_free_ratio),
                osnr_margin=float(self.metrics.osnr_margin),
                worst_link_nli_share=float(self.metrics.worst_link_nli_share),
                fragmentation_damage_num_blocks=float(self.metrics.fragmentation_damage_num_blocks),
                fragmentation_damage_largest_block=float(self.metrics.fragmentation_damage_largest_block),
            ),
        )


@dataclass(frozen=True, slots=True)
class DecisionRulesPayload:
    primary_objective: str
    priority_order: tuple[str, ...]
    tie_rule: str

    def to_mapping(self) -> dict[str, object]:
        mapping = asdict(self)
        mapping["priority_order"] = list(self.priority_order)
        return mapping


@dataclass(frozen=True, slots=True)
class JudgeDecisionPayload:
    topology_context: TopologyContextPayload
    operational_state: OperationalState
    global_regimes: GlobalRegimes
    decision_rules: DecisionRulesPayload
    candidates: tuple[JudgeDecisionCandidate, ...]

    def to_prompt_mapping(self) -> dict[str, object]:
        return {
            "topology_context": asdict(self.topology_context),
            "operational_state": asdict(self.operational_state),
            "global_regimes": asdict(self.global_regimes),
            "decision_rules": self.decision_rules.to_mapping(),
            "candidates": [candidate.to_mapping() for candidate in self.candidates],
        }

    def to_prompt_json(self) -> str:
        return json.dumps(self.to_prompt_mapping(), indent=2, sort_keys=True)


JudgePayload = JudgeDecisionPayload


@dataclass(frozen=True, slots=True)
class DecisiveSignal:
    factor: str
    supports: str
    evidence: str
    importance: str

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    winner: str
    confidence: float
    ranking: tuple[str, ...]
    reason: str
    used_tie_break: bool
    decisive_signals: tuple[DecisiveSignal, ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "winner": self.winner,
            "confidence": float(self.confidence),
            "ranking": list(self.ranking),
            "reason": self.reason,
            "used_tie_break": bool(self.used_tie_break),
            "decisive_signals": [signal.to_mapping() for signal in self.decisive_signals],
        }


@dataclass(frozen=True, slots=True)
class JudgePromptRecord:
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True, slots=True)
class JudgeCallTrace:
    prompt: JudgePromptRecord
    raw_model_response: Mapping[str, object] | None
    parsed_response: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class JudgeModelIORecord:
    raw_model_response: Mapping[str, object] | None
    parsed_response: Mapping[str, object] | None
    fallback_reason: str
    judge_error_message: str


@dataclass(frozen=True, slots=True)
class JudgeAuditSection:
    date: str
    prompt_version: str
    seed: int
    episode_index: int
    step_index: int
    topology_id: str
    candidate_audit: tuple[JudgeCandidate, ...]
    baseline_scores_by_candidate: Mapping[str, CandidateCriterionScores]
    baseline_winner: str
    agrees_with_baseline: bool
    chosen_action: int
    chosen_heuristic: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "date": self.date,
            "prompt_version": self.prompt_version,
            "seed": int(self.seed),
            "episode_index": int(self.episode_index),
            "step_index": int(self.step_index),
            "topology_id": self.topology_id,
            "candidate_audit": [candidate.to_audit_mapping() for candidate in self.candidate_audit],
            "baseline_scores_by_candidate": {
                name: asdict(scores) for name, scores in self.baseline_scores_by_candidate.items()
            },
            "baseline_winner": self.baseline_winner,
            "agrees_with_baseline": bool(self.agrees_with_baseline),
            "chosen_action": int(self.chosen_action),
            "chosen_heuristic": self.chosen_heuristic,
        }


@dataclass(frozen=True, slots=True)
class JudgeAuditRecord:
    audit: JudgeAuditSection
    decision_payload: JudgeDecisionPayload
    prompt: JudgePromptRecord
    model_io: JudgeModelIORecord

    def to_mapping(self) -> dict[str, object]:
        return {
            "audit": self.audit.to_mapping(),
            "decision_payload": self.decision_payload.to_prompt_mapping(),
            "prompt": asdict(self.prompt),
            "model_io": {
                "raw_model_response": None
                if self.model_io.raw_model_response is None
                else dict(self.model_io.raw_model_response),
                "parsed_response": None
                if self.model_io.parsed_response is None
                else dict(self.model_io.parsed_response),
                "fallback_reason": self.model_io.fallback_reason,
                "judge_error_message": self.model_io.judge_error_message,
            },
        }


class HeuristicJudge(Protocol):
    def judge(self, payload: JudgeDecisionPayload) -> JudgeVerdict:
        ...


_FRIENDLY_TOPOLOGY_NAMES = {
    "ring_4": "4-node metro ring",
    "nobel-eu": "Nobel Europe backbone",
    "nsfnet_chen": "NSFNET backbone",
}


def build_topology_profile(topology: TopologyModel) -> TopologyProfile:
    shortest_paths = [path for path in topology.paths if path.k == 0]
    if shortest_paths:
        average_shortest_km = sum(path.length_km for path in shortest_paths) / len(shortest_paths)
        p90_index = max(0, math.ceil(0.9 * len(shortest_paths)) - 1)
        sorted_shortest_km = sorted(path.length_km for path in shortest_paths)
        p90_shortest_km = sorted_shortest_km[p90_index]
        average_hops = sum(path.hops for path in shortest_paths) / len(shortest_paths)
    else:
        average_shortest_km = 0.0
        p90_shortest_km = 0.0
        average_hops = 0.0
    if topology.node_count <= 1:
        density = 0.0
    else:
        density = (2.0 * float(topology.link_count)) / float(topology.node_count * (topology.node_count - 1))

    if average_shortest_km < 600.0:
        route_length_regime = "short"
    elif average_shortest_km < 1500.0:
        route_length_regime = "medium"
    else:
        route_length_regime = "long"

    if p90_shortest_km < 1000.0:
        ase_sensitivity_hint = "low"
    elif p90_shortest_km < 2200.0:
        ase_sensitivity_hint = "moderate"
    else:
        ase_sensitivity_hint = "high"

    if average_hops < 2.0 and density >= 0.25:
        fragmentation_risk_hint = "low"
    elif average_hops < 4.0 and density >= 0.12:
        fragmentation_risk_hint = "medium"
    else:
        fragmentation_risk_hint = "high"

    decision_hint = (
        f"This topology is in a {route_length_regime}-route regime with {ase_sensitivity_hint} ASE sensitivity "
        f"and {fragmentation_risk_hint} fragmentation risk."
    )
    friendly_name = _FRIENDLY_TOPOLOGY_NAMES.get(
        topology.topology_id,
        topology.topology_id.replace("_", " ").replace("-", " ").title(),
    )
    return TopologyProfile(
        friendly_name=friendly_name,
        route_length_regime=route_length_regime,
        ase_sensitivity_hint=ase_sensitivity_hint,
        fragmentation_risk_hint=fragmentation_risk_hint,
        decision_hint=decision_hint,
    )


def build_operational_state(
    *,
    context: RuntimeHeuristicContext,
    info: Mapping[str, object],
) -> OperationalState:
    global_features = context.analysis.global_features
    return OperationalState(
        services_processed=int(info.get("episode_services_processed", info.get("services_processed", 0))),
        active_services_norm=float(global_features[_GLOBAL_FEATURE_INDEX["active_services_norm"]]),
        network_util_mean=float(global_features[_GLOBAL_FEATURE_INDEX["network_util_mean"]]),
        network_util_max=float(global_features[_GLOBAL_FEATURE_INDEX["network_util_max"]]),
        free_slots_ratio=float(global_features[_GLOBAL_FEATURE_INDEX["free_slots_ratio"]]),
        episode_service_blocking_rate=float(info.get("episode_service_blocking_rate", 0.0)),
        episode_bit_rate_blocking_rate=float(info.get("episode_bit_rate_blocking_rate", 0.0)),
        episode_disrupted_services_rate=float(info.get("episode_disrupted_services", 0.0)),
    )


def build_global_regimes(operational_state: OperationalState) -> GlobalRegimes:
    if operational_state.network_util_mean < 0.35 and operational_state.free_slots_ratio > 0.55:
        load_regime = "light"
    elif operational_state.network_util_mean < 0.60 and operational_state.free_slots_ratio > 0.30:
        load_regime = "moderate"
    elif operational_state.network_util_mean < 0.80 and operational_state.free_slots_ratio > 0.15:
        load_regime = "high"
    else:
        load_regime = "critical"

    if (
        operational_state.episode_disrupted_services_rate == 0.0
        and operational_state.episode_bit_rate_blocking_rate < 0.02
    ):
        qot_pressure_regime = "relaxed"
    elif (
        operational_state.episode_disrupted_services_rate < 0.01
        and operational_state.episode_bit_rate_blocking_rate < 0.08
    ):
        qot_pressure_regime = "guarded"
    elif (
        operational_state.episode_disrupted_services_rate < 0.03
        and operational_state.episode_bit_rate_blocking_rate < 0.15
    ):
        qot_pressure_regime = "tight"
    else:
        qot_pressure_regime = "critical"

    return GlobalRegimes(load_regime=load_regime, qot_pressure_regime=qot_pressure_regime)


def build_judge_candidate(
    *,
    context: RuntimeHeuristicContext,
    heuristic_name: str,
    action: int,
    proposed_by: Sequence[str] | None = None,
) -> JudgeCandidate:
    decoded = context.decode_action(action)
    if decoded is None:
        return JudgeCandidate(
            heuristic_name=heuristic_name,
            proposed_by=tuple(proposed_by or (heuristic_name,)),
            raw_action=int(action),
            is_reject=True,
            decoded_action=None,
            metrics=CandidateMetricsPayload(
                required_slots=0,
                path_link_util_mean=0.0,
                path_link_util_max=0.0,
                path_common_free_ratio=0.0,
                osnr_margin=0.0,
                nli_share=0.0,
                worst_link_nli_share=0.0,
                fragmentation_damage_num_blocks=0.0,
                fragmentation_damage_largest_block=0.0,
            ),
        )

    modulation_offset = context.analysis.modulation_offset_for_index(decoded.modulation_index)
    if modulation_offset is None:
        raise RuntimeError(f"modulation_index {decoded.modulation_index} is not present in the current analysis")

    path = context.analysis.paths[decoded.path_index]
    modulation = context.config.modulations[decoded.modulation_index]
    required_slots = int(context.analysis.required_slots_by_path_mod[decoded.path_index, modulation_offset])
    candidate_metrics = context.selected_candidate_metrics(action)
    if candidate_metrics is None:
        raise RuntimeError(f"candidate metrics are unavailable for action {action}")
    path_features = context.analysis.path_features[decoded.path_index]

    return JudgeCandidate(
        heuristic_name=heuristic_name,
        proposed_by=tuple(proposed_by or (heuristic_name,)),
        raw_action=int(action),
        is_reject=False,
        decoded_action=DecodedActionPayload(
            path_index=int(decoded.path_index),
            path_rank_k=int(path.k),
            path_node_names=tuple(str(name) for name in path.node_names),
            path_hops=int(path.hops),
            path_length_km=float(path.length_km),
            source_name=str(path.node_names[0]),
            destination_name=str(path.node_names[-1]),
            modulation_index=int(decoded.modulation_index),
            modulation_name=str(modulation.name),
            modulation_spectral_efficiency=int(modulation.spectral_efficiency),
            initial_slot=int(decoded.initial_slot),
            required_slots=required_slots,
            slot_end_exclusive=int(decoded.initial_slot + required_slots),
        ),
        metrics=CandidateMetricsPayload(
            required_slots=required_slots,
            path_link_util_mean=float(path_features[_PATH_FEATURE_INDEX["path_link_util_mean"]]),
            path_link_util_max=float(path_features[_PATH_FEATURE_INDEX["path_link_util_max"]]),
            path_common_free_ratio=float(path_features[_PATH_FEATURE_INDEX["path_common_free_ratio"]]),
            osnr_margin=float(candidate_metrics.osnr_margin),
            nli_share=float(candidate_metrics.nli_share),
            worst_link_nli_share=float(candidate_metrics.worst_link_nli_share),
            fragmentation_damage_num_blocks=float(candidate_metrics.fragmentation_damage_num_blocks),
            fragmentation_damage_largest_block=float(candidate_metrics.fragmentation_damage_largest_block),
        ),
    )


def _default_decision_rules() -> DecisionRulesPayload:
    return DecisionRulesPayload(
        primary_objective="Minimize overall blocking risk for the current request and future requests.",
        priority_order=(
            "non_reject",
            "qot_safety",
            "fragmentation",
            "load_balance",
            "slot_efficiency",
        ),
        tie_rule="Prefer the candidate that preserves future capacity better; if still tied, lower confidence.",
    )


def score_candidates(
    candidates: Sequence[JudgeCandidate],
) -> tuple[tuple[JudgeCandidate, ...], str]:
    if not candidates:
        raise ValueError("at least one candidate is required")

    non_reject = [candidate for candidate in candidates if not candidate.is_reject]
    max_required_slots = max((candidate.metrics.required_slots for candidate in non_reject), default=1)
    max_frag_blocks = max((candidate.metrics.fragmentation_damage_num_blocks for candidate in non_reject), default=1.0)
    max_frag_largest = max(
        (candidate.metrics.fragmentation_damage_largest_block for candidate in non_reject),
        default=1.0,
    )

    scored: list[JudgeCandidate] = []
    for candidate in candidates:
        if candidate.is_reject:
            scores = CandidateCriterionScores(
                physical_safety_score=0.0,
                fragmentation_score=0.0,
                load_balance_score=0.0,
                efficiency_score=0.0,
                total_score=0.0,
            )
        else:
            physical_safety_score = (
                0.7 * _clamp(candidate.metrics.osnr_margin / 3.0)
                + 0.3 * (1.0 - _clamp(candidate.metrics.worst_link_nli_share))
            )
            fragmentation_score = (
                0.5 * (1.0 - _clamp(candidate.metrics.fragmentation_damage_num_blocks / max(max_frag_blocks, 1.0)))
                + 0.5
                * (1.0 - _clamp(candidate.metrics.fragmentation_damage_largest_block / max(max_frag_largest, 1.0)))
            )
            load_balance_score = (
                0.4 * (1.0 - _clamp(candidate.metrics.path_link_util_max))
                + 0.3 * (1.0 - _clamp(candidate.metrics.path_link_util_mean))
                + 0.3 * _clamp(candidate.metrics.path_common_free_ratio)
            )
            efficiency_score = 1.0 - _clamp(candidate.metrics.required_slots / max(max_required_slots, 1))
            total_score = (
                0.35 * physical_safety_score
                + 0.25 * fragmentation_score
                + 0.25 * load_balance_score
                + 0.15 * efficiency_score
            )
            scores = CandidateCriterionScores(
                physical_safety_score=float(physical_safety_score),
                fragmentation_score=float(fragmentation_score),
                load_balance_score=float(load_balance_score),
                efficiency_score=float(efficiency_score),
                total_score=float(total_score),
            )
        scored.append(replace(candidate, baseline_scores=scores))

    def winner_key(candidate: JudgeCandidate) -> tuple[int, float, float, float, int]:
        assert candidate.baseline_scores is not None
        return (
            0 if candidate.is_reject else 1,
            candidate.baseline_scores.total_score,
            candidate.baseline_scores.physical_safety_score,
            candidate.baseline_scores.load_balance_score,
            -candidate.raw_action,
        )

    baseline_winner = max(scored, key=winner_key).heuristic_name
    return tuple(scored), baseline_winner


def build_judge_payload(
    *,
    topology_profile: TopologyProfile,
    operational_state: OperationalState,
    global_regimes: GlobalRegimes,
    candidates: Sequence[JudgeCandidate],
) -> JudgeDecisionPayload:
    return JudgeDecisionPayload(
        topology_context=TopologyContextPayload(
            route_length_regime=topology_profile.route_length_regime,
            ase_sensitivity_hint=topology_profile.ase_sensitivity_hint,
            fragmentation_risk_hint=topology_profile.fragmentation_risk_hint,
        ),
        operational_state=operational_state,
        global_regimes=global_regimes,
        decision_rules=_default_decision_rules(),
        candidates=tuple(candidate.to_decision_candidate() for candidate in candidates),
    )


def build_judge_audit_record(
    *,
    date: str,
    prompt_version: str,
    seed: int,
    episode_index: int,
    step_index: int,
    topology_id: str,
    decision_payload: JudgeDecisionPayload,
    prompt: JudgePromptRecord,
    raw_model_response: Mapping[str, object] | None,
    parsed_response: Mapping[str, object] | None,
    fallback_reason: str,
    judge_error_message: str,
    candidates: Sequence[JudgeCandidate],
    baseline_winner: str,
    chosen_action: int,
    chosen_heuristic: str,
) -> JudgeAuditRecord:
    baseline_scores_by_candidate: dict[str, CandidateCriterionScores] = {}
    for candidate in candidates:
        if candidate.baseline_scores is None:
            raise ValueError("candidate baseline scores are required for the audit record")
        baseline_scores_by_candidate[candidate.heuristic_name] = candidate.baseline_scores
    return JudgeAuditRecord(
        audit=JudgeAuditSection(
            date=date,
            prompt_version=prompt_version,
            seed=int(seed),
            episode_index=int(episode_index),
            step_index=int(step_index),
            topology_id=topology_id,
            candidate_audit=tuple(candidates),
            baseline_scores_by_candidate=baseline_scores_by_candidate,
            baseline_winner=baseline_winner,
            agrees_with_baseline=(chosen_heuristic == baseline_winner),
            chosen_action=int(chosen_action),
            chosen_heuristic=chosen_heuristic,
        ),
        decision_payload=decision_payload,
        prompt=prompt,
        model_io=JudgeModelIORecord(
            raw_model_response=raw_model_response,
            parsed_response=parsed_response,
            fallback_reason=fallback_reason,
            judge_error_message=judge_error_message,
        ),
    )


__all__ = [
    "CandidateCriterionScores",
    "CandidateMetricsPayload",
    "DecodedActionPayload",
    "DecisiveSignal",
    "DecisionMetricsPayload",
    "DecisionRulesPayload",
    "GlobalRegimes",
    "HeuristicJudge",
    "JudgeAuditRecord",
    "JudgeCallTrace",
    "JudgeCandidate",
    "JudgeDecisionCandidate",
    "JudgeDecisionPayload",
    "JudgeModelIORecord",
    "JudgePayload",
    "JudgePromptRecord",
    "JudgeVerdict",
    "OperationalState",
    "RouteSummaryPayload",
    "TopologyContextPayload",
    "TopologyProfile",
    "build_global_regimes",
    "build_judge_audit_record",
    "build_judge_candidate",
    "build_judge_payload",
    "build_operational_state",
    "build_topology_profile",
    "score_candidates",
]

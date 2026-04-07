from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import json
import math
from typing import Protocol

from optical_networking_gym_v2.heuristics.runtime_heuristics import RuntimeHeuristicContext
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.runtime.request_analysis import (
    GLOBAL_FEATURE_NAMES,
    PATH_FEATURE_NAMES,
    PATH_SLOT_FEATURE_NAMES,
)


_GLOBAL_FEATURE_INDEX = {name: index for index, name in enumerate(GLOBAL_FEATURE_NAMES)}
_PATH_FEATURE_INDEX = {name: index for index, name in enumerate(PATH_FEATURE_NAMES)}
_PATH_SLOT_FEATURE_INDEX = {name: index for index, name in enumerate(PATH_SLOT_FEATURE_NAMES)}
_INTERNAL_SHORTLIST_ROLE_NAMES = frozenset(
    {
        "balanced_anchor",
        "backfill_challenger",
        "same_slot_preservation_anchor",
        "same_path_slot_variant_challenger",
        "same_slot_route_challenger",
        "same_slot_common_free_challenger",
        "same_slot_backfill_challenger",
        "extra_slot_structural_challenger",
        "extra_slot_backfill_challenger",
    }
)


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(min(max(value, lower), upper))


def _round_floats(obj: object, ndigits: int = 4) -> object:
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {key: _round_floats(value, ndigits) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(item, ndigits) for item in obj]
    return obj


def _payload_visible_candidate_roles(candidate_roles: Sequence[str] | None) -> tuple[str, ...]:
    if candidate_roles is None:
        return ()
    visible_roles = [
        str(role)
        for role in candidate_roles
        if str(role) and str(role) not in _INTERNAL_SHORTLIST_ROLE_NAMES
    ]
    return tuple(dict.fromkeys(visible_roles))


@dataclass(frozen=True, slots=True)
class TopologyProfile:
    friendly_name: str
    route_length_regime: str
    ase_sensitivity_hint: str
    fragmentation_risk_hint: str
    decision_hint: str
    average_shortest_path_km: float
    average_hops: float
    p90_shortest_path_km: float


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
class RequestPayload:
    services_processed: int
    source: str
    destination: str
    bit_rate_gbps: int


@dataclass(frozen=True, slots=True)
class NetworkStatePayload:
    episode_service_blocking_rate: float
    network_util_mean: float
    network_util_max: float
    free_slots_ratio: float


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
    path_index: int
    path_hops: int
    path_length_km: float
    modulation_name: str
    initial_slot: int
    required_slots: int


@dataclass(frozen=True, slots=True)
class CandidateMetricsPayload:
    required_slots: int
    path_link_util_mean: float
    path_link_util_max: float
    path_common_free_ratio: float
    path_common_largest_block_ratio: float
    path_common_num_blocks_norm: float
    path_route_cuts_norm: float
    path_route_rss: float
    osnr_margin_db: float
    nli_share: float
    worst_link_nli_share: float
    common_block_length_norm: float
    left_free_span_norm: float
    right_free_span_norm: float
    local_fragmentation: float
    fragmentation_damage_num_blocks: float
    fragmentation_damage_largest_block: float
    fragmentation_added_blocks: int
    largest_block_loss_slots: int
    qot_safe_now: bool = False
    qot_band: str = "unsafe"
    qot_margin_clipped_db: float = 0.0
    qot_excess_db_over_floor: float = 0.0
    fragmentation_added_blocks_norm: float = 0.0
    largest_block_loss_slots_norm: float = 0.0
    slot_cost_vs_best: int = 0
    slot_ratio_vs_best: float = 1.0
    local_damage_score: float = 0.0
    route_pressure_score: float = 0.0
    same_path_only_modulation_tradeoff: bool = False
    same_path_modulation_warning: bool = False
    extra_slots_for_same_path: int = 0
    same_slots_tradeoff: bool = False
    delta_route_pressure_vs_best_peer: float = 0.0
    delta_local_damage_vs_best_peer: float = 0.0
    delta_common_free_ratio_vs_best_peer: float = 0.0
    equal_slot_route_pressure_warning: bool = False
    future_risk_band: str = "high"
    is_pareto_dominated: bool = False
    num_candidates_dominating_this: int = 0
    num_candidates_dominated_by_this: int = 0
    plausibility_score: float = -1.0
    plausibility_rank: int = 0
    blocking_proxy_score: float = -1.0


@dataclass(frozen=True, slots=True)
class DecisionMetricsPayload:
    qot_safe_now: bool
    qot_band: str
    osnr_margin_db: float
    qot_margin_clipped_db: float
    qot_excess_db_over_floor: float
    required_slots: int
    slot_cost_vs_best: int
    slot_ratio_vs_best: float
    local_damage_score: float
    route_pressure_score: float
    path_link_util_mean: float
    path_link_util_max: float
    path_route_cuts_norm: float
    path_route_rss: float
    local_fragmentation: float
    fragmentation_added_blocks: int
    largest_block_loss_slots: int
    path_common_num_blocks_norm: float
    common_block_length_norm: float
    left_free_span_norm: float
    right_free_span_norm: float
    slot_span_total_norm: float
    path_common_free_ratio: float
    support_count: int
    has_multi_heuristic_support: bool
    same_slots_tradeoff: bool
    delta_route_pressure_vs_best_peer: float
    delta_local_damage_vs_best_peer: float
    delta_common_free_ratio_vs_best_peer: float
    equal_slot_route_pressure_warning: bool
    same_path_modulation_warning: bool
    future_risk_band: str
    is_pareto_dominated: bool
    num_candidates_dominating_this: int
    num_candidates_dominated_by_this: int
    same_path_only_modulation_tradeoff: bool
    extra_slots_for_same_path: int


@dataclass(frozen=True, slots=True)
class PairwiseDeltaPayload:
    candidate_id: str
    vs_candidate_id: str
    same_path_same_modulation: bool
    delta_required_slots: int
    delta_route_pressure_score: float
    delta_local_damage_score: float
    delta_path_common_free_ratio: float
    delta_fragmentation_added_blocks: int
    delta_largest_block_loss_slots: int
    delta_local_fragmentation: float
    delta_left_free_span_norm: float
    delta_right_free_span_norm: float
    delta_slot_span_total_norm: float
    delta_qot_margin_clipped_db: float

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromptContextPayload:
    min_required_slots_in_shortlist: int
    same_slot_candidate_ids: tuple[str, ...]
    extra_slot_candidate_ids: tuple[str, ...]
    progress_ratio: float
    congestion_band: str
    same_slot_near_tie_band: str
    same_slot_damage_axes_tie_band: str
    same_slot_route_common_free_alignment: str
    same_path_slot_variant_band: str
    route_gain_band_vs_same_slot_best: str
    local_gain_band_vs_same_slot_best: str
    same_slot_local_support_band: str
    common_free_penalty_band_vs_same_slot_best: str
    future_feasibility_risk_band: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "min_required_slots_in_shortlist": int(self.min_required_slots_in_shortlist),
            "same_slot_candidate_ids": list(self.same_slot_candidate_ids),
            "extra_slot_candidate_ids": list(self.extra_slot_candidate_ids),
            "progress_ratio": float(self.progress_ratio),
            "congestion_band": str(self.congestion_band),
            "same_slot_near_tie_band": str(self.same_slot_near_tie_band),
            "same_slot_damage_axes_tie_band": str(self.same_slot_damage_axes_tie_band),
            "same_slot_route_common_free_alignment": str(self.same_slot_route_common_free_alignment),
            "same_path_slot_variant_band": str(self.same_path_slot_variant_band),
            "route_gain_band_vs_same_slot_best": str(self.route_gain_band_vs_same_slot_best),
            "local_gain_band_vs_same_slot_best": str(self.local_gain_band_vs_same_slot_best),
            "same_slot_local_support_band": str(self.same_slot_local_support_band),
            "common_free_penalty_band_vs_same_slot_best": str(self.common_free_penalty_band_vs_same_slot_best),
            "future_feasibility_risk_band": str(self.future_feasibility_risk_band),
        }


@dataclass(frozen=True, slots=True)
class CandidateCriterionScores:
    slot_efficiency_component: float
    route_pressure_component: float
    local_damage_component: float
    common_free_component: float
    qot_tiebreak_component: float
    dominated_penalty: float
    route_warning_penalty: float
    same_path_penalty: float
    total_score: float


@dataclass(frozen=True, slots=True)
class JudgeDecisionCandidate:
    candidate_id: str
    candidate_roles: tuple[str, ...]
    is_reject: bool
    route: RouteSummaryPayload | None
    metrics: DecisionMetricsPayload

    @property
    def route_summary(self) -> RouteSummaryPayload | None:
        return self.route

    def to_mapping(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_roles": list(self.candidate_roles),
            "is_reject": self.is_reject,
            "route": None if self.route is None else asdict(self.route),
            "metrics": asdict(self.metrics),
        }


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

    def to_decision_candidate(
        self,
        *,
        candidate_id: str = "A",
        candidate_role: str = "backfill_challenger",
        candidate_roles: Sequence[str] | None = None,
    ) -> JudgeDecisionCandidate:
        route = None
        if self.decoded_action is not None:
            route = RouteSummaryPayload(
                path_index=int(self.decoded_action.path_index),
                path_hops=int(self.decoded_action.path_hops),
                path_length_km=float(self.decoded_action.path_length_km),
                modulation_name=str(self.decoded_action.modulation_name),
                initial_slot=int(self.decoded_action.initial_slot),
                required_slots=int(self.metrics.required_slots),
            )
        return JudgeDecisionCandidate(
            candidate_id=candidate_id,
            candidate_roles=tuple(
                str(role) for role in (candidate_roles if candidate_roles is not None else (candidate_role,))
            ),
            is_reject=self.is_reject,
            route=route,
            metrics=DecisionMetricsPayload(
                qot_safe_now=bool(self.metrics.qot_safe_now),
                qot_band=str(self.metrics.qot_band),
                osnr_margin_db=float(self.metrics.osnr_margin_db),
                qot_margin_clipped_db=float(self.metrics.qot_margin_clipped_db),
                qot_excess_db_over_floor=float(self.metrics.qot_excess_db_over_floor),
                required_slots=int(self.metrics.required_slots),
                slot_cost_vs_best=int(self.metrics.slot_cost_vs_best),
                slot_ratio_vs_best=float(self.metrics.slot_ratio_vs_best),
                local_damage_score=float(self.metrics.local_damage_score),
                route_pressure_score=float(self.metrics.route_pressure_score),
                path_link_util_mean=float(self.metrics.path_link_util_mean),
                path_link_util_max=float(self.metrics.path_link_util_max),
                path_route_cuts_norm=float(self.metrics.path_route_cuts_norm),
                path_route_rss=float(self.metrics.path_route_rss),
                local_fragmentation=float(self.metrics.local_fragmentation),
                fragmentation_added_blocks=int(self.metrics.fragmentation_added_blocks),
                largest_block_loss_slots=int(self.metrics.largest_block_loss_slots),
                path_common_num_blocks_norm=float(self.metrics.path_common_num_blocks_norm),
                common_block_length_norm=float(self.metrics.common_block_length_norm),
                left_free_span_norm=float(self.metrics.left_free_span_norm),
                right_free_span_norm=float(self.metrics.right_free_span_norm),
                slot_span_total_norm=float(self.metrics.left_free_span_norm + self.metrics.right_free_span_norm),
                path_common_free_ratio=float(self.metrics.path_common_free_ratio),
                support_count=int(len(self.proposed_by)),
                has_multi_heuristic_support=bool(len(self.proposed_by) > 1),
                same_slots_tradeoff=bool(self.metrics.same_slots_tradeoff),
                delta_route_pressure_vs_best_peer=float(self.metrics.delta_route_pressure_vs_best_peer),
                delta_local_damage_vs_best_peer=float(self.metrics.delta_local_damage_vs_best_peer),
                delta_common_free_ratio_vs_best_peer=float(self.metrics.delta_common_free_ratio_vs_best_peer),
                equal_slot_route_pressure_warning=bool(self.metrics.equal_slot_route_pressure_warning),
                same_path_modulation_warning=bool(self.metrics.same_path_modulation_warning),
                future_risk_band=str(self.metrics.future_risk_band),
                is_pareto_dominated=bool(self.metrics.is_pareto_dominated),
                num_candidates_dominating_this=int(self.metrics.num_candidates_dominating_this),
                num_candidates_dominated_by_this=int(self.metrics.num_candidates_dominated_by_this),
                same_path_only_modulation_tradeoff=bool(self.metrics.same_path_only_modulation_tradeoff),
                extra_slots_for_same_path=int(self.metrics.extra_slots_for_same_path),
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
    prompt_version: str
    request: RequestPayload
    network_state: NetworkStatePayload
    candidates: tuple[JudgeDecisionCandidate, ...]
    pairwise_deltas: tuple[PairwiseDeltaPayload, ...] = ()
    prompt_context: PromptContextPayload | None = None

    def to_prompt_mapping(self) -> dict[str, object]:
        return _round_floats(
            {
                "prompt_version": self.prompt_version,
                "request": asdict(self.request),
                "network_state": asdict(self.network_state),
                "candidates": [candidate.to_mapping() for candidate in self.candidates],
                "pairwise_deltas": [pair.to_mapping() for pair in self.pairwise_deltas],
                "prompt_context": None if self.prompt_context is None else self.prompt_context.to_mapping(),
            }
        )

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
    winner_candidate_id: str
    confidence: float
    decision_basis: str = ""
    ranking: tuple[str, ...] = ()
    reason: str = ""
    used_tie_break: bool = False
    decisive_signals: tuple[DecisiveSignal, ...] = ()

    @property
    def winner(self) -> str:
        return self.winner_candidate_id

    def to_mapping(self) -> dict[str, object]:
        mapping: dict[str, object] = {
            "winner_candidate_id": self.winner_candidate_id,
            "confidence": float(self.confidence),
        }
        if self.decision_basis:
            mapping["decision_basis"] = self.decision_basis
        if self.ranking:
            mapping["ranking"] = list(self.ranking)
        if self.reason:
            mapping["reason"] = self.reason
        if self.used_tie_break:
            mapping["used_tie_break"] = True
        if self.decisive_signals:
            mapping["decisive_signals"] = [signal.to_mapping() for signal in self.decisive_signals]
        return mapping


@dataclass(frozen=True, slots=True)
class JudgePromptRecord:
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True, slots=True)
class JudgeCallTrace:
    prompt: JudgePromptRecord
    raw_model_response: Mapping[str, object] | None
    parsed_response: object | None


@dataclass(frozen=True, slots=True)
class JudgeModelIORecord:
    raw_model_response: Mapping[str, object] | None
    parsed_response: object | None
    fallback_reason: str
    judge_error_message: str
    semantic_warning_flags: tuple[str, ...] = ()
    basis_vs_payload_mismatch: bool = False


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
    reference_winner: str
    agrees_with_reference: bool
    chosen_action: int
    chosen_heuristic: str
    winner_proposed_by: tuple[str, ...]
    controller_decision_source: str
    raw_candidate_count: int
    surviving_candidate_count: int
    pruned_dominated_count: int
    prompt_candidate_count: int
    pre_shuffle_shortlist_actions: tuple[int, ...] = ()
    post_shuffle_shortlist_actions: tuple[int, ...] = ()
    prompt_permutation: tuple[int, ...] = ()
    hidden_balanced_candidate_id: str = ""
    hidden_balanced_candidate_action: int = -1
    hidden_balanced_candidate_heuristic: str = ""

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
            "baseline_winner": self.reference_winner,
            "reference_winner": self.reference_winner,
            "agrees_with_reference": bool(self.agrees_with_reference),
            "chosen_action": int(self.chosen_action),
            "chosen_heuristic": self.chosen_heuristic,
            "winner_proposed_by": list(self.winner_proposed_by),
            "controller_decision_source": self.controller_decision_source,
            "raw_candidate_count": int(self.raw_candidate_count),
            "surviving_candidate_count": int(self.surviving_candidate_count),
            "pruned_dominated_count": int(self.pruned_dominated_count),
            "prompt_candidate_count": int(self.prompt_candidate_count),
            "pre_shuffle_shortlist_actions": [int(action) for action in self.pre_shuffle_shortlist_actions],
            "post_shuffle_shortlist_actions": [int(action) for action in self.post_shuffle_shortlist_actions],
            "prompt_permutation": [int(index) for index in self.prompt_permutation],
            "hidden_balanced_candidate_id": str(self.hidden_balanced_candidate_id),
            "hidden_balanced_candidate_action": int(self.hidden_balanced_candidate_action),
            "hidden_balanced_candidate_heuristic": str(self.hidden_balanced_candidate_heuristic),
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
                "parsed_response": _serialize_json_like(self.model_io.parsed_response),
                "fallback_reason": self.model_io.fallback_reason,
                "judge_error_message": self.model_io.judge_error_message,
                "semantic_warning_flags": list(self.model_io.semantic_warning_flags),
                "basis_vs_payload_mismatch": bool(self.model_io.basis_vs_payload_mismatch),
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


def _serialize_json_like(obj: object | None) -> object | None:
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return dict(obj)
    if isinstance(obj, list):
        return [_serialize_json_like(item) for item in obj]
    return obj


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
        average_shortest_path_km=float(average_shortest_km),
        average_hops=float(average_hops),
        p90_shortest_path_km=float(p90_shortest_km),
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


def _source_destination_names(context: RuntimeHeuristicContext) -> tuple[str, str]:
    topology = context.topology
    request = context.request
    return (
        str(topology.node_names[request.source_id]),
        str(topology.node_names[request.destination_id]),
    )


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
                path_common_largest_block_ratio=0.0,
                path_common_num_blocks_norm=0.0,
                path_route_cuts_norm=0.0,
                path_route_rss=0.0,
                osnr_margin_db=-1.0,
                nli_share=0.0,
                worst_link_nli_share=0.0,
                common_block_length_norm=0.0,
                left_free_span_norm=0.0,
                right_free_span_norm=0.0,
                local_fragmentation=0.0,
                fragmentation_damage_num_blocks=0.0,
                fragmentation_damage_largest_block=0.0,
                fragmentation_added_blocks=0,
                largest_block_loss_slots=0,
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
    slot_features = context.analysis.path_slot_features[decoded.path_index, decoded.initial_slot]
    max_block_count = max(1, math.ceil(context.config.num_spectrum_resources / 2))

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
            path_common_largest_block_ratio=float(
                path_features[_PATH_FEATURE_INDEX["path_common_largest_block_ratio"]]
            ),
            path_common_num_blocks_norm=float(path_features[_PATH_FEATURE_INDEX["path_common_num_blocks_norm"]]),
            path_route_cuts_norm=float(path_features[_PATH_FEATURE_INDEX["path_route_cuts_norm"]]),
            path_route_rss=float(path_features[_PATH_FEATURE_INDEX["path_route_rss"]]),
            osnr_margin_db=float(candidate_metrics.osnr_margin),
            nli_share=float(candidate_metrics.nli_share),
            worst_link_nli_share=float(candidate_metrics.worst_link_nli_share),
            common_block_length_norm=float(slot_features[_PATH_SLOT_FEATURE_INDEX["common_block_length_norm"]]),
            left_free_span_norm=float(slot_features[_PATH_SLOT_FEATURE_INDEX["left_free_span_norm"]]),
            right_free_span_norm=float(slot_features[_PATH_SLOT_FEATURE_INDEX["right_free_span_norm"]]),
            local_fragmentation=float(slot_features[_PATH_SLOT_FEATURE_INDEX["local_fragmentation"]]),
            fragmentation_damage_num_blocks=float(candidate_metrics.fragmentation_damage_num_blocks),
            fragmentation_damage_largest_block=float(candidate_metrics.fragmentation_damage_largest_block),
            fragmentation_added_blocks=int(
                round(float(candidate_metrics.fragmentation_damage_num_blocks) * max_block_count)
            ),
            largest_block_loss_slots=int(
                round(float(candidate_metrics.fragmentation_damage_largest_block) * context.config.num_spectrum_resources)
            ),
        ),
    )


def _qot_band(margin: float) -> str:
    if margin < 0.0:
        return "unsafe"
    if margin < 0.5:
        return "marginal"
    if margin < 2.0:
        return "moderate"
    return "strong"


def _candidate_path_index(candidate: JudgeCandidate) -> int | None:
    if candidate.decoded_action is None:
        return None
    return int(candidate.decoded_action.path_index)


def _is_edge_aligned_slot(metrics: CandidateMetricsPayload, *, atol: float = 1e-9) -> bool:
    return bool(
        float(metrics.left_free_span_norm) <= atol
        or float(metrics.right_free_span_norm) <= atol
    )


def _is_split_slot(metrics: CandidateMetricsPayload, *, atol: float = 1e-9) -> bool:
    return bool(
        float(metrics.left_free_span_norm) > atol
        and float(metrics.right_free_span_norm) > atol
    )


def _recalibrate_same_path_slot_competition(
    candidates: Sequence[JudgeCandidate],
) -> tuple[JudgeCandidate, ...]:
    groups: dict[tuple[int, int], list[JudgeCandidate]] = {}
    for candidate in candidates:
        if not candidate.metrics.qot_safe_now:
            continue
        path_index = _candidate_path_index(candidate)
        if path_index is None:
            continue
        groups.setdefault((int(path_index), int(candidate.metrics.required_slots)), []).append(candidate)

    overrides: dict[int, CandidateMetricsPayload] = {}
    for group_candidates in groups.values():
        if len(group_candidates) <= 1:
            continue
        route_span = max(float(candidate.metrics.route_pressure_score) for candidate in group_candidates) - min(
            float(candidate.metrics.route_pressure_score) for candidate in group_candidates
        )
        common_free_span = max(float(candidate.metrics.path_common_free_ratio) for candidate in group_candidates) - min(
            float(candidate.metrics.path_common_free_ratio) for candidate in group_candidates
        )
        if route_span > 0.03 or common_free_span > 0.08:
            continue
        if not any(_is_edge_aligned_slot(candidate.metrics) for candidate in group_candidates):
            continue

        min_frag_blocks = min(int(candidate.metrics.fragmentation_added_blocks) for candidate in group_candidates)
        min_largest_loss = min(int(candidate.metrics.largest_block_loss_slots) for candidate in group_candidates)
        for candidate in group_candidates:
            metrics = candidate.metrics
            if (
                int(metrics.fragmentation_added_blocks) > min_frag_blocks
                or int(metrics.largest_block_loss_slots) > min_largest_loss
            ):
                continue
            if _is_edge_aligned_slot(metrics):
                overrides[int(candidate.raw_action)] = replace(
                    metrics,
                    local_fragmentation=float(min(float(metrics.local_fragmentation), 0.10)),
                    local_damage_score=float(min(float(metrics.local_damage_score), 0.05)),
                )
                continue
            if _is_split_slot(metrics):
                overrides[int(candidate.raw_action)] = replace(
                    metrics,
                    local_fragmentation=float(max(float(metrics.local_fragmentation), 0.40)),
                    local_damage_score=float(max(float(metrics.local_damage_score), 0.12)),
                )

    if not overrides:
        return tuple(candidates)
    return tuple(
        replace(candidate, metrics=overrides.get(int(candidate.raw_action), candidate.metrics))
        for candidate in candidates
    )


def _dominates(lhs: JudgeCandidate, rhs: JudgeCandidate) -> bool:
    lhs_metrics = lhs.metrics
    rhs_metrics = rhs.metrics
    if not lhs_metrics.qot_safe_now:
        return False
    if not rhs_metrics.qot_safe_now:
        return True
    lhs_qot_floor = min(lhs_metrics.qot_margin_clipped_db, 0.5)
    rhs_qot_floor = min(rhs_metrics.qot_margin_clipped_db, 0.5)
    non_worse = (
        lhs_metrics.slot_ratio_vs_best <= rhs_metrics.slot_ratio_vs_best
        and lhs_metrics.local_damage_score <= rhs_metrics.local_damage_score
        and lhs_metrics.route_pressure_score <= rhs_metrics.route_pressure_score
        and lhs_qot_floor >= rhs_qot_floor
    )
    strictly_better = (
        lhs_metrics.slot_ratio_vs_best < rhs_metrics.slot_ratio_vs_best
        or lhs_metrics.local_damage_score < rhs_metrics.local_damage_score
        or lhs_metrics.route_pressure_score < rhs_metrics.route_pressure_score
        or lhs_qot_floor > rhs_qot_floor
    )
    return non_worse and strictly_better


def score_candidates(
    candidates: Sequence[JudgeCandidate],
) -> tuple[tuple[JudgeCandidate, ...], str]:
    if not candidates:
        raise ValueError("at least one candidate is required")

    safe_candidates = [
        candidate for candidate in candidates if (not candidate.is_reject and candidate.metrics.osnr_margin_db >= 0.0)
    ]
    min_required_slots = min((candidate.metrics.required_slots for candidate in safe_candidates), default=0)
    max_frag_blocks = max((candidate.metrics.fragmentation_added_blocks for candidate in safe_candidates), default=0)
    max_largest_loss = max((candidate.metrics.largest_block_loss_slots for candidate in safe_candidates), default=0)

    min_required_slots_by_path: dict[int, int] = {}
    for candidate in safe_candidates:
        path_index = _candidate_path_index(candidate)
        if path_index is None:
            continue
        current = min_required_slots_by_path.get(path_index)
        if current is None or candidate.metrics.required_slots < current:
            min_required_slots_by_path[path_index] = candidate.metrics.required_slots

    staged: list[JudgeCandidate] = []
    for candidate in candidates:
        metrics = candidate.metrics
        qot_safe_now = (not candidate.is_reject) and metrics.osnr_margin_db >= 0.0
        qot_margin_clipped_db = _clamp(metrics.osnr_margin_db, lower=0.0, upper=3.0)
        qot_excess_db_over_floor = max(metrics.osnr_margin_db - 0.5, 0.0)
        fragmentation_added_blocks_norm = (
            0.0 if max_frag_blocks <= 0 else float(metrics.fragmentation_added_blocks) / float(max_frag_blocks)
        )
        largest_block_loss_slots_norm = (
            0.0 if max_largest_loss <= 0 else float(metrics.largest_block_loss_slots) / float(max_largest_loss)
        )
        slot_cost_vs_best = (
            0 if (not qot_safe_now or min_required_slots <= 0) else int(metrics.required_slots - min_required_slots)
        )
        slot_ratio_vs_best = (
            1.0
            if (not qot_safe_now or min_required_slots <= 0)
            else float(metrics.required_slots) / float(min_required_slots)
        )
        local_damage_score = _clamp(
            (0.40 * fragmentation_added_blocks_norm)
            + (0.35 * largest_block_loss_slots_norm)
            + (0.15 * metrics.local_fragmentation)
            + (0.10 * metrics.path_common_num_blocks_norm)
        )
        route_pressure_score = _clamp(
            (0.35 * metrics.path_link_util_max)
            + (0.25 * metrics.path_link_util_mean)
            + (0.20 * metrics.path_route_cuts_norm)
            + (0.20 * (1.0 - metrics.path_route_rss))
        )
        path_index = _candidate_path_index(candidate)
        min_required_slots_on_same_path = (
            metrics.required_slots
            if path_index is None or path_index not in min_required_slots_by_path
            else min_required_slots_by_path[path_index]
        )
        extra_slots_for_same_path = int(metrics.required_slots - min_required_slots_on_same_path)
        same_path_only_modulation_tradeoff = bool(qot_safe_now and extra_slots_for_same_path > 0)

        staged.append(
            replace(
                candidate,
                metrics=replace(
                    metrics,
                    qot_safe_now=qot_safe_now,
                    qot_band=_qot_band(metrics.osnr_margin_db),
                    qot_margin_clipped_db=float(qot_margin_clipped_db),
                    qot_excess_db_over_floor=float(qot_excess_db_over_floor),
                    fragmentation_added_blocks_norm=float(fragmentation_added_blocks_norm),
                    largest_block_loss_slots_norm=float(largest_block_loss_slots_norm),
                    slot_cost_vs_best=int(slot_cost_vs_best),
                    slot_ratio_vs_best=float(slot_ratio_vs_best),
                    local_damage_score=float(local_damage_score),
                    route_pressure_score=float(route_pressure_score),
                    same_path_only_modulation_tradeoff=same_path_only_modulation_tradeoff,
                    same_path_modulation_warning=same_path_only_modulation_tradeoff,
                    extra_slots_for_same_path=int(extra_slots_for_same_path),
                ),
            )
        )

    staged = list(_recalibrate_same_path_slot_competition(tuple(staged)))

    dominance_counts: dict[str, tuple[int, int]] = {}
    for candidate in staged:
        dominating = 0
        dominated_by_this = 0
        if candidate.metrics.qot_safe_now:
            for other in staged:
                if other.heuristic_name == candidate.heuristic_name:
                    continue
                if _dominates(other, candidate):
                    dominating += 1
                if _dominates(candidate, other):
                    dominated_by_this += 1
        dominance_counts[candidate.heuristic_name] = (dominating, dominated_by_this)

    with_dominance: list[JudgeCandidate] = []
    for candidate in staged:
        dominating, dominated_by_this = dominance_counts[candidate.heuristic_name]
        metrics = replace(
            candidate.metrics,
            is_pareto_dominated=dominating > 0,
            num_candidates_dominating_this=int(dominating),
            num_candidates_dominated_by_this=int(dominated_by_this),
        )
        with_dominance.append(replace(candidate, metrics=metrics))

    with_peer_deltas: list[JudgeCandidate] = []
    for candidate in with_dominance:
        metrics = candidate.metrics
        same_slot_peers = [
            other
            for other in with_dominance
            if other.raw_action != candidate.raw_action
            and other.metrics.qot_safe_now
            and metrics.qot_safe_now
            and other.metrics.required_slots == metrics.required_slots
        ]
        same_slots_tradeoff = bool(same_slot_peers)
        delta_route_pressure_vs_best_peer = 0.0
        delta_local_damage_vs_best_peer = 0.0
        delta_common_free_ratio_vs_best_peer = 0.0
        if same_slot_peers:
            best_peer = min(
                same_slot_peers,
                key=lambda other: (
                    float(other.metrics.route_pressure_score),
                    -float(other.metrics.path_common_free_ratio),
                    float(other.metrics.local_damage_score),
                    -float(other.metrics.qot_margin_clipped_db),
                    int(other.raw_action),
                ),
            )
            delta_route_pressure_vs_best_peer = (
                float(metrics.route_pressure_score) - float(best_peer.metrics.route_pressure_score)
            )
            delta_local_damage_vs_best_peer = (
                float(metrics.local_damage_score) - float(best_peer.metrics.local_damage_score)
            )
            delta_common_free_ratio_vs_best_peer = (
                float(metrics.path_common_free_ratio) - float(best_peer.metrics.path_common_free_ratio)
            )
        with_peer_deltas.append(
            replace(
                candidate,
                metrics=replace(
                    metrics,
                    same_slots_tradeoff=same_slots_tradeoff,
                    delta_route_pressure_vs_best_peer=float(delta_route_pressure_vs_best_peer),
                    delta_local_damage_vs_best_peer=float(delta_local_damage_vs_best_peer),
                    delta_common_free_ratio_vs_best_peer=float(delta_common_free_ratio_vs_best_peer),
                ),
            )
        )

    with_warnings: list[JudgeCandidate] = []
    for candidate in with_peer_deltas:
        metrics = candidate.metrics
        equal_slot_route_pressure_warning = bool(
            metrics.same_slots_tradeoff
            and metrics.delta_route_pressure_vs_best_peer >= 0.03
            and metrics.delta_local_damage_vs_best_peer > -0.03
        )
        future_risk_band = "high"
        if metrics.qot_safe_now:
            if (
                equal_slot_route_pressure_warning
                or metrics.route_pressure_score >= 0.23
                or metrics.local_damage_score >= 0.35
                or (metrics.same_path_modulation_warning and metrics.extra_slots_for_same_path >= 2)
            ):
                future_risk_band = "high"
            elif (
                metrics.route_pressure_score >= 0.19
                or metrics.local_damage_score >= 0.20
                or metrics.qot_margin_clipped_db < 0.75
            ):
                future_risk_band = "medium"
            else:
                future_risk_band = "low"
        with_warnings.append(
            replace(
                candidate,
                metrics=replace(
                    metrics,
                    equal_slot_route_pressure_warning=equal_slot_route_pressure_warning,
                    future_risk_band=future_risk_band,
                ),
            )
        )

    max_slot_cost = max(
        (candidate.metrics.slot_cost_vs_best for candidate in with_warnings if candidate.metrics.qot_safe_now),
        default=0,
    )
    scored_unsorted: list[JudgeCandidate] = []
    for candidate in with_warnings:
        metrics = candidate.metrics
        if metrics.qot_safe_now:
            if max_slot_cost <= 0:
                slot_efficiency_component = 1.0
            else:
                slot_efficiency_component = 1.0 - min(float(metrics.slot_cost_vs_best) / float(max_slot_cost), 1.0)
            route_pressure_component = 1.0 - float(metrics.route_pressure_score)
            local_damage_component = 1.0 - float(metrics.local_damage_score)
            common_free_component = float(metrics.path_common_free_ratio)
            qot_tiebreak_component = min(float(metrics.qot_margin_clipped_db), 1.5) / 1.5
            dominated_penalty = 0.06 if metrics.is_pareto_dominated else 0.0
            route_warning_penalty = 0.10 if metrics.equal_slot_route_pressure_warning else 0.0
            same_path_penalty = (
                0.12 if (metrics.same_path_modulation_warning and metrics.extra_slots_for_same_path >= 2) else 0.0
            )
            total_score = (
                (0.24 * slot_efficiency_component)
                + (0.34 * route_pressure_component)
                + (0.12 * local_damage_component)
                + (0.18 * common_free_component)
                + (0.12 * qot_tiebreak_component)
                - dominated_penalty
                - route_warning_penalty
                - same_path_penalty
            )
        else:
            slot_efficiency_component = 0.0
            route_pressure_component = 0.0
            local_damage_component = 0.0
            common_free_component = 0.0
            qot_tiebreak_component = 0.0
            dominated_penalty = 0.0
            route_warning_penalty = 0.0
            same_path_penalty = 0.0
            total_score = -1.0

        scores = CandidateCriterionScores(
            slot_efficiency_component=float(slot_efficiency_component),
            route_pressure_component=float(route_pressure_component),
            local_damage_component=float(local_damage_component),
            common_free_component=float(common_free_component),
            qot_tiebreak_component=float(qot_tiebreak_component),
            dominated_penalty=float(dominated_penalty),
            route_warning_penalty=float(route_warning_penalty),
            same_path_penalty=float(same_path_penalty),
            total_score=float(total_score),
        )
        scored_unsorted.append(
            replace(
                candidate,
                metrics=replace(
                    metrics,
                    plausibility_score=float(total_score),
                    blocking_proxy_score=float(total_score),
                ),
                baseline_scores=scores,
            )
        )

    ranked_safe = sorted(
        (candidate for candidate in scored_unsorted if candidate.metrics.qot_safe_now),
        key=lambda candidate: (
            float(candidate.metrics.plausibility_score),
            -int(candidate.metrics.required_slots),
            -float(candidate.metrics.route_pressure_score),
            -float(candidate.metrics.local_damage_score),
            float(candidate.metrics.qot_margin_clipped_db),
            -int(candidate.raw_action),
        ),
        reverse=True,
    )
    rank_by_name = {candidate.heuristic_name: index + 1 for index, candidate in enumerate(ranked_safe)}
    scored: list[JudgeCandidate] = [
        replace(
            candidate,
            metrics=replace(
                candidate.metrics,
                plausibility_rank=int(rank_by_name.get(candidate.heuristic_name, 0)),
            ),
        )
        for candidate in scored_unsorted
    ]

    def winner_key(candidate: JudgeCandidate) -> tuple[int, float, int, float, float, float, int]:
        assert candidate.baseline_scores is not None
        safety_tier = 0
        if candidate.metrics.qot_safe_now:
            safety_tier = 2
        elif not candidate.is_reject:
            safety_tier = 1
        return (
            safety_tier,
            float(candidate.metrics.plausibility_score),
            -int(candidate.metrics.required_slots),
            -float(candidate.metrics.route_pressure_score),
            -float(candidate.metrics.local_damage_score),
            float(candidate.metrics.qot_margin_clipped_db),
            -int(candidate.raw_action),
        )

    reference_winner = max(scored, key=winner_key).heuristic_name
    return tuple(scored), reference_winner


def _build_request_payload(context: RuntimeHeuristicContext, operational_state: OperationalState) -> RequestPayload:
    source_name, destination_name = _source_destination_names(context)
    return RequestPayload(
        services_processed=int(operational_state.services_processed),
        source=source_name,
        destination=destination_name,
        bit_rate_gbps=int(context.request.bit_rate),
    )


def _build_network_state_payload(operational_state: OperationalState) -> NetworkStatePayload:
    return NetworkStatePayload(
        episode_service_blocking_rate=float(operational_state.episode_service_blocking_rate),
        network_util_mean=float(operational_state.network_util_mean),
        network_util_max=float(operational_state.network_util_max),
        free_slots_ratio=float(operational_state.free_slots_ratio),
    )


def _decision_candidate_sort_key(candidate: JudgeDecisionCandidate) -> tuple[float, int, float, float, float, str]:
    return (
        float(candidate.metrics.plausibility_score),
        -int(candidate.metrics.required_slots),
        -float(candidate.metrics.route_pressure_score),
        -float(candidate.metrics.local_damage_score),
        float(candidate.metrics.qot_margin_clipped_db),
        str(candidate.candidate_id),
    )


def _positive_delta_band(
    value: float,
    *,
    small: float,
    material: float,
    strong: float,
) -> str:
    if value >= strong:
        return "strong"
    if value >= material:
        return "material"
    if value >= small:
        return "small"
    return "none"


def _resolve_prompt_episode_length(context: RuntimeHeuristicContext) -> int:
    config = getattr(context, "config", None)
    if config is not None:
        episode_length = int(getattr(config, "episode_length", 0) or 0)
        if episode_length > 0:
            return episode_length
    simulator = getattr(context, "simulator", None)
    if simulator is not None:
        episode_length = int(getattr(simulator, "episode_length", 0) or 0)
        if episode_length > 0:
            return episode_length
    return 0


def _resolve_congestion_band(
    *,
    operational_state: OperationalState,
    global_regimes: GlobalRegimes | None,
) -> str:
    if global_regimes is not None:
        return {
            "light": "open",
            "moderate": "building",
            "high": "tight",
            "critical": "fragile",
        }.get(str(global_regimes.load_regime), "building")
    if operational_state.free_slots_ratio > 0.70 and operational_state.network_util_mean < 0.30:
        return "open"
    if operational_state.free_slots_ratio > 0.55 and operational_state.network_util_mean < 0.45:
        return "building"
    if operational_state.free_slots_ratio > 0.35 and operational_state.network_util_mean < 0.65:
        return "tight"
    return "fragile"


def _same_slot_route_leader(candidates: Sequence[JudgeDecisionCandidate]) -> JudgeDecisionCandidate:
    return min(
        candidates,
        key=lambda candidate: (
            float(candidate.metrics.route_pressure_score),
            -float(candidate.metrics.path_common_free_ratio),
            float(candidate.metrics.local_damage_score),
            -float(candidate.metrics.qot_margin_clipped_db),
            str(candidate.candidate_id),
        ),
    )


def _same_slot_route_leader_ids(candidates: Sequence[JudgeDecisionCandidate]) -> tuple[str, ...]:
    if not candidates:
        return ()
    best_route = min(float(candidate.metrics.route_pressure_score) for candidate in candidates)
    return tuple(
        str(candidate.candidate_id)
        for candidate in candidates
        if math.isclose(float(candidate.metrics.route_pressure_score), best_route, abs_tol=1e-9)
    )


def _same_slot_preservation_leader(candidates: Sequence[JudgeDecisionCandidate]) -> JudgeDecisionCandidate:
    return min(
        candidates,
        key=lambda candidate: (
            int(candidate.metrics.fragmentation_added_blocks),
            int(candidate.metrics.largest_block_loss_slots),
            float(candidate.metrics.local_fragmentation),
            float(candidate.metrics.local_damage_score),
            round(-_slot_span_total_norm(candidate), 9),
            round(-float(candidate.metrics.common_block_length_norm), 9),
            -float(candidate.metrics.path_common_free_ratio),
            float(candidate.metrics.route_pressure_score),
            -float(candidate.metrics.qot_margin_clipped_db),
            str(candidate.candidate_id),
        ),
    )


def _same_slot_preservation_leader_ids(candidates: Sequence[JudgeDecisionCandidate]) -> tuple[str, ...]:
    if not candidates:
        return ()
    best_key = min(
        (
            int(candidate.metrics.fragmentation_added_blocks),
            int(candidate.metrics.largest_block_loss_slots),
            round(float(candidate.metrics.local_fragmentation), 9),
            round(float(candidate.metrics.local_damage_score), 9),
            round(-_slot_span_total_norm(candidate), 9),
            round(-float(candidate.metrics.common_block_length_norm), 9),
            round(-float(candidate.metrics.path_common_free_ratio), 9),
            round(float(candidate.metrics.route_pressure_score), 9),
            round(-float(candidate.metrics.qot_margin_clipped_db), 9),
        )
        for candidate in candidates
    )
    return tuple(
        str(candidate.candidate_id)
        for candidate in candidates
        if (
            int(candidate.metrics.fragmentation_added_blocks),
            int(candidate.metrics.largest_block_loss_slots),
            round(float(candidate.metrics.local_fragmentation), 9),
            round(float(candidate.metrics.local_damage_score), 9),
            round(-_slot_span_total_norm(candidate), 9),
            round(-float(candidate.metrics.common_block_length_norm), 9),
            round(-float(candidate.metrics.path_common_free_ratio), 9),
            round(float(candidate.metrics.route_pressure_score), 9),
            round(-float(candidate.metrics.qot_margin_clipped_db), 9),
        ) == best_key
    )


def _same_slot_common_free_leader(candidates: Sequence[JudgeDecisionCandidate]) -> JudgeDecisionCandidate:
    return min(
        candidates,
        key=lambda candidate: (
            -float(candidate.metrics.path_common_free_ratio),
            int(candidate.metrics.fragmentation_added_blocks),
            int(candidate.metrics.largest_block_loss_slots),
            float(candidate.metrics.local_damage_score),
            float(candidate.metrics.route_pressure_score),
            -float(candidate.metrics.qot_margin_clipped_db),
            str(candidate.candidate_id),
        ),
    )


def _same_slot_common_free_leader_ids(candidates: Sequence[JudgeDecisionCandidate]) -> tuple[str, ...]:
    if not candidates:
        return ()
    best_common_free = max(float(candidate.metrics.path_common_free_ratio) for candidate in candidates)
    return tuple(
        str(candidate.candidate_id)
        for candidate in candidates
        if math.isclose(float(candidate.metrics.path_common_free_ratio), best_common_free, abs_tol=1e-9)
    )


def _same_slot_local_damage_leader_ids(candidates: Sequence[JudgeDecisionCandidate]) -> tuple[str, ...]:
    if not candidates:
        return ()
    best_local_damage = min(float(candidate.metrics.local_damage_score) for candidate in candidates)
    return tuple(
        str(candidate.candidate_id)
        for candidate in candidates
        if math.isclose(float(candidate.metrics.local_damage_score), best_local_damage, abs_tol=1e-9)
    )


def _same_slot_qot_leader_ids(candidates: Sequence[JudgeDecisionCandidate]) -> tuple[str, ...]:
    if not candidates:
        return ()
    best_qot_margin = max(float(candidate.metrics.qot_margin_clipped_db) for candidate in candidates)
    return tuple(
        str(candidate.candidate_id)
        for candidate in candidates
        if math.isclose(float(candidate.metrics.qot_margin_clipped_db), best_qot_margin, abs_tol=1e-9)
    )


def _same_path_same_modulation(
    lhs: JudgeDecisionCandidate,
    rhs: JudgeDecisionCandidate,
) -> bool:
    lhs_route = lhs.route
    rhs_route = rhs.route
    if lhs_route is None or rhs_route is None:
        return False
    if lhs_route.path_index != rhs_route.path_index:
        return False
    if str(lhs_route.modulation_name) != str(rhs_route.modulation_name):
        return False
    if int(lhs.metrics.required_slots) != int(rhs.metrics.required_slots):
        return False
    if int(lhs_route.initial_slot) == int(rhs_route.initial_slot):
        return False
    return bool(
        math.isclose(
            float(lhs.metrics.route_pressure_score),
            float(rhs.metrics.route_pressure_score),
            abs_tol=1e-9,
        )
        and math.isclose(
            float(lhs.metrics.path_common_free_ratio),
            float(rhs.metrics.path_common_free_ratio),
            abs_tol=1e-9,
        )
    )


def _slot_span_total_norm(candidate: JudgeDecisionCandidate) -> float:
    return float(candidate.metrics.left_free_span_norm) + float(candidate.metrics.right_free_span_norm)


def _same_path_slot_variant_candidate_ids(
    candidates: Sequence[JudgeDecisionCandidate],
) -> tuple[str, ...]:
    candidate_ids: list[str] = []
    for candidate in candidates:
        has_variant_peer = any(
            peer.candidate_id != candidate.candidate_id and _same_path_same_modulation(candidate, peer)
            for peer in candidates
        )
        if has_variant_peer:
            candidate_ids.append(str(candidate.candidate_id))
    return tuple(candidate_ids)


def _same_path_slot_variant_preservation_leader_ids(
    candidates: Sequence[JudgeDecisionCandidate],
) -> tuple[str, ...]:
    groups: dict[tuple[int, str, int], list[JudgeDecisionCandidate]] = {}
    for candidate in candidates:
        route = candidate.route
        if route is None:
            continue
        group = [
            peer
            for peer in candidates
            if _same_path_same_modulation(candidate, peer) or peer.candidate_id == candidate.candidate_id
        ]
        if len(group) <= 1:
            continue
        group_key = (
            int(route.path_index),
            str(route.modulation_name),
            int(candidate.metrics.required_slots),
        )
        groups[group_key] = group
    leader_ids: list[str] = []
    for group_candidates in groups.values():
        leader = min(
            group_candidates,
            key=lambda candidate: (
                int(candidate.metrics.fragmentation_added_blocks),
                int(candidate.metrics.largest_block_loss_slots),
                round(float(candidate.metrics.local_fragmentation), 9),
                round(-float(candidate.metrics.common_block_length_norm), 9),
                round(-_slot_span_total_norm(candidate), 9),
                round(
                    -max(
                        float(candidate.metrics.left_free_span_norm),
                        float(candidate.metrics.right_free_span_norm),
                    ),
                    9,
                ),
                round(-min(float(candidate.metrics.left_free_span_norm), float(candidate.metrics.right_free_span_norm)), 9),
                round(float(candidate.metrics.local_damage_score), 9),
                round(-float(candidate.metrics.qot_margin_clipped_db), 9),
                int(candidate.route.initial_slot) if candidate.route is not None else 10**9,
                str(candidate.candidate_id),
            ),
        )
        leader_ids.append(str(leader.candidate_id))
    return tuple(dict.fromkeys(leader_ids))


def _resolve_same_path_slot_variant_band(
    candidates: Sequence[JudgeDecisionCandidate],
) -> str:
    has_variant = False
    for candidate_index, candidate in enumerate(candidates):
        for peer in candidates[candidate_index + 1 :]:
            if not _same_path_same_modulation(candidate, peer):
                continue
            has_variant = True
            if (
                abs(int(candidate.metrics.fragmentation_added_blocks) - int(peer.metrics.fragmentation_added_blocks)) >= 1
                or abs(int(candidate.metrics.largest_block_loss_slots) - int(peer.metrics.largest_block_loss_slots)) >= 1
                or abs(float(candidate.metrics.local_fragmentation) - float(peer.metrics.local_fragmentation)) >= 0.05
                or abs(float(candidate.metrics.left_free_span_norm) - float(peer.metrics.left_free_span_norm)) >= 0.05
                or abs(float(candidate.metrics.right_free_span_norm) - float(peer.metrics.right_free_span_norm)) >= 0.05
            ):
                return "material"
    return "present" if has_variant else "none"


def _resolve_same_slot_near_tie_band(
    *,
    route_gain: float,
    local_gain: float,
    common_free_penalty: float,
    qot_gap: float,
    same_slot_count: int,
) -> str:
    del qot_gap
    if same_slot_count <= 1:
        return "single_option"
    if route_gain < 0.02 and local_gain < 0.10 and common_free_penalty < 0.06:
        return "near_tie"
    if route_gain < 0.05 and local_gain < 0.16 and common_free_penalty < 0.12:
        return "soft_tradeoff"
    return "clear_tradeoff"


def _resolve_same_slot_route_common_free_alignment(
    *,
    route_leader_ids: Sequence[str],
    common_free_leader_ids: Sequence[str],
    same_slot_count: int,
) -> str:
    if same_slot_count <= 1:
        return "single_option"
    if set(str(candidate_id) for candidate_id in route_leader_ids).intersection(
        str(candidate_id) for candidate_id in common_free_leader_ids
    ):
        return "aligned"
    return "split"


def _resolve_same_slot_damage_axes_tie_band(
    candidates: Sequence[JudgeDecisionCandidate],
) -> str:
    if len(candidates) <= 1:
        return "single_option"
    structural_pairs = {
        (
            int(candidate.metrics.fragmentation_added_blocks),
            int(candidate.metrics.largest_block_loss_slots),
        )
        for candidate in candidates
    }
    if len(structural_pairs) == 1:
        return "tied"
    return "split"


def _resolve_same_slot_local_support_band(
    candidates: Sequence[JudgeDecisionCandidate],
) -> str:
    if len(candidates) <= 1:
        return "none"

    route_leader_ids = set(_same_slot_route_leader_ids(candidates))
    preservation_leader_ids = set(_same_slot_preservation_leader_ids(candidates))
    same_path_variant_preservation_leader_ids = set(_same_path_slot_variant_preservation_leader_ids(candidates))
    partial_support = False
    for candidate in candidates:
        candidate_is_local_story = (
            candidate.candidate_id in preservation_leader_ids
            or candidate.candidate_id in same_path_variant_preservation_leader_ids
        )
        if not candidate_is_local_story:
            continue
        relevant_rivals = [
            rival
            for rival in candidates
            if rival.candidate_id != candidate.candidate_id
            and (
                rival.candidate_id in route_leader_ids
                or _same_path_same_modulation(candidate, rival)
            )
        ]
        for rival in relevant_rivals:
            if rival.candidate_id == candidate.candidate_id:
                continue
            local_gain = (
                float(rival.metrics.local_damage_score)
                - float(candidate.metrics.local_damage_score)
            )
            route_penalty = (
                float(candidate.metrics.route_pressure_score)
                - float(rival.metrics.route_pressure_score)
            )
            common_free_gain = (
                float(candidate.metrics.path_common_free_ratio)
                - float(rival.metrics.path_common_free_ratio)
            )
            local_fragmentation_gain = (
                float(rival.metrics.local_fragmentation)
                - float(candidate.metrics.local_fragmentation)
            )
            slot_span_gain = _slot_span_total_norm(candidate) - _slot_span_total_norm(rival)
            common_block_gain = (
                float(candidate.metrics.common_block_length_norm)
                - float(rival.metrics.common_block_length_norm)
            )
            hard_structural_support = (
                int(candidate.metrics.fragmentation_added_blocks)
                < int(rival.metrics.fragmentation_added_blocks)
                or int(candidate.metrics.largest_block_loss_slots)
                < int(rival.metrics.largest_block_loss_slots)
            )
            same_path_structural_support = (
                _same_path_same_modulation(candidate, rival)
                and (slot_span_gain >= 0.03 or common_block_gain >= 0.02)
            )
            soft_structural_support = (
                local_fragmentation_gain >= 0.10
                or common_free_gain >= 0.02
            )
            if (
                local_gain >= 0.05
                and route_penalty <= 0.045
                and (
                    hard_structural_support
                    or same_path_structural_support
                )
            ):
                return "material"
            if (
                local_gain >= 0.03
                and route_penalty <= 0.05
                and (
                    hard_structural_support
                    or same_path_structural_support
                    or soft_structural_support
                )
            ):
                partial_support = True
    return "partial" if partial_support else "none"


def _aggregate_same_slot_future_risk_band(
    candidates: Sequence[JudgeDecisionCandidate],
) -> str:
    if not candidates:
        return "low"
    severity_by_band = {"low": 0, "medium": 1, "high": 2}
    severities = [
        severity_by_band.get(str(candidate.metrics.future_risk_band), 1)
        for candidate in candidates
    ]
    peak = max(severities)
    mean = sum(severities) / float(len(severities))
    if peak >= 2 and mean >= 1.5:
        return "high"
    if peak >= 1 or mean >= 0.75:
        return "medium"
    return "low"


def _resolve_future_feasibility_risk_band(
    *,
    progress_ratio: float,
    congestion_band: str,
    same_slot_near_tie_band: str,
    local_gain_band: str,
    common_free_penalty_band: str,
    candidate_future_risk_band: str,
) -> str:
    material_structural_cost = (
        local_gain_band in {"material", "strong"}
        or common_free_penalty_band in {"material", "strong"}
    )
    if candidate_future_risk_band == "high":
        if congestion_band in {"tight", "fragile"}:
            return "high"
        if congestion_band == "building" and progress_ratio >= 0.55:
            return "high"
        if progress_ratio >= 0.60:
            return "high"
        return "medium"
    if candidate_future_risk_band == "medium":
        if congestion_band in {"tight", "fragile"} and (
            material_structural_cost or progress_ratio >= 0.50
        ):
            return "high"
        if (
            congestion_band in {"building", "tight", "fragile"}
            or material_structural_cost
            or progress_ratio >= 0.60
        ):
            return "medium"
        return "low"
    if congestion_band == "fragile":
        if material_structural_cost or progress_ratio >= 0.45:
            return "high"
        return "medium"
    if congestion_band == "tight":
        if material_structural_cost and same_slot_near_tie_band != "clear_tradeoff":
            return "high"
        if material_structural_cost or progress_ratio >= 0.65:
            return "medium"
        return "low"
    if material_structural_cost and progress_ratio >= 0.75:
        return "medium"
    return "low"


def _build_prompt_context_payload(
    candidates: Sequence[JudgeDecisionCandidate],
    *,
    context: RuntimeHeuristicContext,
    operational_state: OperationalState,
    global_regimes: GlobalRegimes | None,
) -> PromptContextPayload | None:
    if not candidates:
        return None
    min_required_slots = min(int(candidate.metrics.required_slots) for candidate in candidates)
    same_slot_candidate_ids = tuple(
        candidate.candidate_id
        for candidate in candidates
        if int(candidate.metrics.required_slots) == min_required_slots
    )
    extra_slot_candidate_ids = tuple(
        candidate.candidate_id
        for candidate in candidates
        if int(candidate.metrics.required_slots) > min_required_slots
    )
    same_slot_candidates = tuple(
        candidate
        for candidate in candidates
        if int(candidate.metrics.required_slots) == min_required_slots
    )
    route_leader = _same_slot_route_leader(same_slot_candidates)
    preservation_leader = _same_slot_preservation_leader(same_slot_candidates)
    common_free_leader = _same_slot_common_free_leader(same_slot_candidates)
    route_leader_ids = _same_slot_route_leader_ids(same_slot_candidates)
    common_free_leader_ids = _same_slot_common_free_leader_ids(same_slot_candidates)
    route_gain = (
        float(preservation_leader.metrics.route_pressure_score)
        - float(route_leader.metrics.route_pressure_score)
    )
    local_gain = (
        float(route_leader.metrics.local_damage_score)
        - float(preservation_leader.metrics.local_damage_score)
    )
    common_free_penalty = (
        float(common_free_leader.metrics.path_common_free_ratio)
        - float(route_leader.metrics.path_common_free_ratio)
    )
    qot_gap = abs(
        float(route_leader.metrics.qot_margin_clipped_db)
        - float(preservation_leader.metrics.qot_margin_clipped_db)
    )
    episode_length = _resolve_prompt_episode_length(context)
    progress_ratio = 0.0 if episode_length <= 0 else min(
        max(float(operational_state.services_processed) / float(episode_length), 0.0),
        1.0,
    )
    congestion_band = _resolve_congestion_band(
        operational_state=operational_state,
        global_regimes=global_regimes,
    )
    candidate_future_risk_band = _aggregate_same_slot_future_risk_band(same_slot_candidates)
    route_gain_band = _positive_delta_band(route_gain, small=0.01, material=0.03, strong=0.06)
    local_gain_band = _positive_delta_band(local_gain, small=0.03, material=0.08, strong=0.14)
    common_free_penalty_band = _positive_delta_band(
        common_free_penalty,
        small=0.02,
        material=0.05,
        strong=0.10,
    )
    same_slot_near_tie_band = _resolve_same_slot_near_tie_band(
        route_gain=route_gain,
        local_gain=local_gain,
        common_free_penalty=common_free_penalty,
        qot_gap=qot_gap,
        same_slot_count=len(same_slot_candidates),
    )
    same_slot_damage_axes_tie_band = _resolve_same_slot_damage_axes_tie_band(
        same_slot_candidates,
    )
    same_slot_route_common_free_alignment = _resolve_same_slot_route_common_free_alignment(
        route_leader_ids=route_leader_ids,
        common_free_leader_ids=common_free_leader_ids,
        same_slot_count=len(same_slot_candidates),
    )
    same_path_slot_variant_band = _resolve_same_path_slot_variant_band(same_slot_candidates)
    same_slot_local_support_band = _resolve_same_slot_local_support_band(same_slot_candidates)
    return PromptContextPayload(
        min_required_slots_in_shortlist=int(min_required_slots),
        same_slot_candidate_ids=same_slot_candidate_ids,
        extra_slot_candidate_ids=extra_slot_candidate_ids,
        progress_ratio=float(progress_ratio),
        congestion_band=congestion_band,
        same_slot_near_tie_band=same_slot_near_tie_band,
        same_slot_damage_axes_tie_band=same_slot_damage_axes_tie_band,
        same_slot_route_common_free_alignment=same_slot_route_common_free_alignment,
        same_path_slot_variant_band=same_path_slot_variant_band,
        route_gain_band_vs_same_slot_best=route_gain_band,
        local_gain_band_vs_same_slot_best=local_gain_band,
        same_slot_local_support_band=same_slot_local_support_band,
        common_free_penalty_band_vs_same_slot_best=common_free_penalty_band,
        future_feasibility_risk_band=_resolve_future_feasibility_risk_band(
            progress_ratio=progress_ratio,
            congestion_band=congestion_band,
            same_slot_near_tie_band=same_slot_near_tie_band,
            local_gain_band=local_gain_band,
            common_free_penalty_band=common_free_penalty_band,
            candidate_future_risk_band=candidate_future_risk_band,
        ),
    )


def _build_pairwise_deltas(
    candidates: Sequence[JudgeDecisionCandidate],
) -> tuple[PairwiseDeltaPayload, ...]:
    pairwise: list[PairwiseDeltaPayload] = []
    for candidate_index, candidate in enumerate(candidates):
        for peer in candidates[candidate_index + 1 :]:
            pairwise.append(
                PairwiseDeltaPayload(
                    candidate_id=str(candidate.candidate_id),
                    vs_candidate_id=str(peer.candidate_id),
                    same_path_same_modulation=_same_path_same_modulation(candidate, peer),
                    delta_required_slots=int(candidate.metrics.required_slots - peer.metrics.required_slots),
                    delta_route_pressure_score=float(
                        candidate.metrics.route_pressure_score - peer.metrics.route_pressure_score
                    ),
                    delta_local_damage_score=float(
                        candidate.metrics.local_damage_score - peer.metrics.local_damage_score
                    ),
                    delta_path_common_free_ratio=float(
                        candidate.metrics.path_common_free_ratio - peer.metrics.path_common_free_ratio
                    ),
                    delta_fragmentation_added_blocks=int(
                        candidate.metrics.fragmentation_added_blocks - peer.metrics.fragmentation_added_blocks
                    ),
                    delta_largest_block_loss_slots=int(
                        candidate.metrics.largest_block_loss_slots - peer.metrics.largest_block_loss_slots
                    ),
                    delta_local_fragmentation=float(
                        candidate.metrics.local_fragmentation - peer.metrics.local_fragmentation
                    ),
                    delta_left_free_span_norm=float(
                        candidate.metrics.left_free_span_norm - peer.metrics.left_free_span_norm
                    ),
                    delta_right_free_span_norm=float(
                        candidate.metrics.right_free_span_norm - peer.metrics.right_free_span_norm
                    ),
                    delta_slot_span_total_norm=float(
                        _slot_span_total_norm(candidate) - _slot_span_total_norm(peer)
                    ),
                    delta_qot_margin_clipped_db=float(
                        candidate.metrics.qot_margin_clipped_db - peer.metrics.qot_margin_clipped_db
                    ),
                )
            )
    return tuple(pairwise)


def build_judge_payload(
    *,
    prompt_version: str,
    context: RuntimeHeuristicContext,
    topology_profile: TopologyProfile,
    operational_state: OperationalState,
    global_regimes: GlobalRegimes,
    candidates: Sequence[JudgeCandidate],
    candidate_ids: Sequence[str] | None = None,
    candidate_roles: Sequence[str] | None = None,
) -> JudgeDecisionPayload:
    del topology_profile
    if candidate_ids is not None and len(candidate_ids) != len(candidates):
        raise ValueError("candidate_ids length must match candidates length")
    if candidate_roles is not None and len(candidate_roles) != len(candidates):
        raise ValueError("candidate_roles length must match candidates length")
    if candidate_ids is None:
        candidate_ids = tuple(f"C{i + 1}" for i in range(len(candidates)))
    if candidate_roles is None:
        candidate_roles = tuple(
            "balanced_anchor" if index == 0 else "backfill_challenger"
            for index in range(len(candidates))
        )
    decision_candidates = tuple(
        candidate.to_decision_candidate(
            candidate_id=str(candidate_id),
            candidate_role="",
            candidate_roles=_payload_visible_candidate_roles((str(candidate_role),)),
        )
        for candidate_id, candidate_role, candidate in zip(candidate_ids, candidate_roles, candidates, strict=True)
    )
    decision_candidates = _annotate_candidate_roles(decision_candidates)
    prompt_context = _build_prompt_context_payload(
        decision_candidates,
        context=context,
        operational_state=operational_state,
        global_regimes=global_regimes,
    )
    decision_candidates = _filter_candidate_roles_for_prompt_context(
        decision_candidates,
        prompt_context=prompt_context,
    )
    return JudgeDecisionPayload(
        prompt_version=str(prompt_version),
        request=_build_request_payload(context, operational_state),
        network_state=_build_network_state_payload(operational_state),
        candidates=decision_candidates,
        pairwise_deltas=_build_pairwise_deltas(decision_candidates),
        prompt_context=prompt_context,
    )


def _annotate_candidate_roles(
    candidates: Sequence[JudgeDecisionCandidate],
) -> tuple[JudgeDecisionCandidate, ...]:
    if not candidates:
        return ()

    min_required_slots = min(int(candidate.metrics.required_slots) for candidate in candidates)
    same_slot_candidates = tuple(
        candidate
        for candidate in candidates
        if int(candidate.metrics.required_slots) == min_required_slots
    )
    route_leader_ids = set(_same_slot_route_leader_ids(same_slot_candidates))
    preservation_leader_ids = set(_same_slot_preservation_leader_ids(same_slot_candidates))
    common_free_leader_ids = set(_same_slot_common_free_leader_ids(same_slot_candidates))
    local_damage_leader_ids = set(_same_slot_local_damage_leader_ids(same_slot_candidates))
    same_path_slot_variant_ids = set(_same_path_slot_variant_candidate_ids(same_slot_candidates))
    same_path_slot_variant_preservation_leader_ids = set(
        _same_path_slot_variant_preservation_leader_ids(same_slot_candidates)
    )

    annotated: list[JudgeDecisionCandidate] = []
    for candidate in candidates:
        roles: list[str] = []
        for role in candidate.candidate_roles:
            role_name = str(role)
            if role_name and role_name not in roles:
                roles.append(role_name)
        if int(candidate.metrics.required_slots) == min_required_slots:
            roles.append("same_slot_candidate")
        else:
            roles.append("extra_slot_candidate")
        if candidate.candidate_id in route_leader_ids:
            roles.append("same_slot_route_leader")
        if candidate.candidate_id in preservation_leader_ids:
            roles.append("same_slot_preservation_leader")
        if candidate.candidate_id in common_free_leader_ids:
            roles.append("same_slot_common_free_leader")
        if candidate.candidate_id in local_damage_leader_ids:
            roles.append("same_slot_local_damage_leader")
        if candidate.candidate_id in same_path_slot_variant_ids:
            roles.append("same_path_slot_variant_candidate")
        if candidate.candidate_id in same_path_slot_variant_preservation_leader_ids:
            roles.append("same_path_slot_variant_preservation_leader")
        deduped_roles = tuple(dict.fromkeys(roles))
        annotated.append(replace(candidate, candidate_roles=deduped_roles))
    return tuple(annotated)


def _filter_candidate_roles_for_prompt_context(
    candidates: Sequence[JudgeDecisionCandidate],
    *,
    prompt_context: PromptContextPayload | None,
) -> tuple[JudgeDecisionCandidate, ...]:
    if prompt_context is None:
        return tuple(candidates)
    local_support_band = str(prompt_context.same_slot_local_support_band)
    damage_axes_tie_band = str(prompt_context.same_slot_damage_axes_tie_band)

    filtered: list[JudgeDecisionCandidate] = []
    for candidate in candidates:
        roles = list(candidate.candidate_roles)
        if local_support_band in {"none", "partial"}:
            roles = [
                role
                for role in roles
                if role not in {"same_slot_preservation_leader", "same_slot_local_damage_leader"}
            ]
        elif damage_axes_tie_band == "tied":
            roles = [role for role in roles if role != "same_slot_local_damage_leader"]
        filtered.append(replace(candidate, candidate_roles=tuple(dict.fromkeys(roles))))
    return tuple(filtered)


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
    parsed_response: object | None,
    fallback_reason: str,
    judge_error_message: str,
    semantic_warning_flags: Sequence[str] = (),
    basis_vs_payload_mismatch: bool = False,
    pre_shuffle_shortlist_actions: Sequence[int] = (),
    post_shuffle_shortlist_actions: Sequence[int] = (),
    prompt_permutation: Sequence[int] = (),
    hidden_balanced_candidate_id: str = "",
    hidden_balanced_candidate_action: int = -1,
    hidden_balanced_candidate_heuristic: str = "",
    candidates: Sequence[JudgeCandidate],
    reference_winner: str,
    chosen_action: int,
    chosen_heuristic: str,
    winner_proposed_by: Sequence[str],
    controller_decision_source: str,
    raw_candidate_count: int,
    surviving_candidate_count: int,
    pruned_dominated_count: int,
    prompt_candidate_count: int,
) -> JudgeAuditRecord:
    baseline_scores_by_candidate = {
        candidate.heuristic_name: (
            candidate.baseline_scores
            if candidate.baseline_scores is not None
            else CandidateCriterionScores(
                slot_efficiency_component=0.0,
                route_pressure_component=0.0,
                local_damage_component=0.0,
                common_free_component=0.0,
                qot_tiebreak_component=0.0,
                dominated_penalty=0.0,
                route_warning_penalty=0.0,
                same_path_penalty=0.0,
                total_score=float(candidate.metrics.plausibility_score),
            )
        )
        for candidate in candidates
    }
    audit = JudgeAuditSection(
        date=str(date),
        prompt_version=str(prompt_version),
        seed=int(seed),
        episode_index=int(episode_index),
        step_index=int(step_index),
        topology_id=str(topology_id),
        candidate_audit=tuple(candidates),
        baseline_scores_by_candidate=baseline_scores_by_candidate,
        reference_winner=str(reference_winner),
        agrees_with_reference=str(chosen_heuristic) == str(reference_winner),
        chosen_action=int(chosen_action),
        chosen_heuristic=str(chosen_heuristic),
        winner_proposed_by=tuple(str(name) for name in winner_proposed_by),
        controller_decision_source=str(controller_decision_source),
        raw_candidate_count=int(raw_candidate_count),
        surviving_candidate_count=int(surviving_candidate_count),
        pruned_dominated_count=int(pruned_dominated_count),
        prompt_candidate_count=int(prompt_candidate_count),
        pre_shuffle_shortlist_actions=tuple(int(action) for action in pre_shuffle_shortlist_actions),
        post_shuffle_shortlist_actions=tuple(int(action) for action in post_shuffle_shortlist_actions),
        prompt_permutation=tuple(int(index) for index in prompt_permutation),
        hidden_balanced_candidate_id=str(hidden_balanced_candidate_id),
        hidden_balanced_candidate_action=int(hidden_balanced_candidate_action),
        hidden_balanced_candidate_heuristic=str(hidden_balanced_candidate_heuristic),
    )
    model_io = JudgeModelIORecord(
        raw_model_response=raw_model_response,
        parsed_response=parsed_response,
        fallback_reason=str(fallback_reason),
        judge_error_message=str(judge_error_message),
        semantic_warning_flags=tuple(str(flag) for flag in semantic_warning_flags),
        basis_vs_payload_mismatch=bool(basis_vs_payload_mismatch),
    )
    return JudgeAuditRecord(
        audit=audit,
        decision_payload=decision_payload,
        prompt=prompt,
        model_io=model_io,
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
    "NetworkStatePayload",
    "OperationalState",
    "PairwiseDeltaPayload",
    "PromptContextPayload",
    "RequestPayload",
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

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from optical_networking_gym_v2 import make_env, set_topology_dir
from optical_networking_gym_v2.heuristics import (
    select_first_fit_runtime_action,
    select_highest_snr_first_fit_runtime_action,
    select_load_balancing_runtime_action,
)
from optical_networking_gym_v2.judge.heuristic_judge import (
    CandidateMetricsPayload,
    DecodedActionPayload,
    JudgeCandidate,
    JudgePromptRecord,
    build_global_regimes,
    build_judge_audit_record,
    build_judge_candidate,
    build_judge_payload,
    build_operational_state,
    build_topology_profile,
    score_candidates,
)
from optical_networking_gym_v2.judge.ollama import (
    _build_verdict_from_response_mapping,
    _extract_json,
    build_ollama_prompt_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_DIR = PROJECT_ROOT.parent / "examples" / "topologies"


def _build_env():
    set_topology_dir(TOPOLOGY_DIR)
    env = make_env(
        topology_name="ring_4",
        modulation_names="QPSK,16QAM",
        seed=7,
        bit_rates=(40,),
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        episode_length=6,
        modulations_to_consider=2,
        k_paths=2,
    )
    observation, info = env.reset(seed=7)
    return env, observation, info


def _manual_candidate(
    *,
    heuristic_name: str,
    raw_action: int,
    path_index: int,
    modulation_name: str,
    required_slots: int,
    osnr_margin_db: float,
    fragmentation_added_blocks: int,
    largest_block_loss_slots: int,
    local_fragmentation: float,
    path_common_num_blocks_norm: float,
    path_route_cuts_norm: float,
    path_route_rss: float,
    path_link_util_mean: float,
    path_link_util_max: float,
    path_common_free_ratio: float,
    common_block_length_norm: float = 0.4,
    left_free_span_norm: float = 0.1,
    right_free_span_norm: float = 0.2,
    initial_slot: int = 0,
) -> JudgeCandidate:
    return JudgeCandidate(
        heuristic_name=heuristic_name,
        proposed_by=(heuristic_name,),
        raw_action=raw_action,
        is_reject=False,
        decoded_action=DecodedActionPayload(
            path_index=path_index,
            path_rank_k=path_index,
            path_node_names=("A", "B"),
            path_hops=1,
            path_length_km=100.0,
            source_name="A",
            destination_name="B",
            modulation_index=0,
            modulation_name=modulation_name,
            modulation_spectral_efficiency=1,
            initial_slot=initial_slot,
            required_slots=required_slots,
            slot_end_exclusive=initial_slot + required_slots,
        ),
        metrics=CandidateMetricsPayload(
            required_slots=required_slots,
            path_link_util_mean=path_link_util_mean,
            path_link_util_max=path_link_util_max,
            path_common_free_ratio=path_common_free_ratio,
            path_common_largest_block_ratio=0.2,
            path_common_num_blocks_norm=path_common_num_blocks_norm,
            path_route_cuts_norm=path_route_cuts_norm,
            path_route_rss=path_route_rss,
            osnr_margin_db=osnr_margin_db,
            nli_share=0.2,
            worst_link_nli_share=0.2,
            common_block_length_norm=common_block_length_norm,
            left_free_span_norm=left_free_span_norm,
            right_free_span_norm=right_free_span_norm,
            local_fragmentation=local_fragmentation,
            fragmentation_damage_num_blocks=0.0,
            fragmentation_damage_largest_block=0.0,
            fragmentation_added_blocks=fragmentation_added_blocks,
            largest_block_loss_slots=largest_block_loss_slots,
        ),
    )


def test_build_judge_candidate_preserves_audit_route_fields() -> None:
    env, _observation, info = _build_env()
    context = env.heuristic_context()
    action = select_first_fit_runtime_action(context)

    candidate = build_judge_candidate(
        context=context,
        heuristic_name="first_fit",
        action=action,
    )

    assert candidate.raw_action == action
    assert candidate.decoded_action is not None
    assert candidate.decoded_action.path_index >= 0
    assert candidate.decoded_action.modulation_name in {"QPSK", "16QAM"}
    assert candidate.decoded_action.initial_slot >= 0
    assert candidate.metrics.required_slots > 0
    assert candidate.metrics.fragmentation_added_blocks >= 0
    assert candidate.metrics.largest_block_loss_slots >= 0
    assert "mask" in info


def test_build_judge_candidate_uses_null_decoded_action_for_reject() -> None:
    env, _observation, _info = _build_env()
    context = env.heuristic_context()

    candidate = build_judge_candidate(
        context=context,
        heuristic_name="reject",
        action=context.reject_action,
    )

    assert candidate.raw_action == context.reject_action
    assert candidate.decoded_action is None


def test_score_candidates_derives_v5_metrics_and_reference_winner() -> None:
    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=10,
            path_index=0,
            modulation_name="16QAM",
            required_slots=2,
            osnr_margin_db=1.2,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=1,
            local_fragmentation=0.2,
            path_common_num_blocks_norm=0.1,
            path_route_cuts_norm=0.2,
            path_route_rss=0.8,
            path_link_util_mean=0.3,
            path_link_util_max=0.4,
            path_common_free_ratio=0.5,
        ),
        _manual_candidate(
            heuristic_name="highest_snr_first_fit",
            raw_action=11,
            path_index=0,
            modulation_name="BPSK",
            required_slots=6,
            osnr_margin_db=7.5,
            fragmentation_added_blocks=2,
            largest_block_loss_slots=5,
            local_fragmentation=0.6,
            path_common_num_blocks_norm=0.2,
            path_route_cuts_norm=0.3,
            path_route_rss=0.7,
            path_link_util_mean=0.3,
            path_link_util_max=0.4,
            path_common_free_ratio=0.5,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=12,
            path_index=1,
            modulation_name="16QAM",
            required_slots=2,
            osnr_margin_db=0.8,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
            local_fragmentation=0.1,
            path_common_num_blocks_norm=0.05,
            path_route_cuts_norm=0.1,
            path_route_rss=0.9,
            path_link_util_mean=0.2,
            path_link_util_max=0.25,
            path_common_free_ratio=0.7,
        ),
    )

    scored_candidates, reference_winner = score_candidates(candidates)
    by_name = {candidate.heuristic_name: candidate for candidate in scored_candidates}

    assert reference_winner == "load_balancing"
    assert by_name["first_fit"].metrics.qot_safe_now is True
    assert by_name["first_fit"].metrics.qot_band == "moderate"
    assert by_name["highest_snr_first_fit"].metrics.qot_band == "strong"
    assert by_name["highest_snr_first_fit"].metrics.qot_margin_clipped_db == pytest.approx(3.0)
    assert by_name["highest_snr_first_fit"].metrics.slot_cost_vs_best == 4
    assert by_name["highest_snr_first_fit"].metrics.slot_ratio_vs_best == pytest.approx(3.0)
    assert by_name["highest_snr_first_fit"].metrics.same_path_only_modulation_tradeoff is True
    assert by_name["highest_snr_first_fit"].metrics.same_path_modulation_warning is True
    assert by_name["highest_snr_first_fit"].metrics.extra_slots_for_same_path == 4
    assert by_name["highest_snr_first_fit"].metrics.is_pareto_dominated is True
    assert by_name["load_balancing"].metrics.is_pareto_dominated is False
    assert by_name["first_fit"].metrics.equal_slot_route_pressure_warning is True
    assert by_name["load_balancing"].metrics.equal_slot_route_pressure_warning is False
    assert by_name["first_fit"].metrics.same_slots_tradeoff is True
    assert by_name["load_balancing"].metrics.same_slots_tradeoff is True
    assert by_name["first_fit"].metrics.delta_route_pressure_vs_best_peer > 0.1
    assert by_name["first_fit"].metrics.delta_local_damage_vs_best_peer > 0.05
    assert by_name["first_fit"].metrics.delta_common_free_ratio_vs_best_peer < 0.0
    assert by_name["load_balancing"].metrics.future_risk_band == "low"
    assert by_name["highest_snr_first_fit"].metrics.future_risk_band == "high"
    assert by_name["load_balancing"].metrics.plausibility_rank == 1
    assert by_name["load_balancing"].metrics.plausibility_score > by_name["first_fit"].metrics.plausibility_score


def test_score_candidates_recalibrates_same_path_edge_vs_split_slot_competition() -> None:
    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=10,
            path_index=0,
            modulation_name="16QAM",
            required_slots=7,
            osnr_margin_db=3.0,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
            local_fragmentation=0.4,
            path_common_num_blocks_norm=0.131,
            path_route_cuts_norm=0.18,
            path_route_rss=0.82,
            path_link_util_mean=0.24,
            path_link_util_max=0.30,
            path_common_free_ratio=0.61,
            left_free_span_norm=0.0,
            right_free_span_norm=0.022,
            initial_slot=61,
        ),
        _manual_candidate(
            heuristic_name="ksp_best_mod_last_fit",
            raw_action=11,
            path_index=0,
            modulation_name="16QAM",
            required_slots=7,
            osnr_margin_db=3.0,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
            local_fragmentation=0.4,
            path_common_num_blocks_norm=0.131,
            path_route_cuts_norm=0.18,
            path_route_rss=0.82,
            path_link_util_mean=0.24,
            path_link_util_max=0.30,
            path_common_free_ratio=0.61,
            left_free_span_norm=0.003,
            right_free_span_norm=0.019,
            initial_slot=313,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=12,
            path_index=0,
            modulation_name="16QAM",
            required_slots=7,
            osnr_margin_db=3.0,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
            local_fragmentation=0.0,
            path_common_num_blocks_norm=0.131,
            path_route_cuts_norm=0.18,
            path_route_rss=0.82,
            path_link_util_mean=0.24,
            path_link_util_max=0.30,
            path_common_free_ratio=0.61,
            left_free_span_norm=0.006,
            right_free_span_norm=0.022,
            initial_slot=63,
        ),
    )

    scored_candidates, reference_winner = score_candidates(candidates)
    by_name = {candidate.heuristic_name: candidate for candidate in scored_candidates}

    assert by_name["first_fit"].metrics.local_fragmentation == pytest.approx(0.1)
    assert by_name["first_fit"].metrics.local_damage_score == pytest.approx(0.05)
    assert by_name["lowest_fragmentation"].metrics.local_fragmentation == pytest.approx(0.4)
    assert by_name["lowest_fragmentation"].metrics.local_damage_score == pytest.approx(0.12)
    assert reference_winner == "first_fit"


def test_build_judge_payload_serializes_v5_prompt_shape_without_route_or_pairwise() -> None:
    env, _observation, info = _build_env()
    context = env.heuristic_context()
    candidates = []
    for heuristic_name, action, proposed_by in (
        ("first_fit", select_first_fit_runtime_action(context), ("first_fit", "load_balancing")),
        ("load_balancing", select_load_balancing_runtime_action(context), ("load_balancing",)),
        ("highest_snr_first_fit", select_highest_snr_first_fit_runtime_action(context), ("highest_snr_first_fit",)),
    ):
        candidates.append(
            replace(
                build_judge_candidate(
                    context=context,
                    heuristic_name=heuristic_name,
                    action=action,
                ),
                proposed_by=proposed_by,
            )
        )
    scored_candidates, _reference_winner = score_candidates(tuple(candidates))
    operational_state = build_operational_state(context=context, info=info)
    payload = build_judge_payload(
        prompt_version="v5_contextual_judge",
        context=context,
        topology_profile=build_topology_profile(context.topology),
        operational_state=operational_state,
        global_regimes=build_global_regimes(operational_state),
        candidates=scored_candidates,
        candidate_ids=("Q7M2", "R8K4", "T9W6"),
    )

    mapping = payload.to_prompt_mapping()
    candidate_mapping = mapping["candidates"][0]

    assert mapping["prompt_version"] == "v5_contextual_judge"
    assert "request" in mapping
    assert "network_state" in mapping
    assert "candidates" in mapping
    assert "pairwise_deltas" in mapping
    assert len(mapping["pairwise_deltas"]) == 3
    assert "prompt_context" in mapping
    assert candidate_mapping["candidate_id"] == "Q7M2"
    assert "route" in candidate_mapping
    assert set(candidate_mapping["route"]) == {
        "path_index",
        "path_hops",
        "path_length_km",
        "modulation_name",
        "initial_slot",
        "required_slots",
    }
    assert "heuristic_name" not in candidate_mapping
    assert "qot_safe_now" in candidate_mapping["metrics"]
    assert "qot_band" in candidate_mapping["metrics"]
    assert "slot_cost_vs_best" in candidate_mapping["metrics"]
    assert "local_damage_score" in candidate_mapping["metrics"]
    assert "route_pressure_score" in candidate_mapping["metrics"]
    assert "is_pareto_dominated" in candidate_mapping["metrics"]
    assert "equal_slot_route_pressure_warning" in candidate_mapping["metrics"]
    assert "same_path_modulation_warning" in candidate_mapping["metrics"]
    assert "same_slots_tradeoff" in candidate_mapping["metrics"]
    assert "delta_route_pressure_vs_best_peer" in candidate_mapping["metrics"]
    assert "delta_local_damage_vs_best_peer" in candidate_mapping["metrics"]
    assert "delta_common_free_ratio_vs_best_peer" in candidate_mapping["metrics"]
    assert "future_risk_band" in candidate_mapping["metrics"]
    assert "path_link_util_mean" in candidate_mapping["metrics"]
    assert "path_link_util_max" in candidate_mapping["metrics"]
    assert "path_route_cuts_norm" in candidate_mapping["metrics"]
    assert "path_route_rss" in candidate_mapping["metrics"]
    assert "local_fragmentation" in candidate_mapping["metrics"]
    assert "fragmentation_added_blocks" in candidate_mapping["metrics"]
    assert "largest_block_loss_slots" in candidate_mapping["metrics"]
    assert "path_common_num_blocks_norm" in candidate_mapping["metrics"]
    assert "common_block_length_norm" in candidate_mapping["metrics"]
    assert "support_count" in candidate_mapping["metrics"]
    assert "has_multi_heuristic_support" in candidate_mapping["metrics"]
    assert candidate_mapping["metrics"]["support_count"] >= 1
    assert mapping["prompt_context"]["min_required_slots_in_shortlist"] >= 1
    assert "same_slot_candidate_ids" in mapping["prompt_context"]
    assert "extra_slot_candidate_ids" in mapping["prompt_context"]
    assert "candidate_role" not in candidate_mapping
    assert "plausibility_score" not in candidate_mapping["metrics"]
    assert "plausibility_rank" not in candidate_mapping["metrics"]
    assert "balanced_candidate_id" not in mapping["prompt_context"]
    assert "candidate_roles" not in mapping["prompt_context"]
    assert "delta_required_slots" in mapping["pairwise_deltas"][0]
    assert "delta_route_pressure_score" in mapping["pairwise_deltas"][0]
    assert "delta_local_damage_score" in mapping["pairwise_deltas"][0]
    assert "delta_path_common_free_ratio" in mapping["pairwise_deltas"][0]
    assert "delta_qot_margin_clipped_db" in mapping["pairwise_deltas"][0]


def test_build_judge_audit_record_preserves_full_trace_and_shuffle_audit() -> None:
    env, _observation, info = _build_env()
    context = env.heuristic_context()
    action = select_first_fit_runtime_action(context)
    candidate = build_judge_candidate(
        context=context,
        heuristic_name="first_fit",
        action=action,
    )
    scored_candidates, reference_winner = score_candidates((candidate,))
    operational_state = build_operational_state(context=context, info=info)
    payload = build_judge_payload(
        prompt_version="v5_contextual_judge",
        context=context,
        topology_profile=build_topology_profile(context.topology),
        operational_state=operational_state,
        global_regimes=build_global_regimes(operational_state),
        candidates=scored_candidates,
        candidate_ids=("J4P8",),
    )
    audit_record = build_judge_audit_record(
        date="23-03",
        prompt_version="v5_contextual_judge",
        seed=7,
        episode_index=0,
        step_index=0,
        topology_id="ring_4",
        decision_payload=payload,
        prompt=JudgePromptRecord(system_prompt="sys", user_prompt="usr"),
        raw_model_response={"content": "{\"winner_candidate_id\":\"J4P8\",\"confidence\":0.91}"},
        parsed_response={"winner_candidate_id": "J4P8", "decision_basis": "balanced_tie_break"},
        fallback_reason="none",
        judge_error_message="",
        candidates=scored_candidates,
        reference_winner=reference_winner,
        chosen_action=action,
        chosen_heuristic="first_fit",
        winner_proposed_by=("first_fit",),
        controller_decision_source="bypass_consensus",
        raw_candidate_count=1,
        surviving_candidate_count=1,
        pruned_dominated_count=0,
        prompt_candidate_count=1,
        pre_shuffle_shortlist_actions=(int(action),),
        post_shuffle_shortlist_actions=(int(action),),
        prompt_permutation=(0,),
        hidden_balanced_candidate_id="J4P8",
        hidden_balanced_candidate_action=int(action),
        hidden_balanced_candidate_heuristic="first_fit",
    )

    mapping = audit_record.to_mapping()

    assert mapping["audit"]["prompt_version"] == "v5_contextual_judge"
    assert mapping["audit"]["reference_winner"] == reference_winner
    assert mapping["audit"]["baseline_winner"] == reference_winner
    assert mapping["audit"]["controller_decision_source"] == "bypass_consensus"
    assert mapping["audit"]["raw_candidate_count"] == 1
    assert mapping["audit"]["surviving_candidate_count"] == 1
    assert mapping["audit"]["pruned_dominated_count"] == 0
    assert mapping["audit"]["prompt_candidate_count"] == 1
    assert "decoded_action" in mapping["audit"]["candidate_audit"][0]
    assert "plausibility_score" in mapping["audit"]["candidate_audit"][0]["metrics"]
    assert mapping["audit"]["pre_shuffle_shortlist_actions"] == [int(action)]
    assert mapping["audit"]["post_shuffle_shortlist_actions"] == [int(action)]
    assert mapping["audit"]["prompt_permutation"] == [0]
    assert mapping["audit"]["hidden_balanced_candidate_id"] == "J4P8"
    assert mapping["audit"]["hidden_balanced_candidate_action"] == int(action)
    assert mapping["audit"]["hidden_balanced_candidate_heuristic"] == "first_fit"
    assert mapping["model_io"]["parsed_response"]["winner_candidate_id"] == "J4P8"
    assert mapping["model_io"]["parsed_response"]["decision_basis"] == "balanced_tie_break"


def test_build_judge_payload_pairwise_deltas_follow_candidate_metrics() -> None:
    env, _observation, info = _build_env()
    context = env.heuristic_context()
    operational_state = build_operational_state(context=context, info=info)
    candidates = (
        replace(
            _manual_candidate(
                heuristic_name="first_fit",
                raw_action=10,
                path_index=0,
                modulation_name="16QAM",
                required_slots=1,
                osnr_margin_db=1.4,
                fragmentation_added_blocks=1,
                largest_block_loss_slots=2,
                local_fragmentation=0.4,
                path_common_num_blocks_norm=0.2,
                path_route_cuts_norm=0.3,
                path_route_rss=0.7,
                path_link_util_mean=0.2,
                path_link_util_max=0.25,
                path_common_free_ratio=0.45,
            ),
            proposed_by=("first_fit", "load_balancing"),
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=11,
            path_index=1,
            modulation_name="16QAM",
            required_slots=1,
            osnr_margin_db=1.0,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
            local_fragmentation=0.1,
            path_common_num_blocks_norm=0.1,
            path_route_cuts_norm=0.1,
            path_route_rss=0.9,
            path_link_util_mean=0.1,
            path_link_util_max=0.15,
            path_common_free_ratio=0.6,
        ),
    )
    scored_candidates, _reference_winner = score_candidates(candidates)
    payload = build_judge_payload(
        prompt_version="v5_contextual_judge",
        context=context,
        topology_profile=build_topology_profile(context.topology),
        operational_state=operational_state,
        global_regimes=build_global_regimes(operational_state),
        candidates=scored_candidates,
        candidate_ids=("CAND", "PEER"),
    )

    mapping = payload.to_prompt_mapping()
    pair = mapping["pairwise_deltas"][0]
    first_metrics = mapping["candidates"][0]["metrics"]
    second_metrics = mapping["candidates"][1]["metrics"]

    assert pair["candidate_id"] == "CAND"
    assert pair["vs_candidate_id"] == "PEER"
    assert pair["delta_required_slots"] == first_metrics["required_slots"] - second_metrics["required_slots"]
    assert pair["delta_route_pressure_score"] == pytest.approx(
        first_metrics["route_pressure_score"] - second_metrics["route_pressure_score"]
    )
    assert pair["delta_local_damage_score"] == pytest.approx(
        first_metrics["local_damage_score"] - second_metrics["local_damage_score"]
    )
    assert pair["delta_path_common_free_ratio"] == pytest.approx(
        first_metrics["path_common_free_ratio"] - second_metrics["path_common_free_ratio"]
    )
    assert pair["delta_qot_margin_clipped_db"] == pytest.approx(
        first_metrics["qot_margin_clipped_db"] - second_metrics["qot_margin_clipped_db"]
    )


def test_build_judge_payload_emits_all_pairs_for_four_candidates() -> None:
    env, _observation, info = _build_env()
    context = env.heuristic_context()
    operational_state = build_operational_state(context=context, info=info)
    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=10,
            path_index=0,
            modulation_name="16QAM",
            required_slots=1,
            osnr_margin_db=1.4,
            fragmentation_added_blocks=1,
            largest_block_loss_slots=2,
            local_fragmentation=0.4,
            path_common_num_blocks_norm=0.2,
            path_route_cuts_norm=0.3,
            path_route_rss=0.7,
            path_link_util_mean=0.2,
            path_link_util_max=0.25,
            path_common_free_ratio=0.45,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=11,
            path_index=1,
            modulation_name="16QAM",
            required_slots=1,
            osnr_margin_db=1.0,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
            local_fragmentation=0.1,
            path_common_num_blocks_norm=0.1,
            path_route_cuts_norm=0.1,
            path_route_rss=0.9,
            path_link_util_mean=0.1,
            path_link_util_max=0.15,
            path_common_free_ratio=0.6,
        ),
        _manual_candidate(
            heuristic_name="ksp_best_mod_last_fit",
            raw_action=12,
            path_index=0,
            modulation_name="16QAM",
            required_slots=1,
            osnr_margin_db=1.2,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=1,
            local_fragmentation=0.2,
            path_common_num_blocks_norm=0.1,
            path_route_cuts_norm=0.2,
            path_route_rss=0.8,
            path_link_util_mean=0.2,
            path_link_util_max=0.2,
            path_common_free_ratio=0.5,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=13,
            path_index=1,
            modulation_name="8QAM",
            required_slots=2,
            osnr_margin_db=2.0,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
            local_fragmentation=0.05,
            path_common_num_blocks_norm=0.1,
            path_route_cuts_norm=0.1,
            path_route_rss=0.95,
            path_link_util_mean=0.1,
            path_link_util_max=0.12,
            path_common_free_ratio=0.58,
        ),
    )
    scored_candidates, _reference_winner = score_candidates(candidates)
    payload = build_judge_payload(
        prompt_version="v5_contextual_judge",
        context=context,
        topology_profile=build_topology_profile(context.topology),
        operational_state=operational_state,
        global_regimes=build_global_regimes(operational_state),
        candidates=scored_candidates,
        candidate_ids=("A1", "B2", "C3", "D4"),
        candidate_roles=(
            "balanced_anchor",
            "same_slot_route_challenger",
            "same_slot_local_challenger",
            "extra_slot_structural_challenger",
        ),
    )

    mapping = payload.to_prompt_mapping()

    assert len(mapping["pairwise_deltas"]) == 6
    assert "candidate_roles" not in mapping["prompt_context"]
    assert "balanced_candidate_id" not in mapping["prompt_context"]


def test_build_ollama_prompt_record_uses_two_stage_judge_instructions() -> None:
    env, _observation, info = _build_env()
    context = env.heuristic_context()
    operational_state = build_operational_state(context=context, info=info)
    candidates = (
        _manual_candidate(
            heuristic_name="first_fit",
            raw_action=10,
            path_index=0,
            modulation_name="16QAM",
            required_slots=1,
            osnr_margin_db=1.2,
            fragmentation_added_blocks=1,
            largest_block_loss_slots=2,
            local_fragmentation=0.3,
            path_common_num_blocks_norm=0.2,
            path_route_cuts_norm=0.3,
            path_route_rss=0.7,
            path_link_util_mean=0.2,
            path_link_util_max=0.25,
            path_common_free_ratio=0.45,
        ),
        _manual_candidate(
            heuristic_name="load_balancing",
            raw_action=11,
            path_index=1,
            modulation_name="16QAM",
            required_slots=1,
            osnr_margin_db=0.9,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
            local_fragmentation=0.1,
            path_common_num_blocks_norm=0.1,
            path_route_cuts_norm=0.1,
            path_route_rss=0.9,
            path_link_util_mean=0.1,
            path_link_util_max=0.15,
            path_common_free_ratio=0.6,
        ),
        _manual_candidate(
            heuristic_name="lowest_fragmentation",
            raw_action=12,
            path_index=1,
            modulation_name="8QAM",
            required_slots=2,
            osnr_margin_db=2.1,
            fragmentation_added_blocks=0,
            largest_block_loss_slots=0,
            local_fragmentation=0.05,
            path_common_num_blocks_norm=0.1,
            path_route_cuts_norm=0.1,
            path_route_rss=0.95,
            path_link_util_mean=0.1,
            path_link_util_max=0.12,
            path_common_free_ratio=0.58,
        ),
    )
    scored_candidates, _reference_winner = score_candidates(candidates)
    payload = build_judge_payload(
        prompt_version="v5_contextual_judge",
        context=context,
        topology_profile=build_topology_profile(context.topology),
        operational_state=operational_state,
        global_regimes=build_global_regimes(operational_state),
        candidates=scored_candidates,
        candidate_ids=("A1", "B2", "C3"),
    )

    prompt = build_ollama_prompt_record(payload)

    assert "same_slot_winner_candidate_id" in prompt.system_prompt
    assert "extra_slot_override" in prompt.system_prompt
    assert "required_slots lower is better" in prompt.system_prompt
    assert "route_pressure_score lower is better" in prompt.system_prompt
    assert "Use same_slot_route_advantage only when a minimum-slot route/common-free story is genuinely stronger" in prompt.system_prompt
    assert "Use same_slot_local_advantage only when structural preservation clearly separates the winner" in prompt.system_prompt
    assert "decision_basis must name the dominant reason only" in prompt.system_prompt
    assert "<example name=\"A\">" in prompt.system_prompt
    assert "<example name=\"B\">" in prompt.system_prompt
    assert "<example name=\"C\">" in prompt.system_prompt
    assert "balanced_anchor" not in prompt.system_prompt
    assert "plausibility_score" not in prompt.user_prompt
    assert "plausibility_rank" not in prompt.user_prompt


def test_build_verdict_from_response_mapping_accepts_minimal_candidate_id_schema() -> None:
    response = {
        "same_slot_winner_candidate_id": "B9X4",
        "extra_slot_override": False,
        "winner_candidate_id": "B9X4",
        "confidence": 0.87,
        "decision_basis": "balanced_tie_break",
        "ranking": ["B9X4", "A1K2"],
        "decisive_signals": ["support_count:B9X4:low"],
    }

    verdict = _build_verdict_from_response_mapping(
        response,
        candidate_names={"A1K2", "B9X4"},
    )

    assert verdict.winner_candidate_id == "B9X4"
    assert verdict.confidence == pytest.approx(0.87)
    assert verdict.decision_basis == "balanced_tie_break"

    with pytest.raises(ValueError):
        _build_verdict_from_response_mapping(
            {**response, "winner_candidate_id": "Z"},
            candidate_names={"A1K2", "B9X4"},
        )


def test_build_verdict_from_response_mapping_accepts_staged_schema_and_text_confidence() -> None:
    response = {
        "same_slot_winner_candidate_id": "A1K2",
        "extra_slot_override": False,
        "winner_candidate_id": "A1K2",
        "confidence": "high",
        "decision_basis": "same_slot_route_advantage",
        "ranking": ["A1K2", "B9X4"],
        "decisive_signals": [
            "route_pressure_score:A1K2:high",
            "A1K2:path_common_free_ratio:medium",
        ],
    }

    verdict = _build_verdict_from_response_mapping(
        response,
        candidate_names={"A1K2", "B9X4"},
    )

    assert verdict.winner_candidate_id == "A1K2"
    assert verdict.confidence == pytest.approx(0.9)
    assert verdict.decision_basis == "same_slot_route_advantage"
    assert verdict.ranking == ("A1K2", "B9X4")
    assert tuple(signal.factor for signal in verdict.decisive_signals) == (
        "route_pressure_score",
        "path_common_free_ratio",
    )
    assert tuple(signal.supports for signal in verdict.decisive_signals) == ("A1K2", "A1K2")


def test_build_verdict_from_response_mapping_accepts_legacy_shapes() -> None:
    legacy_object = {
        "winner": "A1K2",
        "confidence": 0.65,
        "ranking": ["A1K2", "B9X4"],
        "used_tie_break": False,
        "decisive_signals": [
            {
                "factor": "fragmentation",
                "supports": "A1K2",
                "importance": "high",
                "evidence": "legacy object output",
            }
        ],
    }
    legacy_list = [
        {
            "candidate": "B9X4",
            "confidence": 0.52,
            "signals": [
                {
                    "factor": "qot_safety",
                    "supports": "B9X4",
                    "importance": "medium",
                }
            ],
        }
    ]

    object_verdict = _build_verdict_from_response_mapping(
        legacy_object,
        candidate_names={"A1K2", "B9X4"},
    )
    list_verdict = _build_verdict_from_response_mapping(
        legacy_list,
        candidate_names={"A1K2", "B9X4"},
    )

    assert object_verdict.winner_candidate_id == "A1K2"
    assert list_verdict.winner_candidate_id == "B9X4"
    assert list_verdict.confidence == pytest.approx(0.52)


def test_extract_json_sanitizes_control_chars_and_extra_suffix() -> None:
    raw_text = "\x00{\"winner_candidate_id\":\"A1\",\"confidence\":0.9,\"decision_basis\":\"balanced_tie_break\",\"ranking\":[\"A1\"],\"decisive_signals\":[\"support_count:A1:low\"]}\nnoise"

    parsed = _extract_json(raw_text)

    assert parsed["winner_candidate_id"] == "A1"

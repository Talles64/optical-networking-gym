from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_METHOD_RULES_PATH = SCRIPT_DIR / "ONLINE_JUDGE_METHOD_RULES.md"
DEFAULT_BASELINE_PATH = SCRIPT_DIR / "current_judge_heuristic_seed_baseline.json"


@dataclass(frozen=True, slots=True)
class JudgeFailurePackConfig:
    run_dir: Path
    output_path: Path | None = None
    calls_jsonl: Path | None = None
    steps_csv: Path | None = None
    summary_csv: Path | None = None
    method_rules_path: Path = DEFAULT_METHOD_RULES_PATH
    baseline_path: Path = DEFAULT_BASELINE_PATH
    future_regret_detail_csv: Path | None = None
    future_regret_summary_csv: Path | None = None
    representative_case_limit: int = 10
    min_step_index: int = 0
    max_step_index: int | None = None
    decision_basis: str | None = None
    require_judge_vs_reference_mismatch: bool = False
    require_basis_vs_payload_mismatch: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        object.__setattr__(self, "output_path", None if self.output_path is None else Path(self.output_path))
        object.__setattr__(self, "calls_jsonl", None if self.calls_jsonl is None else Path(self.calls_jsonl))
        object.__setattr__(self, "steps_csv", None if self.steps_csv is None else Path(self.steps_csv))
        object.__setattr__(self, "summary_csv", None if self.summary_csv is None else Path(self.summary_csv))
        object.__setattr__(self, "method_rules_path", Path(self.method_rules_path))
        object.__setattr__(self, "baseline_path", Path(self.baseline_path))
        object.__setattr__(
            self,
            "future_regret_detail_csv",
            None if self.future_regret_detail_csv is None else Path(self.future_regret_detail_csv),
        )
        object.__setattr__(
            self,
            "future_regret_summary_csv",
            None if self.future_regret_summary_csv is None else Path(self.future_regret_summary_csv),
        )
        if int(self.representative_case_limit) <= 0:
            raise ValueError("representative_case_limit must be positive")
        if int(self.min_step_index) < 0:
            raise ValueError("min_step_index must be non-negative")
        if self.max_step_index is not None and int(self.max_step_index) < int(self.min_step_index):
            raise ValueError("max_step_index must be greater than or equal to min_step_index")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_single_file(run_dir: Path, pattern: str, explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"missing file: {explicit}")
        return explicit
    matches = sorted(run_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one {pattern} in {run_dir}, found {len(matches)}")
    return matches[0]


def _parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _parse_int(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def _infer_load_from_run_dir(run_dir: Path) -> float | None:
    match = re.search(r"load(?P<load>\d+(?:\.\d+)?)", str(run_dir))
    if match is None:
        return None
    return float(match.group("load"))


def _load_baseline_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_entry_for_load(baseline_payload: dict[str, Any], load: float | None) -> dict[str, Any] | None:
    if load is None:
        return None
    for entry in baseline_payload.get("loads", []):
        if abs(float(entry.get("load", -1.0)) - float(load)) <= 1e-6:
            return dict(entry)
    return None


def _load_method_rules_digest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"source_path": str(path), "sections": {}, "raw_excerpt": ""}
    lines = path.read_text(encoding="utf-8").splitlines()
    sections: dict[str, list[str]] = {}
    current_section = "preamble"
    sections[current_section] = []
    for raw_line in lines:
        line = raw_line.rstrip()
        if line.startswith("## "):
            current_section = line[3:].strip()
            sections.setdefault(current_section, [])
            continue
        if line.startswith("- "):
            sections.setdefault(current_section, []).append(line[2:].strip())
    return {
        "source_path": str(path),
        "sections": sections,
        "raw_excerpt": "\n".join(lines[:80]),
    }


def _load_future_regret_detail(path: Path | None) -> dict[tuple[int, int], dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows = _read_csv_rows(path)
    detail_by_step: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (_parse_int(row.get("episode_index")), _parse_int(row.get("step_index")))
        judge_gap_25 = _parse_int(row.get("judge_blocking_delta_25")) - _parse_int(row.get("future_best_blocking_delta_25"))
        judge_gap_50 = _parse_int(row.get("judge_blocking_delta_50")) - _parse_int(row.get("future_best_blocking_delta_50"))
        judge_gap_100 = _parse_int(row.get("judge_blocking_delta_100")) - _parse_int(row.get("future_best_blocking_delta_100"))
        detail_by_step[key] = {
            "judge_matches_future_best": str(row.get("judge_matches_future_best", "")).lower() == "true",
            "blocking_gap_25": judge_gap_25,
            "blocking_gap_50": judge_gap_50,
            "blocking_gap_100": judge_gap_100,
        }
    return detail_by_step


def _heuristic_choice_histogram(step_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in step_rows:
        heuristic = str(row.get("winner_heuristic", "")).strip()
        if not heuristic:
            continue
        counts[heuristic] = counts.get(heuristic, 0) + 1
    return [
        {"heuristic_name": heuristic_name, "count": count}
        for heuristic_name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _count_pairs(step_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for row in step_rows:
        chosen = str(row.get("winner_heuristic", "")).strip()
        reference = str(row.get("reference_winner", "")).strip()
        if not chosen or not reference or chosen == reference:
            continue
        counts[(chosen, reference)] = counts.get((chosen, reference), 0) + 1
    return [
        {"chosen_heuristic": chosen, "reference_winner": reference, "count": count}
        for (chosen, reference), count in sorted(counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    ]


def _count_decision_basis_mismatches(step_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str, str], int] = {}
    for row in step_rows:
        chosen = str(row.get("winner_heuristic", "")).strip()
        reference = str(row.get("reference_winner", "")).strip()
        decision_basis = str(row.get("winner_decision_basis", "")).strip()
        if not chosen or not reference or chosen == reference:
            continue
        key = (decision_basis or "unknown", chosen, reference)
        counts[key] = counts.get(key, 0) + 1
    return [
        {
            "decision_basis": decision_basis,
            "chosen_heuristic": chosen,
            "reference_winner": reference,
            "count": count,
        }
        for (decision_basis, chosen, reference), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2]),
        )
    ]


def _count_fallback_reasons(step_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for row in step_rows:
        reason = str(row.get("fallback_reason", "")).strip()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return [
        {"fallback_reason": reason, "count": count}
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _blocking_checkpoints(step_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not step_rows:
        return []
    checkpoints = (0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999)
    rows_by_step = {_parse_int(row.get("step_index")): row for row in step_rows}
    resolved: list[dict[str, Any]] = []
    last_row = step_rows[-1]
    for checkpoint in checkpoints:
        candidate = rows_by_step.get(checkpoint)
        if candidate is None:
            candidate = max(
                (row for row in step_rows if _parse_int(row.get("step_index")) <= checkpoint),
                key=lambda row: _parse_int(row.get("step_index")),
                default=last_row,
            )
        resolved.append(
            {
                "step_index": _parse_int(candidate.get("step_index")),
                "episode_service_blocking_rate": _parse_float(candidate.get("episode_service_blocking_rate")),
                "winner_heuristic": str(candidate.get("winner_heuristic", "")).strip(),
                "reference_winner": str(candidate.get("reference_winner", "")).strip(),
            }
        )
    first_positive = next(
        (row for row in step_rows if _parse_float(row.get("episode_service_blocking_rate")) > 0.0),
        None,
    )
    if first_positive is not None:
        resolved.append(
            {
                "step_index": _parse_int(first_positive.get("step_index")),
                "episode_service_blocking_rate": _parse_float(first_positive.get("episode_service_blocking_rate")),
                "winner_heuristic": str(first_positive.get("winner_heuristic", "")).strip(),
                "reference_winner": str(first_positive.get("reference_winner", "")).strip(),
                "label": "first_positive_blocking",
            }
        )
    return resolved


def _trim_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(candidate.get("metrics", {}))
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "candidate_roles": list(candidate.get("candidate_roles", [])),
        "is_reject": bool(candidate.get("is_reject", False)),
        "route": dict(candidate.get("route", {})) if candidate.get("route") is not None else None,
        "metrics": {
            "required_slots": metrics.get("required_slots"),
            "route_pressure_score": metrics.get("route_pressure_score"),
            "local_damage_score": metrics.get("local_damage_score"),
            "path_common_free_ratio": metrics.get("path_common_free_ratio"),
            "qot_margin_clipped_db": metrics.get("qot_margin_clipped_db"),
            "future_risk_band": metrics.get("future_risk_band"),
            "qot_safe_now": metrics.get("qot_safe_now"),
        },
    }


def _case_priority(
    *,
    step_row: dict[str, str],
    future_regret_evidence: dict[str, Any] | None,
    first_positive_step: int | None,
) -> tuple[int, int, int]:
    chosen = str(step_row.get("winner_heuristic", "")).strip()
    reference = str(step_row.get("reference_winner", "")).strip()
    mismatch = 1 if chosen and reference and chosen != reference else 0
    basis_mismatch = 1 if str(step_row.get("basis_vs_payload_mismatch", "")).lower() == "true" else 0
    future_regret_gap = 0
    if future_regret_evidence is not None:
        future_regret_gap = max(
            int(future_regret_evidence.get("blocking_gap_25", 0)),
            int(future_regret_evidence.get("blocking_gap_50", 0)),
            int(future_regret_evidence.get("blocking_gap_100", 0)),
        )
    onset_distance = 999999
    if first_positive_step is not None:
        onset_distance = abs(_parse_int(step_row.get("step_index")) - int(first_positive_step))
    return (mismatch + basis_mismatch, future_regret_gap, -onset_distance)


def _filter_analysis_rows(
    *,
    step_rows: list[dict[str, str]],
    min_step_index: int,
    max_step_index: int | None,
    decision_basis: str | None,
    require_judge_vs_reference_mismatch: bool,
    require_basis_vs_payload_mismatch: bool,
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    expected_basis = None if decision_basis is None else str(decision_basis).strip()
    for row in step_rows:
        step_index = _parse_int(row.get("step_index"))
        if step_index < int(min_step_index):
            continue
        if max_step_index is not None and step_index > int(max_step_index):
            continue
        chosen = str(row.get("winner_heuristic", "")).strip()
        reference = str(row.get("reference_winner", "")).strip()
        if require_judge_vs_reference_mismatch and (not chosen or not reference or chosen == reference):
            continue
        if require_basis_vs_payload_mismatch and str(row.get("basis_vs_payload_mismatch", "")).lower() != "true":
            continue
        if expected_basis is not None and str(row.get("winner_decision_basis", "")).strip() != expected_basis:
            continue
        filtered.append(row)
    return filtered


def _representative_cases(
    *,
    step_rows: list[dict[str, str]],
    call_records: list[dict[str, Any]],
    future_regret_detail: dict[tuple[int, int], dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    calls_by_key = {
        (_parse_int(record["audit"].get("episode_index")), _parse_int(record["audit"].get("step_index"))): record
        for record in call_records
    }
    first_positive_step = next(
        (_parse_int(row.get("step_index")) for row in step_rows if _parse_float(row.get("episode_service_blocking_rate")) > 0.0),
        None,
    )
    ranked_rows = sorted(
        step_rows,
        key=lambda row: (
            -_case_priority(
                step_row=row,
                future_regret_evidence=future_regret_detail.get(
                    (_parse_int(row.get("episode_index")), _parse_int(row.get("step_index")))
                ),
                first_positive_step=first_positive_step,
            )[0],
            -_case_priority(
                step_row=row,
                future_regret_evidence=future_regret_detail.get(
                    (_parse_int(row.get("episode_index")), _parse_int(row.get("step_index")))
                ),
                first_positive_step=first_positive_step,
            )[1],
            -_parse_int(row.get("step_index")),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int]] = set()
    for row in ranked_rows:
        key = (_parse_int(row.get("episode_index")), _parse_int(row.get("step_index")))
        if key in seen_keys:
            continue
        record = calls_by_key.get(key)
        if record is None:
            continue
        future_regret_evidence = future_regret_detail.get(key)
        selection_reasons: list[str] = []
        chosen = str(row.get("winner_heuristic", "")).strip()
        reference = str(row.get("reference_winner", "")).strip()
        if chosen and reference and chosen != reference:
            selection_reasons.append("judge_vs_reference_mismatch")
        if str(row.get("basis_vs_payload_mismatch", "")).lower() == "true":
            selection_reasons.append("basis_vs_payload_mismatch")
        if first_positive_step is not None and abs(key[1] - int(first_positive_step)) <= 25:
            selection_reasons.append("near_blocking_onset")
        if future_regret_evidence is not None and not bool(future_regret_evidence.get("judge_matches_future_best", True)):
            selection_reasons.append("offline_future_regret_disagrees")
        if not selection_reasons:
            continue
        case_payload = {
            "case_id": f"ep{key[0]}-step{key[1]}",
            "episode_index": key[0],
            "step_index": key[1],
            "blocking_rate_so_far": _parse_float(row.get("episode_service_blocking_rate")),
            "controller_decision_source": str(row.get("controller_decision_source", "")).strip(),
            "chosen_heuristic": chosen,
            "reference_winner": reference,
            "winner_candidate_id": str(row.get("winner_candidate_id", "")).strip(),
            "winner_decision_basis": str(row.get("winner_decision_basis", "")).strip(),
            "basis_vs_payload_mismatch": str(row.get("basis_vs_payload_mismatch", "")).lower() == "true",
            "fallback_reason": str(row.get("fallback_reason", "")).strip(),
            "decisive_signals_summary": str(row.get("decisive_signals_summary", "")).strip(),
            "semantic_warning_flags": str(row.get("semantic_warning_flags", "")).strip(),
            "selection_reasons": selection_reasons,
            "prompt_context": dict(record["decision_payload"].get("prompt_context") or {}),
            "candidates": [_trim_candidate(candidate) for candidate in record["decision_payload"].get("candidates", ())],
        }
        if future_regret_evidence is not None:
            case_payload["offline_future_regret_evidence"] = dict(future_regret_evidence)
        selected.append(case_payload)
        seen_keys.add(key)
        if len(selected) >= int(limit):
            break
    return selected


def build_failure_pack(*, config: JudgeFailurePackConfig) -> dict[str, Any]:
    calls_path = _resolve_single_file(config.run_dir, "*-llm-judge-calls.jsonl", config.calls_jsonl)
    steps_path = _resolve_single_file(config.run_dir, "*-llm-judge-steps.csv", config.steps_csv)
    summary_path = _resolve_single_file(config.run_dir, "*-llm-judge-summary.csv", config.summary_csv)

    call_records = _read_json_lines(calls_path)
    step_rows = _read_csv_rows(steps_path)
    summary_rows = _read_csv_rows(summary_path)
    summary_run_row = next((row for row in summary_rows if str(row.get("scope", "")).strip() == "run"), summary_rows[0])
    analysis_step_rows = _filter_analysis_rows(
        step_rows=step_rows,
        min_step_index=int(config.min_step_index),
        max_step_index=None if config.max_step_index is None else int(config.max_step_index),
        decision_basis=config.decision_basis,
        require_judge_vs_reference_mismatch=bool(config.require_judge_vs_reference_mismatch),
        require_basis_vs_payload_mismatch=bool(config.require_basis_vs_payload_mismatch),
    )

    inferred_load = _infer_load_from_run_dir(config.run_dir)
    baseline_payload = _load_baseline_payload(config.baseline_path)
    baseline_entry = _baseline_entry_for_load(baseline_payload, inferred_load)
    method_rules_digest = _load_method_rules_digest(config.method_rules_path)
    future_regret_detail = _load_future_regret_detail(config.future_regret_detail_csv)

    scenario_truth: dict[str, Any] = {
        "scenario_profile": str(baseline_payload.get("scenario_profile", "legacy_benchmark")),
        "topology_id": str(baseline_payload.get("topology_id", "nobel-eu")),
        "load": inferred_load,
        "seed": int(baseline_payload.get("seed", 10)),
        "episode_count": int(baseline_payload.get("episode_count", 1)),
        "episode_length": int(baseline_payload.get("episode_length", 1000)),
    }
    if baseline_entry is not None:
        scenario_truth["environment"] = dict(baseline_entry.get("environment", {}))

    acceptance_target = None
    if baseline_entry is not None:
        acceptance_target = {
            "load": float(baseline_entry["load"]),
            "best_heuristic": str(baseline_entry["best_heuristic"]),
            "best_service_blocking_rate_mean": float(baseline_entry["best_service_blocking_rate_mean"]),
        }

    run_summary = {
        "scope": str(summary_run_row.get("scope", "")),
        "steps": _parse_int(summary_run_row.get("steps")),
        "llm_calls": _parse_int(summary_run_row.get("llm_calls")),
        "fallback_count": _parse_int(summary_run_row.get("fallback_count")),
        "reference_agreement_rate": _parse_float(summary_run_row.get("reference_agreement_rate")),
        "hidden_balanced_agreement_rate": _parse_float(summary_run_row.get("hidden_balanced_agreement_rate")),
        "final_blocking_rate": _parse_float(summary_run_row.get("final_blocking_rate")),
        "mean_episode_service_blocking_rate": _parse_float(summary_run_row.get("mean_episode_service_blocking_rate")),
        "choice_histogram": _heuristic_choice_histogram(step_rows),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(config.run_dir),
        "source_files": {
            "calls_jsonl": str(calls_path),
            "steps_csv": str(steps_path),
            "summary_csv": str(summary_path),
            "future_regret_detail_csv": None if config.future_regret_detail_csv is None else str(config.future_regret_detail_csv),
            "future_regret_summary_csv": None if config.future_regret_summary_csv is None else str(config.future_regret_summary_csv),
        },
        "scenario_truth": scenario_truth,
        "acceptance_target": acceptance_target,
        "run_summary": run_summary,
        "analysis_scope": {
            "source_step_count": len(step_rows),
            "analysis_step_count": len(analysis_step_rows),
            "min_step_index": int(config.min_step_index),
            "max_step_index": None if config.max_step_index is None else int(config.max_step_index),
            "decision_basis": None if config.decision_basis is None else str(config.decision_basis),
            "require_judge_vs_reference_mismatch": bool(config.require_judge_vs_reference_mismatch),
            "require_basis_vs_payload_mismatch": bool(config.require_basis_vs_payload_mismatch),
        },
        "blocking_checkpoints": _blocking_checkpoints(analysis_step_rows or step_rows),
        "heuristic_choice_histogram": _heuristic_choice_histogram(analysis_step_rows),
        "top_mismatch_pairs": _count_pairs(analysis_step_rows)[:10],
        "top_decision_basis_mismatches": _count_decision_basis_mismatches(analysis_step_rows)[:10],
        "top_fallback_reasons": _count_fallback_reasons(analysis_step_rows or step_rows)[:10],
        "representative_cases": _representative_cases(
            step_rows=analysis_step_rows,
            call_records=call_records,
            future_regret_detail=future_regret_detail,
            limit=int(config.representative_case_limit),
        ),
        "method_rules_digest": method_rules_digest,
    }


def write_failure_pack(*, config: JudgeFailurePackConfig) -> Path:
    output_path = config.output_path or (config.run_dir / "failure_pack.json")
    payload = build_failure_pack(config=config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact failure pack for a judge online benchmark run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--calls-jsonl", type=Path, default=None)
    parser.add_argument("--steps-csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--method-rules-path", type=Path, default=DEFAULT_METHOD_RULES_PATH)
    parser.add_argument("--baseline-path", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--future-regret-detail-csv", type=Path, default=None)
    parser.add_argument("--future-regret-summary-csv", type=Path, default=None)
    parser.add_argument("--representative-case-limit", type=int, default=10)
    parser.add_argument("--min-step-index", type=int, default=0)
    parser.add_argument("--max-step-index", type=int, default=None)
    parser.add_argument("--decision-basis", default=None)
    parser.add_argument("--require-judge-vs-reference-mismatch", action="store_true")
    parser.add_argument("--require-basis-vs-payload-mismatch", action="store_true")
    args = parser.parse_args()

    output_path = write_failure_pack(
        config=JudgeFailurePackConfig(
            run_dir=args.run_dir,
            output_path=args.output_path,
            calls_jsonl=args.calls_jsonl,
            steps_csv=args.steps_csv,
            summary_csv=args.summary_csv,
            method_rules_path=args.method_rules_path,
            baseline_path=args.baseline_path,
            future_regret_detail_csv=args.future_regret_detail_csv,
            future_regret_summary_csv=args.future_regret_summary_csv,
            representative_case_limit=args.representative_case_limit,
            min_step_index=args.min_step_index,
            max_step_index=args.max_step_index,
            decision_basis=args.decision_basis,
            require_judge_vs_reference_mismatch=bool(args.require_judge_vs_reference_mismatch),
            require_basis_vs_payload_mismatch=bool(args.require_basis_vs_payload_mismatch),
        )
    )
    print(json.dumps({"output_path": str(output_path)}, ensure_ascii=True))


if __name__ == "__main__":
    main()

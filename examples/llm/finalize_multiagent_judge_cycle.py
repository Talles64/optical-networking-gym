from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRACKER_PATH = SCRIPT_DIR / "Multi-agent_JUDGE_CHANGE_TRACKER.md"
DEFAULT_STATE_PATH = SCRIPT_DIR / "multi_agent_judge_state.json"


def _cycle_module() -> dict[str, Any]:
    return runpy.run_path(str(SCRIPT_DIR / "run_multiagent_judge_cycle.py"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _summarize_analyst(review: dict[str, Any]) -> str:
    evidence = ",".join(str(case_id) for case_id in review.get("evidence_case_ids", []))
    return (
        f"{review.get('primary_failure_mode', 'unknown')}; "
        f"onset={review.get('onset_window', 'unknown')}; "
        f"evidence={evidence or 'none'}"
    )


def _summarize_reviewer(review: dict[str, Any]) -> str:
    layer = str(review.get("allowed_change_layer", "no_change"))
    evidence = ",".join(str(case_id) for case_id in review.get("evidence_case_ids", []))
    change = str(review.get("recommended_single_change", "")).strip() or "no_change"
    return f"{layer}: {change}; evidence={evidence or 'none'}"


def finalize_multiagent_cycle(
    *,
    cycle_id: str,
    analyst_review_path: Path,
    reviewer_review_path: Path,
    state_path: Path = DEFAULT_STATE_PATH,
    tracker_path: Path = DEFAULT_TRACKER_PATH,
) -> dict[str, Any]:
    cycle_module = _cycle_module()
    render_tracker = cycle_module["_render_tracker_markdown"]
    agent_review_status = cycle_module["_agent_review_status"]

    state = _load_json(Path(state_path))
    analyst_review = _load_json(Path(analyst_review_path))
    reviewer_review = _load_json(Path(reviewer_review_path))

    cycle_entry = next(
        (cycle for cycle in state.get("cycles", []) if str(cycle.get("cycle_id")) == str(cycle_id)),
        None,
    )
    if cycle_entry is None:
        raise ValueError(f"cycle_id not found in state: {cycle_id}")

    review_status = agent_review_status(
        analyst_review=analyst_review,
        reviewer_review=reviewer_review,
    )
    raw_status = str(review_status["convergence_status"])
    if raw_status == "converged_ready_for_patch":
        convergence_status = "converged"
    elif raw_status == "converged_no_change":
        convergence_status = "reviewer_no_change"
    else:
        convergence_status = raw_status
    cycle_entry["analyst_findings"] = _summarize_analyst(analyst_review)
    cycle_entry["reviewer_recommendation"] = _summarize_reviewer(reviewer_review)
    cycle_entry["convergence_status"] = convergence_status

    artifacts = dict(cycle_entry.get("artifacts", {}))
    artifacts["analyst_review"] = str(Path(analyst_review_path))
    artifacts["reviewer_review"] = str(Path(reviewer_review_path))
    cycle_entry["artifacts"] = artifacts

    if convergence_status == "converged":
        cycle_entry["applied_change"] = "pending_implementation"
        cycle_entry["decision_next_step"] = (
            "implement recommended single change, run narrow tests, rerun official online benchmark"
        )
    elif convergence_status == "reviewer_no_change":
        cycle_entry["applied_change"] = "no_change"
        cycle_entry["decision_next_step"] = "close cycle without patch because Reviewer returned no_change"
    else:
        cycle_entry["applied_change"] = "no_change"
        cycle_entry["decision_next_step"] = "close cycle without patch because Analyst and Reviewer did not converge"

    state_path = Path(state_path)
    tracker_path = Path(tracker_path)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    tracker_path.write_text(render_tracker(state=state), encoding="utf-8")
    return {
        "cycle_id": str(cycle_id),
        "convergence_status": convergence_status,
        "state_path": str(state_path),
        "tracker_path": str(tracker_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize a multi-agent judge cycle from analyst/reviewer JSON outputs")
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--analyst-review-path", type=Path, required=True)
    parser.add_argument("--reviewer-review-path", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--tracker-path", type=Path, default=DEFAULT_TRACKER_PATH)
    args = parser.parse_args()

    output = finalize_multiagent_cycle(
        cycle_id=str(args.cycle_id),
        analyst_review_path=args.analyst_review_path,
        reviewer_review_path=args.reviewer_review_path,
        state_path=args.state_path,
        tracker_path=args.tracker_path,
    )
    print(json.dumps(output, ensure_ascii=True))


if __name__ == "__main__":
    main()

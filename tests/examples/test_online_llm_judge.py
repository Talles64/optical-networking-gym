from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import runpy

from optical_networking_gym_v2.judge.heuristic_judge import DecisiveSignal, JudgeVerdict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_PATH = PROJECT_ROOT / "examples" / "llm" / "online_heuristic_judge.py"


def _load_module() -> dict[str, object]:
    return runpy.run_path(str(EXAMPLE_PATH))


class FakeJudge:
    def __init__(self) -> None:
        self.calls = 0

    def judge(self, payload):
        self.calls += 1
        ranking = tuple(candidate.heuristic_name for candidate in payload.candidates)
        winner = ranking[0]
        return JudgeVerdict(
            winner=winner,
            confidence=0.91,
            ranking=ranking,
            reason="picked first candidate in fake judge",
            used_tie_break=False,
            decisive_signals=(
                DecisiveSignal(
                    factor="load_balance",
                    supports=winner,
                    evidence="fake judge always prefers the first candidate.",
                    importance="medium",
                ),
            ),
        )


class CountingEnv:
    def __init__(self, env, counter: dict[str, int]) -> None:
        self._env = env
        self._counter = counter

    def __getattr__(self, name: str):
        return getattr(self._env, name)

    def step(self, action: int):
        self._counter["step_calls"] += 1
        return self._env.step(action)


def test_online_llm_judge_smoke_generates_outputs_and_uses_single_step_per_decision(
    tmp_path: Path,
) -> None:
    module = _load_module()
    fake_judge = FakeJudge()
    counter = {"step_calls": 0}
    original_build_env = module["build_env"]

    def counting_build_env(*, scenario):
        return CountingEnv(original_build_env(scenario=scenario), counter)

    module["run_experiment"].__globals__["build_env"] = counting_build_env

    experiment = module["LLMJudgeExperiment"](
        topology_id="ring_4",
        episode_count=1,
        episode_length=6,
        seed=7,
        load=10.0,
        mean_holding_time=8.0,
        num_spectrum_resources=24,
        k_paths=2,
        modulations_to_consider=2,
        output_dir=tmp_path,
    )

    outputs = module["run_experiment"](
        experiment=experiment,
        judge=fake_judge,
        now=datetime(2026, 3, 19),
    )

    assert outputs.steps_csv == tmp_path / "19-03-llm-judge-steps.csv"
    assert outputs.summary_csv == tmp_path / "19-03-llm-judge-summary.csv"
    assert outputs.calls_jsonl == tmp_path / "19-03-llm-judge-calls.jsonl"
    assert outputs.steps_csv.exists()
    assert outputs.summary_csv.exists()
    assert outputs.calls_jsonl.exists()

    with outputs.steps_csv.open("r", encoding="utf-8", newline="") as handle:
        step_rows = list(csv.DictReader(handle))
    with outputs.summary_csv.open("r", encoding="utf-8", newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    with outputs.calls_jsonl.open("r", encoding="utf-8") as handle:
        call_records = [json.loads(line) for line in handle if line.strip()]

    assert step_rows
    assert call_records
    assert counter["step_calls"] == len(step_rows)
    assert fake_judge.calls <= len(step_rows)
    assert "winner_decoded_path_nodes" in step_rows[0]
    assert "winner_modulation_name" in step_rows[0]
    assert "winner_confidence" in step_rows[0]
    assert "decisive_signals_summary" in step_rows[0]
    assert "fallback_reason" in step_rows[0]
    assert "judge_error_message" in step_rows[0]
    assert any(row["scope"] == "run" for row in summary_rows)

    first_record = call_records[0]
    assert "audit" in first_record
    assert "decision_payload" in first_record
    assert "prompt" in first_record
    assert "model_io" in first_record
    assert "baseline_winner" in first_record["audit"]
    assert "prompt_version" in first_record["audit"]
    assert "raw_action" in first_record["audit"]["candidate_audit"][0]
    assert "system_prompt" in first_record["prompt"]
    assert "user_prompt" in first_record["prompt"]
    assert "raw_model_response" in first_record["model_io"]
    assert "parsed_response" in first_record["model_io"]

    candidate_payload = first_record["decision_payload"]["candidates"][0]
    assert "raw_action" not in candidate_payload
    assert "baseline_scores" not in candidate_payload
    assert "route_summary" in candidate_payload

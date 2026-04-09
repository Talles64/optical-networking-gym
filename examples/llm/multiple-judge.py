"""Run the LLM heuristic judge across multiple models and loads sequentially.

Each model runs its full experiment one at a time for each requested load.
The same load sweep is also executed for the fixed heuristic baselines.
Results are merged into a summary_all_models.csv with model_name,
heuristic_name, load, runner_type, and avg_step_time_s columns.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optical_networking_gym_v2.defaults import DEFAULT_SEED
from optical_networking_gym_v2.judge import (
    JudgeCallTrace,
    JudgeDecisionPayload,
    JudgeVerdict,
    build_ollama_prompt_record,
)

SCRIPT_DIR = Path(__file__).resolve().parent
_JUDGE_SCRIPT = SCRIPT_DIR / "online_heuristic_judge.py"
_HEURISTIC_BASELINE_SCRIPT = SCRIPT_DIR / "build_judge_heuristic_seed_baseline.py"
if not _JUDGE_SCRIPT.exists():
    raise RuntimeError(f"Cannot find online_heuristic_judge.py at {_JUDGE_SCRIPT}")
if not _HEURISTIC_BASELINE_SCRIPT.exists():
    raise RuntimeError(
        f"Cannot find build_judge_heuristic_seed_baseline.py at {_HEURISTIC_BASELINE_SCRIPT}"
    )

spec = importlib.util.spec_from_file_location("_online_judge", str(_JUDGE_SCRIPT))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load online_heuristic_judge.py from {_JUDGE_SCRIPT}")
_online = importlib.util.module_from_spec(spec)
sys.modules["_online_judge"] = _online
spec.loader.exec_module(_online)

baseline_spec = importlib.util.spec_from_file_location(
    "_heuristic_baseline",
    str(_HEURISTIC_BASELINE_SCRIPT),
)
if baseline_spec is None or baseline_spec.loader is None:
    raise RuntimeError(
        f"Cannot load build_judge_heuristic_seed_baseline.py from {_HEURISTIC_BASELINE_SCRIPT}"
    )
_heuristic_baseline = importlib.util.module_from_spec(baseline_spec)
sys.modules["_heuristic_baseline"] = _heuristic_baseline
baseline_spec.loader.exec_module(_heuristic_baseline)

LLMJudgeExperiment = _online.LLMJudgeExperiment
run_experiment = _online.run_experiment
_date_prefix = _online._date_prefix
JudgeHeuristicSeedBaselineConfig = _heuristic_baseline.JudgeHeuristicSeedBaselineConfig
build_heuristic_seed_baseline = _heuristic_baseline.build_heuristic_seed_baseline


@dataclass(frozen=True, slots=True)
class ModelConfig:
    name: str
    base_url: str
    model: str
    api_key: str = ""
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class LoadSweepConfig:
    start: float
    end: float
    step: float

    def __post_init__(self) -> None:
        if math.isclose(self.step, 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("load step must be non-zero")
        delta = self.end - self.start
        if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
            return
        if delta > 0 and self.step < 0:
            raise ValueError("load step must be positive when load_end > load_start")
        if delta < 0 and self.step > 0:
            raise ValueError("load step must be negative when load_end < load_start")

    def build_loads(self) -> tuple[float, ...]:
        if math.isclose(self.start, self.end, rel_tol=0.0, abs_tol=1e-12):
            return (_normalize_load_value(self.start),)

        loads: list[float] = []
        current = self.start
        epsilon = abs(self.step) * 1e-9 + 1e-12
        if self.step > 0:
            while current <= self.end + epsilon:
                loads.append(_normalize_load_value(current))
                current += self.step
        else:
            while current >= self.end - epsilon:
                loads.append(_normalize_load_value(current))
                current += self.step

        if not loads:
            raise ValueError(
                f"load sweep produced no values for start={self.start}, end={self.end}, step={self.step}"
            )
        return tuple(loads)


def _extract_json(text: str) -> object:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = "".join(ch for ch in cleaned if ch.isprintable() or ch in "\r\n\t")
    cleaned = cleaned.strip()
    if not cleaned:
        raise ValueError("could not extract JSON from empty model response")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", cleaned):
        try:
            parsed, _ = decoder.raw_decode(cleaned[match.start() :])
            return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"could not extract JSON from response: {cleaned[:200]}")


def _normalize_candidate_name(raw_value: str, *, candidate_names: set[str]) -> str:
    stripped = raw_value.strip()
    if stripped in candidate_names:
        return stripped
    lowered = stripped.lower()
    exact = [name for name in candidate_names if name.lower() == lowered]
    if len(exact) == 1:
        return exact[0]
    substring_matches = [
        name for name in candidate_names if name.lower() in lowered or lowered in name.lower()
    ]
    if len(substring_matches) == 1:
        return substring_matches[0]
    raise ValueError(f"unknown candidate id {raw_value!r}")


def _normalize_confidence(raw: object) -> float:
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    text = str(raw).strip().lower()
    aliases = {
        "very_high": 0.95,
        "high": 0.90,
        "medium": 0.75,
        "med": 0.75,
        "low": 0.60,
        "very_low": 0.45,
    }
    if text in aliases:
        return aliases[text]
    if text.endswith("%"):
        return max(0.0, min(1.0, float(text[:-1]) / 100.0))
    return max(0.0, min(1.0, float(text)))


_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "same_slot_winner_candidate_id": {"type": "string"},
        "extra_slot_override": {"type": "boolean"},
        "winner_candidate_id": {"type": "string"},
        "confidence": {
            "anyOf": [
                {"type": "number", "minimum": 0.0, "maximum": 1.0},
                {"type": "string", "enum": ["very_high", "high", "medium", "med", "low", "very_low"]},
            ]
        },
        "decision_basis": {
            "type": "string",
            "enum": [
                "same_slot_route_advantage",
                "same_slot_local_advantage",
                "extra_slot_structural_advantage",
                "balanced_tie_break",
                "qot_safety_override",
            ],
        },
        "ranking": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "decisive_signals": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    },
    "required": [
        "same_slot_winner_candidate_id",
        "extra_slot_override",
        "winner_candidate_id",
        "confidence",
        "decision_basis",
        "ranking",
        "decisive_signals",
    ],
}


class ChatJudge:
    """Judge supporting Ollama (api_key='') and OpenAI-compatible (api_key set)."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._last_trace: JudgeCallTrace | None = None

    def consume_last_trace(self) -> JudgeCallTrace | None:
        trace, self._last_trace = self._last_trace, None
        return trace

    def _chat(self, messages: list[dict[str, str]]) -> tuple[object, dict[str, object]]:
        if self.api_key:
            return self._openai_chat(messages)
        return self._ollama_chat(messages)

    def _ollama_chat(self, messages: list[dict[str, str]]) -> tuple[object, dict[str, object]]:
        import ollama as ollama_client

        client = ollama_client.Client(host=self.base_url, timeout=self.timeout_s)
        last_err: Exception | None = None
        for _ in range(max(1, self.max_retries)):
            try:
                response = client.chat(
                    model=self.model,
                    messages=messages,
                    format=_JSON_SCHEMA,
                    options={"temperature": self.temperature, "num_think": 0},
                    stream=False,
                )
                raw = response.message.content or ""
                return _extract_json(raw), {"content": raw}
            except (ValueError, RuntimeError):
                raise
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"Ollama failed after {self.max_retries} attempts: {last_err}")

    def _openai_chat(self, messages: list[dict[str, str]]) -> tuple[object, dict[str, object]]:
        import requests

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object", "schema": _JSON_SCHEMA},
        }
        last_err: Exception | None = None
        for _ in range(max(1, self.max_retries)):
            try:
                response = requests.post(url, headers=headers, json=body, timeout=self.timeout_s)
                response.raise_for_status()
                raw = response.json()["choices"][0]["message"]["content"]
                return _extract_json(raw), {"content": raw, "raw_response": response.json()}
            except (ValueError, RuntimeError):
                raise
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"OpenAI-compatible failed after {self.max_retries} attempts: {last_err}")

    def judge(self, payload: JudgeDecisionPayload) -> JudgeVerdict:
        prompt = build_ollama_prompt_record(payload)
        messages = [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": prompt.user_prompt},
        ]
        candidate_names = {candidate.candidate_id for candidate in payload.candidates}
        parsed, raw_response = self._chat(messages)
        self._last_trace = JudgeCallTrace(
            prompt=prompt,
            raw_model_response=raw_response,
            parsed_response=parsed,
        )
        raw_winner = (
            parsed.get("winner_candidate_id")
            or parsed.get("final_winner_candidate_id")
            or parsed.get("winner")
            or parsed.get("candidate")
        )
        if raw_winner is None:
            raise ValueError("winner_candidate_id is required")
        winner_id = _normalize_candidate_name(str(raw_winner), candidate_names=candidate_names)
        if "confidence" not in parsed:
            raise ValueError("confidence is required")
        return JudgeVerdict(
            winner_candidate_id=winner_id,
            confidence=_normalize_confidence(parsed["confidence"]),
            decision_basis=str(parsed.get("decision_basis", "")).strip(),
            ranking=tuple(),
            reason=str(parsed.get("reason", "")),
            used_tie_break=bool(parsed.get("used_tie_break", False)),
            decisive_signals=(),
        )


def _sanitize_model_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", name)


def _normalize_load_value(value: float) -> float:
    normalized = round(float(value), 10)
    nearest_int = round(normalized)
    if math.isclose(normalized, nearest_int, rel_tol=0.0, abs_tol=1e-10):
        return float(nearest_int)
    return normalized


def _format_load_label(value: float) -> str:
    normalized = _normalize_load_value(value)
    if float(normalized).is_integer():
        return str(int(normalized))
    return str(normalized).replace("-", "neg_").replace(".", "_")


def _resolve_loads(*, args: Any, raw: dict[str, Any]) -> tuple[float, ...]:
    cli_range_values = (args.load_start, args.load_end, args.load_step)
    if args.load is not None and any(value is not None for value in cli_range_values):
        raise ValueError("use either --load or --load-start/--load-end/--load-step")
    if any(value is not None for value in cli_range_values):
        if not all(value is not None for value in cli_range_values):
            raise ValueError("provide --load-start, --load-end, and --load-step together")
        return LoadSweepConfig(
            start=float(args.load_start),
            end=float(args.load_end),
            step=float(args.load_step),
        ).build_loads()
    if args.load is not None:
        return (_normalize_load_value(float(args.load)),)

    raw_range_values = (raw.get("load_start"), raw.get("load_end"), raw.get("load_step"))
    if raw.get("load") is not None and any(value is not None for value in raw_range_values):
        raise ValueError("model config must define either 'load' or 'load_start/load_end/load_step'")
    if any(value is not None for value in raw_range_values):
        if not all(value is not None for value in raw_range_values):
            raise ValueError("model config must define load_start, load_end, and load_step together")
        return LoadSweepConfig(
            start=float(raw["load_start"]),
            end=float(raw["load_end"]),
            step=float(raw["load_step"]),
        ).build_loads()
    return (_normalize_load_value(float(raw.get("load", 320.0))),)


def _append_heuristic_summary_rows(
    *,
    merged_rows: list[dict[str, Any]],
    payload: dict[str, Any],
    date_label: str,
) -> None:
    for load_entry in payload.get("loads", []):
        artifacts = dict(load_entry.get("artifacts", {}))
        load_value = _normalize_load_value(float(load_entry["load"]))
        for heuristic_entry in load_entry.get("heuristics", []):
            service_blocking_rate = float(heuristic_entry["service_blocking_rate_mean"])
            merged_rows.append(
                {
                    "runner_type": "heuristic",
                    "model_name": "",
                    "heuristic_name": str(heuristic_entry["heuristic_name"]),
                    "load": str(load_value),
                    "date": date_label,
                    "scope": "heuristic_baseline",
                    "episode_index": "-1",
                    "comparison_service_blocking_rate": str(service_blocking_rate),
                    "service_blocking_rate_mean": str(service_blocking_rate),
                    "final_blocking_rate": str(service_blocking_rate),
                    "mean_episode_service_blocking_rate": str(service_blocking_rate),
                    "bit_rate_blocking_rate_mean": str(
                        heuristic_entry["bit_rate_blocking_rate_mean"]
                    ),
                    "final_disrupted_rate": str(heuristic_entry["disrupted_services_rate_mean"]),
                    "source_summary_csv": str(artifacts.get("summary_csv", "")),
                    "source_episodes_csv": str(artifacts.get("episodes_csv", "")),
                    "source_run_dir": str(artifacts.get("run_dir", "")),
                }
            )


def _write_heuristic_baseline_payload(
    *,
    run_dir: Path,
    date_label: str,
    loads: tuple[float, ...],
    profile: str,
    topology_id: str,
    seed: int,
    episode_count: int,
    episode_length: int,
) -> dict[str, Any]:
    heuristics_root = run_dir / "heuristics"
    heuristics_root.mkdir(parents=True, exist_ok=True)
    output_path = heuristics_root / f"{date_label}-heuristic-baseline.json"
    config = JudgeHeuristicSeedBaselineConfig(
        scenario_profile=profile,
        topology_id=topology_id,
        loads=loads,
        seed=seed,
        episode_count=episode_count,
        episode_length=episode_length,
        output_path=output_path,
        results_root=heuristics_root,
    )
    payload = build_heuristic_seed_baseline(config=config, online_module=dict(vars(_online)))
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run LLM judge across multiple models")
    parser.add_argument("--model-config", type=Path, required=True, help="model config JSON")
    parser.add_argument("--load", type=float, default=None, help="override single load")
    parser.add_argument("--load-start", type=float, default=None, help="initial load for inclusive sweep")
    parser.add_argument("--load-end", type=float, default=None, help="final load for inclusive sweep")
    parser.add_argument("--load-step", type=float, default=None, help="load step for inclusive sweep")
    parser.add_argument("--episode-count", type=int, default=None, help="override episode_count")
    parser.add_argument("--episode-length", type=int, default=None, help="override episode_length")
    parser.add_argument("--seed", type=int, default=None, help="override seed")
    parser.add_argument("--output-dir", type=Path, default=None, help="output directory")
    parser.add_argument(
        "--scenario-profile",
        choices=("legacy_benchmark", "ofc_v1", "graph_load"),
        default=None,
    )
    args = parser.parse_args()

    raw = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    models = [
        ModelConfig(
            name=str(model["name"]),
            base_url=str(model["base_url"]),
            model=str(model["model"]),
            api_key=str(model.get("api_key", "")),
            temperature=float(model.get("temperature", 0.0)),
        )
        for model in raw["models"]
    ]
    ep_len = int(args.episode_length) if args.episode_length else int(raw.get("episode_length", 1000))
    ep_count = int(args.episode_count) if args.episode_count else int(raw.get("episode_count", 5))
    seed = int(args.seed) if args.seed is not None else int(raw.get("seed", DEFAULT_SEED))
    loads = _resolve_loads(args=args, raw=raw)
    profile = args.scenario_profile or str(raw.get("scenario_profile", "legacy_benchmark"))
    topology_id = str(raw.get("topology_id", "nobel-eu"))
    out_dir = args.output_dir or SCRIPT_DIR

    date_label = _date_prefix()
    run_dir = out_dir / date_label
    run_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running {len(models)} model(s) plus heuristic baselines sequentially across loads={list(loads)}, "
        f"{ep_count} eps x {ep_len} steps"
    )
    for model in models:
        backend = "OpenAI-compatible" if model.api_key else "Ollama"
        print(f"  [{model.name}] {backend} - {model.model} @ {model.base_url}")

    merged_rows: list[dict[str, Any]] = []

    print("\n=== Heuristic baselines ===")
    heuristic_payload = _write_heuristic_baseline_payload(
        run_dir=run_dir,
        date_label=date_label,
        loads=loads,
        profile=profile,
        topology_id=topology_id,
        seed=seed,
        episode_count=ep_count,
        episode_length=ep_len,
    )
    _append_heuristic_summary_rows(
        merged_rows=merged_rows,
        payload=heuristic_payload,
        date_label=date_label,
    )
    print(f"Heuristic baseline artifacts: {run_dir / 'heuristics'}")

    for load in loads:
        load_label = _format_load_label(load)
        load_dir = run_dir / f"load_{load_label}"
        load_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Load {load} ===")

        for model_cfg in models:
            start = time.monotonic()
            safe_name = _sanitize_model_name(model_cfg.name)
            model_out = load_dir / safe_name
            if model_out.exists():
                shutil.rmtree(model_out)
            model_out.mkdir(parents=True, exist_ok=True)

            judge = ChatJudge(
                base_url=model_cfg.base_url,
                model=model_cfg.model,
                api_key=model_cfg.api_key,
                temperature=model_cfg.temperature,
            )
            experiment = LLMJudgeExperiment(
                episode_length=ep_len,
                episode_count=ep_count,
                seed=seed,
                load=load,
                scenario_profile=profile,
                output_dir=model_out,
            )

            print(f"\n[{model_cfg.name}] Starting - load={load}, {ep_count} episodes x {ep_len} steps")
            outputs: Any = run_experiment(experiment=experiment, judge=judge)
            elapsed = time.monotonic() - start
            print(f"[{model_cfg.name}] Done in {elapsed:.0f}s")

            total_steps = 0
            if outputs.summary_csv.exists():
                with outputs.summary_csv.open("r", encoding="utf-8") as file:
                    rows = list(csv.DictReader(file))
                total_steps = sum(int(row.get("steps", 0)) for row in rows if row.get("scope") == "episode")
                for row in rows:
                    comparison_service_blocking_rate = (
                        row.get("mean_episode_service_blocking_rate")
                        or row.get("final_blocking_rate")
                        or ""
                    )
                    enriched = {
                        "runner_type": "model",
                        "model_name": model_cfg.name,
                        "heuristic_name": "",
                        "load": str(load),
                        "comparison_service_blocking_rate": comparison_service_blocking_rate,
                        **row,
                    }
                    merged_rows.append(enriched)

            timing_row: dict[str, Any] = {
                "runner_type": "model",
                "model_name": model_cfg.name,
                "heuristic_name": "",
                "load": str(load),
                "date": date_label,
                "scope": "timing_only",
                "episode_index": "-1",
                "total_time_s": round(elapsed, 2),
                "avg_step_time_s": round(elapsed / max(1, total_steps), 4),
                "per_model_note": "timing_only",
            }
            merged_rows.append(timing_row)

    merged_path = run_dir / f"{date_label}-summary_all_models.csv"
    fieldnames: list[str] = []
    for row in merged_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for row in merged_rows:
        for key in fieldnames:
            row.setdefault(key, "")
    with merged_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"\nMerged summary: {merged_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from .heuristic_judge import (
    DecisiveSignal,
    JudgeCallTrace,
    JudgeDecisionPayload,
    JudgePromptRecord,
    JudgeVerdict,
)


@dataclass(frozen=True, slots=True)
class OllamaJudgeConfig:
    base_url: str
    model: str
    temperature: float = 0.0
    timeout_s: float = 20.0
    max_retries: int = 2


def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def load_ollama_judge_config(*, env_path: str | Path | None = None) -> OllamaJudgeConfig:
    if env_path is not None:
        _load_env_file(Path(env_path))
    base_url = os.environ.get("LLM_JUDGE_OLLAMA_BASE_URL", "").strip()
    model = os.environ.get("LLM_JUDGE_OLLAMA_MODEL", "").strip()
    if not base_url:
        raise RuntimeError("LLM_JUDGE_OLLAMA_BASE_URL is required to run the Ollama judge")
    if not model:
        raise RuntimeError("LLM_JUDGE_OLLAMA_MODEL is required to run the Ollama judge")
    return OllamaJudgeConfig(
        base_url=base_url,
        model=model,
        temperature=float(os.environ.get("LLM_JUDGE_TEMPERATURE", "0.0")),
        timeout_s=float(os.environ.get("LLM_JUDGE_TIMEOUT_S", "20.0")),
        max_retries=int(os.environ.get("LLM_JUDGE_MAX_RETRIES", "2")),
    )


def _build_system_prompt() -> str:
    return (
        "You are a rigorous evaluation judge for optical-networking heuristic comparison.\n"
        "Your primary objective is to minimize overall blocking risk for the current request and future requests.\n"
        "Treat QoT safety, fragmentation control, load concentration, and slot efficiency as signals about preserving future network capacity.\n"
        "Use only the JSON payload provided by the user.\n"
        "Ignore audit-oriented identifiers, internal ids, and any expectation of matching a deterministic baseline.\n"
        "Use route_summary to understand the selected route, modulation, and spectral width.\n"
        "In decisive_signals, use only importance values 'high' or 'medium'.\n"
        "In decisive_signals.supports, use only the exact heuristic_name of the supported candidate.\n"
        "Do not invent metrics, hidden assumptions, or physics not present in the payload.\n"
        "Return only the structured output."
    )


def _build_user_prompt(payload: JudgeDecisionPayload) -> str:
    return (
        "Select the candidate that best minimizes blocking risk now and for future requests.\n"
        "Priority order: non_reject, qot_safety, fragmentation, load_balance, slot_efficiency.\n"
        "Use route_summary to interpret the physical and spectral decision.\n"
        "For each decisive_signal, set supports to an exact heuristic_name from the payload.\n"
        "Return 1 to 3 decisive_signals and keep each importance as either high or medium.\n\n"
        f"{payload.to_prompt_json()}"
    )


def build_ollama_prompt_record(payload: JudgeDecisionPayload) -> JudgePromptRecord:
    return JudgePromptRecord(
        system_prompt=_build_system_prompt(),
        user_prompt=_build_user_prompt(payload),
    )


def _serialize_raw_model_response(message: Any) -> dict[str, object]:
    return {
        "content": getattr(message, "content", ""),
        "additional_kwargs": getattr(message, "additional_kwargs", {}),
        "response_metadata": getattr(message, "response_metadata", {}),
        "tool_calls": getattr(message, "tool_calls", []),
        "usage_metadata": getattr(message, "usage_metadata", None),
    }


def _build_verdict_from_response_mapping(
    response: dict[str, Any],
    *,
    candidate_names: set[str],
) -> JudgeVerdict:
    winner = str(response["winner"])
    ranking = tuple(str(name) for name in response["ranking"])
    if winner not in candidate_names:
        raise ValueError(f"winner {winner!r} is not one of the payload candidates")
    if any(name not in candidate_names for name in ranking):
        raise ValueError("ranking contains unknown candidate names")

    decisive_signals_raw = response.get("decisive_signals", [])
    decisive_signals: list[DecisiveSignal] = []
    for signal in decisive_signals_raw:
        factor = _normalize_factor(str(signal["factor"]))
        supports = _normalize_candidate_name(str(signal["supports"]), candidate_names=candidate_names)
        evidence = str(signal["evidence"])
        importance = _normalize_importance(str(signal["importance"]))
        decisive_signals.append(
            DecisiveSignal(
                factor=factor,
                supports=supports,
                evidence=evidence,
                importance=importance,
            )
        )

    return JudgeVerdict(
        winner=winner,
        confidence=float(response["confidence"]),
        ranking=ranking,
        reason=str(response["reason"]),
        used_tie_break=bool(response.get("used_tie_break", False)),
        decisive_signals=tuple(decisive_signals),
    )


def _normalize_factor(raw_factor: str) -> str:
    normalized = raw_factor.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "reject": "reject_status",
        "qot": "qot_safety",
        "qot_margin": "qot_safety",
        "load_balancing": "load_balance",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {
        "reject_status",
        "qot_safety",
        "fragmentation",
        "load_balance",
        "slot_efficiency",
    }:
        raise ValueError(f"unsupported decisive signal factor {raw_factor!r}")
    return normalized


def _normalize_importance(raw_importance: str) -> str:
    normalized = raw_importance.strip().lower()
    if "critical" in normalized or "high" in normalized:
        return "high"
    if "medium" in normalized or "moderate" in normalized or "low" in normalized:
        return "medium"
    raise ValueError(f"unsupported decisive signal importance {raw_importance!r}")


def _normalize_candidate_name(raw_supports: str, *, candidate_names: set[str]) -> str:
    normalized = raw_supports.strip()
    if normalized in candidate_names:
        return normalized

    lowered = normalized.lower()
    matching_names = [name for name in candidate_names if name.lower() in lowered]
    if len(matching_names) == 1:
        return matching_names[0]
    raise ValueError(f"decisive signal supports unknown candidate {raw_supports!r}")


class OllamaHeuristicJudge:
    def __init__(self, config: OllamaJudgeConfig) -> None:
        self.config = config
        self._last_trace: JudgeCallTrace | None = None

    @classmethod
    def from_env(cls, *, env_path: str | Path | None = None) -> "OllamaHeuristicJudge":
        return cls(load_ollama_judge_config(env_path=env_path))

    def consume_last_trace(self) -> JudgeCallTrace | None:
        trace = self._last_trace
        self._last_trace = None
        return trace

    def judge(self, payload: JudgeDecisionPayload) -> JudgeVerdict:
        try:
            from langchain_ollama import ChatOllama
            from pydantic import BaseModel, Field
        except ImportError as exc:
            raise RuntimeError(
                "LLM judge dependencies are missing. Install the optional llm dependencies first."
            ) from exc

        class DecisiveSignalModel(BaseModel):
            factor: str
            supports: str
            evidence: str
            importance: str

        class VerdictModel(BaseModel):
            winner: str
            confidence: float
            ranking: list[str]
            reason: str
            used_tie_break: bool = False
            decisive_signals: list[DecisiveSignalModel] = Field(min_length=1, max_length=3)

        prompt_record = build_ollama_prompt_record(payload)
        llm = ChatOllama(
            base_url=self.config.base_url,
            model=self.config.model,
            temperature=self.config.temperature,
            timeout=self.config.timeout_s,
        )
        structured_llm = llm.with_structured_output(VerdictModel, include_raw=True)
        last_error: Exception | None = None
        candidate_names = {candidate.heuristic_name for candidate in payload.candidates}

        for _attempt in range(max(1, self.config.max_retries)):
            try:
                response = structured_llm.invoke(
                    [
                        ("system", prompt_record.system_prompt),
                        ("human", prompt_record.user_prompt),
                    ]
                )
                raw_model_response = _serialize_raw_model_response(response.get("raw"))
                parsed_model = response.get("parsed")
                parsed_response = None if parsed_model is None else parsed_model.model_dump()
                self._last_trace = JudgeCallTrace(
                    prompt=prompt_record,
                    raw_model_response=raw_model_response,
                    parsed_response=parsed_response,
                )
                parsing_error = response.get("parsing_error")
                if parsing_error is not None:
                    raise parsing_error
                if parsed_response is None:
                    raise ValueError("structured response parsing returned no parsed payload")
                return _build_verdict_from_response_mapping(
                    parsed_response,
                    candidate_names=candidate_names,
                )
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"LLM judge failed after {self.config.max_retries} attempts: {last_error}")


__all__ = [
    "OllamaHeuristicJudge",
    "OllamaJudgeConfig",
    "build_ollama_prompt_record",
    "load_ollama_judge_config",
]

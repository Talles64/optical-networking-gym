from __future__ import annotations

from dataclasses import dataclass
import json as _json
import os
import re
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
    skip_explanation: bool = False
    think: bool | str = False  # False | "low" | "medium" | "high" | True


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
        skip_explanation=os.environ.get("LLM_JUDGE_SKIP_EXPLANATION", "").strip().lower()
        in ("1", "true", "yes"),
    )


def _build_system_prompt(*, skip_explanation: bool = False) -> str:
    base = (
        "You are an evaluation judge for optical-network resource allocation.\n"
        "Objective: select the candidate that minimizes blocking probability (current + future).\n"
        "Use only the JSON payload. Return only structured JSON.\n"
        "\n"
        "RUBRIC (priority order):\n"
        "1. reject_status: Always prefer non-reject. Reject = immediate blocking.\n"
        "2. qot_safety: Higher osnr_margin and lower worst_link_nli_share = lower failure risk.\n"
        "   osnr_margin: >3dB=safe, 1-3dB=moderate, <1dB=risky. worst_link_nli_share: <0.5=good, >0.8=stressed.\n"
        "3. fragmentation: Lower fragmentation_damage_num_blocks = less future blocking. Critical under high load.\n"
        "4. load_balance: Lower path_link_util_max = fewer hotspots. Higher path_common_free_ratio = more capacity.\n"
        "5. slot_efficiency: Fewer required_slots = less waste. Tiebreaker only.\n"
        "\n"
        "CONFIDENCE — lower it when there are trade-offs:\n"
        "- 0.85-1.0: Winner clearly dominates on the top relevant factor with no trade-off.\n"
        "- 0.6-0.85: Winner leads on one factor but loses on another (e.g., best qot but worst fragmentation).\n"
        "- 0.5-0.6: Very close or conflicting signals across factors. Set used_tie_break=true.\n"
        "\n"
        "DECISIVE SIGNALS — return 2-3 signals when factors point to different candidates.\n"
        "A signal supporting a non-winner shows the trade-off the winner had to overcome.\n"
        "factor: reject_status | qot_safety | fragmentation | load_balance | slot_efficiency\n"
        "importance: high | medium. supports: exact heuristic_name.\n"
        "\n"
        "RANKING: ALL heuristic_names from lowest to highest blocking risk."
    )
    if skip_explanation:
        base += (
            "\nOmit 'reason' and 'evidence' fields."
        )
    return base


_REGIME_GUIDANCE = {
    "light": "Low blocking risk now. Focus on qot_safety to avoid signal failures.",
    "moderate": "Moderate blocking risk. Balance qot_safety with fragmentation to prevent future blocking.",
    "high": "High blocking risk. Prioritize fragmentation and load_balance — the network is filling up.",
    "critical": "Critical blocking risk. Every allocation must minimize fragmentation and spread load.",
}


def _build_user_prompt(payload: JudgeDecisionPayload, *, skip_explanation: bool = False) -> str:
    load_regime = payload.global_regimes.load_regime
    qot_regime = payload.global_regimes.qot_pressure_regime
    frag_hint = payload.topology_context.fragmentation_risk_hint
    route_hint = payload.topology_context.route_length_regime
    guidance = _REGIME_GUIDANCE.get(load_regime, "")

    header = (
        f"Select the candidate with the LOWEST blocking risk.\n"
        f"Network: {load_regime} load, {qot_regime} QoT pressure. {guidance}\n"
        f"Topology: {route_hint} routes, {frag_hint} fragmentation risk.\n"
        f"If the winner is strong on one factor but weak on another, include signals for BOTH "
        f"(the winning factor AND the trade-off factor supporting another candidate).\n"
    )
    if skip_explanation:
        header += "Omit reason and evidence.\n"
    return f"{header}\n{payload.to_prompt_json()}"


def build_ollama_prompt_record(
    payload: JudgeDecisionPayload, *, skip_explanation: bool = False,
) -> JudgePromptRecord:
    return JudgePromptRecord(
        system_prompt=_build_system_prompt(skip_explanation=skip_explanation),
        user_prompt=_build_user_prompt(payload, skip_explanation=skip_explanation),
    )


def _serialize_raw_model_response(message: Any) -> dict[str, object]:
    return {
        "content": getattr(message, "content", ""),
        "additional_kwargs": getattr(message, "additional_kwargs", {}),
        "response_metadata": getattr(message, "response_metadata", {}),
        "tool_calls": getattr(message, "tool_calls", []),
        "usage_metadata": getattr(message, "usage_metadata", None),
    }


def _normalize_winner(raw_winner: str, *, candidate_names: set[str]) -> str:
    if raw_winner in candidate_names:
        return raw_winner
    return _normalize_candidate_name(raw_winner, candidate_names=candidate_names)


def _normalize_ranking(
    raw_ranking: list[Any],
    *,
    winner: str,
    candidate_names: set[str],
) -> tuple[str, ...]:
    normalized: list[str] = []
    for entry in raw_ranking:
        try:
            name = _normalize_candidate_name(str(entry), candidate_names=candidate_names)
            if name not in normalized:
                normalized.append(name)
        except ValueError:
            continue
    if not normalized or set(normalized) != candidate_names:
        normalized = [winner] + [n for n in candidate_names if n != winner]
    return tuple(normalized)


def _build_verdict_from_response_mapping(
    response: dict[str, Any],
    *,
    candidate_names: set[str],
) -> JudgeVerdict:
    winner = _normalize_winner(str(response["winner"]), candidate_names=candidate_names)

    # ranking: accept list or comma-separated string
    raw_ranking = response.get("ranking", [])
    if isinstance(raw_ranking, str):
        raw_ranking = [s.strip() for s in raw_ranking.split(",") if s.strip()]
    ranking = _normalize_ranking(
        raw_ranking,
        winner=winner,
        candidate_names=candidate_names,
    )

    # decisive_signals: accept list of dicts, list of strings, or pipe-separated string
    # Also accept "signals" as alias (some models shorten the field name)
    decisive_signals_raw = response.get("decisive_signals") or response.get("signals", [])
    if isinstance(decisive_signals_raw, str):
        decisive_signals_raw = [s.strip() for s in decisive_signals_raw.split("|") if s.strip()]

    decisive_signals: list[DecisiveSignal] = []
    _generic_to_candidate = {
        "best": winner,
        "winner": winner,
        "worst": ranking[-1] if ranking else winner,
        "loser": ranking[-1] if ranking else winner,
    }

    for signal in decisive_signals_raw:
        if isinstance(signal, str):
            # Flat format: "factor:heuristic_name:importance"
            parts = signal.split(":")
            if len(parts) < 3:
                continue
            raw_factor, raw_supports, raw_importance = parts[0], parts[1], parts[2]
            evidence = ""
        elif isinstance(signal, dict):
            raw_factor = str(signal["factor"])
            raw_supports = str(signal["supports"])
            raw_importance = str(signal["importance"])
            evidence = str(signal.get("evidence", ""))
        else:
            continue

        factor = _normalize_factor(raw_factor)
        supports_key = raw_supports.strip().lower()
        if supports_key in _generic_to_candidate:
            supports = _generic_to_candidate[supports_key]
        else:
            supports = _normalize_candidate_name(raw_supports, candidate_names=candidate_names)
        importance = _normalize_importance(raw_importance)
        decisive_signals.append(
            DecisiveSignal(
                factor=factor,
                supports=supports,
                evidence=evidence,
                importance=importance,
            )
        )

    if not decisive_signals:
        raise ValueError("at least one decisive signal is required")

    confidence = max(0.0, min(1.0, float(response["confidence"])))

    return JudgeVerdict(
        winner=winner,
        confidence=confidence,
        ranking=ranking,
        reason=str(response.get("reason", "")),
        used_tie_break=bool(response.get("used_tie_break", False)),
        decisive_signals=tuple(decisive_signals),
    )


_VALID_FACTORS = {
    "reject_status",
    "qot_safety",
    "fragmentation",
    "load_balance",
    "slot_efficiency",
}

_FACTOR_ALIASES = {
    "reject": "reject_status",
    "non_reject": "reject_status",
    "qot": "qot_safety",
    "qot_margin": "qot_safety",
    "osnr": "qot_safety",
    "osnr_margin": "qot_safety",
    "physical_safety": "qot_safety",
    "load_balancing": "load_balance",
    "load_concentration": "load_balance",
    "path_link_util": "load_balance",
    "path_link_util_mean": "load_balance",
    "path_link_util_max": "load_balance",
    "path_common_free_ratio": "load_balance",
    "fragmentation_damage": "fragmentation",
    "fragmentation_damage_num_blocks": "fragmentation",
    "fragmentation_damage_largest_block": "fragmentation",
    "fragmentation_risk": "fragmentation",
    "frag": "fragmentation",
    "slot": "slot_efficiency",
    "slots": "slot_efficiency",
    "required_slots": "slot_efficiency",
    "spectral_efficiency": "slot_efficiency",
    "efficiency": "slot_efficiency",
}


def _normalize_factor(raw_factor: str) -> str:
    normalized = raw_factor.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _FACTOR_ALIASES.get(normalized, normalized)
    if normalized in _VALID_FACTORS:
        return normalized
    # Fuzzy fallback: check if any valid factor is a substring
    for valid in _VALID_FACTORS:
        if valid in normalized or normalized in valid:
            return valid
    raise ValueError(f"unsupported decisive signal factor {raw_factor!r}")


def _normalize_importance(raw_importance: str) -> str:
    normalized = raw_importance.strip().lower()
    if "critical" in normalized or "high" in normalized:
        return "high"
    if "medium" in normalized or "moderate" in normalized or "low" in normalized:
        return "medium"
    raise ValueError(f"unsupported decisive signal importance {raw_importance!r}")


def _normalize_candidate_name(raw_supports: str, *, candidate_names: set[str]) -> str:
    stripped = raw_supports.strip()
    if stripped in candidate_names:
        return stripped

    lowered = stripped.lower()
    exact_lower = [n for n in candidate_names if n.lower() == lowered]
    if len(exact_lower) == 1:
        return exact_lower[0]

    substring = [n for n in candidate_names if n.lower() in lowered or lowered in n.lower()]
    if len(substring) == 1:
        return substring[0]
    raise ValueError(f"decisive signal supports unknown candidate {raw_supports!r}")


def _extract_json(text: str) -> dict[str, Any]:
    """Extract JSON from model response, stripping thinking tags if present."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return _json.loads(cleaned)
    except _json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        return _json.loads(match.group(0))
    raise ValueError(f"could not extract JSON from model response: {cleaned[:200]}")


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
            import ollama as _ollama
        except ImportError as exc:
            raise RuntimeError(
                "ollama package is missing. Run: pip install ollama"
            ) from exc

        skip = self.config.skip_explanation

        prompt_record = build_ollama_prompt_record(payload, skip_explanation=skip)

        # Flat JSON schema — no $defs/$ref — for reliable constrained decoding.
        _signal_props: dict[str, object] = {
            "factor": {"type": "string"},
            "supports": {"type": "string"},
            "importance": {"type": "string"},
        }
        if not skip:
            _signal_props["evidence"] = {"type": "string"}

        _verdict_props: dict[str, object] = {
            "winner": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "ranking": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "used_tie_break": {"type": "boolean"},
            "decisive_signals": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": _signal_props,
                    "required": list(_signal_props),
                },
                "minItems": 1,
                "maxItems": 3,
            },
        }
        if not skip:
            _verdict_props["reason"] = {"type": "string"}

        json_schema = {
            "type": "object",
            "properties": _verdict_props,
            "required": ["winner", "confidence", "ranking", "used_tie_break", "decisive_signals"],
        }

        client = _ollama.Client(
            host=self.config.base_url,
            timeout=self.config.timeout_s,
        )

        _messages = [
            {"role": "system", "content": prompt_record.system_prompt},
            {"role": "user", "content": prompt_record.user_prompt},
        ]

        candidate_names = {candidate.heuristic_name for candidate in payload.candidates}
        last_error: Exception | None = None

        for _attempt in range(max(1, self.config.max_retries)):
            try:
                response = client.chat(
                    model=self.config.model,
                    messages=_messages,
                    format=json_schema,
                    options={"temperature": self.config.temperature},
                    stream=False,
                    think=self.config.think,
                )
                raw_text = response.message.content or ""
                raw_model_response = {"content": raw_text}
                parsed_response = _extract_json(raw_text)

                self._last_trace = JudgeCallTrace(
                    prompt=prompt_record,
                    raw_model_response=raw_model_response,
                    parsed_response=parsed_response,
                )

                return _build_verdict_from_response_mapping(
                    parsed_response,
                    candidate_names=candidate_names,
                )
            except ValueError as ve:
                raise RuntimeError(f"LLM judge validation error: {ve}") from ve
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"LLM judge failed after {self.config.max_retries} attempts: {last_error}")


__all__ = [
    "OllamaHeuristicJudge",
    "OllamaJudgeConfig",
    "build_ollama_prompt_record",
    "load_ollama_judge_config",
]

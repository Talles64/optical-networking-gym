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


def _build_few_shot_examples() -> str:
    return (
        "<examples>\n"
        "<example name=\"A\">Minimum-slot route/common-free evidence is clearly stronger and the gain is material. "
        "Use decision_basis=same_slot_route_advantage.</example>\n"
        "<example name=\"B\">Minimum-slot candidates are close, route and local gains are small, and no structural "
        "axis clearly separates them. Keep extra_slot_override=false and use decision_basis=balanced_tie_break.</example>\n"
        "<example name=\"C\">same_path_slot_variant_band=material with pairwise_deltas.same_path_same_modulation=true "
        "means a slot-placement conflict on the same path and modulation, not a route story. Compare fragmentation and "
        "slot compactness first.</example>\n"
        "</examples>"
    )


def _build_system_prompt(*, skip_explanation: bool = False) -> str:
    del skip_explanation
    return (
        "<role>\n"
        "Choose the best optical-network allocation candidate using only the payload. Reject or unsafe candidates "
        "never win. Ignore candidate order, candidate_id, and do not imitate any hidden heuristic or ranking.\n"
        "</role>\n"
        "<decision_frame>\n"
        "1) Compare minimum-slot candidates first. same_slot_winner_candidate_id must come from the minimum required_slots "
        "group. extra_slot_override=true only when an extra-slot candidate has a clearly material structural gain.\n"
        "2) Judge trade-offs with a simple priority: spectral efficiency first, then future-feasibility preservation, "
        "then route/common-free slack, and QoT last unless safety is at risk.\n"
        "3) required_slots lower is better; fragmentation_added_blocks and largest_block_loss_slots lower are better; "
        "local_fragmentation and local_damage_score lower are better; route_pressure_score lower is better; "
        "path_common_free_ratio higher is better; qot_margin_clipped_db above 1.0 dB is usually only a tie-breaker.\n"
        "</decision_frame>\n"
        "<context>\n"
        "prompt_context, pairwise_deltas, and candidate_roles are descriptive scaffolding, not an answer key. "
        "candidate_roles may mark route, preservation, common-free, or slot-variant leaders; shared roles mean a tied-best "
        "set, not a forced winner. same_slot_near_tie_band, same_slot_damage_axes_tie_band, progress_ratio, congestion_band, "
        "and future_feasibility_risk_band describe how cautious you should be, not what to pick.\n"
        "same_slot_local_support_band tells whether a local-preservation story is structurally real. If it is none, do not "
        "promote a winner from raw local_damage_score alone. same_slot_route_common_free_alignment tells whether route and "
        "common-free evidence point to the same candidate or only define a tied set.\n"
        "If same_path_slot_variant_band is present and pairwise_deltas.same_path_same_modulation=true, the conflict is slot "
        "placement on the same path and modulation. Compare fragmentation_added_blocks, largest_block_loss_slots, "
        "local_fragmentation, delta_slot_span_total_norm, common_block_length_norm, and then smaller route.initial_slot as "
        "a last tie-breaker within the same-path family before telling a route story.\n"
        "</context>\n"
        "<guardrails>\n"
        "Use same_slot_route_advantage only when a minimum-slot route/common-free story is genuinely stronger, not when the "
        "gain is none, small, or still tied. Smaller route.initial_slot alone is not enough for same_slot_route_advantage. "
        "Use same_slot_local_advantage only when structural preservation clearly separates the winner, not from compactness "
        "hints or local_damage_score alone.\n"
        "When route and local stories are both weak, conflicting, or near-tied, use balanced_tie_break and judge the whole "
        "trade-off across fragmentation, spectral efficiency, and future slack. In tied structural cases, prefer "
        "path_common_free_ratio and route_pressure_score before QoT. support_count is a weak tie-breaker.\n"
        "decision_basis must name the dominant reason only.\n"
        "</guardrails>\n"
        f"{_build_few_shot_examples()}\n"
        "Return only JSON with keys same_slot_winner_candidate_id, extra_slot_override, "
        "winner_candidate_id, confidence, decision_basis, ranking, decisive_signals. "
        "ranking must contain candidate_id strings, best to worst. "
        "decisive_signals must contain at most 3 short strings in the form factor:candidate_id:low|medium|high. "
        "decision_basis must be one of same_slot_route_advantage, same_slot_local_advantage, "
        "extra_slot_structural_advantage, balanced_tie_break, qot_safety_override."
    )


def _build_user_prompt(payload: JudgeDecisionPayload, *, skip_explanation: bool = False) -> str:
    del skip_explanation
    return _json.dumps(payload.to_prompt_mapping(), separators=(",", ":"), ensure_ascii=True)


def _build_repair_user_prompt(
    payload: JudgeDecisionPayload,
    *,
    previous_verdict: JudgeVerdict,
    repair_issue: str,
) -> str:
    repair_payload = {
        "repair_note": (
            "Your previous JSON was semantically inconsistent with the payload. "
            "Return corrected JSON with the same schema. Keep winner_candidate_id, "
            "same_slot_winner_candidate_id, decision_basis, and ranking mutually consistent."
        ),
        "repair_issue": str(repair_issue),
        "previous_verdict": previous_verdict.to_mapping(),
        "payload": payload.to_prompt_mapping(),
    }
    return _json.dumps(repair_payload, separators=(",", ":"), ensure_ascii=True)


def build_ollama_prompt_record(
    payload: JudgeDecisionPayload,
    *,
    skip_explanation: bool = False,
    repair_issue: str | None = None,
    previous_verdict: JudgeVerdict | None = None,
) -> JudgePromptRecord:
    if repair_issue is not None and previous_verdict is None:
        raise ValueError("previous_verdict is required when repair_issue is provided")
    user_prompt = (
        _build_user_prompt(payload, skip_explanation=skip_explanation)
        if repair_issue is None
        else _build_repair_user_prompt(
            payload,
            previous_verdict=previous_verdict,
            repair_issue=repair_issue,
        )
    )
    return JudgePromptRecord(
        system_prompt=_build_system_prompt(skip_explanation=skip_explanation),
        user_prompt=user_prompt,
    )


def _normalize_candidate_name(raw_value: str, *, candidate_names: set[str]) -> str:
    stripped = raw_value.strip()
    if stripped in candidate_names:
        return stripped

    lowered = stripped.lower()
    exact_lower = [name for name in candidate_names if name.lower() == lowered]
    if len(exact_lower) == 1:
        return exact_lower[0]

    substring = [name for name in candidate_names if name.lower() in lowered or lowered in name.lower()]
    if len(substring) == 1:
        return substring[0]
    raise ValueError(f"unknown candidate id {raw_value!r}")


def _normalize_ranking(
    raw_ranking: list[Any] | str | None,
    *,
    winner_candidate_id: str,
    candidate_names: set[str],
) -> tuple[str, ...]:
    if raw_ranking is None:
        return ()
    if isinstance(raw_ranking, str):
        raw_entries = [entry.strip() for entry in raw_ranking.split(",") if entry.strip()]
    else:
        raw_entries = list(raw_ranking)
    normalized: list[str] = []
    for entry in raw_entries:
        try:
            candidate_id = _normalize_candidate_name(str(entry), candidate_names=candidate_names)
        except ValueError:
            continue
        if candidate_id not in normalized:
            normalized.append(candidate_id)
    if winner_candidate_id not in normalized:
        normalized.insert(0, winner_candidate_id)
    return tuple(normalized)


def _normalize_decisive_signals(
    raw_signals: object,
    *,
    candidate_names: set[str],
) -> tuple[DecisiveSignal, ...]:
    if not raw_signals:
        return ()
    if isinstance(raw_signals, str):
        raw_signal_items: list[object] = [raw_signals]
    elif isinstance(raw_signals, list):
        raw_signal_items = raw_signals
    else:
        return ()

    normalized: list[DecisiveSignal] = []
    for signal in raw_signal_items:
        if isinstance(signal, dict):
            supports = str(signal.get("supports", signal.get("support", ""))).strip()
            if not supports:
                continue
            try:
                normalized_supports = _normalize_candidate_name(supports, candidate_names=candidate_names)
            except ValueError:
                continue
            normalized.append(
                DecisiveSignal(
                    factor=str(signal.get("factor", "")).strip() or "unknown",
                    supports=normalized_supports,
                    evidence=str(signal.get("evidence", signal.get("description", ""))),
                    importance=str(signal.get("importance", "medium")).strip() or "medium",
                )
            )
        elif isinstance(signal, str):
            parts = [part.strip() for part in signal.split(":") if part.strip()]
            if len(parts) < 3:
                continue
            factor = parts[0]
            candidate_token = parts[1]
            importance = parts[2]
            try:
                normalized_supports = _normalize_candidate_name(candidate_token, candidate_names=candidate_names)
            except ValueError:
                try:
                    normalized_supports = _normalize_candidate_name(parts[0], candidate_names=candidate_names)
                except ValueError:
                    continue
                factor = parts[1]
                importance = parts[2]
            normalized.append(
                DecisiveSignal(
                    factor=factor,
                    supports=normalized_supports,
                    evidence="",
                    importance=importance,
                )
            )
    return tuple(normalized)


def _normalize_confidence(raw_confidence: object) -> float:
    if isinstance(raw_confidence, (int, float)):
        return max(0.0, min(1.0, float(raw_confidence)))
    text = str(raw_confidence).strip().lower()
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
        try:
            return max(0.0, min(1.0, float(text[:-1]) / 100.0))
        except ValueError as exc:
            raise ValueError(f"invalid confidence value {raw_confidence!r}") from exc
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError as exc:
        raise ValueError(f"invalid confidence value {raw_confidence!r}") from exc


def _first_list_mapping(response: list[Any]) -> dict[str, Any]:
    for item in response:
        if isinstance(item, dict):
            return item
    raise ValueError("legacy list response does not contain an object item")


def _build_verdict_from_response_mapping(
    response: object,
    *,
    candidate_names: set[str],
) -> JudgeVerdict:
    if isinstance(response, list):
        response_mapping = _first_list_mapping(response)
    elif isinstance(response, dict):
        response_mapping = response
    else:
        raise ValueError("response must be a JSON object or a list of JSON objects")

    raw_winner = response_mapping.get("winner_candidate_id")
    if raw_winner is None:
        raw_winner = response_mapping.get(
            "final_winner_candidate_id",
            response_mapping.get("winner", response_mapping.get("candidate")),
        )
    if raw_winner is None:
        raise ValueError("winner_candidate_id is required")
    winner_candidate_id = _normalize_candidate_name(str(raw_winner), candidate_names=candidate_names)

    if "confidence" not in response_mapping:
        raise ValueError("confidence is required")
    confidence = _normalize_confidence(response_mapping["confidence"])

    ranking = _normalize_ranking(
        response_mapping.get("ranking"),
        winner_candidate_id=winner_candidate_id,
        candidate_names=candidate_names,
    )
    decisive_signals = _normalize_decisive_signals(
        response_mapping.get("decisive_signals", response_mapping.get("signals")),
        candidate_names=candidate_names,
    )

    return JudgeVerdict(
        winner_candidate_id=winner_candidate_id,
        confidence=confidence,
        decision_basis=str(response_mapping.get("decision_basis", "")).strip(),
        ranking=ranking,
        reason=str(response_mapping.get("reason", "")),
        used_tie_break=bool(response_mapping.get("used_tie_break", False)),
        decisive_signals=decisive_signals,
    )


def _extract_json(text: str) -> object:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = "".join(char for char in cleaned if char.isprintable() or char in "\r\n\t")
    cleaned = cleaned.strip()
    if not cleaned:
        raise ValueError("could not extract JSON from empty model response")
    try:
        return _json.loads(cleaned)
    except _json.JSONDecodeError:
        pass

    decoder = _json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", cleaned):
        try:
            parsed, _index = decoder.raw_decode(cleaned[match.start() :])
        except _json.JSONDecodeError:
            continue
        return parsed
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

    def _chat_with_prompt_record(
        self,
        payload: JudgeDecisionPayload,
        *,
        prompt_record: JudgePromptRecord,
    ) -> JudgeVerdict:
        try:
            import ollama as _ollama
        except ImportError as exc:
            raise RuntimeError("ollama package is missing. Run: pip install ollama") from exc

        json_schema = {
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
                "ranking": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "decisive_signals": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 3,
                },
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

        client = _ollama.Client(host=self.config.base_url, timeout=self.config.timeout_s)
        messages = [
            {"role": "system", "content": prompt_record.system_prompt},
            {"role": "user", "content": prompt_record.user_prompt},
        ]
        candidate_names = {candidate.candidate_id for candidate in payload.candidates}
        last_error: Exception | None = None

        for _attempt in range(max(1, self.config.max_retries)):
            try:
                response = client.chat(
                    model=self.config.model,
                    messages=messages,
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
            except ValueError as exc:
                raise RuntimeError(f"LLM judge validation error: {exc}") from exc
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"LLM judge failed after {self.config.max_retries} attempts: {last_error}")

    def judge(self, payload: JudgeDecisionPayload) -> JudgeVerdict:
        prompt_record = build_ollama_prompt_record(payload, skip_explanation=self.config.skip_explanation)
        return self._chat_with_prompt_record(payload, prompt_record=prompt_record)

    def repair(
        self,
        payload: JudgeDecisionPayload,
        *,
        previous_verdict: JudgeVerdict,
        repair_issue: str,
    ) -> JudgeVerdict:
        prompt_record = build_ollama_prompt_record(
            payload,
            skip_explanation=self.config.skip_explanation,
            repair_issue=repair_issue,
            previous_verdict=previous_verdict,
        )
        return self._chat_with_prompt_record(payload, prompt_record=prompt_record)


__all__ = [
    "OllamaHeuristicJudge",
    "OllamaJudgeConfig",
    "build_ollama_prompt_record",
    "load_ollama_judge_config",
]

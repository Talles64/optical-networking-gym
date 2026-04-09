from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import runpy
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RESULTS_ROOT = REPO_ROOT / "optical_networking_gym_v2" / "examples" / "results"
DEFAULT_METHOD_RULES_PATH = SCRIPT_DIR / "ONLINE_JUDGE_METHOD_RULES.md"
DEFAULT_TRACKER_PATH = SCRIPT_DIR / "Multi-agent_JUDGE_CHANGE_TRACKER.md"
DEFAULT_BASELINE_PATH = SCRIPT_DIR / "current_judge_heuristic_seed_baseline.json"
DEFAULT_STATE_PATH = SCRIPT_DIR / "multi_agent_judge_state.json"


@dataclass(frozen=True, slots=True)
class MultiAgentJudgeCycleConfig:
    cycle_id: str
    target_load: float = 400.0
    scenario_profile: str = "legacy_benchmark"
    topology_id: str = "nobel-eu"
    baseline_loads: tuple[float, ...] = (400.0, 350.0)
    seed: int = 10
    episode_count: int = 1
    episode_length: int = 1000
    hypothesis: str = "bootstrap automated multi-agent cycle"
    results_root: Path = RESULTS_ROOT
    method_rules_path: Path = DEFAULT_METHOD_RULES_PATH
    tracker_path: Path = DEFAULT_TRACKER_PATH
    baseline_path: Path = DEFAULT_BASELINE_PATH
    state_path: Path = DEFAULT_STATE_PATH
    representative_case_limit: int = 10
    skip_judge_run: bool = False
    existing_run_dir: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "results_root", Path(self.results_root))
        object.__setattr__(self, "method_rules_path", Path(self.method_rules_path))
        object.__setattr__(self, "tracker_path", Path(self.tracker_path))
        object.__setattr__(self, "baseline_path", Path(self.baseline_path))
        object.__setattr__(self, "state_path", Path(self.state_path))
        object.__setattr__(self, "baseline_loads", tuple(float(load) for load in self.baseline_loads))
        object.__setattr__(self, "existing_run_dir", None if self.existing_run_dir is None else Path(self.existing_run_dir))
        if not self.cycle_id:
            raise ValueError("cycle_id must be non-empty")
        if float(self.target_load) <= 0:
            raise ValueError("target_load must be positive")
        if not self.baseline_loads:
            raise ValueError("baseline_loads must be non-empty")


def _load_script_module(path: Path) -> dict[str, Any]:
    return runpy.run_path(str(path))


def _online_module() -> dict[str, Any]:
    return _load_script_module(SCRIPT_DIR / "online_heuristic_judge.py")


def _baseline_module() -> dict[str, Any]:
    return _load_script_module(SCRIPT_DIR / "build_judge_heuristic_seed_baseline.py")


def _failure_pack_module() -> dict[str, Any]:
    return _load_script_module(SCRIPT_DIR / "build_judge_failure_pack.py")


def _brief_module() -> dict[str, Any]:
    return _load_script_module(SCRIPT_DIR / "render_codex_agent_brief.py")


def _scenario_snapshot_for_load(*, config: MultiAgentJudgeCycleConfig, online_module: dict[str, Any], load: float) -> dict[str, Any]:
    experiment_cls = online_module["LLMJudgeExperiment"]
    build_base_scenario = online_module["build_base_scenario"]
    experiment = experiment_cls(
        topology_id=config.topology_id,
        scenario_profile=config.scenario_profile,
        episode_count=config.episode_count,
        episode_length=config.episode_length,
        seed=config.seed,
        load=float(load),
    )
    scenario = build_base_scenario(experiment)
    return {
        "scenario_profile": config.scenario_profile,
        "topology_id": experiment.topology_id,
        "load": float(experiment.load),
        "episode_count": int(experiment.episode_count),
        "episode_length": int(experiment.episode_length),
        "seed": int(experiment.seed),
        "k_paths": int(scenario.k_paths),
        "launch_power_dbm": float(scenario.launch_power_dbm),
        "modulations_to_consider": int(scenario.modulations_to_consider),
        "bit_rates": [int(bit_rate) for bit_rate in scenario.bit_rates],
        "num_spectrum_resources": int(scenario.num_spectrum_resources),
        "mean_holding_time": float(scenario.mean_holding_time),
        "qot_constraint": str(scenario.qot_constraint),
        "measure_disruptions": bool(scenario.measure_disruptions),
        "drop_on_disruption": bool(scenario.drop_on_disruption),
        "max_span_length_km": float(scenario.max_span_length_km),
        "default_attenuation_db_per_km": float(scenario.default_attenuation_db_per_km),
        "default_noise_figure_db": float(scenario.default_noise_figure_db),
        "frequency_start": float(scenario.frequency_start),
        "frequency_slot_bandwidth": float(scenario.frequency_slot_bandwidth),
        "bandwidth": float(scenario.bandwidth),
        "margin": float(scenario.margin),
        "scenario_id": str(scenario.scenario_id),
        "modulations": [
            {
                "name": str(modulation.name),
                "maximum_length": float(modulation.maximum_length),
                "spectral_efficiency": int(modulation.spectral_efficiency),
                "minimum_osnr": float(modulation.minimum_osnr),
                "inband_xt": float(modulation.inband_xt),
            }
            for modulation in scenario.modulations
        ],
    }


def _heuristics_from_online_module(online_module: dict[str, Any]) -> list[str]:
    return [str(name) for name in online_module["HEURISTIC_ORDER"]]


def _render_method_rules_markdown(
    *,
    config: MultiAgentJudgeCycleConfig,
    official_snapshot: dict[str, Any],
    heuristics: list[str],
) -> str:
    lines = [
        "# Online Judge Method Rules",
        "",
        "## Permanent Rules",
        "",
        "- O judge LLM online continua sendo o decisor principal real sempre que houver mais de uma alternativa plausivel no prompt.",
        "- Scorer, fallback, shortlist, papeis dos candidatos e qualquer logica auxiliar nao podem virar answer key disfarcada para enviesar a escolha da LLM.",
        "- Melhorias do judge so contam como tal quando vierem de melhor entendimento do trade-off pelo proprio judge.",
        "- Mudancas permitidas para melhorar o judge:",
        "  - prompt / few-shots",
        "  - payload / contexto explicativo",
        "  - shortlist / plausibility sem pre-escolher o vencedor",
        "- Mudancas proibidas como mecanismo de melhoria do judge:",
        "  - override recorrente de controlador",
        "  - fallback ampliado para cobrir fraqueza sistematica",
        "  - scorer como referencia operacional no runtime",
        "  - future regret como answer key, repair ou sinal online",
        "  - amputacao agressiva de opcoes plausiveis apenas para empurrar um vencedor",
        "",
        "## Agent Contract",
        "",
        "- Sem convergencia entre Analyst e Reviewer, o ciclo termina sem mudanca.",
        "- O agente principal continua sendo o unico que altera codigo, roda testes, roda benchmark online e atualiza os documentos.",
        "",
        "### Analyst",
        "",
        "- O Analyst so pode diagnosticar onde o run comecou a degradar, quais erros dominaram e quais casos reais provam isso.",
        "- O Analyst nao pode propor solucao, patch, mudanca de camada ou heuristica alvo.",
        "- O Analyst pode usar future regret apenas como evidencia offline auxiliar para priorizar casos informativos.",
        "",
        "### Reviewer",
        "",
        "- O Reviewer so pode recomendar uma unica mudanca local e testavel.",
        "- O Reviewer so pode escolher uma camada por ciclo:",
        "  - prompt",
        "  - payload",
        "  - shortlist",
        "- O Reviewer deve justificar a recomendacao com casos reais dos logs.",
        "- O Reviewer deve explicar por que as outras camadas nao foram escolhidas.",
        "- O Reviewer deve listar efeitos colaterais proibidos antes de recomendar a mudanca.",
        "",
        "## Future Regret Rule",
        "",
        "- O future regret e estritamente offline.",
        "- Ele pode ser usado para:",
        "  - selecionar casos mais informativos",
        "  - priorizar investigacao",
        "  - apoiar a leitura de degradacao acumulada",
        "- Ele nao pode ser usado para:",
        "  - definir vencedor no runtime",
        "  - reparar resposta da LLM",
        "  - instruir shortlist a esconder alternativas reais",
        "  - funcionar como answer key paralela",
        "",
        "## Official Environment Snapshot",
        "",
        "- Este snapshot registra as configs completas do ambiente oficial antes da primeira run do fluxo multiagente.",
        f"- Runner oficial: `optical_networking_gym_v2/examples/llm/online_heuristic_judge.py`",
        f"- Scenario profile oficial: `{official_snapshot['scenario_profile']}`",
        f"- Topology id oficial: `{official_snapshot['topology_id']}`",
        "- Ordem oficial de aceitacao:",
        "  - primeiro `load=400`",
        "  - depois `load=350`",
        "- Configuracao oficial compartilhada entre as duas cargas:",
        f"  - `episode_count={official_snapshot['episode_count']}`",
        f"  - `episode_length={official_snapshot['episode_length']}`",
        f"  - `seed={official_snapshot['seed']}`",
        f"  - `k_paths={official_snapshot['k_paths']}`",
        f"  - `launch_power_dbm={official_snapshot['launch_power_dbm']}`",
        f"  - `modulations_to_consider={official_snapshot['modulations_to_consider']}`",
        f"  - `mean_holding_time={official_snapshot['mean_holding_time']}`",
        f"  - `num_spectrum_resources={official_snapshot['num_spectrum_resources']}`",
        f"  - `bit_rates={tuple(official_snapshot['bit_rates'])}`",
        f"  - `qot_constraint=\"{official_snapshot['qot_constraint']}\"`",
        f"  - `measure_disruptions={official_snapshot['measure_disruptions']}`",
        f"  - `drop_on_disruption={official_snapshot['drop_on_disruption']}`",
        f"  - `max_span_length_km={official_snapshot['max_span_length_km']}`",
        f"  - `default_attenuation_db_per_km={official_snapshot['default_attenuation_db_per_km']}`",
        f"  - `default_noise_figure_db={official_snapshot['default_noise_figure_db']}`",
        f"  - `frequency_start={official_snapshot['frequency_start']}`",
        f"  - `frequency_slot_bandwidth={official_snapshot['frequency_slot_bandwidth']}`",
        f"  - `bandwidth={official_snapshot['bandwidth']}`",
        f"  - `margin={official_snapshot['margin']}`",
        "- Snapshot explicito da primeira run do fluxo:",
        f"  - `load={official_snapshot['load']}`",
        f"  - `scenario_id=\"{official_snapshot['scenario_id']}\"`",
        f"  - `heuristic_baseline_json=\"{config.baseline_path}\"`",
        f"  - `heuristic_baseline_results_root=\"{config.results_root / 'heuristic_seed_baseline'}\"`",
        "",
        "### Official Modulations",
        "",
    ]
    for modulation in official_snapshot["modulations"]:
        lines.append(
            f"- `{modulation['name']}`: `maximum_length={modulation['maximum_length']}`, "
            f"`spectral_efficiency={modulation['spectral_efficiency']}`, "
            f"`minimum_osnr={modulation['minimum_osnr']}`, `inband_xt={modulation['inband_xt']}`"
        )
    lines.extend(
        [
            "",
            "## Official Heuristic Set",
            "",
            "- O conjunto atual de heuristicas candidatas do runner e:",
        ]
    )
    for heuristic_name in heuristics:
        lines.append(f"  - `{heuristic_name}`")
    lines.extend(
        [
            "- Qualquer baseline ou snapshot do fluxo multiagente deve usar exatamente esse conjunto.",
            "",
            "## Acceptance Order",
            "",
            "- O benchmark oficial aceito continua sendo apenas o run online do judge.",
            "- A ordem do loop e:",
            "  - bater a melhor heuristica em `load=400`",
            "  - depois bater a melhor heuristica em `load=350`",
            "- O snapshot de baseline heuristica precisa ser refeito apenas quando o cenario oficial ou o conjunto de heuristicas mudar.",
        ]
    )
    return "\n".join(lines) + "\n"


def _baseline_summary_rows(baseline_payload: dict[str, Any]) -> list[dict[str, Any]]:
    return list(baseline_payload.get("loads", []))


def _render_tracker_markdown(*, state: dict[str, Any]) -> str:
    baseline_results_root = str(state["baseline"].get("results_root", "")).strip()
    lines = [
        "# Multi-agent Judge Change Tracker",
        "",
        "## Scope",
        "",
        "- Este arquivo e o diario operacional exclusivo do fluxo multiagente no Codex.",
        "- O benchmark oficial continua sendo apenas o runner online em `optical_networking_gym_v2/examples/llm/online_heuristic_judge.py`.",
        "- O tracker antigo `ONLINE_JUDGE_CHANGE_TRACKER.md` permanece como historico do loop anterior.",
        "- Regras permanentes do novo fluxo ficam em `optical_networking_gym_v2/examples/llm/ONLINE_JUDGE_METHOD_RULES.md`.",
        "",
        "## Initial Baseline Snapshot",
        "",
        f"- Snapshot gerado por `optical_networking_gym_v2/examples/llm/build_judge_heuristic_seed_baseline.py`",
        f"- Artefato fonte: `optical_networking_gym_v2/examples/llm/current_judge_heuristic_seed_baseline.json`",
    ]
    if baseline_results_root:
        lines.append(f"- Diretorio de artefatos do baseline: `{baseline_results_root}`")
    lines.extend(
        [
            "- Config comum do snapshot:",
            f"  - `scenario_profile={state['baseline']['scenario_profile']}`",
            f"  - `topology_id={state['baseline']['topology_id']}`",
            f"  - `seed={state['baseline']['seed']}`",
            f"  - `episode_count={state['baseline']['episode_count']}`",
            f"  - `episode_length={state['baseline']['episode_length']}`",
            "  - heuristicas avaliadas:",
        ]
    )
    for heuristic_name in state["baseline"]["heuristics"]:
        lines.append(f"    - `{heuristic_name}`")
    for load_entry in state["baseline"]["loads"]:
        artifacts = dict(load_entry.get("artifacts") or {})
        lines.extend(
            [
                "",
                f"### Load {int(float(load_entry['load'])) if float(load_entry['load']).is_integer() else load_entry['load']}",
                "",
                "| heuristica | service_blocking_rate |",
                "| --- | ---: |",
            ]
        )
        for heuristic_row in load_entry["heuristics"]:
            lines.append(
                f"| `{heuristic_row['heuristic_name']}` | `{heuristic_row['service_blocking_rate_mean']:.3f}` |"
            )
        lines.extend(
            [
                "",
                f"- Melhor heuristica fixa atual em `load={int(float(load_entry['load'])) if float(load_entry['load']).is_integer() else load_entry['load']}`: "
                f"`{load_entry['best_heuristic']}` com `{load_entry['best_service_blocking_rate_mean']:.3f}`",
            ]
        )
        artifacts = load_entry.get("artifacts", {})
        if artifacts:
            lines.extend(
                [
                    f"- CSV resumo do baseline: `{artifacts.get('summary_csv', '')}`",
                    f"- CSV episodios do baseline: `{artifacts.get('episodes_csv', '')}`",
                ]
            )
    lines.extend(
        [
            "",
            "## Active Acceptance Order",
            "",
            "1. Bater a melhor heuristica fixa em `load=400`",
            "2. Depois repetir o mesmo criterio em `load=350`",
        ]
    )
    for cycle in state.get("cycles", []):
        artifacts = dict(cycle.get("artifacts") or {})
        lines.extend(
            [
                "",
                f"## Cycle {cycle['cycle_id']}",
                "",
                f"- `cycle_id`: `{cycle['cycle_id']}`",
                f"- `target_load`: `{cycle['target_load']}`",
                f"- `run_dir`: `{cycle['run_dir']}`",
                f"- `hypothesis`: {cycle['hypothesis']}",
                f"- `analyst_findings`: {cycle['analyst_findings']}",
                f"- `reviewer_recommendation`: {cycle['reviewer_recommendation']}",
                f"- `convergence_status`: `{cycle['convergence_status']}`",
                f"- `applied_change`: `{cycle['applied_change']}`",
                f"- `benchmark_result`: {cycle['benchmark_result']}",
                f"- `decision_next_step`: {cycle['decision_next_step']}",
            ]
        )
        if artifacts:
            lines.append("- `artifacts`:")
            for artifact_key in (
                "steps_csv",
                "summary_csv",
                "calls_jsonl",
                "failure_pack",
                "analyst_brief",
                "reviewer_brief",
                "analyst_review",
                "reviewer_review",
            ):
                artifact_value = str(artifacts.get(artifact_key, "")).strip()
                if artifact_value:
                    lines.append(f"  - `{artifact_key}={artifact_value}`")
    lines.extend(
        [
            "",
            "## Cycle Entry Template",
            "",
            "- `cycle_id`:",
            "- `target_load`:",
            "- `run_dir`:",
            "- `hypothesis`:",
            "- `analyst_findings`:",
            "- `reviewer_recommendation`:",
            "- `convergence_status`:",
            "- `applied_change` ou `no_change`:",
            "- `benchmark_result`:",
            "- `decision_next_step`:",
        ]
    )
    return "\n".join(lines) + "\n"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"cycles": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _load_optional_json(path_value: str) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_failure_mode(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else " " for character in value)
    return " ".join(token for token in normalized.split() if token)


def _failure_mode_tokens(value: str) -> set[str]:
    stop_tokens = {
        "that",
        "with",
        "this",
        "from",
        "into",
        "after",
        "under",
        "over",
        "when",
        "where",
        "while",
        "they",
        "them",
        "their",
        "real",
        "wrong",
        "drifts",
        "appears",
        "candidate",
        "candidates",
        "judge",
        "toward",
        "family",
    }
    return {
        token
        for token in _normalize_failure_mode(value).split()
        if len(token) >= 4 and token not in stop_tokens
    }


def _evidence_case_set(review: dict[str, Any] | None) -> set[str]:
    if review is None:
        return set()
    return {
        str(case_id).strip().lower()
        for case_id in review.get("evidence_case_ids", [])
        if str(case_id).strip()
    }


def _failure_mode_alignment(*, analyst_review: dict[str, Any], reviewer_review: dict[str, Any]) -> dict[str, Any]:
    analyst_mode = str(analyst_review.get("primary_failure_mode", "")).strip()
    reviewer_mode = str(reviewer_review.get("target_failure_mode", "")).strip()
    analyst_normalized = _normalize_failure_mode(analyst_mode)
    reviewer_normalized = _normalize_failure_mode(reviewer_mode)
    analyst_tokens = _failure_mode_tokens(analyst_mode)
    reviewer_tokens = _failure_mode_tokens(reviewer_mode)
    shared_tokens = analyst_tokens & reviewer_tokens
    analyst_evidence = _evidence_case_set(analyst_review)
    reviewer_evidence = _evidence_case_set(reviewer_review)
    shared_evidence = analyst_evidence & reviewer_evidence
    jaccard = 0.0
    union = analyst_tokens | reviewer_tokens
    if union:
        jaccard = len(shared_tokens) / len(union)
    direct_match = bool(analyst_normalized and reviewer_normalized) and (
        analyst_normalized == reviewer_normalized
        or analyst_normalized in reviewer_normalized
        or reviewer_normalized in analyst_normalized
    )
    semantic_match = (
        (len(shared_tokens) >= 3 and len(shared_evidence) >= 1)
        or (len(shared_tokens) >= 4 and jaccard >= 0.3)
        or direct_match
    )
    return {
        "direct_match": direct_match,
        "semantic_match": semantic_match,
        "shared_tokens": sorted(shared_tokens),
        "shared_evidence": sorted(shared_evidence),
        "jaccard": jaccard,
    }


def _agent_review_status(*, analyst_review: dict[str, Any] | None, reviewer_review: dict[str, Any] | None) -> dict[str, str]:
    if analyst_review is None or reviewer_review is None:
        return {
            "analyst_findings": "pending_agent_review",
            "reviewer_recommendation": "pending_agent_review",
            "convergence_status": "pending_agent_review",
            "decision_next_step": "spawn Analyst + Reviewer on generated briefs and update cycle with convergence decision",
        }

    analyst_mode = str(analyst_review.get("primary_failure_mode", "")).strip()
    reviewer_mode = str(reviewer_review.get("target_failure_mode", "")).strip()
    analyst_summary = (
        f"primary_failure_mode={analyst_mode}; evidence={','.join(str(item) for item in analyst_review.get('evidence_case_ids', []))}"
    )
    reviewer_summary = (
        f"layer={reviewer_review.get('allowed_change_layer', '')}; "
        f"change={reviewer_review.get('recommended_single_change', '')}"
    )

    alignment = _failure_mode_alignment(
        analyst_review=analyst_review,
        reviewer_review=reviewer_review,
    )
    if not alignment["semantic_match"]:
        return {
            "analyst_findings": analyst_summary,
            "reviewer_recommendation": reviewer_summary,
            "convergence_status": "no_convergence",
            "decision_next_step": "no_change; analyst and reviewer did not converge on the same failure mode",
        }
    if str(reviewer_review.get("allowed_change_layer", "")).strip() == "no_change":
        return {
            "analyst_findings": analyst_summary,
            "reviewer_recommendation": reviewer_summary,
            "convergence_status": "converged_no_change",
            "decision_next_step": "no_change; reviewer found no safe single change for the converged failure mode",
        }
    return {
        "analyst_findings": analyst_summary,
        "reviewer_recommendation": reviewer_summary,
        "convergence_status": "converged_ready_for_patch",
        "decision_next_step": "apply the single recommended change, run narrow tests, and rerun the official benchmark",
    }


def _run_baseline(*, config: MultiAgentJudgeCycleConfig) -> dict[str, Any]:
    baseline_module = _baseline_module()
    config_cls = baseline_module["JudgeHeuristicSeedBaselineConfig"]
    write_baseline = baseline_module["write_heuristic_seed_baseline"]
    payload_path = write_baseline(
        config=config_cls(
            scenario_profile=config.scenario_profile,
            topology_id=config.topology_id,
            loads=config.baseline_loads,
            seed=config.seed,
            episode_count=config.episode_count,
            episode_length=config.episode_length,
            output_path=config.baseline_path,
            results_root=config.results_root / "heuristic_seed_baseline",
        )
    )
    return json.loads(Path(payload_path).read_text(encoding="utf-8"))


def _cycle_run_dir(*, config: MultiAgentJudgeCycleConfig) -> Path:
    if config.existing_run_dir is not None:
        return config.existing_run_dir
    load_label = int(config.target_load) if float(config.target_load).is_integer() else str(config.target_load).replace(".", "p")
    return config.results_root / f"judge_multiagent_load{load_label}_cycle{config.cycle_id}"


def _resolve_existing_judge_outputs(*, run_dir: Path) -> dict[str, str]:
    summary_csv_matches = sorted(run_dir.glob("*-llm-judge-summary.csv"))
    steps_csv_matches = sorted(run_dir.glob("*-llm-judge-steps.csv"))
    calls_jsonl_matches = sorted(run_dir.glob("*-llm-judge-calls.jsonl"))
    if len(summary_csv_matches) != 1 or len(steps_csv_matches) != 1 or len(calls_jsonl_matches) != 1:
        raise FileNotFoundError(
            f"expected one summary, one steps and one calls file in {run_dir}; "
            f"found summary={len(summary_csv_matches)} steps={len(steps_csv_matches)} calls={len(calls_jsonl_matches)}"
        )
    return {
        "steps_csv": str(steps_csv_matches[0]),
        "summary_csv": str(summary_csv_matches[0]),
        "calls_jsonl": str(calls_jsonl_matches[0]),
    }


def _run_judge(*, config: MultiAgentJudgeCycleConfig, run_dir: Path) -> dict[str, str]:
    online_module = _online_module()
    experiment_cls = online_module["LLMJudgeExperiment"]
    run_experiment = online_module["run_experiment"]
    outputs = run_experiment(
        experiment=experiment_cls(
            topology_id=config.topology_id,
            scenario_profile=config.scenario_profile,
            episode_count=config.episode_count,
            episode_length=config.episode_length,
            seed=config.seed,
            load=config.target_load,
            output_dir=run_dir,
        )
    )
    return {
        "steps_csv": str(outputs.steps_csv),
        "summary_csv": str(outputs.summary_csv),
        "calls_jsonl": str(outputs.calls_jsonl),
    }


def _build_failure_pack_and_briefs(*, config: MultiAgentJudgeCycleConfig, run_dir: Path) -> dict[str, str]:
    failure_pack_module = _failure_pack_module()
    brief_module = _brief_module()

    failure_pack_config_cls = failure_pack_module["JudgeFailurePackConfig"]
    write_failure_pack = failure_pack_module["write_failure_pack"]
    brief_config_cls = brief_module["AgentBriefConfig"]
    write_agent_brief = brief_module["write_agent_brief"]

    failure_pack_path = write_failure_pack(
        config=failure_pack_config_cls(
            run_dir=run_dir,
            method_rules_path=config.method_rules_path,
            baseline_path=config.baseline_path,
            representative_case_limit=config.representative_case_limit,
        )
    )
    analyst_brief_path = write_agent_brief(
        config=brief_config_cls(
            failure_pack_path=failure_pack_path,
            role="analyst",
            output_path=run_dir / "analyst_brief.json",
        )
    )
    reviewer_brief_path = write_agent_brief(
        config=brief_config_cls(
            failure_pack_path=failure_pack_path,
            role="reviewer",
            output_path=run_dir / "reviewer_brief.json",
        )
    )
    analyst_review_path = run_dir / "analyst_review.json"
    reviewer_review_path = run_dir / "reviewer_review.json"
    return {
        "failure_pack": str(failure_pack_path),
        "analyst_brief": str(analyst_brief_path),
        "reviewer_brief": str(reviewer_brief_path),
        "analyst_review": str(analyst_review_path) if analyst_review_path.exists() else "",
        "reviewer_review": str(reviewer_review_path) if reviewer_review_path.exists() else "",
    }


def run_multiagent_cycle(*, config: MultiAgentJudgeCycleConfig) -> dict[str, Any]:
    online_module = _online_module()
    heuristics = _heuristics_from_online_module(online_module)
    official_snapshot = _scenario_snapshot_for_load(
        config=config,
        online_module=online_module,
        load=float(config.target_load),
    )
    config.method_rules_path.parent.mkdir(parents=True, exist_ok=True)
    config.method_rules_path.write_text(
        _render_method_rules_markdown(
            config=config,
            official_snapshot=official_snapshot,
            heuristics=heuristics,
        ),
        encoding="utf-8",
    )

    baseline_payload = _run_baseline(config=config)
    run_dir = _cycle_run_dir(config=config)
    run_dir.mkdir(parents=True, exist_ok=True)
    judge_outputs = (
        _resolve_existing_judge_outputs(run_dir=run_dir)
        if config.skip_judge_run
        else _run_judge(config=config, run_dir=run_dir)
    )
    artifacts = _build_failure_pack_and_briefs(config=config, run_dir=run_dir)
    failure_pack = json.loads(Path(artifacts["failure_pack"]).read_text(encoding="utf-8"))
    review_status = _agent_review_status(
        analyst_review=_load_optional_json(artifacts.get("analyst_review", "")),
        reviewer_review=_load_optional_json(artifacts.get("reviewer_review", "")),
    )

    state = _load_state(config.state_path)
    state["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["baseline"] = baseline_payload
    cycle_entry = {
        "cycle_id": str(config.cycle_id),
        "target_load": float(config.target_load),
        "run_dir": str(run_dir),
        "hypothesis": str(config.hypothesis),
        "analyst_findings": review_status["analyst_findings"],
        "reviewer_recommendation": review_status["reviewer_recommendation"],
        "convergence_status": review_status["convergence_status"],
        "applied_change": "none",
        "benchmark_result": (
            f"final_blocking_rate={failure_pack['run_summary']['final_blocking_rate']:.3f}; "
            f"llm_calls={failure_pack['run_summary']['llm_calls']}; "
            f"fallback_count={failure_pack['run_summary']['fallback_count']}"
        ),
        "decision_next_step": review_status["decision_next_step"],
        "artifacts": {
            **judge_outputs,
            **artifacts,
        },
    }
    existing_cycles = [cycle for cycle in state.get("cycles", []) if str(cycle.get("cycle_id")) != str(config.cycle_id)]
    existing_cycles.append(cycle_entry)
    state["cycles"] = existing_cycles
    _write_json(config.state_path, state)
    config.tracker_path.write_text(_render_tracker_markdown(state=state), encoding="utf-8")
    return {
        "method_rules_path": str(config.method_rules_path),
        "baseline_path": str(config.baseline_path),
        "tracker_path": str(config.tracker_path),
        "state_path": str(config.state_path),
        "run_dir": str(run_dir),
        **judge_outputs,
        **artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Automate a full multi-agent judge cycle bootstrap around the online benchmark")
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--target-load", type=float, default=400.0)
    parser.add_argument("--scenario-profile", default="legacy_benchmark")
    parser.add_argument("--topology-id", default="nobel-eu")
    parser.add_argument("--baseline-loads", default="400,350")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--episode-count", type=int, default=1)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--hypothesis", default="bootstrap automated multi-agent cycle")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--method-rules-path", type=Path, default=DEFAULT_METHOD_RULES_PATH)
    parser.add_argument("--tracker-path", type=Path, default=DEFAULT_TRACKER_PATH)
    parser.add_argument("--baseline-path", type=Path, default=DEFAULT_BASELINE_PATH)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--representative-case-limit", type=int, default=10)
    parser.add_argument("--skip-judge-run", action="store_true")
    parser.add_argument("--existing-run-dir", type=Path, default=None)
    args = parser.parse_args()

    baseline_loads = tuple(float(token) for token in str(args.baseline_loads).split(",") if token.strip())
    outputs = run_multiagent_cycle(
        config=MultiAgentJudgeCycleConfig(
            cycle_id=str(args.cycle_id),
            target_load=float(args.target_load),
            scenario_profile=str(args.scenario_profile),
            topology_id=str(args.topology_id),
            baseline_loads=baseline_loads,
            seed=int(args.seed),
            episode_count=int(args.episode_count),
            episode_length=int(args.episode_length),
            hypothesis=str(args.hypothesis),
            results_root=args.results_root,
            method_rules_path=args.method_rules_path,
            tracker_path=args.tracker_path,
            baseline_path=args.baseline_path,
            state_path=args.state_path,
            representative_case_limit=int(args.representative_case_limit),
            skip_judge_run=bool(args.skip_judge_run),
            existing_run_dir=args.existing_run_dir,
        )
    )
    print(json.dumps(outputs, ensure_ascii=True))


if __name__ == "__main__":
    main()

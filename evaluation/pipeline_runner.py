"""Live pipeline execution for the paper's evaluation.

Three conditions share one artifact format so downstream analysis (scaffold build
rate, traceability coverage) runs identically over any of them:

    B2  run_pipeline(...)                     full system, all six agents
    B1  run_pipeline(..., skip_quality=True)  ablation, Quality agent removed
    B0  run_baseline(...)                     single LLM call for every artifact

Each run persists its artifacts and metrics under
``evaluation/runs/<condition>/project_<id>/`` so a run can be analysed, rebuilt,
or re-scored without re-spending API calls.

This module has no pytest dependency; it is importable from scripts and
notebooks. ``tests/test_integration_pipeline.py`` is a thin wrapper over it.

Instrumentation note
--------------------
Only the Requirements and Design agents record ``model_used`` internally, so the
execution path of the other four cannot be recovered from their output. We
therefore wrap each agent's fallback method before invoking it and count calls.
This is why ``fallbacks`` below is trustworthy for all six agents while
``model_used`` is not.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests as _requests

# Make the project root importable when run as a script.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.constants import DEFAULT_MODEL, DEFAULT_TEMPERATURE  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / "runs"

#: Artifact key -> filename, in pipeline order.
ARTIFACT_FILES = {
    "requirements": "requirements.json",
    "database_schema": "database_schema.json",
    "design": "design.json",
    "cycle_plan": "cycle_plan.json",
    "quality_report": "quality_report.json",
    "support_package": "support_package.json",
}


def _is_error(result: Any) -> bool:
    """Project-wide convention: a failed agent returns a dict containing 'error'."""
    return not isinstance(result, dict) or "error" in result


@dataclass
class PipelineResult:
    condition: str
    project_id: int
    domain: str
    complexity: str
    user_input: str
    artifacts: Dict[str, Any] = field(default_factory=dict)
    latency_s: Dict[str, float] = field(default_factory=dict)
    fallbacks: Dict[str, int] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    total_latency_s: float = 0.0

    @property
    def completed(self) -> bool:
        """True when every agent in this condition produced a non-error artifact.

        This is the sole basis for the Pipeline Completion Rate (PCR) reported in
        the paper. Do not compute PCR from any other source.
        """
        return not self.errors

    def save(self, runs_dir: Path = RUNS_DIR) -> Path:
        out = runs_dir / self.condition / f"project_{self.project_id}"
        out.mkdir(parents=True, exist_ok=True)
        for key, filename in ARTIFACT_FILES.items():
            if key in self.artifacts and not _is_error(self.artifacts[key]):
                (out / filename).write_text(
                    json.dumps(self.artifacts[key], indent=2), encoding="utf-8"
                )
        meta = asdict(self)
        meta.pop("artifacts", None)
        meta["completed"] = self.completed
        (out / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out


def _instrument_fallback(agent: Any, label: str, counters: Dict[str, int]) -> None:
    """Count direct-API fallback invocations on a live agent instance.

    Wrapping the bound method is deliberate: it works without modifying agent
    source, and it captures fallbacks triggered deep inside retry loops that
    never surface in the returned artifact.
    """
    for attr in ("_direct_api_call", "_direct_design_fallback"):
        original: Optional[Callable] = getattr(agent, attr, None)
        if original is None:
            continue

        def make_wrapper(func: Callable, key: str) -> Callable:
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                counters[key] = counters.get(key, 0) + 1
                return func(*args, **kwargs)
            return wrapper

        setattr(agent, attr, make_wrapper(original, label))


def _timed(result_obj: PipelineResult, label: str, fn: Callable[[], Any]) -> Any:
    """Run ``fn``, recording wall-clock latency and any error under ``label``."""
    start = time.perf_counter()
    try:
        value = fn()
    except Exception as exc:  # an agent raising is itself a pipeline failure
        result_obj.latency_s[label] = round(time.perf_counter() - start, 2)
        result_obj.errors[label] = f"{type(exc).__name__}: {exc}"[:300]
        return {"error": str(exc)[:300]}
    result_obj.latency_s[label] = round(time.perf_counter() - start, 2)
    if _is_error(value):
        err = value.get("error") if isinstance(value, dict) else "non-dict result"
        result_obj.errors[label] = str(err)[:300]
    return value


# ---------------------------------------------------------------------------
# B2 / B1 -- the agent pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    project: Dict[str, Any],
    api_key: str,
    skip_quality: bool = False,
    condition: Optional[str] = None,
) -> PipelineResult:
    """Drive all six pipeline agents end-to-end for one project input.

    Agents run in the canonical order declared by
    ``ProjectManagerAgent.DEFAULT_STEPS``: requirements, database, design,
    planning, quality, support. Execution continues after a failing agent so
    that one failure does not mask the others; downstream agents receive
    ``None`` for any artifact that failed, which is the same degradation the
    Streamlit UI applies.

    Set ``skip_quality=True`` for the B1 ablation.
    """
    from agents import (
        RequirementsAnalystAgent,
        DatabaseAgent,
        DesignGenerationAgent,
        PlanningManagerAgent,
        QualityManagerAgent,
        SupportManagerAgent,
    )

    condition = condition or ("B1" if skip_quality else "B2")
    res = PipelineResult(
        condition=condition,
        project_id=project["id"],
        domain=project["domain"],
        complexity=project["complexity"],
        user_input=project["input"],
    )

    run_start = time.perf_counter()

    # 1. Requirements -- note this agent takes no api_key; it reads the environment.
    req_agent = RequirementsAnalystAgent()
    _instrument_fallback(req_agent, "requirements", res.fallbacks)
    requirements = _timed(res, "requirements", lambda: req_agent.analyze_requirements(project["input"]))
    res.artifacts["requirements"] = requirements

    if _is_error(requirements):
        # Every later agent consumes requirements; without them the run is over.
        res.total_latency_s = round(time.perf_counter() - run_start, 2)
        return res

    # 2. Database
    db_agent = DatabaseAgent(api_key)
    _instrument_fallback(db_agent, "database", res.fallbacks)
    schema = _timed(res, "database", lambda: db_agent.generate_schema(requirements))
    res.artifacts["database_schema"] = schema
    schema_ok = None if _is_error(schema) else schema

    # 3. Design
    design_agent = DesignGenerationAgent(api_key)
    _instrument_fallback(design_agent, "design", res.fallbacks)
    design = _timed(res, "design", lambda: design_agent.generate_design(requirements))
    res.artifacts["design"] = design
    design_ok = None if _is_error(design) else design

    # 4. Planning
    plan_agent = PlanningManagerAgent(api_key)
    _instrument_fallback(plan_agent, "planning", res.fallbacks)
    plan = _timed(res, "planning", lambda: plan_agent.generate_cycle_plan(requirements, design_ok))
    res.artifacts["cycle_plan"] = plan
    plan_ok = None if _is_error(plan) else plan

    # 5. Quality -- omitted entirely in the B1 ablation.
    if not skip_quality:
        quality_agent = QualityManagerAgent(api_key)
        _instrument_fallback(quality_agent, "quality", res.fallbacks)
        quality = _timed(
            res, "quality",
            lambda: quality_agent.generate_quality_report(requirements, design_ok, schema_ok),
        )
        res.artifacts["quality_report"] = quality

    # 6. Support
    support_agent = SupportManagerAgent(api_key)
    _instrument_fallback(support_agent, "support", res.fallbacks)
    support = _timed(
        res, "support",
        lambda: support_agent.generate_support_package(requirements, design_ok, schema_ok, plan_ok),
    )
    res.artifacts["support_package"] = support

    res.total_latency_s = round(time.perf_counter() - run_start, 2)
    return res


# ---------------------------------------------------------------------------
# B3 -- native CrewAI multi-agent crew
# ---------------------------------------------------------------------------

def run_crew_pipeline(project: Dict[str, Any], api_key: str,
                      allow_delegation: bool = False) -> PipelineResult:
    """Condition B3: one crew, six agents, context-chained tasks, one kickoff.

    Same agents and same role framing as B2; the only difference is who owns
    the sequencing -- CrewAI here, ordinary Python there. That isolation is the
    point: a B2 vs B3 comparison measures the coordination strategy alone.

    Unlike B2 there is no per-agent fallback, so a mid-chain failure loses the
    whole run. Token usage, which B2 cannot report, is captured here.
    """
    from agents.crew_pipeline import UnifiedCrewPipeline

    res = PipelineResult(
        condition="B3",
        project_id=project["id"],
        domain=project["domain"],
        complexity=project["complexity"],
        user_input=project["input"],
    )

    pipeline = UnifiedCrewPipeline(api_key, allow_delegation=allow_delegation)
    if not pipeline.crew:
        res.errors["crew"] = "crew failed to initialize"
        return res

    outcome = pipeline.run(project["input"])
    res.latency_s = outcome.get("latency_s", {})
    res.total_latency_s = outcome.get("total_latency_s", 0.0)
    res.fallbacks.update(outcome.get("tokens", {}))

    if outcome.get("error"):
        res.errors["crew"] = outcome["error"]
        return res

    for key, artifact in outcome.get("artifacts", {}).items():
        res.artifacts[key] = artifact
        if _is_error(artifact):
            err = artifact.get("error") if isinstance(artifact, dict) else "non-dict"
            res.errors[key] = str(err)[:300]

    missing = [k for k in ARTIFACT_FILES if k not in res.artifacts]
    for key in missing:
        res.errors[key] = "task produced no output"

    return res


# ---------------------------------------------------------------------------
# B0 -- single-agent baseline
# ---------------------------------------------------------------------------

_BASELINE_PROMPT = """You are a software engineering assistant. From the project idea below, produce ALL of the following artifacts in a single JSON object. Output ONLY JSON, no markdown.

PROJECT IDEA:
{idea}

Output a JSON object with exactly these top-level keys:

"requirements": {{
  "project_name": string,
  "features": [{{"id": string, "name": string, "description": string, "priority": "High"|"Medium"|"Low",
                 "user_stories": [string], "acceptance_criteria": [string]}}],
  "constraints": [string],
  "non_functional_requirements": [string]
}}
"database_schema": {{
  "tables": [{{"name": string,
               "columns": [{{"name": string, "type": string, "pk": boolean, "nullable": boolean}}],
               "relationships": [string]}}]
}}
"design": {{
  "screens": [{{"name": string, "purpose": string,
                "key_components": [{{"name": string, "type": string, "interaction": string}}]}}],
  "color_scheme": {{"primary": "#RRGGBB", "secondary": "#RRGGBB", "background": "#RRGGBB",
                    "surface": "#RRGGBB", "text": "#RRGGBB"}},
  "typography": {{"heading_font": string, "body_font": string}}
}}
"cycle_plan": {{"tasks": [{{"name": string, "estimate_hours": number}}],
                "milestones": [string], "risks": [string]}}
"quality_report": {{"gate_decision": "PASS"|"FAIL",
                    "checklist": {{"requirements_complete": boolean, "design_consistent": boolean,
                                   "db_matches_requirements": boolean, "nfr_defined": boolean}},
                    "issues": [{{"severity": string, "item": string, "message": string}}],
                    "required_fixes": [string], "recommendations": [string]}}
"support_package": {{"documentation": [string], "baselines": [string],
                     "glossary": [{{"term": string, "definition": string}}]}}

Every table referenced by a screen must exist in database_schema. Every feature in requirements must be reachable from at least one screen."""


def run_baseline(
    project: Dict[str, Any],
    api_key: str,
    max_tokens: int = 16000,
    timeout: int = 600,
) -> PipelineResult:
    """Condition B0: one LLM call asked to produce every artifact at once.

    Uses the same model, temperature, and token budget as the pipeline so that
    the only difference between B0 and B2 is decomposition. The output is split
    into the same six artifact keys and written in the same layout, so the SBR
    harness and traceability scorer treat it identically.
    """
    from core.utils import parse_json_from_text

    res = PipelineResult(
        condition="B0",
        project_id=project["id"],
        domain=project["domain"],
        complexity=project["complexity"],
        user_input=project["input"],
    )

    start = time.perf_counter()
    try:
        resp = _requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": DEFAULT_MODEL,
                "max_tokens": max_tokens,
                "temperature": DEFAULT_TEMPERATURE,
                "messages": [{"role": "user", "content": _BASELINE_PROMPT.format(idea=project["input"])}],
            },
            timeout=timeout,
        )
        elapsed = round(time.perf_counter() - start, 2)
        res.latency_s["baseline"] = elapsed
        res.total_latency_s = elapsed

        if resp.status_code != 200:
            res.errors["baseline"] = f"API error {resp.status_code}"
            return res

        data = resp.json()
        usage = data.get("usage") or {}
        res.fallbacks["input_tokens"] = usage.get("input_tokens", 0)
        res.fallbacks["output_tokens"] = usage.get("output_tokens", 0)

        text = "\n".join(
            b.get("text", "") for b in (data.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
        parsed = parse_json_from_text(text)
        if _is_error(parsed):
            res.errors["baseline"] = str(parsed.get("error"))[:300]
            return res

        for key in ARTIFACT_FILES:
            value = parsed.get(key)
            if isinstance(value, dict) and value:
                res.artifacts[key] = value
            else:
                res.errors[key] = "missing or empty in baseline output"
    except Exception as exc:
        res.latency_s["baseline"] = round(time.perf_counter() - start, 2)
        res.total_latency_s = res.latency_s["baseline"]
        res.errors["baseline"] = f"{type(exc).__name__}: {exc}"[:300]

    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


#: Auth failures are environmental, never per-project. Repeating the sweep
#: cannot change the outcome, so we abort on the first one.
_AUTH_MARKERS = ("authentication", "401", "invalid x-api-key", "unauthorized")

#: Consecutive whole-run failures before the sweep gives up. Exhausted API
#: credits return HTTP 400 with the reason only in the response body, which the
#: agents discard -- so the status code alone cannot distinguish "out of credit"
#: from "bad prompt". Consecutive total failures are the reliable signal, and
#: aborting matters for correctness, not just cost: a sweep that runs to
#: completion through a billing outage silently reports a deflated completion
#: rate that looks like a property of the system.
_MAX_CONSECUTIVE_FAILURES = 2


def _key_problem(api_key: str) -> Optional[str]:
    """Reject an obviously unusable key before spending a sweep on it."""
    if not api_key:
        return "ANTHROPIC_API_KEY is not set (no shell value and none in .env)"
    if api_key == "test-api-key-12345":
        return "ANTHROPIC_API_KEY is the conftest stub, not a real key"
    if "..." in api_key or api_key.endswith("-"):
        return (
            f"ANTHROPIC_API_KEY looks like a placeholder, not a key: {api_key!r}\n"
            "  A literal 'sk-ant-...' from a copied example overrides .env, because\n"
            "  shell environment wins over the dotenv fallback. Unset it and let .env\n"
            "  supply the key:  unset ANTHROPIC_API_KEY"
        )
    if not api_key.startswith("sk-ant-") or len(api_key) < 40:
        return f"ANTHROPIC_API_KEY does not look like an Anthropic key (length {len(api_key)})"
    return None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the evaluation pipeline over the paper's project inputs.")
    parser.add_argument("--condition", choices=["B0", "B1", "B2", "B3"], default="B2",
                        help="B0 single prompt; B1 pipeline without Quality; "
                             "B2 orchestrated pipeline; B3 native CrewAI multi-agent crew")
    parser.add_argument("--delegation", action="store_true",
                        help="B3 only: give agents CrewAI's delegate/ask tools. Makes "
                             "execution order emergent; expect lower reliability.")
    parser.add_argument("--projects", type=int, nargs="*", help="Project ids to run (default: all)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Runs per project. Use >=3 for the mean+/-SD the paper needs.")
    args = parser.parse_args()

    _load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    problem = _key_problem(api_key)
    if problem:
        print(f"error: {problem}")
        return 1

    from evaluation.projects import PROJECTS

    selected = [p for p in PROJECTS if not args.projects or p["id"] in args.projects]
    completed = 0
    total = 0
    consecutive_failures = 0

    for repeat in range(args.repeats):
        for project in selected:
            total += 1
            label = f"{args.condition} project_{project['id']}"
            if args.repeats > 1:
                label += f" run_{repeat + 1}"
            print(f"\n=== {label}: {project['domain']} ===")

            if args.condition == "B0":
                result = run_baseline(project, api_key)
            elif args.condition == "B3":
                result = run_crew_pipeline(project, api_key, allow_delegation=args.delegation)
            else:
                result = run_pipeline(project, api_key, skip_quality=(args.condition == "B1"))

            if args.repeats > 1:
                result.condition = f"{args.condition}/run_{repeat + 1}"
            out = result.save()

            status = "COMPLETE" if result.completed else "INCOMPLETE"
            completed += int(result.completed)
            print(f"  {status}  {result.total_latency_s}s  -> {out}")
            for agent, seconds in result.latency_s.items():
                fb = result.fallbacks.get(agent, 0)
                note = f"  [fallback x{fb}]" if fb else ""
                print(f"    {agent:<14} {seconds:>7.1f}s{note}")
            for agent, err in result.errors.items():
                print(f"    FAILED {agent}: {err}")

            if any(m in err.lower() for err in result.errors.values() for m in _AUTH_MARKERS):
                print("\nerror: the API rejected the credentials. Aborting -- this is an\n"
                      "environment problem, so the remaining runs would fail identically.\n"
                      "Check that ANTHROPIC_API_KEY is a real key, then delete the partial\n"
                      f"results under evaluation/runs/{args.condition}/ before re-running.")
                return 1

            consecutive_failures = 0 if result.completed else consecutive_failures + 1
            if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                print(f"\nerror: {consecutive_failures} runs failed back to back. Aborting.\n"
                      "Back-to-back total failures are almost always environmental --\n"
                      "exhausted API credits (HTTP 400), rate limiting, or a network fault --\n"
                      "not a property of the system under test. Verify with:\n"
                      "  curl -s https://api.anthropic.com/v1/messages -H \"x-api-key: $ANTHROPIC_API_KEY\" \\\n"
                      "    -H 'anthropic-version: 2023-06-01' -H 'content-type: application/json' \\\n"
                      "    -d '{\"model\":\"claude-sonnet-4-5\",\"max_tokens\":1,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}'\n"
                      f"Then DELETE the affected runs under evaluation/runs/{args.condition}/\n"
                      "before re-running -- keeping them would deflate the completion rate.")
                return 1

    print(f"\nPipeline Completion Rate ({args.condition}): "
          f"{completed}/{total} = {100.0 * completed / total:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

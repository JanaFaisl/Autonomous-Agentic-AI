"""Native CrewAI multi-agent pipeline: ONE crew, six agents, chained tasks.

Contrast with the per-agent modules in this package. There, each agent owns a
private single-agent crew, and ordinary Python passes dictionaries from one to
the next -- CrewAI never sees a pipeline, only six unrelated one-agent jobs.
Here CrewAI owns the sequencing: every task declares ``context=[...]`` naming
the tasks whose output it consumes, and a single ``kickoff()`` runs the chain.

Three consequences worth knowing before using this path:

1. Task descriptions no longer interpolate upstream artifacts. In the per-agent
   modules the design prompt embeds the requirements JSON via an f-string; here
   the requirements arrive through ``context``, and the description only says
   what to do with them.

2. There is no per-agent fallback. The single-agent modules each degrade to a
   direct Anthropic call when orchestration fails. A crew-level kickoff either
   completes or raises, so a failure anywhere loses the whole run. This is a
   real robustness regression and the main cost of the change.

3. Token usage becomes measurable. ``CrewOutput.token_usage`` reports the whole
   chain, which the per-agent path cannot provide.

This module is additive: nothing in agents/*.py changes, so both coordination
strategies can be run and compared on identical inputs.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from core.constants import DEFAULT_MODEL, MAX_EXECUTION_TIME, DESIGN_MAX_EXECUTION_TIME
from core.llm import CREWAI_AVAILABLE, Agent, Task, Crew, Process, get_llm
from core.models import (
    PYDANTIC_AVAILABLE,
    RequirementsOutputModel,
    DatabaseSchemaModel,
    DesignOutputModel,
    CyclePlanOutputModel,
    QualityReportOutputModel,
    SupportGovernanceOutputModel,
)
from core.prompts import (
    UPSTREAM_FROM_CONTEXT,
    build_requirements_prompt,
    build_database_prompt,
    build_planning_prompt,
    build_quality_prompt,
    build_support_prompt,
)
from core.utils import parse_json_from_text
from utils.io_suppression import suppress_stderr, suppress_io

#: Artifact keys, in execution order. Matches the per-agent pipeline exactly so
#: downstream tooling (scaffold renderer, SBR harness) works on either path.
STEP_KEYS = ["requirements", "database_schema", "design", "cycle_plan",
             "quality_report", "support_package"]

_JSON_RULE = (
    "Output ONLY a single JSON object. No markdown fences, no commentary. "
    "Start with { and end with }."
)


def _agent_specs() -> List[Dict[str, Any]]:
    """Role/goal/backstory per agent, lifted verbatim from the federated modules.

    Generated from agents/*.py so the two topologies frame their agents
    identically. Role framing is a strong driver of LLM behaviour, so a
    paraphrase here would confound the B2/B3 comparison exactly as a
    reworded task prompt would. If you change a role in an agent module,
    re-sync this list.
    """
    return [
        {
            "key": 'requirements',
            "role": 'Software Requirements Analyst',
            "goal": 'Analyze user requirements and OUTPUT a complete JSON object with features, technical requirements, and user roles. You must produce the actual JSON output, not just think about it.',
            "backstory": 'You are an expert software requirements analyst with years of experience \n            in breaking down complex ideas into structured, actionable requirements. You excel at \n            identifying features, prioritizing them, and creating comprehensive technical specifications.\n            You always produce complete, valid JSON outputs without unnecessary explanations.',
        },
        {
            "key": 'database_schema',
            "role": 'Database Agent',
            "goal": 'Produce a clear, minimal database schema (tables, columns, relationships) from software requirements.',
            "backstory": 'You are a database architect. You derive practical schemas from requirements and output structured JSON.',
        },
        {
            "key": 'design',
            "role": 'UI/UX Design Specialist',
            "goal": 'OUTPUT a complete JSON object with UI/UX design specifications including components, icons, images, and realistic app previews. You must produce the actual JSON output, not just think about it.',
            "backstory": 'You are a senior product designer and UI/UX architect with extensive experience \n            creating production-ready, polished mobile applications. You think like a real product team, \n            prioritizing clarity, usability, and visual hierarchy. You excel at translating requirements \n            into detailed design specifications that resemble real iOS/Android apps, not wireframes. \n            Your designs are intentional, well-structured, and suitable for interactive prototypes. \n            You always produce complete, valid JSON outputs without unnecessary explanations.',
            "max_execution_time": DESIGN_MAX_EXECUTION_TIME,
        },
        {
            "key": 'cycle_plan',
            "role": 'Planning Manager Agent',
            "goal": 'Create a short-term MVP execution plan suitable for rapid development.',
            "backstory": 'You are an experienced project planner. You produce lightweight, immediately executable plans for single-developer or small-team MVP scenarios with fast delivery.',
        },
        {
            "key": 'quality_report',
            "role": 'Quality Manager Agent',
            "goal": 'Evaluate requirements, design, and database schema for completeness and consistency.',
            "backstory": 'You are a quality assurance expert. You inspect artifacts and produce gate decisions with clear issues and required fixes.',
        },
        {
            "key": 'support_package',
            "role": 'Support Manager Agent',
            "goal": 'Produce support governance artifacts from project state.',
            "backstory": 'You manage baselines, app documentation, and glossary for software projects.',
        },
    ]


def _task_specs() -> Dict[str, Dict[str, Any]]:
    """Description, expected output, and schema per task.

    Every description comes from ``core.prompts`` -- the same builders the
    federated agents use -- so the two topologies issue identical instructions.
    The only substitution is ``UPSTREAM_FROM_CONTEXT`` in place of an inline
    artifact, which is forced by the topology: a unified crew propagates prior
    output through ``context`` rather than by string interpolation. Without this
    sharing, any measured B2/B3 difference could be prompt wording rather than
    coordination, and the comparison would be worthless.

    ``{user_input}`` is interpolated by CrewAI from ``kickoff(inputs=...)``.
    """
    from agents.design_agent import DesignGenerationAgent

    ctx = UPSTREAM_FROM_CONTEXT
    return {
        "requirements": {
            # The federated path appends the transcript; here CrewAI substitutes
            # {user_input} at kickoff, so the placeholder stands in for it.
            "description": build_requirements_prompt("{user_input}"),
            "expected_output": (
                "A complete, valid JSON object starting with { and ending with }, containing: "
                "project_name, features, technical_requirements, user_roles. "
                "No markdown, no explanations, only JSON."
            ),
            "output_json": RequirementsOutputModel,
            "context": [],
        },
        "database_schema": {
            "description": build_database_prompt(ctx),
            "expected_output": (
                "A JSON object with tables and assumptions. Use common SQL types "
                "(uuid, int, bigint, varchar, text, timestamp, boolean, json)."
            ),
            "output_json": DatabaseSchemaModel,
            "context": ["requirements"],
        },
        "design": {
            "description": DesignGenerationAgent._build_task_prompt(ctx),
            "expected_output": (
                "A JSON object with design_overview, color_scheme, typography, navigation, "
                "screens, and ui_components. No markdown, only JSON."
            ),
            "output_json": DesignOutputModel,
            "context": ["requirements", "database_schema"],
        },
        "cycle_plan": {
            "description": build_planning_prompt(ctx, ctx),
            "expected_output": "A JSON object with plan_name, tasks, and risks.",
            "output_json": CyclePlanOutputModel,
            "context": ["requirements", "design"],
        },
        "quality_report": {
            "description": build_quality_prompt(ctx, ctx, ctx),
            "expected_output": (
                "A quality inspection report with gate decision (PASS/FAIL), checklist "
                "results, issues, and required fixes."
            ),
            "output_json": QualityReportOutputModel,
            "context": ["requirements", "database_schema", "design"],
        },
        "support_package": {
            "description": build_support_prompt(ctx, ctx, ctx, ctx),
            "expected_output": (
                "A JSON object with app_documentation, baseline_artifacts, and glossary."
            ),
            "output_json": SupportGovernanceOutputModel,
            "context": ["requirements", "database_schema", "design", "cycle_plan"],
        },
    }


class UnifiedCrewPipeline:
    """One crew, six agents, six context-chained tasks, a single kickoff.

    Args:
        api_key: Anthropic key. Only used to fail fast when absent; the LLM
            itself is built by ``core.llm.get_llm()`` from the environment.
        allow_delegation: When True, agents receive CrewAI's delegate and ask
            tools and can address any coworker in the crew. Off by default:
            delegation makes execution order emergent, which is exactly the
            property the sequential pipeline was chosen to avoid. Turn it on
            deliberately, as an experiment, not as a default.
        max_iter: Reasoning iterations per agent. Delegation is implemented as
            a tool call and therefore needs more than one iteration to happen at
            all, so raising this is a precondition for delegation being real.
    """

    def __init__(
        self,
        api_key: str,
        allow_delegation: bool = False,
        max_iter: int = 1,
    ) -> None:
        self.api_key = api_key
        self.crew: Optional[Any] = None
        self.tasks: Dict[str, Any] = {}
        self.agents: Dict[str, Any] = {}
        self._task_order: List[str] = []

        if not CREWAI_AVAILABLE:
            return

        llm = get_llm()
        specs = _agent_specs()
        task_specs = _task_specs()

        # 1. Build every agent first -- a crew's roster is its address book, so
        #    all agents must exist before any task can name a coworker.
        for spec in specs:
            kwargs = {
                "role": spec["role"],
                "goal": spec["goal"],
                "backstory": spec["backstory"],
                "verbose": False,
                "allow_delegation": allow_delegation,
                "max_iter": max_iter,
                "max_execution_time": spec.get("max_execution_time", MAX_EXECUTION_TIME),
            }
            if llm:
                kwargs["llm"] = llm
            try:
                with suppress_stderr():
                    self.agents[spec["key"]] = Agent(**kwargs)
            except Exception:
                self.agents = {}
                return

        # 2. Build tasks in order, wiring context to the already-built tasks.
        for spec in specs:
            key = spec["key"]
            tspec = task_specs[key]
            kwargs = {
                "name": key,
                "description": tspec["description"],
                "expected_output": tspec["expected_output"],
                "agent": self.agents[key],
                "context": [self.tasks[dep] for dep in tspec["context"]],
            }
            if PYDANTIC_AVAILABLE and tspec.get("output_json") is not None:
                kwargs["output_json"] = tspec["output_json"]
            try:
                self.tasks[key] = Task(**kwargs)
                self._task_order.append(key)
            except Exception:
                self.tasks = {}
                return

        # 3. One crew holding every agent and every task.
        try:
            with suppress_stderr():
                self.crew = Crew(
                    agents=[self.agents[s["key"]] for s in specs],
                    tasks=[self.tasks[s["key"]] for s in specs],
                    process=Process.sequential,
                    verbose=False,
                )
        except Exception:
            self.crew = None

    # ------------------------------------------------------------------
    @staticmethod
    def _artifact_from(task_output: Any) -> Dict[str, Any]:
        """Pull a dict out of a TaskOutput, preferring the validated form."""
        if task_output is None:
            return {"error": "no task output"}
        if isinstance(getattr(task_output, "json_dict", None), dict):
            return task_output.json_dict
        pyd = getattr(task_output, "pydantic", None)
        if pyd is not None and hasattr(pyd, "model_dump"):
            return pyd.model_dump()
        raw = getattr(task_output, "raw", "") or ""
        parsed = parse_json_from_text(raw) if raw else {}
        return parsed if isinstance(parsed, dict) else {"error": "unparseable task output"}

    def run(self, user_input: str) -> Dict[str, Any]:
        """Execute the whole chain in one kickoff.

        Returns a dict with ``artifacts`` (keyed by STEP_KEYS), ``latency_s``
        per task, ``total_latency_s``, ``tokens``, and ``error`` when the
        kickoff failed outright.
        """
        if not CREWAI_AVAILABLE or not self.crew:
            return {"error": "CrewAI unavailable or crew not initialized",
                    "artifacts": {}, "latency_s": {}}

        latency: Dict[str, float] = {}
        marks: Dict[str, float] = {}
        start = time.perf_counter()
        last = {"t": start}

        def _on_task_done(output: Any) -> None:
            # task_callback fires after each task; the gap since the previous
            # callback is that task's wall-clock cost. This is the only route to
            # per-agent latency when a single kickoff runs the whole chain.
            now = time.perf_counter()
            name = getattr(output, "name", None) or f"task_{len(latency) + 1}"
            latency[name] = round(now - last["t"], 2)
            marks[name] = now
            last["t"] = now

        self.crew.task_callback = _on_task_done

        try:
            with suppress_io():
                result = self.crew.kickoff(inputs={"user_input": user_input})
        except Exception as exc:
            return {
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "artifacts": {},
                "latency_s": latency,
                "total_latency_s": round(time.perf_counter() - start, 2),
            }

        artifacts: Dict[str, Any] = {}
        outputs = list(getattr(result, "tasks_output", None) or [])
        for index, key in enumerate(self._task_order):
            artifacts[key] = self._artifact_from(outputs[index] if index < len(outputs) else None)

        usage = getattr(result, "token_usage", None)
        tokens = {}
        if usage is not None:
            for field in ("prompt_tokens", "completion_tokens", "total_tokens",
                          "cached_prompt_tokens", "successful_requests"):
                value = getattr(usage, field, None)
                if value is not None:
                    tokens[field] = value

        return {
            "artifacts": artifacts,
            "latency_s": latency,
            "total_latency_s": round(time.perf_counter() - start, 2),
            "tokens": tokens,
        }

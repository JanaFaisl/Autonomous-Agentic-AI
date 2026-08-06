import json
from typing import Dict, Any, List

from core.constants import (
    DEFAULT_MODEL, MAX_ITERATIONS, MAX_EXECUTION_TIME,
)
from core.models import PYDANTIC_AVAILABLE, PipelinePlanOutputModel
from core.llm import CREWAI_AVAILABLE, Agent, Task, Crew, Process, get_llm
from core.utils import parse_json_from_text, extract_crewai_output
from utils.io_suppression import suppress_stderr, suppress_io

class ProjectManagerAgent:
    """Central orchestrator (CrewAI): controls execution order of AI agents, manages workflow state, stores intermediate artifacts."""

    DEFAULT_STEPS = ["requirements", "database", "design", "planning", "quality", "support"]

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.agent = None
        self.crew = None
        self.task = None

        if not CREWAI_AVAILABLE:
            return

        llm = get_llm()
        agent_kwargs = {
            "role": "Project Manager Agent",
            "goal": "Orchestrate the development pipeline: decide execution order of agents and ensure workflow state and artifacts are managed.",
            "backstory": "You are the central project manager. You control the order in which requirements, design, planning, quality, and support agents run, and you ensure intermediate artifacts are stored.",
            "verbose": False,
            "allow_delegation": False,
            "max_iter": MAX_ITERATIONS,
            "max_execution_time": MAX_EXECUTION_TIME,
        }
        if llm:
            agent_kwargs["llm"] = llm

        try:
            from utils.io_suppression import suppress_stderr
            with suppress_stderr():
                self.agent = Agent(**agent_kwargs)
        except Exception:
            self.agent = None
            return

        task_kwargs = {
            "description": "Given current project state (requirements and which artifacts already exist), output the pipeline execution plan as JSON: which steps to run and in what order.",
            "agent": self.agent,
            "expected_output": "A JSON object with steps: list of step names in order (e.g. requirements, database, design, planning, quality, support), and optional reason.",
        }
        if PYDANTIC_AVAILABLE and PipelinePlanOutputModel is not None:
            task_kwargs["output_json"] = PipelinePlanOutputModel

        self.task = Task(**task_kwargs)

        try:
            from utils.io_suppression import suppress_stderr
            with suppress_stderr():
                self.crew = Crew(agents=[self.agent], tasks=[self.task], verbose=False, process=Process.sequential)
        except Exception:
            self.crew = None

    def get_execution_plan(
        self,
        requirements: Dict[str, Any],
        has_requirements: bool = True,
        has_database: bool = False,
        has_design: bool = False,
        has_planning: bool = False,
        has_quality: bool = False,
        has_support: bool = False,
    ) -> List[str]:
        """Return ordered list of pipeline steps to run. Uses CrewAI when available; otherwise default order."""
        if not CREWAI_AVAILABLE or not self.crew:
            return list(self.DEFAULT_STEPS)

        state = {
            "has_requirements": has_requirements,
            "has_database": has_database,
            "has_design": has_design,
            "has_planning": has_planning,
            "has_quality": has_quality,
            "has_support": has_support,
        }
        req_preview = json.dumps({"project_name": requirements.get("project_name"), "features_count": len(requirements.get("features") or [])}, indent=2)
        self.task.description = f"""Current workflow state: requirements={has_requirements}, database={has_database}, design={has_design}, planning={has_planning}, quality={has_quality}, support={has_support}.
Requirements summary: {req_preview}
Output the pipeline steps to run in order. Use only: requirements, database, design, planning, quality, support. Skip steps already done if you want; otherwise output the standard order: requirements, database, design, planning, quality, support."""

        try:
            from utils.io_suppression import suppress_io
            with suppress_io():
                result = self.crew.kickoff(inputs={"state": json.dumps(state)})

            if hasattr(result, "tasks_output") and result.tasks_output:
                task_out = result.tasks_output[0]
                out = None
                if hasattr(task_out, "json_dict") and isinstance(task_out.json_dict, dict):
                    out = task_out.json_dict
                elif hasattr(task_out, "json") and isinstance(task_out.json, dict):
                    out = task_out.json
                if out and isinstance(out.get("steps"), list) and len(out["steps"]) > 0:
                    filtered = [str(s).lower() for s in out["steps"] if str(s).lower() in self.DEFAULT_STEPS]
                    if filtered:
                        return filtered
            raw = extract_crewai_output(result)
            parsed = parse_json_from_text(raw) if raw else {}
            if isinstance(parsed, dict) and isinstance(parsed.get("steps"), list) and len(parsed["steps"]) > 0:
                filtered = [str(s).lower() for s in parsed["steps"] if str(s).lower() in self.DEFAULT_STEPS]
                if filtered:
                    return filtered
        except Exception:
            pass
        return list(self.DEFAULT_STEPS)

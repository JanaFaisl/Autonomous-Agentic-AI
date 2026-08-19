import json
from typing import Dict, Any, Optional

import requests as _requests

from core.constants import (
    DEFAULT_MODEL, MAX_ITERATIONS, MAX_EXECUTION_TIME,
    ERROR_NO_API_KEY, ERROR_PARSE_JSON, SOLUTION_ADD_API_KEY, SOLUTION_RETRY,
)
from core.models import PYDANTIC_AVAILABLE, QualityReportOutputModel
from core.llm import CREWAI_AVAILABLE, Agent, Task, Crew, Process, get_llm
from core.utils import parse_json_from_text, extract_crewai_output, create_error_response
from core.prompts import build_quality_prompt
from utils.io_suppression import suppress_stderr, suppress_io

class QualityManagerAgent:
    """TSPi Quality Manager: inspects artifacts and produces quality gate decision."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.agent = None
        self.crew = None
        self.task = None

        if not CREWAI_AVAILABLE:
            return

        llm = get_llm()
        agent_kwargs = {
            "role": "Quality Manager Agent",
            "goal": "Evaluate requirements, design, and database schema for completeness and consistency.",
            "backstory": "You are a quality assurance expert. You inspect artifacts and produce gate decisions with clear issues and required fixes.",
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
            "description": "Evaluate the provided artifacts and produce a quality inspection report.",
            "agent": self.agent,
            "expected_output": "A quality inspection report with gate decision (PASS/FAIL), checklist results, issues, and required fixes.",
        }
        if PYDANTIC_AVAILABLE and QualityReportOutputModel is not None:
            task_kwargs["output_json"] = QualityReportOutputModel

        self.task = Task(**task_kwargs)

        try:
            from utils.io_suppression import suppress_stderr
            with suppress_stderr():
                self.crew = Crew(agents=[self.agent], tasks=[self.task], verbose=False, process=Process.sequential)
        except Exception:
            self.crew = None

    def _direct_api_call(self, prompt: str) -> Dict[str, Any]:
        """Fallback: call Anthropic Messages API directly when CrewAI is unavailable or fails."""
        try:
            resp = _requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": DEFAULT_MODEL,
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=90,
            )
            if resp.status_code != 200:
                return create_error_response(f"API error {resp.status_code}", SOLUTION_RETRY)
            data = resp.json()
            text = "\n".join(
                b.get("text", "") for b in (data.get("content") or [])
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            parsed = parse_json_from_text(text) if text else {}
            return parsed if isinstance(parsed, dict) and "error" not in parsed else create_error_response(ERROR_PARSE_JSON, SOLUTION_RETRY)
        except Exception as e:
            return create_error_response(str(e)[:200], SOLUTION_RETRY)

    def generate_quality_report(
        self,
        requirements: Dict[str, Any],
        design: Optional[Dict[str, Any]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate quality inspection report from requirements, design, and database schema."""
        if not self.api_key:
            return create_error_response(ERROR_NO_API_KEY, SOLUTION_ADD_API_KEY)

        req_json = json.dumps(requirements, indent=2)
        design_json = json.dumps(design, indent=2) if design else "No design."
        db_json = json.dumps(database_schema, indent=2) if database_schema else "No database schema."

        prompt = build_quality_prompt(req_json, design_json, db_json)

        # 1) Try CrewAI first — full orchestration with role, goal, and backstory context
        if CREWAI_AVAILABLE and self.crew:
            self.task.description = prompt
            try:
                from utils.io_suppression import suppress_io
                with suppress_io():
                    crew_result = self.crew.kickoff(inputs={"requirements": req_json, "design": design_json, "database": db_json})

                if hasattr(crew_result, "tasks_output") and crew_result.tasks_output:
                    task_out = crew_result.tasks_output[0]
                    if hasattr(task_out, "json_dict") and isinstance(task_out.json_dict, dict):
                        return task_out.json_dict
                    if hasattr(task_out, "json") and isinstance(task_out.json, dict):
                        return task_out.json

                raw = extract_crewai_output(crew_result)
                parsed = parse_json_from_text(raw) if raw else {}
                if isinstance(parsed, dict) and "error" not in parsed:
                    return parsed
            except Exception:
                pass

        # 2) Fallback to direct Anthropic API if CrewAI unavailable or failed
        result = self._direct_api_call(prompt)
        if "error" not in result:
            return result

        return result


# ----------------------------
# SUPPORT MANAGER AGENT (TSPi)
# ----------------------------

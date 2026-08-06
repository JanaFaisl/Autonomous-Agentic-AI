"""
Real-LLM integration test.

Runs ONE actual CrewAI kickoff against the Anthropic API.
Skipped automatically when ANTHROPIC_API_KEY is not a real key
(i.e. is missing or still set to the test stub value).

Run manually:
    ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_integration_real_llm.py -v -s
"""
import os
import pytest

# Load .env so the test works whether or not the shell exported the vars
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

_load_dotenv()

_REAL_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Skip only when no key is set or it's the test stub used in conftest.py
_IS_REAL_KEY = bool(_REAL_KEY) and _REAL_KEY != "test-api-key-12345"

pytestmark = pytest.mark.skipif(
    not _IS_REAL_KEY,
    reason="ANTHROPIC_API_KEY is not set to a real key — skipping live LLM call",
)


PROJECTS = [
    {
        "id": 1,
        "domain": "E-commerce / logistics",
        "complexity": "Medium",
        "input": "A flower delivery app where customers can browse flowers, place orders, and get them delivered to their door.",
    },
    {
        "id": 2,
        "domain": "Productivity",
        "complexity": "Low",
        "input": "A simple task manager app where users can create, complete, and delete tasks.",
    },
    {
        "id": 3,
        "domain": "Healthcare",
        "complexity": "High",
        "input": "A hospital appointment booking system where patients can schedule visits with doctors.",
    },
    {
        "id": 4,
        "domain": "Education / social",
        "complexity": "Medium",
        "input": "A social platform where university students can share study notes and form groups.",
    },
    {
        "id": 5,
        "domain": "Retail / operations",
        "complexity": "High",
        "input": "A real-time inventory management tool for small retail stores with barcode scanning.",
    },
]


@pytest.mark.parametrize("project", PROJECTS, ids=[f"project_{p['id']}" for p in PROJECTS])
def test_requirements_agent_real_crewai_call(project):
    """
    End-to-end: RequirementsAnalystAgent.analyze_requirements() with a real
    Anthropic API key for each of the 5 paper project inputs.
    """
    import json
    from agents.requirements_agent import RequirementsAnalystAgent
    from core.llm import CREWAI_AVAILABLE

    assert CREWAI_AVAILABLE, "CrewAI must be available for this test"

    agent = RequirementsAnalystAgent()
    assert agent.crew is not None, "Crew was not initialized"

    result = agent.analyze_requirements(project["input"])

    # Structural checks
    assert isinstance(result, dict), "Result must be a dict"
    assert "error" not in result, f"Got error: {result.get('error')}"
    path_used = result.get("model_used", "unknown")
    assert "project_name" in result, "Missing 'project_name'"
    assert isinstance(result.get("features"), list) and len(result["features"]) >= 1, (
        "Expected at least one feature"
    )
    assert "timestamp" in result

    print(f"  #{project['id']} {project['domain']:<25} features={len(result['features'])} path={path_used}")

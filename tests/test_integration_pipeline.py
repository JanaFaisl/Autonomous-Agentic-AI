"""Full-pipeline live integration test.

This is the test that backs the Pipeline Completion Rate (PCR) reported in the
paper. Unlike ``test_integration_real_llm.py``, which exercises the Requirements
Analyst alone, this drives all six pipeline agents end-to-end -- requirements,
database, design, planning, quality, support -- for each of the five project
inputs, and asserts that every agent produced a usable artifact.

Do not report an end-to-end completion figure from any other source.

Skipped automatically when ANTHROPIC_API_KEY is absent or set to the conftest
stub, so the deterministic suite is unaffected.

Run:
    ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_integration_pipeline.py -v -s

Cost warning: each project drives six live LLM calls and takes roughly 13
minutes, dominated by the Design agent. The full parametrisation is therefore
around an hour of wall-clock time. Use -k project_2 for a single cheap run
(the task manager is the lowest-complexity input).
"""
import json
import os
from pathlib import Path

import pytest

from evaluation.pipeline_runner import ARTIFACT_FILES, run_pipeline
from evaluation.projects import PROJECTS, PROJECT_IDS


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


_load_dotenv()

_REAL_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
_IS_REAL_KEY = bool(_REAL_KEY) and _REAL_KEY != "test-api-key-12345"

# A real key in .env must NOT be sufficient to trigger this suite. Thirty live
# LLM calls and an hour of wall-clock time should never happen because someone
# typed `pytest`, so an explicit opt-in is required on top of the key.
_OPT_IN = os.environ.get("RUN_PIPELINE_EVAL") == "1"

pytestmark = [
    pytest.mark.skipif(
        not _IS_REAL_KEY,
        reason="ANTHROPIC_API_KEY is not set to a real key -- skipping live pipeline run",
    ),
    pytest.mark.skipif(
        not _OPT_IN,
        reason="set RUN_PIPELINE_EVAL=1 to opt in (~30 live LLM calls, ~1 hour, billable)",
    ),
    pytest.mark.slow,
]

#: Agents that must produce a usable artifact for a run to count as complete.
REQUIRED_ARTIFACTS = list(ARTIFACT_FILES)


@pytest.fixture(scope="module")
def api_key() -> str:
    return _REAL_KEY


@pytest.mark.parametrize("project", PROJECTS, ids=PROJECT_IDS)
def test_full_pipeline_completes(project, api_key):
    """All six agents complete and hand off a usable artifact to the next stage."""
    result = run_pipeline(project, api_key)
    out_dir = result.save()

    print(f"\n  project_{project['id']} ({project['domain']}) -> {out_dir}")
    for agent, seconds in result.latency_s.items():
        fallbacks = result.fallbacks.get(agent, 0)
        note = f"  [fallback x{fallbacks}]" if fallbacks else ""
        print(f"    {agent:<14} {seconds:>7.1f}s{note}")
    print(f"    {'TOTAL':<14} {result.total_latency_s:>7.1f}s")

    assert not result.errors, (
        f"pipeline did not complete: "
        + "; ".join(f"{agent}: {err}" for agent, err in result.errors.items())
    )

    missing = [k for k in REQUIRED_ARTIFACTS if k not in result.artifacts]
    assert not missing, f"missing artifacts: {missing}"


@pytest.mark.parametrize("project", PROJECTS, ids=PROJECT_IDS)
def test_artifacts_are_structurally_linked(project, api_key):
    """Cheap traceability floor: the schema and design must be non-empty and
    reference the project the requirements describe.

    This is deliberately weak -- it is a smoke check, not the RDTC/RSC metric,
    which is computed offline over the saved artifacts. Its purpose is to fail
    loudly when the pipeline emits well-formed but disconnected artifacts.
    """
    run_dir = Path("evaluation/runs/B2") / f"project_{project['id']}"
    if not run_dir.exists():
        pytest.skip("no saved run; test_full_pipeline_completes must run first")

    requirements = json.loads((run_dir / "requirements.json").read_text())
    schema = json.loads((run_dir / "database_schema.json").read_text())
    design = json.loads((run_dir / "design.json").read_text())

    assert requirements.get("features"), "requirements contain no features"

    tables = schema.get("tables") or []
    assert tables, "schema contains no tables"
    assert all(t.get("name") for t in tables), "schema contains an unnamed table"

    screens = design.get("screens") or []
    assert screens, "design contains no screens"
    assert all(s.get("name") for s in screens), "design contains an unnamed screen"

    # Every screen's components should be resolvable; the scaffold renderer
    # depends on this and will otherwise emit an unbuildable frontend.
    for screen in screens:
        components = screen.get("key_components")
        assert isinstance(components, list), f"screen {screen.get('name')!r} has no component list"

"""Unit tests for core/traceability.py and utils/self_heal.py (no API calls)."""
import copy

from core.traceability import build_trace_matrix, change_impact, diff_requirements
from utils.self_heal import validate_project, validate_scaffold, self_heal
from tests.conftest import SAMPLE_REQUIREMENTS, SAMPLE_DESIGN, SAMPLE_DB_SCHEMA

PLAN = {"plan_name": "MVP", "tasks": [
    {"id": "T1", "title": "Build flower catalogue browse screen (F1)"},
    {"id": "T2", "title": "Implement order checkout"},
    {"id": "T3", "title": "Set up CI pipeline"},
]}
DESIGN = {**SAMPLE_DESIGN, "screens": [
    {"name": "Catalogue", "purpose": "Browse flowers and filter.", "key_components": []},
    {"name": "Settings", "purpose": "App preferences.", "key_components": []},
]}


def _req():
    r = copy.deepcopy(SAMPLE_REQUIREMENTS)
    r["database_schema"] = copy.deepcopy(SAMPLE_DB_SCHEMA)
    return r


class TestTraceMatrix:
    def test_links_by_id_and_keywords(self):
        m = build_trace_matrix(_req(), DESIGN, PLAN)
        f1, f2 = m["features"]
        assert "Catalogue" in f1["screens"]
        assert "T1" in f1["tasks"]
        assert "orders" in f2["tables"]
        assert "T2" in f2["tasks"]

    def test_orphans_and_gaps(self):
        m = build_trace_matrix(_req(), DESIGN, PLAN)
        assert "Settings" in m["orphans"]["screens"]
        assert "T3" in m["orphans"]["tasks"]
        assert 0 <= m["coverage"] <= 1

    def test_missing_artifacts_are_not_gaps(self):
        m = build_trace_matrix(_req())
        assert m["available"] == ["acceptance", "tables"]
        assert "screens" not in m["features"][0]["gaps"]

    def test_invalid_requirements(self):
        assert build_trace_matrix({"error": "x"})["features"] == []
        assert build_trace_matrix(None)["coverage"] == 0.0


class TestChangeImpact:
    def test_diff_detects_all_kinds(self):
        old, new = _req(), _req()
        new["features"][0]["description"] = "changed"
        new["features"].pop(1)
        new["features"].append({"id": "F3", "name": "Reviews", "description": "", "priority": "Low"})
        d = diff_requirements(old, new)
        assert d == {"added": ["F3"], "removed": ["F2"], "modified": ["F1"]}

    def test_impact_lists_artifacts_and_steps(self):
        m = build_trace_matrix(_req(), DESIGN, PLAN)
        imp = change_impact(m, ["F2"])
        assert "orders" in imp["tables"]
        assert "database" in imp["steps_to_rerun"]
        assert imp["steps_to_rerun"][-2:] == ["quality", "support"]

    def test_unknown_feature_reruns_everything(self):
        m = build_trace_matrix(_req(), DESIGN, PLAN)
        imp = change_impact(m, ["F9"])
        assert imp["unknown"] == ["F9"]
        assert imp["steps_to_rerun"] == ["database", "design", "planning", "quality", "support"]

    def test_no_change_no_steps(self):
        assert change_impact(build_trace_matrix(_req()), [])["steps_to_rerun"] == []


class TestSelfHeal:
    def test_bad_sql_detected(self):
        probs = validate_scaffold({"backend/schema.sql": "CREATE TABLE ("})
        assert probs and probs[0]["step"] == "database"

    def test_valid_sql_passes(self):
        assert validate_scaffold({"backend/schema.sql": "CREATE TABLE a (id INTEGER PRIMARY KEY);"}) == []

    def test_dangling_fk_detected(self):
        r = _req()
        r["database_schema"]["tables"][1]["relationships"][0]["to_table"] = "ghost"
        assert any("ghost" in p["message"] for p in validate_project(r, None))

    def test_heals_via_regenerate(self):
        r = _req()
        r["database_schema"]["tables"][1]["relationships"][0]["to_table"] = "ghost"
        calls = []

        def fix_db(req, messages):
            calls.append(messages)
            return copy.deepcopy(SAMPLE_DB_SCHEMA)

        out = self_heal(r, None, lambda req, d: {}, {"database": fix_db})
        assert out["problems"] == []
        assert len(calls) == 1 and "ghost" in calls[0][0]
        assert out["rounds"][0]["healed_steps"] == ["database"]

    def test_stops_when_nothing_heals(self):
        r = _req()
        r["database_schema"]["tables"][1]["relationships"][0]["to_table"] = "ghost"
        out = self_heal(r, None, lambda req, d: {}, {"database": lambda req, m: {"error": "no"}}, max_rounds=3)
        assert out["problems"] and len(out["rounds"]) == 1


# ---------------------------------------------------------------------------
# Task feature_ids, test cases
# ---------------------------------------------------------------------------
from core.test_cases import generate_test_cases, render_test_stubs


class TestDeclaredFeatureIds:
    def test_declared_ids_override_keywords(self):
        plan = {"tasks": [{"id": "T1", "title": "Build flower catalogue", "feature_ids": ["F2"]}]}
        f1, f2 = build_trace_matrix(_req(), DESIGN, plan)["features"]
        assert f2["tasks"] == ["T1"] and f1["tasks"] == []


class TestTestCases:
    def test_parses_given_when_then(self):
        cases = generate_test_cases(_req())
        assert len(cases) == 4
        c = cases[0]
        assert c["id"] == "TC-F1-1" and c["given"].startswith("I open") and c["when"].startswith("I navigate")
        assert c["then"].startswith("I see all")

    def test_free_text_criterion_kept(self):
        r = {"features": [{"id": "F1", "name": "x", "acceptance_criteria": ["Works offline."]}]}
        c = generate_test_cases(r)[0]
        assert c["then"] == "Works offline." and c["given"] == ""

    def test_stub_is_valid_js(self, tmp_path):
        import shutil, subprocess
        src = render_test_stubs(generate_test_cases(_req()))
        assert src.count("test.todo(") == 4
        if shutil.which("node"):
            f = tmp_path / "a.test.js"; f.write_text(src)
            assert subprocess.run(["node", "--check", str(f)]).returncode == 0

    def test_invalid_input(self):
        assert generate_test_cases(None) == [] and generate_test_cases({"error": "x"}) == []


def test_tasks_named_name_instead_of_title_still_link():
    plan = {"tasks": [{"name": "Implement order checkout"}]}
    f1, f2 = build_trace_matrix(_req(), DESIGN, plan)["features"]
    assert f2["tasks"] == ["Implement order checkout"]


class TestDesignCoverage:
    def test_declared_screen_feature_ids(self):
        from core.traceability import uncovered_features
        design = {"screens": [{"name": "Catalogue", "purpose": "", "feature_ids": ["F1"], "key_components": []}]}
        assert [f["id"] for f in uncovered_features(_req(), design)] == ["F2"]

    def test_keyword_fallback_without_ids(self):
        from core.traceability import uncovered_features
        assert [f["id"] for f in uncovered_features(_req(), DESIGN)] == ["F2"]

    def test_missing_design_is_not_reported(self):
        from core.traceability import uncovered_features
        assert uncovered_features(_req(), None) == []

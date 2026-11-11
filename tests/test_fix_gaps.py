"""Quality-issue routing and the repair round (no API calls)."""
from utils.fix_gaps import checks_for_issue, failing_checks, fix_gaps, group_fixable, issues_by_check, steps_for_issue

REPORT = {
    "checklist": {"requirements_complete": False, "design_consistent": False, "db_matches_requirements": False, "nfr_defined": True},
    "issues": [
        {"severity": "High", "item": "Database Schema - Missing Password Field", "message": "users has no password_hash."},
        {"severity": "High", "item": "Design vs Database - Appointment status", "message": "UI says Pending, DB says Booked."},
        {"severity": "Medium", "item": "Requirements - Authentication", "message": "No explicit login requirement."},
        {"severity": "Low", "item": "Design", "message": "Minor spacing."},
        {"severity": "High", "item": "Requirements - Non-Functional Requirements", "message": "No performance targets."},
    ],
}


def test_issue_routing():
    assert steps_for_issue(REPORT["issues"][0]) == {"database"}
    assert steps_for_issue(REPORT["issues"][1]) == {"design", "database"}
    assert steps_for_issue({"item": "", "message": "The users table lacks a column"}) == {"database"}
    assert "nfr_defined" in checks_for_issue(REPORT["issues"][4])


def test_group_skips_low_severity_and_orders_steps():
    g = group_fixable(REPORT)
    assert list(g) == ["requirements", "database", "design"]
    assert all("spacing" not in m for m in g["design"]) and len(g["database"]) == 2


def test_issues_explain_failing_checks():
    by = issues_by_check(REPORT)
    assert {i["item"] for i in by["db_matches_requirements"]} >= {"Database Schema - Missing Password Field"}
    assert set(failing_checks(REPORT)) == {"requirements_complete", "design_consistent", "db_matches_requirements"}


def test_fix_round_applies_fixers_in_order_and_survives_failures():
    calls = []
    req = {"features": [{"id": "F1"}], "database_schema": {"tables": [{"name": "users"}]}}
    fixers = {
        "requirements": lambda r, m: calls.append("requirements") or {**r, "features": [{"id": "F1"}, {"id": "F2"}]},
        "database": lambda r, m: calls.append("database") or {"tables": [{"name": "users"}, {"name": "x"}]},
        "design": lambda r, d, m: calls.append("design") or {"error": "boom"},
    }
    out = fix_gaps(req, {"screens": [{"name": "A"}]}, REPORT, fixers, lambda r, d: {"checklist": {"requirements_complete": True}})
    assert calls == ["requirements", "database", "design"]
    assert len(out["requirements"]["features"]) == 2 and len(out["requirements"]["database_schema"]["tables"]) == 2
    assert out["design"] == {"screens": [{"name": "A"}]}                       # failed fixer keeps the old design
    assert [(s["step"], s["ok"]) for s in out["steps"]] == [("requirements", True), ("database", True), ("design", False)]
    assert out["report"]["checklist"]["requirements_complete"] is True

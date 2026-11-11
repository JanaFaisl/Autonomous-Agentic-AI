"""Turn a quality review's issues into targeted fixes by the agent that owns each problem.

Pure orchestration: the caller injects the fixers and the re-check, so nothing here calls an API.
"""
from typing import Any, Callable, Dict, List, Optional, Set

# Checklist item each pipeline step answers for.
STEP_CHECK = {"requirements": "requirements_complete", "database": "db_matches_requirements", "design": "design_consistent"}
# Requirements are fixed first because the database and design are derived from them.
ORDER = ("requirements", "database", "design")
_FIXABLE = ("High", "Medium")


def _text(issue: Dict[str, Any]) -> str:
    return f"{issue.get('item', '')} {issue.get('message', '')}".lower()


def steps_for_issue(issue: Dict[str, Any]) -> Set[str]:
    """Which agents own an issue, judged mainly from its 'item' label ('Design vs Database - ...')."""
    item = str(issue.get("item", "")).lower()
    steps = {s for s, words in (("requirements", ("requirement", "nfr", "non-functional", "scope")),
                                ("database", ("database", "schema", "db ", "table", "column")),
                                ("design", ("design", "screen", "ui ", "ux", "navigation"))) if any(w in item for w in words)}
    if not steps:
        steps = {s for s, words in (("database", ("table", "column", "schema")), ("design", ("screen", "design")),
                                    ("requirements", ("requirement", "feature"))) if any(w in _text(issue) for w in words)}
    return steps or {"requirements"}


def checks_for_issue(issue: Dict[str, Any]) -> Set[str]:
    """Checklist keys an issue explains (non-functional issues also explain nfr_defined)."""
    keys = {STEP_CHECK[s] for s in steps_for_issue(issue)}
    if any(w in _text(issue) for w in ("non-functional", "nfr", "performance", "security requirement")):
        keys.add("nfr_defined")
    return keys


def issues_by_check(report: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for issue in (report or {}).get("issues") or []:
        if isinstance(issue, dict):
            for key in checks_for_issue(issue):
                out.setdefault(key, []).append(issue)
    return out


def group_fixable(report: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """{step: ["item: message", ...]} for High/Medium issues, in fix order."""
    grouped: Dict[str, List[str]] = {}
    for issue in (report or {}).get("issues") or []:
        if not isinstance(issue, dict) or issue.get("severity") not in _FIXABLE:
            continue
        line = f"{issue.get('item', '')}: {issue.get('message', '')}".strip(": ")
        for step in steps_for_issue(issue):
            grouped.setdefault(step, []).append(line)
    return {s: grouped[s] for s in ORDER if s in grouped}


def failing_checks(report: Optional[Dict[str, Any]]) -> List[str]:
    return [k for k, v in ((report or {}).get("checklist") or {}).items() if not v]


def fix_gaps(
    requirements: Dict[str, Any],
    design: Optional[Dict[str, Any]],
    report: Dict[str, Any],
    fixers: Dict[str, Callable[..., Dict[str, Any]]],
    recheck: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Dict[str, Any]],
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """One repair round. Fixers: requirements(req, msgs), database(req, msgs), design(req, design, msgs).

    A fixer that returns an error leaves that artifact untouched, so a failed step never loses work.
    Returns the (possibly updated) artifacts, the fresh report, and what happened per step.
    """
    req, des = dict(requirements), design
    steps: List[Dict[str, Any]] = []
    for step, msgs in group_fixable(report).items():
        fixer = fixers.get(step)
        if not fixer:
            continue
        if progress:
            progress(step)
        if step == "requirements":
            result = fixer(req, msgs)
            ok = isinstance(result, dict) and "error" not in result
            if ok:
                req = result
        elif step == "database":
            result = fixer(req, msgs)
            ok = isinstance(result, dict) and "error" not in result and bool(result.get("tables"))
            if ok:
                req = {**req, "database_schema": result}
        else:
            result = fixer(req, des, msgs)
            ok = isinstance(result, dict) and "error" not in result and bool(result.get("screens"))
            if ok:
                des = result
        steps.append({"step": step, "ok": ok, "issues": len(msgs), "error": "" if ok else str((result or {}).get("error", "no change"))[:160]})
    if progress:
        progress("quality")
    new_report = recheck(req, des)
    return {"requirements": req, "design": des, "report": new_report, "steps": steps}

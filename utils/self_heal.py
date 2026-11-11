"""Validate generated artifacts (schema + scaffold) and feed failures back to the agents that made them."""
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
from typing import Any, Callable, Dict, List, Optional


def validate_scaffold(files: Dict[str, str]) -> List[Dict[str, str]]:
    """Run real checks on scaffold files. Each problem: {step, check, message}."""
    problems: List[Dict[str, str]] = []

    schema_sql = files.get("backend/schema.sql")
    if schema_sql:
        conn = sqlite3.connect(":memory:")
        try:
            conn.executescript(schema_sql)
        except sqlite3.Error as e:
            problems.append({"step": "database", "check": "schema.sql executes", "message": str(e)})
        finally:
            conn.close()

    for path, content in files.items():
        if path.endswith("package.json"):
            try:
                json.loads(content)
            except ValueError as e:
                problems.append({"step": "scaffold", "check": f"{path} is valid JSON", "message": str(e)})

    js_files = {p: c for p, c in files.items() if p.startswith("backend/") and p.endswith(".js")}
    if js_files and shutil.which("node"):
        with tempfile.TemporaryDirectory() as tmp:
            for path, content in js_files.items():
                target = os.path.join(tmp, path)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "w", encoding="utf-8") as fh:
                    fh.write(content)
                res = subprocess.run(["node", "--check", target], capture_output=True, text=True, timeout=20)
                if res.returncode != 0:
                    msg = (res.stderr or "").strip().splitlines()
                    problems.append({"step": "scaffold", "check": f"{path} parses",
                                     "message": msg[-1] if msg else "syntax error"})
    return problems


def validate_project(
    requirements: Optional[Dict[str, Any]],
    design: Optional[Dict[str, Any]],
    scaffold: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    """Schema integrity (reusing DatabaseAgent's checks) plus scaffold checks."""
    problems: List[Dict[str, str]] = []
    schema = (requirements or {}).get("database_schema") if isinstance(requirements, dict) else None
    if isinstance(schema, dict) and schema.get("tables"):
        from agents.database_agent import DatabaseAgent
        for issue in DatabaseAgent.__new__(DatabaseAgent)._validate_schema(schema):
            problems.append({"step": "database", "check": "schema integrity", "message": issue})
    if scaffold:
        problems.extend(validate_scaffold(scaffold))
    return problems


def self_heal(
    requirements: Dict[str, Any],
    design: Optional[Dict[str, Any]],
    build_scaffold: Callable[[Dict[str, Any], Optional[Dict[str, Any]]], Dict[str, str]],
    regenerate: Dict[str, Callable[[Dict[str, Any], List[str]], Dict[str, Any]]],
    max_rounds: int = 2,
) -> Dict[str, Any]:
    """Validate, then ask the owning agent to fix its artifact using the failure messages; repeat.

    `regenerate[step](requirements, messages)` returns the replacement artifact
    (the database schema for "database"), or a dict with "error" to leave it unchanged.
    Returns {"requirements", "problems", "rounds": [{"round", "problems", "healed_steps"}]}.
    """
    req = dict(requirements)
    history: List[Dict[str, Any]] = []
    problems = validate_project(req, design, build_scaffold(req, design))
    for rnd in range(1, max_rounds + 1):
        if not problems:
            break
        healed: List[str] = []
        for step in sorted({p["step"] for p in problems}):
            fn = regenerate.get(step)
            if fn is None:
                continue
            messages = [f"{p['check']}: {p['message']}" for p in problems if p["step"] == step]
            fixed = fn(req, messages)
            if isinstance(fixed, dict) and "error" not in fixed:
                if step == "database":
                    req["database_schema"] = fixed
                healed.append(step)
        history.append({"round": rnd, "problems": problems, "healed_steps": healed})
        if not healed:
            break
        problems = validate_project(req, design, build_scaffold(req, design))
    return {"requirements": req, "problems": problems, "rounds": history}

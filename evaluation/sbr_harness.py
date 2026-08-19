"""Scaffold Build Rate (SBR) harness.

Renders the artifacts produced by ``pipeline_runner`` into a runnable project via
``utils.code_scaffold.build_scaffold_files``, then checks whether that project
actually installs, builds, and runs.

Why this measures something
---------------------------
The renderer performs no LLM calls -- it is a pure function from specification to
source files. So a build failure is always attributable to the specification (a
design screen referencing a table the schema omits, a component type the design
never defines, an unparseable colour value) and never to code-generation
variance. That is what makes SBR an oracle-free quality metric for upstream
artifacts: it cannot be satisfied by construction, and it fails for diagnosable
reasons.

Checks, in order of increasing strictness:

    render          build_scaffold_files() produced backend and/or frontend files
    npm_install     dependencies resolve
    schema_exec     schema.sql executes against a real node:sqlite database
    server_start    the Express server boots and answers GET /api/health
    frontend_build  vite build succeeds

A project counts toward SBR only if every applicable check passes.

Usage
-----
    python -m evaluation.sbr_harness --condition B2
    python -m evaluation.sbr_harness --condition B0 --keep      # keep build dirs
    python -m evaluation.sbr_harness --condition B2 --skip-npm  # render+syntax only

Requires Node >= 22.5 for the built-in ``node:sqlite`` module (checked at start).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.code_scaffold import build_scaffold_files  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent / "runs"
BUILDS_DIR = Path(__file__).resolve().parent / "builds"

NPM_INSTALL_TIMEOUT = 900

#: npm defaults to a 300s fetch timeout with 2 retries, so an unreachable or
#: TLS-broken registry stalls for ~15 minutes per project and surfaces as a
#: timeout with no cause attached. Capping both turns that into a fast, named
#: error ("unable to verify the first certificate") that the operator can act on.
#: Raise NPM_FETCH_TIMEOUT_MS on a genuinely slow-but-working connection.
NPM_FETCH_TIMEOUT_MS = 60000
NPM_FETCH_RETRIES = 1
NPM_NETWORK_FLAGS = [
    "--no-audit", "--no-fund",
    f"--fetch-timeout={NPM_FETCH_TIMEOUT_MS}",
    f"--fetch-retries={NPM_FETCH_RETRIES}",
]
BUILD_TIMEOUT = 300
SERVER_BOOT_TIMEOUT = 30

#: Failures at these stages say nothing about specification quality -- they are
#: registry, network, or toolchain problems. They are reported separately and
#: EXCLUDED from the SBR denominator, because counting an npm timeout as a
#: specification defect would understate the system for reasons unrelated to it.
ENVIRONMENTAL_STAGES = {"npm_install"}

#: Ordered so a report reads as a funnel, cheapest and most diagnostic first.
CHECKS = ["render", "schema_exec", "npm_install", "server_start", "frontend_build"]

#: Checks that need the npm registry; skipped by --skip-npm.
NETWORK_CHECKS = ["npm_install", "server_start", "frontend_build"]


@dataclass
class BuildResult:
    project_id: int
    condition: str
    checks: Dict[str, str] = field(default_factory=dict)   # check -> pass|fail|skip
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    file_count: int = 0
    build_dir: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.failure_stage is None and any(v == "pass" for v in self.checks.values())

    @property
    def environmental(self) -> bool:
        """True when the run failed for a reason unrelated to the specification."""
        return self.failure_stage in ENVIRONMENTAL_STAGES

    def fail(self, stage: str, reason: str) -> "BuildResult":
        self.checks[stage] = "fail"
        self.failure_stage = stage
        self.failure_reason = reason.strip()[:500]
        return self


_SYSTEM_CA_SUPPORTED: Optional[bool] = None


def _node_env() -> Dict[str, str]:
    """Environment for node/npm subprocesses.

    On networks that terminate TLS (corporate proxies, security appliances) the
    registry presents a certificate signed by a CA absent from Node's *bundled*
    trust store, and the handshake fails with UNABLE_TO_VERIFY_LEAF_SIGNATURE.
    ``--use-system-ca`` makes Node consult the OS trust store instead, which
    helps only when the intercepting CA is actually installed there -- it is not
    a general fix, and on a machine where that CA is missing everywhere it
    changes nothing. Enabled when supported because it costs nothing and
    resolves the common case; when it does not help, the operator needs
    NODE_EXTRA_CA_CERTS pointing at the CA, or a different network.
    """
    global _SYSTEM_CA_SUPPORTED
    env = dict(os.environ)
    if _SYSTEM_CA_SUPPORTED is None:
        try:
            probe = subprocess.run(
                ["node", "--use-system-ca", "-e", "0"],
                capture_output=True, text=True, timeout=20,
            )
            _SYSTEM_CA_SUPPORTED = probe.returncode == 0
        except Exception:
            _SYSTEM_CA_SUPPORTED = False
    if _SYSTEM_CA_SUPPORTED:
        existing = env.get("NODE_OPTIONS", "")
        if "--use-system-ca" not in existing:
            env["NODE_OPTIONS"] = (existing + " --use-system-ca").strip()
    return env


def _run(cmd: List[str], cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    """Run a command, converting a timeout into an ordinary non-zero result.

    A hung `npm install` (offline machine, TLS-intercepting proxy, unreachable
    registry) is a build failure to be recorded and categorised, not an
    exception that aborts the whole sweep and loses the other projects' results.
    """
    try:
        return subprocess.run(
            cmd, cwd=str(cwd), timeout=timeout, env=_node_env(),
            capture_output=True, text=True, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=cmd, returncode=124,
            stdout=(exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            stderr=f"timed out after {timeout}s: {' '.join(cmd)}",
        )
    except FileNotFoundError:
        return subprocess.CompletedProcess(
            args=cmd, returncode=127, stdout="", stderr=f"command not found: {cmd[0]}",
        )


#: Lines that carry the actual cause, as opposed to stack frames or noise.
_ERROR_LINE = re.compile(
    r"^\s*(?:[A-Za-z]*Error|error|npm error|ERR!|SyntaxError|Failed|Cannot)\b.*",
    re.IGNORECASE,
)
_STACK_FRAME = re.compile(r"^\s*at\s")


def _tail(text: str, lines: int = 8) -> str:
    """Extract the diagnosable cause from tool output.

    Node prints the real message (``Error: duplicate column name: status``) near
    the top and then a long stack trace, so a naive tail returns ``at [eval]:1:1``
    and tells us nothing. The failure reason is the whole value of this metric --
    it is what turns "the build broke" into "the schema had a duplicate column" --
    so we hunt for the first genuine error line and fall back to the tail only
    when there is none.
    """
    raw = (text or "").strip()
    if not raw:
        return ""
    candidates = [
        line.strip() for line in raw.splitlines()
        if _ERROR_LINE.match(line) and not _STACK_FRAME.match(line)
    ]
    if candidates:
        return "\n".join(candidates[:3])
    meaningful = [line for line in raw.splitlines() if line.strip() and not _STACK_FRAME.match(line)]
    return "\n".join(meaningful[-lines:])


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _check_node() -> Optional[str]:
    """Return an error string if the Node toolchain cannot support the scaffold."""
    if shutil.which("node") is None or shutil.which("npm") is None:
        return "node and npm must be on PATH"
    try:
        out = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=15)
        version = out.stdout.strip().lstrip("v")
        major, minor = (int(p) for p in version.split(".")[:2])
    except Exception as exc:
        return f"could not determine Node version: {exc}"
    if (major, minor) < (22, 5):
        return f"Node {version} is too old; the scaffold uses node:sqlite, which needs >= 22.5"
    return None


def _load_artifacts(run_dir: Path) -> Dict[str, Any]:
    artifacts: Dict[str, Any] = {}
    for key, filename in (
        ("requirements", "requirements.json"),
        ("design", "design.json"),
        ("database_schema", "database_schema.json"),
    ):
        path = run_dir / filename
        if path.exists():
            try:
                artifacts[key] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                artifacts[key] = {"error": f"unreadable: {exc}"}
    return artifacts


def _server_answers_health(backend: Path, port: int) -> Optional[str]:
    """Boot server.js and poll /api/health. Returns an error string, or None on success."""
    env = {**_node_env(), "PORT": str(port)}
    proc = subprocess.Popen(
        ["node", "server.js"], cwd=str(backend), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.time() + SERVER_BOOT_TIMEOUT
        last_error = "server did not answer before timeout"
        while time.time() < deadline:
            if proc.poll() is not None:
                _, err = proc.communicate(timeout=5)
                return f"server exited early: {_tail(err)}"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2) as resp:
                    if resp.status == 200 and b'"ok"' in resp.read():
                        return None
                    last_error = f"health endpoint returned HTTP {resp.status}"
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.4)
        return last_error
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def build_one(
    run_dir: Path,
    project_id: int,
    condition: str,
    work_root: Path,
    skip_npm: bool = False,
) -> BuildResult:
    """Render one run's artifacts and verify the result installs, boots, and builds."""
    res = BuildResult(project_id=project_id, condition=condition)
    artifacts = _load_artifacts(run_dir)

    requirements = artifacts.get("requirements")
    if not isinstance(requirements, dict) or "error" in requirements:
        return res.fail("render", "no usable requirements.json for this run")

    design = artifacts.get("design")
    schema = artifacts.get("database_schema")
    design = design if isinstance(design, dict) and "error" not in design else None
    schema = schema if isinstance(schema, dict) and "error" not in schema else None

    # --- render -------------------------------------------------------------
    try:
        files = build_scaffold_files(requirements, design, schema)
    except Exception as exc:
        return res.fail("render", f"{type(exc).__name__}: {exc}")

    if not files:
        return res.fail("render", "renderer produced no files")

    has_backend = any(p.startswith("backend/") for p in files)
    has_frontend = any(p.startswith("frontend/") for p in files)
    if not has_backend and not has_frontend:
        return res.fail("render", "specification yielded neither a backend nor a frontend")

    work = work_root / condition.replace("/", "_") / f"project_{project_id}"
    if work.exists():
        shutil.rmtree(work)
    for rel, content in files.items():
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    res.file_count = len(files)
    res.build_dir = str(work)
    res.checks["render"] = "pass"

    if skip_npm:
        for check in NETWORK_CHECKS:
            res.checks[check] = "skip"
        if has_backend:
            schema_exec = _run(["node", "-e", "require('./db.js')"], work / "backend", 60)
            if schema_exec.returncode != 0:
                return res.fail("schema_exec", _tail(schema_exec.stderr or schema_exec.stdout))
            res.checks["schema_exec"] = "pass"
        else:
            res.checks["schema_exec"] = "skip"
        return res

    # --- backend ------------------------------------------------------------
    backend = work / "backend"
    if has_backend:
        # schema_exec runs BEFORE npm install on purpose: db.js imports only
        # node:sqlite, path, and fs -- all built in -- so the DDL check needs no
        # dependencies. Ordering it first means an unreachable npm registry
        # still leaves the most diagnostic check (is the generated schema valid
        # SQL?) intact, and attributes schema failures to the specification
        # rather than to the network.
        schema_exec = _run(["node", "-e", "require('./db.js')"], backend, 60)
        if schema_exec.returncode != 0:
            return res.fail("schema_exec", _tail(schema_exec.stderr or schema_exec.stdout))
        res.checks["schema_exec"] = "pass"

        install = _run(["npm", "install", *NPM_NETWORK_FLAGS], backend, NPM_INSTALL_TIMEOUT)
        if install.returncode != 0:
            return res.fail("npm_install", f"backend: {_tail(install.stderr or install.stdout)}")

        boot_error = _server_answers_health(backend, _free_port())
        if boot_error:
            return res.fail("server_start", boot_error)
        res.checks["server_start"] = "pass"
    else:
        res.checks["schema_exec"] = "skip"
        res.checks["server_start"] = "skip"

    # --- frontend -----------------------------------------------------------
    frontend = work / "frontend"
    if has_frontend:
        install = _run(["npm", "install", *NPM_NETWORK_FLAGS], frontend, NPM_INSTALL_TIMEOUT)
        if install.returncode != 0:
            return res.fail("npm_install", f"frontend: {_tail(install.stderr or install.stdout)}")
        res.checks["npm_install"] = "pass"

        build = _run(["npm", "run", "build"], frontend, BUILD_TIMEOUT)
        if build.returncode != 0:
            return res.fail("frontend_build", _tail(build.stderr or build.stdout))
        res.checks["frontend_build"] = "pass"
    else:
        res.checks.setdefault("npm_install", "pass" if has_backend else "skip")
        res.checks["frontend_build"] = "skip"

    return res


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Scaffold Build Rate over saved pipeline runs.")
    parser.add_argument("--condition", default="B2",
                        help="Run directory under evaluation/runs (e.g. B0, B1, B2, B2/run_1)")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--work-dir", type=Path, default=BUILDS_DIR)
    parser.add_argument("--skip-npm", action="store_true",
                        help="Skip the checks needing the npm registry. Still renders and "
                             "executes the generated DDL, which needs no dependencies.")
    parser.add_argument("--keep", action="store_true", help="Keep build directories for inspection")
    args = parser.parse_args()

    if not args.skip_npm:
        node_error = _check_node()
        if node_error:
            print(f"error: {node_error}")
            return 1

    condition_dir = args.runs_dir / args.condition
    if not condition_dir.is_dir():
        print(f"error: no runs found at {condition_dir}")
        print("Run evaluation/pipeline_runner.py first.")
        return 1

    run_dirs = sorted(
        (d for d in condition_dir.iterdir() if d.is_dir() and d.name.startswith("project_")),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not run_dirs:
        print(f"error: {condition_dir} contains no project_* directories")
        return 1

    results: List[BuildResult] = []
    for run_dir in run_dirs:
        project_id = int(run_dir.name.split("_")[1])
        print(f"\n=== {args.condition} project_{project_id} ===")
        result = build_one(run_dir, project_id, args.condition, args.work_dir, args.skip_npm)
        results.append(result)

        for check in CHECKS:
            status = result.checks.get(check, "-")
            mark = {"pass": "PASS", "fail": "FAIL", "skip": "skip"}.get(status, "-")
            print(f"  {check:<16} {mark}")
        if result.failure_reason:
            print(f"  reason: {result.failure_reason.splitlines()[0][:160]}")

    passed = sum(r.passed for r in results)
    env_failed = [r for r in results if r.environmental]
    spec_failed = [r for r in results if not r.passed and not r.environmental]
    scorable = len(results) - len(env_failed)

    stages_run = sorted({c for r in results for c, v in r.checks.items() if v in ("pass", "fail")})
    partial = bool(set(CHECKS) - set(stages_run))

    print("\n" + "=" * 60)
    if scorable:
        label = "PARTIAL Scaffold Build Rate" if partial else "Scaffold Build Rate"
        print(f"{label} ({args.condition}): "
              f"{passed}/{scorable} = {100.0 * passed / scorable:.1f}%")
        if partial:
            skipped = [c for c in CHECKS if c not in stages_run]
            print(f"  stages verified: {', '.join(stages_run)}")
            print(f"  stages SKIPPED : {', '.join(skipped)}")
            print("  This is not a full build rate. Report it as a partial figure and")
            print("  name the stages, or re-run without --skip-npm before publishing.")
    else:
        print(f"Scaffold Build Rate ({args.condition}): not computable -- "
              "every run failed for environmental reasons")
    if env_failed:
        print(f"\n{len(env_failed)} of {len(results)} runs excluded from the denominator: they")
        print("failed at a stage that reflects the build environment, not the")
        print("specification. Re-run once the environment is healthy before")
        print("reporting SBR -- a partial denominator is not a publishable figure.")
        for r in env_failed:
            print(f"  project_{r.project_id}: {r.failure_stage} -- "
                  f"{(r.failure_reason or '').splitlines()[0][:70]}")

    if spec_failed:
        print("\nSpecification failures (report this table in the paper):")
        causes: Dict[str, int] = {}
        for r in spec_failed:
            causes[r.failure_stage or "unknown"] = causes.get(r.failure_stage or "unknown", 0) + 1
        for stage, count in sorted(causes.items(), key=lambda kv: -kv[1]):
            print(f"  {stage:<16} {count}")
        for r in spec_failed:
            print(f"    project_{r.project_id}: "
                  f"{(r.failure_reason or '').splitlines()[0][:80]}")

    failures = {r.failure_stage or "unknown": 0 for r in results if not r.passed}
    for r in results:
        if not r.passed:
            failures[r.failure_stage or "unknown"] += 1

    report = args.runs_dir / args.condition / "sbr_report.json"
    report.write_text(json.dumps({
        "condition": args.condition,
        "partial": partial,
        "stages_verified": stages_run,
        "sbr_percent": round(100.0 * passed / scorable, 1) if scorable else None,
        "passed": passed,
        "total": scorable,
        "runs_attempted": len(results),
        "excluded_environmental": len(env_failed),
        "failure_causes": {k: v for k, v in failures.items()
                           if k not in ENVIRONMENTAL_STAGES},
        "environmental_causes": {k: v for k, v in failures.items()
                                 if k in ENVIRONMENTAL_STAGES},
        "results": [asdict(r) for r in results],
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {report}")

    if not args.keep and args.work_dir.exists():
        shutil.rmtree(args.work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

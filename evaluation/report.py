"""Aggregate every saved run into paper-ready tables.

Reads evaluation/runs/, computes descriptive statistics per condition, and
prints tables in plain text or LaTeX. Reads only what is on disk -- it makes no
API calls, so it is free to re-run as often as you like.

    python -m evaluation.report                # all conditions, plain text
    python -m evaluation.report --latex        # LaTeX bodies to paste
    python -m evaluation.report --condition B2

Incomplete runs are EXCLUDED from every statistic and counted separately, so a
billing outage or network fault cannot silently deflate a reported mean. The
completion-rate line always shows the excluded count.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics as stats
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

RUNS_DIR = Path(__file__).resolve().parent / "runs"

#: Canonical agent order for the per-agent latency table.
AGENT_ORDER = ["requirements", "database", "design", "planning", "quality", "support"]

#: B3 (native crew) names its tasks after artifacts, not agents. Map them onto
#: the same rows so the two conditions can be compared line by line.
B3_ALIASES = {
    "database_schema": "database",
    "cycle_plan": "planning",
    "quality_report": "quality",
    "support_package": "support",
}

PRETTY = {
    "requirements": "Requirements Analyst",
    "database": "Database",
    "design": "Design Generation",
    "planning": "Planning Manager",
    "quality": "Quality/Process Manager",
    "support": "Support Manager",
    "baseline": "Single prompt",
}


def _find_runs(condition_dir: Path) -> List[Path]:
    """Every metrics.json under a condition, at any nesting depth."""
    return sorted(condition_dir.rglob("metrics.json"))


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _normalise_latency(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Rename B3's task labels onto the shared agent rows."""
    out: Dict[str, float] = {}
    for key, value in (metrics.get("latency_s") or {}).items():
        out[B3_ALIASES.get(key, key)] = value
    return out


def _fmt(value: float, width: int = 8, places: int = 1) -> str:
    return f"{value:>{width}.{places}f}"


def collect(condition: str, runs_dir: Path) -> Optional[Dict[str, Any]]:
    condition_dir = runs_dir / condition
    if not condition_dir.is_dir():
        return None

    complete: List[Dict[str, Any]] = []
    incomplete: List[Dict[str, Any]] = []
    for path in _find_runs(condition_dir):
        metrics = _load(path)
        if metrics is None:
            continue
        (complete if metrics.get("completed") else incomplete).append(metrics)

    if not complete and not incomplete:
        return None

    per_agent: Dict[str, List[float]] = {}
    for metrics in complete:
        for agent, seconds in _normalise_latency(metrics).items():
            per_agent.setdefault(agent, []).append(seconds)

    totals = [m["total_latency_s"] for m in complete]
    fallbacks = sum(
        v for m in complete for k, v in (m.get("fallbacks") or {}).items()
        if "token" not in k and k != "successful_requests"
    )
    tokens: Dict[str, int] = {}
    for metrics in complete:
        for key, value in (metrics.get("fallbacks") or {}).items():
            if "token" in key:
                tokens[key] = tokens.get(key, 0) + int(value)

    return {
        "condition": condition,
        "complete": complete,
        "n_complete": len(complete),
        "n_incomplete": len(incomplete),
        "per_agent": per_agent,
        "totals": totals,
        "fallbacks": fallbacks,
        "tokens": tokens,
    }


def _sd(values: List[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def print_condition(data: Dict[str, Any]) -> None:
    n, bad = data["n_complete"], data["n_incomplete"]
    total_runs = n + bad
    print(f"\n{'=' * 72}")
    print(f"CONDITION {data['condition']}")
    print("=" * 72)
    rate = 100.0 * n / total_runs if total_runs else 0.0
    line = f"Completion rate     {n}/{total_runs} = {rate:.1f}%"
    if bad:
        line += f"   ({bad} incomplete, EXCLUDED from statistics below)"
    print(line)

    if not n:
        print("No complete runs -- nothing to summarise.")
        return

    print(f"Fallback activations {data['fallbacks']}")
    if data["tokens"]:
        per_run = {k: v // n for k, v in data["tokens"].items()}
        print("Tokens (total)      " + ", ".join(f"{k}={v:,}" for k, v in data["tokens"].items()))
        print("Tokens (per run)    " + ", ".join(f"{k}={v:,}" for k, v in per_run.items()))

    totals = data["totals"]
    grand = sum(stats.mean(v) for v in data["per_agent"].values()) or 1.0

    print(f"\n{'Agent':<26}{'Mean':>8}{'SD':>8}{'Min':>8}{'Max':>8}{'Share':>8}")
    print("-" * 66)
    ordered = [a for a in AGENT_ORDER if a in data["per_agent"]]
    ordered += [a for a in data["per_agent"] if a not in ordered]
    for agent in ordered:
        v = data["per_agent"][agent]
        mean = stats.mean(v)
        print(f"{PRETTY.get(agent, agent):<26}{_fmt(mean)}{_fmt(_sd(v))}"
              f"{_fmt(min(v))}{_fmt(max(v))}{100 * mean / grand:>7.1f}%")
    print("-" * 66)
    print(f"{'TOTAL PIPELINE':<26}{_fmt(stats.mean(totals))}{_fmt(_sd(totals))}"
          f"{_fmt(min(totals))}{_fmt(max(totals))}")

    print(f"\n{'Project':<30}{'Complexity':<12}{'Mean (s)':>10}{'n':>4}")
    print("-" * 56)
    by_project: Dict[int, List[Dict[str, Any]]] = {}
    for m in data["complete"]:
        by_project.setdefault(m["project_id"], []).append(m)
    for pid in sorted(by_project):
        rows = by_project[pid]
        lat = [r["total_latency_s"] for r in rows]
        print(f"{rows[0]['domain']:<30}{rows[0]['complexity']:<12}"
              f"{stats.mean(lat):>10.1f}{len(rows):>4}")

    # Conditions run with --repeats store results in nested run_N/ directories,
    # so an SBR report may sit one level below the condition root.
    sbr_paths = sorted((RUNS_DIR / data["condition"]).rglob("sbr_report.json"))
    sbr = _load(sbr_paths[0]) if sbr_paths else None
    if sbr and len(sbr_paths) > 1:
        print(f"\n  note: {len(sbr_paths)} SBR reports found; showing "
              f"{sbr_paths[0].parent.name}")
    if sbr:
        label = "Scaffold Build Rate (PARTIAL)" if sbr.get("partial") else "Scaffold Build Rate"
        print(f"\n{label}  {sbr['passed']}/{sbr['total']} = {sbr['sbr_percent']}%")
        if sbr.get("partial"):
            print(f"  verified stages only: {', '.join(sbr.get('stages_verified', []))}")
            print("  NOT a full build rate -- do not report as one.")
        if sbr.get("excluded_environmental"):
            print(f"  {sbr['excluded_environmental']} run(s) excluded (environmental, "
                  f"not specification)")
        for stage, count in (sbr.get("failure_causes") or {}).items():
            print(f"  spec failure at {stage:<14} {count}")
    else:
        print("\nScaffold Build Rate  not computed "
              f"(run: python -m evaluation.sbr_harness --condition {data['condition']})")


def print_comparison(all_data: List[Dict[str, Any]]) -> None:
    usable = [d for d in all_data if d["n_complete"]]
    if len(usable) < 2:
        return
    print(f"\n{'=' * 72}")
    print("CROSS-CONDITION COMPARISON")
    print("=" * 72)
    print(f"{'Condition':<12}{'n':>4}{'Complete':>10}{'Mean (s)':>11}{'SD':>8}{'vs B0':>8}")
    print("-" * 56)
    baseline = next((d for d in usable if d["condition"] == "B0"), None)
    base_mean = stats.mean(baseline["totals"]) if baseline else None
    for d in usable:
        mean = stats.mean(d["totals"])
        ratio = f"{mean / base_mean:.2f}x" if base_mean else "--"
        total_runs = d["n_complete"] + d["n_incomplete"]
        completed = f"{d['n_complete']}/{total_runs}"
        print(f"{d['condition']:<12}{d['n_complete']:>4}{completed:>10}"
              f"{mean:>11.1f}{_sd(d['totals']):>8.1f}{ratio:>8}")


def print_latex(all_data: List[Dict[str, Any]]) -> None:
    for data in all_data:
        if not data["n_complete"]:
            continue
        grand = sum(stats.mean(v) for v in data["per_agent"].values()) or 1.0
        print(f"\n% ---- {data['condition']}: per-agent latency "
              f"(n={data['n_complete']} complete runs) ----")
        print(r"\begin{tabular}{@{}lrrrrr@{}}")
        print(r"\toprule")
        print(r"\textbf{Agent} & \textbf{Mean (s)} & \textbf{SD} & \textbf{Min} & "
              r"\textbf{Max} & \textbf{Share} \\")
        print(r"\midrule")
        ordered = [a for a in AGENT_ORDER if a in data["per_agent"]]
        ordered += [a for a in data["per_agent"] if a not in ordered]
        for agent in ordered:
            v = data["per_agent"][agent]
            mean = stats.mean(v)
            print(f"{PRETTY.get(agent, agent)} & {mean:.1f} & {_sd(v):.1f} & "
                  f"{min(v):.1f} & {max(v):.1f} & {100 * mean / grand:.1f}\\% \\\\")
        totals = data["totals"]
        print(r"\midrule")
        print(f"\\textbf{{Total pipeline}} & \\textbf{{{stats.mean(totals):.1f}}} & "
              f"\\textbf{{{_sd(totals):.1f}}} & {min(totals):.1f} & {max(totals):.1f} & \\\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")


def summary_dict(all_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Machine-readable aggregate: every figure the paper quotes, in one place.

    Written alongside the human-readable tables so the reported numbers exist as
    a durable artifact rather than only as terminal output that must be
    regenerated to be checked.
    """
    out: Dict[str, Any] = {"generated_from": str(RUNS_DIR), "conditions": {}}
    for d in all_data:
        if not d["n_complete"]:
            continue
        totals = d["totals"]
        grand = sum(stats.mean(v) for v in d["per_agent"].values()) or 1.0
        sbr_paths = sorted((RUNS_DIR / d["condition"]).rglob("sbr_report.json"))
        sbr = _load(sbr_paths[0]) if sbr_paths else None
        out["conditions"][d["condition"]] = {
            "n_complete": d["n_complete"],
            "n_incomplete": d["n_incomplete"],
            "completion_rate_pct": round(100.0 * d["n_complete"] /
                                         (d["n_complete"] + d["n_incomplete"]), 1),
            "fallback_activations": d["fallbacks"],
            "tokens_total": d["tokens"] or None,
            "latency_s": {
                "mean": round(stats.mean(totals), 1),
                "sd": round(_sd(totals), 1),
                "min": round(min(totals), 1),
                "max": round(max(totals), 1),
            },
            "per_agent_s": {
                a: {"mean": round(stats.mean(v), 1), "sd": round(_sd(v), 1),
                    "share_pct": round(100 * stats.mean(v) / grand, 1)}
                for a, v in d["per_agent"].items()
            },
            "per_project_mean_s": {
                str(pid): round(stats.mean([m["total_latency_s"] for m in rows]), 1)
                for pid, rows in sorted(
                    {m["project_id"]: [r for r in d["complete"]
                                       if r["project_id"] == m["project_id"]]
                     for m in d["complete"]}.items())
            },
            "scaffold_build_rate": (
                {"passed": sbr["passed"], "total": sbr["total"],
                 "percent": sbr["sbr_percent"], "partial": sbr.get("partial", False)}
                if sbr else None
            ),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate saved runs into paper tables.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--condition", help="Only this condition (default: all found)")
    parser.add_argument("--latex", action="store_true", help="Emit LaTeX table bodies")
    parser.add_argument("--no-save", action="store_true",
                        help="Print only; do not write results_summary.{txt,json}")
    args = parser.parse_args()

    if not args.runs_dir.is_dir():
        print(f"error: {args.runs_dir} does not exist. Run pipeline_runner first.")
        return 1

    conditions = ([args.condition] if args.condition else
                  sorted(d.name for d in args.runs_dir.iterdir() if d.is_dir()))

    all_data = [d for d in (collect(c, args.runs_dir) for c in conditions) if d]
    if not all_data:
        print(f"error: no runs found under {args.runs_dir}")
        return 1

    # Capture rather than print directly, so the same text reaches both the
    # terminal and the saved report without being formatted twice.
    buf = io.StringIO()
    with redirect_stdout(buf):
        if args.latex:
            print_latex(all_data)
        else:
            for data in all_data:
                print_condition(data)
            print_comparison(all_data)
            print()
    text = buf.getvalue()
    print(text, end="")

    if args.no_save or args.condition:
        return 0

    out_dir = args.runs_dir.parent
    suffix = "_latex" if args.latex else ""
    txt = out_dir / f"results_summary{suffix}.txt"
    txt.write_text(text, encoding="utf-8")
    written = [txt]
    if not args.latex:
        js = out_dir / "results_summary.json"
        js.write_text(json.dumps(summary_dict(all_data), indent=2), encoding="utf-8")
        written.append(js)
    print("saved: " + ", ".join(str(p) for p in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate Arabic output against English output on the same projects.

Reads the runs saved by ``python -m evaluation.pipeline_runner --language en|ar --runs-dir evaluation/runs_lang/<lang>``
and reports, per language and paired by project:

  * completion rate, latency
  * language purity of the human-readable text (Arabic runs: share of letters that are Arabic, ignoring
    identifiers, acronyms and an allow-list of technical terms; English runs: share that is Latin)
  * identifier integrity: JSON enums, feature ids and database names must stay English
  * artifact size, traceability coverage, design feature coverage, schema and scaffold validity
  * an LLM judge (1-5) for faithfulness, consistency, language quality and design usability

The judge is an LLM, not a human; its scores are reported as such.

    python -m evaluation.arabic_eval                 # metrics + judge
    python -m evaluation.arabic_eval --no-judge      # metrics only
"""
import argparse
import json
import math
import os
import random
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

BASE = Path(__file__).resolve().parent / "runs_lang"
JUDGE_MODEL = "claude-opus-5-5"

_AR = re.compile(r"[؀-ۿݐ-ݿ]")
_LAT = re.compile(r"[A-Za-z]")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.\-/]*")
_ALLOW = {
    "json", "api", "apis", "sql", "rest", "jwt", "react", "node", "nodejs", "node.js", "postgresql", "sqlite", "https",
    "http", "ui", "ux", "id", "ids", "csv", "crud", "mvp", "gps", "qr", "sms", "pdf", "html", "css", "url", "uuid",
    "oauth", "sso", "ios", "android", "wifi", "pwa", "app", "email", "e-mail", "ocr", "ssl", "tls", "gdpr", "hipaa",
    "bcrypt", "express", "vite", "docker", "git", "github", "stripe", "paypal", "google", "apple", "pos", "sku",
    "ean", "upc", "rfid", "nfc", "saas", "kpi", "sla", "ci", "cd", "qa", "ai", "ml", "web", "mobile", "high", "medium", "low",
}
# English words of the user-story and Given/When/Then templates, kept English on purpose: the test-case
# generator (core/test_cases.py) parses the Gherkin keywords.
_TEMPLATE = {"as", "a", "an", "i", "can", "so", "that", "given", "when", "then", "and"}
_ENUM_PRIORITY = {"High", "Medium", "Low"}
_ENUM_TASK_PRIORITY = {"high", "low"}
_ENUM_AGENT = {"Developer", "Designer", "DBA"}
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FID = re.compile(r"^F\d+$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_runs(lang: str, base: Path = BASE) -> Iterator[Dict[str, Any]]:
    root = base / lang / "B2"
    for run_dir in sorted(root.glob("run_*")):
        for proj in sorted(run_dir.glob("project_*")):
            meta = _read(proj / "metrics.json")
            if not meta:
                continue
            req = _read(proj / "requirements.json")
            schema = _read(proj / "database_schema.json")
            if isinstance(req, dict) and isinstance(schema, dict):
                req = {**req, "database_schema": schema}
            yield {
                "lang": lang, "run": run_dir.name, "project": int(meta["project_id"]), "meta": meta,
                "req": req, "schema": schema, "design": _read(proj / "design.json"),
                "plan": _read(proj / "cycle_plan.json"), "quality": _read(proj / "quality_report.json"),
                "support": _read(proj / "support_package.json"),
            }


# ---------------------------------------------------------------------------
# Text and identifier checks
# ---------------------------------------------------------------------------

def _strs(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for v in value:
            yield from _strs(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strs(v)


def human_text(run: Dict[str, Any]) -> List[str]:
    """Strings a user reads: requirement text, screen text, plan titles, documentation, review remarks."""
    out: List[str] = []
    req = run["req"] if isinstance(run["req"], dict) else {}
    out += [str(req.get("project_name") or "")]
    for f in req.get("features") or []:
        if isinstance(f, dict):
            out += [str(f.get("name") or ""), str(f.get("description") or "")]
            out += list(_strs(f.get("user_stories"))) + list(_strs(f.get("acceptance_criteria")))
    out += list(_strs(req.get("non_functional_requirements")))
    for t in (req.get("database_schema") or {}).get("tables") or []:
        if isinstance(t, dict):
            out += [str(t.get("purpose") or "")]
            for c in t.get("columns") or []:
                if isinstance(c, dict):
                    out += [str(c.get("notes") or "")]
    d = run["design"] if isinstance(run["design"], dict) else {}
    out += [str(d.get("design_overview") or "")]
    for s in d.get("screens") or []:
        if isinstance(s, dict):
            out += [str(s.get("name") or ""), str(s.get("purpose") or ""), str(s.get("user_flow") or "")]
            for c in s.get("key_components") or []:
                if isinstance(c, dict):
                    out += [str(c.get("name") or ""), str(c.get("interaction") or "")]
    p = run["plan"] if isinstance(run["plan"], dict) else {}
    out += [str(t.get("title") or t.get("name") or "") for t in p.get("tasks") or [] if isinstance(t, dict)]
    out += list(_strs(p.get("risks")))
    q = run["quality"] if isinstance(run["quality"], dict) else {}
    out += [str(i.get("message") or "") for i in q.get("issues") or [] if isinstance(i, dict)]
    out += list(_strs(q.get("recommendations"))) + list(_strs(q.get("required_fixes")))
    sp = run["support"] if isinstance(run["support"], dict) else {}
    out += [str(sp.get("app_documentation") or "")] + list(_strs(sp.get("glossary")))
    return [t for t in out if re.search(r"[A-Za-z؀-ۿ]", t)]


def _stray_latin_letters(text: str, skip_template: bool = False) -> int:
    """Latin letters in words that are not identifiers, acronyms or allow-listed technical terms."""
    total = 0
    for tok in _TOKEN.findall(text):
        low = tok.lower().strip(".-/")
        if low in _ALLOW or "_" in tok or any(c.isdigit() for c in tok):
            continue
        if skip_template and low in _TEMPLATE:
            continue
        if tok.isupper() and len(tok) <= 6:
            continue
        if re.search(r"[a-z][A-Z]", tok):  # camelCase identifier
            continue
        total += len(_LAT.findall(tok))
    return total


def language_purity(texts: List[str], lang: str, skip_template: bool = False) -> float:
    """Share of letters in the intended language (see module docstring).

    ``skip_template`` also ignores the English story/Gherkin template words kept on purpose.
    """
    if lang == "ar":
        ar = sum(len(_AR.findall(t)) for t in texts)
        stray = sum(_stray_latin_letters(t, skip_template) for t in texts)
        return ar / (ar + stray) if ar + stray else 0.0
    lat = sum(len(_LAT.findall(t)) for t in texts)
    ar = sum(len(_AR.findall(t)) for t in texts)
    return lat / (lat + ar) if lat + ar else 0.0


def identifier_integrity(run: Dict[str, Any]) -> Tuple[int, int]:
    """(valid, checked) over fields that must stay English regardless of the output language."""
    ok = n = 0

    def check(cond: bool) -> None:
        nonlocal ok, n
        n += 1
        ok += int(cond)

    req = run["req"] if isinstance(run["req"], dict) else {}
    for f in req.get("features") or []:
        if isinstance(f, dict):
            check(bool(_FID.match(str(f.get("id") or ""))))
            check(f.get("priority") in _ENUM_PRIORITY)
    for t in (req.get("database_schema") or {}).get("tables") or []:
        if isinstance(t, dict):
            check(bool(_IDENT.match(str(t.get("name") or ""))))
            for c in t.get("columns") or []:
                if isinstance(c, dict):
                    check(bool(_IDENT.match(str(c.get("name") or ""))))
    plan = run["plan"] if isinstance(run["plan"], dict) else {}
    for t in plan.get("tasks") or []:
        if isinstance(t, dict):
            check(str(t.get("priority")) in _ENUM_TASK_PRIORITY)
            check(t.get("assigned_agent") in _ENUM_AGENT)
    q = run["quality"] if isinstance(run["quality"], dict) else {}
    for i in q.get("issues") or []:
        if isinstance(i, dict):
            check(i.get("severity") in _ENUM_PRIORITY)
    for k in (q.get("checklist") or {}):
        check(bool(_IDENT.match(str(k))))
    d = run["design"] if isinstance(run["design"], dict) else {}
    for s in d.get("screens") or []:
        if isinstance(s, dict):
            for c in s.get("key_components") or []:
                if isinstance(c, dict):
                    check(not _AR.search(str(c.get("type") or "")))
            for fid in s.get("feature_ids") or []:
                check(bool(_FID.match(str(fid))))
    return ok, n


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

def run_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    from core.test_cases import generate_test_cases
    from core.traceability import build_trace_matrix, uncovered_features
    from utils.code_scaffold import build_scaffold_files
    from utils.self_heal import validate_project, validate_scaffold

    req, design, plan = run["req"], run["design"], run["plan"]
    feats = (req or {}).get("features") or [] if isinstance(req, dict) else []
    screens = [s for s in ((design or {}).get("screens") or []) if isinstance(s, dict)] if isinstance(design, dict) else []
    texts = human_text(run)
    ok, n = identifier_integrity(run)
    m: Dict[str, Any] = {
        "completed": bool(run["meta"].get("completed")),
        "latency_s": float(run["meta"].get("total_latency_s") or 0.0),
        "language_purity": language_purity(texts, run["lang"]),
        "purity_excl_templates": language_purity(texts, run["lang"], skip_template=True),
        "identifier_integrity": ok / n if n else None,
        "features": len(feats), "screens": len(screens),
        "components_per_screen": statistics.mean([len(s.get("key_components") or []) for s in screens]) if screens else 0.0,
        "tables": len(((req or {}).get("database_schema") or {}).get("tables") or []) if isinstance(req, dict) else 0,
        "tasks": len((plan or {}).get("tasks") or []) if isinstance(plan, dict) else 0,
        "test_cases": len(generate_test_cases(req)) if isinstance(req, dict) else 0,
    }
    if isinstance(req, dict) and isinstance(design, dict) and feats:
        m["design_feature_coverage"] = 1 - len(uncovered_features(req, design)) / len(feats)
        m["trace_coverage"] = build_trace_matrix(req, design, plan if isinstance(plan, dict) else None)["coverage"]
    else:
        m["design_feature_coverage"] = m["trace_coverage"] = None
    if isinstance(req, dict):
        m["schema_problems"] = len(validate_project(req, design if isinstance(design, dict) else None))
        try:
            files = build_scaffold_files(req, design if isinstance(design, dict) else None, req.get("database_schema"))
            pages = [p for p in files if p.startswith("frontend/src/pages/")]
            m["scaffold_problems"] = len(validate_scaffold(files))
            m["scaffold_pages_ok"] = int(len(pages) == len(screens))
        except Exception:
            m["scaffold_problems"], m["scaffold_pages_ok"] = None, 0
    return m


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _compact(run: Dict[str, Any]) -> str:
    req = run["req"] if isinstance(run["req"], dict) else {}
    lines = [f"PROJECT: {req.get('project_name')}"]
    for f in req.get("features") or []:
        if isinstance(f, dict):
            lines.append(f"- {f.get('id')} {f.get('name')}: {f.get('description')}")
    d = run["design"] if isinstance(run["design"], dict) else {}
    lines.append("SCREENS:")
    for s in d.get("screens") or []:
        if isinstance(s, dict):
            comps = ", ".join(str(c.get("name")) for c in (s.get("key_components") or [])[:12] if isinstance(c, dict))
            lines.append(f"- {s.get('name')} [{','.join(s.get('feature_ids') or [])}]: {s.get('purpose')} | {comps}")
    p = run["plan"] if isinstance(run["plan"], dict) else {}
    lines.append("TASKS:")
    lines += [f"- {t.get('id')} {t.get('title') or t.get('name')}" for t in (p.get("tasks") or [])[:14] if isinstance(t, dict)]
    return "\n".join(lines)[:9000]


def judge(run: Dict[str, Any], api_key: str) -> Optional[Dict[str, Any]]:
    import requests
    from core.utils import parse_json_from_text

    project = next((p for p in __import__("evaluation.projects", fromlist=["PROJECTS"]).PROJECTS
                    if p["id"] == run["project"]), {})
    request = project.get("input_ar") if run["lang"] == "ar" else project.get("input")
    lang_item = ("Modern Standard Arabic fluency, correct technical terminology, natural phrasing "
                 "(not machine-translated), and no unnecessary English" if run["lang"] == "ar"
                 else "clarity, fluency and consistent terminology")
    prompt = (
        "You are a bilingual (Arabic/English) software-requirements expert. Score the generated project artifacts "
        "below from 1 (very poor) to 5 (excellent) on each criterion:\n"
        "- faithfulness: the artifacts implement what the user's request asks for, without inventing unrelated scope\n"
        "- consistency: features, screens and tasks agree with each other and nothing contradicts\n"
        f"- language_quality: {lang_item}\n"
        "- design_usability: the screens form a coherent, usable flow that covers every feature\n"
        'Output ONLY JSON: {"faithfulness":n,"consistency":n,"language_quality":n,"design_usability":n,"notes":"one sentence"}.\n\n'
        f"USER REQUEST: {request}\n\nARTIFACTS:\n{_compact(run)}"
    )
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": JUDGE_MODEL, "max_tokens": 600, "messages": [{"role": "user", "content": prompt}]},
            timeout=180,
        )
        if r.status_code != 200:
            return None
        text = "".join(b.get("text", "") for b in r.json().get("content", []) if isinstance(b, dict))
        parsed = parse_json_from_text(text)
        keys = ("faithfulness", "consistency", "language_quality", "design_usability")
        if isinstance(parsed, dict) and all(isinstance(parsed.get(k), (int, float)) for k in keys):
            return {k: float(parsed[k]) for k in keys} | {"notes": str(parsed.get("notes", ""))}
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

METRICS = [
    ("completed", "Completion rate"), ("latency_s", "Latency (s)"),
    ("language_purity", "Language purity"), ("purity_excl_templates", "Purity excl. story/Gherkin words"), ("identifier_integrity", "Identifier integrity"),
    ("features", "Features"), ("screens", "Screens"), ("components_per_screen", "Components / screen"),
    ("tables", "Tables"), ("tasks", "Tasks"), ("test_cases", "Test cases"),
    ("design_feature_coverage", "Design feature coverage"), ("trace_coverage", "Traceability coverage"),
    ("schema_problems", "Schema problems"), ("scaffold_problems", "Scaffold problems"),
    ("scaffold_pages_ok", "Scaffold pages = screens"),
    ("faithfulness", "Judge: faithfulness"), ("consistency", "Judge: consistency"),
    ("language_quality", "Judge: language quality"), ("design_usability", "Judge: design usability"),
]


def _by_project(rows: List[Dict[str, Any]], key: str) -> Dict[int, float]:
    acc: Dict[int, List[float]] = {}
    for r in rows:
        v = r.get(key)
        if v is not None:
            acc.setdefault(r["project"], []).append(float(v))
    return {p: statistics.mean(v) for p, v in acc.items()}


def compare(rows_en: List[Dict[str, Any]], rows_ar: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for key, label in METRICS:
        en, ar = _by_project(rows_en, key), _by_project(rows_ar, key)
        both = sorted(set(en) & set(ar))
        if not both:
            continue
        e, a = [en[p] for p in both], [ar[p] for p in both]
        d = [y - x for x, y in zip(e, a)]
        row: Dict[str, Any] = {
            "metric": label, "key": key, "n_projects": len(both),
            "en_mean": statistics.mean(e), "en_sd": statistics.pstdev(e) if len(e) > 1 else 0.0,
            "ar_mean": statistics.mean(a), "ar_sd": statistics.pstdev(a) if len(a) > 1 else 0.0,
            "diff": statistics.mean(d), "p": None,
        }
        if len(d) > 1 and statistics.pstdev(d) > 0:
            try:
                from scipy import stats
                row["p"] = float(stats.ttest_rel(a, e).pvalue)
            except Exception:
                pass
        out.append(row)
    return out


def render(table: List[Dict[str, Any]], n_en: int, n_ar: int) -> str:
    lines = [f"Arabic vs English (B2, runs: en={n_en}, ar={n_ar}; means over repeats, paired by project)", "",
             f"{'metric':<28}{'English':>16}{'Arabic':>16}{'AR-EN':>10}{'p':>8}"]
    for r in table:
        p = f"{r['p']:.3f}" if r["p"] is not None else "  -"
        lines.append(f"{r['metric']:<28}{r['en_mean']:>9.2f}±{r['en_sd']:<5.2f}{r['ar_mean']:>9.2f}±{r['ar_sd']:<5.2f}"
                     f"{r['diff']:>+10.2f}{p:>8}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM judge")
    ap.add_argument("--base", type=Path, default=BASE)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    rows: Dict[str, List[Dict[str, Any]]] = {"en": [], "ar": []}
    cache_path = args.base / "judge_cache.json"
    cache: Dict[str, Any] = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    runs = [r for lang in ("en", "ar") for r in load_runs(lang, args.base)]
    random.Random(0).shuffle(runs)  # judge in random order so language is not the call order
    for run in runs:
        m = run_metrics(run)
        m.update(lang=run["lang"], run=run["run"], project=run["project"])
        if not args.no_judge and key and m["completed"]:
            ck = f"{run['lang']}/{run['run']}/{run['project']}"
            if ck not in cache:
                cache[ck] = judge(run, key)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
            if cache.get(ck):
                m.update({k: v for k, v in cache[ck].items() if k != "notes"})
        rows[run["lang"]].append(m)

    table = compare(rows["en"], rows["ar"])
    text = render(table, len(rows["en"]), len(rows["ar"]))
    print(text)
    (args.base / "arabic_results.json").write_text(
        json.dumps({"runs": rows, "comparison": table, "judge_model": JUDGE_MODEL}, ensure_ascii=False, indent=1))
    (args.base / "arabic_results.txt").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

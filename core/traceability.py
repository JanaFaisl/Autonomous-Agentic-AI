"""Requirements traceability, coverage scoring and change-impact analysis.

Pure functions over the pipeline's JSON artifacts, so they cost no API calls.
Links are inferred from explicit feature IDs first, then from keyword overlap.
"""
import json
import re
from typing import Any, Dict, List, Optional, Set

_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "in", "on", "with", "by", "as", "is", "are",
    "can", "i", "my", "user", "users", "app", "screen", "page", "manage", "management", "view",
    "so", "that", "be", "it", "this", "their", "all", "each", "from", "into", "when", "then", "given",
}

# Pipeline step to re-run when an artifact kind is affected.
_KIND_TO_STEP = {"tables": "database", "screens": "design", "tasks": "planning"}


def _ok(d: Any) -> bool:
    return isinstance(d, dict) and "error" not in d


def _tokens(*texts: Any) -> Set[str]:
    out: Set[str] = set()
    for text in texts:
        for word in re.findall(r"[^\W\d_]+", str(text or "").replace("_", " ").lower()):
            if len(word) < 3 or word in _STOPWORDS:
                continue
            if word.endswith("ies") and len(word) > 4:
                word = word[:-3] + "y"
            elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
                word = word[:-1]
            out.add(word)
    return out


def _linked(feature: Dict[str, Any], name_tokens: Set[str], all_tokens: Set[str],
            candidate_tokens: Set[str], candidate_text: str = "") -> bool:
    fid = str(feature.get("id") or "").strip()
    if fid and re.search(rf"(?<![A-Za-z0-9]){re.escape(fid)}(?![A-Za-z0-9])", candidate_text):
        return True
    if name_tokens & candidate_tokens:
        return True
    return len(all_tokens & candidate_tokens) >= 2


def build_trace_matrix(
    requirements: Optional[Dict[str, Any]],
    design: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Link every feature to the screens, tables, tasks and acceptance criteria covering it.

    Returns {"features": [...], "orphans": {...}, "available": [...], "coverage": float}.
    Coverage only counts artifact kinds that exist, so a missing design isn't a gap in itself.
    """
    if not _ok(requirements):
        return {"features": [], "orphans": {}, "available": [], "coverage": 0.0}

    features = [f for f in (requirements.get("features") or []) if isinstance(f, dict)]
    schema = requirements.get("database_schema")
    tables = [t for t in ((schema or {}).get("tables") or []) if isinstance(t, dict) and t.get("name")] if isinstance(schema, dict) else []
    screens = [s for s in ((design or {}).get("screens") or []) if isinstance(s, dict) and s.get("name")] if _ok(design) else []
    tasks = [t for t in ((plan or {}).get("tasks") or []) if isinstance(t, dict)] if _ok(plan) else []

    available = ["acceptance"]
    if screens:
        available.append("screens")
    if tables:
        available.append("tables")
    if tasks:
        available.append("tasks")

    table_toks = [(t["name"], _tokens(t["name"], t.get("purpose")), t["name"]) for t in tables]
    screen_toks = []
    for s in screens:
        comps = " ".join(str(c.get("name", "")) for c in (s.get("key_components") or []) if isinstance(c, dict))
        screen_toks.append((s["name"], _tokens(s["name"], s.get("purpose"), comps), f"{s['name']} {s.get('purpose', '')}"))
    # Screens that declare feature_ids link only to those features; the rest fall back to matching.
    screen_declared = {s["name"]: {str(i) for i in (s.get("feature_ids") or [])} for s in screens}
    def _ttl(t: Dict[str, Any]) -> str:
        return str(t.get("title") or t.get("name") or "")  # single-prompt output names tasks "name"

    task_toks = [(str(t.get("id") or _ttl(t)), _tokens(_ttl(t)), _ttl(t)) for t in tasks]
    # Tasks that declare feature_ids link only to those features; the rest fall back to matching.
    task_declared = {str(t.get("id") or _ttl(t)): {str(i) for i in (t.get("feature_ids") or [])} for t in tasks}

    used: Dict[str, Set[str]] = {"screens": set(), "tables": set(), "tasks": set()}
    rows: List[Dict[str, Any]] = []
    for f in features:
        name_toks = _tokens(f.get("name"))
        all_toks = name_toks | _tokens(f.get("description"), *(f.get("user_stories") or []))
        row = {
            "id": f.get("id") or f.get("name"),
            "name": f.get("name"),
            "priority": f.get("priority"),
            "acceptance_criteria": len(f.get("acceptance_criteria") or []),
            "screens": [], "tables": [], "tasks": [],
        }
        for kind, cands in (("screens", screen_toks), ("tables", table_toks), ("tasks", task_toks)):
            for label, toks, text in cands:
                declared = task_declared if kind == "tasks" else screen_declared if kind == "screens" else {}
                if declared.get(label):
                    if str(f.get("id")) in declared[label]:
                        row[kind].append(label)
                        used[kind].add(label)
                    continue
                if _linked(f, name_toks, all_toks, toks, text):
                    row[kind].append(label)
                    used[kind].add(label)
        present = {"acceptance": row["acceptance_criteria"] > 0}
        for kind in ("screens", "tables", "tasks"):
            present[kind] = bool(row[kind])
        row["gaps"] = [k for k in available if not present[k]]
        row["coverage"] = round(1 - len(row["gaps"]) / len(available), 2)
        rows.append(row)

    orphans = {
        "screens": [l for l, _, _ in screen_toks if l not in used["screens"]],
        "tables": [l for l, _, _ in table_toks if l not in used["tables"]],
        "tasks": [l for l, _, _ in task_toks if l not in used["tasks"]],
    }
    coverage = round(sum(r["coverage"] for r in rows) / len(rows), 2) if rows else 0.0
    return {"features": rows, "orphans": orphans, "available": available, "coverage": coverage}


def uncovered_features(requirements: Optional[Dict[str, Any]], design: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Requirement features that no design screen implements.

    Uses the screens' declared ``feature_ids`` when the design provides them, otherwise keyword matching.
    """
    if not _ok(requirements) or not _ok(design):
        return []
    screens = [s for s in (design.get("screens") or []) if isinstance(s, dict)]
    if not screens:
        return []
    if any(s.get("feature_ids") for s in screens):
        covered = {str(i) for s in screens for i in (s.get("feature_ids") or [])}
        return [f for f in (requirements.get("features") or [])
                if isinstance(f, dict) and str(f.get("id")) not in covered]
    rows = {str(r["id"]): r for r in build_trace_matrix(requirements, design)["features"]}
    return [f for f in (requirements.get("features") or [])
            if isinstance(f, dict) and not rows.get(str(f.get("id") or f.get("name")), {}).get("screens")]


def diff_requirements(old: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Feature IDs that were added, removed or modified between two requirement snapshots."""
    def index(req: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        feats = (req or {}).get("features") or [] if _ok(req) else []
        return {str(f.get("id") or f.get("name")): f for f in feats if isinstance(f, dict)}

    a, b = index(old), index(new)
    return {
        "added": sorted(b.keys() - a.keys()),
        "removed": sorted(a.keys() - b.keys()),
        "modified": sorted(k for k in a.keys() & b.keys()
                           if json.dumps(a[k], sort_keys=True) != json.dumps(b[k], sort_keys=True)),
    }


def change_impact(matrix: Dict[str, Any], feature_ids: List[str]) -> Dict[str, Any]:
    """Artifacts touched by changing the given features, and the pipeline steps to re-run.

    A feature with no linked artifact of a kind (including brand-new ones) still needs that step run,
    so every available step is flagged for it. Quality and support always re-run.
    """
    by_id = {str(r["id"]): r for r in matrix.get("features", [])}
    impact: Dict[str, Any] = {"screens": [], "tables": [], "tasks": [], "unknown": []}
    steps: Set[str] = set()
    for fid in feature_ids:
        row = by_id.get(str(fid))
        if row is None:
            impact["unknown"].append(fid)
            steps.update(_KIND_TO_STEP.values())
            continue
        for kind, step in _KIND_TO_STEP.items():
            for label in row[kind]:
                if label not in impact[kind]:
                    impact[kind].append(label)
            if row[kind] or kind in row["gaps"]:
                steps.add(step)
    if feature_ids:
        steps.update({"quality", "support"})
    order = ["database", "design", "planning", "quality", "support"]
    impact["steps_to_rerun"] = [s for s in order if s in steps]
    return impact

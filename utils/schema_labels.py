"""Arabic display names for database tables and columns.

Identifiers stay English everywhere (SQL, DDL, scaffold); these labels are only shown in the UI.
One small API call translates every distinct table and column name of a schema.
"""
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import requests

from core.constants import model_for
from core.utils import parse_json_from_text


def schema_names(schema: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Distinct table and column names of a schema, in order of appearance."""
    tables: List[str] = []
    columns: List[str] = []
    for t in ((schema or {}).get("tables") or []):
        if not isinstance(t, dict) or not t.get("name"):
            continue
        if t["name"] not in tables:
            tables.append(t["name"])
        for c in (t.get("columns") or []):
            if isinstance(c, dict) and c.get("name") and c["name"] not in columns:
                columns.append(c["name"])
    return {"tables": tables, "columns": columns}


def schema_key(schema: Optional[Dict[str, Any]]) -> str:
    return hashlib.sha1(json.dumps(schema_names(schema), sort_keys=True).encode()).hexdigest()[:16]


def arabic_labels(schema: Optional[Dict[str, Any]], api_key: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    """{"tables": {name: arabic}, "columns": {name: arabic}}; empty when the call fails."""
    names = schema_names(schema)
    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key or not (names["tables"] or names["columns"]):
        return {"tables": {}, "columns": {}}
    purposes = {t["name"]: t.get("purpose", "") for t in (schema.get("tables") or []) if isinstance(t, dict) and t.get("name")}
    prompt = (
        "Translate these database table and column names into short Arabic display names "
        "(1-3 words, natural for a business user, e.g. users -> المستخدمون, created_at -> تاريخ الإنشاء). "
        "Output ONLY JSON: {\"tables\": {name: arabic}, \"columns\": {name: arabic}} with exactly the given names as keys.\n\n"
        f"TABLES (with purpose): {json.dumps(purposes, ensure_ascii=False)}\n"
        f"COLUMNS: {json.dumps(names['columns'])}"
    )
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
            json={"model": model_for("database"), "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}]},
            timeout=90,
        )
        if resp.status_code != 200:
            return {"tables": {}, "columns": {}}
        text = "".join(b.get("text", "") for b in resp.json().get("content", []) if isinstance(b, dict))
        parsed = parse_json_from_text(text)
    except Exception:
        return {"tables": {}, "columns": {}}
    if not isinstance(parsed, dict) or "error" in parsed:
        return {"tables": {}, "columns": {}}
    clean = lambda d, allowed: {k: str(v) for k, v in (d or {}).items() if k in allowed and isinstance(v, str) and v.strip()}
    return {"tables": clean(parsed.get("tables"), names["tables"]), "columns": clean(parsed.get("columns"), names["columns"])}

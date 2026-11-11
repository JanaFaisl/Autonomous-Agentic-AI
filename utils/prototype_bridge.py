"""Lets the live preview save what the user types into the project's database.

The preview is static HTML inside an iframe and cannot call Python, so a tiny HTTP receiver
runs on 127.0.0.1 in a daemon thread and the preview POSTs to it. Only tables and columns
of the current schema are accepted, and every request carries a per-process token.
"""
import json
import re
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

from core.traceability import build_trace_matrix

_MAX_BODY = 16_000
_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {"server": None, "port": None, "token": secrets.token_urlsafe(16), "tables": {}}
_GENERIC = {"enter", "field", "input", "text", "your", "the"}


def _words(text: str) -> set:
    out = set()
    for w in re.findall(r"[^\W_]+", str(text or "").replace("_", " ").lower()):
        if w in _GENERIC:
            continue
        out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return out


def _is_auto(col: Dict[str, Any]) -> bool:
    name, ctype = str(col.get("name", "")).lower(), str(col.get("type", "")).upper()
    if (col.get("pk") or col.get("primary_key")) and ("INT" in ctype or name == "id"):
        return True
    return name in ("created_at", "updated_at") or ("TIMESTAMP" in ctype and col.get("default") is not None)


def editable_columns(table: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [c for c in (table.get("columns") or []) if isinstance(c, dict) and c.get("name") and not _is_auto(c)]


def match_fields(input_names: List[str], table: Dict[str, Any]) -> Dict[str, str]:
    """Map preview input names to column names by word overlap (each column used once)."""
    scored = []
    for inp in input_names:
        iw = _words(inp)
        for col in editable_columns(table):
            cw = _words(col["name"])
            if not iw or not cw:
                continue
            score = 3 if iw == cw else 2 if (iw <= cw or cw <= iw) else 1 if iw & cw else 0
            if score:
                scored.append((score, inp, col["name"]))
    mapping: Dict[str, str] = {}
    used = set()
    for score, inp, col in sorted(scored, key=lambda t: -t[0]):
        if inp not in mapping and col not in used:
            mapping[inp] = col
            used.add(col)
    return mapping


_INPUT_TYPES = ("input", "textarea", "textfield", "text field", "search bar", "searchbar")


def is_input_component(comp: Dict[str, Any]) -> bool:
    """True for design components the preview renders as a typeable field."""
    ctype = str(comp.get("type", "")).lower()
    return any(t in ctype for t in _INPUT_TYPES)


def _screen_inputs(screen: Dict[str, Any]) -> List[str]:
    names = []
    for comp in (screen.get("key_components") or screen.get("components") or []):
        if isinstance(comp, dict) and is_input_component(comp):
            n = str(comp.get("name") or "").lower().replace(" ", "_")
            if n:
                names.append(n)
    return names


def screen_mappings(req: Optional[Dict[str, Any]], design: Optional[Dict[str, Any]], plan: Any = None) -> Dict[str, Dict[str, Any]]:
    """{screen name: {"table": name, "fields": {input name: column}}} for screens that can save data."""
    schema = (req or {}).get("database_schema") if isinstance(req, dict) else None
    tables = [t for t in ((schema or {}).get("tables") or []) if isinstance(t, dict) and t.get("name")] if isinstance(schema, dict) else []
    screens = [s for s in ((design or {}).get("screens") or []) if isinstance(s, dict) and s.get("name")] if isinstance(design, dict) else []
    if not tables or not screens:
        return {}
    by_name = {t["name"]: t for t in tables}
    matrix = build_trace_matrix(req, design, plan)
    out: Dict[str, Dict[str, Any]] = {}
    for s in screens:
        inputs = _screen_inputs(s)
        if not inputs:
            continue
        linked: List[str] = []
        for row in matrix.get("features", []):
            if s["name"] in row.get("screens", []):
                linked += [t for t in row.get("tables", []) if t in by_name and t not in linked]
        best: Tuple[int, str, Dict[str, str]] = (0, "", {})
        for name in linked + [t for t in by_name if t not in linked]:
            m = match_fields(inputs, by_name[name])
            if len(m) > best[0]:
                best = (len(m), name, m)
        if best[0]:
            out[s["name"]] = {"table": best[1], "fields": best[2]}
    return out


def _coerce(value: Any, ctype: str) -> Any:
    ctype = ctype.upper()
    text = str(value).strip()
    if "BOOL" in ctype:
        return 1 if text.lower() in ("1", "true", "yes", "on", "y") else 0
    if "INT" in ctype:
        return int(float(text))
    if any(t in ctype for t in ("DECIMAL", "NUMERIC", "FLOAT", "REAL", "DOUBLE")):
        return float(text)
    return text


def prepare_row(table: Dict[str, Any], data: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Validate and coerce a submitted row against the table's columns. Returns (row, error)."""
    cols = {c["name"]: c for c in editable_columns(table)}
    unknown = [k for k in data if k not in cols]
    if unknown:
        return {}, "Unknown column(s): " + ", ".join(unknown)
    row: Dict[str, Any] = {}
    for name, col in cols.items():
        ctype = str(col.get("type", "TEXT"))
        if name in data and str(data[name]).strip() != "":
            try:
                row[name] = _coerce(data[name], ctype)
            except (ValueError, TypeError):
                return {}, f"'{name}' must be a {ctype.lower()}"
        elif not col.get("nullable", True) and col.get("default") in (None, ""):
            # NOT NULL column with nothing typed (e.g. a foreign key): neutral placeholder.
            row[name] = 0 if any(t in ctype.upper() for t in ("INT", "BOOL", "DECIMAL", "NUMERIC", "FLOAT", "REAL", "DOUBLE")) else ""
    return row, ""


def save_row(table_name: str, data: Dict[str, Any]) -> Tuple[bool, str]:
    from db import table_exists, create_table_from_schema, insert_object
    table = _STATE["tables"].get(table_name.lower())
    if not table:
        return False, "Unknown table"
    row, err = prepare_row(table, data)
    if err:
        return False, err
    if not row:
        return False, "Nothing to save"
    with _LOCK:
        if not table_exists(table["name"]):
            ok, msg = create_table_from_schema(table["name"], table.get("columns") or [])
            if not ok and not table_exists(table["name"]):
                return False, f"Could not create table: {msg}"
        try:
            insert_object(table["name"], row)
        except Exception as e:  # surfaced to the preview's toast
            return False, f"Could not save: {e}"
    return True, ""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep Streamlit's console clean
        pass

    def _send(self, status: int, body: Dict[str, Any]) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length") or 0)
            if size <= 0 or size > _MAX_BODY:
                return self._send(400, {"ok": False, "error": "Bad request size"})
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict) or not secrets.compare_digest(str(body.get("token", "")), _STATE["token"]):
                return self._send(403, {"ok": False, "error": "Forbidden"})
            data = body.get("data")
            if not isinstance(data, dict):
                return self._send(400, {"ok": False, "error": "Bad data"})
            ok, err = save_row(str(body.get("table", "")), data)
            self._send(200 if ok else 422, {"ok": ok, "error": err})
        except Exception as e:
            self._send(500, {"ok": False, "error": str(e)[:200]})


def build_bridge(req: Optional[Dict[str, Any]], design: Optional[Dict[str, Any]], plan: Any = None) -> Optional[Dict[str, Any]]:
    """Start the receiver (once) and return the config the preview needs, or None if nothing can be saved."""
    mappings = screen_mappings(req, design, plan)
    if not mappings:
        return None
    schema = req["database_schema"]
    _STATE["tables"] = {t["name"].lower(): t for t in schema.get("tables", []) if isinstance(t, dict) and t.get("name")}
    if _STATE["server"] is None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        _STATE["server"], _STATE["port"] = server, server.server_address[1]
    return {"url": f"http://127.0.0.1:{_STATE['port']}/submit", "token": _STATE["token"], "screens": mappings}

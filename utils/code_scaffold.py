"""Turns the generated requirements/design/database specs into an actual runnable
scaffold: an Express + SQLite backend and a React (Vite) frontend, wired to the
user's specific tables and screens. Purely deterministic (no LLM calls).

The frontend intentionally mirrors ui/main_ui.py's build_device_html (the Design
tab's live preview) class-for-class and icon-for-icon — same CSS classes (.appbar,
.card, .btn-primary, .bottom-nav, etc.), same get_component_icon_glyph icon
resolution (imported from core.utils, the shared source of truth), and the same
per-component-type rendering rules — so the scaffolded app looks like the preview,
not a generic placeholder UI.
"""
import json
import re
from typing import Any, Dict, List, Optional

from core.utils import get_component_icon_glyph


def _slug(text: str, fallback: str = "item") -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", str(text or "")).strip("-").lower()
    return text or fallback


def _snake(text: str, fallback: str = "item") -> str:
    return _slug(text, fallback).replace("-", "_")


def _pascal(text: str, fallback: str = "Item") -> str:
    parts = re.split(r"[^a-zA-Z0-9]+", str(text or ""))
    name = "".join(p[:1].upper() + p[1:] for p in parts if p)
    return name or fallback


def _camel(text: str, fallback: str = "item") -> str:
    pascal = _pascal(text, fallback[:1].upper() + fallback[1:])
    return pascal[:1].lower() + pascal[1:] if pascal else fallback


def _primary_key(table: Dict[str, Any]) -> str:
    for col in table.get("columns", []) or []:
        if isinstance(col, dict) and col.get("pk"):
            return col.get("name") or "id"
    return "id"


def _writable_columns(table: Dict[str, Any], pk: str) -> List[str]:
    names = []
    for col in table.get("columns", []) or []:
        if isinstance(col, dict):
            name = col.get("name")
            if name and name != pk:
                names.append(name)
    return names


def _js_str_array(items: List[str]) -> str:
    return "[" + ", ".join(json.dumps(i) for i in items) + "]"


# ---------------------------------------------------------------------------
# Backend (Express + node:sqlite)
# ---------------------------------------------------------------------------
# Uses Node's built-in node:sqlite (stable since Node 22.5+) instead of a native
# npm module like better-sqlite3: no node-gyp/C++ compilation, so nothing to break
# across Node/V8 versions or platforms — just `npm install` and go.

def _render_backend_package_json(project_name: str) -> str:
    return json.dumps({
        "name": _slug(project_name, "app") + "-backend",
        "version": "1.0.0",
        "private": True,
        "type": "commonjs",
        "engines": {"node": ">=22.5.0"},
        "scripts": {"start": "node server.js"},
        "dependencies": {
            "express": "^4.19.2",
            "cors": "^2.8.5",
        },
    }, indent=2) + "\n"


def _render_db_index_js() -> str:
    return """const path = require("path");
const fs = require("fs");
const { DatabaseSync } = require("node:sqlite");

const DB_PATH = path.join(__dirname, "app.db");
const SCHEMA_PATH = path.join(__dirname, "schema.sql");

const db = new DatabaseSync(DB_PATH);
db.exec("PRAGMA foreign_keys = ON");

// Apply schema.sql on every boot; CREATE TABLE IF NOT EXISTS makes this idempotent.
const schema = fs.readFileSync(SCHEMA_PATH, "utf8");
db.exec(schema);

module.exports = db;
"""


_AUTOINCREMENT_TYPES = {"integer", "int", "bigint", "smallint", "serial", "bigserial"}


def _pk_is_autoincrement(table: Dict[str, Any], pk: str) -> bool:
    for col in table.get("columns", []) or []:
        if isinstance(col, dict) and col.get("name") == pk:
            raw_type = str(col.get("type", "")).lower().split("(")[0].strip()
            return raw_type in _AUTOINCREMENT_TYPES
    return False


def _render_route_js(table: Dict[str, Any]) -> str:
    table_name = table.get("name", "items")
    pk = _primary_key(table)
    writable = _writable_columns(table, pk)
    columns_js = _js_str_array(writable)
    autoincrement = _pk_is_autoincrement(table, pk)

    if autoincrement:
        post_body = """// POST /api/{table_name} - create
router.post("/", (req, res) => {{
  const data = pick(req.body || {{}});
  const cols = Object.keys(data);
  if (cols.length === 0) return res.status(400).json({{ error: "No valid fields in body" }});
  const placeholders = cols.map(() => "?").join(", ");
  const stmt = db.prepare(`INSERT INTO ${{TABLE}} (${{cols.join(", ")}}) VALUES (${{placeholders}})`);
  const info = stmt.run(...cols.map((c) => data[c]));
  const created = db.prepare(`SELECT * FROM ${{TABLE}} WHERE ${{PK}} = ?`).get(info.lastInsertRowid);
  res.status(201).json(created || {{ [PK]: info.lastInsertRowid, ...data }});
}});""".format(table_name=table_name)
        requires = 'const db = require("../db");'
    else:
        # Non-integer PK (e.g. uuid/text): SQLite won't auto-generate it, so the server does.
        post_body = """// POST /api/{table_name} - create
router.post("/", (req, res) => {{
  const data = pick(req.body || {{}});
  const id = (req.body && req.body[PK]) || randomUUID();
  const cols = [PK, ...Object.keys(data)];
  const values = [id, ...Object.keys(data).map((c) => data[c])];
  const placeholders = cols.map(() => "?").join(", ");
  const stmt = db.prepare(`INSERT INTO ${{TABLE}} (${{cols.join(", ")}}) VALUES (${{placeholders}})`);
  stmt.run(...values);
  const created = db.prepare(`SELECT * FROM ${{TABLE}} WHERE ${{PK}} = ?`).get(id);
  res.status(201).json(created || {{ [PK]: id, ...data }});
}});""".format(table_name=table_name)
        requires = 'const { randomUUID } = require("crypto");\nconst db = require("../db");'

    return f"""const express = require("express");
{requires}

const router = express.Router();
const TABLE = {json.dumps(table_name)};
const PK = {json.dumps(pk)};
const COLUMNS = {columns_js};

function pick(body) {{
  const row = {{}};
  for (const col of COLUMNS) {{
    if (Object.prototype.hasOwnProperty.call(body, col)) row[col] = body[col];
  }}
  return row;
}}

// GET /api/{table_name} - list
router.get("/", (req, res) => {{
  const rows = db.prepare(`SELECT * FROM ${{TABLE}} LIMIT 100`).all();
  res.json(rows);
}});

// GET /api/{table_name}/:id - fetch one
router.get("/:id", (req, res) => {{
  const row = db.prepare(`SELECT * FROM ${{TABLE}} WHERE ${{PK}} = ?`).get(req.params.id);
  if (!row) return res.status(404).json({{ error: `${{TABLE}} not found` }});
  res.json(row);
}});

{post_body}

// PUT /api/{table_name}/:id - update
router.put("/:id", (req, res) => {{
  const data = pick(req.body || {{}});
  const cols = Object.keys(data);
  if (cols.length === 0) return res.status(400).json({{ error: "No valid fields in body" }});
  const assignments = cols.map((c) => `${{c}} = ?`).join(", ");
  const stmt = db.prepare(`UPDATE ${{TABLE}} SET ${{assignments}} WHERE ${{PK}} = ?`);
  const info = stmt.run(...cols.map((c) => data[c]), req.params.id);
  if (info.changes === 0) return res.status(404).json({{ error: `${{TABLE}} not found` }});
  const updated = db.prepare(`SELECT * FROM ${{TABLE}} WHERE ${{PK}} = ?`).get(req.params.id);
  res.json(updated);
}});

// DELETE /api/{table_name}/:id - remove
router.delete("/:id", (req, res) => {{
  const info = db.prepare(`DELETE FROM ${{TABLE}} WHERE ${{PK}} = ?`).run(req.params.id);
  if (info.changes === 0) return res.status(404).json({{ error: `${{TABLE}} not found` }});
  res.status(204).end();
}});

module.exports = router;
"""


def _render_server_js(project_name: str, tables: List[Dict[str, Any]]) -> str:
    mounts = []
    requires = []
    for table in tables:
        name = table.get("name", "items")
        var = _camel(name, "items")
        requires.append(f'const {var}Router = require("./routes/{_slug(name)}");')
        mounts.append(f'app.use("/api/{_slug(name)}", {var}Router);')

    requires_js = "\n".join(requires) if requires else "// No tables in the generated schema yet."
    mounts_js = "\n".join(mounts) if mounts else "// No routes to mount yet."

    return f"""const express = require("express");
const cors = require("cors");
{requires_js}

const app = express();
app.use(cors());
app.use(express.json());

app.get("/api/health", (req, res) => res.json({{ status: "ok", project: {json.dumps(project_name)} }}));

{mounts_js}

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`{project_name} backend listening on http://localhost:${{PORT}}`));
"""


# ---------------------------------------------------------------------------
# Frontend (React + Vite)
# ---------------------------------------------------------------------------

def _render_frontend_package_json(project_name: str) -> str:
    return json.dumps({
        "name": _slug(project_name, "app") + "-frontend",
        "version": "1.0.0",
        "private": True,
        "type": "module",
        "scripts": {
            "dev": "vite",
            "build": "vite build",
            "preview": "vite preview",
        },
        "dependencies": {
            "react": "^18.3.1",
            "react-dom": "^18.3.1",
            "react-router-dom": "^6.26.0",
        },
        "devDependencies": {
            "@vitejs/plugin-react": "^4.3.1",
            "vite": "^5.4.0",
        },
    }, indent=2) + "\n"


def _render_vite_config() -> str:
    return """import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:4000" },
  },
});
"""


def _render_index_html(project_name: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{project_name}</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
"""


def _render_main_jsx() -> str:
    return """import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
"""


def _screen_path(name: str, screens: List[Dict[str, Any]]) -> str:
    """Map a screen name to its route path — first screen is "/", matching App.jsx's routing."""
    for i, s in enumerate(screens):
        if s.get("name") == name:
            return "/" if i == 0 else f"/{_slug(name)}"
    return f"/{_slug(name)}"


def _find_nav_target(cname: str, interaction: str, screens: List[Dict[str, Any]]) -> Optional[str]:
    """Same heuristic as build_device_html: an explicit 'navigate:Screen' in interaction,
    else a substring/word match between the component name and another screen's name."""
    if interaction and "navigate:" in interaction:
        return interaction.split("navigate:")[-1].strip()
    cname_lower = cname.lower()
    words = [w for w in cname_lower.split() if len(w) > 3]
    for screen in screens:
        screen_name = screen.get("name", "")
        screen_name_lower = screen_name.lower()
        if cname_lower in screen_name_lower or any(w in screen_name_lower for w in words):
            return screen_name
    return None


# Section labels grouping components by type, matching build_device_html's type_labels.
_TYPE_LABELS = [
    ("button", "Primary Actions"),
    ("search", "Navigation & Search"),
    ("input", "Forms & Inputs"),
    ("card", "Content Cards"),
    ("list", "Lists & Menus"),
    ("image", "Media & Images"),
]

# Bottom-nav icon heuristics, matching build_device_html's mobile nav icon_map.
_NAV_ICON_MAP = [
    ("home", "⌂"), ("search", "🔍"), ("profile", "●"), ("settings", "⚙"),
    ("favorites", "★"), ("bookmark", "★"), ("notifications", "◉"), ("translate", "文"),
]


def _nav_icon_for(screen_name: str) -> str:
    lower = screen_name.lower()
    for key, glyph in _NAV_ICON_MAP:
        if key in lower:
            return glyph
    return "●"


def _jsx_text(value: str) -> str:
    """A value as a JSX expression child: literal {"escaped text"} — safe against
    quotes/braces/etc. in the value, and (unlike an f-string {json.dumps(value)}
    used directly as JSX children) actually includes the JSX braces themselves."""
    return "{" + json.dumps(value) + "}"


def _section_label_jsx(ctype: str, seen_types: set) -> str:
    for key, label in _TYPE_LABELS:
        if key in ctype and key not in seen_types:
            seen_types.add(key)
            return f'<div className="section-label">{_jsx_text(label)}</div>\n      '
    return ""


def _render_component_jsx(comp: Dict[str, Any], screens: List[Dict[str, Any]], seen_types: set) -> str:
    cname = comp.get("name", "Component")
    ctype = str(comp.get("type", "Component"))
    ct = ctype.lower()
    icon = get_component_icon_glyph(comp)
    has_image = bool(comp.get("has_image", False))
    interaction = comp.get("interaction", "") or ""
    section = _section_label_jsx(ct, seen_types)

    if "button" in ct:
        nav_target = _find_nav_target(cname, interaction, screens)
        onclick = f'onClick={{() => navigate({json.dumps(_screen_path(nav_target, screens))})}}' if nav_target else ""
        return (
            f'      {section}<button className="btn-primary" {onclick}>\n'
            f'        <span>{_jsx_text(icon)}</span> <span>{_jsx_text(cname)}</span>\n'
            f'      </button>'
        )

    if "search" in ct:
        return (
            f'      {section}<div className="input-wrap">\n'
            f'        <div className="left">{_jsx_text(icon)}</div>\n'
            f'        <input className="input" type="text" placeholder="Search..." />\n'
            f'      </div>'
        )

    if "card" in ct or "image" in ct or has_image:
        nav_target = _find_nav_target(cname, interaction, screens)
        onclick = f'onClick={{() => navigate({json.dumps(_screen_path(nav_target, screens))})}}' if nav_target else ""
        cursor = "pointer" if nav_target else "default"
        chevron = '<span style={{ marginLeft: "auto", opacity: 0.35, fontWeight: 900, fontSize: "1.2rem" }}>›</span>' if nav_target else ""
        image_html = '<div className="image-placeholder">🖼️</div>\n        ' if has_image else ""
        description = comp.get("description", ctype) or ctype
        return (
            f'      {section}<div className="card" style={{{{ cursor: "{cursor}" }}}} {onclick}>\n'
            f'        {image_html}<div className="row">\n'
            f'          <div className="ic">{_jsx_text(icon)}</div>\n'
            f'          <div style={{{{ flex: 1 }}}}>\n'
            f'            <div className="t1">{_jsx_text(cname)}</div>\n'
            f'            <div className="t2">{_jsx_text(description)}</div>\n'
            f'          </div>\n'
            f'          {chevron}\n'
            f'        </div>\n'
            f'      </div>'
        )

    if "list" in ct:
        return (
            f'      {section}<div className="card">\n'
            f'        <div className="row" style={{{{ marginBottom: "8px" }}}}>\n'
            f'          <div className="ic">{_jsx_text(icon)}</div>\n'
            f'          <div className="t1">{_jsx_text(cname)}</div>\n'
            f'        </div>\n'
            f'        <div style={{{{ paddingLeft: "52px" }}}}>\n'
            f'          <div className="list-row">{"• Item 1"}</div>\n'
            f'          <div className="list-row">{"• Item 2"}</div>\n'
            f'          <div className="list-row" style={{{{ borderBottom: "none" }}}}>{"• Item 3"}</div>\n'
            f'        </div>\n'
            f'      </div>'
        )

    if "input" in ct:
        return (
            f'      {section}<div style={{{{ margin: "14px 0" }}}}>\n'
            f'        <div className="input-label">\n'
            f'          <span>{_jsx_text(icon)}</span>\n'
            f'          <label>{_jsx_text(cname)}</label>\n'
            f'        </div>\n'
            f'        <div className="input-wrap">\n'
            f'          <div className="left">{_jsx_text(icon)}</div>\n'
            f'          <input className="input" type="text" placeholder={json.dumps(f"Enter {cname.lower()}...")} />\n'
            f'        </div>\n'
            f'      </div>'
        )

    return (
        f'      {section}<div className="card">\n'
        f'        <div className="row">\n'
        f'          <div className="ic">{_jsx_text(icon)}</div>\n'
        f'          <div style={{{{ flex: 1 }}}}>\n'
        f'            <div className="t1">{_jsx_text(cname)}</div>\n'
        f'            <div className="t2">{_jsx_text(ctype)}</div>\n'
        f'          </div>\n'
        f'        </div>\n'
        f'      </div>'
    )


def _render_page_jsx(screen: Dict[str, Any], screens: List[Dict[str, Any]]) -> str:
    name = screen.get("name", "Screen")
    component_name = _pascal(name, "Screen")
    components = screen.get("key_components", []) or []
    seen_types: set = set()
    rendered = "\n".join(_render_component_jsx(c, screens, seen_types) for c in components if isinstance(c, dict))
    if not rendered:
        rendered = (
            '      <div className="empty-state">\n'
            '        <div className="t1">This screen is defined, but not detailed yet</div>\n'
            '        <div className="t2">No components were generated for this screen.</div>\n'
            '      </div>'
        )

    return f"""import {{ useNavigate }} from "react-router-dom";

export default function {component_name}() {{
  const navigate = useNavigate();
  return (
    <>
      <div className="appbar">
        <button className="back-btn icon-btn" onClick={{() => navigate(-1)}} aria-label="Go back">←</button>
        <div className="title">{_jsx_text(name)}</div>
        <button className="icon-btn" onClick={{() => alert("Notifications")}} aria-label="Notifications">◉</button>
      </div>
      <div className="screen-content">
{rendered}
      </div>
    </>
  );
}}
"""


def _render_app_jsx(project_name: str, screens: List[Dict[str, Any]]) -> str:
    if not screens:
        screens = [{"name": "Home", "purpose": "Landing screen", "key_components": []}]

    imports = []
    routes = []
    nav_items = []
    for i, screen in enumerate(screens):
        name = screen.get("name", f"Screen {i+1}")
        component_name = _pascal(name, f"Screen{i+1}")
        path = "/" if i == 0 else f"/{_slug(name)}"
        imports.append(f'import {component_name} from "./pages/{component_name}.jsx";')
        routes.append(f'          <Route path={json.dumps(path)} element={{<{component_name} />}} />')
        if i < 4:  # bottom nav shows at most 4 items, matching build_device_html
            nav_items.append((path, name))

    imports_js = "\n".join(imports)
    routes_js = "\n".join(routes)
    nav_js = "\n".join(
        f'        <NavLink to={json.dumps(path)} end className={{({{ isActive }}) => "nav-item" + (isActive ? " active" : "")}}>\n'
        f'          <div className="nav-ic">{_jsx_text(_nav_icon_for(name))}</div>\n'
        f'          <div className="lbl">{_jsx_text(name)}</div>\n'
        f'        </NavLink>'
        for path, name in nav_items
    )

    return f"""import {{ Routes, Route, NavLink }} from "react-router-dom";
{imports_js}

export default function App() {{
  return (
    <div className="device-frame">
      <div className="screen-shell">
        <Routes>
{routes_js}
        </Routes>
      </div>
      <nav className="bottom-nav">
{nav_js}
      </nav>
    </div>
  );
}}
"""


def _render_index_css(color_scheme: Dict[str, Any]) -> str:
    primary = color_scheme.get("primary", "#667eea")
    secondary = color_scheme.get("secondary", "#f472b6")
    background = color_scheme.get("background", "#F6F7FB")

    # Mirrors ui/main_ui.py's build_device_html <style> block: same variable names,
    # same class names, so the scaffold's rendered UI matches the live preview.
    return f"""* {{ box-sizing: border-box; }}

:root {{
  --bg: {background};
  --surface: #FFFFFF;
  --text: #111827;
  --muted: #6B7280;
  --border: rgba(17,24,39,.10);
  --shadow: 0 10px 30px rgba(17,24,39,.08);
  --shadow-soft: 0 6px 16px rgba(17,24,39,.06);
  --radius: 18px;
  --radius-sm: 14px;
  --primary: {primary};
  --primary2: {secondary};
}}

body {{
  margin: 0;
  font-family: system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
}}

.device-frame {{
  max-width: 480px;
  margin: 24px auto;
  padding: 0 16px 16px;
}}

.screen-shell {{
  background: var(--bg);
  border-radius: var(--radius);
  padding: 14px;
  box-shadow: var(--shadow);
  min-height: 70vh;
}}

.appbar {{
  background: linear-gradient(90deg, {primary}dd, {secondary}dd);
  border-radius: 16px;
  padding: 14px 14px;
  color: #fff;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: var(--shadow-soft);
  margin-bottom: 14px;
  border: none;
}}
.appbar .title {{ font-weight: 800; letter-spacing: .2px; font-size: 18px; flex: 1; text-align: center; }}

.icon-btn {{
  width: 38px;
  height: 38px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255,255,255,.16);
  border: 1px solid rgba(255,255,255,.18);
  cursor: pointer;
  font-size: 18px;
  color: #fff;
}}
.icon-btn:active {{ transform: scale(.96); }}
.icon-btn:hover {{ background: rgba(255,255,255,.22); }}

.section-label {{
  margin: 16px 4px 10px;
  font-size: 12px;
  font-weight: 900;
  letter-spacing: .08em;
  text-transform: uppercase;
  color: var(--muted);
}}

.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 14px;
  box-shadow: 0 2px 10px rgba(17,24,39,.04);
  margin: 12px 0;
}}

.row {{ display: flex; gap: 12px; align-items: center; }}

.ic {{
  width: 40px;
  height: 40px;
  border-radius: 14px;
  background: rgba(109,94,248,.10);
  border: 1px solid rgba(109,94,248,.14);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 900;
  color: var(--primary);
  font-size: 18px;
  flex-shrink: 0;
}}

.t1 {{ font-weight: 900; color: var(--text); font-size: 16px; line-height: 1.1; }}
.t2 {{ color: var(--muted); font-size: 13px; margin-top: 2px; }}

.input {{
  width: 100%;
  padding: 14px 14px 14px 44px;
  border-radius: 14px;
  border: 1px solid var(--border);
  background: #fff;
  font-size: 15px;
  outline: none;
}}
.input:focus {{
  border-color: {primary}88;
  box-shadow: 0 0 0 4px {primary}25;
}}
.input-wrap {{ position: relative; margin: 10px 0; }}
.input-wrap .left {{
  position: absolute;
  left: 14px;
  top: 50%;
  transform: translateY(-50%);
  opacity: .55;
  font-size: 18px;
}}
.input-label {{ display: flex; align-items: center; gap: 8px; margin-bottom: 6px; color: var(--muted); }}
.input-label label {{ font-weight: 700; color: var(--text); font-size: 0.9rem; }}

.btn-primary {{
  width: 100%;
  padding: 14px 16px;
  border-radius: var(--radius-sm);
  background: linear-gradient(135deg, {secondary}, {primary});
  color: white;
  font-weight: 800;
  text-align: center;
  box-shadow: 0 4px 12px rgba(0,0,0,.12);
  margin: 12px 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  cursor: pointer;
  border: none;
  font-size: 15px;
}}
.btn-primary:active {{ transform: scale(.96); }}

.list-row {{ padding: 8px 0; border-bottom: 1px solid var(--border); color: var(--muted); font-size: 14px; }}

.image-placeholder {{
  width: 100%;
  height: 120px;
  border-radius: 12px;
  background: linear-gradient(135deg, #f0f0f0 0%, #e0e0e0 100%);
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 2rem;
}}

.empty-state {{ padding: 50px 20px; text-align: center; }}
.empty-state .t2 {{ margin-top: 6px; line-height: 1.6; }}

.bottom-nav {{
  margin-top: 12px;
  background: rgba(255,255,255,.92);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 10px 6px;
  display: flex;
  justify-content: space-between;
  gap: 6px;
  box-shadow: var(--shadow-soft);
}}

.nav-item {{
  flex: 1;
  border-radius: 14px;
  padding: 10px 6px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  text-decoration: none;
  color: var(--muted);
  min-width: 0;
}}
.nav-item .nav-ic {{
  width: 26px;
  height: 26px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(17,24,39,.10);
  background: #fff;
  font-size: 16px;
}}
.nav-item .lbl {{
  font-size: 12px;
  font-weight: 800;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
}}
.nav-item.active {{ background: {primary}18; color: var(--primary); }}
.nav-item.active .nav-ic {{
  border-color: {primary}58;
  background: {primary}1f;
  color: var(--primary);
}}
"""


# ---------------------------------------------------------------------------
# README + orchestrator
# ---------------------------------------------------------------------------

def _render_readme(
    requirements: Dict[str, Any],
    tables: List[Dict[str, Any]],
    screens: List[Dict[str, Any]],
    has_backend: bool,
    has_frontend: bool,
) -> str:
    project_name = requirements.get("project_name", "Generated App")
    features = requirements.get("features", []) or []
    feature_lines = "\n".join(
        f"- **{f.get('name', 'Feature')}** ({f.get('priority', '—')}): {f.get('description', '')}"
        for f in features if isinstance(f, dict)
    ) or "- (no features captured yet)"

    table_lines = "\n".join(f"- `{t.get('name')}`" for t in tables) or "- (no tables yet — run the Database agent first)"
    screen_lines = "\n".join(f"- {s.get('name')}" for s in screens) or "- (no screens yet — run the Design agent first)"

    setup_parts = [f"# {project_name}\n", "Generated from your requirements, database schema, and UI/UX design.\n"]
    setup_parts.append("## Features\n" + feature_lines + "\n")
    setup_parts.append("## Database tables\n" + table_lines + "\n")
    setup_parts.append("## Screens\n" + screen_lines + "\n")

    setup_parts.append("## Running this project\n")
    if has_backend:
        setup_parts.append(
            "### Backend (Express + SQLite)\n"
            "Requires **Node.js 22.5+** (uses the built-in `node:sqlite` module — no native compilation, "
            "no node-gyp, nothing that can break across Node versions or platforms).\n"
            "```bash\ncd backend\nnpm install\nnpm start\n```\n"
            "Serves a REST API at `http://localhost:4000/api/<table>` for every generated table "
            "(GET list, GET :id, POST, PUT :id, DELETE :id). The SQLite DB file (`app.db`) is created "
            "automatically from `schema.sql` on first run.\n"
        )
    else:
        setup_parts.append("### Backend\nNot included — no database schema was available when this was generated. Run the Database agent first.\n")

    if has_frontend:
        setup_parts.append(
            "### Frontend (React + Vite)\n"
            "```bash\ncd frontend\nnpm install\nnpm run dev\n```\n"
            "Opens a page per generated screen with basic navigation, styled with your design's color scheme. "
            "The dev server proxies `/api` to the backend on port 4000.\n"
        )
    else:
        setup_parts.append("### Frontend\nNot included — no design was available when this was generated. Run the Design agent first.\n")

    setup_parts.append(
        "## Notes\n"
        "This is a starting scaffold, not a finished app: the frontend renders each screen's components "
        "as simple placeholders (buttons/inputs/cards) rather than final visual designs, and backend routes "
        "are generic CRUD. Use it as a working base to build the real feature logic on top of.\n"
    )
    return "\n".join(setup_parts)


def build_scaffold_files(
    requirements: Dict[str, Any],
    design: Optional[Dict[str, Any]],
    database_schema: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    """Return {relative_path: file_content} for a runnable scaffold tied to the given specs."""
    project_name = requirements.get("project_name", "Generated App")
    tables = (database_schema or {}).get("tables") or []
    tables = [t for t in tables if isinstance(t, dict) and t.get("name")]
    screens = (design or {}).get("screens") or []
    screens = [s for s in screens if isinstance(s, dict) and s.get("name")]
    color_scheme = (design or {}).get("color_scheme") or {}

    files: Dict[str, str] = {}

    has_backend = bool(tables)
    has_frontend = bool(design)

    if has_backend:
        from agents.database_agent import DatabaseAgent
        dba = DatabaseAgent.__new__(DatabaseAgent)
        schema_sql = dba.generate_ddl(database_schema, dialect="sqlite")

        files["backend/package.json"] = _render_backend_package_json(project_name)
        files["backend/schema.sql"] = schema_sql
        files["backend/db.js"] = _render_db_index_js()
        files["backend/server.js"] = _render_server_js(project_name, tables)
        for table in tables:
            files[f"backend/routes/{_slug(table.get('name'))}.js"] = _render_route_js(table)

    if has_frontend:
        files["frontend/package.json"] = _render_frontend_package_json(project_name)
        files["frontend/vite.config.js"] = _render_vite_config()
        files["frontend/index.html"] = _render_index_html(project_name)
        files["frontend/src/main.jsx"] = _render_main_jsx()
        files["frontend/src/index.css"] = _render_index_css(color_scheme)
        pages = screens or [{"name": "Home", "purpose": "Landing screen", "key_components": []}]
        files["frontend/src/App.jsx"] = _render_app_jsx(project_name, pages)
        for screen in pages:
            component_name = _pascal(screen.get("name", "Screen"))
            files[f"frontend/src/pages/{component_name}.jsx"] = _render_page_jsx(screen, pages)

    files["README.md"] = _render_readme(requirements, tables, screens, has_backend, has_frontend)

    return files

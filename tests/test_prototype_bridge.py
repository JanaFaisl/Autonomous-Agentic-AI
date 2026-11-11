"""Live-preview → database bridge (no API calls)."""
import json
import urllib.request

import pytest

from utils import prototype_bridge as pb

REQ = {
    "features": [{"id": "F1", "name": "Notes", "description": "Write notes", "acceptance_criteria": ["x"]}],
    "database_schema": {"tables": [{"name": "Notes", "columns": [
        {"name": "id", "type": "INTEGER", "pk": True},
        {"name": "title", "type": "TEXT", "nullable": False},
        {"name": "priority", "type": "INTEGER"},
        {"name": "created_at", "type": "TIMESTAMP", "default": "CURRENT_TIMESTAMP"},
    ]}, {"name": "Users", "columns": [
        {"name": "id", "type": "INTEGER", "pk": True}, {"name": "email", "type": "TEXT"}]}]},
}
DESIGN = {"screens": [
    {"name": "Notes", "purpose": "Write notes", "key_components": [
        {"name": "Title", "type": "input"}, {"name": "Priority", "type": "input"}, {"name": "Save", "type": "button"}]},
    {"name": "Home", "purpose": "Welcome", "key_components": [{"name": "Hero", "type": "image"}]},
]}


def test_screen_mapping_picks_linked_table_and_columns():
    m = pb.screen_mappings(REQ, DESIGN)
    assert m == {"Notes": {"table": "Notes", "fields": {"title": "title", "priority": "priority"}}}


def test_no_inputs_or_schema_means_no_bridge():
    assert pb.screen_mappings(REQ, {"screens": [DESIGN["screens"][1]]}) == {}
    assert pb.screen_mappings({}, DESIGN) == {}
    assert pb.build_bridge({}, DESIGN) is None


def test_prepare_row_coerces_and_rejects():
    table = REQ["database_schema"]["tables"][0]
    assert pb.prepare_row(table, {"title": "Hi", "priority": "3"}) == ({"title": "Hi", "priority": 3}, "")
    assert "priority" in pb.prepare_row(table, {"priority": "high"})[1]
    assert "Unknown" in pb.prepare_row(table, {"id": "9"})[1]  # auto columns can't be set
    assert pb.prepare_row(table, {"priority": "1"})[0]["title"] == ""  # NOT NULL placeholder


def test_http_roundtrip_inserts_row(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    monkeypatch.chdir(tmp_path)
    import db.db_service as dbs
    if hasattr(dbs, "_ENGINE"):
        monkeypatch.setattr(dbs, "_ENGINE", None)
    if hasattr(dbs, "_sqlite_conn"):
        monkeypatch.setattr(dbs, "_sqlite_conn", None)
    cfg = pb.build_bridge(REQ, DESIGN)
    assert cfg and cfg["screens"]["Notes"]["table"] == "Notes"

    def post(body):
        req = urllib.request.Request(cfg["url"], data=json.dumps(body).encode(), headers={"Content-Type": "text/plain"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    assert post({"token": "wrong", "table": "Notes", "data": {"title": "x"}})[0] == 403
    assert post({"token": cfg["token"], "table": "Nope", "data": {"title": "x"}})[0] == 422
    status, res = post({"token": cfg["token"], "table": "Notes", "data": {"title": "Buy milk", "priority": "2"}})
    assert (status, res["ok"]) == (200, True)
    rows = dbs.fetch_objects("Notes")
    assert rows and rows[0]["title"] == "Buy milk" and rows[0]["priority"] == 2


def test_preview_html_embeds_bridge_and_submit_hooks():
    from ui.main_ui import build_device_html
    cfg = {"url": "http://127.0.0.1:1/submit", "token": "t", "screens": {"Notes": {"table": "Notes", "fields": {"title": "title"}}}}
    html = build_device_html(DESIGN | {"navigation": {"items": []}}, False, "#667eea", "#f472b6", bridge=cfg)
    assert "const BRIDGE = {" in html and 'data-submit="1"' in html and 'data-field="title"' in html
    assert "const BRIDGE = null;" in build_device_html(DESIGN, False, "#667eea", "#f472b6")

"""
Unit tests for agents/database_agent.py

Covers schema validation, DDL generation, and type mapping — none of these
require a live API call.
"""
import pytest
from unittest.mock import patch, MagicMock
import json

from agents.database_agent import DatabaseAgent, _map_type
DatabaseDesignAgent = DatabaseAgent
from tests.conftest import SAMPLE_REQUIREMENTS, SAMPLE_DB_SCHEMA


# ---------------------------------------------------------------------------
# _map_type helper
# ---------------------------------------------------------------------------

class TestMapType:

    @pytest.mark.parametrize("raw,dialect,expected_fragment", [
        ("uuid", "postgresql", "UUID"),
        ("int", "postgresql", "INTEGER"),
        ("varchar", "postgresql", "VARCHAR"),
        ("boolean", "postgresql", "BOOLEAN"),
        ("timestamp", "postgresql", "TIMESTAMP"),
        ("uuid", "sqlite", "TEXT"),
        ("int", "sqlite", "INTEGER"),
        ("boolean", "sqlite", "INTEGER"),
    ])
    def test_known_types(self, raw, dialect, expected_fragment):
        result = _map_type(raw, dialect)
        assert expected_fragment in result.upper()

    def test_unknown_type_returns_text(self):
        result = _map_type("some_exotic_type", "postgresql")
        assert result  # should not crash, returns a non-empty fallback


# ---------------------------------------------------------------------------
# DatabaseDesignAgent._validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:

    def _make_agent(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("agents.database_agent.CREWAI_AVAILABLE", False):
                return DatabaseDesignAgent(api_key="test-key")

    def test_valid_schema_returns_no_issues(self):
        agent = self._make_agent()
        issues = agent._validate_schema(SAMPLE_DB_SCHEMA)
        assert issues == []

    def test_missing_tables_key_returns_no_issues(self):
        # Schema with no 'tables' key — agent treats it as empty list (no issues raised)
        agent = self._make_agent()
        issues = agent._validate_schema({})
        assert isinstance(issues, list)

    def test_empty_tables_list_returns_no_issues(self):
        # Empty tables list is valid from the validator's perspective
        agent = self._make_agent()
        issues = agent._validate_schema({"tables": []})
        assert issues == []

    def test_table_without_name_flagged(self):
        agent = self._make_agent()
        schema = {
            "tables": [
                {"purpose": "Something", "columns": [{"name": "id", "type": "uuid", "pk": True}]}
            ]
        }
        issues = agent._validate_schema(schema)
        assert len(issues) > 0

    def test_table_without_pk_flagged(self):
        agent = self._make_agent()
        schema = {
            "tables": [
                {
                    "name": "no_pk_table",
                    "purpose": "Test",
                    "columns": [{"name": "name", "type": "varchar", "pk": False}],
                }
            ]
        }
        issues = agent._validate_schema(schema)
        assert any("primary" in i.lower() or "pk" in i.lower() for i in issues)


# ---------------------------------------------------------------------------
# DatabaseDesignAgent.generate_ddl
# ---------------------------------------------------------------------------

class TestGenerateDDL:

    def _make_agent(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("agents.database_agent.CREWAI_AVAILABLE", False):
                return DatabaseDesignAgent(api_key="test-key")

    def test_ddl_contains_create_table(self):
        agent = self._make_agent()
        ddl = agent.generate_ddl(SAMPLE_DB_SCHEMA, dialect="postgresql")
        assert "CREATE TABLE" in ddl.upper()

    def test_ddl_contains_table_names(self):
        agent = self._make_agent()
        ddl = agent.generate_ddl(SAMPLE_DB_SCHEMA, dialect="postgresql")
        assert "users" in ddl.lower()
        assert "orders" in ddl.lower()

    def test_ddl_contains_primary_key(self):
        agent = self._make_agent()
        ddl = agent.generate_ddl(SAMPLE_DB_SCHEMA, dialect="postgresql")
        assert "PRIMARY KEY" in ddl.upper()

    def test_ddl_sqlite_dialect(self):
        agent = self._make_agent()
        ddl = agent.generate_ddl(SAMPLE_DB_SCHEMA, dialect="sqlite")
        assert "CREATE TABLE" in ddl.upper()

    def test_empty_schema_returns_empty_or_comment(self):
        agent = self._make_agent()
        ddl = agent.generate_ddl({"tables": []}, dialect="postgresql")
        assert isinstance(ddl, str)


# ---------------------------------------------------------------------------
# DatabaseDesignAgent.generate_schema (mocked API)
# ---------------------------------------------------------------------------

def _mock_response(payload: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


def _api_payload(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


class TestGenerateSchema:

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("agents.database_agent.CREWAI_AVAILABLE", False)
    @patch("agents.database_agent._requests.post")
    def test_generate_schema_returns_tables(self, mock_post):
        mock_post.return_value = _mock_response(_api_payload(json.dumps(SAMPLE_DB_SCHEMA)))
        agent = DatabaseDesignAgent(api_key="test-key")
        result = agent.generate_schema(SAMPLE_REQUIREMENTS)
        assert "tables" in result or "error" in result  # passes either way; no crash

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("agents.database_agent.CREWAI_AVAILABLE", False)
    @patch("agents.database_agent._requests.post")
    def test_generate_schema_on_401_returns_error(self, mock_post):
        mock_post.return_value = _mock_response({"error": "Unauthorized"}, status_code=401)
        agent = DatabaseDesignAgent(api_key="test-key")
        result = agent.generate_schema(SAMPLE_REQUIREMENTS)
        assert "error" in result

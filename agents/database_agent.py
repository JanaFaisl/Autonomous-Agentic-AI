import json
from typing import Dict, Any, List, Optional

import requests as _requests

from core.constants import (
    DEFAULT_MODEL, MAX_ITERATIONS, MAX_EXECUTION_TIME,
    ERROR_NO_API_KEY, ERROR_PARSE_JSON, SOLUTION_ADD_API_KEY, SOLUTION_RETRY,
)
from core.models import PYDANTIC_AVAILABLE, DatabaseSchemaModel
from core.llm import CREWAI_AVAILABLE, Agent, Task, Crew, Process, get_llm
from core.utils import parse_json_from_text, extract_crewai_output, create_error_response
from utils.io_suppression import suppress_stderr, suppress_io

# SQL type mappings for DDL generation
_PG_TYPE_MAP = {
    "uuid": "UUID",
    "int": "INTEGER",
    "integer": "INTEGER",
    "bigint": "BIGINT",
    "smallint": "SMALLINT",
    "float": "DOUBLE PRECISION",
    "decimal": "NUMERIC",
    "numeric": "NUMERIC",
    "varchar": "VARCHAR(255)",
    "string": "VARCHAR(255)",
    "text": "TEXT",
    "timestamp": "TIMESTAMP",
    "datetime": "TIMESTAMP",
    "date": "DATE",
    "time": "TIME",
    "boolean": "BOOLEAN",
    "bool": "BOOLEAN",
    "json": "JSONB",
    "jsonb": "JSONB",
    "blob": "BYTEA",
    "serial": "SERIAL",
    "bigserial": "BIGSERIAL",
}

_SQLITE_TYPE_MAP = {
    "uuid": "TEXT",
    "int": "INTEGER",
    "integer": "INTEGER",
    "bigint": "INTEGER",
    "smallint": "INTEGER",
    "float": "REAL",
    "decimal": "REAL",
    "numeric": "NUMERIC",
    "varchar": "TEXT",
    "string": "TEXT",
    "text": "TEXT",
    "timestamp": "TEXT",
    "datetime": "TEXT",
    "date": "TEXT",
    "time": "TEXT",
    "boolean": "INTEGER",
    "bool": "INTEGER",
    "json": "TEXT",
    "jsonb": "TEXT",
    "blob": "BLOB",
    "serial": "INTEGER",
    "bigserial": "INTEGER",
}


def _map_type(raw_type: str, dialect: str) -> str:
    key = raw_type.lower().split("(")[0].strip()
    mapping = _PG_TYPE_MAP if dialect == "postgresql" else _SQLITE_TYPE_MAP
    return mapping.get(key, raw_type.upper())


# Cross-dialect DEFAULT expression translation. Keys are lowercased, whitespace-stripped.
# None means "no safe equivalent — drop the default" rather than emit invalid SQL.
_SQLITE_DEFAULT_FUNCS = {
    "now()": "CURRENT_TIMESTAMP",
    "current_timestamp()": "CURRENT_TIMESTAMP",
    "gen_random_uuid()": None,
    "uuid()": None,
    "uuid_generate_v4()": None,
}

_BAREWORD_KEYWORDS = {"current_timestamp", "current_date", "current_time"}


def _format_default(default: Any, dialect: str) -> Optional[str]:
    """Turn a schema's `default` value into a valid DDL DEFAULT expression for the given
    dialect, or None if it should be omitted (no safe cross-dialect equivalent).
    """
    if default is None:
        return None
    if isinstance(default, bool):
        if dialect == "sqlite":
            return "1" if default else "0"
        return "TRUE" if default else "FALSE"
    if isinstance(default, (int, float)):
        return str(default)

    text = str(default).strip()
    if not text:
        return None
    lower = text.lower()

    if lower == "null":
        return "NULL"
    if lower == "true":
        return "1" if dialect == "sqlite" else "TRUE"
    if lower == "false":
        return "0" if dialect == "sqlite" else "FALSE"
    if lower in _BAREWORD_KEYWORDS:
        return lower.upper()

    compact = lower.replace(" ", "")
    if dialect == "sqlite" and compact in _SQLITE_DEFAULT_FUNCS:
        return _SQLITE_DEFAULT_FUNCS[compact]

    # Function-call-shaped default (e.g. foo() or foo(arg)).
    if "(" in text and text.rstrip().endswith(")"):
        # SQLite requires non-literal expression defaults to be parenthesized;
        # PostgreSQL accepts a bare function call directly.
        return f"({text})" if dialect == "sqlite" else text

    # Plain numeric literal passed as a string.
    try:
        float(text)
        return text
    except ValueError:
        pass

    escaped = text.replace("'", "''")
    return f"'{escaped}'"


class DatabaseAgent:
    """Agent that derives or refines database schema from requirements (CrewAI)."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.agent = None
        self.crew = None
        self.task = None

        if not CREWAI_AVAILABLE:
            return

        llm = get_llm()
        agent_kwargs = {
            "role": "Database Agent",
            "goal": "Produce a clear, minimal database schema (tables, columns, relationships) from software requirements.",
            "backstory": "You are a database architect. You derive practical schemas from requirements and output structured JSON.",
            "verbose": False,
            "allow_delegation": False,
            "max_iter": MAX_ITERATIONS,
            "max_execution_time": MAX_EXECUTION_TIME,
        }
        if llm:
            agent_kwargs["llm"] = llm

        try:
            from utils.io_suppression import suppress_stderr
            with suppress_stderr():
                self.agent = Agent(**agent_kwargs)
        except Exception:
            self.agent = None
            return

        task_kwargs = {
            "description": "From the given requirements, output a database_schema JSON: tables (name, purpose, columns with name/type/pk/nullable/unique/default/notes, indexes, relationships), and assumptions.",
            "agent": self.agent,
            "expected_output": "A JSON object with tables and assumptions. Use common SQL types (uuid, int, bigint, varchar, text, timestamp, boolean, json).",
        }
        if PYDANTIC_AVAILABLE and DatabaseSchemaModel is not None:
            task_kwargs["output_json"] = DatabaseSchemaModel

        self.task = Task(**task_kwargs)

        try:
            from utils.io_suppression import suppress_stderr
            with suppress_stderr():
                self.crew = Crew(agents=[self.agent], tasks=[self.task], verbose=False, process=Process.sequential)
        except Exception:
            self.crew = None

    def _direct_api_call(self, prompt: str) -> Dict[str, Any]:
        """Fallback: call Anthropic Messages API directly when CrewAI is unavailable or fails."""
        try:
            resp = _requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": DEFAULT_MODEL,
                    "max_tokens": 4096,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=90,
            )
            if resp.status_code != 200:
                return create_error_response(f"API error {resp.status_code}", SOLUTION_RETRY)
            data = resp.json()
            text = "\n".join(
                b.get("text", "") for b in (data.get("content") or [])
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            parsed = parse_json_from_text(text) if text else {}
            return parsed if isinstance(parsed, dict) and "error" not in parsed else create_error_response(ERROR_PARSE_JSON, SOLUTION_RETRY)
        except Exception as e:
            return create_error_response(str(e)[:200], SOLUTION_RETRY)

    # ------------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------------

    def _validate_schema(self, schema: Dict[str, Any]) -> List[str]:
        """Return a list of validation issues found in the schema."""
        issues = []
        tables = schema.get("tables", [])
        if not isinstance(tables, list):
            return ["'tables' is not a list"]

        table_names = set()
        table_column_map: Dict[str, set] = {}

        for table in tables:
            name = table.get("name", "").strip()
            if not name:
                issues.append("A table is missing its name.")
                continue

            if name in table_names:
                issues.append(f"Duplicate table name: '{name}'.")
            table_names.add(name)

            columns = table.get("columns", [])
            col_names = {c.get("name", "") for c in columns if isinstance(c, dict)}
            table_column_map[name] = col_names

            has_pk = any(c.get("pk") for c in columns if isinstance(c, dict))
            if not has_pk:
                issues.append(f"Table '{name}' has no primary key column.")

        # Check foreign keys reference real tables and columns
        for table in tables:
            tname = table.get("name", "")
            for rel in table.get("relationships", []):
                if not isinstance(rel, dict):
                    continue
                to_table = rel.get("to_table", "")
                fk_col = rel.get("fk", "")
                ref_col = rel.get("ref", "")

                if to_table and to_table not in table_names:
                    issues.append(
                        f"Table '{tname}' has a relationship to unknown table '{to_table}'."
                    )
                if fk_col and fk_col not in table_column_map.get(tname, set()):
                    issues.append(
                        f"Table '{tname}': foreign key column '{fk_col}' does not exist in this table."
                    )
                if ref_col and to_table in table_column_map and ref_col not in table_column_map[to_table]:
                    issues.append(
                        f"Table '{tname}': referenced column '{ref_col}' does not exist in table '{to_table}'."
                    )

        return issues

    def _fix_schema(self, schema: Dict[str, Any], issues: List[str], requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the model to correct the schema given a list of issues."""
        issues_text = "\n".join(f"- {i}" for i in issues)
        prompt = f"""The following database schema has validation issues. Fix ALL of them and return a corrected schema. Output ONLY a JSON object with no markdown.

ORIGINAL SCHEMA:
{json.dumps(schema, indent=2)}

ISSUES TO FIX:
{issues_text}

ORIGINAL REQUIREMENTS (for context):
{json.dumps(requirements, indent=2)}

Return the corrected JSON with the same structure: tables (name, purpose, columns, indexes, relationships) and assumptions."""

        fixed = self._direct_api_call(prompt)
        if "error" not in fixed:
            return fixed
        return schema  # return original if correction also fails

    # ------------------------------------------------------------------
    # DDL generation
    # ------------------------------------------------------------------

    def generate_ddl(self, schema: Dict[str, Any], dialect: str = "postgresql") -> str:
        """Convert schema JSON to CREATE TABLE SQL statements.

        Args:
            schema: The database schema dict (as returned by generate_schema).
            dialect: 'postgresql' or 'sqlite'.

        Returns:
            A string of SQL DDL statements.
        """
        if dialect not in ("postgresql", "sqlite"):
            dialect = "postgresql"

        tables = schema.get("tables", [])
        if not tables:
            return "-- No tables found in schema."

        lines: List[str] = []
        if dialect == "postgresql":
            lines.append('-- Generated DDL (PostgreSQL)\n')
        else:
            lines.append('-- Generated DDL (SQLite)\n')

        for table in tables:
            tname = table.get("name", "unnamed")
            purpose = table.get("purpose", "")
            columns = table.get("columns", [])
            indexes = table.get("indexes", [])
            relationships = table.get("relationships", [])

            if purpose:
                lines.append(f"-- {purpose}")

            col_defs: List[str] = []
            pk_cols: List[str] = []

            for col in columns:
                if not isinstance(col, dict):
                    continue
                cname = col.get("name", "col")
                ctype = _map_type(col.get("type", "text"), dialect)
                nullable = col.get("nullable", True)
                is_pk = col.get("pk", False)
                unique = col.get("unique", False)
                default = col.get("default")

                # SQLite uses AUTOINCREMENT for integer PKs
                if is_pk and dialect == "sqlite" and ctype == "INTEGER":
                    col_defs.append(f"    {cname} INTEGER PRIMARY KEY AUTOINCREMENT")
                    continue
                # PostgreSQL uses SERIAL / UUID default for PKs
                elif is_pk and dialect == "postgresql":
                    if ctype in ("INTEGER", "BIGINT"):
                        col_defs.append(f"    {cname} {ctype} GENERATED ALWAYS AS IDENTITY PRIMARY KEY")
                    elif ctype == "UUID":
                        col_defs.append(f"    {cname} UUID PRIMARY KEY DEFAULT gen_random_uuid()")
                    else:
                        col_defs.append(f"    {cname} {ctype} PRIMARY KEY")
                    continue

                parts = [f"    {cname}", ctype]
                if not nullable:
                    parts.append("NOT NULL")
                if unique:
                    parts.append("UNIQUE")
                formatted_default = _format_default(default, dialect)
                if formatted_default is not None:
                    parts.append(f"DEFAULT {formatted_default}")
                col_defs.append(" ".join(parts))

                if is_pk:
                    pk_cols.append(cname)

            # PK constraint for columns that fell through the inline-PK branches above
            # (non-integer PKs on any dialect, and composite PKs on postgresql).
            if pk_cols:
                col_defs.append(f"    PRIMARY KEY ({', '.join(pk_cols)})")

            # Foreign key constraints
            for rel in relationships:
                if not isinstance(rel, dict):
                    continue
                fk = rel.get("fk")
                to_table = rel.get("to_table")
                ref = rel.get("ref", "id")
                if fk and to_table:
                    col_defs.append(f"    FOREIGN KEY ({fk}) REFERENCES {to_table}({ref})")

            lines.append(f"CREATE TABLE IF NOT EXISTS {tname} (")
            lines.append(",\n".join(col_defs))
            lines.append(");\n")

            # Indexes. Models sometimes return an already-formed index name (e.g.
            # "idx_users_email") instead of the plain column name the prompt asked for
            # ("email") — resolve back to a real column, or skip rather than emit SQL
            # referencing a column that doesn't exist.
            column_by_lower = {
                c.get("name", "").lower(): c.get("name", "")
                for c in columns if isinstance(c, dict) and c.get("name")
            }
            for idx in indexes:
                if not (isinstance(idx, str) and idx.strip()):
                    continue
                idx = idx.strip()
                resolved = column_by_lower.get(idx.lower())
                if resolved is None:
                    for prefix in (f"idx_{tname}_", f"ix_{tname}_", f"index_{tname}_", "idx_", "ix_", "index_"):
                        if idx.lower().startswith(prefix.lower()):
                            resolved = column_by_lower.get(idx[len(prefix):].lower())
                            if resolved:
                                break
                if not resolved:
                    continue
                col_hint = resolved.replace(" ", "_").lower()
                lines.append(f"CREATE INDEX IF NOT EXISTS idx_{tname}_{col_hint} ON {tname}({resolved});")

            lines.append("")  # blank line between tables

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise the model response to always have top-level 'tables' key.

        Models sometimes wrap the schema in an extra object, e.g.:
          {"database_schema": {"tables": [...], "assumptions": [...]}}
          {"schema": {"tables": [...]}}
        This strips that outer wrapper so callers can always do result["tables"].
        """
        if not isinstance(raw, dict):
            return raw
        # Already the right shape
        if isinstance(raw.get("tables"), list):
            return raw
        # Look one level deep for any value that looks like a schema
        for v in raw.values():
            if isinstance(v, dict) and isinstance(v.get("tables"), list):
                return v
        return raw

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def generate_schema(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Generate or refine database schema from requirements."""
        if not self.api_key:
            return create_error_response(ERROR_NO_API_KEY, SOLUTION_ADD_API_KEY)

        req_json = json.dumps(requirements, indent=2)
        prompt = f"""From these requirements, produce a complete database schema. Rules:
- Use snake_case for all table and column names.
- Every table must have exactly one primary key column.
- Explicitly define foreign key columns (e.g. user_id) and list them in the relationships array with fk/ref fields.
- Add junction tables for many-to-many relationships.
- Output ONLY a flat JSON object (no wrapper key) with no markdown.

REQUIREMENTS:
{req_json}

Output JSON structure exactly:
{{
  "tables": [
    {{
      "name": "table_name",
      "purpose": "...",
      "columns": [{{"name": "id", "type": "uuid", "pk": true, "nullable": false, "unique": false, "default": null, "notes": null}}],
      "indexes": ["col_name"],
      "relationships": [{{"type": "many-to-one", "to_table": "other", "fk": "other_id", "ref": "id", "notes": null}}]
    }}
  ],
  "assumptions": ["..."]
}}"""

        def _process(result: Dict[str, Any]) -> Dict[str, Any]:
            if "error" in result:
                return result
            result = self._unwrap(result)
            # Only validate when we actually have tables — empty tables is a
            # generation failure, not a fixable schema issue.
            if result.get("tables"):
                issues = self._validate_schema(result)
                if issues:
                    result = self._fix_schema(result, issues, requirements)
                    result = self._unwrap(result)
            return result

        # 1) Try CrewAI first — full orchestration with role, goal, and backstory context
        if CREWAI_AVAILABLE and self.crew:
            self.task.description = prompt
            try:
                from utils.io_suppression import suppress_io
                with suppress_io():
                    crew_result = self.crew.kickoff(inputs={"requirements": req_json})

                if hasattr(crew_result, "tasks_output") and crew_result.tasks_output:
                    task_out = crew_result.tasks_output[0]
                    if hasattr(task_out, "json_dict") and isinstance(task_out.json_dict, dict):
                        result = task_out.json_dict
                    elif hasattr(task_out, "json") and isinstance(task_out.json, dict):
                        result = task_out.json
                    else:
                        raw = extract_crewai_output(crew_result)
                        result = parse_json_from_text(raw) if raw else {}

                result = _process(result)
                if isinstance(result, dict) and "error" not in result:
                    return result
            except Exception:
                pass

        # 2) Fallback to direct Anthropic API if CrewAI unavailable or failed
        result = _process(self._direct_api_call(prompt))
        if "error" not in result:
            return result

        return result


# ----------------------------
# PROJECT MANAGER AGENT (Central Orchestrator)
# ----------------------------

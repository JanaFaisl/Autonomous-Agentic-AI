"""
Unit tests for core/utils.py

Tests JSON parsing robustness and CrewAI output extraction — these are
load-bearing utilities that every agent depends on.
"""
import pytest
from core.utils import parse_json_from_text, extract_crewai_output, create_error_response


# ---------------------------------------------------------------------------
# parse_json_from_text
# ---------------------------------------------------------------------------

class TestParseJsonFromText:

    def test_clean_json(self):
        text = '{"key": "value", "number": 42}'
        result = parse_json_from_text(text)
        assert result == {"key": "value", "number": 42}

    def test_json_wrapped_in_markdown_code_block(self):
        text = '```json\n{"project_name": "Test App"}\n```'
        result = parse_json_from_text(text)
        assert result["project_name"] == "Test App"

    def test_json_with_leading_prose(self):
        text = 'Here is the JSON you asked for:\n{"status": "ok"}'
        result = parse_json_from_text(text)
        assert result["status"] == "ok"

    def test_json_with_trailing_comma(self):
        text = '{"a": 1, "b": 2,}'
        result = parse_json_from_text(text)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_nested_json(self):
        text = '{"outer": {"inner": [1, 2, 3]}}'
        result = parse_json_from_text(text)
        assert result["outer"]["inner"] == [1, 2, 3]

    def test_empty_string_returns_error(self):
        result = parse_json_from_text("")
        assert "error" in result

    def test_no_json_object_returns_error(self):
        result = parse_json_from_text("This is just plain text with no JSON.")
        assert "error" in result

    def test_malformed_json_returns_error(self):
        result = parse_json_from_text('{"unclosed": "bracket"')
        assert "error" in result

    def test_array_at_top_level_returns_error(self):
        # Agents always return objects, not bare arrays
        result = parse_json_from_text("[1, 2, 3]")
        assert "error" in result

    def test_large_realistic_requirements_json(self):
        text = """{
            "project_name": "Flower Delivery",
            "features": [{"id": "F1", "name": "Browse", "description": "...", "priority": "High",
                          "user_stories": [], "acceptance_criteria": []}],
            "technical_requirements": {"platform": ["Web"], "technologies": ["React"], "database": "PostgreSQL"},
            "user_roles": ["Customer"]
        }"""
        result = parse_json_from_text(text)
        assert result["project_name"] == "Flower Delivery"
        assert len(result["features"]) == 1
        assert "error" not in result


# ---------------------------------------------------------------------------
# extract_crewai_output
# ---------------------------------------------------------------------------

class TestExtractCrewAIOutput:

    def test_string_result_returned_as_is(self):
        class FakeResult:
            raw = '{"key": "value"}'
        result = extract_crewai_output(FakeResult())
        assert "key" in result

    def test_none_returns_empty_string(self):
        assert extract_crewai_output(None) == ""

    def test_thought_prefix_stripped(self):
        class FakeResult:
            raw = "I now can give a great answer\n{\"answer\": \"yes\"}"
        result = extract_crewai_output(FakeResult())
        assert "I now can give" not in result

    def test_tasks_output_attribute_preferred(self):
        class FakeTaskOutput:
            raw = '{"from": "tasks_output"}'
        class FakeResult:
            tasks_output = [FakeTaskOutput()]
        result = extract_crewai_output(FakeResult())
        assert "tasks_output" in result

    def test_pure_thought_returns_empty_string(self):
        class FakeResult:
            raw = "I can give a great answer to this question"
        result = extract_crewai_output(FakeResult())
        assert result == ""

    def test_dict_result_via_dict_scan(self):
        class FakeResult:
            pass
        fake = FakeResult()
        fake.__dict__["output"] = '{"schema": "found"}'
        result = extract_crewai_output(fake)
        assert "schema" in result


# ---------------------------------------------------------------------------
# create_error_response
# ---------------------------------------------------------------------------

class TestCreateErrorResponse:

    def test_basic_error_response(self):
        resp = create_error_response("Something failed", "Try again")
        assert resp["error"] == "Something failed"
        assert resp["solution"] == "Try again"

    def test_extra_kwargs_included(self):
        resp = create_error_response("err", "fix", code=500, retryable=True)
        assert resp["code"] == 500
        assert resp["retryable"] is True

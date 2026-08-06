"""
Unit tests for agents/requirements_agent.py

Uses mocked HTTP calls so no real API key is needed.
"""
import pytest
from unittest.mock import patch, MagicMock
import json

from agents.requirements_agent import RequirementsAnalystAgent, _anthropic_text_from_messages_api


# ---------------------------------------------------------------------------
# Helper: build a mock requests.Response
# ---------------------------------------------------------------------------

def _mock_response(payload: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = json.dumps(payload)
    return resp


def _api_payload(text: str) -> dict:
    """Wrap text in the Anthropic Messages API response envelope."""
    return {"content": [{"type": "text", "text": text}]}


VALID_REQUIREMENTS_JSON = json.dumps({
    "project_name": "Flower Delivery App",
    "features": [
        {
            "id": "F1",
            "name": "Browse Flowers",
            "description": "Customers browse the catalogue.",
            "priority": "High",
            "user_stories": ["As a customer, I can browse flowers."],
            "acceptance_criteria": ["Given the app is open, When I tap Browse, Then flowers appear."],
        }
    ],
    "technical_requirements": {"platform": ["Web"], "technologies": ["React"], "database": "PostgreSQL"},
    "user_roles": ["Customer"],
})


# ---------------------------------------------------------------------------
# _anthropic_text_from_messages_api
# ---------------------------------------------------------------------------

class TestAnthropicTextFromMessagesApi:

    def test_extracts_text_blocks(self):
        data = {"content": [{"type": "text", "text": "Hello"}, {"type": "tool_use", "id": "xyz"}]}
        assert _anthropic_text_from_messages_api(data) == "Hello"

    def test_multiple_text_blocks_joined(self):
        data = {"content": [{"type": "text", "text": "Part 1"}, {"type": "text", "text": "Part 2"}]}
        result = _anthropic_text_from_messages_api(data)
        assert "Part 1" in result and "Part 2" in result

    def test_empty_content_returns_empty_string(self):
        assert _anthropic_text_from_messages_api({"content": []}) == ""

    def test_missing_content_key(self):
        assert _anthropic_text_from_messages_api({}) == ""


# ---------------------------------------------------------------------------
# RequirementsAnalystAgent
# ---------------------------------------------------------------------------

class TestRequirementsAnalystAgent:

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("agents.requirements_agent.CREWAI_AVAILABLE", False)
    def test_init_without_crewai(self):
        agent = RequirementsAnalystAgent()
        assert agent.api_key == "test-key"
        assert agent.agent is None
        assert agent.crew is None

    @patch.dict("os.environ", {}, clear=True)
    @patch("agents.requirements_agent.CREWAI_AVAILABLE", False)
    def test_analyze_without_api_key_returns_error(self):
        agent = RequirementsAnalystAgent()
        result = agent.analyze_requirements("build me an app")
        assert "error" in result
        assert "api" in result["error"].lower() or "key" in result["error"].lower()

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("agents.requirements_agent.CREWAI_AVAILABLE", False)
    @patch("agents.requirements_agent.requests.post")
    def test_analyze_returns_parsed_json_on_success(self, mock_post):
        mock_post.return_value = _mock_response(_api_payload(VALID_REQUIREMENTS_JSON))
        agent = RequirementsAnalystAgent()
        result = agent.analyze_requirements("I want a flower delivery app")
        assert "error" not in result
        assert result["project_name"] == "Flower Delivery App"
        assert len(result["features"]) >= 1

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("agents.requirements_agent.CREWAI_AVAILABLE", False)
    @patch("agents.requirements_agent.requests.post")
    def test_analyze_adds_model_used_and_timestamp(self, mock_post):
        mock_post.return_value = _mock_response(_api_payload(VALID_REQUIREMENTS_JSON))
        agent = RequirementsAnalystAgent()
        result = agent.analyze_requirements("flower app")
        assert "model_used" in result
        assert "timestamp" in result

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("agents.requirements_agent.CREWAI_AVAILABLE", False)
    @patch("agents.requirements_agent.requests.post")
    def test_analyze_returns_error_on_401(self, mock_post):
        mock_post.return_value = _mock_response({"error": "Unauthorized"}, status_code=401)
        agent = RequirementsAnalystAgent()
        result = agent.analyze_requirements("flower app")
        assert "error" in result
        assert "auth" in result["error"].lower() or "401" in result.get("error", "").lower()

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("agents.requirements_agent.CREWAI_AVAILABLE", False)
    @patch("agents.requirements_agent.requests.post")
    def test_analyze_returns_error_on_500(self, mock_post):
        mock_post.return_value = _mock_response({"error": "Server Error"}, status_code=500)
        agent = RequirementsAnalystAgent()
        result = agent.analyze_requirements("flower app")
        assert "error" in result

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("agents.requirements_agent.CREWAI_AVAILABLE", False)
    @patch("agents.requirements_agent.requests.post")
    def test_analyze_returns_error_when_model_returns_prose(self, mock_post):
        # Model responds successfully but with prose instead of JSON
        mock_post.return_value = _mock_response(_api_payload("Sure! What kind of app would you like?"))
        agent = RequirementsAnalystAgent()
        result = agent.analyze_requirements("flower app")
        assert "error" in result

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("agents.requirements_agent.CREWAI_AVAILABLE", False)
    @patch("agents.requirements_agent.requests.post")
    def test_output_contains_required_keys(self, mock_post):
        mock_post.return_value = _mock_response(_api_payload(VALID_REQUIREMENTS_JSON))
        agent = RequirementsAnalystAgent()
        result = agent.analyze_requirements("flower app")
        for key in ("project_name", "features", "technical_requirements", "user_roles"):
            assert key in result, f"Missing required key: {key}"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("agents.requirements_agent.CREWAI_AVAILABLE", False)
    @patch("agents.requirements_agent.requests.post")
    def test_features_have_required_fields(self, mock_post):
        mock_post.return_value = _mock_response(_api_payload(VALID_REQUIREMENTS_JSON))
        agent = RequirementsAnalystAgent()
        result = agent.analyze_requirements("flower app")
        for feature in result.get("features", []):
            for field in ("id", "name", "description", "priority"):
                assert field in feature, f"Feature missing field: {field}"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"})
    @patch("agents.requirements_agent.CREWAI_AVAILABLE", False)
    @patch("agents.requirements_agent.requests.post")
    def test_timeout_returns_error(self, mock_post):
        import requests as req
        mock_post.side_effect = req.exceptions.Timeout()
        agent = RequirementsAnalystAgent()
        result = agent.analyze_requirements("flower app")
        assert "error" in result

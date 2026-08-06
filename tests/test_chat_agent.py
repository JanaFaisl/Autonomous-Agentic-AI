"""
Unit tests for agents/chat_agent.py

Focuses on prompt-adherence rules critical to the paper's evaluation:
- One MCQ question per turn
- Options appear only inside [ ... ] brackets
- Options are pipe-separated
- Last option is always "Other (please specify)"
- Options are NOT leaked into the question text
"""
import re
import pytest
from unittest.mock import patch, MagicMock

from agents.chat_agent import PROFESSIONAL_SYSTEM_PROMPT, call_chat_assistant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_options_block(text: str):
    """Return the content inside the first [ ... ] block, or None."""
    match = re.search(r"\[([^\]]+)\]", text)
    return match.group(1) if match else None


def _extract_question(text: str):
    """Return everything before the first [ line."""
    lines = text.strip().split("\n")
    question_lines = []
    for line in lines:
        if line.strip().startswith("["):
            break
        question_lines.append(line)
    return "\n".join(question_lines).strip()


MESSAGES_FLOWER = [
    {"role": "user", "content": "I want to build a flower delivery app."}
]

MESSAGES_WITH_HISTORY = [
    {"role": "user", "content": "I want to build a flower delivery app."},
    {"role": "assistant", "content": "Nice — a flower app! 🌷 Who are you mainly building it for?\n[ Individual customers | Florists | Event planners | Other (please specify) ]"},
    {"role": "user", "content": "Individual customers"},
]

VALID_MCQ_RESPONSE = (
    "Nice — a flower app! 🌷 Who are you mainly building it for?\n"
    "[ Individual customers | Florists | Event planners | Other (please specify) ]"
)

VALID_MCQ_RESPONSE_STEP2 = (
    "Got it — individual customers. What's the most important sign it's working?\n"
    "[ Sales revenue | Active buyers | Repeat purchases | Other (please specify) ]"
)


# ---------------------------------------------------------------------------
# PROFESSIONAL_SYSTEM_PROMPT content checks
# ---------------------------------------------------------------------------

class TestSystemPrompt:

    def test_prompt_defines_discovery_roadmap(self):
        assert "DISCOVERY ROADMAP" in PROFESSIONAL_SYSTEM_PROMPT

    def test_prompt_defines_six_steps(self):
        steps = ["PROBLEM", "AUDIENCE", "SUCCESS", "SCOPE", "CONSTRAINT", "SUMMARY"]
        for step in steps:
            assert step in PROFESSIONAL_SYSTEM_PROMPT, f"Step missing from prompt: {step}"

    def test_prompt_forbids_inline_options(self):
        assert "NEVER list" in PROFESSIONAL_SYSTEM_PROMPT or "NEVER" in PROFESSIONAL_SYSTEM_PROMPT

    def test_prompt_requires_bracket_format(self):
        assert "[" in PROFESSIONAL_SYSTEM_PROMPT and "|" in PROFESSIONAL_SYSTEM_PROMPT

    def test_prompt_requires_other_option(self):
        assert "Other (please specify)" in PROFESSIONAL_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# MCQ format validation (applied to any agent response)
# ---------------------------------------------------------------------------

class TestMCQFormatCompliance:
    """Validate that a model response follows the required MCQ format rules."""

    @pytest.mark.parametrize("response", [VALID_MCQ_RESPONSE, VALID_MCQ_RESPONSE_STEP2])
    def test_response_contains_options_block(self, response):
        assert _extract_options_block(response) is not None, "Response must contain [ ... ] options block"

    @pytest.mark.parametrize("response", [VALID_MCQ_RESPONSE, VALID_MCQ_RESPONSE_STEP2])
    def test_options_are_pipe_separated(self, response):
        block = _extract_options_block(response)
        assert block is not None
        options = [o.strip() for o in block.split("|")]
        assert len(options) >= 2, "Must have at least 2 options"

    @pytest.mark.parametrize("response", [VALID_MCQ_RESPONSE, VALID_MCQ_RESPONSE_STEP2])
    def test_last_option_is_other(self, response):
        block = _extract_options_block(response)
        assert block is not None
        options = [o.strip() for o in block.split("|")]
        assert "other" in options[-1].lower(), "Last option must be 'Other (please specify)'"

    @pytest.mark.parametrize("response", [VALID_MCQ_RESPONSE, VALID_MCQ_RESPONSE_STEP2])
    def test_question_does_not_mention_options(self, response):
        block = _extract_options_block(response)
        assert block is not None
        options = [o.strip().lower() for o in block.split("|") if "other" not in o.lower()]
        question = _extract_question(response).lower()
        for opt in options:
            # The option text (stripped) should not appear verbatim in the question sentence
            assert opt not in question, (
                f"Option '{opt}' leaked into the question text. "
                "Options must only appear inside [ ... ]."
            )

    @pytest.mark.parametrize("response", [VALID_MCQ_RESPONSE, VALID_MCQ_RESPONSE_STEP2])
    def test_options_count_is_three_to_four(self, response):
        block = _extract_options_block(response)
        assert block is not None
        options = [o.strip() for o in block.split("|")]
        assert 3 <= len(options) <= 4, f"Expected 3–4 options, got {len(options)}"

    @pytest.mark.parametrize("response", [VALID_MCQ_RESPONSE, VALID_MCQ_RESPONSE_STEP2])
    def test_options_block_on_new_line(self, response):
        lines = response.strip().split("\n")
        bracket_lines = [l for l in lines if l.strip().startswith("[")]
        assert len(bracket_lines) >= 1, "Options block must be on its own line starting with ["

    def test_invalid_response_without_brackets_fails_check(self):
        bad_response = "What kind of app? A. Delivery B. Marketplace C. Other"
        assert _extract_options_block(bad_response) is None

    def test_response_with_inline_options_fails_question_check(self):
        """Simulate a bad model response that leaks option text into the question."""
        bad_response = (
            "Are you targeting individual customers or florists or event planners?\n"
            "[ Individual customers | Florists | Event planners | Other (please specify) ]"
        )
        block = _extract_options_block(bad_response)
        options = [o.strip().lower() for o in block.split("|") if "other" not in o.lower()]
        question = _extract_question(bad_response).lower()
        leaked = [opt for opt in options if opt in question]
        assert len(leaked) > 0, "This test expects leaked options to be detected"


# ---------------------------------------------------------------------------
# call_chat_assistant (mocked Anthropic fallback)
# ---------------------------------------------------------------------------

class TestCallChatAssistant:

    @patch("agents.chat_agent.CREWAI_AVAILABLE", False)
    @patch("agents.chat_agent.st")
    def test_fallback_returns_response_on_success(self, mock_st):
        mock_st.spinner = MagicMock(return_value=MagicMock(
            __enter__=lambda s, *a: None, __exit__=lambda s, *a: None
        ))
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text=VALID_MCQ_RESPONSE)]
        )
        fake_agent = MagicMock()
        fake_agent.api_key = "test-key"
        # Patch the Anthropic class inside the namespace where chat_agent imports it
        with patch.dict("sys.modules", {"anthropic": MagicMock(Anthropic=MagicMock(return_value=mock_client))}):
            result = call_chat_assistant(fake_agent, MESSAGES_FLOWER)
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("agents.chat_agent.CREWAI_AVAILABLE", False)
    @patch("agents.chat_agent.st")
    def test_fallback_returns_error_on_api_exception(self, mock_st):
        mock_st.spinner = MagicMock(return_value=MagicMock(
            __enter__=lambda s, *a: None, __exit__=lambda s, *a: None
        ))
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("Network error")
        fake_agent = MagicMock()
        fake_agent.api_key = "test-key"
        with patch("anthropic.Anthropic", return_value=mock_client):
            result = call_chat_assistant(fake_agent, MESSAGES_FLOWER)
        assert "error" in result.lower() or "⚠️" in result

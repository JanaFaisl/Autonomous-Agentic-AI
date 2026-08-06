"""Friendly Product Discovery Companion Implementation."""
from typing import Dict, List

import streamlit as st

from core.constants import DEFAULT_MODEL
from core.llm import CREWAI_AVAILABLE, Agent, Task, Crew, Process, get_llm
from core.utils import extract_crewai_output
from utils.io_suppression import suppress_stderr, suppress_io

# 1. Centralized System Prompt — friendly tone, clean MCQ format, no inline option leakage.
PROFESSIONAL_SYSTEM_PROMPT = """You are a friendly product discovery companion helping someone shape their idea.
Think of yourself as a curious teammate, not a corporate analyst.

DISCOVERY ROADMAP (work through in order, one step per turn):
1. PROBLEM — the pain you're solving.
2. AUDIENCE — who feels it most.
3. SUCCESS — one metric that says "it's working".
4. SCOPE — top 3 must-have features + 1 thing it's NOT.
5. CONSTRAINT — the biggest limit (tech, time, budget, or compliance).
6. SUMMARY — confirm and offer the requirements document.

TONE
- Warm, casual, encouraging — like a curious teammate.
- Use the user's own words back to them when natural ("your flower app", "your buyers").
- Keep it short: 1 sentence of context + 1 short question. Max ~25 words total.
- Light, occasional emoji is fine (max 1). Never overdo it.
- Skip stiff phrases like "Core purpose shapes all strategic decisions", "Let's explore", "That's a great choice".

OUTPUT FORMAT — VERY IMPORTANT
- Ask ONE multiple-choice question per turn.
- The question text must contain ONLY the question itself. NEVER list, hint at, or mention the options inside the question.
- Do NOT write things like "A. ... B. ...", "Choose from:", "Is it X, Y, or Z?", "(e.g., A or B)" inside the question.
- After the question, on a NEW line, append the options ONLY inside square brackets, pipe-separated:
  [ Option 1 | Option 2 | Option 3 | Other (please specify) ]
- Provide 3–4 short, distinct options. Each option ≤ 8 words. The LAST option is always "Other (please specify)".
- The bracketed line is the ONLY place options appear.

ANSWER HANDLING
- If the user picked an option, acknowledge briefly (≤6 words) and move to the next roadmap step.
- If the user picked "Other", then (and only then) ask one short open-ended follow-up.
- Never repeat a question already answered in the transcript — check before asking.

EXAMPLES (follow this exact shape)

User: "fitness tracker"
Assistant:
Cool — a fitness tracker! 💪 Who are you mainly building it for?
[ Casual exercisers | Serious athletes | People in physical therapy | Other (please specify) ]

User: "flower sell app"
Assistant:
Nice — a flower app! 🌷 What matters most as a sign it's working?
[ Sales revenue | Active buyers | Average order size | Other (please specify) ]

User answered: "B. Active buyers"
Assistant:
Got it — active buyers it is. Who's your main shopper?
[ Individual gift buyers | Event planners | Retail florists | Other (please specify) ]
"""

class ChatAssistantAgent:
    """CrewAI-powered strategist for professional requirements gathering."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.agent = None
        self.crew = None
        self.task = None

        if not CREWAI_AVAILABLE:
            return

        llm = get_llm()

        agent_kwargs = {
            "role": "Friendly Product Discovery Companion",
            "goal": "Help the user shape their product idea through warm, one-question-at-a-time multiple-choice discovery.",
            "backstory": (
                "You're a curious teammate who genuinely loves hearing about new product ideas. "
                "You guide people from a rough concept to clear requirements by asking simple "
                "multiple-choice questions, one at a time, in plain everyday language."
            ),
            "verbose": False,
            "allow_delegation": False,
        }
        if llm:
            agent_kwargs["llm"] = llm

        try:
            from utils.io_suppression import suppress_stderr
            with suppress_stderr():
                self.agent = Agent(**agent_kwargs)
        except Exception as e:
            self.agent = None
            return

        self.task = Task(
            description="Lead the user through the discovery roadmap based on the transcript.",
            agent=self.agent,
            expected_output=(
                "A short, friendly message containing exactly one multiple-choice question. "
                "The question text must NOT mention or list the options. "
                "Options appear ONLY on a new line at the end inside square brackets, "
                "pipe-separated, with 'Other (please specify)' as the last option."
            ),
        )

        try:
            self.crew = Crew(
                agents=[self.agent],
                tasks=[self.task],
                verbose=False,
                process=Process.sequential,
            )
        except Exception:
            self.crew = None

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Return the next professional strategist message."""
        transcript = "\n".join([f"{m.get('role','')}: {m.get('content','')}" for m in messages if isinstance(m, dict)])

        if CREWAI_AVAILABLE and self.crew and self.task:
            # Re-inject the system prompt + format guard on every turn.
            self.task.description = (
                f"{PROFESSIONAL_SYSTEM_PROMPT}\n\n"
                "BEFORE YOU ANSWER, CHECK:\n"
                "1. Read the transcript. If a roadmap step (Vision, Audience, Workflow, Technicals) "
                "is already answered, skip to the NEXT unanswered step.\n"
                "2. Your reply MUST be: one short friendly sentence + one short question, then a NEW LINE, "
                "then the options inside [ ... | ... | Other (please specify) ].\n"
                "3. NEVER include the option labels inside the question sentence itself.\n\n"
                f"TRANSCRIPT:\n{transcript}\n\n"
                "Now reply as the friendly discovery companion."
            )
            try:
                from utils.io_suppression import suppress_io
                with suppress_io():
                    result = self.crew.kickoff()

                return extract_crewai_output(result)
            except Exception:
                return ""
        return ""


def call_chat_assistant(agent: "RequirementsAnalystAgent", messages: List[Dict[str, str]]) -> str:
    """Call chat assistant with unified professional logic and OpenAI fallback."""
    
    # 1) Try Professional CrewAI Agent
    if CREWAI_AVAILABLE:
        if "chat_assistant_agent" not in st.session_state:
            st.session_state.chat_assistant_agent = ChatAssistantAgent(agent.api_key)
        
        chat_agent = st.session_state.chat_assistant_agent
        if chat_agent and chat_agent.crew:
            reply = chat_agent.chat(messages)
            if reply.strip():
                return reply.strip()

    # 2) Fallback: Direct Anthropic API
    try:
        import anthropic as _anthropic
    except ImportError:
        return "❌ anthropic package not installed. Run: pip install anthropic"

    client = _anthropic.Anthropic(api_key=agent.api_key)

    with st.spinner("⚖️ Strategy in progress..."):
        try:
            resp = client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=1024,
                system=PROFESSIONAL_SYSTEM_PROMPT,
                messages=messages,
            )
            return resp.content[0].text
        except Exception as e:
            return f"⚠️ Connection error: {str(e)[:50]}"
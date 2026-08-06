"""CrewAI setup and LLM factory. Import this instead of ui.main_ui for agent use."""
import io
import os
import sys
import warnings

# Disable CrewAI/LangChain telemetry before import
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "1")
os.environ.setdefault("CREWAI_TESTING", "true")
os.environ.setdefault("DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_VERBOSE", "false")
os.environ.setdefault("CREWAI_VERBOSE", "false")
os.environ.setdefault("LANGCHAIN_ENDPOINT", "")
os.environ.setdefault("LANGCHAIN_API_KEY", "")
os.environ.setdefault("LANGCHAIN_PROJECT", "")
os.environ.setdefault("CREWAI_PLUS_API_KEY", "")
os.environ.setdefault("CREWAI_PLUS_ENABLED", "false")

warnings.filterwarnings("ignore", category=RuntimeWarning, module="crewai")
warnings.filterwarnings("ignore", message=".*signal.*", category=RuntimeWarning)

try:
    _old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        from crewai import Agent, LLM, Task, Crew
        from crewai.process import Process

        try:
            from crewai.events.listeners.tracing.utils import set_suppress_tracing_messages
            set_suppress_tracing_messages(True)
        except Exception:
            pass

        CREWAI_AVAILABLE = True
    except Exception:
        CREWAI_AVAILABLE = False
        Agent = LLM = Task = Crew = Process = None
    finally:
        sys.stderr = _old_stderr
except Exception:
    CREWAI_AVAILABLE = False
    Agent = LLM = Task = Crew = Process = None


def get_llm():
    """Return a CrewAI LLM instance backed by Anthropic, or None if unavailable."""
    if not CREWAI_AVAILABLE or LLM is None:
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    from core.constants import DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS

    model_str = DEFAULT_MODEL if DEFAULT_MODEL.startswith("anthropic/") else f"anthropic/{DEFAULT_MODEL}"
    try:
        return LLM(
            model=model_str,
            temperature=DEFAULT_TEMPERATURE,
            api_key=api_key,
            # Design generation alone commonly takes ~400s at 8192 output tokens (see README);
            # DEFAULT_MAX_TOKENS is now ~2x that, so a shorter timeout here would abort the call
            # mid-generation and look like a network error.
            timeout=600,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
    except Exception:
        return None

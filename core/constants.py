import os

# Model configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-5")
DEFAULT_TEMPERATURE = 0.2
# CrewAI's native Anthropic provider defaults max_tokens to 4096 if unset, which truncates
# verbose agents (e.g. Design: 4-6 screens x 8-12 components) mid-JSON and fails Pydantic validation.
# Truncation point scales linearly with this value (confirmed: 4096->line 304, 8192->line 582),
# so it must comfortably exceed the largest schema (Design). The configured model supports far
# more than this (check via GET /v1/models/{model} -> max_tokens), so headroom here is cheap.
DEFAULT_MAX_TOKENS = 16000
MAX_RETRIES = 2
MAX_EXECUTION_TIME = 180
# Design routinely runs ~400s at 8192 output tokens (see README); DEFAULT_MAX_TOKENS above is
# now ~2x that, so give it a matching execution budget instead of the shared, lighter-agent one.
DESIGN_MAX_EXECUTION_TIME = 650
MAX_ITERATIONS = 1

# Retry configuration
RETRY_DELAY = 0.5
EXPONENTIAL_BACKOFF_BASE = 1.5

# Error messages
ERROR_NO_API_KEY = "Anthropic API key not found"
ERROR_CREWAI_NOT_INSTALLED = "CrewAI not installed"
ERROR_AGENT_NOT_INITIALIZED = "CrewAI agent not initialized"
ERROR_EMPTY_RESPONSE = "Empty response from CrewAI agent"
ERROR_PARSE_JSON = "Could not parse JSON"

# Solutions
SOLUTION_ADD_API_KEY = "Add ANTHROPIC_API_KEY to your .env file"
SOLUTION_INSTALL_CREWAI = "Run: pip install crewai"
SOLUTION_CHECK_CONFIG = "Check your ANTHROPIC_API_KEY configuration"
SOLUTION_RETRY = "Please try again or check the agent configuration."

# UI messages
MSG_ANALYZING = "Analyzing requirements…"
MSG_GENERATING_DESIGN = "🎨 Generating UI/UX design…"
MSG_RETRYING = "Retrying…"
MSG_CONNECTION_ISSUE = "⚠️ Connection issue detected. Retrying in {wait_time}s…"
MSG_AGENT_EMPTY = "⚠️ Agent returned empty/thought-only response. Retrying with more explicit instructions…"
MSG_PARSE_ERROR = "⚠️ Could not parse JSON from agent response. Retrying…"

# CrewAI thought patterns to strip from output
THOUGHT_PATTERNS = [
    r'^I\s+now\s+can\s+give',
    r'^I\s+can\s+give',
    r'^Now\s+I\s+can',
    r'^I\s+will\s+now',
    r'great\s+answer',
    r'let\s+me\s+think',
    r'^I\s+think',
    r'^Let\s+me',
    r'^I\s+now',
    r'can\s+give\s+a\s+great',
]

THOUGHT_INDICATORS = [
    'i now can give',
    'i can give',
    'great answer',
    'now i can',
    'let me think',
]

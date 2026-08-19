import html as _html
import os
import re
import json
import time
import warnings
import sys
import io
from typing import Dict, List, Any, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from utils.project_export import build_files_zip
from utils.code_scaffold import build_scaffold_files
from core.utils import get_component_icon_glyph

try:
    from db import (
        init_db, save_input, save_click, save_navigation,
        get_screen_state, get_all_session_state, clear_session,
        save_artifact, get_latest_artifact
    )
    # Initialize database on import
    init_db()
except ImportError:
    # Fallback if db.py doesn't exist yet
    def init_db(): pass
    def save_input(*args, **kwargs): pass
    def save_click(*args, **kwargs): pass
    def save_navigation(*args, **kwargs): pass
    def get_screen_state(*args, **kwargs): return {}
    def get_all_session_state(*args, **kwargs): return {}
    def clear_session(*args, **kwargs): pass
    def save_artifact(*args, **kwargs): return 0
    def get_latest_artifact(*args, **kwargs): return None

# ----------------------------
# ENVIRONMENT SETUP
# ----------------------------
load_dotenv(override=True)

# Disable CrewAI telemetry and tracing to avoid signal handler errors and trace output in Streamlit
os.environ["CREWAI_DISABLE_TELEMETRY"] = "1"
# Skip first-run trace prompts and "tracing preference saved" panels (non-interactive / Streamlit)
os.environ["CREWAI_TESTING"] = "true"
os.environ["DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
# Disable interactive prompts and execution traces
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_VERBOSE"] = "false"
os.environ["CREWAI_VERBOSE"] = "false"
# Disable LangSmith and CrewAI Plus tracing
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""
os.environ["CREWAI_PLUS_API_KEY"] = ""
os.environ["CREWAI_PLUS_ENABLED"] = "false"

# Suppress warnings from CrewAI telemetry
warnings.filterwarnings("ignore", category=RuntimeWarning, module="crewai")
warnings.filterwarnings("ignore", message=".*signal.*", category=RuntimeWarning)

try:
    import anthropic as _anthropic_module
except ImportError:
    _anthropic_module = None

# Keep OpenAI import for backward compatibility if any code still references it
OpenAI = None

try:
    # Suppress telemetry signal handler errors during import
    # CrewAI tries to register signal handlers which fails in Streamlit's thread context
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    try:
        from crewai import Agent, LLM, Task, Crew
        from crewai.process import Process

        try:
            from crewai.events.listeners.tracing.utils import (
                set_suppress_tracing_messages,
            )

            set_suppress_tracing_messages(True)
        except Exception:
            pass

        CREWAI_AVAILABLE = True
    except Exception:
        # If import fails, we'll handle it below
        CREWAI_AVAILABLE = False
    finally:
        # Restore stderr
        sys.stderr = old_stderr
        
    if not CREWAI_AVAILABLE:
        Agent = None
        LLM = None
        Task = None
        Crew = None
        Process = None
        
except ImportError:
    CREWAI_AVAILABLE = False
    Agent = None
    LLM = None
    Task = None
    Crew = None
    Process = None


# ----------------------------
# CONSTANTS
# ----------------------------
# Model Configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "claude-3-5-sonnet-latest")
DEFAULT_TEMPERATURE = 0.2  # Lower temperature for faster, more deterministic responses
MAX_RETRIES = 2  # Reduced retries for faster failure detection
MAX_EXECUTION_TIME = 40  # 40 seconds (reduced for faster response)
MAX_ITERATIONS = 1  # Single iteration for faster response

# Retry Configuration
RETRY_DELAY = 0.5  # seconds (reduced delay)
EXPONENTIAL_BACKOFF_BASE = 1.5  # seconds (reduced backoff)

# Error Messages
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

# UI Messages
MSG_ANALYZING = "Analyzing requirements…"
MSG_GENERATING_DESIGN = "🎨 Generating UI/UX design…"
MSG_RETRYING = "Retrying…"
MSG_CONNECTION_ISSUE = "⚠️ Connection issue detected. Retrying in {wait_time}s…"
MSG_AGENT_EMPTY = "⚠️ Agent returned empty/thought-only response. Retrying with more explicit instructions…"
MSG_PARSE_ERROR = "⚠️ Could not parse JSON from agent response. Retrying…"

# Thought Patterns to Filter
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

# ----------------------------
# PYDANTIC MODELS FOR CREWAI OUTPUT
# ----------------------------
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
    Field = None

if PYDANTIC_AVAILABLE:
    # Requirements Output Model
    class DBColumnModel(BaseModel):
        name: str
        type: str
        pk: bool = False
        nullable: bool = True
        unique: bool = False
        default: Optional[str] = None
        notes: Optional[str] = None

    class DBRelationshipModel(BaseModel):
        type: str  # one-to-many, many-to-one, many-to-many, one-to-one
        to_table: str
        fk: Optional[str] = None
        ref: Optional[str] = None
        notes: Optional[str] = None

    class DBTableModel(BaseModel):
        name: str
        purpose: str
        columns: List[DBColumnModel] = []
        indexes: List[str] = []
        relationships: List[DBRelationshipModel] = []

    class DatabaseSchemaModel(BaseModel):
        tables: List[DBTableModel] = []
        assumptions: List[str] = []

    class FeatureModel(BaseModel):
        id: str
        name: str
        description: str
        priority: str  # High/Medium/Low
        user_stories: List[str] = []
        acceptance_criteria: List[str] = []

    class TechnicalRequirementsModel(BaseModel):
        platform: List[str] = []
        technologies: List[str] = []
        database: Optional[str] = None

    class RequirementsOutputModel(BaseModel):
        project_name: str
        features: List[FeatureModel] = []
        technical_requirements: TechnicalRequirementsModel
        user_roles: List[str] = []
        database_schema: Optional[DatabaseSchemaModel] = None

    # Design Output Models
    class ColorSchemeModel(BaseModel):
        primary: str
        secondary: str
        accent: str
        surface: str
        background: str
        text_primary: str
        error: str
        success: str
        warning: str

    class TypographySizesModel(BaseModel):
        small: Optional[str] = None
        medium: Optional[str] = None
        large: Optional[str] = None
        xlarge: Optional[str] = None

    class TypographyWeightsModel(BaseModel):
        normal: Optional[str] = None
        bold: Optional[str] = None

    class TypographyModel(BaseModel):
        font_family: Optional[str] = None
        heading_font: Optional[str] = None
        body_font: Optional[str] = None
        sizes: Optional[TypographySizesModel] = None
        weights: Optional[TypographyWeightsModel] = None

    class NavigationModel(BaseModel):
        type: str  # top/bottom/side
        items: List[str] = []

    class ComponentModel(BaseModel):
        name: str
        type: str
        position: Optional[str] = None
        size: Optional[str] = None
        styling: Optional[str] = None
        interaction: Optional[str] = None
        icon: Optional[str] = None
        has_image: bool = False
        image_type: Optional[str] = None

    class ScreenModel(BaseModel):
        name: str
        purpose: str
        key_components: List[ComponentModel] = []
        user_flow: Optional[str] = None

    class DesignOutputModel(BaseModel):
        design_overview: Optional[str] = None
        color_scheme: ColorSchemeModel
        typography: TypographyModel
        navigation: NavigationModel
        screens: List[ScreenModel] = []
        ui_components: List[ComponentModel] = []
        responsive_design: Optional[str] = None
        accessibility: Optional[str] = None
        animations: Optional[str] = None
        icons: Optional[str] = None

    # TSPi Agent Output Models
    class CycleTaskModel(BaseModel):
        id: str
        title: str
        assigned_agent: Optional[str] = None
        priority: Optional[str] = None

    class CyclePlanOutputModel(BaseModel):
        plan_name: str
        tasks: List[CycleTaskModel] = []
        risks: List[str] = []

    class QualityIssueModel(BaseModel):
        severity: Optional[str] = None
        item: Optional[str] = None
        message: Optional[str] = None

    class QualityReportOutputModel(BaseModel):
        gate_decision: str  # PASS or FAIL
        artifact_reviewed: List[str] = []
        checklist: Optional[Dict[str, bool]] = None
        issues: List[QualityIssueModel] = []
        required_fixes: List[str] = []
        recommendations: List[str] = []

    class SupportGovernanceOutputModel(BaseModel):
        app_documentation: Optional[str] = None
        baseline_artifacts: List[str] = []
        glossary: Optional[Dict[str, str]] = None

    class PipelinePlanOutputModel(BaseModel):
        """Execution order of pipeline steps (orchestrator output)."""
        steps: List[str] = []  # e.g. ["design", "planning", "quality", "support"]
        reason: Optional[str] = None
else:
    # Placeholders when Pydantic is not available
    RequirementsOutputModel = None
    DesignOutputModel = None
    CyclePlanOutputModel = None
    QualityReportOutputModel = None
    SupportGovernanceOutputModel = None
    PipelinePlanOutputModel = None

# ----------------------------
# UTILITY FUNCTIONS
# ----------------------------
def suppress_stderr(func):
    """Context manager to suppress stderr during function execution."""
    def wrapper(*args, **kwargs):
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            return func(*args, **kwargs)
        finally:
            sys.stderr = old_stderr
    return wrapper


def suppress_stdin_stderr(func):
    """Context manager to suppress stdin/stderr during function execution."""
    def wrapper(*args, **kwargs):
        old_stdin = sys.stdin
        old_stderr = sys.stderr
        sys.stdin = io.StringIO("n\n")  # Auto-answer "no" to prompts
        sys.stderr = io.StringIO()
        try:
            return func(*args, **kwargs)
        finally:
            sys.stdin = old_stdin
            sys.stderr = old_stderr
    return wrapper


def create_error_response(error: str, solution: str, **kwargs) -> Dict[str, Any]:
    """Create a standardized error response dictionary."""
    response = {
        "error": error,
        "solution": solution,
    }
    response.update(kwargs)
    return response


def extract_hex(color_str: Optional[str], fallback: str) -> str:
    """Extract hex color code from string."""
    if not color_str:
        return fallback
    hex_match = re.search(r"#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})", color_str)
    if hex_match:
        return hex_match.group(0)
    parts = str(color_str).strip().split()
    if parts and parts[0].startswith("#"):
        return parts[0]
    return fallback


def safe_str(x: Any, default: str = "") -> str:
    """Safely convert value to string."""
    return x if isinstance(x, str) and x.strip() else default


def _h(s: str) -> str:
    """Escape a string for safe embedding in HTML text or attribute values."""
    return _html.escape(str(s), quote=True)


def _j(s: str) -> str:
    """Escape a string for safe embedding in a JavaScript single-quoted string."""
    return str(s).replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")


def parse_choice_block(text: str) -> Optional[Tuple[str, List[str]]]:
    """Detect and parse multiple choice format: 'Question text [A|B|C|D]'."""
    if not text:
        return None
    m = re.search(r"(.+?)\s*\[([^\]]+)\]", text, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    q = m.group(1).strip()
    choices = [c.strip() for c in m.group(2).split("|") if c.strip()]
    if len(choices) < 2:
        return None
    return q, choices


def strip_choice_block(text: str) -> str:
    """Remove the trailing '[A | B | C]' marker so the chat bubble shows the question only.
    Keeps original text untouched if no choice block is present."""
    if not text:
        return text
    parsed = parse_choice_block(text)
    if parsed is None:
        return text
    q, _ = parsed
    return q

def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Robustly parse JSON from text, handling markdown code blocks and extra text."""
    try:
        if not text:
            return {
                "error": "Empty response",
                "raw_response": "",
            }
        
        text = text.strip()
        
        # Remove markdown code blocks
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*", "", text)
        
        # Try to find JSON object - look for opening brace
        json_start = text.find('{')
        if json_start == -1:
            return {
                "error": "No JSON object found in response",
                "raw_response": text[:1000],
            }
        
        # Find matching closing brace
        brace_count = 0
        json_end = -1
        for i in range(json_start, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end == -1:
            # Try to find the last closing brace as fallback
            json_end = text.rfind('}')
            if json_end > json_start:
                json_end += 1
            else:
                return {
                    "error": "Incomplete JSON object (missing closing brace)",
                    "raw_response": text[:1000],
                }
        
        candidate = text[json_start:json_end]
        
        # Fix common JSON issues
        candidate = re.sub(r",\s*}", "}", candidate)  # Trailing commas before }
        candidate = re.sub(r",\s*]", "]", candidate)  # Trailing commas before ]
        candidate = re.sub(r",\s*,", ",", candidate)  # Double commas
        
        # Try to parse
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

        return {
            "error": "Response JSON is not an object",
            "raw_response": text[:1000],
            "extracted_json": candidate[:500],
        }
    except json.JSONDecodeError as e:
        # Provide more helpful error message
        error_pos = getattr(e, 'pos', None)
        error_msg = str(e)
        if error_pos:
            start = max(0, error_pos - 50)
            end = min(len(text), error_pos + 50)
            context = text[start:end]
            error_msg = f"{error_msg} (near position {error_pos}: ...{context}...)"
        
        return {
            "error": f"JSON parsing error: {error_msg}",
            "raw_response": text[:1000],
            "json_error": str(e),
        }
    except Exception as e:
        return {
            "error": f"Could not parse JSON: {e}",
            "raw_response": text[:1000] if text else "",
        }


def extract_crewai_output(result) -> str:
    """
    Robustly extract output from CrewAI result object.
    Handles various result structures and returns the actual task output.
    """
    if not result:
        return ""
    
    content = None
    original_content = None
    
    # Method 1: Try tasks_output (most reliable in newer CrewAI versions)
    if hasattr(result, 'tasks_output'):
        tasks_output = result.tasks_output
        if isinstance(tasks_output, list) and len(tasks_output) > 0:
            task_output = tasks_output[0]
            # Priority: json_dict (Pydantic-validated), json, then raw/output/content
            # Note: json_dict/json are dicts, not strings, so we skip them here
            # They should be handled by the caller before calling this function
            for attr in ['raw', 'output', 'content', 'result', 'messages']:
                if hasattr(task_output, attr):
                    val = getattr(task_output, attr)
                    if isinstance(val, str) and val.strip():
                        content = val
                        original_content = val
                        break
                    elif isinstance(val, list) and len(val) > 0:
                        # If messages list, try to extract content from last message
                        if attr == 'messages':
                            for msg in reversed(val):
                                if hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content.strip():
                                    content = msg.content
                                    original_content = content
                                    break
                            if content:
                                break
            
            # If no direct attribute, check __dict__ deeply
            if not content and hasattr(task_output, '__dict__'):
                task_dict = task_output.__dict__
                # Look for any string attribute that might contain output
                json_candidates = []
                other_candidates = []
                
                for key, val in task_dict.items():
                    if isinstance(val, str) and val.strip() and len(val) > 10:
                        # Prefer attributes that likely contain JSON
                        if '{' in val or '[' in val:
                            json_candidates.append((key, val))
                        else:
                            other_candidates.append((key, val))
                    elif isinstance(val, (dict, list)):
                        # Recursively search nested structures
                        def search_nested(obj, path=""):
                            found = []
                            if isinstance(obj, dict):
                                for k, v in obj.items():
                                    if isinstance(v, str) and v.strip() and len(v) > 10:
                                        if '{' in v or '[' in v:
                                            found.append((f"{path}.{k}", v))
                                        else:
                                            found.append((f"{path}.{k}", v))
                                    elif isinstance(v, (dict, list)):
                                        found.extend(search_nested(v, f"{path}.{k}"))
                            elif isinstance(obj, list):
                                for i, item in enumerate(obj):
                                    if isinstance(item, str) and item.strip() and len(item) > 10:
                                        if '{' in item or '[' in item:
                                            found.append((f"{path}[{i}]", item))
                                    elif isinstance(item, (dict, list)):
                                        found.extend(search_nested(item, f"{path}[{i}]"))
                            return found
                        
                        nested_results = search_nested(val, key)
                        for path, nested_val in nested_results:
                            if '{' in nested_val or '[' in nested_val:
                                json_candidates.append((path, nested_val))
                            else:
                                other_candidates.append((path, nested_val))
                
                # Prefer JSON-like content
                if json_candidates:
                    content = json_candidates[0][1]
                    original_content = content
                elif other_candidates:
                    content = other_candidates[0][1]
                    original_content = content
                
                # Last resort: convert task_output to string
                if not content:
                    content = str(task_output)
                    original_content = content
    
    # Method 2: Try direct result attributes
    if not content:
        for attr in ['raw', 'output', 'content', 'result']:
            if hasattr(result, attr):
                val = getattr(result, attr)
                if isinstance(val, str) and val.strip():
                    content = val
                    original_content = val
                    break
                elif isinstance(val, list) and len(val) > 0:
                    # If it's a list, try to get string from first item
                    first_item = val[0]
                    if isinstance(first_item, str) and first_item.strip():
                        content = first_item
                        original_content = content
                        break
                    elif hasattr(first_item, 'raw'):
                        raw_val = getattr(first_item, 'raw', '')
                        if raw_val and isinstance(raw_val, str) and raw_val.strip():
                            content = raw_val
                            original_content = content
                            break
    
    # Method 3: Inspect __dict__ for any string values
    if not content and hasattr(result, '__dict__'):
        result_dict = result.__dict__
        # Look for string values, prioritizing those with JSON-like content
        json_candidates = []
        other_candidates = []
        
        def search_dict(d, path=""):
            for key, val in d.items():
                if isinstance(val, str) and val.strip() and len(val) > 10:
                    if '{' in val or '[' in val:
                        json_candidates.append((f"{path}.{key}", val))
                    else:
                        other_candidates.append((f"{path}.{key}", val))
                elif isinstance(val, list) and len(val) > 0:
                    for i, item in enumerate(val):
                        if isinstance(item, str) and item.strip() and len(item) > 10:
                            if '{' in item or '[' in item:
                                json_candidates.append((f"{path}.{key}[{i}]", item))
                            else:
                                other_candidates.append((f"{path}.{key}[{i}]", item))
                        elif isinstance(item, dict) and hasattr(item, '__dict__'):
                            search_dict(item.__dict__, f"{path}.{key}[{i}]")
                elif isinstance(val, dict) and hasattr(val, '__dict__'):
                    search_dict(val.__dict__, f"{path}.{key}")
        
        search_dict(result_dict)
        
        # Prefer JSON-like content
        if json_candidates:
            content = json_candidates[0][1]
            original_content = content
        elif other_candidates:
            content = other_candidates[0][1]
            original_content = content
    
    # Method 4: Last resort - string conversion
    if not content:
        content = str(result)
        original_content = content
    
    # Clean up the content
    if content:
        content = content.strip()
        original_content = original_content.strip() if original_content else content
        
        # Remove common CrewAI/LangChain prefixes
        content = re.sub(r'^Thought:\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        content = re.sub(r'^Action:\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        content = re.sub(r'^Observation:\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        content = re.sub(r'^Final Answer:\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        content = re.sub(r'^Reflection:\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        
        # Filter out pure thought/reflection lines (but keep if it's the only content)
        lines = content.split('\n')
        filtered_lines = []
        has_json = any('{' in line or '[' in line for line in lines)
        
        # Common thought patterns to filter
        thought_patterns = THOUGHT_PATTERNS
        
        for line in lines:
            line_stripped = line.strip()
            # Only filter thoughts if we have JSON content elsewhere
            if has_json:
                is_thought = False
                for pattern in thought_patterns:
                    if re.search(pattern, line_stripped, re.IGNORECASE):
                        is_thought = True
                        break
                if is_thought:
                    continue
            filtered_lines.append(line)
        
        content = '\n'.join(filtered_lines).strip()
        
        # Check if content is ONLY thoughts (no JSON)
        is_only_thought = False
        if content and not has_json:
            # Check if the entire content matches thought patterns
            content_lower = content.lower()
            if any(indicator in content_lower for indicator in THOUGHT_INDICATORS):
                is_only_thought = True
        
        # If we only have thoughts and no JSON, return empty to trigger retry
        if is_only_thought:
            return ""
        
        # If filtering removed everything but we had original content, 
        # and original doesn't have JSON, return empty to trigger retry
        if not content and original_content:
            # Check if original is just thoughts
            original_lower = original_content.lower()
            if any(indicator in original_lower for indicator in THOUGHT_INDICATORS):
                return ""  # Return empty to trigger retry with more explicit prompt
    
    return content if content else ""


# ----------------------------
# TAB NAVIGATION
# ----------------------------
def navigate_tabs_js(target_index: int) -> None:
    """JavaScript hack to programmatically switch Streamlit tabs."""
    js = f"""
    <script>
    (function () {{
      function isScrollable(el) {{
        if (!el) return false;
        const style = window.getComputedStyle(el);
        const oy = style.overflowY;
        return (oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight;
      }}

      function scrollEverythingToTop(doc) {{
        try {{
          const candidates = [
            doc.documentElement,
            doc.body,
            doc.querySelector('section.main'),
            doc.querySelector('div[data-testid="stAppViewContainer"]'),
            doc.querySelector('div[data-testid="stMain"]'),
            doc.querySelector('.main'),
            doc.querySelector('.block-container'),
          ].filter(Boolean);

          for (const el of candidates) {{
            try {{ el.scrollTop = 0; }} catch (e) {{}}
          }}

          const all = doc.querySelectorAll('*');
          for (const el of all) {{
            if (isScrollable(el)) {{
              try {{ el.scrollTop = 0; }} catch (e) {{}}
            }}
          }}

          const anchor = doc.querySelector('#top-anchor');
          if (anchor) {{
            try {{
              doc.defaultView.location.hash = 'top-anchor';
            }} catch(e) {{}}
            try {{
              anchor.scrollIntoView({{ behavior: 'auto', block: 'start', inline: 'nearest' }});
            }} catch(e) {{}}
          }}
        }} catch (e) {{}}
      }}

      function clickTab(doc) {{
        try {{
          // Try multiple selectors for Streamlit tabs
          let tabs = null;
          let clicked = false;
          
          // Method 1: Standard Streamlit tab selector
          tabs = doc.querySelectorAll('[data-baseweb="tab"]');
          if (tabs && tabs.length > {target_index}) {{
            const tab = tabs[{target_index}];
            if (tab && !tab.hasAttribute('aria-selected') || tab.getAttribute('aria-selected') !== 'true') {{
              tab.click();
              // Also dispatch events to ensure it works
              tab.dispatchEvent(new MouseEvent('click', {{ bubbles: true, cancelable: true }}));
              clicked = true;
            }}
          }}
          
          // Method 2: Button with tab role
          if (!clicked) {{
            tabs = doc.querySelectorAll('button[data-baseweb="tab"]');
            if (tabs && tabs.length > {target_index}) {{
              const tab = tabs[{target_index}];
              tab.click();
              tab.dispatchEvent(new MouseEvent('click', {{ bubbles: true, cancelable: true }}));
              clicked = true;
            }}
          }}
          
          // Method 3: Role-based selector
          if (!clicked) {{
            tabs = doc.querySelectorAll('[role="tab"]');
            if (tabs && tabs.length > {target_index}) {{
              const tab = tabs[{target_index}];
              tab.click();
              tab.dispatchEvent(new MouseEvent('click', {{ bubbles: true, cancelable: true }}));
              clicked = true;
            }}
          }}
          
          // Method 4: Find by tab list container
          if (!clicked) {{
            const tabList = doc.querySelector('[data-baseweb="tab-list"]');
            if (tabList) {{
              const tabButtons = tabList.querySelectorAll('button, [role="tab"]');
              if (tabButtons && tabButtons.length > {target_index}) {{
                const tab = tabButtons[{target_index}];
                tab.click();
                tab.dispatchEvent(new MouseEvent('click', {{ bubbles: true, cancelable: true }}));
                clicked = true;
              }}
            }}
          }}
          
          return clicked;
        }} catch (e) {{
          console.error('Tab click error:', e);
        }}
        return false;
      }}

      function runOnce() {{
        let pdoc = null;
        try {{ pdoc = window.parent.document; }} catch(e) {{}}
        const cdoc = document;

        let success = false;
        if (pdoc) {{
          success = clickTab(pdoc) || success;
        }}
        success = clickTab(cdoc) || success;

        if (success) {{
          if (pdoc) scrollEverythingToTop(pdoc);
          scrollEverythingToTop(cdoc);
        }}
        return success;
      }}

      // Wait for DOM to be ready
      function waitForTabs() {{
        let tries = 0;
        const maxTries = 80;  // Increased tries for better reliability
        const interval = setInterval(() => {{
          tries++;
          const success = runOnce();
          if (success) {{
            clearInterval(interval);
            return;
          }}
          if (tries >= maxTries) {{
            clearInterval(interval);
            console.warn('Tab navigation: Max tries reached');
          }}
        }}, 30);  // Faster interval for quicker response
      }}

      // Run when DOM is ready
      if (document.readyState === 'loading') {{
        document.addEventListener('DOMContentLoaded', waitForTabs);
      }} else {{
        waitForTabs();
      }}

      // Also try immediately and with delays as fallback
      setTimeout(runOnce, 0);
      setTimeout(runOnce, 10);
      setTimeout(runOnce, 50);
      setTimeout(runOnce, 100);
      setTimeout(runOnce, 200);
      setTimeout(runOnce, 400);
    }})();
    </script>
    """
    # st.iframe does not allow height=0; "content" keeps the injected script block minimal.
    st.iframe(js, height="content", width="stretch")


# ----------------------------
# CREWAI SETUP
# ----------------------------
def get_llm():
    """Get CrewAI LLM instance (crewai.llm.LLM), not LangChain chat models.

    CrewAI Agent.llm must be a model name string or crewai.llms.base_llm.BaseLLM;
    LangChain's ChatOpenAI is not accepted.
    """
    if not CREWAI_AVAILABLE or not LLM:
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    # CrewAI uses LiteLLM under the hood — prefix with "anthropic/" for correct routing
    model_str = DEFAULT_MODEL if DEFAULT_MODEL.startswith("anthropic/") else f"anthropic/{DEFAULT_MODEL}"
    try:
        return LLM(
            model=model_str,
            temperature=DEFAULT_TEMPERATURE,
            api_key=api_key,
            timeout=30,
        )
    except Exception:
        return None


# ----------------------------
# REQUIREMENT ANALYST AGENT (CrewAI)
# ----------------------------


try:
    from agents.requirements_agent import RequirementsAnalystAgent
    from agents.design_agent import DesignGenerationAgent
    from agents.chat_agent import ChatAssistantAgent, call_chat_assistant
    from agents.planning_agent import PlanningManagerAgent
    from agents.quality_agent import QualityManagerAgent
    from agents.support_agent import SupportManagerAgent
    from agents.database_agent import DatabaseAgent
    from agents.project_manager_agent import ProjectManagerAgent
except Exception:
    RequirementsAnalystAgent = None
    DesignGenerationAgent = None
    ChatAssistantAgent = None
    PlanningManagerAgent = None
    QualityManagerAgent = None
    SupportManagerAgent = None
    DatabaseAgent = None
    ProjectManagerAgent = None
    call_chat_assistant = None


def _ensure_agent_imports() -> None:
    """Lazily import agent classes so `agents/*` can import ui.main_ui without circular import."""
    global RequirementsAnalystAgent
    global DesignGenerationAgent
    global ChatAssistantAgent
    global PlanningManagerAgent
    global QualityManagerAgent
    global SupportManagerAgent
    global DatabaseAgent
    global ProjectManagerAgent
    global call_chat_assistant

    if (
        RequirementsAnalystAgent is not None
        and DesignGenerationAgent is not None
        and ChatAssistantAgent is not None
        and PlanningManagerAgent is not None
        and QualityManagerAgent is not None
        and SupportManagerAgent is not None
        and DatabaseAgent is not None
        and ProjectManagerAgent is not None
        and call_chat_assistant is not None
    ):
        return

    from agents.requirements_agent import RequirementsAnalystAgent as _RequirementsAnalystAgent
    from agents.design_agent import DesignGenerationAgent as _DesignGenerationAgent
    from agents.chat_agent import ChatAssistantAgent as _ChatAssistantAgent
    from agents.chat_agent import call_chat_assistant as _call_chat_assistant
    from agents.planning_agent import PlanningManagerAgent as _PlanningManagerAgent
    from agents.quality_agent import QualityManagerAgent as _QualityManagerAgent
    from agents.support_agent import SupportManagerAgent as _SupportManagerAgent
    from agents.database_agent import DatabaseAgent as _DatabaseAgent
    from agents.project_manager_agent import ProjectManagerAgent as _ProjectManagerAgent

    RequirementsAnalystAgent = _RequirementsAnalystAgent
    DesignGenerationAgent = _DesignGenerationAgent
    ChatAssistantAgent = _ChatAssistantAgent
    PlanningManagerAgent = _PlanningManagerAgent
    QualityManagerAgent = _QualityManagerAgent
    SupportManagerAgent = _SupportManagerAgent
    DatabaseAgent = _DatabaseAgent
    ProjectManagerAgent = _ProjectManagerAgent
    call_chat_assistant = _call_chat_assistant


def run_full_pipeline(api_key: str, session_id: str) -> Dict[str, Any]:
    """Run the full pipeline: requirements → database → design → planning → quality → support. Manages workflow state and stores intermediate artifacts."""
    _ensure_agent_imports()
    req = st.session_state.get("requirements_for_review")
    chat_history = st.session_state.get("chat_history") or []
    has_valid_req = req and isinstance(req, dict) and "error" not in req
    if not has_valid_req and not chat_history:
        return create_error_response("No valid requirements and no chat history", "Generate requirements in Chat → Review first, or run pipeline after a conversation.")

    req = req if has_valid_req else {}
    project_mgr = ProjectManagerAgent(api_key)
    steps = project_mgr.get_execution_plan(
        req if req else {"project_name": "From chat", "features": []},
        has_requirements=has_valid_req,
        has_database=bool(has_valid_req and (req or {}).get("database_schema") and isinstance((req or {}).get("database_schema"), dict) and (req or {}).get("database_schema", {}).get("tables")),
        has_design=bool(st.session_state.get("interface_design") and "error" not in (st.session_state.get("interface_design") or {})),
        has_planning=bool(st.session_state.get("cycle_plan") and "error" not in (st.session_state.get("cycle_plan") or {})),
        has_quality=bool(st.session_state.get("quality_report") and "error" not in (st.session_state.get("quality_report") or {})),
        has_support=bool(st.session_state.get("support_governance") and "error" not in (st.session_state.get("support_governance") or {})),
    )

    st.session_state.setdefault("pipeline_step", None)
    st.session_state.setdefault("pipeline_completed_steps", [])
    st.session_state["pipeline_running"] = True
    st.session_state["pipeline_errors"] = {}
    st.session_state["pipeline_completed_steps"] = []

    for step in steps:
        st.session_state["pipeline_step"] = step
        req = st.session_state.get("requirements_for_review") or {}
        if not isinstance(req, dict) or "error" in req:
            req = {}
        db_schema = req.get("database_schema") if isinstance(req, dict) else None

        try:
            if step == "requirements":
                # Requirements is the only hard-stop: everything else depends on it
                if has_valid_req or not chat_history:
                    st.session_state["pipeline_completed_steps"].append("requirements")
                    continue
                convo = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history])
                analyzer = st.session_state.get("analyzer")
                if not analyzer or not getattr(analyzer, "api_key", None):
                    analyzer = RequirementsAnalystAgent()
                    analyzer.api_key = api_key
                    analyzer.headers = {
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    } if api_key else {}
                else:
                    analyzer.api_key = api_key
                    analyzer.headers = {
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    } if api_key else {}
                result = analyzer.analyze_requirements(convo, max_retries=1)
                st.session_state.requirements_for_review = result
                if "error" in result:
                    # Requirements failing is fatal — nothing else can run without them
                    st.session_state["pipeline_errors"]["requirements"] = result.get("error", "Requirements failed")
                    break
                st.session_state["pipeline_completed_steps"].append("requirements")
                try:
                    save_artifact(session_id, "requirements", result)
                except Exception:
                    pass
                req = result

            elif step == "database":
                if not req or "error" in req:
                    st.session_state["pipeline_errors"]["database"] = "Requirements not available"
                    continue  # non-fatal: keep running other steps
                db_agent = DatabaseAgent(api_key)
                schema_result = db_agent.generate_schema(req)
                if "error" not in schema_result:
                    req = dict(st.session_state.get("requirements_for_review") or {})
                    req["database_schema"] = schema_result
                    st.session_state.requirements_for_review = req
                    st.session_state["pipeline_completed_steps"].append("database")
                    try:
                        save_artifact(session_id, "requirements", req)
                    except Exception:
                        pass
                else:
                    st.session_state["pipeline_errors"]["database"] = schema_result.get("error", "Database schema failed")
                    # non-fatal: continue with other steps

            elif step == "design":
                design_agent = DesignGenerationAgent(api_key)
                design = design_agent.generate_design(req)
                st.session_state.interface_design = design
                st.session_state.requirements_confirmed = True
                if "error" not in design:
                    st.session_state["pipeline_completed_steps"].append("design")
                    try:
                        save_artifact(session_id, "design", design)
                    except Exception:
                        pass
                else:
                    st.session_state["pipeline_errors"]["design"] = design.get("error", "Design failed")

            elif step == "planning":
                plan_agent = PlanningManagerAgent(api_key)
                plan = plan_agent.generate_cycle_plan(req, st.session_state.get("interface_design"))
                st.session_state.cycle_plan = plan
                if "error" not in plan:
                    st.session_state["pipeline_completed_steps"].append("planning")
                    try:
                        save_artifact(session_id, "cycle_plan", plan)
                    except Exception:
                        pass
                else:
                    st.session_state["pipeline_errors"]["planning"] = plan.get("error", "Planning failed")

            elif step == "quality":
                quality_agent = QualityManagerAgent(api_key)
                report = quality_agent.generate_quality_report(
                    req, st.session_state.get("interface_design"), db_schema
                )
                st.session_state.quality_report = report
                if "error" not in report:
                    st.session_state["pipeline_completed_steps"].append("quality")
                    try:
                        save_artifact(session_id, "quality_report", report)
                    except Exception:
                        pass
                else:
                    st.session_state["pipeline_errors"]["quality"] = report.get("error", "Quality check failed")

            elif step == "support":
                support_agent = SupportManagerAgent(api_key)
                pkg = support_agent.generate_support_package(
                    req,
                    st.session_state.get("interface_design"),
                    db_schema,
                    st.session_state.get("cycle_plan"),
                )
                st.session_state.support_governance = pkg
                if "error" not in pkg:
                    st.session_state["pipeline_completed_steps"].append("support")
                    try:
                        save_artifact(session_id, "support_governance", pkg)
                    except Exception:
                        pass
                else:
                    st.session_state["pipeline_errors"]["support"] = pkg.get("error", "Support package failed")

        except Exception as e:
            st.session_state["pipeline_errors"][step] = str(e)[:200]
            if step == "requirements":
                break  # requirements failure is the only hard stop

    st.session_state["pipeline_running"] = False
    st.session_state["pipeline_step"] = None
    errors = st.session_state.get("pipeline_errors", {})
    if errors:
        # Return partial success — caller decides how to surface per-step errors
        return {
            "ok": len(st.session_state.get("pipeline_completed_steps", [])) > 0,
            "completed_steps": st.session_state.get("pipeline_completed_steps", []),
            "step_errors": errors,
        }
    return {"ok": True, "completed_steps": st.session_state.get("pipeline_completed_steps", [])}


# ----------------------------
# UI SIMULATION (Device HTML)
# ----------------------------
def build_device_html(design: Dict[str, Any], is_mobile: bool, primary_hex: str, secondary_hex: str, screen_index: int = 0) -> str:
    """Build HTML preview of device interface."""
    nav = design.get("navigation") or {}
    nav_items = nav.get("items") or ["Home", "Features", "About", "Contact"]

    screens = design.get("screens") if isinstance(design.get("screens"), list) else []
    # Render all screens for interactive navigation (not just one)
    # screen_index is used as the initial active screen

    bg1 = "#0b1220"
    bg2 = "#111827"
    surface = "#ffffff"
    border = "rgba(255,255,255,0.08)"

    # Navigation will be generated per screen in the screen loop
    nav_html = ""  # Not used anymore - each screen has its own appbar

    get_component_icon = get_component_icon_glyph

    def generate_image_placeholder(comp: dict, primary: str, secondary: str) -> str:
        """Generate a styled image placeholder."""
        img_type = comp.get("image_type", "content")
        patterns = {
            "product": "linear-gradient(135deg, #f0f0f0 0%, #e0e0e0 100%)",
            "user": "radial-gradient(circle, #e3f2fd 0%, #bbdefb 100%)",
            "background": f"linear-gradient(135deg, {primary}22 0%, {secondary}22 100%)",
            "content": "linear-gradient(45deg, #f5f5f5 25%, transparent 25%), linear-gradient(-45deg, #f5f5f5 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #f5f5f5 75%), linear-gradient(-45deg, transparent 75%, #f5f5f5 75%)"
        }
        pattern = patterns.get(img_type, patterns["content"])
        return f"""
        <div style="
            width: 100%;
            height: 120px;
        border-radius: 12px;
            background: {pattern};
            background-size: 20px 20px;
            background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #999;
            font-size: 2rem;
        ">🖼️</div>
        """

    screens_html = ""
    # Get global UI components as fallback
    global_ui_components = design.get("ui_components") if isinstance(design.get("ui_components"), list) else []
    
    for idx, screen in enumerate(screens):
        screen_name = safe_str(screen.get("name"), f"Screen {idx+1}")
        screen_purpose = safe_str(screen.get("purpose"), "")
        
        # Try multiple possible keys for components
        comps = []
        if isinstance(screen.get("key_components"), list):
            comps = screen.get("key_components")
        elif isinstance(screen.get("components"), list):
            comps = screen.get("components")
        elif isinstance(screen.get("ui_components"), list):
            comps = screen.get("ui_components")
        elif global_ui_components:
            # Fallback to global UI components if screen has none
            comps = global_ui_components
        
        # Show more components: 12 for mobile, 15 for desktop
        comps = comps[: (12 if is_mobile else 15)]

        comp_cards = ""
        if not comps:
            # Improved empty state messaging
            comp_cards = f"""
            <div style="
                padding: 50px 30px;
                text-align: center;
                color: #6b7280;
                font-size: 1rem;
            ">
                <div style="font-size: 3.5rem; margin-bottom: 20px; opacity: 0.6;">📱</div>
                <div style="font-weight: 900; font-size: 1.1rem; color: #111827; margin-bottom: 8px;">
                    This screen is defined, but not detailed yet
                </div>
                <div style="opacity: 0.7; margin-top: 6px; line-height: 1.6;">
                    The AI generated the screen structure but did not assign components.
                </div>
            </div>
            """
        
        # Track component types for section headers
        seen_types = set()
        type_labels = {
            "button": "Primary Actions",
            "search": "Navigation & Search",
            "input": "Forms & Inputs",
            "card": "Content Cards",
            "list": "Lists & Menus",
            "image": "Media & Images"
        }
        for comp in comps:
            if not isinstance(comp, dict):
                comp = {"name": str(comp), "type": "Component"}
            
            cname = safe_str(comp.get("name"), "Component")
            ctype = safe_str(comp.get("type"), "Component")
            icon = get_component_icon(comp)
            has_image = comp.get("has_image", False)
            ct = ctype.lower()
            
            # Add section header for first component of each type
            section_label = None
            for key, label in type_labels.items():
                if key in ct and key not in seen_types:
                    section_label = label
                    seen_types.add(key)
                    break
            
            if section_label:
                comp_cards += f"""
                <div class="section-label">{section_label}</div>
                """

            # Generate component HTML based on type - Professional card structure
            if "button" in ct:
                # Extract navigation target from interaction or infer from component name
                interaction = comp.get("interaction", "")
                nav_target = None
                if interaction and "navigate:" in interaction:
                    nav_target = interaction.split("navigate:")[-1].strip()
                else:
                    # Try to find matching screen name from component name
                    for screen in screens:
                        screen_name_lower = safe_str(screen.get("name"), "").lower()
                        if cname.lower() in screen_name_lower or any(word in screen_name_lower for word in cname.lower().split() if len(word) > 3):
                            nav_target = safe_str(screen.get("name"), "")
                            break

                if nav_target:
                    onclick = f"onclick=\"saveClick('{_j(screen_name)}', '{_j(cname)}'); navigateTo('{_j(nav_target)}')\""
                else:
                    onclick = f"onclick=\"saveClick('{_j(screen_name)}', '{_j(cname)}')\""
                tooltip = 'title="Static preview element"' if not nav_target else ""
                role_attr = f'role="button" tabindex="0" aria-label="Navigate to {_h(cname)}"' if nav_target else ''
                keypress = f"onkeypress=\"if(event.key==='Enter') {{ saveClick('{_j(screen_name)}', '{_j(cname)}'); navigateTo('{_j(nav_target)}') }}\"" if nav_target else ''
                opacity_style = "1" if nav_target else "0.85"

                comp_cards += f"""
                <button {onclick} {tooltip} {role_attr} {keypress} class="btn-primary" style="opacity: {opacity_style};">
                    {icon} {_h(cname)}
                </button>
                """
            elif "search" in ct:
                field_id = f"search_{_j(screen_name.lower().replace(' ', '_'))}"
                comp_cards += f"""
                <div class="input-wrap">
                    <div class="left">{icon}</div>
                    <input class="input" type="text" placeholder="Search..." id="{field_id}" data-screen="{_h(screen_name)}" data-field="search" onblur="saveInput('{_j(screen_name)}', 'search', this.value)" />
                </div>
                """
            elif "card" in ct or "image" in ct or has_image:
                # Extract navigation target for cards
                interaction = comp.get("interaction", "")
                nav_target = None
                if interaction and "navigate:" in interaction:
                    nav_target = interaction.split("navigate:")[-1].strip()
                else:
                    # Try to find matching screen name
                    for screen in screens:
                        screen_name_lower = safe_str(screen.get("name"), "").lower()
                        if cname.lower() in screen_name_lower or any(word in screen_name_lower for word in cname.lower().split() if len(word) > 3):
                            nav_target = safe_str(screen.get("name"), "")
                            break

                onclick = f"onclick=\"navigateTo('{_j(nav_target)}')\"" if nav_target else ""
                image_html = generate_image_placeholder(comp, primary_hex, secondary_hex) if has_image else ""
                chevron = '<span style="margin-left: auto; opacity: 0.35; font-weight: 900; font-size: 1.2rem;">›</span>' if nav_target else ""
                tooltip = 'title="Static preview element"' if not nav_target else ""
                role_attr = f'role="button" tabindex="0" aria-label="Navigate to {_h(cname)}"' if nav_target else ''
                keypress = f"onkeypress=\"if(event.key==='Enter') navigateTo('{_j(nav_target)}')\"" if nav_target else ''
                cursor_style = "pointer" if nav_target else "default"
                opacity_style = "1" if nav_target else "0.85"
                desc = _h(safe_str(comp.get("description", ctype), ""))

                comp_cards += f"""
                <div {onclick} {tooltip} {role_attr} {keypress} class="card" style="cursor: {cursor_style}; opacity: {opacity_style};">
                    {image_html}
                    <div class="row">
                        <div class="ic">{icon}</div>
                        <div style="flex: 1;">
                            <div class="t1">{_h(cname)}</div>
                            <div class="t2">{desc}</div>
                        </div>
                        {chevron}
                    </div>
                </div>
                """
            elif "list" in ct:
                comp_cards += f"""
                <div class="card">
                    <div class="row" style="margin-bottom: 8px;">
                        <div class="ic">{icon}</div>
                        <div class="t1">{_h(cname)}</div>
                    </div>
                    <div style="padding-left: 52px;">
                        <div style="padding: 8px 0; border-bottom: 1px solid var(--border); color: var(--muted); font-size: 14px;">• Item 1</div>
                        <div style="padding: 8px 0; border-bottom: 1px solid var(--border); color: var(--muted); font-size: 14px;">• Item 2</div>
                        <div style="padding: 8px 0; color: var(--muted); font-size: 14px;">• Item 3</div>
                    </div>
                </div>
                """
            elif "input" in ct:
                field_id = f"{_j(cname.lower().replace(' ', '_'))}_{_j(screen_name.lower().replace(' ', '_'))}"
                field_name = cname.lower().replace(' ', '_')
                comp_cards += f"""
                <div style="margin: 14px 0;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                        <span style="font-size: 1.1rem; color: var(--muted);">{icon}</span>
                        <label style="font-weight: 700; color: var(--text); font-size: 0.9rem;">{_h(cname)}</label>
                    </div>
                    <div class="input-wrap">
                        <div class="left">{icon}</div>
                        <input class="input" type="text" placeholder="Enter {_h(cname.lower())}..." id="{field_id}" data-screen="{_h(screen_name)}" data-field="{_h(field_name)}" onblur="saveInput('{_j(screen_name)}', '{_j(field_name)}', this.value)" />
                    </div>
                </div>
                """
            else:
                # Generic component - use card structure
                comp_cards += f"""
                <div class="card">
                    <div class="row">
                        <div class="ic">{icon}</div>
                        <div style="flex: 1;">
                            <div class="t1">{_h(cname)}</div>
                            <div class="t2">{_h(ctype)}</div>
                        </div>
                    </div>
                </div>
                """

        # Append screen HTML with professional appbar structure
        is_active = "active" if idx == screen_index else ""
        screens_html += f"""
        <div id="screen-{idx}" class="screen {is_active}" data-screen-name="{_h(screen_name)}" style="
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid rgba(0,0,0,0.07);
            box-shadow: 0 18px 40px rgba(0,0,0,0.10);
            margin: 14px 0;
            display: {'block' if idx == screen_index else 'none'};
            background: {surface};
        ">
            <div class="screen-scroll" style="padding: 14px;max-height: {'580px' if is_mobile else '600px'};overflow-y:auto;box-sizing:border-box;">
                <div class="screen-shell">
                    <div class="appbar">
                        <div onclick="goBack()" class="back-btn icon-btn" role="button" tabindex="0" aria-label="Go back" title="Go Back" onkeypress="if(event.key==='Enter') goBack()">←</div>
                        <div class="title">{_h(screen_name)}</div>
                        <div class="icon-btn" onclick="toast('Notifications')" aria-label="Notifications" role="button" tabindex="0">◉</div>
                    </div>
                    <div id="crumb" style="opacity:.65;font-size:.78rem;font-weight:800;text-align:center;margin-bottom:8px;color:var(--muted);"></div>
                    <div class="screen-content">
                        {comp_cards}
                    </div>
                </div>
            </div>
        </div>
        """
    
    # If no screens at all, show a fallback message
    if not screens_html:
        screens_html = f"""
        <div style="
            border-radius: 18px;
            overflow:hidden;
            border: 1px solid rgba(0,0,0,0.07);
            box-shadow: 0 18px 40px rgba(0,0,0,0.10);
            margin: 14px 0;
            padding: 60px 40px;
            text-align: center;
            background: {surface};
        ">
            <div style="font-size: 4rem; margin-bottom: 20px;">🎨</div>
            <div style="font-size: 1.5rem; font-weight: 900; color: #111827; margin-bottom: 12px;">No Screens Found</div>
            <div style="color: #6b7280; font-size: 1rem; line-height: 1.6;">
                The design specification doesn't contain any screens with components.<br/>
                Please check the Design Specs tab to see what was generated.
            </div>
        </div>
        """

    # Build bottom navigation for mobile - Professional style
    bottom_nav_html = ""
    if is_mobile and screens:
        nav_items_html = ""
        for idx, screen in enumerate(screens[:4]):  # Limit to 4 items for mobile
            screen_name = safe_str(screen.get("name"), f"Screen {idx+1}")
            # Use simple glyphs instead of emojis
            icon_map = {
                "home": "⌂", "search": "🔍", "profile": "●", "settings": "⚙",
                "favorites": "★", "bookmark": "★", "notifications": "◉",
                "translate": "文", "translat": "文"
            }
            icon = "●"
            screen_lower = screen_name.lower()
            for key, glyph in icon_map.items():
                if key in screen_lower:
                    icon = glyph
                    break
            
            # Ensure label doesn't truncate - use shorter version if needed
            label = screen_name
            if len(label) > 10:
                # Try to shorten intelligently
                if "translate" in screen_lower:
                    label = "Translate"
                elif "notification" in screen_lower:
                    label = "Alerts"
                elif "favorite" in screen_lower or "bookmark" in screen_lower:
                    label = "Saved"
                else:
                    label = screen_name[:8]
            
            is_active = "active" if idx == screen_index else ""
            nav_items_html += f"""
            <div onclick="navigateTo('{screen_name}')" data-nav="{screen_name}" class="nav-item {is_active}">
                <div class="nav-ic">{icon}</div>
                <div class="lbl">{label}</div>
            </div>
            """
        bottom_nav_html = f"""
        <div class="bottom-nav" style="
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            width: 100%;
            z-index: 100;
        ">
            {nav_items_html}
        </div>
        """
    
    # JavaScript for navigation with enhanced UX
    screen_names_js = [safe_str(s.get("name"), f"Screen {i+1}") for i, s in enumerate(screens)]
    nav_js = f"""
    <script>
    let navStack = [];
    const screenNames = {json.dumps(screen_names_js)};
    const scrollMemory = {{}}; // screenName -> scrollTop
    let modalOnConfirm = null;
    let touchStartX = null;
    let touchStartY = null;
    
    function toast(msg) {{
        const el = document.getElementById('toast');
        if (!el) return;
        el.textContent = msg;
        el.style.opacity = '1';
        el.style.transform = 'translateX(-50%) translateY(0)';
        setTimeout(() => {{
            el.style.opacity = '0';
            el.style.transform = 'translateX(-50%) translateY(-6px)';
        }}, 1100);
    }}
    
    function openModal(title, body, onConfirm) {{
        modalOnConfirm = onConfirm || null;
        document.getElementById('modalTitle').textContent = title;
        document.getElementById('modalBody').textContent = body;
        document.getElementById('modal').style.display = 'block';
    }}
    
    function closeModal() {{
        document.getElementById('modal').style.display = 'none';
        modalOnConfirm = null;
    }}
    
    function confirmModal() {{
        if (modalOnConfirm) modalOnConfirm();
        closeModal();
    }}
    
    function rememberScroll() {{
        const active = document.querySelector('.screen.active .screen-scroll');
        const name = document.querySelector('.screen.active')?.getAttribute('data-screen-name');
        if (active && name) scrollMemory[name] = active.scrollTop;
    }}
    
    function restoreScroll(screenName) {{
        const el = document.querySelector(`[data-screen-name="${{screenName}}"] .screen-scroll`);
        if (!el) return;
        const y = scrollMemory[screenName] || 0;
        setTimeout(() => {{ el.scrollTop = y; }}, 50);
    }}
    
    function showSkeleton() {{
        const active = document.querySelector('.screen.active');
        if (!active) return;
        const scrollEl = active.querySelector('.screen-scroll');
        if (!scrollEl) return;
        const shell = document.createElement('div');
        shell.className = 'skeletonShell';
        shell.innerHTML = `
            <div class="sk sk1"></div>
            <div class="sk sk2"></div>
            <div class="sk sk3"></div>
        `;
        scrollEl.appendChild(shell);
        setTimeout(() => shell.remove(), 220);
    }}
    
    function updateBackState() {{
        const backBtns = document.querySelectorAll('.back-btn');
        backBtns.forEach(btn => {{
            btn.style.opacity = navStack.length > 0 ? '1' : '0.3';
            btn.style.cursor = navStack.length > 0 ? 'pointer' : 'default';
        }});
    }}
    
    function updateBottomNav(activeName) {{
        document.querySelectorAll('.nav-item').forEach(el => {{
            const navName = el.getAttribute('data-nav');
            if (navName === activeName) {{
                el.classList.add('active');
            }} else {{
                el.classList.remove('active');
            }}
        }});
    }}
    
    // Persistence functions - save data to database via URL query params
    function saveInput(screen, field, value) {{
        if (!value || value.trim() === '') return;
        
        // Store in localStorage as backup
        const key = `prototype_${{screen}}_${{field}}`;
        localStorage.setItem(key, value);
        
        // Update URL query params to trigger Streamlit rerun
        try {{
            const params = new URLSearchParams(window.location.search);
            params.set('prototype_event', JSON.stringify({{
                type: 'prototype_input',
                payload: {{ screen: screen, field: field, value: value }}
            }}));
            // Use history.replaceState to avoid adding to browser history
            const newUrl = window.location.pathname + '?' + params.toString();
            window.history.replaceState({{}}, '', newUrl);
            // Trigger a small delay then reload query params (Streamlit will pick it up on next interaction)
        }} catch(e) {{
            console.log('Persistence update failed:', e);
        }}
    }}
    
    function saveClick(screen, buttonName) {{
        try {{
            const params = new URLSearchParams(window.location.search);
            params.set('prototype_event', JSON.stringify({{
                type: 'prototype_click',
                payload: {{ screen: screen, button: buttonName }}
            }}));
            const newUrl = window.location.pathname + '?' + params.toString();
            window.history.replaceState({{}}, '', newUrl);
        }} catch(e) {{
            console.log('Click save failed:', e);
        }}
    }}
    
    function showScreen(screenName, direction = 'forward') {{
        rememberScroll();
        const active = document.querySelector('.screen.active');
        const target = document.querySelector(`[data-screen-name="${{screenName}}"]`);
        if (!target || active === target) return;
        
        if (active) {{
            active.style.transition = 'opacity 0.2s ease, transform 0.2s ease';
            active.style.opacity = '0';
            active.style.transform = direction === 'back' ? 'translateX(6px)' : 'translateX(-6px)';
            setTimeout(() => {{
                active.classList.remove('active');
                active.style.display = 'none';
            }}, 200);
        }}
        
        target.style.display = 'block';
        target.style.opacity = '0';
        target.style.transform = direction === 'back' ? 'translateX(-6px)' : 'translateX(6px)';
        target.classList.add('active');
        
        requestAnimationFrame(() => {{
            target.style.transition = 'opacity 0.25s ease, transform 0.25s ease';
            target.style.opacity = '1';
            target.style.transform = 'translateX(0)';
        }});
        
        restoreScroll(screenName);
        updateBackState();
        updateBottomNav(screenName);
        
        // Restore saved input values from localStorage
        setTimeout(() => {{
            const inputs = target.querySelectorAll('input[data-screen][data-field]');
            inputs.forEach(input => {{
                const key = `prototype_${{input.getAttribute('data-screen')}}_${{input.getAttribute('data-field')}}`;
                const saved = localStorage.getItem(key);
                if (saved) input.value = saved;
            }});
        }}, 100);
        
        // Update breadcrumb
        const crumb = document.getElementById('crumb');
        if (crumb) crumb.textContent = screenName;
    }}
    
    function navigateTo(screenName) {{
        // Haptic feedback (if available)
        if (navigator.vibrate) navigator.vibrate(8);
        
        const active = document.querySelector('.screen.active');
        if (active) {{
            const current = active.getAttribute('data-screen-name');
            if (current && current !== screenName) {{
                // Save navigation event
                if (window.parent && window.parent !== window) {{
                    window.parent.postMessage({{
                        type: 'prototype_navigation',
                        payload: {{
                            from: current,
                            to: screenName
                        }}
                    }}, '*');
                }}
                navStack.push(current);
                showSkeleton();
            }}
        }}
        showScreen(screenName, 'forward');
        toast('Navigated');
    }}
    
    function goBack() {{
        if (navStack.length === 0) return;
        const prev = navStack.pop();
        showScreen(prev, 'back');
        toast('Back');
    }}
    
    // Gesture navigation (swipe back)
    document.addEventListener('touchstart', (e) => {{
        const t = e.touches[0];
        touchStartX = t.clientX;
        touchStartY = t.clientY;
    }}, {{passive: true}});
    
    document.addEventListener('touchend', (e) => {{
        if (touchStartX === null) return;
        const t = e.changedTouches[0];
        const dx = t.clientX - touchStartX;
        const dy = t.clientY - touchStartY;
        
        // swipe right = back (only if mostly horizontal + strong enough)
        if (dx > 70 && Math.abs(dy) < 50) {{
            goBack();
        }}
        
        touchStartX = null;
        touchStartY = null;
    }}, {{passive: true}});
    
    // Keyboard navigation
    document.addEventListener('keydown', (e) => {{
        if (e.key === 'Escape') closeModal();
        if (e.key === 'Backspace' && !e.target.matches('input, textarea')) goBack();
    }});
    
    // Initialize back button state, bottom nav, and page load
    document.addEventListener('DOMContentLoaded', () => {{
        updateBackState();
        // Set initial bottom nav state
        const active = document.querySelector('.screen.active');
        if (active) {{
            const activeName = active.getAttribute('data-screen-name');
            updateBottomNav(activeName);
            const crumb = document.getElementById('crumb');
            if (crumb) crumb.textContent = activeName;
        }}
        // Page load illusion
        const container = document.querySelector('.device-container');
        if (container) {{
            container.style.opacity = '0';
            container.style.transition = 'opacity 0.4s ease';
            requestAnimationFrame(() => {{
                container.style.opacity = '1';
            }});
        }}
        
        // Make inputs actually focusable (prevent parent handlers from stealing clicks)
        document.querySelectorAll('input, textarea').forEach((el) => {{
            el.addEventListener('pointerdown', (e) => e.stopPropagation(), true);
            el.addEventListener('mousedown', (e) => e.stopPropagation(), true);
            el.addEventListener('click', (e) => e.stopPropagation(), true);
            el.addEventListener('touchstart', (e) => e.stopPropagation(), {{ passive: true, capture: true }});
        }});
    }});
    </script>
    <style>
    :root {{
        --bg: #F6F7FB;
        --surface: #FFFFFF;
        --text: #111827;
        --muted: #6B7280;
        --border: rgba(17,24,39,.10);
        --shadow: 0 10px 30px rgba(17,24,39,.08);
        --shadow-soft: 0 6px 16px rgba(17,24,39,.06);
        --radius: 18px;
        --radius-sm: 14px;
        --primary: {primary_hex};
        --primary2: {secondary_hex};
    }}
    
    * {{ box-sizing: border-box; }}
    
    .screen {{
        transition: opacity 0.25s ease, transform 0.25s ease;
    }}
    
    .screen-scroll {{
        height: 100%;
        overflow-y: auto;
        -webkit-overflow-scrolling: touch;
        box-sizing: border-box;
    }}
    
    .screen-shell {{
        background: var(--bg);
        border-radius: var(--radius);
        padding: 14px;
        box-shadow: var(--shadow);
    }}
    
    /* App bar (single, clean) */
    .appbar {{
        background: linear-gradient(90deg, {primary_hex}dd, {secondary_hex}dd);
        border-radius: 16px;
        padding: 14px 14px;
        color: #fff;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: var(--shadow-soft);
        margin-bottom: 14px;
    }}
    
    .appbar .title {{
        font-weight: 800;
        letter-spacing: .2px;
        font-size: 18px;
        flex: 1;
        text-align: center;
    }}
    
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
        transition: transform .15s ease, background .15s ease;
        user-select: none;
        font-size: 18px;
        color: #fff;
    }}
    .icon-btn:active {{ transform: scale(.96); }}
    .icon-btn:hover {{ background: rgba(255,255,255,.22); }}
    
    .back-btn {{
        transition: opacity 0.2s ease;
    }}
    
    /* Section label */
    .section-label {{
        margin: 16px 4px 10px;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: var(--muted);
    }}
    
    /* Cards */
    .card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 14px;
        box-shadow: 0 2px 10px rgba(17,24,39,.04);
        margin: 12px 0;
    }}
    
    .row {{
        display: flex;
        gap: 12px;
        align-items: center;
    }}
    
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
    
    /* Inputs */
    .input {{
        width: 100%;
        padding: 14px 14px 14px 44px;
        border-radius: 14px;
        border: 1px solid var(--border);
        background: #fff;
        font-size: 15px;
        outline: none;
        transition: box-shadow .15s ease, border-color .15s ease;
        /* --- INPUT FIX: allow typing/focus inside Streamlit iframe --- */
        pointer-events: auto !important;
        user-select: text !important;
        -webkit-user-select: text !important;
        cursor: text !important;
    }}
    .input:focus {{
        border-color: {primary_hex}88;
        box-shadow: 0 0 0 4px {primary_hex}25;
    }}
    
    /* Ensure all inputs and textareas are focusable and allow text selection */
    input, textarea {{
        pointer-events: auto !important;
        user-select: text !important;
        -webkit-user-select: text !important;
        cursor: text !important;
    }}
    
    .input-wrap {{
        position: relative;
        margin: 10px 0;
    }}
    .input-wrap .left {{
        position: absolute;
        left: 14px;
        top: 50%;
        transform: translateY(-50%);
        opacity: .55;
        font-size: 18px;
    }}
    
    /* Buttons */
    .btn-primary {{
        padding: 14px 16px;
        border-radius: var(--radius-sm);
        background: linear-gradient(135deg, {secondary_hex}, {primary_hex});
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
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        user-select: none;
        border: none;
    }}
    .btn-primary:active {{ transform: scale(.96); }}
    
    /* Bottom nav */
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
        user-select: none;
        transition: background .15s ease, transform .15s ease;
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
    
    .nav-item.active {{
        background: {primary_hex}18;
        color: var(--primary);
    }}
    .nav-item.active .nav-ic {{
        border-color: {primary_hex}58;
        background: {primary_hex}1f;
        color: var(--primary);
    }}
    .nav-item:active {{ transform: scale(.98); }}
    
    .skeleton-card {{
        height: 80px;
        border-radius: 12px;
        background: linear-gradient(90deg, #eee 0%, #f5f5f5 50%, #eee 100%);
        background-size: 200% 100%;
        animation: pulse 1.2s infinite;
        margin: 14px 0;
    }}
    
    .skeletonShell {{
        padding: 12px;
    }}
    
    .sk {{
        height: 14px;
        border-radius: 10px;
        margin: 10px 0;
        background: linear-gradient(90deg, rgba(0,0,0,.06), rgba(0,0,0,.10), rgba(0,0,0,.06));
        animation: shimmer 1.1s infinite;
    }}
    
    .sk1 {{ width: 70%; height: 18px; }}
    .sk2 {{ width: 92%; }}
    .sk3 {{ width: 84%; }}
    
    @keyframes pulse {{
        0% {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
    }}
    
    @keyframes shimmer {{
        0% {{ filter: brightness(1); }}
        50% {{ filter: brightness(1.15); }}
        100% {{ filter: brightness(1); }}
    }}
    
    * {{
        -webkit-tap-highlight-color: transparent;
        touch-action: manipulation;
    }}
    
    button, [onclick] {{
        user-select: none;
        -webkit-user-select: none;
    }}
    
    /* Exclude inputs from button user-select rules */
    button input, button textarea, [onclick] input, [onclick] textarea {{
        user-select: text !important;
        -webkit-user-select: text !important;
    }}
    </style>
    
            """
    
    # Toast and Modal HTML (to be included in containers)
    toast_modal_html = f"""
    <!-- Toast Notification -->
    <div id="toast" style="
        position: absolute;
        left: 50%;
        top: 16px;
        transform: translateX(-50%);
        background: rgba(17,24,39,.92);
        color: #fff;
        padding: 10px 14px;
        border-radius: 999px;
        font-size: .85rem;
        font-weight: 800;
        opacity: 0;
        pointer-events: none;
        transition: opacity .2s ease, transform .2s ease;
        z-index: 9999;
    "></div>
    
    <!-- Modal Dialog -->
    <div id="modal" onclick="if(event.target.id === 'modal') closeModal()" style="
        position: absolute;
        inset: 0;
        display: none;
        z-index: 9998;
        background: rgba(0,0,0,.45);
    ">
        <div id="modalCard" style="
            position: absolute;
            left: 50%;
            top: 50%;
            transform: translate(-50%,-50%);
            width: 86%;
            max-width: 320px;
            background: #fff;
            border-radius: 18px;
            padding: 16px;
            box-shadow: 0 18px 60px rgba(0,0,0,.25);
        ">
            <div id="modalTitle" style="font-weight: 900; font-size: 1rem; margin-bottom: 6px; color: #111827;">Title</div>
            <div id="modalBody" style="opacity: .8; font-size: .92rem; line-height: 1.35; color: #6b7280; margin-bottom: 14px;">Body</div>
            <div style="display: flex; gap: 10px;">
                <button onclick="closeModal()" style="
                    flex: 1;
                    padding: 10px;
                    border-radius: 12px;
                    border: 1px solid rgba(17,24,39,.15);
                    background: #fff;
                    font-weight: 800;
                    cursor: pointer;
                    color: #111827;
                ">Cancel</button>
                <button onclick="confirmModal()" style="
                    flex: 1;
                    padding: 10px;
                    border-radius: 12px;
                    border: none;
                    background: {primary_hex};
                    color: #fff;
                    font-weight: 900;
                    cursor: pointer;
                ">OK</button>
            </div>
        </div>
    </div>
    """

    if is_mobile:
        return f"""
        <div class="device-container" style="
            max-width: 380px;
            margin: 0 auto;
            padding: 14px 12px;
            border-radius: 36px;
            background: linear-gradient(180deg, {bg1} 0%, {bg2} 100%);
            border: 1px solid {border};
            box-shadow: 0 30px 80px rgba(0,0,0,0.35);
            position: relative;
        ">
            <div style="
                position: absolute;
                top: 10px;
                right: 12px;
                background: rgba(0,0,0,0.55);
                color: white;
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 0.7rem;
                font-weight: 800;
                letter-spacing: 0.04em;
                z-index: 10;
            ">PROTOTYPE</div>
            <div style="height:20px;width:90px;margin: 6px auto 12px auto;border-radius:999px;background: rgba(255,255,255,0.12)"></div>
            <div style="
                background: {surface}; 
                border-radius: 26px; 
                padding: 14px; 
                padding-bottom: 80px;
                min-height: 620px;
                box-shadow: 0 -8px 24px rgba(0,0,0,0.12);
                position: relative;
            ">
                {screens_html}
                {bottom_nav_html}
            </div>
            {toast_modal_html}
            {nav_js}
        </div>
        """
    return f"""
    <div class="device-container" style="
        max-width: 1200px;
        margin: 0 auto;
        padding: 16px;
        border-radius: 18px;
        background: linear-gradient(180deg, rgba(17,24,39,0.06) 0%, rgba(17,24,39,0.02) 100%);
        border: 1px solid rgba(17,24,39,0.08);
        position: relative;
    ">
        <div style="
            position: absolute;
            top: 10px;
            right: 16px;
            background: rgba(0,0,0,0.55);
            color: white;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            z-index: 10;
        ">PROTOTYPE</div>
        {screens_html}
        {toast_modal_html}
        {nav_js}
    </div>
    """


# ----------------------------
# DISPLAY FUNCTIONS
# ----------------------------
def format_requirements_for_display(data: Dict[str, Any]) -> None:
    """Format and display requirements in Streamlit."""
    if "error" in data:
        st.error(f"❌ {data['error']}")
        if data.get("solution"):
            st.info(f"💡 {data['solution']}")
        if data.get("raw_response"):
            with st.expander("View raw response"):
                st.code(data["raw_response"])
        return
    
    if data.get("model_used"):
        st.caption(f"Model: {data.get('model_used')} • Generated: {data.get('timestamp', 'N/A')}")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### 📌 Project")
        st.markdown(f"**{data.get('project_name', 'Unnamed Project')}**")
    with c2:
        features_count = len(data.get("features", []) or [])
        roles_count = len(data.get("user_roles", []) or [])
        st.metric("Features", features_count)
        st.metric("Roles", roles_count)
                
        st.divider()

    st.markdown("### 🚀 Features")
    feats = data.get("features") or []
    if not feats:
        st.info("No features found. Try providing more detail.")
    else:
        for f in feats:
            fid = f.get("id", "F?")
            name = f.get("name", "Unnamed Feature")
            pr = f.get("priority", "Medium")
            pr_color = {"High": "🟥", "Medium": "🟨", "Low": "🟩"}.get(pr, "⬜")

            with st.expander(f"{pr_color} {fid} • {name}  —  Priority: {pr}"):
                st.markdown(f"**Description**: {f.get('description', '—')}")
                if f.get("user_stories"):
                    st.markdown("**User Stories**")
                    for s in f["user_stories"]:
                        st.write(f"• {s}")
                if f.get("acceptance_criteria"):
                    st.markdown("**Acceptance Criteria**")
                    for a in f["acceptance_criteria"]:
                        st.write(f"✅ {a}")

    st.divider()

    st.markdown("### 🧩 Technical Requirements")
    tech = data.get("technical_requirements") or {}
    if tech:
        a, b = st.columns(2)
        with a:
            if tech.get("platform"):
                st.markdown("**Platform**")
                for p in tech["platform"]:
                    st.write(f"• {p}")
            if tech.get("technologies"):
                st.markdown("**Technologies**")
                for t in tech["technologies"]:
                    st.write(f"• {t}")
        with b:
            if tech.get("frontend"):
                st.markdown("**Frontend**")
                for x in tech["frontend"]:
                    st.write(f"• {x}")
            if tech.get("backend"):
                st.markdown("**Backend**")
                for x in tech["backend"]:
                    st.write(f"• {x}")
            if tech.get("database"):
                st.markdown(f"**Database**: {tech['database']}")
            else:
                st.info("No technical requirements returned.")

    roles = data.get("user_roles") or []
    if roles:
        st.divider()
        st.markdown("### 👥 User Roles")
        cols = st.columns(min(4, len(roles)))
        for i, r in enumerate(roles):
            with cols[i % len(cols)]:
                st.info(f"👤 {r}")


# ----------------------------
# CSS STYLING
# ----------------------------
def inject_css(dark: bool) -> None:
    """Inject custom CSS styling."""
    if dark:
        bg = "#0b1220"
        card = "rgba(255,255,255,0.06)"
        text = "rgba(255,255,255,0.92)"
        muted = "rgba(255,255,255,0.70)"
        border = "rgba(255,255,255,0.10)"
    else:
        bg = "#f8fafc"
        card = "#ffffff"
        text = "#0f172a"
        muted = "#475569"
        border = "rgba(15,23,42,0.10)"

    st.markdown(
        f"""
<style>
/* Increase main content width */
.block-container {{
    max-width: 960px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
}}

.main {{ background: {bg}; padding-top: 1.2rem; }}

h1, h2, h3, h4 {{ color: {text} !important; letter-spacing: -0.02em; }}
p, li, span, div {{ color: {text}; }}

.hero {{
  border: 1px solid {border};
  background: linear-gradient(135deg, rgba(102,126,234,0.16) 0%, rgba(244,114,182,0.12) 100%);
  border-radius: 18px;
  padding: 18px 18px;
  margin: 8px 0 18px 0;
  box-shadow: 0 4px 14px rgba(0,0,0,0.06);
}}
.hero-title {{ font-size: 2.0rem; font-weight: 900; margin: 0; text-align: center; }}
.hero-sub {{ margin-top: 6px; color: {muted}; font-weight: 600; }}

.stTabs [data-baseweb="tab-list"] {{
  gap: 4px;
  background: transparent;
  flex-wrap: nowrap;
  overflow-x: visible;
  padding: 6px 0 10px 0;
  display: flex !important;
}}
.stTabs [data-baseweb="tab"] {{
  flex: 1 1 0 !important;
  min-width: 0 !important;
  border-radius: 999px;
  padding: 8px 12px;
  border: 1px solid {border};
  background: {card};
  font-weight: 800;
  font-size: 0.9rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  text-align: center;
}}
.stTabs [aria-selected="true"] {{
  background: linear-gradient(135deg, rgba(102,126,234,0.18) 0%, rgba(244,114,182,0.16) 100%);
  box-shadow: 0 3px 10px rgba(0,0,0,0.08);
}}

.stButton > button {{
  border-radius: 14px;
  font-weight: 900;
  border: 1px solid {border};
  padding: 0.70rem 1.1rem;
}}

.stChatMessage {{ border-radius: 18px; border: 1px solid {border}; }}

.stChatInput {{
  padding: 1rem 0 !important;
}}

.stChatInput > div {{
  background: {card} !important;
  border-radius: 20px !important;
  border: 1.5px solid {border} !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
  overflow: hidden !important;
  padding: 0 !important;
}}

.stChatInput > div > div {{
  background: {card} !important;
  padding: 0 !important;
  margin: 0 !important;
}}

.stChatInput > div > div > div {{
  background: {card} !important;
}}

.stChatInput textarea {{
  border-radius: 0 !important;
  border: none !important;
  background: {card} !important;
  padding: 0.875rem 1.125rem !important;
  font-size: 0.95rem !important;
  color: {text} !important;
  line-height: 1.5 !important;
  margin: 0 !important;
}}

.stChatInput textarea::placeholder {{
  color: {muted} !important;
}}

.stChatInput button {{
  border-radius: 12px !important;
  margin-left: 0.5rem !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 0.5rem !important;
}}

.stChatInput button svg {{
  margin: 0 !important;
}}

details {{ border-radius: 16px !important; }}
</style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# APPLICATION STATE
# ----------------------------
def ensure_state() -> None:
    """Initialize and ensure all session state variables exist."""
    _ensure_agent_imports()
    if "analyzer" not in st.session_state:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        st.session_state.analyzer = RequirementsAnalystAgent() if api_key else None

    # Initialize session ID for prototype persistence
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{int(time.time() * 1000)}"

    st.session_state.setdefault("dark_mode", False)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("initial_chat_mode", True)
    st.session_state.setdefault("processing_choice", None)
    st.session_state.setdefault("requirements_for_review", None)
    st.session_state.setdefault("requirements_confirmed", False)
    st.session_state.setdefault("interface_design", None)
    st.session_state.setdefault("generating_design", False)
    st.session_state.setdefault("active_tab", 0)
    st.session_state.setdefault("navigate_to_tab", None)
    st.session_state.setdefault("_nav_pending", False)
    st.session_state.setdefault("cycle_plan", None)
    st.session_state.setdefault("quality_report", None)
    st.session_state.setdefault("support_governance", None)
    st.session_state.setdefault("pipeline_step", None)
    st.session_state.setdefault("pipeline_completed_steps", [])
    st.session_state.setdefault("pipeline_running", False)
    st.session_state.setdefault("pipeline_errors", {})


def _build_full_export_zip() -> Optional[bytes]:
    """Bundle everything generated so far into one ZIP: specs at the root, plus a
    runnable Express+SQLite backend / React frontend scaffold under app/, when there's
    a design and/or database schema to build one from. Returns None if nothing exists yet.
    """
    req = st.session_state.get("requirements_for_review")
    design = st.session_state.get("interface_design")
    plan = st.session_state.get("cycle_plan")
    quality = st.session_state.get("quality_report")
    support = st.session_state.get("support_governance")

    def ok(d: Any) -> bool:
        return isinstance(d, dict) and "error" not in d

    req_ok = ok(req)
    design_ok = ok(design)
    db_schema = req.get("database_schema") if req_ok else None
    db_schema_ok = isinstance(db_schema, dict) and bool(db_schema.get("tables"))

    files: Dict[str, Any] = {}
    if req_ok:
        files["requirements.json"] = json.dumps(req, indent=2)
        if db_schema_ok:
            files["database_schema.json"] = json.dumps(db_schema, indent=2)
            try:
                from agents.database_agent import DatabaseAgent as _DBA
                _dba = _DBA.__new__(_DBA)
                files["schema_postgresql.sql"] = _dba.generate_ddl(db_schema, dialect="postgresql")
                files["schema_sqlite.sql"] = _dba.generate_ddl(db_schema, dialect="sqlite")
            except Exception:
                pass
    if design_ok:
        files["design.json"] = json.dumps(design, indent=2)
    if ok(plan):
        files["plan.json"] = json.dumps(plan, indent=2)
    if ok(quality):
        files["quality_report.json"] = json.dumps(quality, indent=2)
    if ok(support):
        files["support_package.json"] = json.dumps(support, indent=2)

    if req_ok and (design_ok or db_schema_ok):
        scaffold = build_scaffold_files(req, design if design_ok else None, db_schema if db_schema_ok else None)
        for path, content in scaffold.items():
            files[f"app/{path}"] = content

    if not files:
        return None
    return build_files_zip(files)


# ----------------------------
# MAIN APPLICATION
# ----------------------------
def main() -> None:
    """Main Streamlit application."""
    _ensure_agent_imports()
    st.set_page_config(
        page_title="Requirements Agent",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={"Get Help": None, "Report a bug": None, "About": "AI-powered requirements → UI design"},
    )

    ensure_state()
    inject_css(st.session_state.dark_mode)

    analyzer: Optional[RequirementsAnalystAgent] = st.session_state.analyzer

    

    # Hero section
    hero_col1, hero_col2, hero_col3 = st.columns([1, 20, 1])
    with hero_col2:
        st.markdown(
            """
            <div class="hero">
            <div class="hero-title">🤖 Requirements Agent</div>
                </div>
            """,
            unsafe_allow_html=True,
        )

    center = st.columns([1, 20, 1])[1]
    with center:
        tab_chat, tab_review, tab_design, tab_db, tab_planning, tab_quality, tab_support = st.tabs([
            "💬 Chat", "📋 Review", "🎨 Design", "🗄️ Database",
            "📅 Planning", "✅ Quality", "🛟 Support"
        ])

        # Handle prototype persistence events from JavaScript via query params
        # The JS sends postMessage, which we intercept via a hidden component
        query_params = st.query_params
        raw = query_params.get("prototype_event")
        if raw:
            try:
                # Fix: Handle both list and string formats (Streamlit version differences)
                if isinstance(raw, list):
                    raw = raw[0]
                event_data = json.loads(raw)
                event_type = event_data.get("type")
                payload = event_data.get("payload", {})
                session_id = st.session_state.get("session_id", f"session_{int(time.time() * 1000)}")
                
                if event_type == "prototype_input":
                    save_input(session_id, payload.get("screen", ""), payload.get("field", ""), payload.get("value", ""))
                elif event_type == "prototype_click":
                    save_click(session_id, payload.get("screen", ""), payload.get("button", ""))
                elif event_type == "prototype_navigation":
                    save_navigation(session_id, payload.get("from", ""), payload.get("to", ""))
                
                # Clear the query param to avoid reprocessing
                if "prototype_event" in query_params:
                    st.query_params.pop("prototype_event", None)
            except Exception as e:
                pass  # Silently fail for persistence events

        # Tab navigation - must happen AFTER tabs are created
        # Check multiple navigation flags for reliability
        nav_target = st.session_state.get("navigate_to_tab")
        nav_pending = st.session_state.get("_nav_pending", False)
        nav_target_alt = st.session_state.get("_nav_target")
        
        # Determine which target to use (priority: navigate_to_tab > _nav_target > _nav_pending)
        target = None
        if nav_target is not None:
            target = nav_target
        elif nav_target_alt is not None:
            target = nav_target_alt
        elif nav_pending:
            target = 1  # Default to Review tab (index 1)
        
        if target is not None:
            # Inject navigation JavaScript
            navigate_tabs_js(int(target))
            
            # Clear navigation flags after JS injection (but keep _nav_target for one more cycle)
            if nav_target is not None:
                st.session_state.navigate_to_tab = None
            if nav_pending:
                st.session_state._nav_pending = False
            # Don't clear _nav_target immediately - let it persist for one cycle in case JS needs retry
        elif "app_initialized" not in st.session_state:
            st.session_state.active_tab = 0
            st.session_state.app_initialized = True

        # TAB 1: CHAT
        with tab_chat:
            st.session_state.active_tab = 0

            if st.session_state.initial_chat_mode and not st.session_state.chat_history:
                st.session_state.chat_history = [
                    {
                        "role": "assistant",
                        "content": "Hey there! 👋 \n\nWhat's the main idea behind your project? ",
                    }
                ]
                st.session_state.initial_chat_mode = False
            
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    # Strip the [A | B | C] marker from assistant messages — the
                    # bracket text is rendered as buttons below, so showing it in
                    # the bubble would just duplicate the options.
                    display = (
                        strip_choice_block(msg["content"])
                        if msg.get("role") == "assistant"
                        else msg["content"]
                    )
                    st.markdown(display)

            if st.session_state.processing_choice:
                choice = st.session_state.processing_choice
                st.session_state.processing_choice = None
                st.session_state.chat_history.append({"role": "user", "content": choice})
                
                if analyzer and analyzer.api_key:
                    ai = call_chat_assistant(analyzer, st.session_state.chat_history)
                    st.session_state.chat_history.append({"role": "assistant", "content": ai})
                    st.rerun()
                else:
                    st.error("Anthropic API key missing. Add ANTHROPIC_API_KEY to .env")

            last = st.session_state.chat_history[-1]["content"] if st.session_state.chat_history else ""
            parsed = parse_choice_block(last)
            has_choices = parsed is not None

            if has_choices:
                _, choices = parsed
                st.markdown("#### Choose one option 👇")
                cols = st.columns(min(4, len(choices)))
                for i, c in enumerate(choices):
                    with cols[i % len(cols)]:
                        if st.button(c, key=f"mc_{i}", width="stretch"):
                            st.session_state.processing_choice = c
                            st.rerun()
                st.info("Pick one option above to continue.")

            if not has_choices:
                prompt = st.chat_input("Type your message…")
                if prompt:
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    
                    if analyzer and analyzer.api_key:
                        ai = call_chat_assistant(analyzer, st.session_state.chat_history)
                        st.session_state.chat_history.append({"role": "assistant", "content": ai})
                        st.rerun()
                    else:
                        st.error("Anthropic API key missing. Add ANTHROPIC_API_KEY to .env")
            
            if len(st.session_state.chat_history) > 3:
                st.divider()
                if st.button("✨ Turn this conversation into structured Requirements", type="primary", width="stretch"):
                    convo = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])

                    with st.spinner("🔄 Analyzing requirements…"):
                        result = analyzer.analyze_requirements(convo, max_retries=1)

                    st.session_state.requirements_for_review = result
                    st.session_state.requirements_confirmed = False

                    ok = (
                        isinstance(result, dict)
                        and "error" not in result
                        and (result.get("project_name") or (isinstance(result.get("features"), list) and len(result["features"]) > 0))
                    )

                    if not ok:
                        st.warning("⚠️ Requirements were not generated correctly. Open Review tab to see raw output/errors.")
                        st.session_state.navigate_to_tab = 1
                        st.session_state._nav_pending = True
                        st.session_state._nav_target = 1
                        st.rerun()
                    else:
                        try:
                            version = save_artifact(st.session_state.session_id, "requirements", result)
                            if version > 0:
                                st.caption(f"💾 Saved requirements (v{version})")
                        except Exception:
                            pass

                        st.success("✅ Requirements generated — review them before continuing.")

                        st.session_state.navigate_to_tab = 1
                        st.session_state._nav_pending = True
                        st.session_state._nav_target = 1
                        st.rerun()
        
        # TAB 2: REVIEW
        with tab_review:
            st.session_state.active_tab = 1

            if st.session_state.requirements_for_review:
                st.markdown("## 📋 Review Requirements")
                req_data = st.session_state.requirements_for_review
                
                # Check if it's a valid requirements dict or an error
                if isinstance(req_data, dict):
                    format_requirements_for_display(req_data)
                else:
                    st.error(f"❌ Invalid requirements data: {type(req_data)}")
                    st.json(req_data)
                
                st.divider()
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("✏️ Continue Chat", width="stretch"):
                        st.info("Go to the Chat tab to refine the idea.")
                with c2:
                    json_str = json.dumps(st.session_state.requirements_for_review, indent=2)
                    st.download_button(
                        "📥 Download Requirements JSON",
                        data=json_str,
                        file_name="requirements.json",
                        mime="application/json",
                        width="stretch",
                    )
                st.divider()
                st.caption("Central orchestrator (Project Manager Agent) runs Requirements → Database → Design → Planning → Quality → Support and stores artifacts.")
                if st.button("🚀 Run full pipeline", type="primary", key="run_full_pipeline", width="stretch"):
                    if not analyzer or not analyzer.api_key:
                        st.error("Anthropic API key missing. Add ANTHROPIC_API_KEY to your .env file.")
                    else:
                        with st.spinner("Project Manager Agent orchestrating pipeline…"):
                            out = run_full_pipeline(analyzer.api_key, st.session_state.session_id)
                        completed = out.get("completed_steps", [])
                        step_errors = out.get("step_errors", {})
                        if completed:
                            st.success(f"Pipeline finished. Completed: {', '.join(completed)}.")
                        for step_name, err_msg in step_errors.items():
                            st.warning(f"⚠️ {step_name.capitalize()} step failed: {err_msg}")
                        if not completed and not step_errors:
                            st.info("No steps were run.")
                        st.rerun()

                full_zip = _build_full_export_zip()
                if full_zip:
                    st.download_button(
                        "📦 Download project (.zip)",
                        data=full_zip,
                        file_name="generated_project.zip",
                        mime="application/zip",
                        width="stretch",
                        key="download_full_project",
                    )
                    st.caption(
                        "Includes requirements, database schema/DDL, design, plan, quality report, and support "
                        "package as JSON/SQL — plus, under app/, an actual runnable Express+SQLite backend and "
                        "React (Vite) frontend scaffolded from your specific tables and screens. "
                        "Run `npm install` in app/backend/ and app/frontend/ to try it."
                    )
            else:
                st.info("No requirements yet. Generate them from the Chat tab.")

        # TAB 3: DESIGN
        with tab_design:
            st.session_state.active_tab = 2

            if not st.session_state.requirements_confirmed:
                st.info("Confirm requirements first (Review tab).")
            else:
                if st.session_state.generating_design and analyzer and analyzer.api_key:
                    with st.spinner("🎨 Generating UI/UX design... This may take 30-60 seconds..."):
                        try:
                            design_agent = DesignGenerationAgent(analyzer.api_key)
                            design = design_agent.generate_design(st.session_state.requirements_for_review)
                            st.session_state.interface_design = design
                            st.session_state.generating_design = False
                            if "error" not in design:
                                st.success("✅ Design generated successfully!")
                                # Save design as versioned artifact (Chef-style)
                                try:
                                    version = save_artifact(st.session_state.session_id, "design", design)
                                    if version > 0:
                                        st.caption(f"💾 Saved design (v{version})")
                                except Exception:
                                    pass  # Fail silently to keep UX smooth
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error generating design: {str(e)}")
                            st.session_state.interface_design = {
                                "error": f"Error generating design: {str(e)}",
                                "solution": "Please try again or check your API key configuration."
                            }
                            st.session_state.generating_design = False
                            st.rerun()
                elif st.session_state.generating_design:
                    if not analyzer:
                        st.error("❌ Analyzer not initialized. Please check your configuration.")
                        st.session_state.generating_design = False
                    elif not analyzer.api_key:
                        st.error("❌ Anthropic API key missing. Add ANTHROPIC_API_KEY to your .env file")
                        st.session_state.generating_design = False
                
                design = st.session_state.interface_design
                if not design:
                    st.warning("No design generated yet.")
                elif "error" in design:
                    st.error(design["error"])
                    if design.get("solution"):
                        st.info(f"💡 {design['solution']}")
                    
                    # Show detailed debugging information for JSON parsing errors
                    if design.get("parse_error"):
                        with st.expander("🔍 Parse Error Details", expanded=True):
                            st.error(f"**Parse Error:** {design['parse_error']}")
                            if design.get("json_error"):
                                st.warning(f"**JSON Error:** {design['json_error']}")
                    
                    if design.get("raw_response"):
                        with st.expander("📄 Raw Response (for debugging)", expanded=False):
                            st.code(design["raw_response"], language="text")
                    
                    if design.get("extracted_content"):
                        with st.expander("📝 Extracted Content", expanded=False):
                            st.code(design["extracted_content"], language="text")
                    
                    if design.get("extracted_json"):
                        with st.expander("🔧 Extracted JSON (attempted)", expanded=False):
                            st.code(design["extracted_json"], language="json")
                    
                    if design.get("design_text"):
                        with st.expander("Raw design output"):
                            st.code(design["design_text"])
                    
                    if design.get("raw_error"):
                        with st.expander("Technical details"):
                            st.code(design["raw_error"])
                else:
                    # Debug: Show design structure if screens are empty
                    screens = design.get("screens") if isinstance(design.get("screens"), list) else []
                    has_components = any(
                        (isinstance(s.get("key_components"), list) and len(s.get("key_components", [])) > 0) or
                        (isinstance(s.get("components"), list) and len(s.get("components", [])) > 0) or
                        (isinstance(s.get("ui_components"), list) and len(s.get("ui_components", [])) > 0)
                        for s in screens if isinstance(s, dict)
                    )
                    if not screens or not has_components:
                        with st.expander("🔍 Debug: Design Structure", expanded=True):
                            st.json({
                                "has_screens": bool(screens),
                                "screens_count": len(screens) if screens else 0,
                                "has_ui_components": bool(design.get("ui_components")),
                                "ui_components_count": len(design.get("ui_components", [])) if isinstance(design.get("ui_components"), list) else 0,
                                "screens_detail": [{
                                    "name": s.get("name", "Unknown"),
                                    "keys": list(s.keys()) if isinstance(s, dict) else [],
                                    "has_key_components": bool(s.get("key_components")),
                                    "key_components_count": len(s.get("key_components", [])) if isinstance(s.get("key_components"), list) else 0,
                                    "has_components": bool(s.get("components")),
                                    "components_count": len(s.get("components", [])) if isinstance(s.get("components"), list) else 0,
                                    "has_ui_components": bool(s.get("ui_components")),
                                    "ui_components_count": len(s.get("ui_components", [])) if isinstance(s.get("ui_components"), list) else 0,
                                } for s in screens[:5]] if screens else [],
                                "design_keys": list(design.keys())
                            })
                    
                    st.markdown("## 🎨 Visual Prototype")

                    left, mid, right = st.columns([1, 2, 1])
                    with mid:
                        device = st.radio("Device view", ["💻 Desktop", "📱 Mobile"], horizontal=True)
                        is_mobile = "Mobile" in device

                    screens = design.get("screens") if isinstance(design.get("screens"), list) else []
                    screen_names = [safe_str(s.get("name"), f"Screen {i+1}") for i, s in enumerate(screens)]
                    selected_index = 0
                    if screen_names:
                        selected_name = st.selectbox("Select screen", screen_names)
                        selected_index = screen_names.index(selected_name)

                    st.divider()

                    sim_tab, spec_tab = st.tabs(["👁️ Live Preview", "📋 Design Specs"])

                    colors = design.get("color_scheme", {}) or {}
                    # Default color palette if not provided
                    default_colors = {
                        "primary": "#667eea",
                        "secondary": "#f472b6",
                        "accent": "#8b5cf6",
                        "surface": "#ffffff",
                        "background": "#f8fafc",
                        "text_primary": "#0f172a",
                        "text": "#0f172a",
                        "error": "#ef4444",
                        "success": "#10b981",
                        "warning": "#f59e0b"
                    }
                    # Merge defaults with actual colors (actual colors take precedence)
                    colors = {**default_colors, **colors}
                    primary_hex = extract_hex(colors.get("primary"), "#667eea")
                    secondary_hex = extract_hex(colors.get("secondary"), "#f472b6")

                    with sim_tab:
                        html = build_device_html(
                            design=design,
                            is_mobile=is_mobile,
                            primary_hex=primary_hex,
                            secondary_hex=secondary_hex,
                            screen_index=selected_index,
                        )
                        st.iframe(html, height=780 if is_mobile else 720, width="stretch")

                    with spec_tab:
                        st.markdown("### 📐 Overview")
                        st.info(design.get("design_overview", "—"))

                        st.markdown("### 🎨 Color Palette")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            primary_color = extract_hex(colors.get("primary"), "#667eea")
                            secondary_color = extract_hex(colors.get("secondary"), "#f472b6")
                            st.code(primary_color)
                            st.code(secondary_color)
                        with c2:
                            accent_color = extract_hex(colors.get("accent"), "#8b5cf6")
                            surface_color = extract_hex(colors.get("surface"), "#ffffff")
                            st.code(accent_color)
                            st.code(surface_color)
                        with c3:
                            bg_color = extract_hex(colors.get("background"), "#f8fafc")
                            text_color = extract_hex(colors.get("text_primary") or colors.get("text"), "#0f172a")
                            st.code(bg_color)
                            st.code(text_color)
                        with c4:
                            error_color = extract_hex(colors.get("error"), "#ef4444")
                            success_color = extract_hex(colors.get("success"), "#10b981")
                            warning_color = extract_hex(colors.get("warning"), "#f59e0b")
                            st.code(f"error: {error_color}")
                            st.code(f"success: {success_color}")
                            st.code(f"warning: {warning_color}")

                        st.divider()

                        st.markdown("### ✍️ Typography")
                        typo = design.get("typography", {}) or {}
                        st.write(f"**Font family:** {typo.get('font_family','—')}")
                        st.write(f"**Heading font:** {typo.get('heading_font','—')}")
                        st.write(f"**Body font:** {typo.get('body_font','—')}")
                        if typo.get("sizes"):
                            st.markdown("**Sizes**")
                            st.json(typo["sizes"])
                        if typo.get("weights"):
                            st.markdown("**Weights**")
                            st.json(typo["weights"])

                        st.divider()

                        st.markdown("### 🧭 Navigation")
                        st.json(design.get("navigation", {}))

                        st.markdown("### 🧱 UI Components")
                        if isinstance(design.get("ui_components"), list):
                            for comp in design["ui_components"][:8]:
                                if isinstance(comp, dict):
                                    comp_name = comp.get('name', 'Component')
                                    comp_type = comp.get('type', '—')
                                    with st.expander(f"{comp_name} • {comp_type}"):
                                        st.json(comp)
                                else:
                                    # Handle case where comp is a string or other type
                                    with st.expander(f"Component • {type(comp).__name__}"):
                                        st.write(comp)

                        st.markdown("### 📱 Responsive & Accessibility")
                        st.write(design.get("responsive_design", "—"))
                        st.write(design.get("accessibility", "—"))

                        st.markdown("### ✨ Animations & Icons")
                        st.write(design.get("animations", "—"))
                        st.write(design.get("icons", "—"))

                    st.divider()
                    st.download_button(
                        "📥 Download Design JSON",
                        data=json.dumps(design, indent=2),
                        file_name="design.json",
                        mime="application/json",
                        width="stretch",
                    )

        # TAB 4: DATABASE
        with tab_db:
            st.session_state.active_tab = 3
            
            # Use unified db.py for domain tables
            try:
                from db import (
                    create_table_from_schema, table_exists, list_domain_tables,
                    fetch_objects, insert_object, get_table_columns,
                    count_rows, get_database_url,
                    register_project_table, list_project_tables
                )
                DB_AVAILABLE = True
            except ImportError:
                DB_AVAILABLE = False
                def list_domain_tables(): return []
                def fetch_objects(*args, **kwargs): return []
                def insert_object(*args, **kwargs): pass
                def get_table_columns(*args): return []
                def create_table_from_schema(*args, **kwargs): return False, "Database not available"
                def table_exists(*args): return False
                def count_rows(*args): return -1
                def get_database_url(): return ""
                def register_project_table(*args, **kwargs): pass
                def list_project_tables(*args): return []

            _TABULAR_EXTS = (".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".xlsx", ".xls", ".parquet")
            _MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024  # 200 MB, matching Streamlit's upload cap

            def _detect_format(name: str) -> str:
                """Infer a tabular format from a filename or URL path."""
                lowered = (name or "").lower().split("?")[0].split("#")[0]
                for ext in _TABULAR_EXTS:
                    if lowered.endswith(ext):
                        return ext.lstrip(".")
                return ""

            def _read_tabular(raw: bytes, fmt: str):
                """Parse raw file bytes into a list of row dicts. Returns (rows, error)."""
                import io

                fmt = (fmt or "").lower()
                try:
                    if PANDAS_AVAILABLE:
                        buf = io.BytesIO(raw)
                        if fmt == "csv":
                            df = pd.read_csv(buf)
                        elif fmt == "tsv":
                            df = pd.read_csv(buf, sep="\t")
                        elif fmt in ("jsonl", "ndjson"):
                            df = pd.read_json(buf, lines=True)
                        elif fmt == "json":
                            df = pd.read_json(buf)
                        elif fmt in ("xlsx", "xls"):
                            df = pd.read_excel(buf)
                        elif fmt == "parquet":
                            df = pd.read_parquet(buf)
                        else:
                            return None, f"Unsupported format '{fmt}'. Supported: {', '.join(_TABULAR_EXTS)}."
                        df = df.where(pd.notnull(df), None)
                        return df.to_dict(orient="records"), None

                    # Fallback without pandas: stdlib handles the two text formats.
                    text = raw.decode("utf-8", errors="replace")
                    if fmt in ("csv", "tsv"):
                        import csv as _csv
                        reader = _csv.DictReader(io.StringIO(text), delimiter="\t" if fmt == "tsv" else ",")
                        return list(reader), None
                    if fmt in ("json", "jsonl", "ndjson"):
                        if fmt == "json":
                            data = json.loads(text)
                            if isinstance(data, dict):
                                data = data.get("data") or data.get("rows") or [data]
                        else:
                            data = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
                        rows = [r for r in data if isinstance(r, dict)]
                        return rows, (None if rows else "No object rows found in the JSON.")
                    return None, f"pandas is required to read '{fmt}' files."
                except Exception as e:
                    return None, str(e)[:300]

            def _infer_columns(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """Infer a column schema from sample row dicts (scans up to 200 rows)."""
                sample = rows[:200]
                names: List[str] = []
                for r in sample:
                    for k in r.keys():
                        if k not in names:
                            names.append(k)

                columns = []
                for name in names:
                    values = [r.get(name) for r in sample if r.get(name) is not None]
                    if any(isinstance(v, (dict, list)) for v in values):
                        ctype = "json"
                    elif values and all(isinstance(v, bool) for v in values):
                        ctype = "boolean"
                    elif values and all(isinstance(v, int) and not isinstance(v, bool) for v in values):
                        ctype = "bigint"
                    elif values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                        ctype = "float"
                    else:
                        ctype = "text"
                    columns.append({
                        "name": str(name), "type": ctype, "pk": False, "nullable": True,
                        "unique": False, "default": None, "notes": None,
                    })
                return columns

            def _import_rows(table_name: str, columns: List[Dict[str, Any]],
                             rows: List[Dict[str, Any]], purpose: str, assumption: str) -> None:
                """Shared tail for every import path: register the table in the schema,
                create it, insert the rows, and report the result."""
                new_table = {
                    "name": table_name, "purpose": purpose,
                    "columns": columns, "indexes": [], "relationships": [],
                }
                _merge_table_into_schema(new_table, assumption)

                created, create_err = create_table_from_schema(table_name, columns)
                if not created and "already exists" not in (create_err or "").lower():
                    st.warning(f"Schema saved, but table creation failed: {create_err}")
                    return
                register_project_table(st.session_state.session_id, table_name, purpose)

                col_names = [c["name"] for c in columns]
                json_cols = {c["name"] for c in columns if c["type"] == "json"}
                inserted, failed = 0, 0
                bar = st.progress(0.0, text="Inserting rows…")
                for i, row in enumerate(rows):
                    payload = {}
                    for cname in col_names:
                        val = row.get(cname)
                        if cname in json_cols or isinstance(val, (dict, list)):
                            val = json.dumps(val, ensure_ascii=False, default=str) if val is not None else None
                        elif val is not None and not isinstance(val, (str, int, float, bool)):
                            val = str(val)
                        payload[cname] = val
                    try:
                        insert_object(table_name, payload)
                        inserted += 1
                    except Exception:
                        failed += 1
                    if i % 25 == 0:
                        bar.progress(min((i + 1) / len(rows), 1.0), text="Inserting rows…")
                bar.empty()

                st.success(f"✅ Imported {inserted} row(s) into `{table_name}` ({len(columns)} columns).")
                if failed:
                    st.warning(f"{failed} row(s) could not be inserted.")
                st.session_state["db_show_import_form"] = False
                st.rerun()

            _HF_ROWS_API = "https://datasets-server.huggingface.co"

            def _hf_list_splits(dataset_id: str):
                """List available (config, split) pairs for a Hugging Face dataset."""
                try:
                    resp = requests.get(
                        f"{_HF_ROWS_API}/splits", params={"dataset": dataset_id.strip()}, timeout=30
                    )
                    if resp.status_code != 200:
                        return None, f"Hugging Face returned {resp.status_code}. Check the dataset ID is correct and public."
                    splits = resp.json().get("splits") or []
                    if not splits:
                        return None, "No splits found for this dataset."
                    return [(s.get("config"), s.get("split")) for s in splits], None
                except Exception as e:
                    return None, str(e)[:300]

            def _hf_type_to_sql(feature_type: Any) -> str:
                """Map a Hugging Face feature dtype to a schema type our DDL understands.
                Nested/complex features (lists, dicts, audio, images) are stored as JSON text."""
                if not isinstance(feature_type, dict) or feature_type.get("_type") != "Value":
                    return "json"
                dtype = str(feature_type.get("dtype", "")).lower()
                if dtype.startswith(("int8", "int16", "int32", "uint8", "uint16")):
                    return "int"
                if dtype.startswith(("int64", "uint32", "uint64")):
                    return "bigint"
                if dtype.startswith(("float", "double")):
                    return "float"
                if dtype == "bool":
                    return "boolean"
                return "text"

            def _hf_fetch(dataset_id: str, config: str, split: str, max_rows: int, progress=None):
                """Fetch up to max_rows from a Hugging Face dataset via the datasets-server
                REST API (no `datasets` package required). Returns (columns, rows, error)."""
                dataset_id = dataset_id.strip()
                columns, rows = None, []
                page_size = 100
                try:
                    while len(rows) < max_rows:
                        resp = requests.get(
                            f"{_HF_ROWS_API}/rows",
                            params={
                                "dataset": dataset_id, "config": config, "split": split,
                                "offset": len(rows), "length": min(page_size, max_rows - len(rows)),
                            },
                            timeout=60,
                        )
                        if resp.status_code != 200:
                            return None, None, f"Hugging Face returned {resp.status_code} while fetching rows."
                        payload = resp.json()

                        if columns is None:
                            columns = [
                                {
                                    "name": f.get("name", ""),
                                    "type": _hf_type_to_sql(f.get("type")),
                                    "pk": False, "nullable": True, "unique": False,
                                    "default": None, "notes": None,
                                }
                                for f in (payload.get("features") or []) if f.get("name")
                            ]
                            if not columns:
                                return None, None, "Could not determine the dataset's columns."

                        page = payload.get("rows") or []
                        if not page:
                            break
                        rows.extend(r.get("row", {}) for r in page)
                        if progress:
                            progress(min(len(rows) / max_rows, 1.0))
                        if len(page) < page_size:
                            break
                    return columns, rows[:max_rows], None
                except Exception as e:
                    return None, None, str(e)[:300]

            def _sanitize_table_name(raw: str) -> str:
                """Turn a dataset id like 'Buraaq/quran-md-ayahs' into a safe table name."""
                base = raw.strip().split("/")[-1]
                cleaned = "".join(ch if ch.isalnum() else "_" for ch in base).strip("_").lower()
                return cleaned or "imported_dataset"

            def _introspect_sqlite_fallback(url: str):
                """Introspect a sqlite:// URL with the stdlib sqlite3 module (used when
                SQLAlchemy isn't installed — mirrors db/db_service.py's own fallback)."""
                import sqlite3
                path = url.split("sqlite:///")[-1] if "sqlite:///" in url else url.split("sqlite://")[-1]
                if not path:
                    return None, "Could not parse a file path from the sqlite URL."
                try:
                    conn = sqlite3.connect(path)
                    cur = conn.cursor()
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                    table_names = [r[0] for r in cur.fetchall()]
                    tables = []
                    for tname in table_names:
                        cur.execute(f'PRAGMA table_info("{tname}")')
                        cols = []
                        for _, cname, ctype, notnull, dflt, pk in cur.fetchall():
                            cols.append({
                                "name": cname,
                                "type": (ctype or "text").lower(),
                                "pk": bool(pk),
                                "nullable": not bool(notnull),
                                "unique": False,
                                "default": dflt,
                                "notes": None,
                            })
                        cur.execute(f'PRAGMA foreign_key_list("{tname}")')
                        rels = [
                            {"type": "many-to-one", "to_table": row[2], "fk": row[3], "ref": row[4] or "id", "notes": None}
                            for row in cur.fetchall()
                        ]
                        tables.append({
                            "name": tname, "purpose": "Imported from existing database",
                            "columns": cols, "indexes": [], "relationships": rels,
                        })
                    conn.close()
                    return tables, None
                except Exception as e:
                    return None, str(e)[:300]

            def _introspect_database_url(url: str):
                """Introspect a database URL's schema, preferring SQLAlchemy and falling
                back to sqlite3 for sqlite:// URLs when SQLAlchemy isn't installed."""
                url = url.strip()
                try:
                    from sqlalchemy import create_engine, inspect as _sa_inspect
                except ImportError:
                    if url.startswith("sqlite:"):
                        return _introspect_sqlite_fallback(url)
                    return None, "SQLAlchemy is not installed, so only sqlite:// URLs can be imported directly."

                try:
                    engine = create_engine(url)
                    inspector = _sa_inspect(engine)
                    tables = []
                    for tname in inspector.get_table_names():
                        pk_cols = set(
                            (inspector.get_pk_constraint(tname) or {}).get("constrained_columns") or []
                        )
                        cols = []
                        for col in inspector.get_columns(tname):
                            cols.append({
                                "name": col["name"],
                                "type": str(col["type"]).lower(),
                                "pk": col["name"] in pk_cols,
                                "nullable": bool(col.get("nullable", True)),
                                "unique": False,
                                "default": str(col["default"]) if col.get("default") is not None else None,
                                "notes": None,
                            })
                        rels = []
                        for fk in inspector.get_foreign_keys(tname):
                            if fk.get("constrained_columns") and fk.get("referred_table"):
                                rels.append({
                                    "type": "many-to-one",
                                    "to_table": fk["referred_table"],
                                    "fk": fk["constrained_columns"][0],
                                    "ref": (fk.get("referred_columns") or ["id"])[0],
                                    "notes": None,
                                })
                        tables.append({
                            "name": tname, "purpose": "Imported from existing database",
                            "columns": cols, "indexes": [], "relationships": rels,
                        })
                    engine.dispose()
                    return tables, None
                except Exception as e:
                    return None, str(e)[:300]

            def _merge_table_into_schema(new_table: Dict[str, Any], assumption: str = "") -> Dict[str, Any]:
                """Add (or replace) a table in the session's database_schema and persist it."""
                req_local = st.session_state.get("requirements_for_review")
                if not isinstance(req_local, dict) or "error" in req_local:
                    req_local = {"project_name": new_table["name"]}
                schema = req_local.get("database_schema") or {"tables": [], "assumptions": []}
                schema["tables"] = [
                    t for t in (schema.get("tables") or []) if t.get("name") != new_table["name"]
                ] + [new_table]
                if assumption:
                    schema["assumptions"] = list(dict.fromkeys((schema.get("assumptions") or []) + [assumption]))
                req_local["database_schema"] = schema
                st.session_state.requirements_for_review = req_local
                try:
                    save_artifact(st.session_state.session_id, "requirements", req_local)
                except Exception:
                    pass
                return req_local

            # --- Dataset source actions: create a new dataset from scratch, or
            # import an existing dataset (database URL or Hugging Face dataset) ---
            col_create_ds, col_import_ds = st.columns(2)
            with col_create_ds:
                if st.button("🆕 Create Dataset", key="db_create_dataset_btn", width="stretch"):
                    st.session_state["db_show_create_form"] = not st.session_state.get("db_show_create_form", False)
                    st.session_state["db_show_import_form"] = False
            with col_import_ds:
                if st.button("🔗 Import Existing Dataset", key="db_import_dataset_btn", width="stretch"):
                    st.session_state["db_show_import_form"] = not st.session_state.get("db_show_import_form", False)
                    st.session_state["db_show_create_form"] = False

            _COL_TYPE_OPTIONS = [
                "uuid", "int", "bigint", "varchar", "text", "timestamp",
                "boolean", "json", "float", "decimal",
            ]
            _TYPE_ALIASES = {
                "integer": "int", "serial": "int", "smallint": "int",
                "bigserial": "bigint", "string": "varchar", "char": "varchar",
                "datetime": "timestamp", "date": "timestamp", "time": "timestamp",
                "bool": "boolean", "jsonb": "json", "double": "float", "real": "float",
                "numeric": "decimal",
            }
            _MAX_FORM_COLS = 50

            def _nearest_col_type(raw: Any) -> str:
                """Snap a schema type (e.g. 'VARCHAR(255)', 'jsonb') to a dropdown option."""
                base = str(raw or "text").lower().split("(")[0].strip()
                if base in _COL_TYPE_OPTIONS:
                    return base
                return _TYPE_ALIASES.get(base, "text")

            if st.session_state.get("db_show_create_form"):
                st.markdown("### 🆕 Create New Dataset")

                # Tables the Database Agent derived from the requirements — pick one to
                # prefill the form instead of typing every column by hand.
                _req_now = st.session_state.get("requirements_for_review")
                agent_tables = []
                if isinstance(_req_now, dict) and "error" not in _req_now:
                    agent_tables = [
                        t for t in ((_req_now.get("database_schema") or {}).get("tables") or [])
                        if isinstance(t, dict) and t.get("name")
                    ]

                blank_label = "✏️ Blank (define columns myself)"
                source_table: Optional[Dict[str, Any]] = None
                if agent_tables:
                    choice = st.selectbox(
                        "Start from",
                        [blank_label] + [t["name"] for t in agent_tables],
                        key="db_create_source_choice",
                        help="Tables the Database Agent designed from your requirements. "
                             "Pick one to prefill the form, then edit before creating.",
                    )
                    if choice != blank_label:
                        source_table = next((t for t in agent_tables if t["name"] == choice), None)
                        if source_table and source_table.get("purpose"):
                            st.caption(f"📋 {source_table['purpose']}")
                else:
                    st.caption(
                        "No Database Agent schema yet — generate requirements from the Chat tab "
                        "to pick from agent-designed tables here. Define columns manually below."
                    )

                # Widget keys are namespaced by the chosen source so that switching the
                # selection re-instantiates the inputs with that table's values as defaults.
                # (Assigning to widget state directly would fight Streamlit's own bookkeeping.)
                src_cols = [
                    c for c in ((source_table or {}).get("columns") or [])
                    if isinstance(c, dict) and c.get("name")
                ][:_MAX_FORM_COLS]
                slug = _sanitize_table_name(source_table["name"]) if source_table else "blank"

                with st.form("create_dataset_form", clear_on_submit=False):
                    new_table_name = st.text_input(
                        "Table name", value=(source_table or {}).get("name", ""),
                        key=f"new_table_name_{slug}", placeholder="e.g. customers",
                    )
                    num_cols = st.number_input(
                        "Number of columns", min_value=1, max_value=_MAX_FORM_COLS,
                        value=max(len(src_cols), 1) if src_cols else 3,
                        key=f"new_table_num_cols_{slug}",
                    )
                    st.caption("Define each column. Leave a column name blank to skip it.")

                    new_columns = []
                    for i in range(int(num_cols)):
                        d = src_cols[i] if i < len(src_cols) else {}
                        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
                        with c1:
                            cname = st.text_input(
                                "Name", value=d.get("name", ""), key=f"new_col_name_{slug}_{i}",
                                label_visibility="collapsed" if i else "visible",
                            )
                        with c2:
                            ctype = st.selectbox(
                                "Type", _COL_TYPE_OPTIONS,
                                index=_COL_TYPE_OPTIONS.index(_nearest_col_type(d.get("type"))) if d else 0,
                                key=f"new_col_type_{slug}_{i}",
                                label_visibility="collapsed" if i else "visible",
                            )
                        with c3:
                            cpk = st.checkbox("PK", value=bool(d.get("pk")), key=f"new_col_pk_{slug}_{i}")
                        with c4:
                            cnull = st.checkbox(
                                "Nullable", value=bool(d.get("nullable", True)),
                                key=f"new_col_null_{slug}_{i}",
                            )
                        new_columns.append({
                            "name": cname.strip(), "type": ctype, "pk": cpk, "nullable": cnull,
                            "unique": d.get("unique", False), "default": d.get("default"),
                            "notes": d.get("notes"),
                        })

                    submitted_create = st.form_submit_button("Create Dataset", type="primary", width="stretch")
                    if submitted_create:
                        valid_columns = [c for c in new_columns if c["name"]]
                        if not new_table_name.strip():
                            st.error("Table name is required.")
                        elif not valid_columns:
                            st.error("Add at least one column.")
                        else:
                            if not any(c["pk"] for c in valid_columns):
                                valid_columns.insert(0, {
                                    "name": "id", "type": "uuid", "pk": True, "nullable": False,
                                    "unique": False, "default": None, "notes": "Auto-added primary key",
                                })
                            # Carry over the agent's own metadata when this row started
                            # life as a Database Agent table and kept its name.
                            source = next(
                                (t for t in agent_tables if t.get("name") == new_table_name.strip()), None
                            )
                            new_table = {
                                "name": new_table_name.strip(),
                                "purpose": (source or {}).get("purpose") or "Manually created dataset",
                                "columns": valid_columns,
                                "indexes": (source or {}).get("indexes") or [],
                                "relationships": (source or {}).get("relationships") or [],
                            }
                            _merge_table_into_schema(new_table)

                            created, err = create_table_from_schema(new_table["name"], valid_columns)
                            if created:
                                register_project_table(
                                    st.session_state.session_id, new_table["name"], new_table["purpose"]
                                )
                                st.success(f"✅ Dataset '{new_table['name']}' created!")
                            else:
                                st.warning(f"Schema saved, but table creation failed: {err}")

                            st.session_state["db_show_create_form"] = False
                            st.rerun()

            if st.session_state.get("db_show_import_form"):
                st.markdown("### 🔗 Import Existing Dataset")
                import_source = st.radio(
                    "Source",
                    ["📁 Upload file", "🌐 File URL", "🤗 Hugging Face dataset", "🗄️ Database URL"],
                    key="db_import_source",
                    horizontal=True,
                )

                if import_source.endswith("Upload file"):
                    st.caption(
                        "Upload a CSV, TSV, JSON, JSONL, Excel, or Parquet file. "
                        "Columns and types are inferred from the contents."
                    )
                    uploaded = st.file_uploader(
                        "Dataset file",
                        type=[e.lstrip(".") for e in _TABULAR_EXTS],
                        key="import_file_uploader",
                    )
                    with st.form("import_file_form", clear_on_submit=False):
                        default_name = _sanitize_table_name(uploaded.name.rsplit(".", 1)[0]) if uploaded else ""
                        up_table_name = st.text_input(
                            "Table name", value=default_name, key="import_file_table_name",
                            placeholder="my_dataset",
                        )
                        up_max_rows = st.number_input(
                            "Max rows to import", min_value=10, max_value=50000, value=5000, step=500,
                            key="import_file_max_rows",
                        )
                        submitted_file = st.form_submit_button("Import File", type="primary", width="stretch")
                        if submitted_file:
                            if uploaded is None:
                                st.error("Please choose a file to upload.")
                            else:
                                fmt = _detect_format(uploaded.name)
                                with st.spinner(f"Reading {uploaded.name}…"):
                                    rows, read_err = _read_tabular(uploaded.getvalue(), fmt)
                                if read_err:
                                    st.error(f"Could not read the file: {read_err}")
                                elif not rows:
                                    st.warning("The file contained no rows.")
                                else:
                                    rows = rows[: int(up_max_rows)]
                                    table_name = _sanitize_table_name(up_table_name or default_name or uploaded.name)
                                    _import_rows(
                                        table_name, _infer_columns(rows), rows,
                                        f"Imported from uploaded file {uploaded.name}",
                                        f"Table '{table_name}' was imported from the uploaded file {uploaded.name}.",
                                    )

                elif import_source.endswith("File URL"):
                    with st.form("import_url_form", clear_on_submit=False):
                        st.caption(
                            "Paste a direct link to a data file — a GitHub raw URL, a public bucket object, "
                            "or a Google Sheets CSV export link."
                        )
                        file_url = st.text_input(
                            "File URL", key="import_file_url",
                            placeholder="https://raw.githubusercontent.com/user/repo/main/data.csv",
                        )
                        fmt_choice = st.selectbox(
                            "Format", ["auto"] + [e.lstrip(".") for e in _TABULAR_EXTS],
                            key="import_url_format",
                        )
                        url_table_name = st.text_input(
                            "Table name (blank = derive from the file name)", key="import_url_table_name"
                        )
                        url_max_rows = st.number_input(
                            "Max rows to import", min_value=10, max_value=50000, value=5000, step=500,
                            key="import_url_max_rows",
                        )
                        submitted_url = st.form_submit_button("Fetch & Import", type="primary", width="stretch")
                        if submitted_url:
                            url = file_url.strip()
                            if not url:
                                st.error("Please provide a file URL.")
                            elif not url.lower().startswith(("http://", "https://")):
                                st.error("Only http:// and https:// URLs are supported.")
                            else:
                                fmt = fmt_choice if fmt_choice != "auto" else _detect_format(url)
                                if not fmt:
                                    st.error(
                                        "Could not tell the file format from the URL. "
                                        "Pick one explicitly in the Format dropdown."
                                    )
                                else:
                                    raw, dl_err = None, None
                                    try:
                                        with st.spinner("Downloading…"):
                                            resp = requests.get(url, timeout=120, stream=True)
                                            if resp.status_code != 200:
                                                dl_err = f"the server returned {resp.status_code}."
                                            else:
                                                chunks, total = [], 0
                                                for chunk in resp.iter_content(chunk_size=1 << 20):
                                                    chunks.append(chunk)
                                                    total += len(chunk)
                                                    if total > _MAX_DOWNLOAD_BYTES:
                                                        dl_err = "the file is larger than the 200 MB limit."
                                                        break
                                                raw = b"".join(chunks) if dl_err is None else None
                                    except Exception as e:
                                        dl_err = str(e)[:300]

                                    if dl_err:
                                        st.error(f"Download failed: {dl_err}")
                                    else:
                                        with st.spinner("Parsing…"):
                                            rows, read_err = _read_tabular(raw, fmt)
                                        if read_err:
                                            st.error(f"Could not read the file: {read_err}")
                                        elif not rows:
                                            st.warning("The file contained no rows.")
                                        else:
                                            rows = rows[: int(url_max_rows)]
                                            derived = url.split("?")[0].rstrip("/").split("/")[-1].rsplit(".", 1)[0]
                                            table_name = _sanitize_table_name(url_table_name or derived)
                                            _import_rows(
                                                table_name, _infer_columns(rows), rows,
                                                f"Imported from {url}",
                                                f"Table '{table_name}' was imported from {url}.",
                                            )

                elif import_source.endswith("Database URL"):
                    with st.form("import_database_form", clear_on_submit=False):
                        st.caption(
                            "Provide a database connection URL to introspect its schema, e.g. "
                            "`postgresql://user:pass@host:5432/dbname` or `sqlite:///path/to/file.db`."
                        )
                        db_url = st.text_input("Database URL", key="import_db_url_input", type="password")
                        submitted_import = st.form_submit_button("Import Schema", type="primary", width="stretch")
                        if submitted_import:
                            if not db_url.strip():
                                st.error("Please provide a database URL.")
                            else:
                                imported_tables, import_err = _introspect_database_url(db_url)
                                if import_err:
                                    st.error(f"Failed to import: {import_err}")
                                elif not imported_tables:
                                    st.warning("Connected successfully, but no tables were found.")
                                else:
                                    req = st.session_state.get("requirements_for_review")
                                    if not isinstance(req, dict) or "error" in req:
                                        req = {"project_name": "Imported Database"}
                                    req["database_schema"] = {
                                        "tables": imported_tables,
                                        "assumptions": ["Imported from an existing database via connection URL."],
                                    }
                                    st.session_state.requirements_for_review = req
                                    try:
                                        save_artifact(st.session_state.session_id, "requirements", req)
                                    except Exception:
                                        pass
                                    st.success(f"✅ Imported {len(imported_tables)} table(s) from the database.")
                                    st.session_state["db_show_import_form"] = False
                                    st.rerun()
                else:
                    with st.form("import_hf_form", clear_on_submit=False):
                        st.caption(
                            "Enter a Hugging Face dataset ID — the same value you'd pass to "
                            "`load_dataset(...)`, e.g. `Buraaq/quran-md-ayahs`. "
                            "The dataset's schema and rows are imported into a new table."
                        )
                        hf_id = st.text_input(
                            "Dataset ID", key="import_hf_id_input", placeholder="Buraaq/quran-md-ayahs"
                        )
                        hf_split_choice = st.text_input(
                            "Split (leave blank for the first available)", key="import_hf_split_input",
                            placeholder="train",
                        )
                        hf_max_rows = st.number_input(
                            "Max rows to import", min_value=10, max_value=5000, value=500, step=100,
                            key="import_hf_max_rows",
                            help="Large datasets are paged in via the Hugging Face datasets-server API; "
                                 "keep this modest to avoid a long import.",
                        )
                        submitted_hf = st.form_submit_button("Import Dataset", type="primary", width="stretch")
                        if submitted_hf:
                            if not hf_id.strip():
                                st.error("Please provide a dataset ID.")
                            else:
                                with st.spinner(f"Looking up `{hf_id.strip()}` on Hugging Face…"):
                                    splits, split_err = _hf_list_splits(hf_id)
                                if split_err:
                                    st.error(f"Failed to load dataset: {split_err}")
                                else:
                                    wanted = hf_split_choice.strip()
                                    match = next(
                                        (cs for cs in splits if not wanted or cs[1] == wanted), None
                                    )
                                    if match is None:
                                        available = ", ".join(sorted({s for _, s in splits}))
                                        st.error(f"Split '{wanted}' not found. Available splits: {available}")
                                    else:
                                        config, split = match
                                        bar = st.progress(0.0, text="Downloading rows…")
                                        columns, rows, fetch_err = _hf_fetch(
                                            hf_id, config, split, int(hf_max_rows),
                                            progress=lambda p: bar.progress(p, text="Downloading rows…"),
                                        )
                                        bar.empty()

                                        if fetch_err:
                                            st.error(f"Failed to load dataset: {fetch_err}")
                                        elif not rows:
                                            st.warning("The dataset returned no rows.")
                                        else:
                                            table_name = _sanitize_table_name(hf_id)
                                            _import_rows(
                                                table_name, columns, rows,
                                                f"Imported from Hugging Face dataset {hf_id.strip()} ({split})",
                                                f"Table '{table_name}' was imported from the Hugging Face dataset "
                                                f"{hf_id.strip()} (split: {split}).",
                                            )

            st.divider()

            # ----------------------------------------------------------------
            # Browse the live database — every table that actually exists,
            # including ones absent from the requirements schema.
            # ----------------------------------------------------------------
            st.markdown("## 🗃️ Browse Database")

            if not DB_AVAILABLE:
                st.info("Database helper not available. Please check db/db_service.py is accessible.")
            else:
                def _mask_db_url(url: str) -> str:
                    """Hide any password in a connection URL before displaying it."""
                    if "@" not in url or "://" not in url:
                        return url
                    scheme, rest = url.split("://", 1)
                    creds, host = rest.rsplit("@", 1)
                    user = creds.split(":", 1)[0]
                    return f"{scheme}://{user}:••••@{host}"

                all_tables = list_domain_tables()
                db_url_display = _mask_db_url(get_database_url() or "")
                if db_url_display:
                    st.caption(f"Connected to `{db_url_display}`")

                # The database is shared across sessions, so it also holds tables from
                # unrelated past projects. This project's tables are the ones registered
                # to this session plus any table named in the current schema.
                existing = {t.lower(): t for t in all_tables}
                project_names = list(list_project_tables(st.session_state.session_id))
                _req_browse = st.session_state.get("requirements_for_review")
                if isinstance(_req_browse, dict) and "error" not in _req_browse:
                    project_names += [
                        t.get("name", "")
                        for t in ((_req_browse.get("database_schema") or {}).get("tables") or [])
                        if isinstance(t, dict) and t.get("name")
                    ]

                seen = set()
                project_tables = []
                for name in project_names:
                    real = existing.get(str(name).lower())
                    if real and real not in seen:
                        seen.add(real)
                        project_tables.append(real)

                other_count = len(all_tables) - len(project_tables)
                show_all = st.checkbox(
                    f"Show all tables in the database ({other_count} unrelated to this project)",
                    key="db_browse_show_all",
                    help="The database file is shared across sessions, so it also contains "
                         "tables left over from earlier projects.",
                    disabled=other_count <= 0,
                )
                live_tables = all_tables if show_all else project_tables

                if not live_tables:
                    if project_tables or not all_tables:
                        st.info(
                            "No tables for this project yet. Use **Create Dataset** or "
                            "**Import Existing Dataset** above to add one."
                        )
                    else:
                        st.info(
                            f"No tables belong to this project yet. The database holds "
                            f"{other_count} table(s) from earlier sessions — tick the box "
                            "above to view them."
                        )
                else:
                    overview = []
                    for tname in live_tables:
                        n = count_rows(tname)
                        overview.append({
                            "Table": tname,
                            "Columns": len(get_table_columns(tname)),
                            "Rows": n if n >= 0 else "—",
                        })

                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Tables shown", len(live_tables))
                    with c2:
                        st.metric(
                            "Total rows",
                            sum(r["Rows"] for r in overview if isinstance(r["Rows"], int)),
                        )

                    if PANDAS_AVAILABLE:
                        st.dataframe(pd.DataFrame(overview), width="stretch", hide_index=True)
                    else:
                        for r in overview:
                            st.write(f"• **{r['Table']}** — {r['Columns']} cols, {r['Rows']} rows")

                    st.markdown("### 🔍 Inspect a table")
                    picked = st.selectbox("Table", live_tables, key="db_browse_table")
                    if picked:
                        browse_limit = st.slider(
                            "Rows to show", 10, 500, 50, step=10, key="db_browse_limit"
                        )
                        picked_rows = fetch_objects(picked, limit=int(browse_limit))
                        if not picked_rows:
                            st.info(f"Table `{picked}` is empty.")
                        elif PANDAS_AVAILABLE:
                            df_browse = pd.DataFrame(picked_rows)
                            for field in ("password", "pwd", "secret", "token", "key", "api_key", "access_token"):
                                if field in df_browse.columns:
                                    df_browse[field] = df_browse[field].astype(str).apply(lambda _: "••••••")
                            st.dataframe(df_browse, width="stretch", hide_index=True)
                            st.download_button(
                                f"📥 Download {picked} as CSV",
                                data=df_browse.to_csv(index=False),
                                file_name=f"{picked}.csv",
                                mime="text/csv",
                                width="stretch",
                                key=f"db_browse_dl_{picked}",
                            )
                        else:
                            st.json(picked_rows[:20])

            st.divider()

            # Show actual database rows (objects) from requirements
            req = st.session_state.get("requirements_for_review")
            if req and isinstance(req, dict) and "error" not in req:
                db = req.get("database_schema", {}) or {}
                schema_tables = db.get("tables", []) if isinstance(db.get("tables"), list) else []
                
                if schema_tables and DB_AVAILABLE:
                    st.markdown("## 📦 Objects (Table Rows)")
                    
                    # Get table names from requirements schema
                    table_names = [t.get("name", "") for t in schema_tables if t.get("name")]
                    
                    if table_names:
                        # Display each table from requirements
                        for table_info in schema_tables:
                            table_name = table_info.get("name", "")
                            if not table_name:
                                continue
                            
                            # Auto-create table if it doesn't exist
                            if not table_exists(table_name):
                                columns = table_info.get("columns", [])
                                if columns:
                                    with st.spinner(f"Creating table `{table_name}` from schema..."):
                                        created, error_msg = create_table_from_schema(table_name, columns)
                                        if created:
                                            st.success(f"✅ Table `{table_name}` created successfully!")
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Could not create table `{table_name}`")
                                            st.code(f"Error: {error_msg}", language="text")
                                            # Show schema for debugging
                                            with st.expander(f"🔍 Debug: Schema for {table_name}"):
                                                st.json({"name": table_name, "columns": columns})
                                            continue
                                else:
                                    st.warning(f"⚠️ Table `{table_name}` has no column definitions. Skipping.")
                                    continue
                            
                            # Check if table exists in database (after creation attempt)
                            tables_in_db = list_domain_tables()
                            if table_name.lower() in [t.lower() for t in tables_in_db]:
                                st.markdown(f"### 📊 {table_name}")
                                
                                limit = st.slider(
                                    f"Rows to show ({table_name}):",
                                    10, 200, 50,
                                    key=f"db_row_limit_{table_name}"
                                )
                                
                                rows = fetch_objects(table_name, limit=limit)
                                if rows:
                                    if PANDAS_AVAILABLE:
                                        df = pd.DataFrame(rows)
                                        
                                        # Mask sensitive fields
                                        sensitive_fields = ["password", "pwd", "secret", "token", "key", "api_key", "access_token"]
                                        for field in sensitive_fields:
                                            if field in df.columns:
                                                df[field] = df[field].astype(str).apply(lambda _: "••••••")
                                        
                                        st.dataframe(df, width="stretch", hide_index=True)
                                        
                                        # Download CSV
                                        csv = df.to_csv(index=False)
                                        st.download_button(
                                            f"📥 Download {table_name} as CSV",
                                            data=csv,
                                            file_name=f"{table_name}_data.csv",
                                            mime="text/csv",
                                            width="stretch",
                                            key=f"download_{table_name}"
                                        )
                                    else:
                                        # Fallback: show as JSON
                                        st.json(rows[:20])
                                else:
                                    st.info(f"Table `{table_name}` exists but is empty.")
                                
                                # Form to add new row (generic for any table)
                                st.markdown(f"### ➕ Add row to {table_name}")
                                table_columns = get_table_columns(table_name)
                                # Remove 'id' if it's auto-increment
                                editable_columns = [col for col in table_columns if col.lower() != "id"]
                                
                                if editable_columns:
                                    with st.form(f"add_row_{table_name}", clear_on_submit=True):
                                        form_data = {}
                                        for col in editable_columns:
                                            # Use password input for password fields
                                            if "password" in col.lower() or "pwd" in col.lower():
                                                form_data[col] = st.text_input(col, type="password", key=f"input_{table_name}_{col}")
                                            else:
                                                form_data[col] = st.text_input(col, key=f"input_{table_name}_{col}")
                                        
                                        submitted = st.form_submit_button(f"Insert Row into {table_name}", width="stretch")
                                        if submitted:
                                            # Check if all required fields are filled
                                            if all(form_data.values()):
                                                try:
                                                    insert_object(table_name, form_data)
                                                    st.success(f"Row inserted into {table_name} ✅")
                                                    st.rerun()
                                                except Exception as e:
                                                    st.error(f"Error inserting row: {str(e)}")
                                            else:
                                                st.error("Please fill all fields.")
                                
                                st.divider()
                            else:
                                # Table doesn't exist in database yet
                                st.info(f"Table `{table_name}` is defined in schema but doesn't exist in database yet.")
                    else:
                        st.info("No tables found in requirements schema.")
                elif schema_tables:
                    st.info("Database helper not available. Please check db.py is accessible.")
            
            st.markdown("## 🗄️ Database Schema")

            req = st.session_state.get("requirements_for_review")
            if not req or not isinstance(req, dict) or "error" in req:
                st.info("Generate requirements first (Chat → Turn into structured Requirements).")
            else:
                db = req.get("database_schema", {}) or {}
                tables = db.get("tables", []) if isinstance(db.get("tables"), list) else []

                # Quick summary
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Tables", len(tables))
                with c2:
                    st.metric("Assumptions", len(db.get("assumptions", []) or []))

                if db.get("assumptions"):
                    with st.expander("📝 Assumptions", expanded=False):
                        for a in db["assumptions"]:
                            st.write(f"• {a}")

                if not tables:
                    st.warning("No tables were generated. Try regenerating requirements with more details.")
                else:
                    st.divider()
                    
                    # Unified table view: All columns from all tables in one table
                    if PANDAS_AVAILABLE:
                        all_columns_rows = []
                        all_relationships_rows = []
                        
                        for t in tables:
                            table_name = t.get("name", "table")
                            cols = t.get("columns", []) if isinstance(t.get("columns"), list) else []
                            rels = t.get("relationships", []) if isinstance(t.get("relationships"), list) else []
                            
                            # Collect all columns
                            for c in cols:
                                all_columns_rows.append({
                                    "Table": table_name,
                                    "Column": c.get("name", ""),
                                    "Type": c.get("type", ""),
                                    "PK": "✅" if c.get("pk") else "",
                                    "Nullable": "✅" if c.get("nullable") else "",
                                    "Unique": "✅" if c.get("unique") else "",
                                    "Default": c.get("default", "") or "—",
                                    "Notes": c.get("notes", "") or "—",
                                })
                            
                            # Collect all relationships
                            for r in rels:
                                all_relationships_rows.append({
                                    "From Table": table_name,
                                    "To Table": r.get("to_table", ""),
                                    "Type": r.get("type", ""),
                                    "FK": r.get("fk", "") or "—",
                                    "References": r.get("ref", "") or "—",
                                    "Notes": r.get("notes", "") or "—",
                                })
                        
                        # Show unified columns table
                        if all_columns_rows:
                            st.markdown("### 📊 All Columns (Unified View)")
                            df_columns = pd.DataFrame(all_columns_rows)
                            # Force column order
                            column_order = ["Table", "Column", "Type", "PK", "Nullable", "Unique", "Default", "Notes"]
                            df_columns = df_columns[[col for col in column_order if col in df_columns.columns]]
                            st.dataframe(df_columns, width="stretch", hide_index=True)
                            
                            # Download button for CSV
                            csv = df_columns.to_csv(index=False)
                            st.download_button(
                                "📥 Download Columns as CSV",
                                data=csv,
                                file_name="database_columns.csv",
                                mime="text/csv",
                                width="stretch"
                            )
                            st.divider()
                        
                        # Show unified relationships table
                        if all_relationships_rows:
                            st.markdown("### 🔗 All Relationships (Unified View)")
                            df_relationships = pd.DataFrame(all_relationships_rows)
                            # Force column order
                            rel_order = ["From Table", "To Table", "Type", "FK", "References", "Notes"]
                            df_relationships = df_relationships[[col for col in rel_order if col in df_relationships.columns]]
                            st.dataframe(df_relationships, width="stretch", hide_index=True)
                            
                            # Download button for CSV
                            csv = df_relationships.to_csv(index=False)
                            st.download_button(
                                "📥 Download Relationships as CSV",
                                data=csv,
                                file_name="database_relationships.csv",
                                mime="text/csv",
                                width="stretch"
                            )
                            st.divider()
                        
                        # Show sample data rows if available
                        all_sample_rows = []
                        for t in tables:
                            table_name = t.get("name", "table")
                            # Check for sample_rows, sample_data, or data fields
                            sample_data = (
                                t.get("sample_rows") or 
                                t.get("sample_data") or 
                                t.get("data") or 
                                t.get("rows") or
                                []
                            )
                            
                            if isinstance(sample_data, list) and len(sample_data) > 0:
                                for row in sample_data:
                                    if isinstance(row, dict):
                                        row_with_table = {"Table": table_name}
                                        row_with_table.update(row)
                                        all_sample_rows.append(row_with_table)
                        
                        # Display sample data if available
                        if all_sample_rows:
                            st.markdown("### 📋 Sample Data (Table Records)")
                            df_sample = pd.DataFrame(all_sample_rows)
                            
                            # Mask sensitive fields (password, pwd, secret, token, key, etc.)
                            sensitive_fields = ["password", "pwd", "secret", "token", "key", "api_key", "access_token"]
                            for field in sensitive_fields:
                                if field in df_sample.columns:
                                    df_sample[field] = df_sample[field].astype(str).apply(lambda _: "••••••")
                            
                            # Reorder: Table first, then other columns
                            if "Table" in df_sample.columns:
                                other_cols = [col for col in df_sample.columns if col != "Table"]
                                df_sample = df_sample[["Table"] + other_cols]
                            
                            st.dataframe(df_sample, width="stretch", hide_index=True)
                            
                            # Download button for CSV
                            csv = df_sample.to_csv(index=False)
                            st.download_button(
                                "📥 Download Sample Data as CSV",
                                data=csv,
                                file_name="database_sample_data.csv",
                                mime="text/csv",
                                width="stretch"
                            )
                            st.divider()
                        
                        # Try to query actual database tables if they exist
                        try:
                            from db import db_session
                            from sqlalchemy import text, inspect
                            
                            # Get list of actual tables from database
                            with db_session() as s:
                                inspector = inspect(s.bind)
                                actual_tables = inspector.get_table_names()
                            
                            # Filter to only tables that match our schema
                            schema_table_names = {t.get("name", "").lower() for t in tables if t.get("name")}
                            matching_tables = [t for t in actual_tables if t.lower() in schema_table_names]
                            
                            if matching_tables:
                                st.markdown("### 💾 Actual Database Records")
                                st.info(f"Found {len(matching_tables)} matching table(s) in database. Showing sample records...")
                                
                                selected_table = st.selectbox(
                                    "Select table to view:",
                                    options=matching_tables,
                                    key="db_table_selector"
                                )
                                
                                if selected_table:
                                    limit = st.slider("Rows to show:", 10, 100, 50, key="db_row_limit")
                                    
                                    try:
                                        with db_session() as s:
                                            # Use parameterized query for safety
                                            result = s.execute(
                                                text(f'SELECT * FROM "{selected_table}" LIMIT :limit'),
                                                {"limit": limit}
                                            )
                                            rows = [dict(row._mapping) for row in result]
                                        
                                        if rows:
                                            df_actual = pd.DataFrame(rows)
                                            
                                            # Mask sensitive fields
                                            for field in sensitive_fields:
                                                if field in df_actual.columns:
                                                    df_actual[field] = df_actual[field].astype(str).apply(lambda _: "••••••")
                                            
                                            st.dataframe(df_actual, width="stretch", hide_index=True)
                                            
                                            # Download button
                                            csv = df_actual.to_csv(index=False)
                                            st.download_button(
                                                f"📥 Download {selected_table} as CSV",
                                                data=csv,
                                                file_name=f"{selected_table}_data.csv",
                                                mime="text/csv",
                                                width="stretch"
                                            )
                                        else:
                                            st.info(f"Table '{selected_table}' is empty.")
                                    except Exception as e:
                                        st.warning(f"Could not query table '{selected_table}': {str(e)}")
                        except (ImportError, Exception):
                            # SQLAlchemy or database not available, skip actual table querying
                            pass
                    
                    # Detailed view: Each table in its own expander
                    st.markdown("### 📋 Detailed Table View")
                    for t in tables:
                        name = t.get("name", "table")
                        purpose = t.get("purpose", "—")
                        cols = t.get("columns", []) if isinstance(t.get("columns"), list) else []
                        rels = t.get("relationships", []) if isinstance(t.get("relationships"), list) else []
                        idxs = t.get("indexes", []) if isinstance(t.get("indexes"), list) else []

                        with st.expander(f"🧩 {name} — {purpose}", expanded=False):
                            # Columns table
                            if cols:
                                if PANDAS_AVAILABLE:
                                    rows = []
                                    for c in cols:
                                        rows.append({
                                            "Column": c.get("name", ""),
                                            "Type": c.get("type", ""),
                                            "PK": "✅" if c.get("pk") else "",
                                            "Nullable": "✅" if c.get("nullable") else "",
                                            "Unique": "✅" if c.get("unique") else "",
                                            "Default": c.get("default", "") or "—",
                                            "Notes": c.get("notes", "") or "—",
                                        })
                                    df = pd.DataFrame(rows)
                                    # Force column order
                                    col_order = ["Column", "Type", "PK", "Nullable", "Unique", "Default", "Notes"]
                                    df = df[[col for col in col_order if col in df.columns]]
                                    st.dataframe(df, width="stretch", hide_index=True)
                                else:
                                    # Fallback if pandas not available
                                    rows = []
                                    for c in cols:
                                        rows.append({
                                            "Column": c.get("name", ""),
                                            "Type": c.get("type", ""),
                                            "PK": "✅" if c.get("pk") else "",
                                            "Nullable": "✅" if c.get("nullable") else "",
                                            "Unique": "✅" if c.get("unique") else "",
                                            "Default": c.get("default", "") or "—",
                                            "Notes": c.get("notes", "") or "—",
                                        })
                                    st.dataframe(rows, width="stretch", hide_index=True)
                            else:
                                st.info("No columns defined.")

                            # Sample data rows for this table
                            sample_data = (
                                t.get("sample_rows") or 
                                t.get("sample_data") or 
                                t.get("data") or 
                                t.get("rows") or
                                []
                            )
                            
                            if isinstance(sample_data, list) and len(sample_data) > 0:
                                st.markdown("**📋 Sample Data**")
                                if PANDAS_AVAILABLE:
                                    df_sample = pd.DataFrame(sample_data)
                                    
                                    # Mask sensitive fields
                                    sensitive_fields = ["password", "pwd", "secret", "token", "key", "api_key", "access_token"]
                                    for field in sensitive_fields:
                                        if field in df_sample.columns:
                                            df_sample[field] = df_sample[field].astype(str).apply(lambda _: "••••••")
                                    
                                    st.dataframe(df_sample, width="stretch", hide_index=True)
                                else:
                                    # Fallback: show as JSON
                                    st.json(sample_data[:10])  # Limit to 10 rows

                            # Indexes
                            if idxs:
                                st.markdown("**📑 Indexes**")
                                for ix in idxs:
                                    st.write(f"• `{ix}`")

                            # Relationships as table
                            if rels:
                                st.markdown("**🔗 Relationships**")
                                if PANDAS_AVAILABLE:
                                    rel_rows = []
                                    for r in rels:
                                        rel_rows.append({
                                            "To Table": r.get("to_table", ""),
                                            "Type": r.get("type", ""),
                                            "FK": r.get("fk", "") or "—",
                                            "References": r.get("ref", "") or "—",
                                            "Notes": r.get("notes", "") or "—",
                                        })
                                    df_rel = pd.DataFrame(rel_rows)
                                    if not df_rel.empty:
                                        st.dataframe(df_rel, width="stretch", hide_index=True)
                                else:
                                    # Fallback: show as list
                                    for r in rels:
                                        rel_type = r.get("type", "")
                                        to_table = r.get("to_table", "")
                                        fk = r.get("fk", "")
                                        ref = r.get("ref", "")
                                        notes = r.get("notes", "")
                                        
                                        st.write(
                                            f"• `{name}` → `{to_table}` "
                                            f"({rel_type})"
                                        )
                                        if fk and ref:
                                            st.caption(f"  FK: `{fk}` → `{ref}`")
                                        if notes:
                                            st.caption(f"  {notes}")

                # Provide SQL export
                st.divider()
                if tables:
                    col_json, col_pg, col_sqlite = st.columns(3)
                    with col_json:
                        st.download_button(
                            "📥 Download Schema JSON",
                            data=json.dumps(db, indent=2),
                            file_name="database_schema.json",
                            mime="application/json",
                            use_container_width=True,
                        )
                    with col_pg:
                        try:
                            from agents.database_agent import DatabaseAgent as _DBA
                            _dba = _DBA.__new__(_DBA)
                            pg_ddl = _dba.generate_ddl(db, dialect="postgresql")
                        except Exception:
                            pg_ddl = "-- DDL generation failed"
                        st.download_button(
                            "📥 Download PostgreSQL DDL",
                            data=pg_ddl,
                            file_name="schema_postgresql.sql",
                            mime="text/plain",
                            use_container_width=True,
                        )
                    with col_sqlite:
                        try:
                            from agents.database_agent import DatabaseAgent as _DBA
                            _dba = _DBA.__new__(_DBA)
                            sqlite_ddl = _dba.generate_ddl(db, dialect="sqlite")
                        except Exception:
                            sqlite_ddl = "-- DDL generation failed"
                        st.download_button(
                            "📥 Download SQLite DDL",
                            data=sqlite_ddl,
                            file_name="schema_sqlite.sql",
                            mime="text/plain",
                            use_container_width=True,
                        )

        # TAB 5: PLANNING
        with tab_planning:
            st.session_state.active_tab = 4
            st.markdown("## 📅 Planning Manager Agent")
            st.info(
                "Creates an executable MVP (Minimum Viable Product) plan (tasks, risks) "
                "from your requirements and design. Generate requirements and design first."
            )
            req = st.session_state.get("requirements_for_review")
            design = st.session_state.get("interface_design")

            if req and isinstance(req, dict) and "error" not in req:
                project_name = req.get("project_name", "Project")
                features = req.get("features", [])
                st.markdown("### Inputs")
                st.write(f"**{project_name}** — {len(features)} feature(s) | Design: {'✅' if design and isinstance(design, dict) else '—'}")

                if st.button("📅 Generate plan", type="primary", key="gen_planning"):
                    if analyzer and analyzer.api_key:
                        planning_agent = PlanningManagerAgent(analyzer.api_key)
                        with st.spinner("Generating plan…"):
                            plan = planning_agent.generate_cycle_plan(req, design)
                        if isinstance(plan, dict) and "error" not in plan:
                            st.session_state.cycle_plan = plan
                            try:
                                save_artifact(st.session_state.session_id, "plan", plan)
                            except Exception:
                                pass
                            st.rerun()
                        else:
                            st.error(plan.get("error", "Failed to generate plan."))
                            if plan.get("solution"):
                                st.info(plan["solution"])
                    else:
                        st.error("Anthropic API key not found.")

                plan = st.session_state.get("cycle_plan")
                if plan and isinstance(plan, dict) and "error" not in plan:
                    st.divider()
                    st.markdown("### Development plan")
                    st.write(f"**{plan.get('plan_name', 'Plan')}**")
                    if not (plan.get("tasks") or plan.get("risks")):
                        st.warning("Agent returned minimal content. Try generating again or add more detail to requirements.")
                    if plan.get("tasks"):
                        if PANDAS_AVAILABLE:
                            task_rows = []
                            for t in plan["tasks"]:
                                task_rows.append({
                                    "ID": t.get("id", ""),
                                    "Title": t.get("title", ""),
                                    "Agent": t.get("assigned_agent", ""),
                                    "Priority": t.get("priority", ""),
                                })
                            st.dataframe(pd.DataFrame(task_rows), width="stretch", hide_index=True)
                        else:
                            for t in plan["tasks"]:
                                st.write(f"• **{t.get('id', '')}** {t.get('title', '')} — {t.get('priority', '')}")
                    if plan.get("risks"):
                        st.markdown("**Risks**")
                        for r in plan["risks"]:
                            st.write(f"• ⚠️ {r}")
                    with st.expander("View full JSON"):
                        st.json(plan)

                    st.download_button(
                        "📥 Download Plan JSON",
                        data=json.dumps(plan, indent=2),
                        file_name="plan.json",
                        mime="application/json",
                        width="stretch",
                    )
            else:
                st.info("Generate and confirm requirements first (Chat → Review) to enable planning.")

        # TAB 6: QUALITY
        with tab_quality:
            st.session_state.active_tab = 5
            st.markdown("## ✅ Quality Manager Agent")
            st.info(
                "Evaluates requirements, design, and database schema. Produces a gate decision "
                "(PASS/FAIL), checklist, issues, and required fixes."
            )
            req = st.session_state.get("requirements_for_review")
            design = st.session_state.get("interface_design")
            db_schema = (req or {}).get("database_schema") if req and isinstance(req, dict) else None

            if req and isinstance(req, dict) and "error" not in req:
                st.markdown("### Inputs")
                st.write(f"Requirements ✅ | Design: {'✅' if design and isinstance(design, dict) else '—'} | DB schema: {'✅' if db_schema else '—'}")

                if st.button("✅ Generate quality report", type="primary", key="gen_quality"):
                    if analyzer and analyzer.api_key:
                        quality_agent = QualityManagerAgent(analyzer.api_key)
                        with st.spinner("Generating quality report…"):
                            report = quality_agent.generate_quality_report(req, design, db_schema)
                        if isinstance(report, dict) and "error" not in report:
                            st.session_state.quality_report = report
                            try:
                                save_artifact(st.session_state.session_id, "quality_report", report)
                            except Exception:
                                pass
                            st.rerun()
                        else:
                            st.error(report.get("error", "Failed to generate report."))
                            if report.get("solution"):
                                st.info(report["solution"])
                    else:
                        st.error("Anthropic API key not found.")

                report = st.session_state.get("quality_report")
                if report and isinstance(report, dict) and "error" not in report:
                    st.divider()
                    st.markdown("### Quality assessment report")
                    if not report.get("checklist") and not report.get("issues") and not report.get("recommendations"):
                        st.warning("Agent returned minimal content. Try generating again.")
                    decision = report.get("gate_decision", "")
                    if decision.upper() == "PASS":
                        st.success("✅ Gate decision: PASS")
                    else:
                        st.error(f"❌ Gate decision: {decision}")

                    if report.get("checklist"):
                        st.markdown("**Checklist**")
                        _checklist_descriptions = {
                            "requirements_complete": "Requirements document is complete and covers scope.",
                            "design_consistent": "Design is consistent with requirements and architecture.",
                            "db_matches_requirements": "Database schema and entities match stated requirements.",
                            "nfr_defined": "Non-functional requirements are defined (performance, security, etc.).",
                        }
                        for k, v in report["checklist"].items():
                            desc = _checklist_descriptions.get(k, "")
                            if desc:
                                st.write(f"• **{k}**: {'✅' if v else '❌'} — *{desc}*")
                            else:
                                st.write(f"• {k}: {'✅' if v else '❌'}")

                    if report.get("issues") and isinstance(report["issues"], list):
                        st.markdown("**Issues**")
                        for iss in report["issues"]:
                            if isinstance(iss, dict):
                                sev = (iss.get("severity") or "—").upper()
                                st.write(f"• [{sev}] {iss.get('item', '')}: {iss.get('message', '')}")

                    if report.get("required_fixes"):
                        st.markdown("**Required fixes**")
                        for f in report["required_fixes"]:
                            st.write(f"• {f}")

                    if report.get("recommendations"):
                        st.markdown("**Recommendations**")
                        for r in report["recommendations"]:
                            st.write(f"• {r}")

                    with st.expander("View full JSON"):
                        st.json(report)

                    st.download_button(
                        "📥 Download Quality Report JSON",
                        data=json.dumps(report, indent=2),
                        file_name="quality_report.json",
                        mime="application/json",
                        width="stretch",
                    )
            else:
                st.info("Generate requirements first (Chat → Review) to enable quality checks.")

        # TAB 7: SUPPORT
        with tab_support:
            st.session_state.active_tab = 6
            st.markdown("## 🛟 Support Manager Agent")
            with st.expander("📖 App documentation", expanded=False):
                st.markdown("""
**LLM Requirements Analyzer** helps you turn natural language into structured requirements, design, and project governance using AI agents (TSPi-style).

**Tabs and workflow**
- **Chat** — Describe your project in plain language; the Requirements Agent turns it into structured requirements.
- **Review** — Inspect and edit requirements, then lock them for downstream use.
- **Design** — Generate interface/screen design from requirements; edit and lock the design.
- **Database** — Derive or edit database schema from requirements.
- **Planning** — Create cycle/sprint plans and briefs from requirements and design.
- **Quality** — Run the Quality Manager Agent to gate-check requirements, design, and DB consistency; view checklist, issues, and recommendations.
- **Support** — Generate a support governance package (baselines, app documentation, glossary) and read this documentation.

**How to use**
1. Set your **Anthropic API key** (`ANTHROPIC_API_KEY` in `.env`).
2. In **Chat**, describe the app; use **Review** to finalize requirements.
3. Optionally run **Design**, **Database**, and **Planning**; then **Quality** to validate.
4. In **Support**, generate the governance package when requirements (and optionally design/DB/plan) are ready.
                """)
            st.info(
                "Produces support governance artifacts: documentation of your app, "
                "baseline artifacts, and glossary."
            )
            req = st.session_state.get("requirements_for_review")
            design = st.session_state.get("interface_design")
            db_schema = (req or {}).get("database_schema") if req and isinstance(req, dict) else None
            cycle_plan = st.session_state.get("cycle_plan")

            if req and isinstance(req, dict) and "error" not in req:
                st.markdown("### Inputs")
                st.write(f"Requirements ✅ | Design: {'✅' if design else '—'} | DB: {'✅' if db_schema else '—'} | MVP plan: {'✅' if cycle_plan else '—'}")

                if st.button("🛟 Generate support package", type="primary", key="gen_support"):
                    if analyzer and analyzer.api_key:
                        support_agent = SupportManagerAgent(analyzer.api_key)
                        with st.spinner("Generating support governance package…"):
                            pkg = support_agent.generate_support_package(req, design, db_schema, cycle_plan)
                        if isinstance(pkg, dict) and "error" not in pkg:
                            st.session_state.support_governance = pkg
                            try:
                                save_artifact(st.session_state.session_id, "support_governance", pkg)
                            except Exception:
                                pass
                            st.rerun()
                        else:
                            st.error(pkg.get("error", "Failed to generate package."))
                            if pkg.get("solution"):
                                st.info(pkg["solution"])
                    else:
                        st.error("Anthropic API key not found.")

                pkg = st.session_state.get("support_governance")
                if pkg and isinstance(pkg, dict) and "error" not in pkg:
                    st.divider()
                    st.markdown("### Support governance package")
                    if not (pkg.get("app_documentation") or pkg.get("baseline_artifacts") or pkg.get("glossary")):
                        st.warning("Agent returned minimal content. Try generating again or ensure requirements/design are complete.")

                    if pkg.get("app_documentation"):
                        st.markdown("**Documentation (your app)**")
                        st.markdown(pkg.get("app_documentation", ""))

                    if pkg.get("baseline_artifacts"):
                        st.markdown("**Baseline artifacts**")
                        for a in pkg["baseline_artifacts"]:
                            st.write(f"• {a}")

                    if pkg.get("glossary") and isinstance(pkg["glossary"], dict):
                        st.markdown("**Glossary**")
                        for term, defn in pkg["glossary"].items():
                            st.write(f"• **{term}**: {defn}")

                    with st.expander("View full JSON"):
                        st.json(pkg)

                    st.download_button(
                        "📥 Download Support Package JSON",
                        data=json.dumps(pkg, indent=2),
                        file_name="support_package.json",
                        mime="application/json",
                        width="stretch",
                    )
            else:
                st.info("Generate requirements first (Chat → Review) to enable support package generation.")


if __name__ == "__main__":
    main()
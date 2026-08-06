import errno
import json
import time
from typing import Dict, Any

import requests
import streamlit as st

from core.constants import (
    DEFAULT_MODEL, MAX_RETRIES, MAX_ITERATIONS, DESIGN_MAX_EXECUTION_TIME,
    EXPONENTIAL_BACKOFF_BASE, ERROR_NO_API_KEY, SOLUTION_ADD_API_KEY,
    MSG_CONNECTION_ISSUE, MSG_RETRYING,
)
from core.models import PYDANTIC_AVAILABLE, DesignOutputModel
from core.llm import CREWAI_AVAILABLE, Agent, Task, Crew, Process, get_llm
from core.utils import parse_json_from_text, extract_crewai_output, create_error_response
from utils.io_suppression import suppress_stderr, suppress_io


def _snip_crew_error(message: str, limit: int = 2800) -> str:
    """CrewAI often prefixes exceptions with the full task description; keep the tail for debugging."""
    message = (message or "").strip()
    if len(message) <= limit:
        return message
    return "…(truncated)\n" + message[-limit:]


def _retryable_transport_error(exc: BaseException) -> bool:
    """True for likely network/transport failures — avoid matching 'connection' inside requirement text."""
    et = type(exc).__name__
    if et in (
        "ConnectionError",
        "TimeoutError",
        "ConnectTimeout",
        "ReadTimeout",
        "SSLError",
        "APIConnectionError",
        "NameResolutionError",
        "gaierror",
    ):
        return True
    if isinstance(exc, OSError):
        err = getattr(exc, "errno", None)
        if err in (
            errno.ECONNREFUSED,
            errno.ETIMEDOUT,
            errno.ENETUNREACH,
            errno.EHOSTUNREACH,
            errno.ECONNRESET,
        ):
            return True
    try:
        import httpx

        if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout)):
            return True
    except Exception:
        pass
    try:
        if isinstance(
            exc,
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.ConnectTimeout,
            ),
        ):
            return True
    except Exception:
        pass
    tail = str(exc).lower()[-2000:]
    return any(
        needle in tail
        for needle in (
            "connection refused",
            "connection reset",
            "temporary failure in name resolution",
            "name or service not known",
            "getaddrinfo failed",
            "sslerror",
            "tlsv",
            "timed out",
            "timeout expired",
            "read timed out",
            "connect timed out",
            "could not resolve host",
            "network is unreachable",
        )
    )


class DesignGenerationAgent:
    """Agent responsible for generating UI/UX design specifications using CrewAI."""

    @classmethod
    def _build_task_prompt(cls, requirements_summary: str) -> str:
        """Build the full task description, injecting requirements into the template."""
        return f"""You are a Senior Product Designer and UX Architect working at a high-quality software company.

Your responsibility is to design a **real, production-ready user interface** based on structured software requirements.
The output must resemble a **modern iOS / Android application** that users can realistically tap, navigate, and interact with — not a wireframe, concept sketch, or decorative mockup.

You are designing for:
• usability
• clarity
• realistic user behavior
• interactive live preview

━━━━━━━━━━━━━━━━━━━━━━
DESIGN MINDSET (MANDATORY)
━━━━━━━━━━━━━━━━━━━━━━
Think like a real product team shipping a usable app:

• Every screen must have a clear purpose
• Every component must justify its existence
• Primary actions must be visually dominant
• Secondary actions must be discoverable but subtle
• Layouts must feel intentional, spaced, and balanced
• UI must feel clickable and touch-friendly

Avoid:
✗ decorative filler
✗ vague components
✗ empty screens
✗ generic labels
✗ wireframe-style layouts

━━━━━━━━━━━━━━━━━━━━━━
VISUAL & UX PRINCIPLES
━━━━━━━━━━━━━━━━━━━━━━
• Use strong visual hierarchy (Primary → Secondary → Supporting)
• Prefer cards, sections, lists, and containers over flat layouts
• Use rounded corners, soft shadows, and subtle borders
• Buttons must look tappable
• Inputs must look editable
• Cards must suggest navigation when appropriate
• Use whitespace intentionally — never clutter
• Gradients are allowed ONLY for headers or primary actions (subtle only)

━━━━━━━━━━━━━━━━━━━━━━
INTERACTION & NAVIGATION
━━━━━━━━━━━━━━━━━━━━━━
Design the UI assuming it will be **fully interactive**:

• Screens must connect logically
• Navigation patterns must feel native (top bar, bottom nav, back)
• Buttons should clearly imply navigation or actions
• Cards may navigate to detail screens
• Inputs must include placeholders
• Lists must feel scannable
• Use realistic user flows (Home → Details → Action → Result)

━━━━━━━━━━━━━━━━━━━━━━
ICONOGRAPHY RULES (STRICT)
━━━━━━━━━━━━━━━━━━━━━━
• DO NOT use emojis
• Use short, consistent icon names ONLY
• Allowed icons:
  home, search, user, settings, bell, bookmark,
  heart, filter, plus, edit, trash, share
• Icons must match interaction purpose
• Be consistent across screens

━━━━━━━━━━━━━━━━━━━━━━
COMPONENT EXPECTATIONS
━━━━━━━━━━━━━━━━━━━━━━
Headers: clearly communicate screen purpose, include navigation when appropriate
Buttons: clearly labeled, obvious hierarchy (primary vs secondary), always feel clickable
Inputs: include placeholders, look editable, sized for touch
Cards: group related content, often navigable, may include images
Lists: clean, readable, scannable

━━━━━━━━━━━━━━━━━━━━━━
STRUCTURED REQUIREMENTS TO DESIGN FROM
━━━━━━━━━━━━━━━━━━━━━━
{requirements_summary}

━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT (STRICT)
━━━━━━━━━━━━━━━━━━━━━━
Return ONLY valid JSON using EXACTLY this structure:

{{
  "design_overview": "Brief professional summary of the design approach",
  "color_scheme": {{
    "primary": "#hex", "secondary": "#hex", "accent": "#hex",
    "surface": "#hex", "background": "#hex", "text_primary": "#hex",
    "error": "#hex", "success": "#hex", "warning": "#hex"
  }},
  "typography": {{
    "font_family": "...", "heading_font": "...", "body_font": "...",
    "sizes": {{...}}, "weights": {{...}}
  }},
  "navigation": {{"type": "top/bottom/side", "items": ["Home", "Search", "Profile"]}},
  "screens": [
    {{
      "name": "...", "purpose": "...",
      "key_components": [
        {{
          "name": "...", "type": "Button/Input/Card/List/Image/Header/Footer/Modal/etc",
          "position": "top/center/bottom", "size": "small/medium/large",
          "styling": "Clear visual description", "interaction": "What happens when tapped",
          "icon": "home/search/user/etc", "has_image": true, "image_type": "product/user/background/content/none"
        }}
      ],
      "user_flow": "How users reach and leave this screen"
    }}
  ],
  "ui_components": [...],
  "responsive_design": "...",
  "accessibility": "...",
  "animations": "...",
  "icons": "..."
}}

━━━━━━━━━━━━━━━━━━━━━━
QUALITY BAR
━━━━━━━━━━━━━━━━━━━━━━
• Include 4–6 screens, each with 8–12 meaningful components
• Screens must feel connected and realistic
• UI must look suitable for startup demos, usability testing, and live clickable preview

FINAL RULES: Output ONLY JSON. Start with {{ and end with }}. No markdown, no explanations."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        # Initialize attributes to None first
        self.agent = None
        self.crew = None
        self.task = None
        self.api_base = "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        } if self.api_key else {}

        if not CREWAI_AVAILABLE:
            return
    
        llm = get_llm()
        
        design_prompt = """You are an expert UI/UX designer. Create detailed, practical UI/UX design specifications with realistic components, icons, and visual elements.

Return a JSON object with this EXACT structure:

{
    "design_overview": "Brief overview of the design approach and key principles",
    "color_scheme": {
        "primary": "#hexcolor",
        "secondary": "#hexcolor",
        "accent": "#hexcolor",
        "surface": "#hexcolor",
        "background": "#hexcolor",
        "text_primary": "#hexcolor",
        "error": "#hexcolor",
        "success": "#hexcolor",
        "warning": "#hexcolor"
    },
    "typography": {
        "font_family": "Primary font family name",
        "heading_font": "Font for headings",
        "body_font": "Font for body text",
        "sizes": {
            "xs": "0.75rem", "sm": "0.875rem", "base": "1rem",
            "lg": "1.125rem", "xl": "1.25rem", "2xl": "1.5rem"
        },
        "weights": {
            "normal": "400", "medium": "500", "semibold": "600", "bold": "700"
        }
    },
    "navigation": {
        "type": "top/bottom/side",
        "items": ["Home", "Features", "About", "Contact"]
    },
    "screens": [
        {
            "name": "Screen Name",
            "purpose": "What this screen is for",
            "key_components": [
                {
                    "name": "Component Name",
                    "type": "Button/Input/Card/List/Image/Search/Header/Footer/Modal/etc",
                    "position": "top/center/bottom",
                    "size": "small/medium/large",
                    "styling": "Description of styling",
                    "interaction": "What happens when user interacts",
                    "icon": "one of: home, search, user, settings, bell, bookmark, heart, filter, plus, edit, trash, share",
                    "has_image": true/false,
                    "image_type": "product/user/background/content/none"
                }
            ],
            "user_flow": "How users navigate to/from this screen"
        }
    ],
    "ui_components": [
        {
            "name": "Component Name",
            "type": "Button/Input/Card/List/Image/Search/Header/Footer/Modal/etc",
            "description": "What it does",
            "styling": "Visual details",
            "states": ["default", "hover", "active", "disabled"],
            "icon": "one of: home, search, user, settings, bell, bookmark, heart, filter, plus, edit, trash, share",
            "has_image": true/false
        }
    ],
    "responsive_design": "How the design adapts to different screen sizes",
    "accessibility": "Accessibility considerations and features",
    "animations": "Animation and transition details",
    "icons": "Icon system and usage guidelines"
}

CRITICAL REQUIREMENTS:
- Return ONLY the JSON object, nothing else.
- Include at least 4-6 screens with detailed components (8-12 components per screen).
- For each component, specify an "icon" field - DO NOT use emojis for icons. Use short icon names only (one of: home, search, user, settings, bell, bookmark, heart, filter, plus, edit, trash, share).
- Set "has_image": true for components that should display images (cards, product listings, profiles, etc.).
- Make components specific and actionable with realistic names and descriptions.
- Use hex color codes for all colors.
- Include diverse component types: buttons, cards, lists, inputs, search bars, headers, images, etc.

DEFAULT/COMMON FEATURES (MANDATORY - Always Include):
⚠️ CRITICAL: Even if the user doesn't explicitly mention these features, you MUST ALWAYS include them. These are standard features found in virtually all modern apps and should be part of every design.

1. NAVIGATION & HEADER (REQUIRED):
   - Navigation bar/header with: Home (icon: "home"), Search (icon: "search"), Notifications (icon: "bell"), Profile/User (icon: "user")
   - Bottom navigation bar for mobile apps (Home, Search, Favorites/Bookmarks, Profile) - use icons: "home", "search", "bookmark", "user"
   - Menu/Hamburger icon - use icon: "settings" or "filter"
   - Logo or app name in header
   - Back button/arrow for navigation history

2. SEARCH FUNCTIONALITY (REQUIRED):
   - Search bar component with search icon (icon: "search") - MUST appear in at least one screen
   - Search input field prominently placed (usually in header or as main component)
   - Search icon button
   - Search results screen or search functionality

3. USER PROFILE & SETTINGS (REQUIRED):
   - User profile section/icon (icon: "user") - MUST be accessible
   - Settings/Preferences screen (icon: "settings") - MUST be included as a screen
   - Account management options
   - User avatar/profile picture component
   - Logout/Sign out option

4. HOME/DASHBOARD SCREEN (REQUIRED):
   - Main landing/home screen as the primary entry point - MUST be the first screen
   - Dashboard with key information and quick actions
   - Content cards, featured items, or main content area
   - Welcome message or personalized greeting
   - Quick action buttons

5. COMMON UI COMPONENTS (Include in multiple screens):
   - Primary action buttons (with appropriate icons like "plus", "edit", "share", "trash")
   - Content cards (with images where appropriate - set has_image: true)
   - Lists with icons (navigation lists, menu items, item lists)
   - Input fields (forms, search, text entry)
   - Image placeholders/content images (use has_image: true for visual components)
   - Icons throughout: use icon names only (home, search, user, settings, bell, bookmark, heart, filter, plus, edit, trash, share) - DO NOT use emojis

6. ADDITIONAL STANDARD SCREENS (Include at least 2-3 of these):
   - Settings/Preferences screen (icon: "settings") - HIGHLY RECOMMENDED
   - Profile/Account screen (icon: "user") - HIGHLY RECOMMENDED
   - Search/Discovery screen (icon: "search")
   - Notifications screen (icon: "bell")
   - Favorites/Bookmarks screen (icon: "bookmark")
   - Help/Support screen (icon: "settings")

7. VISUAL ELEMENTS (Enhance realism):
   - Include images in cards, product listings, profiles (set has_image: true)
   - Use icons for all interactive elements
   - Add visual hierarchy with colors, spacing, and typography
   - Include placeholder images for content areas

REMEMBER: These default features should be integrated naturally into the design alongside the user's specific requirements. The design should feel complete, production-ready, and include all standard app features that users expect. Always include navigation, search, profile, and settings - these are non-negotiable.
"""
        
        # Create CrewAI agent (uses get_llm() from ui.main_ui, typically Anthropic)
        agent_kwargs = {
            "role": "UI/UX Design Specialist",
            "goal": "OUTPUT a complete JSON object with UI/UX design specifications including components, icons, images, and realistic app previews. You must produce the actual JSON output, not just think about it.",
            "backstory": """You are a senior product designer and UI/UX architect with extensive experience 
            creating production-ready, polished mobile applications. You think like a real product team, 
            prioritizing clarity, usability, and visual hierarchy. You excel at translating requirements 
            into detailed design specifications that resemble real iOS/Android apps, not wireframes. 
            Your designs are intentional, well-structured, and suitable for interactive prototypes. 
            You always produce complete, valid JSON outputs without unnecessary explanations.""",
            "verbose": False,
            "allow_delegation": False,
            "max_iter": MAX_ITERATIONS,
            # Design prompts are large; allow more time than the global default.
            "max_execution_time": DESIGN_MAX_EXECUTION_TIME,
        }
        if llm:
            agent_kwargs["llm"] = llm
        
        try:
            from utils.io_suppression import suppress_stderr
            with suppress_stderr():
                self.agent = Agent(**agent_kwargs)
        except Exception as e:
            # Only show error if it's not a telemetry-related error
            error_msg = str(e)
            if "signal" not in error_msg.lower() and "telemetry" not in error_msg.lower():
                st.error(f"Failed to create CrewAI DesignGenerationAgent: {e}")
            self.agent = None
            self.crew = None
            return
        
        # Create task — requirements are injected per call in generate_design()
        task_description = self._build_task_prompt("(requirements will be provided at generation time)")
        
        task_kwargs = {
            "description": task_description,
            "agent": self.agent,
            "expected_output": "A complete, valid JSON object starting with { and ending with }, containing design_overview, color_scheme, typography, navigation, screens (4-6 screens with 8-12 components each), ui_components, responsive_design, accessibility, animations, and icons. No markdown, no explanations, just pure JSON.",
        }
        
        # Add output_json if Pydantic is available
        if PYDANTIC_AVAILABLE and DesignOutputModel:
            task_kwargs["output_json"] = DesignOutputModel
        
        self.task = Task(**task_kwargs)
        
        # Create crew (suppress telemetry errors)
        try:
            from utils.io_suppression import suppress_stderr
            with suppress_stderr():
                self.crew = Crew(
                    agents=[self.agent],
                    tasks=[self.task],
                    verbose=False,
                    process=Process.sequential,
                )
        except Exception as e:
            error_msg = str(e)
            if "signal" not in error_msg.lower() and "telemetry" not in error_msg.lower():
                st.error(f"Failed to create CrewAI crew: {e}")
            self.crew = None

    def _direct_design_fallback(self, requirements_summary: str) -> dict:
        """Call Anthropic Messages API directly when CrewAI is unavailable or fails."""
        if not self.api_key:
            return {
                "error": "Anthropic API key not found",
                "solution": "Add ANTHROPIC_API_KEY to your .env file",
            }
        prompt = (
            "Create UI/UX design specs as strict JSON only.\n"
            "Return ONLY one JSON object with keys: design_overview, color_scheme, typography, "
            "navigation, screens, ui_components, responsive_design, accessibility, animations, icons.\n\n"
            "Keep response compact to avoid truncation:\n"
            "- screens: exactly 4 items\n"
            "- key_components per screen: 4-6 items\n"
            "- ui_components: 8-12 items\n"
            "- design_overview/animations/icons/accessibility/responsive_design: short strings\n"
            "- no prose outside JSON\n\n"
            f"Requirements:\n{requirements_summary}"
        )
        try:
            resp = requests.post(
                f"{self.api_base}/messages",
                headers=self.headers,
                json={
                    "model": DEFAULT_MODEL,
                    "max_tokens": 3200,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["content"][0]["text"]
                parsed = parse_json_from_text(content)
                if isinstance(parsed, dict) and "error" not in parsed:
                    parsed["model_used"] = f"{DEFAULT_MODEL} (direct)"
                    parsed["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    return parsed
                return {
                    "error": "Could not parse UI design JSON",
                    "solution": "Direct API returned non-JSON output. Please retry.",
                    "raw_response": (content or "")[:4000],
                }
            if resp.status_code == 401:
                return {
                    "error": "Authentication failed",
                    "solution": "Check ANTHROPIC_API_KEY in your .env file",
                    "raw_error": resp.text,
                }
            return {
                "error": f"Anthropic API error {resp.status_code}",
                "solution": "Please retry shortly. If it persists, verify model name and account limits.",
                "raw_error": resp.text[:800],
            }
        except requests.exceptions.RequestException as e:
            return {
                "error": "Connection error. Unable to reach Anthropic API.",
                "solution": "Check network, VPN/firewall, and retry.",
                "raw_error": str(e),
            }

    def generate_design(self, requirements: Dict[str, Any], max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
        """Generate UI/UX design specification from requirements with retry logic.
        
        Args:
            requirements: Dictionary containing structured requirements.
            max_retries: Maximum number of retry attempts (default: MAX_RETRIES).
        
        Returns:
            Dictionary containing design specification or error information.
        """
        if not self.api_key:
            return create_error_response(ERROR_NO_API_KEY, SOLUTION_ADD_API_KEY)

        requirements_summary = json.dumps(requirements, indent=2)

        # 1) Try CrewAI first — full orchestration with role, goal, and backstory context
        # If CrewAI unavailable, fall straight through to the direct API fallback below
        if not CREWAI_AVAILABLE or not self.crew or not self.task:
            return self._direct_design_fallback(requirements_summary)

        # Update task with requirements - force actual output
        self.task.description = self._build_task_prompt(requirements_summary)
        for attempt in range(max_retries):
            # Bind before any code that might throw (avoids UnboundLocalError on `result`).
            result = None
            try:
                from utils.io_suppression import suppress_io
                start_time = time.time()
                with suppress_io():
                    result = self.crew.kickoff(inputs={"requirements": requirements_summary})
                elapsed = time.time() - start_time
                if elapsed > 20:
                    st.caption(f"⏱️ Design generation completed in {elapsed:.1f}s")

                # ---- result processing (only reached when kickoff succeeded) ----
                if result is None:
                    if attempt < max_retries - 1:
                        continue
                    return {"error": "CrewAI agent returned no result.", "solution": "Please try again."}

                # Priority 1: Pydantic-validated dict
                if hasattr(result, "tasks_output") and result.tasks_output:
                    task_out = result.tasks_output[0]
                    if hasattr(task_out, "json_dict") and isinstance(task_out.json_dict, dict):
                        return task_out.json_dict
                    if hasattr(task_out, "json") and isinstance(task_out.json, dict):
                        return task_out.json

                # Priority 2: Extract and parse raw text
                content = extract_crewai_output(result)

                if not content:
                    return {
                        "error": "Empty response from CrewAI agent",
                        "solution": "The agent did not return any content. Please try again or check the agent configuration.",
                        "raw_response": str(result)[:800],
                    }

                parsed = parse_json_from_text(content)

                if "error" not in parsed:
                    return parsed

                error_details = {
                    "error": "Could not parse UI design JSON",
                    "parse_error": parsed.get("error", "Unknown error"),
                    "raw_response": parsed.get("raw_response", content[:1500]),
                    "extracted_content": content[:2000] if len(content) > 2000 else content,
                    "solution": "The agent returned content but it's not valid JSON. Please try again.",
                }
                if "json_error" in parsed:
                    error_details["json_error"] = parsed["json_error"]
                if "extracted_json" in parsed:
                    error_details["extracted_json"] = parsed["extracted_json"]
                return error_details
                    
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                snipped = _snip_crew_error(error_msg)
                transport = _retryable_transport_error(e)

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * EXPONENTIAL_BACKOFF_BASE
                    if transport:
                        st.warning(MSG_CONNECTION_ISSUE.format(wait_time=wait_time) + f" ({attempt + 1}/{max_retries})")
                    else:
                        st.warning(f"⚠️ Error: {snipped[:400]}. {MSG_RETRYING} ({attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue

                # Final attempt: fall back to direct Anthropic API (CrewAI timed out or hit rate limits)
                fb = self._direct_design_fallback(requirements_summary)
                if isinstance(fb, dict) and "error" not in fb:
                    return fb

                fb_err = fb.get("error", "") if isinstance(fb, dict) else str(fb)
                return {
                    "error": f"Design generation failed ({error_type})",
                    "solution": (
                        "CrewAI reported an error and the direct Anthropic fallback also failed. "
                        "Check ANTHROPIC_API_KEY, DEFAULT_MODEL in .env, network/VPN, and Anthropic status. "
                        "If you see 'execution timed out', try again or increase CrewAI task time limits."
                    ),
                    "raw_error": snipped,
                    "fallback_error": fb_err,
                    "fallback_raw": (fb.get("raw_error") if isinstance(fb, dict) else None),
                }
        
        return {"error": "Max retries exceeded. Please try again later."}


# ----------------------------
# CHAT ASSISTANT AGENT (CrewAI)
# ----------------------------

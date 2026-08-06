"""Shared utility functions for all agents."""
import json
import re
from typing import Any, Dict

from core.constants import THOUGHT_PATTERNS, THOUGHT_INDICATORS


_ICON_GLYPH_MAP = {
    "button": "▶", "search": "🔍", "input": "✎", "card": "▦",
    "list": "☰", "image": "▣", "header": "▤", "footer": "▥",
    "modal": "◉", "menu": "☰", "user": "●", "settings": "⚙",
    "home": "⌂", "notification": "◉", "heart": "♥", "share": "↗",
    "like": "▲", "comment": "◐", "save": "⬇", "edit": "✎",
    "filter": "⧉", "bookmark": "★", "translate": "文", "profile": "●",
}

_EMOJI_TO_GLYPH = {
    "🔍": "🔍", "🏠": "⌂", "👤": "●", "⚙️": "⚙", "🔔": "◉",
    "🔖": "★", "❤️": "♥", "📤": "↗", "⭐": "★", "🔽": "⧉",
    "➕": "+", "✏️": "✎", "💾": "⬇", "📋": "☰", "🖼️": "▣",
}


def get_component_icon_glyph(comp: Dict[str, Any]) -> str:
    """Resolve a design component to its display glyph.

    Single source of truth for icon rendering — used by both the live design
    preview (ui/main_ui.py's build_device_html) and the generated app scaffold
    (utils/code_scaffold.py), so the two never drift apart.
    """
    if not isinstance(comp, dict):
        return "●"
    icon = comp.get("icon", "")
    ctype = str(comp.get("type", "")).lower()
    cname = str(comp.get("name", "")).lower()

    if icon:
        if icon in _EMOJI_TO_GLYPH:
            return _EMOJI_TO_GLYPH[icon]
        if len(icon) <= 2:
            return icon

    for key, glyph in _ICON_GLYPH_MAP.items():
        if key in ctype or key in cname:
            return glyph

    if cname:
        return cname[0].upper()
    if ctype:
        return ctype[0].upper()
    return "●"


def create_error_response(error: str, solution: str, **kwargs) -> Dict[str, Any]:
    response = {"error": error, "solution": solution}
    response.update(kwargs)
    return response


def parse_json_from_text(text: str) -> Dict[str, Any]:
    """Robustly parse a JSON object from free-form text."""
    try:
        if not text:
            return {"error": "Empty response", "raw_response": ""}

        text = text.strip()
        text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```\s*", "", text)

        json_start = text.find("{")
        if json_start == -1:
            return {"error": "No JSON object found in response", "raw_response": text[:1000]}

        brace_count = 0
        json_end = -1
        for i in range(json_start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        if json_end == -1:
            json_end = text.rfind("}")
            if json_end > json_start:
                json_end += 1
            else:
                return {"error": "Incomplete JSON object (missing closing brace)", "raw_response": text[:1000]}

        candidate = text[json_start:json_end]
        candidate = re.sub(r",\s*}", "}", candidate)
        candidate = re.sub(r",\s*]", "]", candidate)
        candidate = re.sub(r",\s*,", ",", candidate)

        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

        return {"error": "Response JSON is not an object", "raw_response": text[:1000], "extracted_json": candidate[:500]}

    except json.JSONDecodeError as e:
        error_pos = getattr(e, "pos", None)
        error_msg = str(e)
        if error_pos:
            start = max(0, error_pos - 50)
            end = min(len(text), error_pos + 50)
            error_msg = f"{error_msg} (near position {error_pos}: ...{text[start:end]}...)"
        return {"error": f"JSON parsing error: {error_msg}", "raw_response": text[:1000], "json_error": str(e)}
    except Exception as e:
        return {"error": f"Could not parse JSON: {e}", "raw_response": text[:1000] if text else ""}


def extract_crewai_output(result) -> str:
    """Robustly extract string output from a CrewAI kickoff result."""
    if not result:
        return ""

    content = None
    original_content = None

    # Method 1: tasks_output attribute (preferred in newer CrewAI)
    if hasattr(result, "tasks_output"):
        tasks_output = result.tasks_output
        if isinstance(tasks_output, list) and len(tasks_output) > 0:
            task_output = tasks_output[0]
            for attr in ["raw", "output", "content", "result", "messages"]:
                if hasattr(task_output, attr):
                    val = getattr(task_output, attr)
                    if isinstance(val, str) and val.strip():
                        content = val
                        original_content = val
                        break
                    elif isinstance(val, list) and attr == "messages":
                        for msg in reversed(val):
                            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
                                content = msg.content
                                original_content = content
                                break
                        if content:
                            break

            if not content and hasattr(task_output, "__dict__"):
                json_candidates, other_candidates = _scan_dict(task_output.__dict__)
                content = original_content = (json_candidates or other_candidates or [("", "")])[0][1] or None
                if not content:
                    content = original_content = str(task_output)

    # Method 2: direct result attributes
    if not content:
        for attr in ["raw", "output", "content", "result"]:
            if hasattr(result, attr):
                val = getattr(result, attr)
                if isinstance(val, str) and val.strip():
                    content = val
                    original_content = val
                    break
                elif isinstance(val, list) and val:
                    first = val[0]
                    if isinstance(first, str) and first.strip():
                        content = first
                        original_content = first
                        break
                    elif hasattr(first, "raw"):
                        raw_val = getattr(first, "raw", "")
                        if raw_val and isinstance(raw_val, str):
                            content = raw_val
                            original_content = raw_val
                            break

    # Method 3: scan result.__dict__
    if not content and hasattr(result, "__dict__"):
        json_candidates, other_candidates = _scan_dict(result.__dict__)
        if json_candidates:
            content = original_content = json_candidates[0][1]
        elif other_candidates:
            content = original_content = other_candidates[0][1]

    # Method 4: last resort string conversion
    if not content:
        content = original_content = str(result)

    if not content:
        return ""

    content = content.strip()
    original_content = (original_content or content).strip()

    # Strip common CrewAI thought prefixes
    for prefix in ("Thought:", "Action:", "Observation:", "Final Answer:", "Reflection:"):
        content = re.sub(rf"^{prefix}\s*", "", content, flags=re.IGNORECASE | re.MULTILINE)

    lines = content.split("\n")
    has_json = any("{" in line or "[" in line for line in lines)

    filtered = []
    for line in lines:
        if has_json and any(re.search(p, line.strip(), re.IGNORECASE) for p in THOUGHT_PATTERNS):
            continue
        filtered.append(line)

    content = "\n".join(filtered).strip()

    if content and not has_json:
        if any(ind in content.lower() for ind in THOUGHT_INDICATORS):
            return ""

    if not content and original_content:
        if any(ind in original_content.lower() for ind in THOUGHT_INDICATORS):
            return ""

    return content if content else ""


def _scan_dict(d: dict, path: str = "") -> tuple:
    """Return (json_candidates, other_candidates) from a flat dict scan."""
    json_candidates = []
    other_candidates = []
    for key, val in d.items():
        if isinstance(val, str) and val.strip() and len(val) > 10:
            target = json_candidates if ("{" in val or "[" in val) else other_candidates
            target.append((f"{path}.{key}", val))
        elif isinstance(val, (dict, list)):
            j, o = _scan_collection(val, f"{path}.{key}")
            json_candidates.extend(j)
            other_candidates.extend(o)
    return json_candidates, other_candidates


def _scan_collection(obj, path: str = "") -> tuple:
    json_candidates = []
    other_candidates = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and v.strip() and len(v) > 10:
                target = json_candidates if ("{" in v or "[" in v) else other_candidates
                target.append((f"{path}.{k}", v))
            elif isinstance(v, (dict, list)):
                j, o = _scan_collection(v, f"{path}.{k}")
                json_candidates.extend(j)
                other_candidates.extend(o)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str) and item.strip() and len(item) > 10:
                target = json_candidates if ("{" in item or "[" in item) else other_candidates
                target.append((f"{path}[{i}]", item))
            elif isinstance(item, (dict, list)):
                j, o = _scan_collection(item, f"{path}[{i}]")
                json_candidates.extend(j)
                other_candidates.extend(o)
    return json_candidates, other_candidates

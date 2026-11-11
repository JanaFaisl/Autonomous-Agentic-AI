"""Interactive HTML preview of a generated design.

Turns the Design Agent's JSON into a clickable prototype: a device frame, per-screen app bar,
real navigation (tab bar on mobile, top bar on desktop) and a purpose-built widget for every
component type (headers, forms, lists, segmented controls, empty states, ...). The design text
fields are specifications ("white card, 18px radius, '3 of 7 done'"), so only their quoted
strings are shown as copy; everything else drives the layout.

Behaviour is wired with data attributes and one delegated click handler, so no user-supplied
text is ever placed inside inline JavaScript.
"""
import html
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from core.language import is_rtl

from utils.prototype_bridge import is_input_component

# ---------------------------------------------------------------------------
# Icons (24x24 outline set)
# ---------------------------------------------------------------------------
_ICONS = {
    "home": '<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "user": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
    "search": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>',
    "plus": '<line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>',
    "settings": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
    "bell": '<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/>',
    "heart": '<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>',
    "star": '<polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>',
    "bookmark": '<path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>',
    "list": '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "check-circle": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "calendar": '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    "mail": '<path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"/><polyline points="22,6 12,13 2,6"/>',
    "lock": '<rect x="3" y="11" width="18" height="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/>',
    "cart": '<circle cx="9" cy="21" r="1"/><circle cx="20" cy="21" r="1"/><path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6"/>',
    "trash": '<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>',
    "edit": '<path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4L16.5 3.5z"/>',
    "filter": '<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>',
    "pin": '<path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/>',
    "message": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "alert": '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
    "x": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "chevron": '<polyline points="9 18 15 12 9 6"/>',
    "back": '<line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>',
    "grid": '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>',
    "chart": '<line x1="12" y1="20" x2="12" y2="10"/><line x1="18" y1="20" x2="18" y2="4"/><line x1="6" y1="20" x2="6" y2="16"/>',
    "share": '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>',
    "eye": '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>',
    "logout": '<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>',
    "menu": '<line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="18" x2="21" y2="18"/>',
    "inbox": '<polyline points="22 12 16 12 14 15 10 15 8 12 2 12"/><path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
    "file": '<path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/>',
    "tag": '<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/>',
    "phone": '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.96.36 1.9.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.34 1.85.57 2.81.7A2 2 0 0 1 22 16.92z"/>',
    "camera": '<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    "dollar": '<line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/>',
    "play": '<polygon points="5 3 19 12 5 21 5 3"/>',
    "refresh": '<polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>',
}

# (keyword in icon hint / name, icon) in priority order
_ICON_KEYWORDS: List[Tuple[str, str]] = [
    ("empty", "inbox"), ("logout", "logout"), ("log out", "logout"), ("sign out", "logout"),
    ("search", "search"), ("home", "home"), ("dashboard", "grid"), ("profile", "user"), ("account", "user"),
    ("user", "user"), ("avatar", "user"), ("people", "user"), ("contact", "user"),
    ("setting", "settings"), ("preference", "settings"), ("notif", "bell"), ("bell", "bell"), ("alert", "alert"),
    ("warn", "alert"), ("error", "alert"), ("heart", "heart"), ("like", "heart"), ("favorite", "heart"),
    ("star", "star"), ("rating", "star"), ("review", "star"), ("bookmark", "bookmark"), ("save", "bookmark"),
    ("calendar", "calendar"), ("date", "calendar"), ("due", "calendar"), ("schedule", "calendar"),
    ("mail", "mail"), ("email", "mail"), ("password", "lock"), ("lock", "lock"), ("secur", "lock"),
    ("cart", "cart"), ("basket", "cart"), ("order", "cart"), ("checkout", "cart"), ("price", "dollar"),
    ("pay", "dollar"), ("trash", "trash"), ("delete", "trash"), ("remove", "trash"), ("edit", "edit"),
    ("pencil", "edit"), ("filter", "filter"), ("sort", "filter"), ("clock", "clock"), ("time", "clock"),
    ("history", "clock"), ("image", "image"), ("photo", "image"), ("picture", "image"), ("camera", "camera"),
    ("location", "pin"), ("map", "pin"), ("address", "pin"), ("chat", "message"), ("message", "message"),
    ("comment", "message"), ("info", "info"), ("help", "info"), ("about", "info"), ("share", "share"),
    ("phone", "phone"), ("call", "phone"), ("download", "download"), ("upload", "upload"),
    ("chart", "chart"), ("stat", "chart"), ("report", "chart"), ("analytic", "chart"), ("progress", "chart"),
    ("tag", "tag"), ("label", "tag"), ("category", "tag"), ("eye", "eye"), ("view", "eye"),
    ("refresh", "refresh"), ("play", "play"), ("menu", "menu"),
    ("add", "plus"), ("new", "plus"), ("create", "plus"), ("plus", "plus"), ("check", "check-circle"),
    ("done", "check-circle"), ("complete", "check-circle"), ("task", "check-circle"), ("todo", "check-circle"),
    ("list", "list"),
    ("file", "file"), ("doc", "file"), ("note", "file"),
]

_TYPE_ICON = {"button": "chevron", "card": "grid", "list": "list", "image": "image", "input": "edit"}


def _icon_name(*hints: Any) -> str:
    for hint in hints:
        text = str(hint or "").lower().strip()
        if not text:
            continue
        if text in _ICONS:
            return text
        for key, name in _ICON_KEYWORDS:
            if re.search(rf"(?<![a-z]){re.escape(key)}", text):  # word start: "preview" is not "review"
                return name
    return "grid"


def _svg(name: str, size: int = 20, cls: str = "") -> str:
    body = _ICONS.get(name, _ICONS["grid"])
    return (f'<svg class="ico {cls}" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" '
            f'aria-hidden="true">{body}</svg>')


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _e(text: Any) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


_QUOTE_RE = re.compile(r"'([^'\n]{1,48})'|“([^”\n]{1,48})”|\"([^\"\n]{1,48})\"")
_NOISE_RE = re.compile(r"^(#[0-9a-f]{3,8}|[\d.]+\s*(px|rem|em|%|pt|ms|s)|[\d.]+)$", re.I)
_STRIP_WORDS = re.compile(r"\b(button|btn|input|field|card|component|section|area|container|widget|control|element|cta|header|زر|أزرار|حقل|إدخال|بطاقة|مكوّن|مكون|منطقة|حاوية|قسم)\b", re.I)


def _quotes(*texts: Any) -> List[str]:
    """Copy the design wrote in quotes, minus colours, sizes and other spec noise."""
    out: List[str] = []
    for text in texts:
        for m in _QUOTE_RE.finditer(str(text or "")):
            q = next(g for g in m.groups() if g is not None).strip()
            if q and not _NOISE_RE.match(q) and re.search(r"[A-Za-z0-9]", q) and q not in out:
                out.append(q)
    return out


def _clean(name: str, fallback: str = "") -> str:
    """'Add Task Button' -> 'Add Task'; keeps the original if stripping would empty it."""
    cleaned = re.sub(r"\s{2,}", " ", _STRIP_WORDS.sub("", str(name or ""))).strip(" -–:_")
    return cleaned or str(name or "").strip() or fallback


def _first_sentence(text: Any, limit: int = 80) -> str:
    s = re.split(r"(?<=[.!?])\s", str(text or "").strip())[0].strip()
    return (s[: limit - 1] + "…") if len(s) > limit else s


def _singular(word: str) -> str:
    w = word.strip()
    if w.lower().endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.lower().endswith("s") and not re.search(r"(ss|us|is)$", w.lower()) and len(w) > 3:
        return w[:-1]  # keeps "Status", "Address", "Canvas"
    return w


_GERUND = {"opens": "Opening", "shows": "Showing", "displays": "Showing", "launches": "Launching", "sends": "Sending",
           "downloads": "Downloading", "saves": "Saving", "calls": "Calling", "shares": "Sharing", "copies": "Copying",
           "loads": "Loading", "reloads": "Reloading", "refreshes": "Refreshing", "applies": "Applying", "starts": "Starting",
           "plays": "Playing", "prints": "Printing", "exports": "Exporting", "uploads": "Uploading", "scans": "Scanning"}


def _toast_text(label: str, interaction: str) -> str:
    """User-facing feedback for a button: 'Opens the Stripe receipt.' -> 'Opening Stripe receipt…'."""
    sentence = _first_sentence(interaction, 70).rstrip(".")
    m = re.match(r"(\w+)\s+(?:the\s+|a\s+|an\s+)?(.+)$", sentence)
    if m and m.group(1).lower() in _GERUND:
        return f"{_GERUND[m.group(1).lower()]} {m.group(2)}…"
    return label


def _initials(text: str) -> str:
    parts = re.findall(r"[A-Za-z]+", text)
    return ("".join(p[0] for p in parts[:2]) or "A").upper()


_AUTH_SCREEN = re.compile(r"login|log in|sign ?in|sign ?up|register|welcome|splash|onboard|forgot|reset|verify", re.I)


def _find_target(text: str, screens: List[str], exclude: str = "") -> str:
    """Screen a sentence refers to, e.g. 'Taps the avatar to open Profile' -> 'Profile'."""
    low = str(text or "").lower()
    for name in sorted(screens, key=len, reverse=True):
        if name == exclude or not name:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(name.lower())}(?![a-z0-9])", low):
            return name
    return ""


# ---------------------------------------------------------------------------
# Component classification
# ---------------------------------------------------------------------------
def _kind(comp: Dict[str, Any]) -> str:
    name = str(comp.get("name") or "").lower()
    ctype = str(comp.get("type") or "").lower()
    both = f"{name} {ctype}"
    if re.search(r"bottom nav|navigation bar|nav ?bar|tab ?bar|navigation|sidebar|side menu|drawer", both):
        return "nav"
    if re.search(r"toast|snackbar", both):
        return "skip"
    if re.search(r"modal|dialog|bottom sheet|popup", both):
        return "modal"
    if re.search(r"\b(filter|category|subject|specialty|specialization|course|occasion|topic|tag)s?\s+chips?\b|\bchips?\s+(row|group|bar|list)\b|\bchips\b", name) and not is_input_component(comp):
        return "chips"
    if re.search(r"empty|no results|zero state|placeholder state", name):
        return "empty"
    if re.search(r"\bfab\b|floating", both):
        return "fab"
    if is_input_component(comp):
        return "input"
    if re.search(r"icon ?button", both):
        return "iconbtn"
    if re.search(r"\blink\b", ctype):
        return "link"
    if ("button" in ctype or re.search(r"\bbutton\b|\bcta\b", name)) and _ICON_ONLY.match(_clean(comp.get("name") or "")):
        return "iconbtn"
    if "button" in ctype or re.search(r"\bbutton\b|\bcta\b", name):
        return "button"
    if re.search(r"segment|tabs|filter bar|chip group|toggle group", both):
        return "segmented"
    if "progress" in both or "stepper" in both:
        return "progress"
    if re.search(r"switch|toggle", both):
        return "switch"
    if re.search(r"checkbox|radio|check box", both):
        return "check"
    if re.search(r"badge|chip|tag|pill", ctype) or re.search(r"\bbadge\b", name):
        return "badge"
    if re.search(r"alert|notice|banner|message|notification", ctype) or re.search(r"alert|notice|banner", name):
        return "alert"
    if re.search(r"divider|separator", both):
        return "divider"
    if "footer" in both:
        return "footer"
    if re.search(r"header|app ?bar|toolbar|hero|title bar", both):
        return "header"
    if re.search(r"carousel|slider|swipe", both):
        return "carousel"
    if re.search(r"list|table|feed|grid|timeline", ctype):
        return "list"
    if re.search(r"\btext\b|label|heading|paragraph|title|caption", ctype):
        return "text"
    if re.search(r"image|avatar|photo|map|chart|graph|video|carousel|gallery|media", ctype):
        return "media"
    return "card"


_ICON_ONLY = re.compile(r"^(notifications?|alerts?|bell|cart|basket|bag|menu|filters?|profile|settings|share|favou?rites?|bookmarks?|wishlist|search|more|close|back|messages?|inbox)$", re.I)


def _chip_options(styling: str, interaction: str) -> List[str]:
    """Chip labels from quotes, or from a list like 'chips: All, Math, Physics' / '(Cardiology, Dermatology)'."""
    quotes = [q for q in _quotes(styling) if len(q) <= 22]
    if len(quotes) >= 2:
        return quotes[:8]
    for text in (styling, interaction):
        m = re.search(r"(?:chips?|pills?|tabs?|filters?|options?)[^:()\n]*[:(]\s*([^.;)\n]{4,200})", str(text or ""), re.I)
        if not m:
            m = re.search(r"\(\s*(?:e\.g\.,?\s*)?([A-Z][^()\n]{4,160})\)", str(text or ""))
        if not m:
            continue
        parts = re.split(r",|\band\b|/", re.sub(r"\b(e\.g\.|etc\.?|and more|with .*)$", "", m.group(1), flags=re.I))
        opts = []
        for part in parts:
            item = part.strip(" '\"\u201c\u201d.")
            if item and len(item) <= 22 and len(item.split()) <= 3 and item.lower() not in ("etc", "more") and item not in opts:
                opts.append(item)
        if len(opts) >= 2:
            return opts[:8]
    return []


def _variant(name: str, styling: str) -> str:
    text = f"{name} {styling}".lower()
    if re.search(r"delete|remove|destructive|danger|log ?out|sign ?out", text):
        return "danger"
    if re.search(r"cancel|secondary|outline|ghost|text button|tertiary|skip|later|back", text):
        return "ghost"
    return "primary"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
class _Ctx:
    def __init__(self, screens: List[str], bridge: Optional[Dict[str, Any]], nav_targets: List[str]):
        self.screens = screens
        self.bridge = bridge
        self.nav_targets = nav_targets
        self.sheets: List[str] = []
        self.header_done = False

    def add_sheet(self, title: str, subtitle: str, body: str, actions: str) -> str:
        """Register a bottom-sheet template and return its id."""
        sid = f"sh{len(self.sheets)}"
        self.sheets.append(
            f'<template id="{sid}"><div class="grab"></div><h3>{_e(title)}</h3>'
            f'{f"<p class=sub>{_e(subtitle)}</p>" if subtitle else ""}{body}<div class="acts">{actions}</div></template>')
        return sid

    def submittable(self, screen: str) -> bool:
        return bool(self.bridge and (self.bridge.get("screens") or {}).get(screen))


def _sw(label: str, on: bool = False) -> str:
    return (f'<label class="toggle-row"><span>{_e(label)}</span>'
            f'<input type="checkbox" class="switch" data-toggle="1"{" checked" if on else ""}/></label>')


def _field(label: str, value: str = "", kind: str = "text") -> str:
    return (f'<label class="field"><span class="field-label">{_e(label)}</span>'
            f'<div class="field-box"><input type="{kind}" value="{_e(value)}" placeholder="{_e(label)}"/></div></label>')


def _row_sheet(ctx: "_Ctx", title: str, parent: str, interaction: str) -> str:
    """Detail sheet opened by a list row that has no screen of its own."""
    low = title.lower()
    done = '<button class="btn ghost" data-close="1">Close</button>'
    if re.search(r"edit profile|personal|account details", low):
        return ctx.add_sheet(title, "Update how you appear in the app.", _field("Name", "Alex Morgan") + _field("Email", "alex@example.com", "email"),
                             '<button class="btn primary" data-close="1" data-toast="' + _e(_t("Profile updated")) + '">' + _e(_t("Save changes")) + '</button>' + done)
    if "notif" in low:
        return ctx.add_sheet(title, "Choose what you hear about.", _sw("Push notifications", True) + _sw("Email updates") + _sw("Reminders", True), done)
    if re.search(r"privacy|security|password", low):
        return ctx.add_sheet(title, "Control your data and sign-in.", _sw("Private account") + _sw("Two-factor authentication"),
                             '<button class="btn primary" data-close="1" data-toast="Password reset link sent">Change password</button>' + done)
    if re.search(r"help|support|faq|contact", low):
        body = '<div class="rows">' + "".join(
            f'<div class="row"><span class="row-text"><b>{_e(q)}</b></span></div>' for q in
            ("How do I reset my password?", "How do I delete my account?", "Where is my data stored?")) + "</div>"
        return ctx.add_sheet(title, "Quick answers and how to reach us.", body,
                             '<button class="btn primary" data-close="1" data-toast="Message sent to support">Contact support</button>' + done)
    detail = _first_sentence(interaction, 120) or f"This opens the {title.lower()} view of {parent.lower() or 'the app'}."
    return ctx.add_sheet(title, parent, f'<p class="copy">{_e(detail)}</p>', done)


_ACTS = (
    ("logout", r"log ?out|sign ?out"),
    ("delete", r"\b(delete|remove|trash|discard|clear)\b"),
    ("done", r"mark (?:as )?(?:done|complete)|\bcomplete[sd]?\b|check off|\btick\b|\bfinish"),
    ("like", r"\b(like|favou?rite|bookmark|heart|wishlist|save for later|star)\b"),
    ("add", r"\b(add|create|save|submit|post|send|book|place|register|sign ?up|publish|upload|confirm|checkout|pay|log ?in|sign ?in|apply|update)\b"),
)


def _action_attrs(label: str, interaction: str, screen: str, ctx: _Ctx, *, submit_ok: bool) -> str:
    """data-* attributes that make an element do something visible when clicked.

    Order of preference: open the screen it mentions, go back, run a recognised in-page action
    (delete / mark done / favourite / add an item / log out), otherwise say what it does.
    """
    target = _find_target(interaction, ctx.screens, exclude=screen) or _find_target(label, ctx.screens, exclude=screen)
    low = f"{label} {interaction}".lower()
    attrs: List[str] = []
    act = next((name for name, rx in _ACTS if re.search(rx, low)), "")
    if act == "add" and not submit_ok:
        act = ""
    if act == "logout" and not target:
        target = next((n for n in ctx.screens if re.search(r"log ?in|sign ?in|welcome", n, re.I) and n != screen), "")
        if not target:
            sid = ctx.add_sheet("Signed out", "You've been signed out of this prototype.",
                                '<p class="copy">Your session has ended. Sign in again to keep going.</p>',
                                f'<button class="btn primary" data-close="1" data-nav="{_e(ctx.screens[0] if ctx.screens else "")}">Sign in again</button>')
            return f'data-sheet="{sid}"'

    if re.search(r"\b(go back|goes back|return to previous|cancel|dismiss|close)\b", low) and not target:
        attrs.append('data-back="1"')
    elif target:
        attrs.append(f'data-nav="{_e(target)}"')
    if submit_ok and ctx.submittable(screen):
        attrs.append('data-submit="1"')
    if act and act != "logout" and not target:
        attrs.append(f'data-act="{act}"')
    elif act == "add" and target:
        attrs.append('data-act="add"')  # add the item, then open the screen it names
    if act == "delete" or (not target and re.search(r"confirm|are you sure|modal|dialog", low)):
        attrs.append(f'data-confirm="{_e(_first_sentence(interaction) or "Delete this item?")}"')
    if not target and not act and "data-back" not in " ".join(attrs):
        attrs.append(f'data-toast="{_e(_toast_text(label, interaction))}"')
    return " ".join(attrs)


def _render_component(comp: Dict[str, Any], screen: Dict[str, Any], ctx: _Ctx, has_header: bool) -> str:
    sname = str(screen.get("name") or "")
    name = str(comp.get("name") or "Component")
    styling = str(comp.get("styling") or "")
    interaction = str(comp.get("interaction") or "")
    kind = _kind(comp)
    label = _clean(name, name)
    quotes = _quotes(styling, interaction)
    icon = _icon_name(name, comp.get("icon"), comp.get("type"))
    if kind in ("nav", "skip"):
        return ""

    if kind == "modal":
        return ""  # dialogs open from the button that mentions them (data-confirm)

    if kind == "header" and ctx.header_done:
        return f'<h2 class="sect">{_e(label)}</h2>'  # one hero per screen; later headers are section titles
    if kind == "header":
        ctx.header_done = True
        purpose = _first_sentence(screen.get("purpose"), 96)
        avatar = ""
        if str(comp.get("image_type") or "").lower() == "user" or "avatar" in f"{styling} {interaction}".lower():
            target = _find_target(interaction, ctx.screens, exclude=sname)
            attr = f'data-nav="{_e(target)}"' if target else f'data-toast="{_e(_toast_text("Profile", interaction))}"'
            avatar = f'<button class="avatar" {attr} aria-label="Open profile">{_e(_initials(sname if not quotes else quotes[-1]))}</button>'
        return (f'<header class="hero"><button class="back" data-back="1" aria-label="Back">{_svg("back")}</button>'
                f'<div class="hero-text"><h1>{_e(sname)}</h1>'
                f'{f"<p>{_e(purpose)}</p>" if purpose else ""}</div>{avatar}</header>')

    if kind == "footer":
        text = quotes[0] if quotes else ""
        return f'<footer class="foot">{_e(text)}</footer>' if text else ""

    if kind == "divider":
        return '<hr class="divider"/>'

    if kind == "button" or kind == "link":
        variant = "link" if kind == "link" else _variant(name, styling)
        text = next((q for q in quotes if len(q) <= 26 and not q.endswith(".")), label)
        attrs = _action_attrs(text, interaction, sname, ctx, submit_ok=(variant == "primary"))
        lead = _svg(icon, 18) if variant == "primary" and icon in ("plus", "check-circle", "logout", "trash", "download", "upload", "share") else ""
        return f'<button class="btn {variant}" {attrs}>{lead}<span>{_e(text)}</span></button>'

    if kind == "iconbtn":
        attrs = _action_attrs(label, interaction, sname, ctx, submit_ok=False)
        return f'<button class="iconbtn" {attrs} aria-label="{_e(label)}" title="{_e(label)}">{_svg(icon, 20)}</button>'

    if kind == "fab":
        attrs = _action_attrs(label, interaction, sname, ctx, submit_ok=True)
        plus = icon if icon != "grid" else "plus"
        return f'<button class="fab" {attrs} aria-label="{_e(label)}" title="{_e(label)}">{_svg(plus, 24)}</button>'

    if kind == "input":
        lowname = name.lower()
        field = lowname.replace(" ", "_")
        placeholder = ""
        m = re.search(r"placeholder[^'\"“]*['\"“]([^'\"”]{1,40})", styling, re.I)
        if m:
            placeholder = m.group(1)
        label_txt = _clean(re.sub(r"\b(textarea|text area|textfield|text field)\b", "", name, flags=re.I), name)
        is_search = "search" in lowname or icon == "search"
        if is_search:
            return (f'<div class="search">{_svg("search", 18)}<input type="text" placeholder="{_e(placeholder or _t("Search"))}" '
                    f'data-screen="{_e(sname)}" data-field="{_e(field)}" aria-label="{_e(label_txt)}"/></div>')
        kindtype = "password" if re.search(r"password|passcode|pin\b", lowname) else "email" if "email" in lowname else \
            "date" if re.search(r"\bdate\b|datepicker", lowname) else "number" if re.search(r"quantity|amount|price|age\b|count", lowname) else "text"
        ph = placeholder or (f"Enter {label_txt.lower()}" if kindtype == "text" else "")
        control = (f'<textarea rows="3" placeholder="{_e(ph)}" data-screen="{_e(sname)}" data-field="{_e(field)}"></textarea>'
                   if re.search(r"textarea|description|notes|message|comment|bio\b", lowname) else
                   f'<div class="field-box">{_svg(icon if icon != "grid" else "edit", 18)}'
                   f'<input type="{kindtype}" placeholder="{_e(ph)}" data-screen="{_e(sname)}" data-field="{_e(field)}"/></div>')
        return f'<label class="field"><span class="field-label">{_e(label_txt)}</span>{control}</label>'

    if kind == "chips":
        opts = _chip_options(styling, interaction)
        if len(opts) < 2:
            opts = ["All", "Popular", "New", "Nearby"]
        target = _find_target(interaction, ctx.screens, exclude=sname)
        nav = f' data-nav="{_e(target)}"' if target else ""
        multi = ' data-multi="1"' if re.search(r"multiple|multi-select|several", interaction, re.I) else ""
        chips = "".join(
            f'<button class="chip{" on" if i == 0 and re.match(r"all\b", o, re.I) else ""}" data-chip="{i}"'
            f'{" data-all=1" if re.match(r"all\b", o, re.I) else ""}{nav}>{_e(o)}</button>' for i, o in enumerate(opts))
        return f'<div class="chips"{multi} role="group" aria-label="{_e(label)}">{chips}</div>'

    if kind == "segmented":
        opts = [q for q in quotes if len(q) <= 22]
        if len(opts) < 2:
            return ""
        items = "".join(f'<button class="seg{" on" if i == 0 else ""}" data-seg="1">{_e(o)}</button>' for i, o in enumerate(opts[:5]))
        return f'<div class="segmented" role="tablist">{items}</div>'

    if kind == "progress":
        text = " ".join(quotes) or styling
        pct = 60
        m = re.search(r"(\d{1,3})\s*%", text)
        m2 = re.search(r"(\d+)\s*(?:of|/)\s*(\d+)", text)
        if m:
            pct = min(100, int(m.group(1)))
        elif m2 and int(m2.group(2)):
            pct = min(100, round(100 * int(m2.group(1)) / int(m2.group(2))))
        caption = next((q for q in quotes if not q.endswith("%")), "")
        return (f'<div class="progress"><div class="progress-row"><span>{_e(caption or label)}</span><b>{pct}%</b></div>'
                f'<div class="bar"><i style="width:{pct}%"></i></div></div>')

    if kind in ("switch", "check"):
        cls = "switch" if kind == "switch" else "checkbox"
        return (f'<label class="toggle-row"><span>{_e(label)}</span>'
                f'<input type="checkbox" class="{cls}" data-toggle="1"/></label>')

    if kind == "badge":
        return f'<span class="badge">{_e(quotes[0] if quotes else label)}</span>'

    if kind == "alert":
        tone = "danger" if re.search(r"error|fail|invalid|danger", f"{name} {styling}", re.I) else \
            "success" if re.search(r"success|saved|done|complete", f"{name} {styling}", re.I) else \
            "warn" if re.search(r"warn|caution", f"{name} {styling}", re.I) else "info"
        icon_name = {"danger": "alert", "warn": "alert", "success": "check-circle", "info": "info"}[tone]
        title = quotes[0] if quotes else label
        body = quotes[1] if len(quotes) > 1 else _first_sentence(interaction, 90)
        vld = " vld" if tone == "danger" and re.search(r"validat|empty|required|invalid|inline|whitespace|shake|submit", f"{name} {styling} {interaction}", re.I) else ""
        return (f'<div class="alert {tone}{vld}" role="status">{_svg(icon_name, 20)}'
                f'<div><b>{_e(title)}</b>{f"<span>{_e(body)}</span>" if body else ""}</div></div>')

    if kind == "empty":
        title = quotes[0] if quotes else _t("Nothing here yet")
        body = quotes[1] if len(quotes) > 1 else "Items you add will show up here."
        return (f'<div class="empty"><div class="empty-ic">{_svg("inbox", 30)}</div>'
                f'<h3>{_e(title)}</h3><p>{_e(body)}</p></div>')

    if kind == "text":
        text = quotes[0] if quotes else _first_sentence(screen.get("purpose"), 140) or label
        return f'<p class="copy">{_e(text)}</p>'

    if kind == "carousel":
        noun = _singular(re.sub(r"\b(carousel|slider|gallery|row|section|rail|grid|list)\b", "", label, flags=re.I).strip() or "Item")
        items = quotes[:6] if len(quotes) >= 2 else [f"{noun} {i}" for i in (1, 2, 3, 4)]
        target = _find_target(interaction, ctx.screens, exclude=sname)
        attr = f'data-nav="{_e(target)}"' if target else 'data-select="1"'
        tiles = "".join(
            f'<button class="tile" {attr}><span class="tile-media">{_svg(icon if icon != "grid" else "image", 26)}</span>'
            f'<b>{_e(t)}</b><i class="sk"></i></button>' for t in items)
        return f'<section class="group"><h2>{_e(label)}</h2><div class="hscroll">{tiles}</div></section>'

    if kind == "list":
        noun = _singular(re.sub(r"\b(list|table|feed|grid|timeline|view|carousel|slider|gallery|row|section|rail)\b", "", label, flags=re.I).strip() or "Item")
        rows = quotes[:6] if len(quotes) >= 2 else [f"{noun} {i}" for i in (1, 2, 3)]
        target = _find_target(interaction, ctx.screens, exclude=sname)
        toggles = not target and re.search(_ACTS[2][1], interaction.lower())
        def row_attr(r: str) -> str:
            if target:
                return f'data-nav="{_e(target)}"'
            if toggles:
                return 'data-rowdone="1"'
            if re.search(r"edit profile|personal|account details|notif|privacy|security|password|help|support|faq|contact", r.lower()):
                return f'data-sheet="{_row_sheet(ctx, r, label, interaction)}"'
            return 'data-select="1"'
        body = "".join(
            f'<button class="row" {row_attr(r)}><span class="row-ic">{_svg(_icon_name(r) if _icon_name(r) != "grid" else (icon if icon != "grid" else "list"), 18)}</span>'
            f'<span class="row-text"><b>{_e(r)}</b><i class="sk"></i></span>{_svg("chevron", 16, "chev")}</button>'
            for r in rows)
        return f'<section class="group"><h2>{_e(label)}</h2><div class="rows">{body}</div></section>'

    if kind == "media":
        tall = "tall" if re.search(r"large|full|hero|cover", str(comp.get("size") or "") + styling, re.I) else ""
        return (f'<div class="media {tall}"><div class="media-ic">{_svg(icon if icon != "grid" else "image", 28)}</div>'
                f'<span>{_e(label)}</span></div>')

    # card (default)
    target = _find_target(interaction, ctx.screens, exclude=sname)
    attr = (f'data-nav="{_e(target)}"' if target else (f'data-toast="{_e(_toast_text(label, interaction))}"' if interaction else ""))
    sub = " · ".join(quotes[:2])
    media = ""
    if comp.get("has_image") and str(comp.get("image_type") or "none").lower() != "none":
        media = f'<div class="card-media">{_svg("image", 26)}</div>'
    tag = "button" if attr else "div"
    chev = _svg("chevron", 16, "chev") if target else ""
    return (f'<{tag} class="card" {attr}>{media}<div class="card-row"><span class="card-ic">{_svg(icon, 18)}</span>'
            f'<span class="card-text"><b>{_e(label)}</b>{f"<small>{_e(sub)}</small>" if sub else ""}</span>{chev}</div></{tag}>')


_NAV_STOP = {"screen", "page", "view", "the", "and", "for", "with", "app", "main", "new"}


def _words(text: Any) -> set:
    return {w for w in re.findall(r"[^\W\d_]+", str(text or "").lower()) if len(w) > 2 and w not in _NAV_STOP}


def _same_word(a: str, b: str) -> bool:
    """'orders'/'order', 'scan'/'scanner', 'settings'/'setting' count as the same word."""
    return a == b or (min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)))


def _overlap(a: set, b: set) -> int:
    return sum(1 for x in a if any(_same_word(x, y) for y in b))


def _match_screens(labels: List[str], screens: List[Dict[str, Any]]) -> Dict[str, str]:
    """Best screen for each navigation label: name words weigh most, purpose and components break ties.

    Tabs get their own screen where possible; a tab shares one only when nothing else matches its name.
    """
    scored: List[Tuple[int, int, str, str]] = []
    for li, label in enumerate(labels):
        lw = _words(label)
        for s in screens:
            comps = " ".join(str(c.get("name") or "") for c in (s.get("key_components") or []) if isinstance(c, dict))
            name_hits = _overlap(lw, _words(s["name"]))
            soft_hits = _overlap(lw, _words(s.get("purpose"))) + _overlap(lw, _words(comps))
            score = 3 * name_hits + min(soft_hits, 2)
            if name_hits or soft_hits >= 2:
                scored.append((score, -li, label, s["name"]))
    out: Dict[str, str] = {}
    used: set = set()
    for score, _, label, target in sorted(scored, reverse=True):
        if label not in out and target not in used:
            out[label] = target
            used.add(target)
    # A screen covering two topics ("Profile & Order History") may serve both tabs.
    for score, _, label, target in sorted(scored, reverse=True):
        if label not in out and score >= 3:
            out[label] = target
    return out


def _nav_items(design: Dict[str, Any], screens: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Tabs that actually lead somewhere. Navigation entries with no matching screen are left out."""
    names = [s["name"] for s in screens]
    nav = design.get("navigation") if isinstance(design.get("navigation"), dict) else {}
    raw = nav.get("items") if isinstance(nav.get("items"), list) else []
    entries: List[Tuple[str, str, str]] = []  # label, declared screen, icon hint
    for it in raw:
        label = str(it.get("label") or it.get("name") or it.get("title") or "") if isinstance(it, dict) else str(it)
        if label.strip():
            declared = str(it.get("screen") or "") if isinstance(it, dict) else ""
            entries.append((label.strip(), declared if declared in names else "", str(it.get("icon") or "") if isinstance(it, dict) else ""))
    matched = _match_screens([l for l, d, _ in entries if not d], screens)
    items: List[Dict[str, str]] = []
    for label, declared, icon_hint in entries:
        target = declared or matched.get(label, "")
        if target:
            items.append({"label": label, "target": target, "icon": _icon_name(label, icon_hint)})
    if len(items) < 2:  # navigation missing or mostly unmatched: tabs for the main screens instead
        main = [n for n in names if not _AUTH_SCREEN.search(n)][:4]
        items = [{"label": re.sub(r"\b(screen|page)\b", "", n, flags=re.I).strip() or n, "target": n, "icon": _icon_name(n)}
                 for n in main] if len(main) >= 2 else items
    return items[:5]


_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{--primary:%(primary)s;--secondary:%(secondary)s;--bg:%(bg)s;--surface:%(surface)s;--text:%(text)s;
--muted:color-mix(in srgb,var(--text) 58%%,transparent);--line:color-mix(in srgb,var(--text) 10%%,transparent);
--soft:color-mix(in srgb,var(--primary) 10%%,var(--surface));--ok:#10b981;--warn:#f59e0b;--bad:#ef4444;
--font:%(font)s}
html,body{height:100%%;background:transparent;font-family:var(--font);color:var(--text);-webkit-font-smoothing:antialiased}
button{font:inherit;color:inherit;border:0;background:none;cursor:pointer;text-align:left}
.stage{height:100%%;display:flex;align-items:center;justify-content:center;padding:10px;
background:radial-gradient(1200px 500px at 50%% -10%%,color-mix(in srgb,var(--primary) 18%%,#e8ecf5),#eef1f8)}
.device{position:relative;display:flex;flex-direction:column;overflow:hidden;background:var(--bg);box-shadow:0 24px 60px rgba(15,23,42,.28),0 2px 6px rgba(15,23,42,.18)}
.device.mobile{width:min(392px,100%%);height:min(790px,100%%);border-radius:40px;border:9px solid #0f172a}
.device.desktop{width:100%%;height:100%%;border-radius:14px;border:1px solid rgba(15,23,42,.18)}
.chrome{display:flex;align-items:center;gap:7px;padding:10px 14px;background:#f1f3f9;border-bottom:1px solid var(--line)}
.chrome i{width:11px;height:11px;border-radius:50%%;background:#ff5f57}.chrome i:nth-child(2){background:#febc2e}.chrome i:nth-child(3){background:#28c840}
.chrome span{margin-left:14px;flex:1;max-width:420px;background:#fff;border-radius:7px;padding:4px 12px;font-size:12px;color:#64748b;border:1px solid var(--line)}
.topnav{display:flex;gap:4px;align-items:center;padding:8px 18px;background:var(--surface);border-bottom:1px solid var(--line)}
.topnav button{display:flex;gap:8px;align-items:center;padding:8px 14px;border-radius:10px;font-size:14px;font-weight:600;color:var(--muted)}
.topnav button.active{color:var(--primary);background:var(--soft)}
.screens{position:relative;flex:1;min-height:0}
.screen{display:none;position:absolute;inset:0;flex-direction:column}
.screen.active{display:flex;animation:in .22s ease}
@keyframes in{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.appbar{display:flex;align-items:center;gap:6px;padding:14px 14px 10px;background:var(--surface);border-bottom:1px solid var(--line)}
.appbar h1{flex:1;font-size:17px;font-weight:700;letter-spacing:-.01em;padding-left:4px}
.back{display:none;width:36px;height:36px;border-radius:12px;align-items:center;justify-content:center;color:var(--text)}
.has-back .back{display:inline-flex}.back:hover{background:var(--soft)}
.content>*{flex-shrink:0}
.content{flex:1;overflow:auto;padding:16px 16px 28px;display:flex;flex-direction:column;gap:14px}
.desktop .content{padding:24px 32px 36px;max-width:980px;width:100%%;margin:0 auto}
.hero{display:flex;gap:10px;align-items:center;margin:-16px -16px 2px;padding:26px 18px 28px;color:#fff;
background:linear-gradient(135deg,var(--primary),var(--secondary));border-radius:0 0 26px 26px}
.desktop .hero{margin:0;border-radius:18px}
.hero .back{color:#fff}.hero .back:hover{background:rgba(255,255,255,.18)}
.hero-text{flex:1;min-width:0}.hero h1{font-size:24px;font-weight:700;letter-spacing:-.02em}
.hero p{margin-top:4px;font-size:13.5px;opacity:.86;line-height:1.4}
.avatar{width:40px;height:40px;border-radius:50%%;background:rgba(255,255,255,.22);border:2px solid rgba(255,255,255,.55);color:#fff;font-weight:700;font-size:14px;text-align:center}
.card{display:block;width:100%%;background:var(--surface);border:1px solid var(--line);border-radius:16px;padding:14px;box-shadow:0 1px 2px rgba(15,23,42,.04)}
button.card:hover{border-color:color-mix(in srgb,var(--primary) 40%%,var(--line));box-shadow:0 6px 16px rgba(15,23,42,.08)}
.card-media,.media{display:flex;align-items:center;justify-content:center;gap:10px;color:var(--primary);
background:linear-gradient(135deg,color-mix(in srgb,var(--primary) 14%%,var(--surface)),color-mix(in srgb,var(--secondary) 14%%,var(--surface)))}
.card-media{height:120px;border-radius:12px;margin-bottom:12px}
.media{height:150px;border-radius:16px;flex-direction:column;font-size:13px;font-weight:600;color:var(--muted)}.media.tall{height:220px}
.media-ic{color:var(--primary)}
.card-row{display:flex;align-items:center;gap:12px}
.card-ic,.row-ic{width:38px;height:38px;border-radius:12px;display:inline-flex;align-items:center;justify-content:center;background:var(--soft);color:var(--primary);flex:none}
.card-text{display:flex;flex-direction:column;gap:2px;flex:1;min-width:0}.card-text b{font-size:15px;font-weight:650}
.card-text small{font-size:13px;color:var(--muted)}.chev{color:var(--muted);flex:none}
.group h2{font-size:13px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:2px 2px 8px}
.rows{background:var(--surface);border:1px solid var(--line);border-radius:16px;overflow:hidden}
.row{display:flex;align-items:center;gap:12px;width:100%%;padding:12px 14px}.row+.row{border-top:1px solid var(--line)}
.row:hover{background:var(--soft)}.row-text{flex:1;display:flex;flex-direction:column;gap:6px}.row-text b{font-size:15px;font-weight:600}
.sk{display:block;height:7px;width:55%%;border-radius:6px;background:var(--line)}
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;min-height:46px;padding:0 20px;border-radius:14px;font-size:15px;font-weight:650;width:100%%;transition:transform .08s,box-shadow .15s}
.btn:active{transform:scale(.985)}
.btn.primary{color:#fff;background:linear-gradient(135deg,var(--primary),color-mix(in srgb,var(--primary) 78%%,var(--secondary)));box-shadow:0 6px 16px color-mix(in srgb,var(--primary) 38%%,transparent)}
.btn.ghost{color:var(--text);background:var(--surface);border:1px solid var(--line)}
.btn.danger{color:var(--bad);background:color-mix(in srgb,var(--bad) 9%%,var(--surface));border:1px solid color-mix(in srgb,var(--bad) 28%%,var(--line))}
.btn.link{width:auto;min-height:0;padding:2px 0;color:var(--primary);background:none;justify-content:flex-start}
.desktop .btn:not(.link){width:auto;align-self:flex-start;min-width:180px}
.row.done b{text-decoration:line-through;color:var(--muted)}.row.done .row-ic{background:color-mix(in srgb,var(--ok) 16%%,var(--surface));color:var(--ok)}
.row.gone,.row.hide{display:none}.row.fresh{animation:in .35s}
.on svg{fill:currentColor;color:var(--bad)}
.sheetwrap{display:none;position:absolute;inset:0;z-index:45;background:rgba(15,23,42,.45);align-items:flex-end}
.sheetwrap.open{display:flex}
.bsheet{width:100%%;max-height:82%%;overflow:auto;background:var(--surface);border-radius:26px 26px 0 0;padding:10px 18px 22px;display:flex;flex-direction:column;gap:12px;animation:up .24s ease}
@keyframes up{from{transform:translateY(40px);opacity:0}to{transform:none;opacity:1}}
.grab{width:38px;height:4px;border-radius:4px;background:var(--line);margin:0 auto 4px}
.bsheet h3{font-size:18px;font-weight:700;letter-spacing:-.01em}.bsheet .sub{font-size:13.5px;color:var(--muted);margin-top:-8px}
.bsheet .acts{display:flex;flex-direction:column;gap:8px;margin-top:4px}
.desktop .sheetwrap{align-items:center;justify-content:center}.desktop .bsheet{max-width:440px;border-radius:22px}
.chips{display:flex;gap:8px;overflow-x:auto;padding:2px 0 4px;scrollbar-width:none}.chips::-webkit-scrollbar{display:none}
.chip{flex:none;padding:8px 16px;border-radius:99px;font-size:13.5px;font-weight:650;background:var(--surface);border:1.5px solid var(--line);color:var(--text);white-space:nowrap;transition:all .15s}
.chip.on{background:var(--primary);border-color:var(--primary);color:#fff;box-shadow:0 4px 12px color-mix(in srgb,var(--primary) 32%%,transparent)}
.row.sel{background:var(--soft);box-shadow:inset 3px 0 0 var(--primary)}.row.sel .row-ic{background:var(--primary);color:#fff}
.hide{display:none!important}
.sect{font-size:13px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:4px 2px 0}
.hscroll{display:flex;gap:12px;overflow-x:auto;padding:2px 2px 8px;scrollbar-width:none}.hscroll::-webkit-scrollbar{display:none}
.tile{flex:none;width:150px;display:flex;flex-direction:column;gap:7px;padding:8px;border-radius:16px;background:var(--surface);border:1.5px solid var(--line)}
.tile-media{height:104px;border-radius:11px;display:flex;align-items:center;justify-content:center;color:var(--primary);background:linear-gradient(135deg,color-mix(in srgb,var(--primary) 16%%,var(--surface)),color-mix(in srgb,var(--secondary) 16%%,var(--surface)))}
.tile b{font-size:14px;font-weight:650}.tile .sk{width:60%%}
.tile.sel{border-color:var(--primary);box-shadow:0 0 0 3px color-mix(in srgb,var(--primary) 18%%,transparent)}
.iconbtn{width:42px;height:42px;border-radius:50%%;display:inline-flex;align-items:center;justify-content:center;background:var(--surface);border:1px solid var(--line);color:var(--text);align-self:flex-start}
.fab{position:absolute;right:18px;bottom:18px;z-index:5;width:56px;height:56px;border-radius:50%%;display:flex;align-items:center;justify-content:center;color:#fff;
background:linear-gradient(135deg,var(--primary),var(--secondary));box-shadow:0 10px 24px color-mix(in srgb,var(--primary) 45%%,transparent)}
.field{display:flex;flex-direction:column;gap:7px}.field-label{font-size:13px;font-weight:650;color:var(--text)}
.field-box,.search{display:flex;align-items:center;gap:10px;background:var(--surface);border:1.5px solid var(--line);border-radius:14px;padding:0 14px;color:var(--muted);transition:border-color .15s,box-shadow .15s}
.search{border-radius:999px}
.field-box:focus-within,.search:focus-within,textarea:focus{border-color:var(--primary);box-shadow:0 0 0 4px color-mix(in srgb,var(--primary) 16%%,transparent)}
input[type=text],input[type=email],input[type=password],input[type=number],input[type=date]{flex:1;min-width:0;height:46px;border:0;outline:0;background:none;font:inherit;font-size:15px;color:var(--text)}
textarea{width:100%%;resize:none;background:var(--surface);border:1.5px solid var(--line);border-radius:14px;padding:12px 14px;font:inherit;font-size:15px;color:var(--text);outline:0}
::placeholder{color:color-mix(in srgb,var(--text) 38%%,transparent)}
.segmented{display:flex;gap:4px;padding:4px;background:color-mix(in srgb,var(--text) 7%%,var(--surface));border-radius:14px}
.seg{flex:1;text-align:center;padding:9px 6px;border-radius:11px;font-size:13.5px;font-weight:600;color:var(--muted)}
.seg.on{background:var(--surface);color:var(--primary);box-shadow:0 1px 4px rgba(15,23,42,.14)}
.progress-row{display:flex;justify-content:space-between;font-size:14px;font-weight:600;margin-bottom:8px}.progress-row b{color:var(--primary)}
.bar{height:9px;border-radius:9px;background:var(--line);overflow:hidden}.bar i{display:block;height:100%%;border-radius:9px;background:linear-gradient(90deg,var(--primary),var(--secondary))}
.toggle-row{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:12px 14px;background:var(--surface);border:1px solid var(--line);border-radius:14px;font-size:15px;font-weight:600}
input.switch{appearance:none;width:46px;height:27px;border-radius:99px;background:color-mix(in srgb,var(--text) 18%%,transparent);position:relative;cursor:pointer;transition:background .18s}
input.switch::after{content:"";position:absolute;top:3px;left:3px;width:21px;height:21px;border-radius:50%%;background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.3);transition:transform .18s}
input.switch:checked{background:var(--primary)}input.switch:checked::after{transform:translateX(19px)}
input.checkbox{width:22px;height:22px;accent-color:var(--primary)}
.badge{align-self:flex-start;padding:5px 12px;border-radius:99px;font-size:12.5px;font-weight:700;color:var(--primary);background:var(--soft)}
.alert{display:flex;gap:12px;align-items:flex-start;padding:13px 14px;border-radius:14px;font-size:14px;line-height:1.4;border:1px solid}
.alert div{display:flex;flex-direction:column;gap:2px}.alert span{color:var(--muted)}
.alert.info{background:color-mix(in srgb,var(--primary) 8%%,var(--surface));border-color:color-mix(in srgb,var(--primary) 25%%,var(--line));color:var(--primary)}
.alert.success{background:color-mix(in srgb,var(--ok) 9%%,var(--surface));border-color:color-mix(in srgb,var(--ok) 30%%,var(--line));color:var(--ok)}
.alert.warn{background:color-mix(in srgb,var(--warn) 10%%,var(--surface));border-color:color-mix(in srgb,var(--warn) 32%%,var(--line));color:#b45309}
.alert.danger{background:color-mix(in srgb,var(--bad) 8%%,var(--surface));border-color:color-mix(in srgb,var(--bad) 28%%,var(--line));color:var(--bad)}
.alert b{color:var(--text)}
.alert.vld{display:none}.alert.vld.show{display:flex;animation:shake .35s}
@keyframes shake{20%%{transform:translateX(-5px)}60%%{transform:translateX(5px)}100%%{transform:none}}
.empty{display:flex;flex-direction:column;align-items:center;text-align:center;gap:6px;padding:34px 20px;border:1.5px dashed var(--line);border-radius:18px}
.empty-ic{width:64px;height:64px;border-radius:50%%;display:flex;align-items:center;justify-content:center;background:var(--soft);color:var(--primary);margin-bottom:6px}
.empty h3{font-size:16px;font-weight:700}.empty p{font-size:14px;color:var(--muted);max-width:260px;line-height:1.45}
.copy{font-size:14.5px;line-height:1.55;color:var(--muted)}
.divider{border:0;border-top:1px solid var(--line);margin:4px 0}
.foot{text-align:center;font-size:12.5px;color:var(--muted);padding:10px 0}
.tabbar{display:flex;justify-content:space-around;padding:8px 8px 12px;background:var(--surface);border-top:1px solid var(--line)}
.tab{flex:1;display:flex;flex-direction:column;align-items:center;gap:3px;padding:6px 4px;border-radius:12px;font-size:11.5px;font-weight:600;color:var(--muted);text-align:center}
.tab.active{color:var(--primary)}.tab.active .ico{stroke-width:2.3}
.device.notabs .tabbar{display:none}
.toast{position:absolute;left:50%%;bottom:84px;transform:translate(-50%%,12px);max-width:86%%;padding:10px 16px 10px 14px;border-radius:14px;background:var(--surface);color:var(--text);border:1px solid var(--line);border-left:4px solid var(--primary);box-shadow:0 8px 24px rgba(15,23,42,.14);font-size:13.5px;font-weight:600;opacity:0;pointer-events:none;transition:all .2s;z-index:30}
.toast.show{opacity:1;transform:translate(-50%%,0)}
.toast.err{border-left-color:var(--bad);color:var(--bad)}
.desktop .toast{bottom:28px}
.modal{display:none;position:absolute;inset:0;z-index:40;background:rgba(15,23,42,.5);align-items:center;justify-content:center;padding:24px}
.modal.open{display:flex}.sheet{background:var(--surface);border-radius:20px;padding:22px;width:100%%;max-width:340px;box-shadow:0 20px 50px rgba(0,0,0,.35)}
.sheet h3{font-size:17px;margin-bottom:6px}.sheet p{font-size:14px;color:var(--muted);line-height:1.5;margin-bottom:18px}
.sheet .acts{display:flex;gap:10px}.sheet .btn{min-height:42px}
"""


_AR = {
    "Search": "بحث", "Save changes": "حفظ التغييرات", "Profile updated": "تم تحديث الملف الشخصي",
    "Nothing here yet": "لا يوجد شيء هنا بعد", "Confirm": "تأكيد", "Cancel": "إلغاء",
    "Saved to": "تم الحفظ في", "Could not save": "تعذر الحفظ", "Database not reachable": "قاعدة البيانات غير متاحة",
    "Nothing to delete": "لا يوجد ما يُحذف", "New item": "عنصر جديد",
}


def _t(text: str) -> str:
    """Static preview text, translated when the output language is Arabic."""
    return _AR.get(text, text) if is_rtl() else text


_PROFILE_SCREEN = {
    "name": "Profile",
    "purpose": "Your account, preferences and sign-out.",
    "key_components": [
        {"name": "Profile Header", "type": "Header", "image_type": "user"},
        {"name": "Account Summary", "type": "Card", "icon": "user",
         "styling": "'Signed in', 'Manage your details'"},
        {"name": "Account Settings", "type": "List", "icon": "settings",
         "styling": "'Edit profile', 'Notifications', 'Privacy & security', 'Help & support'"},
        {"name": "Push Notifications", "type": "Switch"},
        {"name": "Log Out Button", "type": "Button", "interaction": "Signs you out and returns to the login screen."},
    ],
}


def needs_profile(design: Dict[str, Any], screens: List[Dict[str, Any]]) -> bool:
    """True when the design points at a profile/account page but never designed one."""
    if any(re.search(r"profile|account|settings", str(s.get("name") or ""), re.I) for s in screens):
        return False
    nav = design.get("navigation") if isinstance(design.get("navigation"), dict) else {}
    texts = [str(i.get("label") if isinstance(i, dict) else i) for i in (nav.get("items") or [])]
    for s in screens:
        for c in (s.get("key_components") or []):
            if isinstance(c, dict):
                texts += [str(c.get("name") or ""), str(c.get("interaction") or "")]
    return any(re.search(r"\b(profile|account)\b", t, re.I) for t in texts)


def build_device_html(design: Dict[str, Any], is_mobile: bool, primary_hex: str, secondary_hex: str,
                      screen_index: int = 0, bridge: Optional[Dict[str, Any]] = None,
                      colors: Optional[Dict[str, str]] = None) -> str:
    """Self-contained HTML for the clickable prototype.

    ``bridge`` (utils.prototype_bridge.build_bridge) lets primary buttons save the text typed on a
    screen into the project database.
    """
    colors = colors or {}
    screens = [s for s in (design.get("screens") if isinstance(design.get("screens"), list) else []) if isinstance(s, dict)]
    names = [str(s.get("name") or f"Screen {i + 1}") for i, s in enumerate(screens)]
    for s, n in zip(screens, names):
        s["name"] = n
    if screens and needs_profile(design, screens):
        screens = screens + [dict(_PROFILE_SCREEN)]
        names.append("Profile")
    screen_index = screen_index if 0 <= screen_index < len(screens) else 0
    typography = design.get("typography") if isinstance(design.get("typography"), dict) else {}
    family = re.sub(r"[^A-Za-z0-9 ,'-]", "", str(typography.get("font_family") or typography.get("body_font") or ""))
    font = (f"'{family.split(',')[0].strip()}', " if family.strip() else "") + "-apple-system,'Segoe UI',Roboto,Inter,system-ui,sans-serif"

    ctx = _Ctx(names, bridge, [])
    nav = _nav_items(design, screens)
    sections = []
    for i, screen in enumerate(screens):
        comps = [c for c in (screen.get("key_components") or screen.get("components") or []) if isinstance(c, dict)]
        has_header = any(_kind(c) == "header" for c in comps)
        ctx.header_done = False
        body = "".join(_render_component(c, screen, ctx, has_header) for c in comps)
        if not body.strip():
            body = ('<div class="empty"><div class="empty-ic">' + _svg("inbox", 30) +
                    f'</div><h3>{_e(screen["name"])}</h3><p>{_e(_first_sentence(screen.get("purpose"), 110) or "No components designed for this screen yet.")}</p></div>')
        appbar = "" if has_header else (
            f'<div class="appbar"><button class="back" data-back="1" aria-label="Back">{_svg("back")}</button>'
            f'<h1>{_e(screen["name"])}</h1></div>')
        notab = ' data-notabs="1"' if _AUTH_SCREEN.search(screen["name"]) else ""
        sections.append(
            f'<section class="screen{" active" if i == screen_index else ""}"{notab} data-screen-name="{_e(screen["name"])}">'
            f'{appbar}<div class="content">{body}</div></section>')
    if not sections:
        sections.append('<section class="screen active" data-screen-name="Preview"><div class="content"><div class="empty">'
                        '<h3>No screens yet</h3><p>Generate a design to see the prototype.</p></div></div></section>')

    def nav_buttons(cls: str) -> str:
        out = ""
        for it in nav:
            attr = f'data-tab="{_e(it["target"])}"'
            icon = _svg(it["icon"], 21) if cls == "tab" else _svg(it["icon"], 17)
            out += f'<button class="{cls}" {attr} data-label="{_e(it["label"])}">{icon}<span>{_e(it["label"])}</span></button>'
        return out

    chrome = '<div class="chrome"><i></i><i></i><i></i><span>app.preview</span></div>' if not is_mobile else ""
    topnav = f'<nav class="topnav">{nav_buttons("navlink")}</nav>' if (not is_mobile and nav) else ""
    tabbar = f'<nav class="tabbar">{nav_buttons("tab")}</nav>' if (is_mobile and nav) else ""
    bridge_json = json.dumps(bridge or None).replace("</", "<\\/")
    t_json = json.dumps({k: _t(v) for k, v in {"saved": "Saved to", "fail": "Could not save", "unreach": "Database not reachable", "nodel": "Nothing to delete", "newitem": "New item", "confirm": "Confirm"}.items()}, ensure_ascii=False)
    css = _CSS % {
        "primary": primary_hex, "secondary": secondary_hex,
        "bg": colors.get("background", "#f6f7fb"), "surface": colors.get("surface", "#ffffff"),
        "text": colors.get("text", "#0f172a"), "font": font,
    }
    rtl = is_rtl()
    if rtl:
        css += ".back svg{transform:scaleX(-1)}body{font-family:Tahoma,'Segoe UI','Noto Sans Arabic',var(--font),sans-serif}"
    return f"""<!doctype html><html{' dir="rtl" lang="ar"' if rtl else ''}><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>{css}</style></head><body>
<div class="stage"><div class="device {'mobile' if is_mobile else 'desktop'}" id="device">
{chrome}{topnav}<div class="screens" id="screens">{''.join(sections)}</div>{tabbar}
<div class="toast" id="toast"></div>
{''.join(ctx.sheets)}
<div class="sheetwrap" id="sheetWrap"><div class="bsheet" id="sheetBody"></div></div>
<div class="modal" id="modal"><div class="sheet"><h3 id="mTitle">{_t("Confirm")}</h3><p id="mBody"></p>
<div class="acts"><button class="btn ghost" id="mNo">{_t("Cancel")}</button><button class="btn primary" id="mYes">{_t("Confirm")}</button></div></div></div>
</div></div>
<script>
const BRIDGE = {bridge_json};
const T = {t_json};
const stack = [];
let lastSave = 0, tt = null;
const $$ = (s, r) => Array.from((r || document).querySelectorAll(s));
function toast(msg, err) {{
  const el = document.getElementById('toast'); el.textContent = msg; el.classList.toggle('err', !!err); el.classList.add('show');
  clearTimeout(tt); tt = setTimeout(() => el.classList.remove('show'), 1800);
}}
function current() {{ const a = document.querySelector('.screen.active'); return a ? a.dataset.screenName : ''; }}
function sync() {{
  const cur = current();
  const act = document.querySelector('.screen.active');
  document.getElementById('device').classList.toggle('has-back', stack.length > 0);
  document.getElementById('device').classList.toggle('notabs', !!(act && act.dataset.notabs));
  $$('.screen').forEach(s => s.classList.toggle('has-back', stack.length > 0));
  $$('[data-tab]').forEach(b => b.classList.toggle('active', b.dataset.tab === cur));
}}
function show(name, push) {{
  const next = document.querySelector('.screen[data-screen-name="' + CSS.escape(name) + '"]');
  if (!next) return false;
  const cur = current();
  if (push && cur && cur !== name) stack.push(cur);
  $$('.screen').forEach(s => s.classList.remove('active'));
  next.classList.add('active'); next.querySelector('.content').scrollTop = 0; sync(); return true;
}}
function back() {{ if (stack.length) show(stack.pop(), false); else toast('Already on the first screen'); }}
function submitScreen(screen) {{
  const cfg = BRIDGE && BRIDGE.screens && BRIDGE.screens[screen];
  if (!cfg) return Promise.resolve(false);
  const data = {{}}, used = [];
  $$('input[data-screen],textarea[data-screen]').forEach(inp => {{
    if (inp.dataset.screen !== screen) return;
    const col = cfg.fields[inp.dataset.field], v = inp.value.trim();
    if (col && v) {{ data[col] = v; used.push(inp); }}
  }});
  if (!used.length) return Promise.resolve(false);
  return fetch(BRIDGE.url, {{method: 'POST', headers: {{'Content-Type': 'text/plain'}},
    body: JSON.stringify({{token: BRIDGE.token, table: cfg.table, data: data}})}})
    .then(r => r.json()).then(j => {{
      lastSave = Date.now();
      if (j.ok) {{ used.forEach(i => i.value = ''); toast(T.saved + ' ' + cfg.table + ' ✓'); }} else toast(j.error || T.fail, true);
      return !!j.ok;
    }}).catch(() => {{ lastSave = Date.now(); toast(T.unreach, true); return false; }});
}}
function rowsOf(scr) {{ return $$('.rows .row:not(.gone)', scr); }}
function act(el) {{
  const kind = el.dataset.act, scr = document.querySelector('.screen.active');
  if (kind === 'like') el.classList.toggle('on');
  else if (kind === 'delete') {{
    const row = rowsOf(scr)[0];
    if (row) row.classList.add('gone'); else toast(T.nodel);
  }} else if (kind === 'done') {{
    const rows = rowsOf(scr), open = rows.find(r => !r.classList.contains('done'));
    if (open) open.classList.add('done'); else rows.forEach(r => r.classList.remove('done'));
    applyFilter(scr);
  }} else if (kind === 'add') {{
    const typed = $$('input[data-screen],textarea[data-screen]', scr).map(i => i.value.trim()).find(Boolean) || '';
    const dest = el.dataset.nav ? document.querySelector('.screen[data-screen-name="' + CSS.escape(el.dataset.nav) + '"]')
      : $$('.screen').find(s => s !== scr && s.querySelector('.rows .row'));
    const model = dest && dest.querySelector('.rows .row');
    if (model) {{
      const row = model.cloneNode(true);
      row.classList.remove('done', 'gone', 'hide'); row.classList.add('fresh');
      row.querySelector('b').textContent = typed || T.newitem;
      model.parentNode.insertBefore(row, model.parentNode.firstChild);
      if (!el.dataset.nav) {{ stack.length = 0; show(dest.dataset.screenName, false); }}
    }}
  }}
}}
function filterByChips(bar) {{
  // Prototype filtering: each chip keeps its share of the items on the screen so the list visibly reacts.
  const scr = bar.closest('.screen'), real = $$('.chip', bar).filter(c => !c.hasAttribute('data-all')), on = real.filter(c => c.classList.contains('on'));
  const items = $$('.rows .row, .card', scr).filter(el => !el.closest('.chips'));
  items.forEach((el, i) => el.classList.toggle('hide', on.length > 0 && !on.some(c => i % real.length === real.indexOf(c))));
}}
function applyFilter(scr) {{
  const on = scr.querySelector('.seg.on'); if (!on) return;
  const t = on.textContent.toLowerCase();
  const wantDone = /done|complete|finish|closed|delivered/.test(t), wantOpen = /active|open|pending|todo|to do|upcoming/.test(t);
  $$('.rows .row', scr).forEach(r => {{
    const d = r.classList.contains('done');
    r.classList.toggle('hide', (wantDone && !d) || (wantOpen && d));
  }});
}}
document.addEventListener('input', ev => {{
  const box = ev.target.closest('.search input'); if (!box) return;
  const q = box.value.trim().toLowerCase(), scr = box.closest('.screen');
  $$('.rows .row', scr).forEach(r => r.classList.toggle('hide', !!q && !r.textContent.toLowerCase().includes(q)));
}});
function openSheet(id) {{
  const t = document.getElementById(id); if (!t) return;
  document.getElementById('sheetBody').innerHTML = t.innerHTML;
  document.getElementById('sheetWrap').classList.add('open');
}}
function closeSheet() {{ document.getElementById('sheetWrap').classList.remove('open'); }}
let pending = null;
document.addEventListener('click', ev => {{
  if (ev.target.id === 'sheetWrap') {{ closeSheet(); return; }}
  const chip = ev.target.closest('[data-chip]');
  if (chip) {{
    const bar = chip.parentNode, multi = bar.dataset.multi !== undefined, all = bar.querySelector('[data-all]');
    if (chip.hasAttribute('data-all')) $$('.chip', bar).forEach(c => c.classList.toggle('on', c === chip));
    else {{
      if (!multi) $$('.chip', bar).forEach(c => {{ if (c !== chip) c.classList.remove('on'); }});
      chip.classList.toggle('on');
      if (all) all.classList.remove('on');
    }}
    if (all && !bar.querySelector('.chip.on')) all.classList.add('on');
    filterByChips(bar);
    if (chip.dataset.nav) show(chip.dataset.nav, true);
    return;
  }}
  const pick = ev.target.closest('[data-select]');
  if (pick) {{ $$('[data-select]', pick.parentNode).forEach(r => r.classList.toggle('sel', r === pick && !r.classList.contains('sel'))); return; }}
  const rd = ev.target.closest('[data-rowdone]');
  if (rd) {{ rd.classList.toggle('done'); applyFilter(rd.closest('.screen')); return; }}
  const sh = ev.target.closest('[data-sheet]');
  if (sh) {{ openSheet(sh.dataset.sheet); return; }}
  const cl = ev.target.closest('[data-close]');
  if (cl) {{
    closeSheet();
    if (cl.dataset.toast) toast(cl.dataset.toast);
    if (cl.dataset.nav) {{ stack.length = 0; show(cl.dataset.nav, false); }}
    return;
  }}
  const seg = ev.target.closest('[data-seg]');
  if (seg) {{
    $$('.seg', seg.parentNode).forEach(s => s.classList.toggle('on', s === seg));
    applyFilter(seg.closest('.screen')); return;
  }}
  const tab = ev.target.closest('[data-tab]');
  if (tab) {{ stack.length = 0; show(tab.dataset.tab, false); return; }}
  const el = ev.target.closest('[data-nav],[data-back],[data-toast],[data-submit],[data-confirm],[data-act]');
  if (!el) return;
  const run = () => {{
    const go = () => {{
      if (el.dataset.act) act(el);
      if (el.dataset.nav) show(el.dataset.nav, true);
      else if (el.dataset.back !== undefined) back();
      else if (el.dataset.toast && Date.now() - lastSave > 1500) toast(el.dataset.toast);
    }};
    if (el.dataset.submit !== undefined || el.dataset.act === 'add') {{
      const scr = document.querySelector('.screen.active');
      const filled = $$('input[data-screen],textarea[data-screen]', scr).some(i => i.value.trim());
      const vld = $$('.vld', scr);
      if (!filled && vld.length) {{ vld.forEach(v => {{ v.classList.remove('show'); void v.offsetWidth; v.classList.add('show'); }}); return; }}
      submitScreen(current()).then(go);
    }} else go();
  }};
  if (el.dataset.confirm) {{
    pending = run;
    document.getElementById('mTitle').textContent = (el.textContent || T.confirm).trim();
    document.getElementById('mBody').textContent = el.dataset.confirm;
    document.getElementById('modal').classList.add('open');
  }} else run();
}});
document.getElementById('mNo').onclick = () => {{ document.getElementById('modal').classList.remove('open'); pending = null; }};
document.getElementById('mYes').onclick = () => {{ document.getElementById('modal').classList.remove('open'); if (pending) pending(); pending = null; }};
sync();
</script></body></html>"""

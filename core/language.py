"""Output language for the agents (English or Arabic).

The setting is process-wide and is applied by the UI on every rerun before any agent runs.
Only human-readable text changes language; JSON keys, enum values, IDs and database
identifiers stay English because downstream code depends on them.
"""
from typing import Dict

LANGUAGES: Dict[str, str] = {"en": "English", "ar": "العربية"}
_RTL = {"ar"}
_current = "en"

_ARABIC_NOTE = (
    "LANGUAGE: Write every human-readable text value (names, descriptions, user stories, acceptance "
    "criteria, screen and component text, messages, labels, notes, recommendations) in Modern Standard "
    "Arabic. Keep in English: JSON keys, enum values (High/Medium/Low, mobile/web, data types, "
    "assigned agents), IDs (F1, T1, S1), and all database identifiers (table and column names). "
    "Name UI components by their visible label only, without a component-type word (write \"تسجيل الدخول\", not \"زر تسجيل الدخول\"). "
    "Output valid JSON only."
)
_ARABIC_CHAT_NOTE = (
    "LANGUAGE: Converse in Modern Standard Arabic. Keep the same bracketed option format, but write the "
    "options in Arabic and make the last option \"أخرى (يرجى التحديد)\"."
)


def set_language(code: str) -> None:
    global _current
    _current = code if code in LANGUAGES else "en"


def get_language() -> str:
    return _current


def is_rtl(code: str = "") -> bool:
    return (code or _current) in _RTL


def with_language(text: str, chat: bool = False) -> str:
    """Append the language instruction to a prompt (no-op for English)."""
    if _current == "ar":
        return f"{text}\n\n{_ARABIC_CHAT_NOTE if chat else _ARABIC_NOTE}"
    return text

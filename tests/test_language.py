"""Arabic output support: prompt directive, Unicode-safe matching and RTL preview (no API calls)."""
import pytest

from core import language
from core.language import set_language, with_language
from core.prompts import build_planning_prompt, build_quality_prompt
from core.traceability import build_trace_matrix
from ui.prototype_render import build_device_html
from tests.conftest import SAMPLE_DESIGN


@pytest.fixture(autouse=True)
def reset_language():
    yield
    set_language("en")


def test_english_prompts_unchanged():
    assert with_language("x") == "x"
    assert "LANGUAGE:" not in build_quality_prompt("{}", "{}", "{}")


def test_arabic_directive_in_every_builder():
    set_language("ar")
    assert "Modern Standard Arabic" in build_planning_prompt("{}", "{}")
    assert "JSON keys" in build_quality_prompt("{}", "{}", "{}")
    assert "أخرى" in with_language("x", chat=True)


def test_unknown_code_falls_back_to_english():
    set_language("fr")
    assert language.get_language() == "en" and not language.is_rtl()


def test_arabic_features_link_to_arabic_screens():
    req = {"features": [{"id": "F1", "name": "تصفح الزهور", "description": "عرض الزهور",
                         "acceptance_criteria": ["x"]}]}
    design = {"screens": [{"name": "تصفح الزهور", "purpose": "عرض الزهور", "key_components": []}]}
    assert build_trace_matrix(req, design)["features"][0]["screens"] == ["تصفح الزهور"]


def test_preview_is_rtl_only_in_arabic():
    d = {**SAMPLE_DESIGN, "screens": [{"name": "الرئيسية", "purpose": "", "key_components": []}]}
    assert 'dir="rtl"' not in build_device_html(d, True, "#111111", "#222222")
    set_language("ar")
    html = build_device_html(d, True, "#111111", "#222222")
    assert 'dir="rtl"' in html and "الرئيسية" in html


def test_arabic_component_type_words_are_stripped():
    from ui.prototype_render import _clean
    assert _clean("زر تسجيل الدخول") == "تسجيل الدخول"
    assert _clean("حقل إدخال البريد الإلكتروني") == "البريد الإلكتروني"
    assert _clean("زر") == "زر"  # never empties the label

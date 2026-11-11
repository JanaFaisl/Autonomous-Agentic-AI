"""Unit tests for the Arabic-vs-English evaluation metrics (no API calls)."""
from evaluation.arabic_eval import compare, identifier_integrity, language_purity


def test_purity_ignores_identifiers_and_terms():
    ar = ["يستطيع المستخدم تسجيل الدخول عبر REST API وحفظ user_id في قاعدة PostgreSQL"]
    assert language_purity(ar, "ar") > 0.95
    assert language_purity(["The user can log in and save tasks"], "en") == 1.0


def test_purity_penalises_stray_english_words():
    mixed = ["المستخدم يستطيع create a new account بسهولة"]
    assert language_purity(mixed, "ar") < 0.8


def test_identifier_integrity_flags_translated_enums():
    run = {"req": {"features": [{"id": "F1", "priority": "High"}, {"id": "ف1", "priority": "عالية"}],
                   "database_schema": {"tables": [{"name": "users", "columns": [{"name": "id"}, {"name": "اسم"}]}]}},
           "plan": {"tasks": [{"priority": "high", "assigned_agent": "Developer"}]},
           "quality": None, "design": None}
    assert identifier_integrity(run) == (6, 9)


def test_compare_is_paired_by_project():
    en = [{"project": 1, "x": 1.0}, {"project": 2, "x": 3.0}]
    ar = [{"project": 1, "x": 2.0}, {"project": 2, "x": 3.0}, {"project": 9, "x": 7.0}]
    import evaluation.arabic_eval as ev
    saved = list(ev.METRICS)
    ev.METRICS[:] = [("x", "X")]
    try:
        row = compare(en, ar)[0]
    finally:
        ev.METRICS[:] = saved
    assert row["n_projects"] == 2 and row["diff"] == 0.5

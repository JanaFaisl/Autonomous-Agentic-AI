"""Arabic widget labels (no API calls)."""
import pytest

from core.language import set_language
from ui.i18n import tr


@pytest.fixture(autouse=True)
def reset_language():
    yield
    set_language("en")


def test_english_is_untouched():
    assert tr("🛠️ Fix these issues") == "🛠️ Fix these issues"


def test_arabic_keeps_emoji_prefix():
    set_language("ar")
    assert tr("🛠️ Fix these issues") == "🛠️ إصلاح هذه المشاكل"
    assert tr("Confirm requirements") == "تأكيد المتطلبات"


def test_arabic_patterns_and_unknown_text():
    set_language("ar")
    assert tr("Required fixes (3)") == "الإصلاحات المطلوبة (3)"
    assert tr("📥 Download orders as CSV") == "📥 تنزيل orders (CSV)"
    assert tr("something new") == "something new"
    assert tr(5) == 5


def test_headings_messages_and_fragments():
    set_language("ar")
    from ui.i18n import tr_text
    assert tr_text("## 🔗 Traceability & Coverage") == "## 🔗 التتبّع والتغطية"
    assert tr_text("**Acceptance Criteria**") == "**معايير القبول**"
    assert tr_text("No columns defined.") == "لم تُعرَّف أعمدة."
    assert "فحوص ناجحة" in tr_text("**1/4 checks passed** · 3 issues (2 high, 1 medium, 0 low)")
    assert tr_text("<style>.a{display:none}</style>") == "<style>.a{display:none}</style>"
    assert tr_text("وصف عربي") == "وصف عربي"
    set_language("en")
    assert tr_text("No columns defined.") == "No columns defined."


def test_dataframe_gets_arabic_headers_and_values():
    import pandas as pd
    from ui.i18n import localize_df
    df = pd.DataFrame([{"ID": "T1", "Agent": "DBA", "Priority": "high", "Gaps": "screens, tasks"}])
    assert list(localize_df(df).columns) == ["ID", "Agent", "Priority", "Gaps"]  # English: untouched
    set_language("ar")
    out = localize_df(df)
    assert list(out.columns) == ["النواقص", "الأولوية", "الوكيل", "المعرّف"]
    assert out.iloc[0].tolist() == ["الشاشات, المهام", "عالية", "مدير قاعدة البيانات", "T1"]
    assert df.iloc[0]["Priority"] == "high"  # the original is not modified


def test_schema_tables_show_arabic_names_types_and_defaults():
    import pandas as pd
    from ui.i18n import localize_df, set_schema_labels
    from utils.schema_labels import schema_names
    set_language("ar")
    set_schema_labels({"tables": {"users": "المستخدمون", "roles": "الأدوار"},
                       "columns": {"role_id": "معرّف الدور", "id": "المعرّف"}})
    try:
        df = pd.DataFrame([{"From Table": "users", "To Table": "roles", "Type": "many-to-one",
                            "FK": "role_id", "References": "id", "Notes": "x"}])
        out = localize_df(df)
        assert out.iloc[0]["من الجدول"] == "المستخدمون" and out.iloc[0]["إلى الجدول"] == "الأدوار"
        assert out.iloc[0]["النوع"] == "متعدد إلى واحد" and out.iloc[0]["مفتاح أجنبي"] == "معرّف الدور"
        cols = pd.DataFrame([{"Table": "users", "Column": "id", "Type": "varchar(50)", "Default": "gen_random_uuid()"},
                             {"Table": "users", "Column": "x", "Type": "uuid", "Default": "true"}])
        o = localize_df(cols)
        assert o["النوع"].tolist() == ["نص (50)", "معرّف فريد"]
        assert o["الافتراضي"].tolist() == ["معرّف عشوائي", "نعم"]
        assert o["العمود"].tolist() == ["المعرّف", "x"]  # no label known -> identifier kept
    finally:
        set_schema_labels({})
    schema = {"tables": [{"name": "users", "columns": [{"name": "id"}, {"name": "email"}]},
                         {"name": "tasks", "columns": [{"name": "id"}]}]}
    assert schema_names(schema) == {"tables": ["users", "tasks"], "columns": ["id", "email"]}

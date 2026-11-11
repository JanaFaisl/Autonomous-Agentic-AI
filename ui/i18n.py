"""Arabic labels for the app's own widgets (buttons, tabs, expanders, form fields).

Labels are translated at the Streamlit call, so call sites keep their English text and widget
return values are unchanged (options are translated through ``format_func``, never replaced).
Leading emoji are kept; text not in the table is left as is.
"""
import re
from typing import Any, Callable, Dict, List, Tuple

import streamlit as st

from core.language import is_rtl

_AR: Dict[str, str] = {
    "Chat": "المحادثة", "Review": "المراجعة", "Design": "التصميم", "Database": "قاعدة البيانات",
    "Planning": "التخطيط", "Quality": "الجودة", "Support": "الدعم",
    "Turn this conversation into structured Requirements": "حوّل هذه المحادثة إلى متطلبات منظّمة",
    "Confirm requirements": "تأكيد المتطلبات",
    "Continue Chat": "متابعة المحادثة",
    "Type your message…": "اكتب رسالتك…",
    "Download Requirements JSON": "تنزيل المتطلبات (JSON)",
    "Download project (.zip)": "تنزيل المشروع (.zip)",
    "Live Preview": "معاينة مباشرة", "Design Specs": "مواصفات التصميم",
    "Device view": "عرض الجهاز", "Desktop": "سطح المكتب", "Mobile": "جوال", "Select screen": "اختر الشاشة",
    "Download Design JSON": "تنزيل التصميم (JSON)",
    "View raw response": "عرض الاستجابة الخام", "Raw design output": "مخرجات التصميم الخام",
    "Technical details": "تفاصيل تقنية", "Parse Error Details": "تفاصيل خطأ التحليل",
    "Raw Response (for debugging)": "الاستجابة الخام (للتصحيح)", "Extracted Content": "المحتوى المستخرج",
    "Extracted JSON (attempted)": "JSON المستخرج (محاولة)", "Debug: Design Structure": "تصحيح: بنية التصميم",
    "Create Dataset": "إنشاء مجموعة بيانات", "Import Existing Dataset": "استيراد مجموعة بيانات موجودة",
    "Source": "المصدر", "Start from": "ابدأ من", "Table name": "اسم الجدول", "Number of columns": "عدد الأعمدة",
    "Dataset file": "ملف مجموعة البيانات", "Table": "الجدول", "Name": "الاسم", "Type": "النوع", "PK": "مفتاح أساسي",
    "Nullable": "يقبل القيمة الفارغة", "File URL": "رابط الملف", "Format": "الصيغة",
    "Table name (blank = derive from the file name)": "اسم الجدول (فارغ = من اسم الملف)",
    "Max rows to import": "الحد الأقصى للصفوف", "Import File": "استيراد الملف", "Fetch & Import": "جلب واستيراد",
    "Database URL": "رابط قاعدة البيانات", "Import Schema": "استيراد المخطط", "Dataset ID": "معرّف مجموعة البيانات",
    "Split (leave blank for the first available)": "التقسيم (اتركه فارغاً لأول متاح)", "Import Dataset": "استيراد مجموعة البيانات",
    "Select table to view:": "اختر جدولاً للعرض:", "Assumptions": "الافتراضات",
    "Download Schema JSON": "تنزيل المخطط (JSON)", "Download PostgreSQL DDL": "تنزيل DDL لـ PostgreSQL",
    "Download SQLite DDL": "تنزيل DDL لـ SQLite", "Download Columns as CSV": "تنزيل الأعمدة (CSV)",
    "Download Relationships as CSV": "تنزيل العلاقات (CSV)", "Download Sample Data as CSV": "تنزيل بيانات العينة (CSV)",
    "Generate plan": "إنشاء الخطة", "Download Plan JSON": "تنزيل الخطة (JSON)",
    "Generate quality report": "إنشاء تقرير الجودة", "Download Quality Report JSON": "تنزيل تقرير الجودة (JSON)",
    "Fix these issues": "إصلاح هذه المشاكل", "View full JSON": "عرض JSON كاملاً",
    "Mark current requirements as baseline": "اعتماد المتطلبات الحالية كخط أساس", "Set baseline now": "تعيين خط الأساس الآن",
    "What if I change…": "ماذا لو غيّرت…", "Download Traceability JSON": "تنزيل التتبّع (JSON)",
    "Download test cases JSON": "تنزيل حالات الاختبار (JSON)", "Validate & self-heal": "تحقق وأصلح تلقائياً",
    "App documentation": "وثائق التطبيق", "Generate support package": "إنشاء حزمة الدعم",
    "Download Support Package JSON": "تنزيل حزمة الدعم (JSON)",
}

_AR_TEXT: Dict[str, str] = {
    '**LLM Requirements Analyzer** helps you turn natural language into structured requirements, design, and project governance using AI agents (TSPi-style).\n\n**Tabs and workflow**\n- **Chat** — Describe your project in plain language; the Requirements Agent turns it into structured requirements.\n- **Review** — Inspect and edit requirements, then lock them for downstream use.\n- **Design** — Generate interface/screen design from requirements; edit and lock the design.\n- **Database** — Derive or edit database schema from requirements.\n- **Planning** — Create cycle/sprint plans and briefs from requirements and design.\n- **Quality** — Run the Quality Manager Agent to gate-check requirements, design, and DB consistency; view checklist, issues, and recommendations.\n- **Support** — Generate a support governance package (baselines, app documentation, glossary) and read this documentation.\n\n**How to use**\n1. Set your **Anthropic API key** (`ANTHROPIC_API_KEY` in `.env`).\n2. In **Chat**, describe the app; use **Review** to finalize requirements.\n3. Optionally run **Design**, **Database**, and **Planning**; then **Quality** to validate.\n4. In **Support**, generate the governance package when requirements (and optionally design/DB/plan) are ready.': '**محلّل المتطلبات بالذكاء الاصطناعي** يساعدك على تحويل اللغة الطبيعية إلى متطلبات منظّمة وتصميم وحوكمة للمشروع باستخدام وكلاء ذكاء اصطناعي (على نمط TSPi).\n\n**التبويبات وسير العمل**\n- **المحادثة** — صِف مشروعك بلغة بسيطة؛ يحوّله وكيل المتطلبات إلى متطلبات منظّمة.\n- **المراجعة** — افحص المتطلبات وعدّلها ثم اعتمدها للاستخدام في الخطوات التالية.\n- **التصميم** — أنشئ تصميم الواجهات والشاشات من المتطلبات؛ عدّله واعتمده.\n- **قاعدة البيانات** — استنتج مخطط قاعدة البيانات من المتطلبات أو عدّله.\n- **التخطيط** — أنشئ خطط الدورات/السبرنتات والملخصات من المتطلبات والتصميم.\n- **الجودة** — شغّل وكيل مدير الجودة لفحص اتساق المتطلبات والتصميم وقاعدة البيانات؛ اطّلع على قائمة الفحص والمشاكل والتوصيات.\n- **الدعم** — أنشئ حزمة حوكمة الدعم (خطوط الأساس ووثائق التطبيق والمصطلحات) واقرأ الوثائق.\n\n**طريقة الاستخدام**\n1. عيّن **مفتاح Anthropic API** (`ANTHROPIC_API_KEY` في ملف `.env`).\n2. في **المحادثة** صِف التطبيق، ثم استخدم **المراجعة** لإنهاء المتطلبات.\n3. شغّل اختيارياً **التصميم** و**قاعدة البيانات** و**التخطيط**؛ ثم **الجودة** للتحقق.\n4. في **الدعم** أنشئ حزمة الحوكمة عند جاهزية المتطلبات (واختيارياً التصميم وقاعدة البيانات والخطة).',
    "Quality Manager Agent": "وكيل مدير الجودة", "Support Manager Agent": "وكيل مدير الدعم",
    "Visual Prototype": "النموذج المرئي", "Planning Manager Agent": "وكيل مدير التخطيط",
    "Review Requirements": "مراجعة المتطلبات", "Objects (Table Rows)": "العناصر (صفوف الجداول)",
    "Traceability & Coverage": "التتبّع والتغطية", "Browse Database": "استعراض قاعدة البيانات",
    "Database Schema": "مخطط قاعدة البيانات", "Change impact": "أثر التغيير", "Development plan": "خطة التطوير",
    "Inputs": "المدخلات", "Quality assessment report": "تقرير تقييم الجودة", "Requirements coverage": "تغطية المتطلبات",
    "Support governance package": "حزمة حوكمة الدعم", "Test cases": "حالات الاختبار",
    "Validate generated code": "التحقق من الشيفرة المولّدة", "Typography": "الخطوط", "Animations & Icons": "الحركات والأيقونات",
    "Create New Dataset": "إنشاء مجموعة بيانات جديدة", "Color Palette": "لوحة الألوان", "User Roles": "أدوار المستخدمين",
    "Actual Database Records": "سجلات قاعدة البيانات الفعلية", "All Columns (Unified View)": "كل الأعمدة (عرض موحّد)",
    "Detailed Table View": "عرض الجدول التفصيلي", "Sample Data (Table Records)": "بيانات العينة (سجلات الجدول)",
    "Project": "المشروع", "Non-Functional Requirements": "المتطلبات غير الوظيفية", "Overview": "نظرة عامة",
    "Responsive & Accessibility": "التجاوب وإمكانية الوصول", "Inspect a table": "فحص جدول",
    "All Relationships (Unified View)": "كل العلاقات (عرض موحّد)", "Import Existing Dataset": "استيراد مجموعة بيانات موجودة",
    "Features": "الميزات", "Technical Requirements": "المتطلبات التقنية", "Navigation": "التنقّل", "UI Components": "مكوّنات الواجهة",
    "Acceptance Criteria": "معايير القبول", "Backend": "الخلفية", "Baseline artifacts": "مخرجات خط الأساس",
    "Documentation (your app)": "وثائق (تطبيقك)", "Frontend": "الواجهة الأمامية", "Glossary": "المصطلحات",
    "Not linked to any requirement": "غير مرتبط بأي متطلب", "Platform": "المنصة", "Risks": "المخاطر", "Sizes": "الأحجام",
    "Technologies": "التقنيات", "User Stories": "قصص المستخدمين", "Weights": "الأوزان", "Sample Data": "بيانات العينة",
    "Indexes": "الفهارس", "Relationships": "العلاقات",
    "Add at least one column.": "أضف عموداً واحداً على الأقل.",
    "Agent returned minimal content. Try generating again or add more detail to requirements.": "أعاد الوكيل محتوى قليلاً. أعد المحاولة أو أضف تفاصيل أكثر للمتطلبات.",
    "Agent returned minimal content. Try generating again or ensure requirements/design are complete.": "أعاد الوكيل محتوى قليلاً. أعد المحاولة أو تأكد من اكتمال المتطلبات والتصميم.",
    "Agent returned minimal content. Try generating again.": "أعاد الوكيل محتوى قليلاً. أعد المحاولة.",
    "Anthropic API key missing. Add ANTHROPIC_API_KEY to .env": "مفتاح Anthropic API مفقود. أضف ANTHROPIC_API_KEY إلى ملف .env",
    "Anthropic API key missing. Add ANTHROPIC_API_KEY to your .env file.": "مفتاح Anthropic API مفقود. أضف ANTHROPIC_API_KEY إلى ملف .env",
    "Anthropic API key missing. Add ANTHROPIC_API_KEY to your .env file": "مفتاح Anthropic API مفقود. أضف ANTHROPIC_API_KEY إلى ملف .env",
    "Anthropic API key not found.": "لم يُعثر على مفتاح Anthropic API.",
    "Central orchestrator (Project Manager Agent) runs Requirements → Database → Design → Planning → Quality → Support and stores artifacts.": "المنسّق المركزي (وكيل مدير المشروع) يشغّل: المتطلبات ← قاعدة البيانات ← التصميم ← التخطيط ← الجودة ← الدعم، ويحفظ المخرجات.",
    "Choose one option to continue 👇": "اختر خياراً للمتابعة 👇",
    "Confirm requirements first (Review tab).": "أكّد المتطلبات أولاً (تبويب المراجعة).",
    "Connected successfully, but no tables were found.": "تم الاتصال بنجاح، لكن لم يُعثر على جداول.",
    "Could not tell the file format from the URL. Pick one explicitly in the Format dropdown.": "تعذّر تحديد صيغة الملف من الرابط. اخترها من قائمة الصيغة.",
    "Coverage score": "درجة التغطية",
    "Creates an executable MVP (Minimum Viable Product) plan (tasks, risks) from your requirements and design. Generate requirements and design first.": "ينشئ خطة منتج أولي قابل للتنفيذ (مهام ومخاطر) من متطلباتك وتصميمك. أنشئ المتطلبات والتصميم أولاً.",
    "Database helper not available. Please check db.py is accessible.": "أداة قاعدة البيانات غير متاحة. تحقق من إمكانية الوصول إلى db.py.",
    "Database helper not available. Please check db/db_service.py is accessible.": "أداة قاعدة البيانات غير متاحة. تحقق من إمكانية الوصول إلى db/db_service.py.",
    "Define each column. Leave a column name blank to skip it.": "عرّف كل عمود. اترك اسم العمود فارغاً لتخطيه.",
    "Downloading…": "جارٍ التنزيل…",
    "Features with gaps": "ميزات بها نواقص",
    "Fixing issues…": "جارٍ إصلاح المشاكل…",
    "Generate a database schema or design first.": "أنشئ مخطط قاعدة بيانات أو تصميماً أولاً.",
    "Generate and confirm requirements first (Chat → Review) to enable planning.": "أنشئ المتطلبات وأكّدها أولاً (المحادثة ← المراجعة) لتفعيل التخطيط.",
    "Generate requirements first (Chat → Review) to enable quality checks.": "أنشئ المتطلبات أولاً (المحادثة ← المراجعة) لتفعيل فحوص الجودة.",
    "Generate requirements first (Chat → Review) to enable support package generation.": "أنشئ المتطلبات أولاً (المحادثة ← المراجعة) لتفعيل إنشاء حزمة الدعم.",
    "Generate requirements first (Chat → Turn into structured Requirements).": "أنشئ المتطلبات أولاً (المحادثة ← تحويل إلى متطلبات منظّمة).",
    "Generating plan…": "جارٍ إنشاء الخطة…", "Generating quality report…": "جارٍ إنشاء تقرير الجودة…",
    "Generating support governance package…": "جارٍ إنشاء حزمة حوكمة الدعم…",
    "Includes requirements, database schema/DDL, design, plan, quality report, and support package as JSON/SQL — plus, under app/, an actual runnable Express+SQLite backend and React (Vite) frontend scaffolded from your specific tables and screens. Run `npm install` in app/backend/ and app/frontend/ to try it.": "يتضمن المتطلبات ومخطط قاعدة البيانات وDDL والتصميم والخطة وتقرير الجودة وحزمة الدعم بصيغة JSON/SQL، وفي المجلد app/ خلفية Express+SQLite وواجهة React (Vite) قابلتان للتشغيل مبنيتان على جداولك وشاشاتك. شغّل `npm install` في app/backend/ وapp/frontend/ للتجربة.",
    "Links every requirement to the screens, tables, tasks and acceptance criteria that cover it, shows what a change would affect, and validates the generated code. No API calls needed.": "يربط كل متطلب بالشاشات والجداول والمهام ومعايير القبول التي تغطيه، ويُظهر ما سيتأثر بأي تغيير، ويتحقق من الشيفرة المولّدة. لا يحتاج إلى استدعاءات API.",
    "No Database Agent schema yet — generate requirements from the Chat tab to pick from agent-designed tables here. Define columns manually below.": "لا يوجد مخطط من وكيل قاعدة البيانات بعد — أنشئ المتطلبات من تبويب المحادثة لاختيار جداول مصمّمة هنا. أو عرّف الأعمدة يدوياً أدناه.",
    "No acceptance criteria in the requirements yet.": "لا توجد معايير قبول في المتطلبات بعد.",
    "No columns defined.": "لم تُعرَّف أعمدة.", "No design generated yet.": "لم يُنشأ تصميم بعد.",
    "No features found. Try providing more detail.": "لم يُعثر على ميزات. جرّب تقديم تفاصيل أكثر.",
    "No requirements yet. Generate them from the Chat tab.": "لا توجد متطلبات بعد. أنشئها من تبويب المحادثة.",
    "No steps were run.": "لم تُنفَّذ أي خطوات.",
    "No tables for this project yet. Use **Create Dataset** or **Import Existing Dataset** above to add one.": "لا توجد جداول لهذا المشروع بعد. استخدم **إنشاء مجموعة بيانات** أو **استيراد مجموعة بيانات موجودة** أعلاه لإضافة جدول.",
    "No tables found in requirements schema.": "لم يُعثر على جداول في مخطط المتطلبات.",
    "No tables were generated. Try regenerating requirements with more details.": "لم تُنشأ جداول. أعد إنشاء المتطلبات بتفاصيل أكثر.",
    "No technical requirements returned.": "لم تُرجَع متطلبات تقنية.",
    "Only http:// and https:// URLs are supported.": "الروابط المدعومة هي http:// وhttps:// فقط.",
    "Parsing…": "جارٍ التحليل…",
    "Paste a direct link to a data file — a GitHub raw URL, a public bucket object, or a Google Sheets CSV export link.": "الصق رابطاً مباشراً لملف بيانات — رابط GitHub الخام أو ملف في مخزن عام أو رابط تصدير CSV من Google Sheets.",
    "Please choose a file to upload.": "يرجى اختيار ملف لرفعه.", "Please fill all fields.": "يرجى ملء جميع الحقول.",
    "Please provide a database URL.": "يرجى إدخال رابط قاعدة البيانات.", "Please provide a dataset ID.": "يرجى إدخال معرّف مجموعة البيانات.",
    "Please provide a file URL.": "يرجى إدخال رابط الملف.",
    "Produces support governance artifacts: documentation of your app, baseline artifacts, and glossary.": "ينتج مخرجات حوكمة الدعم: وثائق تطبيقك ومخرجات خط الأساس والمصطلحات.",
    "Project Manager Agent orchestrating pipeline…": "وكيل مدير المشروع ينسّق خط المعالجة…",
    "Requirements match the last pipeline run — nothing to re-run.": "المتطلبات مطابقة لآخر تشغيل — لا حاجة لإعادة التشغيل.",
    "Roles": "الأدوار", "Run the pipeline (or set a baseline) to track edits.": "شغّل خط المعالجة (أو عيّن خط أساس) لتتبّع التعديلات.",
    "Runs the schema checks, executes the SQLite DDL, and syntax-checks the backend (needs Node). On failure, the owning agent is asked to fix it using the error messages.": "يشغّل فحوص المخطط وينفّذ DDL الخاص بـ SQLite ويفحص صياغة الخلفية (يحتاج Node). عند الفشل يُطلب من الوكيل المسؤول الإصلاح اعتماداً على رسائل الخطأ.",
    "Sends High/Medium issues to the owning agent (requirements → database → design), then re-reviews. ~1–3 min.": "يرسل المشاكل العالية والمتوسطة إلى الوكيل المسؤول (المتطلبات ← قاعدة البيانات ← التصميم) ثم يعيد المراجعة. نحو 1–3 دقائق.",
    "Table name is required.": "اسم الجدول مطلوب.", "Tables": "الجداول", "Tables shown": "الجداول المعروضة",
    "The dataset returned no rows.": "لم تُرجِع مجموعة البيانات أي صفوف.", "The file contained no rows.": "لا يحتوي الملف على صفوف.",
    "Total rows": "إجمالي الصفوف",
    "Traceability & coverage of every requirement, plus an agent review of requirements, design, and database schema with a checklist, issues, and required fixes.": "تتبّع وتغطية كل متطلب، إضافة إلى مراجعة وكيل للمتطلبات والتصميم ومخطط قاعدة البيانات مع قائمة فحص ومشاكل وإصلاحات مطلوبة.",
    "Upload a CSV, TSV, JSON, JSONL, Excel, or Parquet file. Columns and types are inferred from the contents.": "ارفع ملف CSV أو TSV أو JSON أو JSONL أو Excel أو Parquet. تُستنتج الأعمدة والأنواع من المحتوى.",
    "Validating…": "جارٍ التحقق…",
    "Requirements were not generated correctly. Open Review tab to see raw output/errors.": "لم تُنشأ المتطلبات بشكل صحيح. افتح تبويب المراجعة لرؤية المخرجات والأخطاء.",
    "Design generated successfully!": "تم إنشاء التصميم بنجاح!",
    "Requirements generated — review them before continuing.": "تم إنشاء المتطلبات — راجعها قبل المتابعة.",
    "Analyzer not initialized. Please check your configuration.": "لم يُهيّأ المحلّل. يرجى التحقق من الإعدادات.",
    "Generating UI/UX design... This may take 30-60 seconds...": "جارٍ إنشاء تصميم الواجهة... قد يستغرق 30–60 ثانية...",
    "Analyzing requirements…": "جارٍ تحليل المتطلبات…",
    "Describe your idea in chat → review requirements → get a design, database, plan and more.": "صِف فكرتك في المحادثة ← راجع المتطلبات ← احصل على التصميم وقاعدة البيانات والخطة وغير ذلك.",
    "Hey there! 👋 \n\nWhat's the main idea behind your project?": "أهلاً بك! 👋 \n\nما الفكرة الرئيسية وراء مشروعك؟",
    "Permanently drops these tables and their rows. This project's tables are kept.": "يحذف هذه الجداول وصفوفها نهائياً. جداول هذا المشروع تبقى.",
    "I understand this cannot be undone": "أفهم أنه لا يمكن التراجع عن هذا",
    "Delete tables from earlier sessions": "حذف جداول الجلسات السابقة",
    "Requirements Agent": "وكيل المتطلبات", "Requirements complete": "المتطلبات مكتملة", "Design consistent with requirements": "التصميم متّسق مع المتطلبات", "Database matches requirements": "قاعدة البيانات تطابق المتطلبات", "Non-functional requirements defined": "المتطلبات غير الوظيفية محدّدة", "Done": "تم", "Priority": "الأولوية",
}
# Fragments of formatted (f-string) messages; applied by substring when no exact entry matches.
_FRAGMENTS: List[Tuple[str, str]] = [
    ("Design:", "التصميم:"), ("DB schema:", "مخطط قاعدة البيانات:"), ("MVP plan:", "خطة المنتج الأولي:"),
    ("Requirements ✅", "المتطلبات ✅"), ("feature(s)", "ميزة"), ("Model:", "النموذج:"), ("Generated:", "تاريخ الإنشاء:"),
    ("Description**:", "الوصف**:"), ("Database**:", "قاعدة البيانات**:"), ("Screens:", "الشاشات:"), ("Tables:", "الجداول:"),
    ("Tasks:", "المهام:"), ("Re-run:", "إعادة التشغيل:"), ("Affected screens:", "الشاشات المتأثرة:"),
    ("Affected tables:", "الجداول المتأثرة:"), ("Affected tasks:", "المهام المتأثرة:"),
    ("case(s) generated from acceptance criteria; the export includes runnable stubs.", "حالة اختبار مولّدة من معايير القبول؛ يتضمن التصدير قوالب قابلة للتشغيل."),
    ("checks passed", "فحوص ناجحة"), ("issues (", "مشكلة ("), (" high,", " عالية،"), (" medium,", " متوسطة،"), (" low)", " منخفضة)"),
    ("Since the last pipeline run — added:", "منذ آخر تشغيل — المضاف:"), ("modified:", "المعدَّل:"), ("removed:", "المحذوف:"),
    ("gaps:", "النواقص:"), ("problem(s) remain", "مشكلة متبقية"), ("problem(s), asked:", "مشكلة، المطلوب من:"),
    ("Last fix run:", "آخر تشغيل للإصلاح:"), ("failing checks", "فحوص فاشلة"),
    ("Pipeline finished. Completed:", "اكتمل خط المعالجة. المكتمل:"), ("step failed:", "فشلت الخطوة:"),
    ("Font family:", "عائلة الخط:"), ("Heading font:", "خط العناوين:"), ("Body font:", "خط النص:"),
    ("Saved requirements", "تم حفظ المتطلبات"),
    ("Saved design", "تم حفظ التصميم"), ("Error generating design:", "خطأ في إنشاء التصميم:"),
    ("Parse Error:", "خطأ التحليل:"), ("JSON Error:", "خطأ JSON:"), ("Add row to", "إضافة صف إلى"),
    ("Dataset '", "مجموعة البيانات '"), ("' created!", "' أُنشئت!"), ("Reading", "جارٍ قراءة"),
    ("Could not read the file:", "تعذّرت قراءة الملف:"), ("Creating table", "جارٍ إنشاء الجدول"), ("from schema...", "من المخطط..."),
    ("Failed to import:", "فشل الاستيراد:"), ("Looking up", "جارٍ البحث عن"), ("on Hugging Face…", "في Hugging Face…"),
    ("Failed to load dataset:", "فشل تحميل مجموعة البيانات:"), ("created successfully!", "أُنشئ بنجاح!"),
    ("Could not query table", "تعذّر الاستعلام عن الجدول"), ("Download failed:", "فشل التنزيل:"),
    ("row(s) into", "صف(وف) إلى"), ("table(s) from the database.", "جدول(جداول) من قاعدة البيانات."),
    ("not found. Available splits:", "غير موجود. التقسيمات المتاحة:"),
    ("Could not create table", "تعذّر إنشاء الجدول"), ("Row inserted into", "تم إدراج صف في"),
    ("Error inserting row:", "خطأ في إدراج الصف:"), ("Connected to", "متصل بـ"),
    ("Schema saved, but table creation failed:", "تم حفظ المخطط لكن فشل إنشاء الجدول:"),
    ("row(s) could not be inserted.", "صف(وف) تعذّر إدراجها."), ("Invalid requirements data:", "بيانات متطلبات غير صالحة:"),
    ("No tables belong to this project yet. The database holds", "لا توجد جداول لهذا المشروع بعد. تحتوي قاعدة البيانات على"),
    ("table(s) from earlier sessions — tick the box above to view them.", "جدول(جداول) من جلسات سابقة — فعّل الخيار أعلاه لعرضها."),
    ("Show all tables in the database (", "عرض كل الجداول في قاعدة البيانات ("), ("unrelated to this project)", "غير مرتبطة بهذا المشروع)"),
    ("Unnamed Project", "مشروع بلا اسم"), ("Requirements Agent", "وكيل المتطلبات"),
    ("Describe your idea in chat → review requirements → get a design, database, plan and more.", "صِف فكرتك في المحادثة ← راجع المتطلبات ← احصل على التصميم وقاعدة البيانات والخطة وغير ذلك."),
    ]
_REGEX_FRAGMENTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"Deleted (\d+) table\(s\) from earlier sessions\."), r"تم حذف \1 جدول(جداول) من الجلسات السابقة."),
    (re.compile(r"Table (`[^`]+`|'[^']+') is empty\."), r"الجدول \1 فارغ."),
    (re.compile(r"Table (`[^`]+`) exists but is empty\."), r"الجدول \1 موجود لكنه فارغ."),
    (re.compile(r"Table (`[^`]+`) has no column definitions\. Skipping\."), r"الجدول \1 بلا تعريفات أعمدة. تم التخطي."),
    (re.compile(r"Table (`[^`]+`) is defined in schema but doesn't exist in database yet\."), r"الجدول \1 معرّف في المخطط لكنه غير موجود في قاعدة البيانات بعد."),
    (re.compile(r"Table (`[^`]+`) created successfully!"), r"تم إنشاء الجدول \1 بنجاح!"),
    (re.compile(r"Table '([^']+)' was imported from"), r"تم استيراد الجدول '\1' من"),
    (re.compile(r"Checked against: (.*)\. Generate more artifacts to widen the check\."), r"تم الفحص مقابل: \1. أنشئ مخرجات أكثر لتوسيع الفحص."),
    (re.compile(r"Found (\d+) matching table\(s\) in database\. Showing sample records\.\.\."), r"تم العثور على \1 جدول(جداول) مطابقة في قاعدة البيانات. عرض سجلات عيّنة..."),
    (re.compile(r"Found (\d+) matching"), r"تم العثور على \1 "),
    (re.compile(r"Imported (\d+) "), r"تم استيراد \1 "),
    (re.compile(r"Round (\d+):"), r"الجولة \1:"),
    (re.compile(r"(\d+) cols, (\d+) rows"), r"\1 أعمدة، \2 صفوف"),
    (re.compile(r"Split '"), "التقسيم '"),
    (re.compile(r"gaps: none"), "النواقص: لا شيء"),
]
_PRIORITY = {"High": "عالية", "Medium": "متوسطة", "Low": "منخفضة"}
_PATTERNS: List[Tuple[re.Pattern, Callable[[re.Match], str]]] = [
    (re.compile(r"^Delete tables from earlier sessions \((\d+)\)$"), lambda m: f"حذف جداول الجلسات السابقة ({m[1]})"),
    (re.compile(r"^Rows to show(?: \((.+?)\))?:?$"), lambda m: "عدد الصفوف المعروضة" + (f" ({m[1]})" if m[1] else "")),
    (re.compile(r"^Required fixes \((\d+)\)$"), lambda m: f"الإصلاحات المطلوبة ({m[1]})"),
    (re.compile(r"^Recommendations \((\d+)\)$"), lambda m: f"التوصيات ({m[1]})"),
    (re.compile(r"^(.*) — (\d+) issue\(s\)$"), lambda m: f"{tr(m[1])} — {m[2]} مشكلة"),
    (re.compile(r"^Download (.+) as CSV$"), lambda m: f"تنزيل {m[1]} (CSV)"),
    (re.compile(r"^Insert Row into (.+)$"), lambda m: f"إدراج صف في {m[1]}"),
    (re.compile(r"^(.*)Priority: (\w+)$"), lambda m: f"{m[1]}الأولوية: {_PRIORITY.get(m[2], m[2])}"),
]
_PREFIX = re.compile(r"^([^\w(]*)(.*)$", re.S)


def tr(text: Any) -> Any:
    """Arabic text for a widget label, or the input unchanged."""
    if not is_rtl() or not isinstance(text, str):
        return text
    m = _PREFIX.match(text)
    prefix, body = (m[1], m[2]) if m else ("", text)
    if body in _AR:
        return prefix + _AR[body]
    if body in _AR_TEXT:
        return prefix + _AR_TEXT[body]
    for pat, fn in _PATTERNS:
        hit = pat.match(body)
        if hit:
            return prefix + fn(hit)
    return tr_text(text)


_AR_ALL: Dict[str, str] = {}
_TAIL = re.compile(r"^(.*?)([\s*_:]*)$", re.S)


def tr_text(text: Any) -> Any:
    """Arabic for body text (headings, captions, messages): exact entry, else known fragments."""
    if not is_rtl() or not isinstance(text, str) or not text.strip():
        return text
    if not _AR_ALL:
        _AR_ALL.update(_AR)
        _AR_ALL.update(_AR_TEXT)
    if text.strip() in _AR_ALL:
        return text.replace(text.strip(), _AR_ALL[text.strip()])
    lead = re.match(r"^[\s#>*_•\-]*(?:[^\w\s(]\ufe0f?\s*)*", text)
    prefix = lead[0] if lead else ""
    rest = text[len(prefix):]
    core, tail = _TAIL.match(rest).groups()
    if core in _AR_ALL:
        return prefix + _AR_ALL[core] + tail
    if core.strip() in _AR_ALL:
        return prefix + _AR_ALL[core.strip()] + tail
    if "<style" in text or "<script" in text:
        return text
    out = text
    for rx, rep in _REGEX_FRAGMENTS:
        out = rx.sub(rep, out)
    for en, ar in _FRAGMENTS:
        if en in out:
            out = out.replace(en, ar)
    return out


def _text_wrapper(fn: Callable) -> Callable:
    def wrapper(body: Any = None, *args: Any, **kwargs: Any) -> Any:
        return fn(tr_text(body), *args, **kwargs)
    wrapper.__wrapped_i18n__ = True  # type: ignore[attr-defined]
    return wrapper


def _label_wrapper(fn: Callable, options: bool = False, placeholder: str = "") -> Callable:
    def wrapper(label: Any = None, *args: Any, **kwargs: Any) -> Any:
        if placeholder and is_rtl() and "placeholder" not in kwargs:
            kwargs["placeholder"] = placeholder
        if options and "format_func" not in kwargs and len(args) < 2:
            kwargs["format_func"] = lambda o: tr(o) if isinstance(o, str) else o
        if "placeholder" in kwargs:
            kwargs["placeholder"] = tr(kwargs["placeholder"])
        return fn(tr(label), *args, **kwargs)
    wrapper.__wrapped_i18n__ = True  # type: ignore[attr-defined]
    return wrapper


def _tabs_wrapper(fn: Callable) -> Callable:
    def wrapper(tabs: Any, *args: Any, **kwargs: Any) -> Any:
        return fn([tr(t) for t in tabs], *args, **kwargs)
    wrapper.__wrapped_i18n__ = True  # type: ignore[attr-defined]
    return wrapper


_COLS = {
    "ID": "المعرّف", "Title": "العنوان", "Agent": "الوكيل", "Priority": "الأولوية", "Features": "الميزات",
    "Points": "النقاط", "Depends on": "يعتمد على", "Feature": "الميزة", "Screens": "الشاشات", "Tables": "الجداول",
    "Tasks": "المهام", "Criteria": "المعايير", "Coverage": "التغطية", "Gaps": "النواقص", "Given": "بافتراض",
    "When": "عندما", "Then": "فإن", "Table": "الجدول", "Column": "العمود", "Type": "النوع", "PK": "مفتاح أساسي",
    "Nullable": "يقبل الفارغ", "Unique": "فريد", "Default": "الافتراضي", "Notes": "ملاحظات", "From Table": "من الجدول",
    "To Table": "إلى الجدول", "FK": "مفتاح أجنبي", "References": "يشير إلى", "Columns": "الأعمدة", "Rows": "الصفوف",
}
_CELLS = {
    "Priority": {"high": "عالية", "medium": "متوسطة", "low": "منخفضة"},
    "Agent": {"dba": "مدير قاعدة البيانات", "developer": "مطوّر", "designer": "مصمّم"},
}
_GAP_WORDS = {"acceptance": "معايير القبول", "screens": "الشاشات", "tables": "الجداول", "tasks": "المهام"}


_LABELS: Dict[str, Dict[str, str]] = {"tables": {}, "columns": {}}
_TYPE_WORDS = {
    "uuid": "معرّف فريد", "varchar": "نص", "character varying": "نص", "string": "نص", "char": "نص", "text": "نص طويل",
    "int": "عدد صحيح", "integer": "عدد صحيح", "bigint": "عدد صحيح", "smallint": "عدد صحيح", "serial": "عدد صحيح",
    "decimal": "عدد عشري", "numeric": "عدد عشري", "float": "عدد عشري", "double": "عدد عشري", "real": "عدد عشري",
    "boolean": "منطقي", "bool": "منطقي", "timestamp": "تاريخ ووقت", "timestamptz": "تاريخ ووقت", "datetime": "تاريخ ووقت",
    "date": "تاريخ", "time": "وقت", "json": "JSON", "jsonb": "JSON", "enum": "قائمة قيم",
}
_REL_TYPES = {"many-to-one": "متعدد إلى واحد", "one-to-many": "واحد إلى متعدد", "one-to-one": "واحد إلى واحد",
              "many-to-many": "متعدد إلى متعدد"}
_DEFAULTS = {"gen_random_uuid()": "معرّف عشوائي", "uuid_generate_v4()": "معرّف عشوائي", "now()": "الوقت الحالي",
             "current_timestamp": "الوقت الحالي", "true": "نعم", "false": "لا"}


def set_schema_labels(labels: Any) -> None:
    """Arabic display names for table/column identifiers (see utils.schema_labels); {} clears them."""
    _LABELS["tables"] = dict((labels or {}).get("tables") or {})
    _LABELS["columns"] = dict((labels or {}).get("columns") or {})


def label_table(name: Any) -> Any:
    """Arabic display name of a table (identifier unchanged when none is known or in English)."""
    return _LABELS["tables"].get(name, name) if is_rtl() else name


def _type_ar(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    m = re.match(r"^\s*([A-Za-z ]+?)\s*(\(.*\))?\s*$", v)
    word = _TYPE_WORDS.get(m[1].lower()) if m else None
    return f"{word} {m[2]}" if word and m[2] else (word or v)


def _name_ar(kind: str) -> Callable[[Any], Any]:
    def convert(v: Any) -> Any:
        if not isinstance(v, str):
            return v
        return ".".join(_LABELS[kind].get(part, part) for part in v.split(".")) if kind == "columns" else _LABELS[kind].get(v, v)
    return convert


def localize_df(df: Any) -> Any:
    """Display copy of a table with Arabic headers and enum values, columns in right-to-left order."""
    if not is_rtl() or not hasattr(df, "columns") or not hasattr(df, "rename"):
        return df
    out = df.copy()
    for col, fn in (("Table", _name_ar("tables")), ("From Table", _name_ar("tables")), ("To Table", _name_ar("tables")),
                    ("Column", _name_ar("columns")), ("FK", _name_ar("columns")), ("References", _name_ar("columns")),
                    ("Type", None), ("Default", None)):
        if col not in out.columns:
            continue
        if col == "Type":
            # Relationship types and column data types share this header.
            out[col] = out[col].map(lambda v: _REL_TYPES.get(str(v).lower(), _type_ar(v)))
        elif col == "Default":
            out[col] = out[col].map(lambda v: _DEFAULTS.get(str(v).strip().lower(), v))
        elif _LABELS["tables"] or _LABELS["columns"]:
            out[col] = out[col].map(fn)
    for col, mapping in _CELLS.items():
        if col in out.columns:
            out[col] = out[col].map(lambda v, m=mapping: m.get(str(v).strip().lower(), v))
    if "Gaps" in out.columns:
        out["Gaps"] = out["Gaps"].map(
            lambda v: ", ".join(_GAP_WORDS.get(w.strip(), w.strip()) for w in v.split(",")) if isinstance(v, str) else v)
    out = out[list(out.columns)[::-1]]
    return out.rename(columns={c: _COLS[c] for c in out.columns if c in _COLS})


def _dataframe_wrapper(fn: Callable) -> Callable:
    def wrapper(data: Any = None, *args: Any, **kwargs: Any) -> Any:
        return fn(localize_df(data), *args, **kwargs)
    wrapper.__wrapped_i18n__ = True  # type: ignore[attr-defined]
    return wrapper


def _patch_class() -> None:
    """Also patch the methods used on columns/containers (``col.metric(...)``)."""
    from streamlit.delta_generator import DeltaGenerator
    label_names = ("button", "download_button", "form_submit_button", "expander", "text_input", "text_area",
                   "checkbox", "number_input", "metric", "slider")
    text_names = ("markdown", "caption", "header", "subheader", "info", "success", "warning", "error", "write")
    for name in label_names + text_names:
        fn = getattr(DeltaGenerator, name, None)
        if fn is None or getattr(fn, "__wrapped_i18n__", False):
            continue
        conv = tr if name in label_names else tr_text

        def make(orig: Callable, conv: Callable = conv) -> Callable:
            def method(self: Any, first: Any = None, *args: Any, **kwargs: Any) -> Any:
                return orig(self, conv(first), *args, **kwargs)
            method.__wrapped_i18n__ = True  # type: ignore[attr-defined]
            return method
        setattr(DeltaGenerator, name, make(fn))


def install() -> None:
    """Patch the Streamlit widgets used by the UI (idempotent; no effect while the language is English)."""
    for name in ("button", "download_button", "form_submit_button", "expander", "text_input", "text_area",
                 "checkbox", "chat_input", "file_uploader", "number_input", "metric", "slider"):
        fn = getattr(st, name, None)
        if fn is not None and not getattr(fn, "__wrapped_i18n__", False):
            setattr(st, name, _label_wrapper(fn))
    for name, ph in (("radio", ""), ("selectbox", "اختر خياراً"), ("multiselect", "اختر خيارات")):
        fn = getattr(st, name, None)
        if fn is not None and not getattr(fn, "__wrapped_i18n__", False):
            setattr(st, name, _label_wrapper(fn, options=True, placeholder=ph))
    for name in ("markdown", "caption", "header", "subheader", "title", "info", "success", "warning", "error",
                 "spinner", "status", "toast", "write"):
        fn = getattr(st, name, None)
        if fn is not None and not getattr(fn, "__wrapped_i18n__", False):
            setattr(st, name, _text_wrapper(fn))
    if not getattr(st.dataframe, "__wrapped_i18n__", False):
        st.dataframe = _dataframe_wrapper(st.dataframe)
    if not getattr(st.tabs, "__wrapped_i18n__", False):
        st.tabs = _tabs_wrapper(st.tabs)
    _patch_class()

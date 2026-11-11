"""Single source of truth for the project inputs used in the paper's evaluation.

tests/test_integration_real_llm.py currently carries its own copy of this list.
Migrate it to import from here so the two cannot drift apart.
"""
from typing import Any, Dict, List

PROJECTS: List[Dict[str, Any]] = [
    {
        "id": 1,
        "domain": "E-commerce / logistics",
        "complexity": "Medium",
        "input": "A flower delivery app where customers can browse flowers, place orders, and get them delivered to their door.",
        "input_ar": "تطبيق لتوصيل الورود يستطيع فيه العملاء تصفح الورود وتقديم الطلبات واستلامها عند باب منزلهم.",
    },
    {
        "id": 2,
        "domain": "Productivity",
        "complexity": "Low",
        "input": "A simple task manager app where users can create, complete, and delete tasks.",
        "input_ar": "تطبيق بسيط لإدارة المهام يستطيع فيه المستخدمون إنشاء المهام وإكمالها وحذفها.",
    },
    {
        "id": 3,
        "domain": "Healthcare",
        "complexity": "High",
        "input": "A hospital appointment booking system where patients can schedule visits with doctors.",
        "input_ar": "نظام لحجز المواعيد في المستشفى يستطيع فيه المرضى جدولة زيارات مع الأطباء.",
    },
    {
        "id": 4,
        "domain": "Education / social",
        "complexity": "Medium",
        "input": "A social platform where university students can share study notes and form groups.",
        "input_ar": "منصة اجتماعية يستطيع فيها طلاب الجامعة مشاركة ملاحظات الدراسة وتكوين مجموعات.",
    },
    {
        "id": 5,
        "domain": "Retail / operations",
        "complexity": "High",
        "input": "A real-time inventory management tool for small retail stores with barcode scanning.",
        "input_ar": "أداة لإدارة المخزون في الوقت الفعلي لمتاجر التجزئة الصغيرة مع مسح الباركود.",
    },
]

# "input_ar" is the same request written in Arabic (Section: Arabic evaluation).
PROJECT_IDS = [f"project_{p['id']}" for p in PROJECTS]

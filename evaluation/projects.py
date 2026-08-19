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
    },
    {
        "id": 2,
        "domain": "Productivity",
        "complexity": "Low",
        "input": "A simple task manager app where users can create, complete, and delete tasks.",
    },
    {
        "id": 3,
        "domain": "Healthcare",
        "complexity": "High",
        "input": "A hospital appointment booking system where patients can schedule visits with doctors.",
    },
    {
        "id": 4,
        "domain": "Education / social",
        "complexity": "Medium",
        "input": "A social platform where university students can share study notes and form groups.",
    },
    {
        "id": 5,
        "domain": "Retail / operations",
        "complexity": "High",
        "input": "A real-time inventory management tool for small retail stores with barcode scanning.",
    },
]

PROJECT_IDS = [f"project_{p['id']}" for p in PROJECTS]

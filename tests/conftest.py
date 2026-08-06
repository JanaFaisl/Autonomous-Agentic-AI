"""Shared fixtures and configuration for all test modules."""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Make the project root importable without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Stub streamlit before any project module imports it so tests run headless
# ---------------------------------------------------------------------------
st_mock = MagicMock()
st_mock.session_state = {}
st_mock.spinner = MagicMock(return_value=MagicMock(__enter__=lambda s, *a: None, __exit__=lambda s, *a: None))
sys.modules.setdefault("streamlit", st_mock)

# ---------------------------------------------------------------------------
# Shared sample data used by multiple test modules
# ---------------------------------------------------------------------------

SAMPLE_PROJECT_DESCRIPTION = (
    "I want to build a flower delivery app where customers can browse flowers, "
    "place orders, and get them delivered to their door."
)

SAMPLE_REQUIREMENTS = {
    "project_name": "Flower Delivery App",
    "features": [
        {
            "id": "F1",
            "name": "Browse Flowers",
            "description": "Customers can view a catalogue of available flowers.",
            "priority": "High",
            "user_stories": [
                "As a customer, I can browse flowers so that I can choose what to order.",
                "As a customer, I can filter flowers by category so that I find what I need quickly.",
            ],
            "acceptance_criteria": [
                "Given I open the app, When I navigate to the catalogue, Then I see all available flowers.",
                "Given I select a filter, When I apply it, Then only matching flowers are shown.",
            ],
        },
        {
            "id": "F2",
            "name": "Place Order",
            "description": "Customers can add flowers to a cart and check out.",
            "priority": "High",
            "user_stories": [
                "As a customer, I can place an order so that flowers are delivered to me.",
                "As a customer, I can pay online so that checkout is seamless.",
            ],
            "acceptance_criteria": [
                "Given I select flowers, When I click 'Order', Then the order is created.",
                "Given I enter payment details, When I confirm, Then payment is processed.",
            ],
        },
    ],
    "technical_requirements": {
        "platform": ["Web", "Mobile"],
        "technologies": ["React", "FastAPI", "PostgreSQL"],
        "database": "PostgreSQL",
    },
    "user_roles": ["Customer", "Admin", "Delivery Driver"],
    "model_used": "claude-3-5-sonnet-latest",
    "timestamp": "2025-01-01 00:00:00",
}

SAMPLE_DESIGN = {
    "design_overview": "Clean, modern flower delivery UI with green palette.",
    "color_scheme": {
        "primary": "#2E7D32",
        "secondary": "#A5D6A7",
        "accent": "#FF8F00",
        "surface": "#FFFFFF",
        "background": "#F1F8E9",
        "text_primary": "#1B1B1B",
        "error": "#D32F2F",
        "success": "#388E3C",
        "warning": "#F57C00",
    },
    "typography": {"font_family": "Inter", "heading_font": "Playfair Display", "body_font": "Inter"},
    "navigation": {"type": "bottom_tab", "items": ["Home", "Browse", "Orders", "Profile"]},
    "screens": [
        {"name": "Home", "purpose": "Landing screen with featured flowers.", "key_components": [], "user_flow": "User opens app → sees featured items."},
    ],
    "ui_components": [],
}

SAMPLE_DB_SCHEMA = {
    "tables": [
        {
            "name": "users",
            "purpose": "Stores customer accounts.",
            "columns": [
                {"name": "id", "type": "uuid", "pk": True, "nullable": False},
                {"name": "email", "type": "varchar", "pk": False, "nullable": False, "unique": True},
                {"name": "name", "type": "varchar", "pk": False, "nullable": False},
            ],
            "indexes": ["email"],
            "relationships": [],
        },
        {
            "name": "orders",
            "purpose": "Tracks customer orders.",
            "columns": [
                {"name": "id", "type": "uuid", "pk": True, "nullable": False},
                {"name": "user_id", "type": "uuid", "pk": False, "nullable": False},
                {"name": "total", "type": "decimal", "pk": False, "nullable": False},
                {"name": "status", "type": "varchar", "pk": False, "nullable": False},
            ],
            "indexes": ["user_id"],
            "relationships": [{"type": "many-to-one", "to_table": "users", "fk": "user_id", "ref": "id"}],
        },
    ],
    "assumptions": ["UUIDs are generated server-side.", "Soft deletes via status field."],
}

SAMPLE_API_KEY = "test-api-key-12345"


@pytest.fixture
def api_key():
    return SAMPLE_API_KEY


@pytest.fixture
def sample_requirements():
    return dict(SAMPLE_REQUIREMENTS)


@pytest.fixture
def sample_design():
    return dict(SAMPLE_DESIGN)


@pytest.fixture
def sample_db_schema():
    return dict(SAMPLE_DB_SCHEMA)


@pytest.fixture
def sample_project_description():
    return SAMPLE_PROJECT_DESCRIPTION

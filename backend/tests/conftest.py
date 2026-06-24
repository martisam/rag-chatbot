import sys
import os

# Fallback for IDEs and direct pytest invocations outside the project root.
# The pythonpath setting in pyproject.toml handles this for normal runs.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_rag_system():
    """Shared mock RAGSystem for API and integration tests."""
    m = MagicMock()
    m.query.return_value = (
        "Mock answer.",
        [{"label": "Course A - Lesson 1", "link": "http://example.com/lesson/1"}],
    )
    m.get_course_analytics.return_value = {
        "total_courses": 1,
        "course_titles": ["Course A"],
    }
    m.session_manager.create_session.return_value = "mock-session-id"
    return m

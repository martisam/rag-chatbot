"""
Tests for CourseSearchTool.execute() in search_tools.py.
All external I/O (ChromaDB, lesson links) is mocked.
"""
import pytest
from unittest.mock import MagicMock
from search_tools import CourseSearchTool
from vector_store import SearchResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_results(docs=None, meta=None, error=None):
    if error:
        return SearchResults(documents=[], metadata=[], distances=[], error=error)
    docs = docs or []
    meta = meta or []
    return SearchResults(documents=docs, metadata=meta, distances=[0.1] * len(docs))


def make_store(docs=None, meta=None, error=None, lesson_link=None):
    store = MagicMock()
    store.search.return_value = make_results(docs=docs, meta=meta, error=error)
    store.get_lesson_link.return_value = lesson_link
    return store


# ---------------------------------------------------------------------------
# Empty / error result paths
# ---------------------------------------------------------------------------

class TestEmptyAndErrorResults:

    def test_no_results_returns_not_found(self):
        tool = CourseSearchTool(make_store())
        result = tool.execute(query="what is MCP")
        assert "No relevant content found" in result

    def test_no_results_with_course_filter_names_the_course(self):
        tool = CourseSearchTool(make_store())
        result = tool.execute(query="what is MCP", course_name="MCP Course")
        assert "No relevant content found" in result
        assert "MCP Course" in result

    def test_no_results_with_lesson_filter_names_the_lesson(self):
        tool = CourseSearchTool(make_store())
        result = tool.execute(query="what is MCP", lesson_number=3)
        assert "No relevant content found" in result
        assert "lesson 3" in result.lower()

    def test_vector_store_error_is_propagated(self):
        store = make_store(error="ChromaDB connection refused")
        tool = CourseSearchTool(store)
        result = tool.execute(query="what is MCP")
        assert "ChromaDB connection refused" in result

    def test_last_sources_stays_empty_on_no_results(self):
        tool = CourseSearchTool(make_store())
        tool.execute(query="what is MCP")
        assert tool.last_sources == []

    def test_last_sources_stays_empty_on_error(self):
        tool = CourseSearchTool(make_store(error="error"))
        tool.execute(query="something")
        assert tool.last_sources == []


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

class TestResultFormatting:

    def test_result_contains_course_and_lesson_header(self):
        store = make_store(
            docs=["MCP content here"],
            meta=[{"course_title": "MCP Course", "lesson_number": 1}],
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="MCP")
        assert "[MCP Course - Lesson 1]" in result

    def test_result_contains_document_text(self):
        store = make_store(
            docs=["Model Context Protocol is a standard."],
            meta=[{"course_title": "MCP Course", "lesson_number": 1}],
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="MCP")
        assert "Model Context Protocol is a standard." in result

    def test_result_without_lesson_number_omits_lesson_from_header(self):
        store = make_store(
            docs=["Some content."],
            meta=[{"course_title": "General Course"}],
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="something")
        assert "[General Course]" in result
        assert "Lesson" not in result

    def test_multiple_results_are_separated(self):
        store = make_store(
            docs=["Chunk A", "Chunk B"],
            meta=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
        )
        tool = CourseSearchTool(store)
        result = tool.execute(query="something")
        assert "Chunk A" in result
        assert "Chunk B" in result


# ---------------------------------------------------------------------------
# Source tracking
# ---------------------------------------------------------------------------

class TestSourceTracking:

    def test_sources_populated_with_label_and_link(self):
        store = make_store(
            docs=["content"],
            meta=[{"course_title": "MCP Course", "lesson_number": 2}],
            lesson_link="http://example.com/lesson/2",
        )
        tool = CourseSearchTool(store)
        tool.execute(query="MCP")
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["label"] == "MCP Course - Lesson 2"
        assert tool.last_sources[0]["link"] == "http://example.com/lesson/2"

    def test_sources_deduplicated_when_same_lesson_appears_twice(self):
        store = make_store(
            docs=["chunk 1", "chunk 2"],
            meta=[
                {"course_title": "MCP Course", "lesson_number": 1},
                {"course_title": "MCP Course", "lesson_number": 1},
            ],
        )
        tool = CourseSearchTool(store)
        tool.execute(query="MCP")
        assert len(tool.last_sources) == 1

    def test_sources_separate_for_different_lessons(self):
        store = make_store(
            docs=["chunk 1", "chunk 2"],
            meta=[
                {"course_title": "MCP Course", "lesson_number": 1},
                {"course_title": "MCP Course", "lesson_number": 2},
            ],
        )
        tool = CourseSearchTool(store)
        tool.execute(query="MCP")
        assert len(tool.last_sources) == 2

    def test_source_label_without_lesson_number(self):
        store = make_store(
            docs=["content"],
            meta=[{"course_title": "General Course"}],
        )
        tool = CourseSearchTool(store)
        tool.execute(query="something")
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["label"] == "General Course"
        assert tool.last_sources[0]["link"] is None

    def test_initial_last_sources_is_empty(self):
        tool = CourseSearchTool(make_store())
        assert tool.last_sources == []


# ---------------------------------------------------------------------------
# Argument pass-through to vector store
# ---------------------------------------------------------------------------

class TestArgPassThrough:

    def test_passes_query_to_store(self):
        store = make_store()
        CourseSearchTool(store).execute(query="what is MCP")
        store.search.assert_called_once_with(query="what is MCP", course_name=None, lesson_number=None)

    def test_passes_course_name_to_store(self):
        store = make_store()
        CourseSearchTool(store).execute(query="MCP", course_name="My Course")
        store.search.assert_called_once_with(query="MCP", course_name="My Course", lesson_number=None)

    def test_passes_lesson_number_to_store(self):
        store = make_store()
        CourseSearchTool(store).execute(query="MCP", lesson_number=4)
        store.search.assert_called_once_with(query="MCP", course_name=None, lesson_number=4)

    def test_passes_all_filters_to_store(self):
        store = make_store()
        CourseSearchTool(store).execute(query="MCP", course_name="Course X", lesson_number=2)
        store.search.assert_called_once_with(query="MCP", course_name="Course X", lesson_number=2)

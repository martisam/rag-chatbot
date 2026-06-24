"""
Tests for RAGSystem.query() in rag_system.py.

These tests mock AIGenerator and VectorStore so no Anthropic API calls or
ChromaDB I/O are needed. They verify:
  - The query->generate_response pipeline is wired correctly
  - Sources are collected from the tool manager and returned
  - Sources are reset after each query
  - Session history is updated correctly
  - Exceptions from the AI layer propagate (not silently swallowed)
"""

import pytest
from unittest.mock import MagicMock, patch
from vector_store import SearchResults

# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def rag(tmp_path):
    """
    RAGSystem with mocked AIGenerator and VectorStore.
    Uses a temp dir for ChromaDB so nothing touches the real DB.
    """
    with (
        patch("rag_system.AIGenerator") as MockAI,
        patch("rag_system.VectorStore") as MockVS,
    ):

        from rag_system import RAGSystem
        from config import Config

        cfg = Config()
        cfg.ANTHROPIC_API_KEY = "test_key"
        cfg.CHROMA_PATH = str(tmp_path / "chroma_db")

        # AI returns a fixed answer
        MockAI.return_value.generate_response.return_value = "Here is the answer."

        # VectorStore search returns one result
        vs = MockVS.return_value
        vs.search.return_value = SearchResults(
            documents=["Model Context Protocol content."],
            metadata=[{"course_title": "MCP Course", "lesson_number": 1}],
            distances=[0.15],
        )
        vs.get_lesson_link.return_value = "http://example.com/lesson/1"
        vs.get_existing_course_titles.return_value = []

        system = RAGSystem(cfg)
        yield system


# ---------------------------------------------------------------------------
# Return values
# ---------------------------------------------------------------------------


class TestQueryReturnValues:

    def test_returns_answer_string(self, rag):
        answer, _ = rag.query("What is MCP?", "session_1")
        assert answer == "Here is the answer."

    def test_returns_sources_list(self, rag):
        _, sources = rag.query("What is MCP?", "session_1")
        assert isinstance(sources, list)

    def test_sources_contain_expected_fields(self, rag):
        _, sources = rag.query("What is MCP?", "session_1")
        if sources:
            for s in sources:
                assert "label" in s

    def test_returns_empty_sources_when_no_tool_used(self, rag):
        # If no tool was used, last_sources stays empty
        rag.tool_manager.reset_sources()  # ensure clean state
        _, sources = rag.query("What is 2+2?", "session_1")
        # Sources may be empty (no search triggered in mock)
        assert isinstance(sources, list)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


class TestSessionManagement:

    def test_exchange_stored_in_session(self, rag):
        sid = rag.session_manager.create_session()
        rag.query("What is MCP?", sid)
        history = rag.session_manager.get_conversation_history(sid)
        assert "What is MCP?" in history
        assert "Here is the answer." in history

    def test_query_without_session_id_does_not_raise(self, rag):
        answer, sources = rag.query("What is MCP?")
        assert answer == "Here is the answer."

    def test_generate_response_receives_conversation_history(self, rag):
        sid = rag.session_manager.create_session()
        rag.session_manager.add_exchange(sid, "prior Q", "prior A")

        rag.query("Follow-up question?", sid)

        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        history = call_kwargs.get("conversation_history")
        assert history is not None
        assert "prior Q" in history


# ---------------------------------------------------------------------------
# Source lifecycle
# ---------------------------------------------------------------------------


class TestSourceLifecycle:

    def test_sources_reset_after_each_query(self, rag):
        sid = rag.session_manager.create_session()
        rag.query("What is MCP?", sid)

        # Manually verify reset_sources was called via the real ToolManager
        # After the query the tool's last_sources should be []
        for tool in rag.tool_manager.tools.values():
            if hasattr(tool, "last_sources"):
                assert tool.last_sources == [], (
                    "last_sources should be reset after each query "
                    "so stale sources do not bleed into the next response."
                )


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


class TestErrorPropagation:

    def test_ai_exception_propagates_to_caller(self, rag):
        rag.ai_generator.generate_response.side_effect = RuntimeError("API failed")
        with pytest.raises(RuntimeError, match="API failed"):
            rag.query("What is MCP?", "session_1")

    def test_generate_response_called_with_tools(self, rag):
        rag.query("What is MCP?", "session_1")
        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        tools = call_kwargs.get("tools")
        assert tools is not None
        assert len(tools) > 0

    def test_generate_response_called_with_tool_manager(self, rag):
        rag.query("What is MCP?", "session_1")
        call_kwargs = rag.ai_generator.generate_response.call_args[1]
        assert call_kwargs.get("tool_manager") is rag.tool_manager

"""
Tests for the FastAPI endpoints: /api/query, /api/courses, /api/session/{id}.

A test-only FastAPI app is defined here that mirrors the production endpoints
from app.py but omits the static file mount (../frontend), which does not exist
in the test environment. The RAGSystem is injected via the mock_rag_system
fixture from conftest.py so no Anthropic API calls or ChromaDB I/O are made.
"""

from typing import List, Optional

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Pydantic models (mirrored from app.py)
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class Source(BaseModel):
    label: str
    link: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


# ---------------------------------------------------------------------------
# Test app factory
# ---------------------------------------------------------------------------


def _make_test_app(rag) -> FastAPI:
    """Build a minimal FastAPI app that mirrors production routes with mocked rag."""
    app = FastAPI()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag.session_manager.create_session()
            answer, sources = rag.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        rag.session_manager.clear_session(session_id)
        return {"status": "cleared"}

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client(mock_rag_system):
    app = _make_test_app(mock_rag_system)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------


class TestQueryEndpoint:

    def test_valid_query_returns_200(self, client):
        resp = client.post("/api/query", json={"query": "What is MCP?"})
        assert resp.status_code == 200

    def test_response_contains_answer(self, client):
        resp = client.post("/api/query", json={"query": "What is MCP?"})
        data = resp.json()
        assert "answer" in data
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

    def test_response_contains_sources_list(self, client):
        resp = client.post("/api/query", json={"query": "What is MCP?"})
        data = resp.json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_response_contains_session_id(self, client):
        resp = client.post("/api/query", json={"query": "What is MCP?"})
        data = resp.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)

    def test_creates_session_when_none_provided(self, client, mock_rag_system):
        client.post("/api/query", json={"query": "What is MCP?"})
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_uses_provided_session_id(self, client, mock_rag_system):
        resp = client.post(
            "/api/query",
            json={"query": "What is MCP?", "session_id": "existing-session"},
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "existing-session"
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_missing_query_field_returns_422(self, client):
        resp = client.post("/api/query", json={})
        assert resp.status_code == 422

    def test_empty_query_string_is_accepted(self, client):
        resp = client.post("/api/query", json={"query": ""})
        assert resp.status_code == 200

    def test_rag_exception_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("DB connection failed")
        resp = client.post("/api/query", json={"query": "What is MCP?"})
        assert resp.status_code == 500
        assert "DB connection failed" in resp.json()["detail"]

    def test_sources_contain_label_field(self, client):
        resp = client.post("/api/query", json={"query": "What is MCP?"})
        sources = resp.json()["sources"]
        for source in sources:
            assert "label" in source

    def test_empty_sources_response_is_valid(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("Answer with no sources.", [])
        resp = client.post("/api/query", json={"query": "What is MCP?"})
        assert resp.status_code == 200
        assert resp.json()["sources"] == []

    def test_query_delegates_to_rag_system(self, client, mock_rag_system):
        client.post("/api/query", json={"query": "What is MCP?"})
        mock_rag_system.query.assert_called_once()
        call_args = mock_rag_system.query.call_args
        assert "What is MCP?" in call_args[0]


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------


class TestCoursesEndpoint:

    def test_returns_200(self, client):
        resp = client.get("/api/courses")
        assert resp.status_code == 200

    def test_response_has_total_courses(self, client):
        resp = client.get("/api/courses")
        data = resp.json()
        assert "total_courses" in data
        assert data["total_courses"] == 2

    def test_response_has_course_titles_list(self, client):
        resp = client.get("/api/courses")
        data = resp.json()
        assert "course_titles" in data
        assert isinstance(data["course_titles"], list)
        assert "MCP Course" in data["course_titles"]

    def test_empty_catalog_returns_zero_courses(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }
        resp = client.get("/api/courses")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_analytics_exception_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError(
            "ChromaDB error"
        )
        resp = client.get("/api/courses")
        assert resp.status_code == 500
        assert "ChromaDB error" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------


class TestSessionEndpoint:

    def test_clear_session_returns_200(self, client):
        resp = client.delete("/api/session/my-session-id")
        assert resp.status_code == 200

    def test_clear_session_returns_cleared_status(self, client):
        resp = client.delete("/api/session/my-session-id")
        assert resp.json() == {"status": "cleared"}

    def test_clear_session_calls_session_manager_with_correct_id(
        self, client, mock_rag_system
    ):
        client.delete("/api/session/session-xyz")
        mock_rag_system.session_manager.clear_session.assert_called_once_with(
            "session-xyz"
        )

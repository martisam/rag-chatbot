"""
Tests for the FastAPI HTTP layer.

Uses an inline test app that mirrors app.py's routes but skips the
StaticFiles mount and module-level RAGSystem init, both of which require
a real filesystem layout that doesn't exist during test runs.
"""
import pytest
from fastapi import FastAPI, HTTPException
from httpx import AsyncClient, ASGITransport
from pydantic import BaseModel
from typing import List, Optional


# ---------------------------------------------------------------------------
# Inline test app - same contract as app.py, no static files
# ---------------------------------------------------------------------------

def build_test_app(rag_system):
    app = FastAPI()

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

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/session/{session_id}")
    async def clear_session(session_id: str):
        rag_system.session_manager.clear_session(session_id)
        return {"status": "cleared"}

    return app


# ---------------------------------------------------------------------------
# Client fixture - uses mock_rag_system from conftest.py
# ---------------------------------------------------------------------------

@pytest.fixture
async def client(mock_rag_system):
    app = build_test_app(mock_rag_system)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

class TestQueryEndpoint:

    async def test_returns_200_on_valid_query(self, client):
        resp = await client.post("/api/query", json={"query": "What is MCP?"})
        assert resp.status_code == 200

    async def test_response_shape(self, client):
        data = (await client.post("/api/query", json={"query": "What is MCP?"})).json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    async def test_answer_matches_rag_output(self, client):
        data = (await client.post("/api/query", json={"query": "What is MCP?"})).json()
        assert data["answer"] == "Mock answer."

    async def test_sources_contain_label_and_link(self, client):
        data = (await client.post("/api/query", json={"query": "What is MCP?"})).json()
        assert len(data["sources"]) == 1
        assert data["sources"][0]["label"] == "Course A - Lesson 1"
        assert data["sources"][0]["link"] == "http://example.com/lesson/1"

    async def test_auto_creates_session_when_none_provided(self, client):
        data = (await client.post("/api/query", json={"query": "What is MCP?"})).json()
        assert data["session_id"] == "mock-session-id"

    async def test_uses_provided_session_id(self, client, mock_rag_system):
        data = (
            await client.post(
                "/api/query",
                json={"query": "What is MCP?", "session_id": "existing-session"},
            )
        ).json()
        assert data["session_id"] == "existing-session"
        assert mock_rag_system.query.call_args.args[1] == "existing-session"

    async def test_returns_422_on_missing_query_field(self, client):
        resp = await client.post("/api/query", json={})
        assert resp.status_code == 422

    async def test_returns_500_when_rag_raises(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = RuntimeError("AI failure")
        resp = await client.post("/api/query", json={"query": "What is MCP?"})
        assert resp.status_code == 500
        assert "AI failure" in resp.json()["detail"]

    async def test_rag_query_called_with_user_query(self, client, mock_rag_system):
        await client.post("/api/query", json={"query": "What is MCP?"})
        assert mock_rag_system.query.call_args.args[0] == "What is MCP?"


# ---------------------------------------------------------------------------
# GET /api/courses
# ---------------------------------------------------------------------------

class TestCoursesEndpoint:

    async def test_returns_200(self, client):
        resp = await client.get("/api/courses")
        assert resp.status_code == 200

    async def test_response_shape(self, client):
        data = (await client.get("/api/courses")).json()
        assert "total_courses" in data
        assert "course_titles" in data

    async def test_total_courses_matches_analytics(self, client):
        data = (await client.get("/api/courses")).json()
        assert data["total_courses"] == 1

    async def test_course_titles_match_analytics(self, client):
        data = (await client.get("/api/courses")).json()
        assert data["course_titles"] == ["Course A"]

    async def test_returns_500_when_analytics_raises(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = RuntimeError("DB error")
        resp = await client.get("/api/courses")
        assert resp.status_code == 500
        assert "DB error" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE /api/session/{session_id}
# ---------------------------------------------------------------------------

class TestClearSessionEndpoint:

    async def test_returns_200_with_cleared_status(self, client):
        resp = await client.delete("/api/session/test-session-abc")
        assert resp.status_code == 200
        assert resp.json() == {"status": "cleared"}

    async def test_calls_clear_session_with_correct_id(self, client, mock_rag_system):
        await client.delete("/api/session/my-session-id")
        mock_rag_system.session_manager.clear_session.assert_called_once_with("my-session-id")

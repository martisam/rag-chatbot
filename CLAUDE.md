# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

Requires Python 3.13+ and `uv`. On Windows, use Git Bash. Always use `uv` - never `pip` directly.

```bash
# Install dependencies
uv sync

# Start the server (runs from backend/ automatically)
./run.sh

# Or manually
cd backend && uv run uvicorn app:app --reload --port 8000
```

App available at `http://localhost:8000`. Swagger UI at `http://localhost:8000/docs`.

Requires a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your-key-here
```

## Architecture

**Entry point:** `backend/app.py` - FastAPI app. `main.py` in the root is an unused stub.

**Request flow for `POST /api/query`:**
1. `app.py` validates the request and delegates to `RAGSystem.query()`
2. `rag_system.py` builds the prompt, fetches session history, and calls `AIGenerator.generate_response()`
3. `ai_generator.py` makes a first Claude API call with `tool_choice: auto` and the `search_course_content` tool
4. If Claude calls the tool (`stop_reason == "tool_use"`), `_handle_tool_execution()` runs it and makes a second Claude API call to synthesize the result
5. Sources (course + lesson labels) are pulled from `ToolManager` after the search and returned alongside the answer

**Vector store layout (ChromaDB):**
- `course_catalog` collection - one document per course (title + metadata); used for fuzzy course name resolution
- `course_content` collection - chunked lesson text; what Claude actually searches

**Document format** (`docs/*.txt`): structured plaintext with `Course Title:`, `Course Link:`, `Course Instructor:` on the first three lines, then `Lesson N: Title` / `Lesson Link:` markers. `DocumentProcessor` parses this format exactly - deviating from it will silently produce malformed chunks.

**Session state** is in-memory only (`SessionManager`). It resets on server restart. `MAX_HISTORY` in `config.py` is set to 2 exchanges.

## Key config values (`backend/config.py`)

| Setting | Default | Notes |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Outdated - current ID is `claude-sonnet-4-6` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Used by ChromaDB via `sentence-transformers` |
| `CHUNK_SIZE` | 800 chars | Sentence-boundary aware chunking |
| `CHUNK_OVERLAP` | 100 chars | |
| `MAX_RESULTS` | 5 | Max chunks returned per search |
| `CHROMA_PATH` | `./chroma_db` | Relative to `backend/` |

## Adding a new tool

1. Create a class extending `Tool` (ABC) in `search_tools.py` - implement `get_tool_definition()` and `execute()`
2. Register it in `RAGSystem.__init__()` via `self.tool_manager.register_tool(your_tool)`
3. If the tool tracks sources for the UI, add a `last_sources` list attribute - `ToolManager.get_last_sources()` checks for it automatically

## Adding course documents

Drop `.txt`, `.pdf`, or `.docx` files into `docs/`. On startup, `app.py` calls `rag_system.add_course_folder("../docs")` and skips any course title already in the vector store. To force a full rebuild, call `add_course_folder(..., clear_existing=True)`.

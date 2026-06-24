# Code Quality Tooling - Changes

## Summary

Added `black` for automatic Python code formatting and created development scripts for running quality checks across the codebase.

## Changes Made

### `pyproject.toml`
- Added `black>=24.0` to the `[dependency-groups] dev` section
- Added `[tool.black]` configuration block:
  - `line-length = 88` (black default)
  - `target-version = ["py313"]` (matches project's Python requirement)
  - Excludes `.git`, `.venv`, and `chroma_db` directories

### `format.sh` (new file)
- Shell script to apply black formatting to all `backend/` Python files
- Run with: `./format.sh`

### `check.sh` (new file)
- Shell script that runs the full quality gate: black format check + pytest suite
- Exits non-zero on any failure so it can be used in CI
- Run with: `./check.sh`

### `backend/` - 12 files reformatted by black
- `app.py`
- `ai_generator.py`
- `config.py`
- `document_processor.py`
- `models.py`
- `rag_system.py`
- `search_tools.py`
- `session_manager.py`
- `vector_store.py`
- `tests/test_ai_generator.py`
- `tests/test_course_search_tool.py`
- `tests/test_rag_system.py`

Changes applied by black include consistent blank lines between class definitions and methods, normalized trailing whitespace, and standardized string quoting.

## Usage

```bash
# Format all files
./format.sh

# Run full quality check (format check + tests)
./check.sh
```

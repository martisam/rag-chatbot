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

---

# Frontend Changes - Theme Toggle

## Feature: Dark/Light Theme Toggle with Smooth Transitions

### What was added

#### `frontend/style.css`
- Added `[data-theme="light"]` CSS variable block with a clean light palette:
  - Background: `#f8fafc`, Surface: `#ffffff`, Text: `#0f172a`
  - Secondary text: `#64748b`, Borders: `#e2e8f0`, Assistant messages: `#f1f5f9`
- Added a "Smooth Theme Transitions" block applying `transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease` to all major themed elements (`body`, `.sidebar`, `.chat-messages`, `.message-content`, `#chatInput`, etc.)
- Updated `.new-chat-btn` from `display: block; width: 100%` to `flex: 1` so it shares the row with the toggle button
- Added `.theme-toggle-btn` styles: borderless icon button with hover highlight using `--primary-color`
- Added `.new-chat-section { display: flex; align-items: center; gap: 0.5rem; }` to lay out the new chat button and toggle side by side

#### `frontend/index.html`
- Added a `<button class="theme-toggle-btn" id="themeToggleBtn">` inside `.new-chat-section`, next to the existing "New Chat" button
- Button contains two SVG icons: a sun (shown in dark mode) and a moon (shown in light mode), toggled via `display` style

#### `frontend/script.js`
- Added `initTheme()` - reads saved theme from `localStorage` (defaults to `dark`) and applies it on page load
- Added `applyTheme(theme)` - sets `data-theme` attribute on `<html>`, swaps the sun/moon icon visibility, and persists the choice to `localStorage`
- Added `toggleTheme()` - reads current theme attribute and flips it
- Wired `themeToggleBtn` click event in `setupEventListeners()`
- Called `initTheme()` at the top of the `DOMContentLoaded` handler so theme is applied before first render

### How it works
1. On load, the saved theme preference is read from `localStorage` and applied instantly via `data-theme` on `<html>`
2. Clicking the sun/moon button in the sidebar calls `toggleTheme()`, which flips `data-theme` between `dark` and `light`
3. CSS variables cascade from `[data-theme="light"]` override the `:root` dark defaults
4. The `transition` declarations on all major elements produce a smooth 300ms color fade across the entire UI
5. Preference persists across sessions via `localStorage`

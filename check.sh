#!/bin/bash
# Run all code quality checks: formatting verification + tests

set -e

echo "--- Checking formatting (black) ---"
uv run black --check backend/

echo ""
echo "--- Running tests ---"
cd backend && uv run pytest tests/ -v

echo ""
echo "All checks passed."

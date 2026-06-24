#!/bin/bash
# Check code quality without modifying files. Exits non-zero if any check fails.

set -e

PASS=true

echo "Checking formatting (black)..."
if ! uv run black --check backend/ main.py; then
    PASS=false
fi

echo ""
echo "Checking import order (isort)..."
if ! uv run isort --check-only backend/ main.py; then
    PASS=false
fi

echo ""
echo "Running tests..."
if ! uv run pytest; then
    PASS=false
fi

echo ""
if [ "$PASS" = true ]; then
    echo "All checks passed."
else
    echo "One or more checks failed. Run ./format.sh to fix formatting issues."
    exit 1
fi

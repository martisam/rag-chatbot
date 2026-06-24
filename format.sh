#!/bin/bash
# Apply black formatting to all backend Python files

set -e

echo "Formatting Python files with black..."
uv run black backend/

echo "Done. All files formatted."

#!/bin/bash
# Start the factor signal analyzer Streamlit UI
# Usage: bash /dfs/data/tools/analyzer/run.sh [port]

PORT=${1:-8551}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# Use conda base python which has streamlit, plotly, scipy installed
exec /root/miniconda3/bin/python -m streamlit run app.py \
    --server.headless true \
    --server.port $PORT \
    --browser.gatherUsageStats false

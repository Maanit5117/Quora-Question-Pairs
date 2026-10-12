#!/usr/bin/env bash
# Script to run the Streamlit frontend for Quora Question Pairs Duplicate Detector

# Prefer Anaconda python/streamlit if present, otherwise fall back to system streamlit
if [ -f "/opt/anaconda3/bin/streamlit" ]; then
    STREAMLIT_CMD="/opt/anaconda3/bin/streamlit"
elif command -v streamlit &> /dev/null; then
    STREAMLIT_CMD="streamlit"
else
    echo "Streamlit not found. Please run: pip install -r requirements.txt"
    exit 1
fi

echo "Starting Streamlit application..."
"$STREAMLIT_CMD" run app.py

#!/bin/bash
# Advanced Algorithmic Trading Platform Runner Script
#
# This script provides an easy way to run the trading system
# with various configurations and modes.

# Change to the script directory
cd "$(dirname "$0")"

# Check if Python exists
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv

    echo "Installing required packages..."
    source .venv/bin/activate
    pip install --upgrade pip
    pip install pandas numpy scipy matplotlib seaborn streamlit alpaca-py \
        requests apscheduler pytz scikit-learn tqdm psutil \
        plotly tabulate

    echo "Virtual environment setup complete."
else
    source .venv/bin/activate
fi

# Create default directories if they don't exist
mkdir -p data logs

# Functions for different run modes
run_dashboard() {
    echo "Starting trading system with dashboard..."
    python -m main --config config.ini --dashboard
}

run_cli() {
    echo "Starting trading system with CLI..."
    python -m main --config config.ini
}

run_headless() {
    echo "Starting trading system in headless mode..."
    python -m main --config config.ini --headless --auto-start
}

# Display menu
echo "===== Advanced Algorithmic Trading Platform ====="
echo "1. Run with Dashboard (Web UI)"
echo "2. Run with CLI (Command Line)"
echo "3. Run Headless (Background)"
echo "4. Exit"
echo "==============================================="

# Get user choice
read -p "Select an option (1-4): " choice
case $choice in
    1)
        run_dashboard
        ;;
    2)
        run_cli
        ;;
    3)
        run_headless
        ;;
    4)
        echo "Exiting."
        exit 0
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac
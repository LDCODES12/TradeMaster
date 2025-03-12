# Advanced Algorithmic Trading Platform

A professional-grade algorithmic trading system for options trading with focus on:
- Performance & Analytics Dashboard
- Risk Management
- Continuous Operation

Built for Mac M2 compatibility.

## Features

- **Advanced Analytics**: Comprehensive performance metrics, visualization, and reporting
- **Risk Management**: Position sizing using Kelly criterion, exposure limits, and automated circuit breakers
- **Multi-Interface**: Web dashboard, CLI, and headless operation modes
- **Options Strategy**: Event-driven options strategy for earnings and catalyst events
- **Continuous Operation**: Robust scheduling, monitoring, and notification system

## Installation

### Prerequisites

- Python 3.8+ installed
- Mac M2 (or any modern Mac)
- Alpaca Paper Trading account with API keys

### Quick Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trading-system.git
   cd trading-system
   ```

2. Run the setup script:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

   This will:
   - Create a Python virtual environment
   - Install all required packages
   - Present options for running the system

## Configuration

Edit the `config.ini` file to add your API keys and customize trading parameters:

```ini
[alpaca]
api_key = YOUR_ALPACA_API_KEY
api_secret = YOUR_ALPACA_API_SECRET

[trading_system]
trading_hours_start = 09:30
trading_hours_end = 16:00
check_interval_minutes = 10
max_positions = 5
...
```

## Usage

The system can be run in three different modes:

### 1. Web Dashboard Mode

```bash
./run.sh
```

Then select option 1 "Run with Dashboard (Web UI)"

This launches a Streamlit dashboard with:
- Real-time portfolio monitoring
- Performance analytics
- Interactive charts
- Trading controls

### 2. Command Line Mode

```bash
./run.sh
```

Then select option 2 "Run with CLI (Command Line)"

This provides a command-line interface with:
- System commands (start/stop trading)
- Position monitoring
- Performance summaries

### 3. Headless Mode

```bash
./run.sh
```

Then select option 3 "Run Headless (Background)"

This runs the system in the background with:
- Automatic trading (if configured)
- Logging to files
- No interactive UI

## System Architecture

The system consists of several integrated components:

```
trading_system/
├── config/            # Configuration management
├── core/              # Core trading system and strategy
├── data/              # Database and data management
├── risk/              # Risk management system
├── analytics/         # Performance analytics
├── ui/                # User interfaces (web, CLI)
└── utils/             # Utility functions
```

## Extending the System

### Adding New Strategies

Create a new strategy class in the `core` directory and integrate it with the trading system.

### Custom Analytics

Extend the `analytics/engine.py` module to add custom metrics and visualizations.

### Risk Management

Customize risk parameters in `config.ini` or extend the `risk/manager.py` module for advanced risk controls.

## Disclaimer

This system is for educational and research purposes only. It is not financial advice. Trade at your own risk.

## License

MIT License
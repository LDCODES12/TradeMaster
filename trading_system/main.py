#!/usr/bin/env python3
"""
Advanced Algorithmic Trading Platform
------------------------------------
An enterprise-grade algorithmic trading platform with:
- Performance & Analytics Dashboard
- Risk Management System
- Continuous Operation Infrastructure

Built for Mac M2 compatibility
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Create necessary directories
for directory in ['logs', 'data']:
    os.makedirs(directory, exist_ok=True)

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/trading_system.log", 'a'),
        logging.StreamHandler(sys.stdout)
    ]
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Advanced Algorithmic Trading Platform")
    parser.add_argument("--config", default="config.ini", help="Path to configuration file")
    parser.add_argument("--dashboard", action="store_true", help="Run with web dashboard")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument("--auto-start", action="store_true", help="Automatically start trading")
    return parser.parse_args()


def main():
    """Main entry point for the trading system"""
    args = parse_arguments()

    # Import core modules here to avoid circular imports
    from config.settings import ConfigManager
    from core.trading_system import TradingSystem

    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()

    # Initialize the trading system
    trading_system = TradingSystem(config)

    # Run in appropriate mode
    if args.dashboard:
        from ui.dashboard import create_dashboard
        create_dashboard(trading_system)
    elif args.headless:
        # Headless mode - just start the system and let it run
        if args.auto_start:
            trading_system.start_trading()
        # Keep the process running
        try:
            trading_system.scheduler.start()
            trading_system.scheduler._thread.join()
        except (KeyboardInterrupt, SystemExit):
            trading_system.stop_trading("User interrupted")
            if trading_system.scheduler.running:
                trading_system.scheduler.shutdown()
    else:
        # CLI mode
        from ui.cli import TradingSystemCLI
        cli = TradingSystemCLI(trading_system)
        cli.start()


if __name__ == "__main__":
    main()
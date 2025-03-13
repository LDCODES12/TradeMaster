"""
Configuration manager for the trading system.
Handles loading and validating config from INI files.
"""

import configparser
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages loading and accessing configuration settings"""

    def __init__(self, config_path: str):
        """Initialize with the path to a config file"""
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file with sensible defaults"""
        config = configparser.ConfigParser()

        # Check if config file exists
        try:
            config.read(self.config_path)
            logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._create_default_config()

        # Convert to dictionary with defaults
        config_dict = {s: dict(config.items(s)) for s in config.sections()}

        # Add trading system defaults if section exists but missing values
        if 'trading_system' in config_dict:
            trading_defaults = {
                'trading_hours_start': '09:30',
                'trading_hours_end': '16:00',
                'check_interval_minutes': '10',
                'monitor_interval_minutes': '5',
                'max_positions': '5',
                'trading_budget': '5000',
                'min_sharpe': '0.25',
                'enable_circuit_breakers': 'true',
                'auto_start': 'false'
            }

            for key, default in trading_defaults.items():
                if key not in config_dict['trading_system']:
                    config_dict['trading_system'][key] = default

        # Add risk management defaults
        if 'risk_management' in config_dict:
            risk_defaults = {
                'max_position_size': '0.05',
                'max_strategy_allocation': '0.20',
                'max_sector_allocation': '0.30',
                'max_daily_drawdown': '0.03',
                'max_total_drawdown': '0.10',
                'circuit_breakers_enabled': 'true'
            }

            for key, default in risk_defaults.items():
                if key not in config_dict['risk_management']:
                    config_dict['risk_management'][key] = default

        # Add database defaults
        if 'database' in config_dict:
            db_defaults = {
                'db_type': 'sqlite',
                'sqlite_path': 'data/trading_system.db',
                'postgres_host': 'localhost',
                'postgres_port': '5432',
                'postgres_user': 'postgres',
                'postgres_password': '',
                'postgres_dbname': 'trading_system'
            }

            for key, default in db_defaults.items():
                if key not in config_dict['database']:
                    config_dict['database'][key] = default
        else:
            # Create database section with defaults
            config_dict['database'] = {
                'db_type': 'sqlite',
                'sqlite_path': 'data/trading_system.db',
                'postgres_host': 'localhost',
                'postgres_port': '5432',
                'postgres_user': 'postgres',
                'postgres_password': '',
                'postgres_dbname': 'trading_system'
            }

        return config_dict

    def _create_default_config(self) -> Dict[str, Any]:
        """Create a default configuration"""
        logger.warning("Creating default configuration")

        default_config = {
            'alpaca': {
                'api_key': '',
                'api_secret': ''
            },
            'finnhub': {
                'api_key': ''
            },
            'alphavantage': {
                'api_key': ''
            },
            'polygon': {
                'api_key': ''
            },
            'trading_system': {
                'trading_hours_start': '09:30',
                'trading_hours_end': '16:00',
                'check_interval_minutes': '10',
                'monitor_interval_minutes': '5',
                'max_positions': '5',
                'trading_budget': '5000',
                'min_sharpe': '0.25',
                'enable_circuit_breakers': 'true',
                'auto_start': 'false'
            },
            'risk_management': {
                'max_position_size': '0.05',
                'max_strategy_allocation': '0.20',
                'max_sector_allocation': '0.30',
                'max_daily_drawdown': '0.03',
                'max_total_drawdown': '0.10',
                'circuit_breakers_enabled': 'true'
            },
            'database': {
                'db_type': 'sqlite',
                'sqlite_path': 'data/trading_system.db',
                'postgres_host': 'localhost',
                'postgres_port': '5432',
                'postgres_user': 'postgres',
                'postgres_password': '',
                'postgres_dbname': 'trading_system'
            },
            'notifications': {
                'enabled': 'true'
            }
        }

        # Write default config to file
        self._write_config(default_config)

        return default_config

    def _write_config(self, config_dict: Dict[str, Any]):
        """Write configuration to file"""
        config = configparser.ConfigParser()

        for section, options in config_dict.items():
            config[section] = options

        try:
            with open(self.config_path, 'w') as f:
                config.write(f)
            logger.info(f"Default configuration written to {self.config_path}")
        except Exception as e:
            logger.error(f"Error writing configuration: {e}")

    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary"""
        return self.config

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get a specific configuration section"""
        if section in self.config:
            return self.config[section]
        logger.warning(f"Configuration section '{section}' not found")
        return {}
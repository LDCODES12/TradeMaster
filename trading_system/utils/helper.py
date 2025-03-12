"""
Utility functions for the trading system.
Provides common helpers used across multiple components.
"""

import logging
import json
import os
from datetime import datetime, date
from typing import Dict, List, Any, Union

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """
    Set up logging configuration

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Set log level
    level = getattr(logging, log_level.upper())

    # Get current date for log filename
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/trading_system_{current_date}.log"),
            logging.StreamHandler()
        ]
    )

    # Set up specialized loggers
    for logger_name in ['Strategy', 'RiskManagement', 'Execution', 'Infrastructure']:
        specialized_logger = logging.getLogger(logger_name)
        handler = logging.FileHandler(f"{log_dir}/{logger_name.lower()}_{current_date}.log")
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        specialized_logger.addHandler(handler)
        specialized_logger.setLevel(level)

    logger.info(f"Logging initialized at level {log_level}")


def json_serial(obj):
    """
    JSON serializer for objects not serializable by default json code
    Used for datetime, date objects in JSON dumps
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def save_to_json(data: Any, filename: str):
    """
    Save data to a JSON file

    Args:
        data: Data to save
        filename: File path to save to
    """
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, default=json_serial, indent=4)
        logger.debug(f"Data saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {e}")
        return False


def load_from_json(filename: str) -> Any:
    """
    Load data from a JSON file

    Args:
        filename: File path to load from

    Returns:
        Loaded data or None if error
    """
    try:
        if not os.path.exists(filename):
            logger.warning(f"File not found: {filename}")
            return None

        with open(filename, 'r') as f:
            data = json.load(f)
        logger.debug(f"Data loaded from {filename}")
        return data
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {e}")
        return None


def calculate_metrics(values: List[float]) -> Dict[str, float]:
    """
    Calculate common metrics for a list of values

    Args:
        values: List of numeric values

    Returns:
        Dictionary of metrics
    """
    if not values:
        return {
            'count': 0,
            'sum': 0,
            'mean': 0,
            'median': 0,
            'min': 0,
            'max': 0,
            'range': 0
        }

    # Sort values for percentile calculation
    sorted_values = sorted(values)
    count = len(sorted_values)

    # Calculate metrics
    metrics = {
        'count': count,
        'sum': sum(sorted_values),
        'mean': sum(sorted_values) / count,
        'median': sorted_values[count // 2] if count % 2 == 1 else (sorted_values[count // 2 - 1] + sorted_values[
            count // 2]) / 2,
        'min': sorted_values[0],
        'max': sorted_values[-1],
        'range': sorted_values[-1] - sorted_values[0]
    }

    return metrics


def format_currency(value: float, include_cents: bool = True) -> str:
    """
    Format a value as currency

    Args:
        value: Numeric value to format
        include_cents: Whether to include cents

    Returns:
        Formatted currency string
    """
    if include_cents:
        return f"${value:,.2f}"
    else:
        return f"${int(value):,}"


def format_percent(value: float, decimal_places: int = 2) -> str:
    """
    Format a value as percentage

    Args:
        value: Numeric value to format (0.1 = 10%)
        decimal_places: Number of decimal places

    Returns:
        Formatted percentage string
    """
    format_str = f"{{:.{decimal_places}f}}%"
    return format_str.format(value * 100)


def get_market_hours():
    """
    Get market open and close times

    Returns:
        Dictionary with market hours
    """
    # Standard market hours (Eastern Time)
    # In a real system, this would check for holidays, early closures, etc.
    return {
        'market_open': '09:30',
        'market_close': '16:00',
        'pre_market_open': '04:00',
        'after_hours_close': '20:00'
    }


def clean_option_symbol(symbol: str) -> Dict[str, Any]:
    """
    Parse an option symbol into its components

    Args:
        symbol: Option symbol (e.g., "AAPL_2023-07-21_150.0_C")

    Returns:
        Dictionary with option components
    """
    try:
        parts = symbol.split('_')
        if len(parts) == 4:
            underlying, expiration, strike, option_type = parts
            return {
                'underlying': underlying,
                'expiration': expiration,
                'strike': float(strike),
                'option_type': 'call' if option_type == 'C' else 'put',
                'is_call': option_type == 'C',
                'is_put': option_type == 'P'
            }
        else:
            # Handle other formats or return partial information
            return {'symbol': symbol}
    except Exception:
        return {'symbol': symbol}


def calculate_days_between(date1: Union[str, datetime], date2: Union[str, datetime] = None) -> int:
    """
    Calculate the number of days between two dates

    Args:
        date1: First date (string or datetime)
        date2: Second date (string or datetime), defaults to today

    Returns:
        Number of days between dates
    """
    # Convert string dates to datetime if needed
    if isinstance(date1, str):
        date1 = datetime.fromisoformat(date1.replace('Z', '+00:00'))

    if date2 is None:
        date2 = datetime.now()
    elif isinstance(date2, str):
        date2 = datetime.fromisoformat(date2.replace('Z', '+00:00'))

    # Calculate difference in days
    delta = date2 - date1
    return abs(delta.days)


def get_system_info() -> Dict[str, Any]:
    """
    Get system information

    Returns:
        Dictionary with system information
    """
    import platform
    import psutil

    # Get CPU info
    cpu_percent = psutil.cpu_percent()
    cpu_count = psutil.cpu_count()

    # Get memory info
    memory = psutil.virtual_memory()
    memory_used_gb = memory.used / (1024 ** 3)
    memory_total_gb = memory.total / (1024 ** 3)
    memory_percent = memory.percent

    # Get disk info
    disk = psutil.disk_usage('/')
    disk_used_gb = disk.used / (1024 ** 3)
    disk_total_gb = disk.total / (1024 ** 3)
    disk_percent = disk.percent

    # Get system info
    system_info = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'platform_version': platform.version(),
        'architecture': platform.machine(),
        'hostname': platform.node(),
        'processor': platform.processor(),
        'cpu_count': cpu_count,
        'cpu_percent': cpu_percent,
        'memory_used_gb': round(memory_used_gb, 2),
        'memory_total_gb': round(memory_total_gb, 2),
        'memory_percent': memory_percent,
        'disk_used_gb': round(disk_used_gb, 2),
        'disk_total_gb': round(disk_total_gb, 2),
        'disk_percent': disk_percent
    }

    return system_info
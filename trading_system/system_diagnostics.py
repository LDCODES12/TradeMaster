#!/usr/bin/env python3
"""
Trading System Free Tier Diagnostic Tool

This script tests the trading system with the constraints of free tier API usage,
verifying caching, rate limiting, and fallbacks work correctly.

Usage:
    python3 system_diagnostics.py [options]

Options:
    --config FILE      Path to configuration file (default: config.ini)
    --trade           Execute test trades (use with caution)
    --full            Run full diagnostics including intensive tests
    --skip-streaming  Skip WebSocket streaming tests
    --output FILE     Output file for results (default: diagnostics_TIMESTAMP.json)
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from tabulate import tabulate
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import required modules
try:
    # Core trading system components
    from config.settings import ConfigManager
    from data.database import DatabaseManager
    from risk.manager import RiskManager
    from analytics.engine import AnalyticsEngine
    from ui.notifications import NotificationSystem
    from utils.helper import setup_logging

    # Import options provider to test caching and rate limiting
    from utils.options_provider import MultiProviderOptionsData

    # Import market data stream manager
    from utils.market_data_stream import MarketDataStreamManager

    # Import for testing sentiment if available
    try:
        from utils.sentiment_analyzer import NewsSentimentAnalyzer

        SENTIMENT_AVAILABLE = True
    except ImportError:
        SENTIMENT_AVAILABLE = False

    # Alpaca API components
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
    from alpaca.data.timeframe import TimeFrame

    logger.info("✅ Successfully imported required modules")
except ImportError as e:
    logger.error(f"❌ Failed to import required modules: {e}")
    logger.error("Make sure you're running from the trading system directory")
    sys.exit(1)


class FreeTierDiagnostics:
    """Trading system diagnostic tool optimized for free tier API usage"""

    def __init__(self, config_path: str):
        """Initialize the diagnostic tool with configuration"""
        self.config_path = config_path
        self.results = {
            "system_info": {"status": "not_tested", "details": {}},
            "config": {"status": "not_tested", "details": {}},
            "database": {"status": "not_tested", "details": {}},
            "alpaca_api": {"status": "not_tested", "details": {}},
            "polygon_api": {"status": "not_tested", "details": {}},
            "options_provider": {"status": "not_tested", "details": {}},
            "market_data": {"status": "not_tested", "details": {}},
            "risk_manager": {"status": "not_tested", "details": {}},
            "sentiment": {"status": "not_tested", "details": {}},
            "stream_manager": {"status": "not_tested", "details": {}},
            "trading": {"status": "not_tested", "details": {}}
        }

        # Common test symbols
        self.common_symbols = ["SPY", "AAPL", "MSFT", "AMZN", "GOOGL"]
        self.option_symbols = ["SPY", "AAPL"]

        # Initialize components
        self.config = None
        self.config_manager = None
        self.db_manager = None
        self.trading_client = None
        self.data_client = None
        self.risk_manager = None
        self.options_provider = None
        self.analytics = None
        self.notifications = None
        self.sentiment_analyzer = None
        self.market_data_stream = None

        # Attempt to initialize the system
        try:
            self._initialize_system()
        except Exception as e:
            logger.error(f"❌ Failed to initialize the system: {e}")
            self.results["system_info"] = {"status": "failed", "error": str(e)}

    def _initialize_system(self):
        """Initialize trading system components"""
        # Load configuration
        try:
            logger.info(f"Loading configuration from {self.config_path}")
            self.config_manager = ConfigManager(self.config_path)
            self.config = self.config_manager.get_config()

            # Extract basic info
            self.results["system_info"] = {
                "status": "initializing",
                "details": {
                    "config_path": self.config_path,
                    "sections": list(self.config.keys()),
                    "timestamp": datetime.now().isoformat()
                }
            }

            # Check config for API keys
            api_keys_status = self._check_api_keys()
            self.results["config"] = api_keys_status

            # Create logging directory if needed
            os.makedirs("logs", exist_ok=True)
            setup_logging()

            logger.info("✅ Configuration loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            raise RuntimeError(f"Configuration error: {e}")

        # Initialize API clients
        try:
            alpaca_config = self.config.get('alpaca', {})
            self.api_key = alpaca_config.get('api_key', '')
            self.api_secret = alpaca_config.get('api_secret', '')

            if not self.api_key or not self.api_secret:
                raise ValueError("Alpaca API credentials missing in configuration")

            self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
            self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)

            logger.info("✅ Alpaca API clients initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Alpaca API clients: {e}")
            raise

        # Initialize database
        try:
            self.db_manager = DatabaseManager(self.config.get('database', {}))
            logger.info("✅ Database manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            self.db_manager = None

        # Initialize other components if possible
        try:
            if self.db_manager:
                self.risk_manager = RiskManager(self.db_manager, self.config.get('risk_management', {}))
                self.analytics = AnalyticsEngine(self.db_manager)
                logger.info("✅ Risk manager and analytics initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize risk/analytics: {e}")

        # Initialize notifications (not critical)
        try:
            self.notifications = NotificationSystem(self.config.get('notifications', {}))
            logger.info("✅ Notification system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize notifications: {e}")

        # Try to initialize options provider
        try:
            # Create API config object for options provider
            api_config = {
                'alpaca': {
                    'api_key': self.api_key,
                    'api_secret': self.api_secret
                },
                'polygon': {'api_key': self.config.get('polygon', {}).get('api_key', '')},
                'provider_preference': ['polygon', 'alpaca']  # Match your production config
            }

            self.options_provider = MultiProviderOptionsData(api_config)
            logger.info("✅ Options provider initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize options provider: {e}")

        # Try to initialize sentiment analyzer if possible
        try:
            if SENTIMENT_AVAILABLE and 'polygon' in self.config:
                polygon_key = self.config.get('polygon', {}).get('api_key', '')
                if polygon_key:
                    self.sentiment_analyzer = NewsSentimentAnalyzer(
                        polygon_key,
                        calls_per_minute=5  # Free tier limit
                    )
                    logger.info("✅ Sentiment analyzer initialized with free tier rate limiting")
        except Exception as e:
            logger.error(f"❌ Failed to initialize sentiment analyzer: {e}")

        # Try to initialize market data stream manager
        try:
            self.market_data_stream = MarketDataStreamManager(
                self.api_key,
                self.api_secret,
                disable_ssl_verification=False  # Will test both ways
            )
            logger.info("✅ Market data stream manager initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize market data stream manager: {e}")

        # Update system info status
        self.results["system_info"]["status"] = "initialized"

    def _check_api_keys(self):
        """Check configuration for necessary API keys"""
        api_keys = {}
        config_status = "success"

        # Check for alpaca keys
        alpaca_config = self.config.get('alpaca', {})
        alpaca_key = alpaca_config.get('api_key', '')
        alpaca_secret = alpaca_config.get('api_secret', '')
        api_keys['alpaca'] = {
            'present': bool(alpaca_key and alpaca_secret),
            'is_paper': True  # Assuming this for now
        }

        # Check for polygon key
        polygon_key = self.config.get('polygon', {}).get('api_key', '')
        api_keys['polygon'] = {
            'present': bool(polygon_key)
        }

        # Check for finnhub key (optional)
        finnhub_key = self.config.get('finnhub', {}).get('api_key', '')
        api_keys['finnhub'] = {
            'present': bool(finnhub_key),
            'optional': True
        }

        # Check for alphavantage key (optional)
        alpha_key = self.config.get('alphavantage', {}).get('api_key', '')
        api_keys['alphavantage'] = {
            'present': bool(alpha_key),
            'optional': True
        }

        # Determine overall status
        required_keys_present = all(
            info['present'] for key, info in api_keys.items()
            if not info.get('optional', False)
        )

        if not required_keys_present:
            config_status = "failed"
            logger.error("❌ Required API keys missing in config")

        return {
            "status": config_status,
            "details": {
                "api_keys": api_keys,
                "required_keys_present": required_keys_present
            }
        }

    def test_system_info(self):
        """Test system information and environment"""
        logger.info("Testing system information and environment...")

        try:
            import platform
            import psutil

            # System info
            system_info = {
                "os": platform.system(),
                "python_version": platform.python_version(),
                "timestamp": datetime.now().isoformat(),
                "hostname": platform.node(),
                "cpu_count": psutil.cpu_count(),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024 ** 3), 2),
                "disk_free_gb": round(psutil.disk_usage('/').free / (1024 ** 3), 2),
            }

            # Check DB version if available
            if self.db_manager:
                try:
                    db_info = self.db_manager.execute_query("SELECT version();")
                    if db_info:
                        system_info["db_version"] = db_info[0].get('version', 'Unknown')
                except Exception as e:
                    system_info["db_version"] = f"Error: {e}"

            # Check for critical resource issues
            health_checks = {
                "sufficient_memory": system_info["memory_available_gb"] > 1.0,  # At least 1GB free
                "sufficient_disk": system_info["disk_free_gb"] > 5.0,  # At least 5GB free
                "cpu_count_adequate": system_info["cpu_count"] >= 2,  # At least 2 CPUs
            }

            status = "success"
            if not all(health_checks.values()):
                status = "warning"
                logger.warning("⚠️ System resource check warnings:")
                for check, result in health_checks.items():
                    if not result:
                        logger.warning(f"  - Failed check: {check}")

            # Update results
            self.results["system_info"] = {
                "status": status,
                "details": {
                    "system_info": system_info,
                    "health_checks": health_checks,
                    "config_sections": list(self.config.keys())
                }
            }

            logger.info(f"✅ System info testing completed: {status}")
            return True

        except Exception as e:
            logger.error(f"❌ System info testing failed: {e}")
            self.results["system_info"] = {"status": "failed", "error": str(e)}
            return False

    def test_database(self):
        """Test database connection and operations"""
        if not self.db_manager:
            logger.error("❌ Database manager not initialized, skipping test")
            self.results["database"] = {"status": "skipped", "error": "Database manager not initialized"}
            return False

        logger.info("Testing database connection and operations...")

        db_results = {}

        # Test basic connectivity
        try:
            # Execute simple query
            version_info = self.db_manager.execute_query("SELECT version();")

            db_results['connection'] = {
                'status': 'success',
                'details': f"Connected to PostgreSQL: {version_info[0]['version'] if version_info else 'Unknown version'}"
            }
            logger.info("✅ Database connection successful")

            # Test table access
            tables_test = self._test_db_tables()
            db_results['tables'] = tables_test

            # Test data operations
            operations_test = self._test_db_operations()
            db_results['operations'] = operations_test

            # Test connection pooling (critical for free tier with limited connections)
            pool_test = self._test_db_connection_pool()
            db_results['connection_pool'] = pool_test

            # Calculate overall status
            status = "success"
            if db_results['connection']['status'] != 'success':
                status = "failed"
            elif any(r.get('status') != 'success' for r in db_results.values()):
                status = "partial"

            self.results["database"] = {
                "status": status,
                "details": db_results
            }

            logger.info(f"✅ Database testing completed: {status}")
            return status != "failed"

        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            self.results["database"] = {"status": "failed", "error": str(e)}
            return False

    def _test_db_tables(self):
        """Test database tables existence and structure"""
        required_tables = ['trades', 'portfolio_snapshots', 'risk_metrics', 'system_logs']
        found_tables = {}

        try:
            # Query to list all tables
            tables_query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """
            tables = self.db_manager.execute_query(tables_query)
            table_names = [t['table_name'] for t in tables]

            # Check each required table
            for table in required_tables:
                if table in table_names:
                    # Verify columns in this table
                    columns_query = f"""
                        SELECT column_name, data_type
                        FROM information_schema.columns
                        WHERE table_name = '{table}'
                    """
                    columns = self.db_manager.execute_query(columns_query)

                    found_tables[table] = {
                        'exists': True,
                        'column_count': len(columns),
                        'columns': {c['column_name']: c['data_type'] for c in columns}
                    }
                else:
                    found_tables[table] = {'exists': False}

            # Determine status
            all_exist = all(t['exists'] for t in found_tables.values())
            status = "success" if all_exist else "partial"

            return {
                'status': status,
                'details': {
                    'required_tables': required_tables,
                    'found_tables': found_tables,
                    'all_exist': all_exist
                }
            }

        except Exception as e:
            logger.error(f"❌ Error checking database tables: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _test_db_operations(self):
        """Test basic database operations"""
        try:
            # Create a test log entry
            test_time = datetime.now()
            test_message = f"Diagnostic test message at {test_time.isoformat()}"

            # Insert query
            insert_query = """
                INSERT INTO system_logs (timestamp, level, component, message)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """
            params = (test_time, "INFO", "DIAGNOSTICS", test_message)

            # Execute insert
            result = self.db_manager.execute_query(insert_query, params)

            if result and len(result) > 0 and 'id' in result[0]:
                log_id = result[0]['id']

                # Verify by reading it back
                read_query = "SELECT * FROM system_logs WHERE id = %s"
                read_result = self.db_manager.execute_query(read_query, (log_id,))

                read_success = len(read_result) > 0 and read_result[0]['message'] == test_message

                return {
                    'status': 'success' if read_success else 'partial',
                    'details': {
                        'insert_success': True,
                        'read_success': read_success,
                        'test_id': log_id
                    }
                }
            else:
                return {
                    'status': 'failed',
                    'details': {
                        'insert_success': False,
                        'error': 'Insert did not return expected ID'
                    }
                }

        except Exception as e:
            logger.error(f"❌ Error testing database operations: {e}")
            return {'status': 'failed', 'error': str(e)}

    def _test_db_connection_pool(self):
        """Test database connection pooling (important for free tier)"""
        try:
            # Get multiple connections in parallel to test pool
            results = []

            # Execute 5 parallel queries to test pool
            for i in range(5):
                query = f"SELECT {i} as test_value, pg_backend_pid() as backend_pid"
                result = self.db_manager.execute_query(query)
                results.append(result[0] if result else {})

            # Check if we got results from all queries
            success = len(results) == 5

            # Check if we're getting different backends (connection reuse)
            pids = set(r.get('backend_pid') for r in results if 'backend_pid' in r)

            return {
                'status': 'success' if success else 'failed',
                'details': {
                    'queries_executed': len(results),
                    'unique_backends': len(pids),
                    'connection_reuse': len(pids) < len(results),
                    'backend_pids': list(pids)
                }
            }
        except Exception as e:
            logger.error(f"❌ Error testing connection pool: {e}")
            return {'status': 'failed', 'error': str(e)}

    def test_alpaca_api(self):
        """Test Alpaca API with free tier constraints"""
        if not self.trading_client or not self.data_client:
            logger.error("❌ Alpaca clients not initialized, skipping test")
            self.results["alpaca_api"] = {"status": "skipped", "error": "Alpaca clients not initialized"}
            return False

        logger.info("Testing Alpaca API with free tier constraints...")

        alpaca_results = {}

        # Test account access (basic functionality)
        try:
            account = self.trading_client.get_account()
            alpaca_results['account'] = {
                'status': 'success',
                'details': {
                    'account_number': account.account_number[:4] + "..." if account.account_number else "Unknown",
                    'buying_power': float(account.buying_power),
                    'equity': float(account.equity),
                    'is_pattern_day_trader': account.pattern_day_trader,
                    'is_paper': account.account_number.startswith('PA') if account.account_number else False
                }
            }
            logger.info(f"✅ Alpaca account access successful")
        except Exception as e:
            alpaca_results['account'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Alpaca account access failed: {e}")

        # Test market data with rate limit consideration
        try:
            # We'll test just a few symbols to be mindful of rate limits
            test_symbols = self.common_symbols[:2]  # Just 2 symbols
            quotes_success = 0

            for symbol in test_symbols:
                try:
                    request_params = StockQuotesRequest(symbol_or_symbols=symbol)
                    quotes = self.data_client.get_stock_latest_quote(request_params)

                    if symbol in quotes:
                        quotes_success += 1
                        logger.info(f"✅ Got quote for {symbol}: ${float(quotes[symbol].ask_price)}")
                except Exception as symbol_error:
                    logger.warning(f"⚠️ Failed to get quote for {symbol}: {symbol_error}")

            alpaca_results['market_data'] = {
                'status': 'success' if quotes_success > 0 else 'failed',
                'details': {
                    'symbols_tested': len(test_symbols),
                    'quotes_success': quotes_success
                }
            }

            if quotes_success > 0:
                logger.info(f"✅ Alpaca market data access successful ({quotes_success}/{len(test_symbols)})")
            else:
                logger.error("❌ Alpaca market data access failed completely")

        except Exception as e:
            alpaca_results['market_data'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Alpaca market data test failed: {e}")

        # Test clock endpoint (low impact)
        try:
            clock = self.trading_client.get_clock()
            alpaca_results['clock'] = {
                'status': 'success',
                'details': {
                    'is_open': clock.is_open,
                    'next_open': clock.next_open,
                    'next_close': clock.next_close
                }
            }
            logger.info(f"✅ Alpaca clock: Market is {'open' if clock.is_open else 'closed'}")
        except Exception as e:
            alpaca_results['clock'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Alpaca clock test failed: {e}")

        # Determine overall status
        success_count = sum(1 for r in alpaca_results.values() if r['status'] == 'success')
        test_count = len(alpaca_results)

        status = "success"
        if success_count == 0:
            status = "failed"
        elif success_count < test_count:
            status = "partial"

        self.results["alpaca_api"] = {
            "status": status,
            "details": {
                "tests_run": test_count,
                "tests_successful": success_count,
                "results": alpaca_results
            }
        }

        logger.info(f"Alpaca API testing completed: {status}")
        return status != "failed"

    def test_polygon_api(self):
        """Test Polygon API with free tier constraints"""
        # Get Polygon API key
        polygon_key = self.config.get('polygon', {}).get('api_key', '')

        if not polygon_key:
            logger.warning("⚠️ Polygon API key not found, skipping test")
            self.results["polygon_api"] = {"status": "skipped", "error": "Polygon API key not found"}
            return True  # Not critical if not using Polygon

        logger.info("Testing Polygon API with free tier constraints...")

        polygon_results = {}

        # Test reference data endpoint with rate limiting (free tier = 5 calls/minute)
        try:
            import requests
            import time

            # Ticker reference endpoint (low impact)
            url = f"https://api.polygon.io/v3/reference/tickers/AAPL?apiKey={polygon_key}"

            start_time = time.time()
            response = requests.get(url, timeout=10)
            request_time = time.time() - start_time

            if response.status_code == 200:
                ticker_data = response.json().get('results', {})
                polygon_results['ticker_info'] = {
                    'status': 'success',
                    'details': {
                        'name': ticker_data.get('name', 'Unknown'),
                        'market_cap': ticker_data.get('market_cap', 0),
                        'request_time_seconds': round(request_time, 2)
                    }
                }
                logger.info(f"✅ Polygon ticker info successful: {ticker_data.get('name', 'Unknown')}")
            else:
                polygon_results['ticker_info'] = {
                    'status': 'failed',
                    'error': f"Status code: {response.status_code}, Response: {response.text[:100]}"
                }
                logger.error(f"❌ Polygon ticker info failed: {response.status_code}")
        except Exception as e:
            polygon_results['ticker_info'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Polygon ticker info test failed: {e}")

        # Wait to respect rate limits (free tier = 5 calls per minute)
        time.sleep(13)  # Ensure we don't hit rate limits

        # Test aggregates endpoint (historical data)
        try:
            # Aggs endpoint
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')

            url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/{start_date}/{end_date}?apiKey={polygon_key}"

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                aggs_data = response.json()
                results_count = len(aggs_data.get('results', []))
                polygon_results['aggs'] = {
                    'status': 'success',
                    'details': {
                        'results_count': results_count,
                        'ticker': aggs_data.get('ticker', 'Unknown'),
                        'timespan': '1 day'
                    }
                }
                logger.info(f"✅ Polygon aggs successful: {results_count} days")
            else:
                polygon_results['aggs'] = {
                    'status': 'failed',
                    'error': f"Status code: {response.status_code}, Response: {response.text[:100]}"
                }
                logger.error(f"❌ Polygon aggs failed: {response.status_code}")
        except Exception as e:
            polygon_results['aggs'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Polygon aggs test failed: {e}")

        # Wait to respect rate limits
        time.sleep(13)

        # Test last quote endpoint
        try:
            url = f"https://api.polygon.io/v2/last/nbbo/AAPL?apiKey={polygon_key}"

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                quote_data = response.json().get('results', {})
                polygon_results['last_quote'] = {
                    'status': 'success',
                    'details': {
                        'ticker': quote_data.get('T', 'Unknown'),
                        'bid_price': quote_data.get('p', 0),
                        'ask_price': quote_data.get('P', 0)
                    }
                }
                logger.info(f"✅ Polygon last quote successful")
            else:
                polygon_results['last_quote'] = {
                    'status': 'failed',
                    'error': f"Status code: {response.status_code}, Response: {response.text[:100]}"
                }
                logger.error(f"❌ Polygon last quote failed: {response.status_code}")
        except Exception as e:
            polygon_results['last_quote'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Polygon last quote test failed: {e}")

        # Determine overall status
        success_count = sum(1 for r in polygon_results.values() if r['status'] == 'success')
        test_count = len(polygon_results)

        status = "success"
        if success_count == 0:
            status = "failed"
        elif success_count < test_count:
            status = "partial"

        self.results["polygon_api"] = {
            "status": status,
            "details": {
                "tests_run": test_count,
                "tests_successful": success_count,
                "results": polygon_results
            }
        }

        logger.info(f"Polygon API testing completed: {status}")
        return True  # Not critical for system operation

    def test_options_provider(self):
        """Test options provider with caching and rate limiting"""
        if not self.options_provider:
            logger.error("❌ Options provider not initialized, skipping test")
            self.results["options_provider"] = {"status": "skipped", "error": "Options provider not initialized"}
            return False

        logger.info("Testing options provider with caching and rate limiting...")

        options_results = {}

        # Test 1: Basic functionality with a single symbol
        test_symbol = self.option_symbols[0]  # Just one symbol to respect rate limits

        try:
            # First call - should go to API
            start_time = time.time()
            options = self.options_provider.get_option_chain(test_symbol)
            first_call_time = time.time() - start_time

            options_count = len(options)

            # Basic validation
            options_results['first_call'] = {
                'status': 'success' if options_count > 0 else 'failed',
                'details': {
                    'symbol': test_symbol,
                    'options_count': options_count,
                    'time_taken': round(first_call_time, 2),
                    'sample': options[0] if options_count > 0 else {}
                }
            }

            if options_count > 0:
                logger.info(f"✅ Options provider first call: {options_count} contracts in {first_call_time:.2f}s")
            else:
                logger.error("❌ Options provider first call failed: No contracts returned")

            # Test 2: Caching - Second call should be faster
            start_time = time.time()
            cached_options = self.options_provider.get_option_chain(test_symbol)
            second_call_time = time.time() - start_time

            cache_working = second_call_time < first_call_time

            options_results['caching'] = {
                'status': 'success' if cache_working else 'warning',
                'details': {
                    'first_call_time': round(first_call_time, 2),
                    'second_call_time': round(second_call_time, 2),
                    'speedup_factor': round(first_call_time / second_call_time, 2) if second_call_time > 0 else 0,
                    'cache_working': cache_working
                }
            }

            if cache_working:
                logger.info(f"✅ Options provider caching working: {second_call_time:.2f}s vs {first_call_time:.2f}s")
            else:
                logger.warning(f"⚠️ Options provider caching may not be working optimally")

            # Test 3: Current price retrieval
            try:
                start_time = time.time()
                price = self.options_provider.get_current_price(test_symbol)
                price_time = time.time() - start_time

                options_results['price'] = {
                    'status': 'success' if price > 0 else 'failed',
                    'details': {
                        'price': price,
                        'time_taken': round(price_time, 2)
                    }
                }

                if price > 0:
                    logger.info(f"✅ Options provider price retrieval: ${price} in {price_time:.2f}s")
                else:
                    logger.error("❌ Options provider price retrieval failed: Invalid price")
            except Exception as e:
                options_results['price'] = {'status': 'failed', 'error': str(e)}
                logger.error(f"❌ Options provider price retrieval failed: {e}")

            # Test 4: Provider selection logic (polygon vs alpaca)
            # Verify the provider preference is correctly configured
            try:
                providers = getattr(self.options_provider, 'provider_preference', ['unknown'])

                options_results['providers'] = {
                    'status': 'success',
                    'details': {
                        'provider_preference': providers,
                        'polygon_available': getattr(self.options_provider, 'polygon_available', False),
                        'alpaca_available': getattr(self.options_provider, 'alpaca_available', False)
                    }
                }

                logger.info(f"✅ Options provider preference: {', '.join(providers)}")
            except Exception as e:
                options_results['providers'] = {'status': 'failed', 'error': str(e)}
                logger.error(f"❌ Options provider selection check failed: {e}")

            # Test 5: Check rate limiting attributes
            try:
                polygon_rate_limit = getattr(self.options_provider, 'polygon_rate_limit', None)
                alpaca_rate_limit = getattr(self.options_provider, 'alpaca_rate_limit', None)

                rate_limits_configured = polygon_rate_limit is not None or alpaca_rate_limit is not None

                if rate_limits_configured:
                    # Try to extract the calls per minute setting
                    polygon_calls = getattr(polygon_rate_limit, 'calls_per_minute', 0) if polygon_rate_limit else 0
                    alpaca_calls = getattr(alpaca_rate_limit, 'calls_per_minute', 0) if alpaca_rate_limit else 0

                    options_results['rate_limiting'] = {
                        'status': 'success',
                        'details': {
                            'polygon_calls_per_minute': polygon_calls,
                            'alpaca_calls_per_minute': alpaca_calls,
                            'is_free_tier_compliant': polygon_calls <= 5  # Free tier is 5 calls/minute
                        }
                    }

                    logger.info(
                        f"✅ Options provider rate limiting: Polygon={polygon_calls}/min, Alpaca={alpaca_calls}/min")
                else:
                    options_results['rate_limiting'] = {
                        'status': 'warning',
                        'details': {
                            'rate_limits_configured': False
                        }
                    }
                    logger.warning("⚠️ Options provider rate limiting not found")
            except Exception as e:
                options_results['rate_limiting'] = {'status': 'failed', 'error': str(e)}
                logger.error(f"❌ Options provider rate limiting check failed: {e}")

        except Exception as e:
            options_results['first_call'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Options provider test failed: {e}")

        # Determine overall status
        success_count = sum(1 for r in options_results.values() if r['status'] == 'success')
        warning_count = sum(1 for r in options_results.values() if r['status'] == 'warning')
        test_count = len(options_results)

        status = "success"
        if options_results.get('first_call', {}).get('status') != 'success':
            status = "failed"  # Basic functionality must work
        elif success_count < test_count - warning_count:
            status = "partial"

        self.results["options_provider"] = {
            "status": status,
            "details": {
                "tests_run": test_count,
                "tests_successful": success_count,
                "results": options_results
            }
        }

        logger.info(f"Options provider testing completed: {status}")
        return status != "failed"

    def test_market_data(self):
        """Test market data retrieval with constraints"""
        if not self.data_client:
            logger.error("❌ Data client not initialized, skipping test")
            self.results["market_data"] = {"status": "skipped", "error": "Data client not initialized"}
            return False

        logger.info("Testing market data retrieval with API constraints...")

        market_results = {}

        # Test 1: Get quotes for a limited set of symbols
        test_symbols = self.common_symbols[:2]  # Limit to 2 symbols for free tier friendliness

        try:
            quotes_success = 0
            quotes_data = {}

            for symbol in test_symbols:
                try:
                    request_params = StockQuotesRequest(symbol_or_symbols=symbol)
                    quotes = self.data_client.get_stock_latest_quote(request_params)

                    if symbol in quotes:
                        quote = quotes[symbol]
                        quotes_data[symbol] = {
                            'ask': float(quote.ask_price) if quote.ask_price else None,
                            'bid': float(quote.bid_price) if quote.bid_price else None,
                            'timestamp': quote.timestamp
                        }
                        quotes_success += 1
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get quote for {symbol}: {e}")

                # Slight delay to avoid rate limits
                time.sleep(0.5)

            market_results['quotes'] = {
                'status': 'success' if quotes_success > 0 else 'failed',
                'details': {
                    'symbols_tested': len(test_symbols),
                    'quotes_success': quotes_success,
                    'quotes_data': quotes_data
                }
            }

            if quotes_success > 0:
                logger.info(f"✅ Market data quotes: {quotes_success}/{len(test_symbols)} successful")
            else:
                logger.error("❌ Market data quotes failed completely")
        except Exception as e:
            market_results['quotes'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Market data quotes test failed: {e}")

        # Test 2: Get bars for a limited set of symbols with modest time range
        try:
            bars_success = 0
            bars_data = {}

            # Use a shorter time range for free tier friendliness
            end_date = datetime.now()
            start_date = end_date - timedelta(days=3)  # Just 3 days

            for symbol in test_symbols:
                try:
                    request_params = StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame.Day,  # Daily bars to minimize data
                        start=start_date,
                        end=end_date
                    )

                    bars = self.data_client.get_stock_bars(request_params)

                    if symbol in bars and len(bars[symbol]) > 0:
                        bars_data[symbol] = {
                            'count': len(bars[symbol]),
                            'first': {
                                'close': float(bars[symbol][0].close),
                                'volume': int(bars[symbol][0].volume)
                            } if len(bars[symbol]) > 0 else {}
                        }
                        bars_success += 1
                except Exception as e:
                    logger.warning(f"⚠️ Failed to get bars for {symbol}: {e}")

                # Slight delay to avoid rate limits
                time.sleep(0.5)

            market_results['bars'] = {
                'status': 'success' if bars_success > 0 else 'failed',
                'details': {
                    'symbols_tested': len(test_symbols),
                    'bars_success': bars_success,
                    'timeframe': 'Day',
                    'days_requested': 3,
                    'bars_data': bars_data
                }
            }

            if bars_success > 0:
                logger.info(f"✅ Market data bars: {bars_success}/{len(test_symbols)} successful")
            else:
                logger.error("❌ Market data bars failed completely")
        except Exception as e:
            market_results['bars'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Market data bars test failed: {e}")

        # Test 3: Verify we can get simple stats like VWAP which are used by the strategies
        if bars_success > 0:
            try:
                # Pick the first successful symbol
                test_symbol = next(symbol for symbol, data in bars_data.items() if data.get('count', 0) > 0)

                # Get 5-minute bars for VWAP calculation (needed by strategies)
                request_params = StockBarsRequest(
                    symbol_or_symbols=test_symbol,
                    timeframe=TimeFrame.Minute,
                    start=end_date - timedelta(days=1),  # Just 1 day
                    end=end_date,
                    limit=100  # Limiting data
                )

                bars = self.data_client.get_stock_bars(request_params)

                if test_symbol in bars and len(bars[test_symbol]) > 0:
                    # Convert to DataFrame for VWAP calculation
                    df = pd.DataFrame([{
                        'close': bar.close,
                        'high': bar.high,
                        'low': bar.low,
                        'open': bar.open,
                        'volume': bar.volume,
                        'timestamp': bar.timestamp
                    } for bar in bars[test_symbol]])

                    # Calculate typical price
                    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

                    # Calculate VWAP
                    df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()

                    market_results['vwap'] = {
                        'status': 'success',
                        'details': {
                            'symbol': test_symbol,
                            'data_points': len(df),
                            'latest_vwap': float(df['vwap'].iloc[-1]) if not df.empty else None,
                            'calculation_successful': not df.empty and 'vwap' in df.columns
                        }
                    }

                    logger.info(f"✅ VWAP calculation successful for {test_symbol}")
                else:
                    market_results['vwap'] = {
                        'status': 'failed',
                        'error': 'No minute bars returned'
                    }
                    logger.warning(f"⚠️ No minute bars returned for {test_symbol}, cannot calculate VWAP")
            except Exception as e:
                market_results['vwap'] = {'status': 'failed', 'error': str(e)}
                logger.error(f"❌ VWAP calculation failed: {e}")

        # Determine overall status
        success_count = sum(1 for r in market_results.values() if r['status'] == 'success')
        test_count = len(market_results)

        status = "success"
        if success_count == 0:
            status = "failed"
        elif success_count < test_count:
            status = "partial"

        self.results["market_data"] = {
            "status": status,
            "details": {
                "tests_run": test_count,
                "tests_successful": success_count,
                "results": market_results
            }
        }

        logger.info(f"Market data testing completed: {status}")
        return status != "failed"

    def test_risk_manager(self):
        """Test risk management functionality"""
        if not self.risk_manager:
            logger.error("❌ Risk manager not initialized, skipping test")
            self.results["risk_manager"] = {"status": "skipped", "error": "Risk manager not initialized"}
            return False

        logger.info("Testing risk management functionality...")

        risk_results = {}

        # Test circuit breakers
        try:
            circuit_breakers_clear, reason = self.risk_manager.check_circuit_breakers()

            risk_results['circuit_breakers'] = {
                'status': 'success',
                'details': {
                    'clear': circuit_breakers_clear,
                    'reason': reason
                }
            }
            logger.info(f"✅ Circuit breakers check: {circuit_breakers_clear}, {reason}")
        except Exception as e:
            risk_results['circuit_breakers'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Circuit breakers test failed: {e}")

        # Test position sizing
        try:
            # Create a sample opportunity for testing
            test_opportunity = {
                'option': {
                    'symbol': 'SPY240315C00430000',
                    'price': 5.25,
                    'strike': 430.0
                },
                'profit_probability': 0.60,
                'event': {
                    'event_type': 'earnings',
                    'symbol': 'SPY',
                    'event_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                }
            }

            position_size = self.risk_manager.calculate_position_size(test_opportunity)

            risk_results['position_sizing'] = {
                'status': 'success',
                'details': {
                    'test_opportunity': test_opportunity,
                    'calculated_position_size': position_size
                }
            }
            logger.info(f"✅ Position sizing test: {position_size} contracts")
        except Exception as e:
            risk_results['position_sizing'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Position sizing test failed: {e}")

        # Test trade approval
        try:
            approved, reason = self.risk_manager.check_trade_approval(test_opportunity)

            risk_results['trade_approval'] = {
                'status': 'success',
                'details': {
                    'approved': approved,
                    'reason': reason
                }
            }
            logger.info(f"✅ Trade approval test: {approved}, {reason}")
        except Exception as e:
            risk_results['trade_approval'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Trade approval test failed: {e}")

        # Test risk metrics calculation
        try:
            risk_metrics = self.risk_manager.calculate_risk_metrics()

            risk_results['risk_metrics'] = {
                'status': 'success',
                'details': {
                    'var_95': risk_metrics['var_95'],
                    'sharpe_ratio': risk_metrics['sharpe_ratio'],
                    'win_rate': risk_metrics['win_rate']
                }
            }
            logger.info(
                f"✅ Risk metrics calculation: VaR={risk_metrics['var_95']:.2%}, Sharpe={risk_metrics['sharpe_ratio']:.2f}")
        except Exception as e:
            risk_results['risk_metrics'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Risk metrics calculation failed: {e}")

        # Determine overall status
        components_working = sum(1 for r in risk_results.values() if r['status'] == 'success')
        components_tested = len(risk_results)

        status = "success"
        if components_working == 0:
            status = "failed"
        elif components_working < components_tested:
            status = "partial"

        self.results["risk_manager"] = {
            "status": status,
            "details": {
                "components_tested": components_tested,
                "components_working": components_working,
                "risk_results": risk_results
            }
        }

        logger.info(f"Risk management testing completed: {status}")
        return status != "failed"

    def test_sentiment(self):
        """Test sentiment analysis with free tier rate limiting"""
        if not self.sentiment_analyzer:
            logger.warning("⚠️ Sentiment analyzer not initialized, skipping test")
            self.results["sentiment"] = {"status": "skipped", "error": "Sentiment analyzer not initialized"}
            return True  # Not critical

        logger.info("Testing sentiment analysis with free tier rate limiting...")

        sentiment_results = {}

        # Verify rate limiting is set correctly
        try:
            # Check if rate limiting is configured properly
            calls_per_minute = getattr(self.sentiment_analyzer, 'calls_per_minute', 0)

            sentiment_results['rate_limiting'] = {
                'status': 'success',
                'details': {
                    'calls_per_minute': calls_per_minute,
                    'is_free_tier_compliant': calls_per_minute <= 5  # Free tier is 5 calls/minute
                }
            }

            if calls_per_minute <= 5:
                logger.info(f"✅ Sentiment analyzer rate limiting set correctly: {calls_per_minute}/minute")
            else:
                logger.warning(f"⚠️ Sentiment analyzer rate limiting may exceed free tier: {calls_per_minute}/minute")
        except Exception as e:
            sentiment_results['rate_limiting'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Sentiment rate limiting check failed: {e}")

        # Test caching mechanism
        try:
            # Check if caching is configured
            cache_dir = getattr(self.sentiment_analyzer, 'cache_dir', None)
            memory_cache = getattr(self.sentiment_analyzer, 'memory_cache', None)

            caching_configured = cache_dir is not None and memory_cache is not None

            if caching_configured:
                # Check if cache directory exists
                cache_exists = os.path.exists(cache_dir) if cache_dir else False

                sentiment_results['caching'] = {
                    'status': 'success',
                    'details': {
                        'cache_dir': cache_dir,
                        'cache_dir_exists': cache_exists,
                        'memory_cache_items': len(memory_cache) if memory_cache else 0
                    }
                }

                logger.info(f"✅ Sentiment analyzer caching configured: {cache_dir}")
            else:
                sentiment_results['caching'] = {
                    'status': 'warning',
                    'details': {
                        'caching_configured': False
                    }
                }
                logger.warning("⚠️ Sentiment analyzer caching may not be properly configured")
        except Exception as e:
            sentiment_results['caching'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Sentiment caching check failed: {e}")

        # Test a single sentiment score (minimal API usage)
        try:
            # Test on a single symbol to minimize API calls
            test_symbol = "AAPL"

            start_time = time.time()
            sentiment_score = self.sentiment_analyzer.get_sentiment_score(test_symbol)
            first_call_time = time.time() - start_time

            sentiment_results['sentiment_score'] = {
                'status': 'success',
                'details': {
                    'symbol': test_symbol,
                    'score': sentiment_score,
                    'time_taken': round(first_call_time, 2)
                }
            }

            logger.info(f"✅ Sentiment score for {test_symbol}: {sentiment_score:.2f} in {first_call_time:.2f}s")

            # Test caching with a second call
            start_time = time.time()
            cached_score = self.sentiment_analyzer.get_sentiment_score(test_symbol)
            second_call_time = time.time() - start_time

            cache_working = second_call_time < first_call_time

            sentiment_results['caching_performance'] = {
                'status': 'success' if cache_working else 'warning',
                'details': {
                    'first_call_time': round(first_call_time, 2),
                    'second_call_time': round(second_call_time, 2),
                    'speedup_factor': round(first_call_time / second_call_time, 2) if second_call_time > 0 else 0,
                    'cache_appears_functional': cache_working
                }
            }

            if cache_working:
                logger.info(f"✅ Sentiment caching working: {second_call_time:.2f}s vs {first_call_time:.2f}s")
            else:
                logger.warning(f"⚠️ Sentiment caching may not be working optimally")
        except Exception as e:
            sentiment_results['sentiment_score'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Sentiment score test failed: {e}")

        # Determine overall status
        success_count = sum(1 for r in sentiment_results.values() if r['status'] == 'success')
        warning_count = sum(1 for r in sentiment_results.values() if r['status'] == 'warning')
        test_count = len(sentiment_results)

        status = "success"
        if sentiment_results.get('sentiment_score', {}).get('status') != 'success':
            status = "failed"  # Basic functionality must work
        elif success_count < test_count - warning_count:
            status = "partial"

        self.results["sentiment"] = {
            "status": status,
            "details": {
                "tests_run": test_count,
                "tests_successful": success_count,
                "results": sentiment_results
            }
        }

        logger.info(f"Sentiment analysis testing completed: {status}")
        return True  # Not critical for system operation

    def test_stream_manager(self):
        """Test market data stream manager"""
        if not self.market_data_stream:
            logger.warning("⚠️ Market data stream manager not initialized, skipping test")
            self.results["stream_manager"] = {"status": "skipped", "error": "Stream manager not initialized"}
            return True  # Not critical

        logger.info("Testing market data stream manager...")

        stream_results = {}

        # Test 1: Test initialization and configuration
        try:
            # Check basic attributes
            stream_results['configuration'] = {
                'status': 'success',
                'details': {
                    'disable_ssl_verification': getattr(self.market_data_stream, 'disable_ssl_verification', None),
                    'calls_per_minute': getattr(self.market_data_stream, 'calls_per_minute', None),
                    'is_running': self.market_data_stream.is_running()
                }
            }

            logger.info(f"✅ Stream manager configuration verified")
        except Exception as e:
            stream_results['configuration'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Stream manager configuration check failed: {e}")

        # Test 2: Test basic setup function
        try:
            # Define dummy handlers
            async def dummy_handler(*args, **kwargs):
                pass

            # Set up stream with minimal configuration
            self.market_data_stream.setup(
                trade_handler=dummy_handler,
                quote_handler=dummy_handler,
                symbols=["SPY"]  # Just one symbol
            )

            stream_results['setup'] = {
                'status': 'success',
                'details': {
                    'handlers_registered': True
                }
            }

            logger.info(f"✅ Stream manager setup successful")
        except Exception as e:
            stream_results['setup'] = {'status': 'failed', 'error': str(e)}
            logger.error(f"❌ Stream manager setup failed: {e}")

        # Test 3: Short connection test (don't keep it running)
        # Only if setup was successful
        if stream_results.get('setup', {}).get('status') == 'success':
            try:
                # Try to start the stream
                start_success = self.market_data_stream.start()

                # Don't keep it running for long
                time.sleep(2)

                # Stop after a short time
                self.market_data_stream.stop()

                stream_results['connection'] = {
                    'status': 'success' if start_success else 'failed',
                    'details': {
                        'start_success': start_success
                    }
                }

                if start_success:
                    logger.info(f"✅ Stream manager connection test successful")
                else:
                    logger.warning(f"⚠️ Stream manager connection test failed")
            except Exception as e:
                stream_results['connection'] = {'status': 'failed', 'error': str(e)}
                logger.error(f"❌ Stream manager connection test failed: {e}")

        # Determine overall status
        success_count = sum(1 for r in stream_results.values() if r['status'] == 'success')
        test_count = len(stream_results)

        status = "success"
        if stream_results.get('configuration', {}).get('status') != 'success':
            status = "failed"  # Basic configuration must work
        elif success_count < test_count:
            status = "partial"

        self.results["stream_manager"] = {
            "status": status,
            "details": {
                "tests_run": test_count,
                "tests_successful": success_count,
                "results": stream_results
            }
        }

        logger.info(f"Stream manager testing completed: {status}")
        return True  # Not critical for system operation

    def test_trading(self, execute_order=False):
        """Test trading capabilities (optional)"""
        if not self.trading_client:
            logger.error("❌ Trading client not initialized, skipping test")
            self.results["trading"] = {"status": "skipped", "error": "Trading client not initialized"}
            return False

        if not execute_order:
            logger.info("🔸 Skipping order execution test (dry run)")
            self.results["trading"] = {
                "status": "skipped",
                "details": {"reason": "execute_order=False (dry run)"}
            }
            return True

        logger.info("Testing basic trading capabilities...")

        try:
            # Choose a cheap, liquid stock for testing
            test_symbol = "SPY"
            test_quantity = 0.01  # Fractional share for minimal cost

            # 1. Place market order
            logger.info(f"🔹 Testing market order for {test_quantity} shares of {test_symbol}")

            market_order_data = MarketOrderRequest(
                symbol=test_symbol,
                qty=test_quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )

            try:
                market_order = self.trading_client.submit_order(market_order_data)
                logger.info(f"✅ Market order placed: ID {market_order.id}")

                # 2. Check order status
                order_status = self.trading_client.get_order_by_id(market_order.id)
                logger.info(f"✅ Order status retrieved: {order_status.status}")

                # 3. Cancel order if still open
                if order_status.status in ['new', 'accepted', 'pending_new']:
                    self.trading_client.cancel_order_by_id(market_order.id)
                    logger.info(f"✅ Order {market_order.id} canceled")

                self.results["trading"] = {
                    "status": "success",
                    "details": {
                        "order_id": market_order.id,
                        "order_status": order_status.status,
                        "cancel_success": True if order_status.status in ['new', 'accepted',
                                                                          'pending_new'] else "not_needed"
                    }
                }

                return True

            except Exception as e:
                logger.error(f"❌ Order placement failed: {e}")
                self.results["trading"] = {
                    "status": "failed",
                    "details": {"error": str(e)}
                }
                return False

        except Exception as e:
            logger.error(f"❌ Trading test failed: {e}")
            self.results["trading"] = {"status": "failed", "error": str(e)}
            return False

    def run_free_tier_tests(self, execute_trade=False, skip_streaming=False, full_test=False):
        """Run all tests with free tier API constraints in mind"""
        tests_passed = 0
        tests_run = 0
        critical_tests_passed = 0
        critical_tests_run = 0

        logger.info("=" * 60)
        logger.info("TRADING SYSTEM FREE TIER DIAGNOSTICS")
        logger.info("=" * 60)

        # System information
        tests_run += 1
        if self.test_system_info():
            tests_passed += 1

        # Database (critical)
        tests_run += 1
        critical_tests_run += 1
        if self.test_database():
            tests_passed += 1
            critical_tests_passed += 1

        # Alpaca API (critical)
        tests_run += 1
        critical_tests_run += 1
        if self.test_alpaca_api():
            tests_passed += 1
            critical_tests_passed += 1

        # Polygon API (non-critical, free tier)
        tests_run += 1
        if self.test_polygon_api():
            tests_passed += 1

        # Options provider (critical)
        tests_run += 1
        critical_tests_run += 1
        if self.test_options_provider():
            tests_passed += 1
            critical_tests_passed += 1

        # Market data (critical)
        tests_run += 1
        critical_tests_run += 1
        if self.test_market_data():
            tests_passed += 1
            critical_tests_passed += 1

        # Risk manager
        tests_run += 1
        if self.test_risk_manager():
            tests_passed += 1

        # Trading (if requested)
        if execute_trade:
            tests_run += 1
            critical_tests_run += 1
            if self.test_trading(execute_order=True):
                tests_passed += 1
                critical_tests_passed += 1
        else:
            self.test_trading(execute_order=False)

        # For full testing, test additional components
        if full_test:
            # Sentiment analysis
            tests_run += 1
            if self.test_sentiment():
                tests_passed += 1

            # Stream manager
            if not skip_streaming:
                tests_run += 1
                if self.test_stream_manager():
                    tests_passed += 1

        # Final summary
        logger.info("=" * 60)
        logger.info(f"DIAGNOSTIC SUMMARY: {tests_passed}/{tests_run} tests passed")
        logger.info(f"CRITICAL COMPONENTS: {critical_tests_passed}/{critical_tests_run} tests passed")
        logger.info("-" * 60)

        # Component summaries
        for component, result in self.results.items():
            if result["status"] != "not_tested":
                status = result["status"]
                status_icon = "✅" if status == "success" else "⚠️" if status in ["partial", "skipped"] else "❌"
                logger.info(f"{status_icon} {component}: {status}")

        logger.info("=" * 60)

        # Alert about critical issues
        if critical_tests_passed < critical_tests_run:
            logger.error("❌ CRITICAL: Some critical components failed diagnostics!")
            logger.error("   The trading system may not function correctly.")

        return critical_tests_passed == critical_tests_run

    def save_results(self, filename=None):
        """Save diagnostic results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"free_tier_diagnostics_{timestamp}.json"

        try:
            # Add system status summary
            self.results["summary"] = {
                "timestamp": datetime.now().isoformat(),
                "tests_passed": sum(1 for r in self.results.values()
                                    if r.get("status") == "success" and r.get("status") != "not_tested"),
                "tests_run": sum(1 for r in self.results.values() if r.get("status") != "not_tested"),
                "critical_components_status": {
                    "database": self.results["database"]["status"],
                    "alpaca_api": self.results["alpaca_api"]["status"],
                    "options_provider": self.results["options_provider"]["status"],
                    "market_data": self.results["market_data"]["status"]
                }
            }

            # Remove sensitive info
            safe_results = self._sanitize_results()

            with open(filename, 'w') as f:
                json.dump(safe_results, f, indent=2, default=str)
            logger.info(f"Diagnostic results saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False

    def _sanitize_results(self):
        """Remove sensitive information from results"""
        safe_results = self.results.copy()

        # Sanitize Alpaca API
        if "alpaca_api" in safe_results and "details" in safe_results["alpaca_api"]:
            if "results" in safe_results["alpaca_api"]["details"]:
                results = safe_results["alpaca_api"]["details"]["results"]
                if "account" in results and "details" in results["account"]:
                    account = results["account"]["details"]
                    if "account_number" in account:
                        account["account_number"] = "REDACTED"

        # Sanitize database connection info
        if "database" in safe_results and "details" in safe_results["database"]:
            if "connection" in safe_results["database"]["details"]:
                # Remove any connection strings or credentials
                conn_details = safe_results["database"]["details"]["connection"].get("details", "")
                if isinstance(conn_details, str) and "password" in conn_details.lower():
                    safe_results["database"]["details"]["connection"]["details"] = "REDACTED"

        return safe_results

    def print_summary_table(self):
        """Print a formatted summary table of the diagnostic results"""
        # Prepare data for table
        data = []

        for component, result in sorted(self.results.items()):
            if result["status"] == "not_tested":
                continue

            status = result["status"]
            status_emoji = "✅" if status == "success" else "⚠️" if status in ["partial", "skipped"] else "❌"

            # Get component details
            details = ""
            if component == "alpaca_api" and "details" in result:
                accounts_status = result["details"].get("results", {}).get("account", {}).get("status", "")
                details = f"Account: {accounts_status}"
            elif component == "polygon_api" and "details" in result:
                success_count = result["details"].get("tests_successful", 0)
                test_count = result["details"].get("tests_run", 0)
                details = f"{success_count}/{test_count} tests passed"
            elif component == "options_provider" and "details" in result:
                first_call = result["details"].get("results", {}).get("first_call", {}).get("status", "")
                details = f"Basic functionality: {first_call}"
            elif component == "market_data" and "details" in result:
                quotes = result["details"].get("results", {}).get("quotes", {}).get("status", "")
                bars = result["details"].get("results", {}).get("bars", {}).get("status", "")
                details = f"Quotes: {quotes}, Bars: {bars}"
            elif component == "database" and "details" in result:
                conn = result["details"].get("connection", {}).get("status", "")
                details = f"Connection: {conn}"
            elif status == "skipped":
                if "error" in result:
                    details = result["error"]
                else:
                    details = "Test skipped"
            elif status == "failed":
                if "error" in result:
                    details = result["error"]
                else:
                    details = "Test failed"

            # Format row
            data.append([
                component.replace("_", " ").title(),
                f"{status_emoji} {status.upper()}",
                details
            ])

        # Print table
        print("\n" + "=" * 80)
        print(f" TRADING SYSTEM FREE TIER DIAGNOSTIC SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        print(tabulate(data, headers=["Component", "Status", "Details"], tablefmt="simple"))

        # Overall assessment
        critical_components = ["database", "alpaca_api", "options_provider", "market_data"]
        critical_passed = sum(1 for c in critical_components
                              if self.results.get(c, {}).get("status") == "success")

        print("\n" + "-" * 80)
        if critical_passed == len(critical_components):
            print("✅ OVERALL: All critical components passed - system is ready for operation")
        elif critical_passed >= len(critical_components) - 1:
            print("⚠️ OVERALL: Most critical components passed - system may function with limitations")
        else:
            print("❌ OVERALL: Critical component failures detected - system is NOT ready")

        # Print rate limiting information
        print("-" * 80)
        print("📊 FREE TIER API USAGE INFORMATION:")

        # Polygon rate limit
        if "polygon_api" in self.results and self.results["polygon_api"]["status"] != "skipped":
            print("  • Polygon API: 5 calls per minute limit (free tier)")

        # Options provider rate limits
        rate_limits = self.results.get("options_provider", {}).get("details", {}).get(
            "results", {}).get("rate_limiting", {}).get("details", {})
        if rate_limits:
            polygon_rate = rate_limits.get("polygon_calls_per_minute", "Unknown")
            is_compliant = rate_limits.get("is_free_tier_compliant", False)
            print(f"  • Options Provider: {polygon_rate} calls per minute to Polygon")
            print(f"    {'✅ Compliant with free tier' if is_compliant else '⚠️ May exceed free tier limits'}")

        # Sentiment analyzer rate limits
        sentiment_limits = self.results.get("sentiment", {}).get("details", {}).get(
            "results", {}).get("rate_limiting", {}).get("details", {})
        if sentiment_limits:
            calls_per_min = sentiment_limits.get("calls_per_minute", "Unknown")
            is_compliant = sentiment_limits.get("is_free_tier_compliant", False)
            print(f"  • Sentiment Analyzer: {calls_per_min} calls per minute to Polygon")
            print(f"    {'✅ Compliant with free tier' if is_compliant else '⚠️ May exceed free tier limits'}")

        print("-" * 80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Trading System Free Tier Diagnostic Tool")
    parser.add_argument("--config", default="config.ini", help="Path to configuration file")
    parser.add_argument("--trade", action="store_true", help="Execute test trades (use with caution)")
    parser.add_argument("--full", action="store_true", help="Run full diagnostics including intensive tests")
    parser.add_argument("--skip-streaming", action="store_true", help="Skip WebSocket streaming tests")
    parser.add_argument("--output", help="Output file for results")
    args = parser.parse_args()

    try:
        # Create diagnostic tool
        diagnostics = FreeTierDiagnostics(args.config)

        # Run all tests
        success = diagnostics.run_free_tier_tests(
            execute_trade=args.trade,
            skip_streaming=args.skip_streaming,
            full_test=args.full
        )

        # Print summary table
        diagnostics.print_summary_table()

        # Save results
        diagnostics.save_results(args.output)

        # Exit code based on test results
        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"Diagnostics failed with critical error: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
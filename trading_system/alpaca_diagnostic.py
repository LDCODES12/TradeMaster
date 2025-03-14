#!/usr/bin/env python3
"""
Alpaca API Comprehensive Diagnostic Tool

This script performs in-depth testing of all Alpaca API functionality
required for options trading, providing detailed feedback on any issues.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
import argparse
import pandas as pd
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Alpaca SDK
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, OrderType
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.live import StockDataStream

    logger.info("Successfully imported Alpaca SDK modules")
except ImportError as e:
    logger.error(f"Failed to import Alpaca SDK: {e}")
    logger.error("Please install required packages: pip install alpaca-py")
    exit(1)


class AlpacaAPITester:
    """Comprehensive Alpaca API testing tool"""

    def __init__(self, api_key, api_secret, paper=True):
        """Initialize with API credentials"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper = paper
        self.results = {
            "account": {"status": "not_tested", "details": {}},
            "market_data": {"status": "not_tested", "details": {}},
            "trading": {"status": "not_tested", "details": {}},
            "options": {"status": "not_tested", "details": {}},
            "streaming": {"status": "not_tested", "details": {}}
        }

        # Initialize clients
        self.trading_client = None
        self.data_client = None
        self.stream = None

        # Test symbols
        self.common_symbols = ["SPY", "AAPL", "MSFT", "AMZN", "GOOGL"]
        self.option_symbols = ["SPY", "AAPL"]  # Highly liquid options

    def initialize_clients(self):
        """Initialize API clients and check basic connectivity"""
        try:
            self.trading_client = TradingClient(self.api_key, self.api_secret, paper=self.paper)
            logger.info("✅ TradingClient initialized")

            self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)
            logger.info("✅ StockHistoricalDataClient initialized")

            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize clients: {e}")
            return False

    def test_account_info(self):
        """Test account information access"""
        try:
            account = self.trading_client.get_account()

            # Extract key account data
            account_data = {
                "id": account.id,
                "account_number": account.account_number,
                "status": account.status,
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
                "created_at": account.created_at
            }

            logger.info(f"✅ Account info retrieved: ID {account_data['id']}, " +
                        f"Portfolio Value ${account_data['portfolio_value']:.2f}")

            # Run health checks on account data
            health_checks = {
                "is_active": account.status == "ACTIVE",
                "has_buying_power": float(account.buying_power) > 0,
                "not_blocked": not account.trading_blocked and not account.account_blocked
            }

            # Update results
            self.results["account"] = {
                "status": "success" if all(health_checks.values()) else "warning",
                "details": {
                    "account_data": account_data,
                    "health_checks": health_checks
                }
            }

            if not all(health_checks.values()):
                logger.warning("⚠️ Account has issues that might prevent trading")
                for check, passed in health_checks.items():
                    if not passed:
                        logger.warning(f"  - Failed check: {check}")

            return all(health_checks.values())

        except Exception as e:
            logger.error(f"❌ Account info test failed: {e}")
            self.results["account"] = {"status": "failed", "error": str(e)}
            return False

    def test_market_data(self):
        """Test market data retrieval for quotes and bars"""
        try:
            quote_results = {}
            bar_results = {}

            # 1. Test getting latest quotes
            for symbol in self.common_symbols:
                try:
                    request_params = StockQuotesRequest(
                        symbol_or_symbols=symbol,
                    )
                    quotes = self.data_client.get_stock_latest_quote(request_params)

                    if symbol in quotes:
                        quote = quotes[symbol]
                        quote_data = {
                            "ask_price": float(quote.ask_price) if quote.ask_price else None,
                            "ask_size": int(quote.ask_size) if quote.ask_size else None,
                            "bid_price": float(quote.bid_price) if quote.bid_price else None,
                            "bid_size": int(quote.bid_size) if quote.bid_size else None,
                            "timestamp": quote.timestamp
                        }
                        quote_results[symbol] = {
                            "status": "success",
                            "data": quote_data
                        }
                        logger.info(
                            f"✅ Quote for {symbol}: Ask ${quote_data['ask_price']}, Bid ${quote_data['bid_price']}")
                    else:
                        quote_results[symbol] = {"status": "failed", "error": "Symbol not in response"}
                        logger.warning(f"⚠️ No quote data for {symbol}")
                except Exception as e:
                    quote_results[symbol] = {"status": "failed", "error": str(e)}
                    logger.error(f"❌ Failed to get quote for {symbol}: {e}")

            # 2. Test getting historical bars
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)

            for symbol in self.common_symbols:
                try:
                    request_params = StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date
                    )

                    bars = self.data_client.get_stock_bars(request_params)

                    if symbol in bars and len(bars[symbol]) > 0:
                        bar_data = [{
                            "timestamp": bar.timestamp,
                            "open": float(bar.open),
                            "high": float(bar.high),
                            "low": float(bar.low),
                            "close": float(bar.close),
                            "volume": int(bar.volume)
                        } for bar in bars[symbol]]

                        bar_results[symbol] = {
                            "status": "success",
                            "count": len(bar_data),
                            "sample": bar_data[0] if bar_data else None
                        }
                        logger.info(f"✅ Retrieved {len(bar_data)} bars for {symbol}")
                    else:
                        bar_results[symbol] = {"status": "failed", "error": "No bars returned"}
                        logger.warning(f"⚠️ No bar data for {symbol}")
                except Exception as e:
                    bar_results[symbol] = {"status": "failed", "error": str(e)}
                    logger.error(f"❌ Failed to get bars for {symbol}: {e}")

            # Analyze results
            quotes_success = sum(1 for r in quote_results.values() if r["status"] == "success")
            bars_success = sum(1 for r in bar_results.values() if r["status"] == "success")

            market_data_status = "success"
            if quotes_success == 0 or bars_success == 0:
                market_data_status = "failed"
            elif quotes_success < len(self.common_symbols) or bars_success < len(self.common_symbols):
                market_data_status = "partial"

            self.results["market_data"] = {
                "status": market_data_status,
                "details": {
                    "quotes_tested": len(self.common_symbols),
                    "quotes_success": quotes_success,
                    "bars_tested": len(self.common_symbols),
                    "bars_success": bars_success,
                    "quote_results": quote_results,
                    "bar_results": bar_results
                }
            }

            return market_data_status == "success" or market_data_status == "partial"

        except Exception as e:
            logger.error(f"❌ Market data test failed: {e}")
            self.results["market_data"] = {"status": "failed", "error": str(e)}
            return False

    def test_trading_capabilities(self, execute_order=False):
        """Test order creation and management capabilities"""
        if not execute_order:
            logger.info("🔸 Skipping order execution test (dry run)")
            self.results["trading"] = {
                "status": "skipped",
                "details": {"reason": "execute_order=False (dry run)"}
            }
            return True

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

    def test_options_data(self):
        """Test options data availability and retrieval"""
        try:
            options_results = {}

            # For each test symbol, try to get options data
            for symbol in self.option_symbols:
                symbol_results = {
                    "expirations": {"status": "not_tested"},
                    "strikes": {"status": "not_tested"},
                    "contracts": {"status": "not_tested"},
                    "quotes": {"status": "not_tested"}
                }

                # 1. Try to get options expirations
                try:
                    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/options/expirations"
                    headers = {
                        "Apca-Api-Key-Id": self.api_key,
                        "Apca-Api-Secret-Key": self.api_secret
                    }

                    # Use requests for direct API access
                    import requests
                    response = requests.get(url, headers=headers, timeout=10)

                    if response.status_code == 200:
                        expirations_data = response.json()
                        if "expirations" in expirations_data:
                            expirations = expirations_data["expirations"]
                            symbol_results["expirations"] = {
                                "status": "success",
                                "count": len(expirations),
                                "sample": expirations[:3] if expirations else []
                            }
                            logger.info(f"✅ Retrieved {len(expirations)} expirations for {symbol}")

                            # If we found expirations, try to get strikes for the nearest expiration
                            if expirations:
                                nearest_exp = sorted(expirations)[0]

                                # 2. Get strikes for the nearest expiration
                                strikes_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/options/strikes?expiration={nearest_exp}"
                                strikes_response = requests.get(strikes_url, headers=headers, timeout=10)

                                if strikes_response.status_code == 200:
                                    strikes_data = strikes_response.json()
                                    if "strikes" in strikes_data:
                                        strikes = strikes_data["strikes"]
                                        symbol_results["strikes"] = {
                                            "status": "success",
                                            "count": len(strikes),
                                            "sample": strikes[:5] if strikes else [],
                                            "expiration": nearest_exp
                                        }
                                        logger.info(f"✅ Retrieved {len(strikes)} strikes for {symbol} ({nearest_exp})")

                                        # 3. If we have strikes, try to get options chain
                                        if strikes:
                                            # Find an ATM strike
                                            # First get the current price
                                            quote_request = StockQuotesRequest(symbol_or_symbols=symbol)
                                            quotes = self.data_client.get_stock_latest_quote(quote_request)
                                            if symbol in quotes:
                                                current_price = float(quotes[symbol].ask_price)

                                                # Find closest strike to current price
                                                atm_strike = min(strikes, key=lambda x: abs(float(x) - current_price))

                                                # Get options chain
                                                chain_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/options/chain"
                                                params = {
                                                    "expiration": nearest_exp,
                                                    "strike": atm_strike
                                                }

                                                chain_response = requests.get(chain_url, headers=headers, params=params,
                                                                              timeout=10)

                                                if chain_response.status_code == 200:
                                                    chain_data = chain_response.json()
                                                    if "options" in chain_data:
                                                        options = chain_data["options"]
                                                        symbol_results["contracts"] = {
                                                            "status": "success",
                                                            "count": len(options),
                                                            "sample": options[0] if options else None,
                                                            "expiration": nearest_exp,
                                                            "strike": atm_strike
                                                        }
                                                        logger.info(
                                                            f"✅ Retrieved {len(options)} option contracts for {symbol}")

                                                        # 4. If we have contracts, try to get quotes for one
                                                        if options and len(options) > 0:
                                                            option_symbol = options[0]["symbol"]
                                                            quote_url = f"https://data.alpaca.markets/v2/stocks/quotes/latest?symbols={option_symbol}"
                                                            quote_response = requests.get(quote_url, headers=headers,
                                                                                          timeout=10)

                                                            if quote_response.status_code == 200:
                                                                quote_data = quote_response.json()
                                                                if "quotes" in quote_data and option_symbol in \
                                                                        quote_data["quotes"]:
                                                                    symbol_results["quotes"] = {
                                                                        "status": "success",
                                                                        "data": quote_data["quotes"][option_symbol]
                                                                    }
                                                                    logger.info(
                                                                        f"✅ Retrieved quotes for option {option_symbol}")
                                                                else:
                                                                    symbol_results["quotes"] = {
                                                                        "status": "failed",
                                                                        "error": "No quotes in response"
                                                                    }
                                                                    logger.warning(
                                                                        f"⚠️ No quote data for option {option_symbol}")
                                                            else:
                                                                symbol_results["quotes"] = {
                                                                    "status": "failed",
                                                                    "error": f"Status code: {quote_response.status_code}",
                                                                    "response": quote_response.text[:100]
                                                                }
                                                                logger.warning(
                                                                    f"⚠️ Failed to get quotes for option {option_symbol}: {quote_response.status_code}")
                                                    else:
                                                        symbol_results["contracts"] = {
                                                            "status": "failed",
                                                            "error": "No options in response"
                                                        }
                                                        logger.warning(f"⚠️ No options contracts returned for {symbol}")
                                                else:
                                                    symbol_results["contracts"] = {
                                                        "status": "failed",
                                                        "error": f"Status code: {chain_response.status_code}",
                                                        "response": chain_response.text[:100]
                                                    }
                                                    logger.warning(
                                                        f"⚠️ Failed to get options chain: {chain_response.status_code}")
                                        else:
                                            logger.warning(f"⚠️ No strikes available for {symbol}")
                                    else:
                                        symbol_results["strikes"] = {
                                            "status": "failed",
                                            "error": "No strikes in response"
                                        }
                                        logger.warning(f"⚠️ No strikes data in response for {symbol}")
                                else:
                                    symbol_results["strikes"] = {
                                        "status": "failed",
                                        "error": f"Status code: {strikes_response.status_code}",
                                        "response": strikes_response.text[:100]
                                    }
                                    logger.warning(f"⚠️ Failed to get strikes: {strikes_response.status_code}")
                        else:
                            symbol_results["expirations"] = {
                                "status": "failed",
                                "error": "No expirations in response"
                            }
                            logger.warning(f"⚠️ No expirations data in response for {symbol}")
                    else:
                        symbol_results["expirations"] = {
                            "status": "failed",
                            "error": f"Status code: {response.status_code}",
                            "response": response.text[:100]
                        }
                        logger.warning(f"⚠️ Failed to get expirations: {response.status_code}")

                except Exception as e:
                    symbol_results["expirations"] = {"status": "failed", "error": str(e)}
                    logger.error(f"❌ Failed to test options for {symbol}: {e}")

                options_results[symbol] = symbol_results

            # Analyze results
            options_available = False
            partial_success = False

            for symbol, results in options_results.items():
                if results["expirations"]["status"] == "success":
                    partial_success = True
                    if results["contracts"]["status"] == "success":
                        options_available = True

            if options_available:
                options_status = "success"
            elif partial_success:
                options_status = "partial"
            else:
                options_status = "failed"

            self.results["options"] = {
                "status": options_status,
                "details": options_results
            }

            # Final assessment
            if options_status == "failed":
                logger.error("❌ Options data unavailable - critical functionality affected")
            elif options_status == "partial":
                logger.warning("⚠️ Options data partially available - some functionality may be limited")
            else:
                logger.info("✅ Options data available - full functionality supported")

            return options_status != "failed"

        except Exception as e:
            logger.error(f"❌ Options data test failed: {e}")
            self.results["options"] = {"status": "failed", "error": str(e)}
            return False

    def test_streaming(self, timeout=10):
        """Test WebSocket streaming data"""
        try:
            events_received = {symbol: False for symbol in self.common_symbols}
            connection_success = False

            # Initialize counter for tracking received messages
            msg_count = 0

            async def _msg_handler(msg):
                nonlocal msg_count
                nonlocal events_received

                msg_count += 1

                # Check if we got data for our test symbols
                if 'S' in msg:  # Symbol field
                    symbol = msg['S']
                    if symbol in events_received:
                        events_received[symbol] = True

            logger.info("🔹 Testing WebSocket connection...")

            # Create stream instance
            self.stream = StockDataStream(self.api_key, self.api_secret)

            # Add message handler
            self.stream.add_raw_data_handler(_msg_handler)

            # Start the connection
            try:
                import asyncio

                # Create event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                # Define async test function
                async def run_test():
                    nonlocal connection_success

                    try:
                        # Start the websocket connection
                        await self.stream.connect()
                        connection_success = True
                        logger.info("✅ WebSocket connected")

                        # Subscribe to quotes for our test symbols
                        await self.stream.subscribe_quotes(_msg_handler, *self.common_symbols)
                        logger.info(f"✅ Subscribed to quotes for {len(self.common_symbols)} symbols")

                        # Wait for data or timeout
                        start_time = time.time()
                        while time.time() - start_time < timeout and not all(events_received.values()):
                            await asyncio.sleep(0.1)

                        # Clean up
                        await self.stream.disconnect()

                    except Exception as e:
                        logger.error(f"❌ WebSocket error: {e}")
                        raise

                # Run the test
                loop.run_until_complete(run_test())

            except Exception as e:
                logger.error(f"❌ WebSocket test failed: {e}")

            # Analyze results
            symbols_with_data = sum(1 for received in events_received.values() if received)

            if not connection_success:
                streaming_status = "failed"
            elif symbols_with_data == 0:
                streaming_status = "connected_no_data"
            elif symbols_with_data < len(self.common_symbols):
                streaming_status = "partial"
            else:
                streaming_status = "success"

            self.results["streaming"] = {
                "status": streaming_status,
                "details": {
                    "connection_success": connection_success,
                    "messages_received": msg_count,
                    "symbols_with_data": symbols_with_data,
                    "symbols_tested": len(self.common_symbols),
                    "per_symbol": events_received
                }
            }

            if streaming_status == "success":
                logger.info("✅ WebSocket streaming fully functional")
            elif streaming_status == "partial":
                logger.warning("⚠️ WebSocket streaming partially working")
            elif streaming_status == "connected_no_data":
                logger.warning("⚠️ WebSocket connected but no data received")
            else:
                logger.error("❌ WebSocket streaming failed")

            return streaming_status != "failed"

        except Exception as e:
            logger.error(f"❌ Streaming test failed: {e}")
            self.results["streaming"] = {"status": "failed", "error": str(e)}
            return False

    def run_all_tests(self, execute_order=False, skip_streaming=False):
        """Run all diagnostic tests"""
        tests_passed = 0
        tests_run = 0

        logger.info("=" * 60)
        logger.info("ALPACA API DIAGNOSTICS")
        logger.info(f"Paper Trading: {'Yes' if self.paper else 'No'}")
        logger.info("=" * 60)

        # Step 1: Initialize clients
        if not self.initialize_clients():
            logger.error("❌ Critical failure: Unable to initialize API clients")
            return False

        # Step 2: Test account information
        tests_run += 1
        if self.test_account_info():
            tests_passed += 1

        # Step 3: Test market data
        tests_run += 1
        if self.test_market_data():
            tests_passed += 1

        # Step 4: Test trading capabilities (optional)
        if execute_order:
            tests_run += 1
            if self.test_trading_capabilities(execute_order=True):
                tests_passed += 1
        else:
            self.test_trading_capabilities(execute_order=False)

        # Step 5: Test options data
        tests_run += 1
        if self.test_options_data():
            tests_passed += 1

        # Step 6: Test streaming (optional)
        if not skip_streaming:
            tests_run += 1
            if self.test_streaming():
                tests_passed += 1

        # Final summary
        logger.info("=" * 60)
        logger.info(f"DIAGNOSTIC SUMMARY: {tests_passed}/{tests_run} tests passed")
        logger.info("-" * 60)

        # Account summary
        account_status = self.results["account"]["status"]
        status_icon = "✅" if account_status == "success" else "⚠️" if account_status == "warning" else "❌"
        logger.info(f"{status_icon} Account: {account_status}")

        # Market data summary
        market_status = self.results["market_data"]["status"]
        status_icon = "✅" if market_status == "success" else "⚠️" if market_status == "partial" else "❌"
        if market_status != "not_tested":
            details = self.results["market_data"]["details"]
            logger.info(
                f"{status_icon} Market Data: {market_status} ({details.get('quotes_success', 0)}/{details.get('quotes_tested', 0)} quotes, {details.get('bars_success', 0)}/{details.get('bars_tested', 0)} bars)")
        else:
            logger.info(f"{status_icon} Market Data: {market_status}")

        # Trading summary
        trade_status = self.results["trading"]["status"]
        status_icon = "✅" if trade_status == "success" else "⚠️" if trade_status == "skipped" else "❌"
        logger.info(f"{status_icon} Trading: {trade_status}")

        # Options summary
        options_status = self.results["options"]["status"]
        status_icon = "✅" if options_status == "success" else "⚠️" if options_status == "partial" else "❌"
        logger.info(f"{status_icon} Options Data: {options_status}")

        # Streaming summary
        streaming_status = self.results["streaming"]["status"]
        if streaming_status != "not_tested":
            status_icon = "✅" if streaming_status == "success" else "⚠️" if streaming_status in ["partial",
                                                                                                 "connected_no_data"] else "❌"
            details = self.results["streaming"]["details"]
            logger.info(
                f"{status_icon} WebSocket: {streaming_status} ({details.get('symbols_with_data', 0)}/{details.get('symbols_tested', 0)} symbols)")
        else:
            logger.info(f"- Streaming: {streaming_status}")

        logger.info("=" * 60)

        # Critical functionality check for options trading
        if options_status == "failed":
            logger.error("❌ CRITICAL: Options data is unavailable - Options trading will not function!")
            logger.error("   Check if your account has options trading permissions and API access.")
        elif options_status == "partial":
            logger.warning("⚠️ CAUTION: Options data is partially available - Some trading function may be limited")
            logger.warning("   Options metadata is accessible but options quotes may be unavailable.")
        elif options_status == "success":
            logger.info("✅ OK: Options data is fully available - Trading system should function properly")

        return tests_passed == tests_run

    def save_results(self, filename=None):
        """Save diagnostic results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alpaca_diagnostics_{timestamp}.json"

        try:
            # Remove sensitive info
            safe_results = self.results.copy()
            if "account" in safe_results and "details" in safe_results["account"] and "account_data" in \
                    safe_results["account"]["details"]:
                # Remove sensitive account details but keep structure
                account_data = safe_results["account"]["details"]["account_data"]
                for field in ["id", "account_number"]:
                    if field in account_data:
                        account_data[field] = f"{account_data[field][:4]}...REDACTED"

            with open(filename, 'w') as f:
                json.dump(safe_results, f, indent=2, default=str)
            logger.info(f"Diagnostic results saved to {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return False


def load_config():
    """Load Alpaca API credentials from config.ini"""
    import configparser

    config = configparser.ConfigParser()
    config_file = "config.ini"

    if not os.path.exists(config_file):
        logger.error(f"Configuration file {config_file} not found")
        return None, None

    try:
        config.read(config_file)

        if 'alpaca' not in config:
            logger.error("Alpaca section missing from config.ini")
            return None, None

        alpaca_cfg = config['alpaca']

        api_key = alpaca_cfg.get('api_key', '')
        api_secret = alpaca_cfg.get('api_secret', '')

        if not api_key or not api_secret:
            logger.error("Alpaca API credentials missing from config.ini")
            return None, None

        return api_key, api_secret

    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None, None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Alpaca API Diagnostic Tool")
    parser.add_argument("--key", help="Alpaca API key")
    parser.add_argument("--secret", help="Alpaca API secret")
    parser.add_argument("--live", action="store_true", help="Use live trading (default: paper)")
    parser.add_argument("--execute", action="store_true", help="Execute test orders (use with caution)")
    parser.add_argument("--skip-streaming", action="store_true", help="Skip WebSocket streaming tests")
    parser.add_argument("--output", help="Output file for results")
    args = parser.parse_args()

    # Get API credentials
    api_key = args.key
    api_secret = args.secret

    # If not provided as arguments, try to load from config
    if not api_key or not api_secret:
        api_key, api_secret = load_config()

    if not api_key or not api_secret:
        logger.error("API credentials not provided. Use --key and --secret or configure config.ini")
        return

    # Create and run tester
    tester = AlpacaAPITester(api_key, api_secret, paper=not args.live)
    success = tester.run_all_tests(execute_order=args.execute, skip_streaming=args.skip_streaming)

    # Save results
    tester.save_results(args.output)

    # Exit code based on test results
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
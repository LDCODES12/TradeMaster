"""
Core options trading strategy implementation with real API integrations.
Analyzes market data and finds trading opportunities.
"""
import os
import json
import logging
import time
import random
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Alpaca APIs
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.requests import LimitOrderRequest


# Initialize logger
logger = logging.getLogger(__name__)


class PrecisionOptionsArbitrage:
    """
    Core strategy for finding and executing options arbitrage opportunities
    using real market data and events
    """

    def __init__(self, api_config: Dict[str, Dict[str, str]]):
        """Initialize the strategy with API connections"""
        # Store the config
        self.config = api_config

        # Get Alpaca credentials
        alpaca = api_config.get('alpaca', {})
        self.api_key = alpaca.get('api_key', '')
        self.api_secret = alpaca.get('api_secret', '')

        # Initialize API clients
        if not self.api_key or not self.api_secret:
            logger.error("API credentials not provided. Strategy cannot run in live mode.")
            raise ValueError("Missing API credentials")

        # Initialize Alpaca clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)

        # Get API keys for other services
        self.finnhub_key = api_config.get('finnhub', {}).get('api_key', '')
        self.alpha_key = api_config.get('alphavantage', {}).get('api_key', '')
        self.polygon_key = api_config.get('polygon', {}).get('api_key', '')

        # Strategy state
        self.positions = []
        self.buying_power = 0.0
        self.account = None
        self.price_cache = {}  # Cache for frequently accessed prices

        # Strategy adaptation parameters
        self.position_size_factor = 1.0
        self.min_edge_threshold = 0.2
        self.min_probability = 0.55
        self.profit_target_factor = 1.0
        self.stop_loss_factor = 1.0
        self.call_bias_factor = 1.0
        self.put_bias_factor = 1.0
        self.duration_preference = 'balanced'
        self.recovery_mode = False
        self.trading_paused = False

        # Load account info
        self._load_account()

        # Test API connections and log results
        api_status = self.test_api_connections()

        # Check if any required APIs failed
        if api_status['alpaca']['status'] != 'success':
            logger.error("CRITICAL: Alpaca API connection failed. Trading cannot proceed.")
            raise RuntimeError("Failed to connect to Alpaca API - check credentials")

        # Warn about missing or failed secondary APIs
        missing_apis = [api for api, result in api_status.items()
                        if api != 'alpaca' and result['status'] != 'success']

        if missing_apis:
            logger.warning(f"Warning: The following APIs have issues: {', '.join(missing_apis)}")
            logger.warning("Real market data may be limited - check API keys and connectivity")

        # Load events calendar
        self.events_calendar = self._load_events_calendar()

        self.production_mode = True  # Set to True for stricter validation
        self.skip_synthetic = True  # Set to True to skip synthetic data completely

        logger.info("Precision Options Arbitrage strategy initialized with live market data")

    def _load_account(self):
        """Load account information from Alpaca"""
        try:
            self.account = self.trading_client.get_account()
            self.buying_power = float(self.account.buying_power)
            logger.info(
                f"Account loaded: ${self.account.portfolio_value} portfolio value, ${self.buying_power} buying power")
        except Exception as e:
            logger.error(f"Error loading account: {e}")
            raise RuntimeError("Failed to load account information") from e

    def test_api_connections(self):
        """Test all API connections and report status"""
        results = {}

        # Test Alpaca connection
        try:
            account = self.trading_client.get_account()
            results['alpaca'] = {
                'status': 'success',
                'details': f"Connected to Alpaca {'paper' if account.account_number.startswith('PA') else 'live'} account",
                'account_id': account.account_number
            }
        except Exception as e:
            results['alpaca'] = {'status': 'failed', 'error': str(e)}

        # Test Finnhub connection
        if self.finnhub_key:
            try:
                url = f"https://finnhub.io/api/v1/stock/symbol?exchange=US&token={self.finnhub_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    results['finnhub'] = {
                        'status': 'success',
                        'details': f"Connected to Finnhub API, got {len(response.json())} symbols"
                    }
                else:
                    results['finnhub'] = {
                        'status': 'failed',
                        'error': f"Status code: {response.status_code}, Response: {response.text[:100]}"
                    }
            except Exception as e:
                results['finnhub'] = {'status': 'failed', 'error': str(e)}
        else:
            results['finnhub'] = {'status': 'missing', 'error': 'No API key provided'}

        # Test Alpha Vantage connection
        if self.alpha_key:
            try:
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=SPY&apikey={self.alpha_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200 and "Global Quote" in response.text:
                    results['alphavantage'] = {'status': 'success', 'details': "Connected to Alpha Vantage API"}
                else:
                    results['alphavantage'] = {
                        'status': 'failed',
                        'error': f"Status code: {response.status_code}, Response: {response.text[:100]}"
                    }
            except Exception as e:
                results['alphavantage'] = {'status': 'failed', 'error': str(e)}
        else:
            results['alphavantage'] = {'status': 'missing', 'error': 'No API key provided'}

        # Test Polygon connection
        if self.polygon_key:
            try:
                url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09?apiKey={self.polygon_key}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    results['polygon'] = {'status': 'success', 'details': "Connected to Polygon API"}
                else:
                    results['polygon'] = {
                        'status': 'failed',
                        'error': f"Status code: {response.status_code}, Response: {response.text[:100]}"
                    }
            except Exception as e:
                results['polygon'] = {'status': 'failed', 'error': str(e)}
        else:
            results['polygon'] = {'status': 'missing', 'error': 'No API key provided'}

        # Print diagnostic information
        logger.info("API Connection Test Results:")
        for api, result in results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            logger.info(f"{status_icon} {api.upper()}: {result['status']}")
            if result['status'] == 'success':
                logger.info(f"  Details: {result['details']}")
            else:
                logger.error(f"  Error: {result['error']}")

        return results

    def _load_events_calendar(self) -> List[Dict[str, Any]]:
        """Load real market events calendar from multiple sources with fallbacks"""
        events = []

        # Try to load from cache first to reduce API calls
        cached_events = self._load_cached_events()
        if cached_events:
            logger.info(f"Loaded {len(cached_events)} events from cache")
            return cached_events

        # 1. Try Finnhub for earnings calendar (primary source)
        if self.finnhub_key:
            try:
                today = datetime.now().strftime('%Y-%m-%d')
                future = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')

                url = f"https://finnhub.io/api/v1/calendar/earnings?from={today}&to={future}&token={self.finnhub_key}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    for earning in data.get('earningsCalendar', []):
                        # Determine importance based on company metrics
                        importance = 'medium'
                        if float(earning.get('epsEstimate', 0) or 0) > 0:
                            importance = 'high'

                        events.append({
                            'symbol': earning['symbol'],
                            'event_type': 'earnings',
                            'event_date': earning['date'],
                            'importance': importance,
                            'data': {
                                'eps_estimate': earning.get('epsEstimate'),
                                'quarter': earning.get('quarter'),
                                'time': earning.get('hour', 'bmo')  # bmo = before market open
                            }
                        })
                    logger.info(f"Loaded {len(events)} earnings events from Finnhub")
            except Exception as e:
                logger.warning(f"Error fetching earnings from Finnhub: {e}")

        # 2. Try Alpha Vantage as backup for earnings
        if not events and self.alpha_key:
            try:
                url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={self.alpha_key}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    # Alpha Vantage returns CSV for this endpoint
                    lines = response.text.strip().split('\n')
                    if len(lines) > 1:
                        headers = lines[0].split(',')
                        alpha_events = []

                        for line in lines[1:]:
                            fields = line.split(',')
                            if len(fields) >= len(headers):
                                event_data = dict(zip(headers, fields))
                                alpha_events.append({
                                    'symbol': event_data.get('symbol', ''),
                                    'event_type': 'earnings',
                                    'event_date': event_data.get('reportDate', ''),
                                    'importance': 'high',
                                    'data': {
                                        'eps_estimate': event_data.get('estimate', ''),
                                        'time': event_data.get('time', 'bmo')
                                    }
                                })

                        events.extend(alpha_events)
                        logger.info(f"Loaded {len(alpha_events)} earnings events from Alpha Vantage")
            except Exception as e:
                logger.warning(f"Error fetching earnings from Alpha Vantage: {e}")

        # 3. Add dividend events from Polygon if available
        if self.polygon_key:
            try:
                today = datetime.now().strftime('%Y-%m-%d')
                url = f"https://api.polygon.io/v2/reference/dividends?limit=50&apiKey={self.polygon_key}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    dividend_events = []

                    for dividend in data.get('results', []):
                        # Filter to only future ex-dividend dates
                        ex_date = dividend.get('ex_dividend_date')
                        if ex_date and ex_date >= today:
                            dividend_events.append({
                                'symbol': dividend.get('ticker', ''),
                                'event_type': 'dividend',
                                'event_date': ex_date,
                                'importance': 'medium',
                                'data': {
                                    'amount': dividend.get('amount'),
                                    'payment_date': dividend.get('payment_date')
                                }
                            })

                    events.extend(dividend_events)
                    logger.info(f"Loaded {len(dividend_events)} dividend events from Polygon")
            except Exception as e:
                logger.warning(f"Error fetching dividends from Polygon: {e}")

        # 4. Add known major economic events
        econ_events = self._get_economic_events()
        if econ_events:
            events.extend(econ_events)
            logger.info(f"Added {len(econ_events)} economic events")

        # 5. Fallback to default watchlist if no events found
        if not events:
            logger.warning("No events found from any source - using fallback watchlist")
            events = self._create_watchlist_events(self._get_default_watchlist())

        # Filter events to keep only relevant future dates
        filtered_events = self._filter_events(events)
        logger.info(f"Filtered to {len(filtered_events)} relevant events")

        # Cache events to avoid repeated API calls
        self._cache_events(filtered_events)

        return filtered_events

    def _load_cached_events(self) -> List[Dict[str, Any]]:
        """Load events from cache if available and not expired"""
        try:
            cache_file = os.path.join("data", "events_cache.json")
            if os.path.exists(cache_file):
                # Check if cache is still valid (not older than 12 hours)
                file_modified = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_modified < timedelta(hours=12):
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading cached events: {e}")
            return []

    def _cache_events(self, events: List[Dict[str, Any]]):
        """Cache events to file for future use"""
        try:
            cache_file = os.path.join("data", "events_cache.json")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(events, f)
        except Exception as e:
            logger.error(f"Error caching events: {e}")

    def _filter_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter events to keep only relevant dates and prioritize by importance"""
        today = datetime.now().date()

        filtered_events = []
        for event in events:
            try:
                event_date = datetime.strptime(event['event_date'], '%Y-%m-%d').date()
                # Keep events in the next 30 days (not past events)
                if today <= event_date <= (today + timedelta(days=30)):
                    filtered_events.append(event)
            except (ValueError, TypeError):
                # Skip events with invalid dates
                continue

        # Sort first by importance, then by date
        return sorted(
            filtered_events,
            key=lambda e: (
                0 if e.get('importance') == 'high' else
                1 if e.get('importance') == 'medium' else 2,
                e['event_date']
            )
        )

    def _get_economic_events(self) -> List[Dict[str, Any]]:
        """Get major economic events that might impact the market"""
        # In a production system, these would come from an economic calendar API
        today = datetime.now().strftime('%Y-%m-%d')
        # Calculate next FOMC date (approximate)
        next_month = (datetime.now() + timedelta(days=30)).replace(day=1)
        fomc_date = next_month.replace(day=15).strftime('%Y-%m-%d')  # Approximate

        return [
            {
                'symbol': 'SPY',  # Use market ETFs for economic events
                'event_type': 'economic',
                'event_date': fomc_date,
                'importance': 'high',
                'data': {'name': 'FOMC Meeting', 'description': 'Federal Reserve interest rate decision'}
            }
        ]

    def _get_default_watchlist(self) -> List[str]:
        """Get default watchlist of major stocks"""
        return [
            'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AMD',  # Tech
            'JPM', 'BAC', 'GS', 'MS',  # Financials
            'JNJ', 'PFE', 'MRK',  # Healthcare
            'COST', 'WMT', 'HD',  # Retail
            'XOM', 'CVX'  # Energy
        ]

    def _create_watchlist_events(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """Create events from a watchlist when no API events are available"""
        events = []
        today = datetime.now()

        for i, symbol in enumerate(symbols):
            # Spread events over next 30 days
            days_offset = (i % 4) * 7 + 1  # 1, 8, 15, 22 days out
            event_date = (today + timedelta(days=days_offset)).strftime('%Y-%m-%d')

            events.append({
                'symbol': symbol,
                'event_type': 'watchlist',
                'event_date': event_date,
                'importance': 'medium'
            })

        logger.info(f"Created {len(events)} watchlist events as fallback")
        return events

    def _get_option_contracts(self, symbol: str) -> List[Dict[str, Any]]:
        """Get real options chain data with fallback mechanisms and data validation"""
        options = []

        # Try to use cache first to reduce API calls
        cached_options = self._get_cached_options(symbol)
        if cached_options:
            logger.info(f"Using cached options data for {symbol}")
            return cached_options

        # Get current price for strike selection
        try:
            current_price = self._get_current_price(symbol)
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            return []

        # 1. PRIMARY SOURCE: Alpaca Options API
        if self.api_key and self.api_secret:
            try:
                options = self._get_options_from_alpaca(symbol, current_price)
                if options:
                    logger.info(f"Successfully fetched {len(options)} option contracts from Alpaca for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get options from Alpaca: {e}")

        # 2. FALLBACK: Polygon Options API
        if not options and self.polygon_key:
            try:
                options = self._get_options_from_polygon(symbol, current_price)
                if options:
                    logger.info(f"Successfully fetched {len(options)} option contracts from Polygon for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to get options from Polygon: {e}")

        # 3. FINAL FALLBACK: Synthetic options (for testing/development)
        if not options:
            logger.warning(f"No options data available from any source for {symbol}, using synthetic data")
            options = self._generate_synthetic_options(symbol, current_price)

        # Cache options data
        self._cache_options(symbol, options)

        return options

    def _get_options_from_alpaca(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Fetch option contracts from Alpaca API"""
        options = []

        try:
            # Get available expirations
            url = f"https://data.alpaca.markets/v2/stocks/{symbol}/options/expirations"
            headers = {
                "Apca-Api-Key-Id": self.api_key,
                "Apca-Api-Secret-Key": self.api_secret
            }

            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                logger.error(f"Failed to get option expirations from Alpaca: {response.text}")
                return options

            expirations = response.json().get('expirations', [])

            # Filter to reasonable expirations (7-60 days out)
            today = datetime.now().date()
            valid_expirations = []

            for exp in expirations:
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                days_to_expiry = (exp_date - today).days
                if 7 <= days_to_expiry <= 60:
                    valid_expirations.append(exp)

            # Limit to 3 expirations to avoid API overload
            valid_expirations = valid_expirations[:3]

            # Define strike range - more focused around current price to get liquid options
            min_strike = current_price * 0.85
            max_strike = current_price * 1.15

            # Process each expiration date
            for expiration in valid_expirations:
                exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                days_to_expiry = (exp_date - today).days

                # Get option chain
                chain_url = f"https://data.alpaca.markets/v2/stocks/{symbol}/options/chain"
                params = {
                    "expiration": expiration,
                    "min_strike": min_strike,
                    "max_strike": max_strike
                }

                response = requests.get(chain_url, headers=headers, params=params, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"Failed to get option chain for {expiration}: {response.text}")
                    continue

                chain_data = response.json().get('options', [])

                # Batch process quotes for efficiency
                option_symbols = [contract.get('symbol') for contract in chain_data]
                quotes = self._get_option_quotes_batch(option_symbols, headers)

                # Process each option contract
                for contract in chain_data:
                    option_symbol = contract.get('symbol')
                    if option_symbol not in quotes:
                        continue

                    quote = quotes[option_symbol]

                    # Validate basic parameters for trading viability
                    if not self._is_option_viable(quote, contract):
                        continue

                    # Create option object
                    option = {
                        'symbol': option_symbol,
                        'underlying': symbol,
                        'expiration': expiration,
                        'strike': float(contract.get('strike_price', 0)),
                        'option_type': contract.get('side', '').lower(),  # 'call' or 'put'
                        'bid': float(quote.get('bid', 0)),
                        'ask': float(quote.get('ask', 0)),
                        'price': float(quote.get('ask', 0)),  # Conservative pricing
                        'iv': float(contract.get('implied_volatility', 0.5)),
                        'days_to_expiry': days_to_expiry,
                        'volume': int(quote.get('volume', 0)),
                        'open_interest': int(contract.get('open_interest', 0)),
                        'delta': float(contract.get('delta', 0)),
                        'gamma': float(contract.get('gamma', 0)),
                        'theta': float(contract.get('theta', 0)),
                        'vega': float(contract.get('vega', 0))
                    }

                    options.append(option)

            return options

        except Exception as e:
            logger.error(f"Error in Alpaca options fetching: {e}")
            return []

    def _get_option_quotes_batch(self, option_symbols: List[str], headers: dict) -> Dict[str, dict]:
        """Get quotes for multiple options in batch for efficiency"""
        quotes = {}

        try:
            # Process in batches of 50 to avoid overloading the API
            batch_size = 50
            for i in range(0, len(option_symbols), batch_size):
                batch = option_symbols[i:i + batch_size]
                symbols_str = ",".join(batch)

                url = f"https://data.alpaca.markets/v2/stocks/quotes/latest?symbols={symbols_str}"
                response = requests.get(url, headers=headers, timeout=15)

                if response.status_code != 200:
                    continue

                data = response.json()
                for symbol, quote_data in data.get('quotes', {}).items():
                    quotes[symbol] = quote_data

            return quotes
        except Exception as e:
            logger.error(f"Error getting batch option quotes: {e}")
            return {}

    def _get_options_from_polygon(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Fetch option contracts from Polygon API as backup"""
        options = []

        try:
            # Get available expirations
            url = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={symbol}&apiKey={self.polygon_key}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                logger.error(f"Failed to get option contracts from Polygon: {response.text}")
                return options

            data = response.json()

            # Filter contracts
            today = datetime.now().date()
            viable_contracts = []

            for contract in data.get('results', []):
                exp_date_str = contract.get('expiration_date')
                if not exp_date_str:
                    continue

                exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d').date()
                days_to_expiry = (exp_date - today).days

                # Filter by days to expiry
                if days_to_expiry < 7 or days_to_expiry > 60:
                    continue

                # Filter by strike
                strike = float(contract.get('strike_price', 0))
                if strike < current_price * 0.85 or strike > current_price * 1.15:
                    continue

                viable_contracts.append(contract)

            # Limit to reasonable number of contracts
            viable_contracts = viable_contracts[:100]

            # Process contracts
            for contract in viable_contracts:
                option_symbol = contract.get('ticker')

                # Get quote for this option
                quote_url = f"https://api.polygon.io/v2/last/trade/{option_symbol}?apiKey={self.polygon_key}"
                quote_resp = requests.get(quote_url, timeout=5)

                if quote_resp.status_code != 200:
                    continue

                quote_data = quote_resp.json()
                last_price = quote_data.get('results', {}).get('price', 0)

                # Create option object
                option = {
                    'symbol': option_symbol,
                    'underlying': symbol,
                    'expiration': contract.get('expiration_date'),
                    'strike': float(contract.get('strike_price', 0)),
                    'option_type': 'call' if contract.get('contract_type') == 'call' else 'put',
                    'bid': last_price * 0.95,  # Estimate
                    'ask': last_price * 1.05,  # Estimate
                    'price': last_price,
                    'iv': 0.5,  # Default IV when not available
                    'days_to_expiry': days_to_expiry,
                    'volume': int(quote_data.get('results', {}).get('volume', 0)),
                    'open_interest': 0  # Not available from this endpoint
                }

                options.append(option)

            return options

        except Exception as e:
            logger.error(f"Error in Polygon options fetching: {e}")
            return []

    def _is_option_viable(self, quote: dict, contract: dict) -> bool:
        """Check if an option is viable for trading"""
        # Check for minimum price
        bid = float(quote.get('bid', 0))
        ask = float(quote.get('ask', 0))

        if bid <= 0.05 or ask <= 0.05:
            return False  # Too cheap, likely illiquid

        # Check for reasonable spread
        if bid > 0 and ask > 0:
            spread_pct = (ask - bid) / ask
            if spread_pct > 0.15:  # More than 15% spread
                return False  # Too wide spread

        # Check volume if available
        volume = int(quote.get('volume', 0))
        if volume < 10:  # Minimal volume requirement
            return False

        return True

    def _get_cached_options(self, symbol: str) -> List[Dict[str, Any]]:
        """Get cached options if available and recent"""
        try:
            cache_file = os.path.join("data", "options_cache", f"{symbol}_options.json")
            if os.path.exists(cache_file):
                # Check if cache is fresh (< 30 minutes)
                file_modified = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_modified < timedelta(minutes=30):
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error reading options cache: {e}")
            return []

    def _cache_options(self, symbol: str, options: List[Dict[str, Any]]):
        """Cache options data to reduce API calls"""
        try:
            cache_dir = os.path.join("data", "options_cache")
            os.makedirs(cache_dir, exist_ok=True)

            cache_file = os.path.join(cache_dir, f"{symbol}_options.json")
            with open(cache_file, 'w') as f:
                json.dump(options, f)
        except Exception as e:
            logger.error(f"Error caching options data: {e}")

    def _generate_synthetic_options(self, symbol: str, current_price: float) -> List[Dict[str, Any]]:
        """Generate synthetic options for testing/development"""
        options = []
        today = datetime.now().date()

        # Generate a few expiration dates
        expirations = [
            (today + timedelta(days=14)).strftime('%Y-%m-%d'),  # 2 weeks
            (today + timedelta(days=30)).strftime('%Y-%m-%d'),  # 1 month
            (today + timedelta(days=45)).strftime('%Y-%m-%d')  # 1.5 months
        ]

        # Generate strikes around current price
        strikes = [
            current_price * 0.9,  # 10% ITM
            current_price * 0.95,  # 5% ITM
            current_price,  # ATM
            current_price * 1.05,  # 5% OTM
            current_price * 1.1  # 10% OTM
        ]

        for expiration in expirations:
            days_to_expiry = (datetime.strptime(expiration, '%Y-%m-%d').date() - today).days

            for strike in strikes:
                # Generate synthetic call
                iv = 0.4 + (random.random() * 0.2)  # 40-60% IV
                synthetic_price = self._calc_synthetic_option_price(current_price, strike, days_to_expiry, iv, 'call')

                call = {
                    'symbol': f"{symbol}_{expiration}_C_{strike:.2f}",
                    'underlying': symbol,
                    'expiration': expiration,
                    'strike': strike,
                    'option_type': 'call',
                    'bid': synthetic_price * 0.95,
                    'ask': synthetic_price * 1.05,
                    'price': synthetic_price,
                    'iv': iv,
                    'days_to_expiry': days_to_expiry,
                    'volume': random.randint(50, 1000),
                    'open_interest': random.randint(100, 5000),
                    'is_synthetic': True
                }
                options.append(call)

                # Generate synthetic put
                iv = 0.4 + (random.random() * 0.2)  # 40-60% IV
                synthetic_price = self._calc_synthetic_option_price(current_price, strike, days_to_expiry, iv, 'put')

                put = {
                    'symbol': f"{symbol}_{expiration}_P_{strike:.2f}",
                    'underlying': symbol,
                    'expiration': expiration,
                    'strike': strike,
                    'option_type': 'put',
                    'bid': synthetic_price * 0.95,
                    'ask': synthetic_price * 1.05,
                    'price': synthetic_price,
                    'iv': iv,
                    'days_to_expiry': days_to_expiry,
                    'volume': random.randint(50, 1000),
                    'open_interest': random.randint(100, 5000),
                    'is_synthetic': True
                }
                options.append(put)

        return options

    def _calc_synthetic_option_price(self, spot: float, strike: float, days: int, iv: float, option_type: str) -> float:
        """Calculate a synthetic option price using Black-Scholes approximation"""
        try:
            # Simple approximation for demo/fallback
            moneyness = spot / strike
            time_factor = np.sqrt(days / 365)

            if option_type == 'call':
                if moneyness > 1:  # ITM call
                    return max(0.1, (moneyness - 1) * strike + (iv * spot * time_factor * 0.4))
                else:  # OTM call
                    return max(0.1, iv * spot * time_factor * 0.4)
            else:  # put
                if moneyness < 1:  # ITM put
                    return max(0.1, (1 - moneyness) * strike + (iv * spot * time_factor * 0.4))
                else:  # OTM put
                    return max(0.1, iv * spot * time_factor * 0.4)
        except Exception:
            return 1.0  # Default fallback price

    def _get_current_price(self, symbol: str) -> float:
        """Get current price using Alpaca market data"""
        # Check cache for recent price
        cache_time = 300  # 5 minutes
        if symbol in self.price_cache:
            cached_time, cached_price = self.price_cache[symbol]
            if (datetime.now() - cached_time).seconds < cache_time:
                return cached_price

        try:
            # Request latest quote from Alpaca
            request_params = StockQuotesRequest(
                symbol_or_symbols=symbol,
            )

            quotes = self.data_client.get_stock_latest_quote(request_params)

            if symbol in quotes:
                price = float(quotes[symbol].ask_price)  # Use ask for conservative estimates
                self.price_cache[symbol] = (datetime.now(), price)
                return price

            # Fallback to latest bar if quote unavailable
            bars_request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                limit=1
            )

            bars = self.data_client.get_stock_bars(bars_request)
            if symbol in bars and len(bars[symbol]) > 0:
                price = float(bars[symbol][0].close)
                self.price_cache[symbol] = (datetime.now(), price)
                return price

            logger.error(f"Could not get price for {symbol}")
            raise ValueError(f"Price data unavailable for {symbol}")

        except Exception as e:
            logger.error(f"Error getting current price: {e}")

            # Last resort fallback to cached price if it exists
            if symbol in self.price_cache:
                logger.warning(f"Using cached price for {symbol} due to API error")
                return self.price_cache[symbol][1]

            raise RuntimeError(f"Failed to get price for {symbol}") from e

    def _get_historical_volatility(self, symbol: str, days: int) -> float:
        """Calculate historical volatility over a specific period"""
        try:
            # Fetch historical daily prices
            end_date = datetime.now()
            start_date = end_date - timedelta(days=max(60, days * 2))  # Get enough data

            # Request historical data from Alpaca
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )

            bars = self.data_client.get_stock_bars(request_params)

            if symbol not in bars or len(bars[symbol]) < 10:
                logger.warning(f"Insufficient price history for {symbol}")
                return 0.0

            # Extract close prices
            closes = [bar.close for bar in bars[symbol]]

            # Calculate daily returns
            returns = [np.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]

            # Calculate annualized volatility
            daily_vol = np.std(returns)
            annual_vol = daily_vol * np.sqrt(252)  # Annualize

            return annual_vol

        except Exception as e:
            logger.error(f"Error calculating historical volatility: {e}")
            return 0.0

    def _get_historical_event_performance(self, symbol: str, event_type: str,
                                          option_type: str, days_to_event: int,
                                          days_to_expiry: int) -> Dict[str, Any]:
        """Get historical performance of similar trades"""
        try:
            # Query our database for past trades matching similar parameters
            similar_trades = self.db.execute_query(
                """
                SELECT * FROM trades 
                WHERE underlying = %s 
                AND event_type = %s 
                AND trade_type = %s
                AND status = 'CLOSED'
                AND ABS(EXTRACT(DAY FROM event_date::timestamp - entry_time) - %s) < 5
                AND ABS(days_to_expiry - %s) < 7
                """,
                (symbol, event_type, option_type, days_to_event, days_to_expiry)
            )

            if not similar_trades:
                # Not enough historical data
                return {}

            # Calculate performance metrics
            wins = sum(1 for t in similar_trades if t['pnl'] > 0)
            total = len(similar_trades)
            win_rate = wins / total if total > 0 else 0.5

            # Calculate average return
            if total > 0:
                avg_return = sum(t['pnl_percent'] for t in similar_trades) / total
            else:
                avg_return = 0

            return {
                'win_rate': win_rate,
                'avg_return': avg_return,
                'sample_size': total
            }

        except Exception as e:
            logger.error(f"Error getting historical performance: {e}")
            return {}

    def analyze_opportunities(self, min_sharpe: float = 0.25) -> List[Dict[str, Any]]:
        """
        Analyze options for trading opportunities with statistical validation

        Args:
            min_sharpe: Minimum acceptable Sharpe ratio

        Returns:
            List of validated trading opportunities
        """
        opportunities = []
        start_time = datetime.now()

        try:
            logger.info("Starting opportunity analysis...")

            # Get relevant events
            events = self._filter_upcoming_events()
            if not events:
                logger.info("No upcoming events to analyze")
                return []

            # Limit analysis to conserve API calls
            events = events[:10]  # Analyze top 10 events

            logger.info(f"Analyzing {len(events)} upcoming events")

            # Track overall stats
            analyzed_options = 0
            potential_opportunities = 0

            # For each event, analyze potential trades
            for event in events:
                symbol = event['symbol']
                event_type = event['event_type']
                event_date = event['event_date']

                logger.debug(f"Analyzing {event_type} event for {symbol} on {event_date}")

                # Skip events that have already passed
                if datetime.strptime(event_date, '%Y-%m-%d') < datetime.now():
                    continue

                # Skip if event is too far in the future (>30 days)
                days_to_event = (datetime.strptime(event_date, '%Y-%m-%d') - datetime.now()).days
                if days_to_event > 30:
                    continue

                # Get options chain for this symbol
                options_chain = self._get_option_contracts(symbol)
                if not options_chain:
                    logger.warning(f"No options available for {symbol}")
                    continue

                analyzed_options += len(options_chain)

                # Skip synthetic data in production if configured
                if self.skip_synthetic and any(opt.get('is_synthetic', False) for opt in options_chain):
                    logger.warning(f"Skipping analysis of synthetic data for {symbol}")
                    continue

                # Get historical volatility for comparative analysis
                historical_vol = self._get_historical_volatility(symbol, 30)

                # Filter options based on days to expiry
                valid_options = [
                    opt for opt in options_chain
                    if (
                        # Must expire after the event
                            datetime.strptime(opt['expiration'], '%Y-%m-%d') > datetime.strptime(event_date, '%Y-%m-%d')
                            # But not too far out
                            and opt['days_to_expiry'] < 60
                            # Must have reasonable bid/ask
                            and opt['bid'] > 0 and opt['ask'] > 0
                            # Exclude synthetic in production mode
                            and (not self.production_mode or not opt.get('is_synthetic', False))
                    )
                ]

                # Find the best opportunities based on event type and strategy
                valid_strategies = self._get_valid_strategies(event)

                for strategy_name, strategy_params in valid_strategies.items():
                    # Apply strategy-specific option filters
                    strategy_options = self._filter_options_for_strategy(
                        valid_options,
                        strategy_name,
                        strategy_params
                    )

                    for option in strategy_options:
                        # Calculate quantitative edge (used for all strategies)
                        edge = self._calculate_edge(option, event)

                        # Skip options with insufficient edge
                        if edge < 0.2:  # Minimum 20% edge
                            continue

                        # Statistically validate this trade idea
                        validation_results = self._validate_with_historical_data(
                            symbol,
                            event_type,
                            option['option_type'],
                            option['strike'],
                            option['days_to_expiry'],
                            days_to_event
                        )

                        # Skip opportunities with insufficient historical evidence
                        if not self._is_historically_validated(validation_results):
                            continue

                        # Calculate expected return and probability metrics
                        expected_roi, profit_prob = self._calculate_expected_return(
                            option,
                            edge,
                            validation_results
                        )

                        # Calculate risk-adjusted metrics
                        sharpe_ratio = self._calculate_sharpe_ratio(
                            expected_roi,
                            option,
                            validation_results
                        )

                        # Final filter by Sharpe ratio
                        if sharpe_ratio >= min_sharpe:
                            potential_opportunities += 1

                            # Determine position sizing
                            kelly_fraction = self._calculate_kelly_fraction(profit_prob, expected_roi)

                            # Create the opportunity object
                            opportunity = {
                                'option': option,
                                'event': event,
                                'strategy': strategy_name,
                                'edge': edge,
                                'expected_roi': expected_roi,
                                'profit_probability': profit_prob,
                                'sharpe_ratio': sharpe_ratio,
                                'kelly_fraction': kelly_fraction,
                                'historical_validation': validation_results,
                                'max_contracts': int(10000 / (option['price'] * 100)),  # Basic cash limit
                                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }

                            opportunities.append(opportunity)

            # Sort opportunities by Sharpe ratio
            opportunities.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

            # Log analysis summary
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Analysis complete: processed {analyzed_options} options in {elapsed:.1f}s")
            logger.info(f"Found {len(opportunities)} viable opportunities out of {potential_opportunities} candidates")

            return opportunities

        except Exception as e:
            logger.error(f"Error in opportunity analysis: {e}", exc_info=True)
            return []

    def _filter_upcoming_events(self) -> List[Dict[str, Any]]:
        """Filter events to only include upcoming relevant events"""
        today = datetime.now().date()

        # Get all events
        all_events = self._load_events_calendar()

        # Filter to upcoming events within reasonable timeframe
        upcoming_events = []
        for event in all_events:
            try:
                event_date = datetime.strptime(event['event_date'], '%Y-%m-%d').date()

                # Must be in the future but not too far out
                if today <= event_date <= (today + timedelta(days=30)):
                    upcoming_events.append(event)
            except Exception:
                continue

        # Sort by importance then date
        return sorted(
            upcoming_events,
            key=lambda e: (
                0 if e.get('importance') == 'high' else
                1 if e.get('importance') == 'medium' else 2,
                e['event_date']
            )
        )

    def _get_valid_strategies(self, event: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get valid strategies for a given event type"""
        event_type = event['event_type']

        strategies = {}

        if event_type == 'earnings':
            # For earnings events, consider multiple strategies
            strategies['long_straddle'] = {
                'moneyness_min': 0.97,  # Near ATM
                'moneyness_max': 1.03,
                'days_min': 3,  # Not too close to expiration
                'days_max': 45  # Not too far out
            }

            strategies['long_call'] = {
                'moneyness_min': 0.95,  # Slightly ITM to ATM
                'moneyness_max': 1.05,  # to slightly OTM
                'days_min': 5,
                'days_max': 30
            }

        elif event_type == 'dividend':
            # Dividend events strategies
            strategies['covered_call'] = {
                'moneyness_min': 1.02,  # Slightly OTM
                'moneyness_max': 1.15,
                'days_min': 10,
                'days_max': 45
            }

        elif event_type in ['product_launch', 'watchlist']:
            # For other events, prefer calls
            strategies['long_call'] = {
                'moneyness_min': 0.95,
                'moneyness_max': 1.10,
                'days_min': 7,
                'days_max': 45
            }

        elif event_type == 'economic':
            # For economic events like FOMC
            strategies['long_straddle'] = {
                'moneyness_min': 0.98,
                'moneyness_max': 1.02,
                'days_min': 1,
                'days_max': 10
            }

        return strategies

    def _filter_options_for_strategy(self, options: List[Dict[str, Any]],
                                     strategy: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter options based on strategy parameters"""
        filtered_options = []

        # Get current price for moneyness calculation
        if options:
            current_price = self._get_current_price(options[0]['underlying'])
        else:
            return []

        for option in options:
            # Calculate moneyness
            strike = option['strike']
            moneyness = strike / current_price

            # Check moneyness range
            if moneyness < params['moneyness_min'] or moneyness > params['moneyness_max']:
                continue

            # Check days to expiry
            if option['days_to_expiry'] < params['days_min'] or option['days_to_expiry'] > params['days_max']:
                continue

            # Strategy-specific filters
            if strategy == 'long_straddle':
                # For straddles, we want liquid options
                bid_ask_spread = (option['ask'] - option['bid']) / option['ask'] if option['ask'] > 0 else 1.0
                if bid_ask_spread > 0.1:  # More than 10% spread
                    continue

            elif strategy == 'long_call' and option['option_type'] != 'call':
                continue

            elif strategy == 'covered_call' and option['option_type'] != 'call':
                continue

            # Add option if it passes all filters
            filtered_options.append(option)

        return filtered_options

    def _validate_with_historical_data(self, symbol: str, event_type: str,
                                       option_type: str, strike_price: float,
                                       days_to_expiry: int, days_to_event: int) -> Dict[str, Any]:
        """
        Validate a potential trade using historical performance data

        Returns:
            Dictionary with validation metrics
        """
        try:
            # Look up similar historical trades in our database
            similar_trades = self.db.execute_query(
                """
                SELECT * FROM trades 
                WHERE underlying = %s 
                AND event_type = %s 
                AND trade_type = %s
                AND status = 'CLOSED'
                AND ABS(EXTRACT(DAY FROM event_date::timestamp - entry_time) - %s) < 7
                AND ABS(days_to_expiry - %s) < 10
                """,
                (symbol, event_type, option_type, days_to_event, days_to_expiry)
            )

            # If not enough data in our system, check market-wide similar events
            if len(similar_trades) < 5:
                # This would be an external API call or lookup to a research database
                # For now, we'll simulate this with reasonable defaults based on event type
                market_win_rate = self._get_market_statistics(event_type, option_type)

                # Create baseline validation results
                return {
                    'sample_size': 0,
                    'win_rate': market_win_rate,
                    'avg_return': 0.0,
                    'max_profit': 0.0,
                    'max_loss': 0.0,
                    'avg_holding_days': days_to_expiry / 2,  # Estimate
                    'source': 'market_estimate',
                    'confidence': 'low'
                }

            # Calculate metrics from similar trades
            wins = [t for t in similar_trades if t['pnl'] > 0]
            losses = [t for t in similar_trades if t['pnl'] <= 0]

            win_rate = len(wins) / len(similar_trades) if similar_trades else 0.5

            avg_return = sum(t['pnl_percent'] for t in similar_trades) / len(similar_trades) if similar_trades else 0

            max_profit = max([t['pnl_percent'] for t in wins]) if wins else 0
            max_loss = min([t['pnl_percent'] for t in losses]) if losses else 0

            # Calculate average holding period
            holding_days = []
            for trade in similar_trades:
                if trade['entry_time'] and trade['exit_time']:
                    entry = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                    exit = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00'))
                    days = (exit - entry).total_seconds() / 86400  # Convert to days
                    holding_days.append(days)

            avg_holding = sum(holding_days) / len(holding_days) if holding_days else days_to_expiry / 2

            # Determine confidence level based on sample size
            confidence = 'low'
            if len(similar_trades) >= 10:
                confidence = 'medium'
            if len(similar_trades) >= 20:
                confidence = 'high'

            return {
                'sample_size': len(similar_trades),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'avg_holding_days': avg_holding,
                'source': 'historical_trades',
                'confidence': confidence
            }

        except Exception as e:
            logger.error(f"Error in historical validation: {e}")
            # Return default values
            return {
                'sample_size': 0,
                'win_rate': 0.5,
                'avg_return': 0.0,
                'max_profit': 0.0,
                'max_loss': 0.0,
                'avg_holding_days': days_to_expiry / 2,
                'source': 'error',
                'confidence': 'none'
            }

    def _get_market_statistics(self, event_type: str, option_type: str) -> float:
        """Get market-wide statistics for similar events"""
        # These would ideally come from a research database
        # Using reasonable defaults based on event type and option type

        # Default win rates based on event type and option type
        win_rates = {
            'earnings': {
                'call': 0.52,
                'put': 0.48,
                'straddle': 0.55
            },
            'dividend': {
                'call': 0.60,
                'put': 0.40
            },
            'product_launch': {
                'call': 0.58,
                'put': 0.42
            },
            'economic': {
                'call': 0.51,
                'put': 0.49,
                'straddle': 0.54
            },
            'watchlist': {
                'call': 0.51,
                'put': 0.49
            }
        }

        # Get appropriate win rate
        if event_type in win_rates and option_type in win_rates[event_type]:
            return win_rates[event_type][option_type]

        # Default
        return 0.50

    def _is_historically_validated(self, validation: Dict[str, Any]) -> bool:
        """Check if a trade is sufficiently validated by historical data"""
        # Must have positive expected return
        if validation['avg_return'] <= 0:
            return False

        # Must have reasonable win rate
        if validation['win_rate'] < 0.52:  # At least 52% win rate
            return False

        # Check confidence level
        if validation['confidence'] == 'none':
            return False

        # In production mode, require higher standards
        if self.production_mode:
            if validation['sample_size'] < 5:
                return False
            if validation['win_rate'] < 0.55:  # Higher win rate in production
                return False

        return True

    def _calculate_sharpe_ratio(self, expected_roi: float, option: Dict[str, Any],
                                validation: Dict[str, Any]) -> float:
        """Calculate Sharpe ratio for risk-adjusted returns"""
        # Get volatility from option or validation data
        if validation['sample_size'] > 5 and 'max_loss' in validation:
            # Use historical volatility if available
            volatility = abs(validation['max_loss']) / 2
        else:
            # Otherwise use option's implied volatility
            volatility = option['iv'] * 100  # Convert to percentage

        # Avoid division by zero
        if volatility <= 0:
            volatility = 20  # Default 20% volatility

        # Adjust volatility based on days to expiry
        time_factor = np.sqrt(option['days_to_expiry'] / 252)  # Annualize
        adjusted_vol = volatility * time_factor

        # Calculate Sharpe (risk-adjusted return)
        sharpe = expected_roi / adjusted_vol if adjusted_vol > 0 else 0

        return max(sharpe, 0.0)  # Ensure non-negative

    def _calculate_kelly_fraction(self, win_probability: float, expected_roi: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion

        Args:
            win_probability: Probability of winning (0-1)
            expected_roi: Expected return on investment (percentage)

        Returns:
            Kelly fraction (0-1)
        """
        # Sanity checks
        if win_probability <= 0 or win_probability >= 1:
            return 0.05  # Default to 5% if probability is invalid

        if expected_roi <= 0:
            return 0

        # Convert ROI to decimal gain (e.g., 20% -> 0.2)
        expected_gain = expected_roi / 100

        # Calculate loss factor (e.g., if we lose 100%)
        loss_probability = 1 - win_probability
        loss_factor = 1.0  # Full loss by default for options

        # Apply Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = gain/loss ratio
        gain_loss_ratio = expected_gain / loss_factor
        kelly = (win_probability * gain_loss_ratio - loss_probability) / gain_loss_ratio

        # Apply conservative adjustment (half-Kelly)
        conservative_kelly = kelly * 0.5

        # Limit to reasonable bounds
        return max(min(conservative_kelly, 0.25), 0.0)  # Cap at 25% of capital

    def _calculate_edge(self, option: Dict[str, Any], event: Dict[str, Any]) -> float:
        """
        Calculate the true trading edge based on market inefficiencies
        Returns a value between 0-1 representing the estimated edge
        """
        try:
            # Get underlying data
            symbol = option['underlying']
            option_type = option['option_type']
            strike = option['strike']
            days_to_expiry = option['days_to_expiry']
            current_price = self._get_current_price(symbol)
            iv = option['iv']

            # 1. Calculate historical vs implied volatility edge
            # Get historical volatility for matching timeframe
            historical_vol = self._get_historical_volatility(symbol, days_to_expiry)

            # No edge if historical data unavailable
            if historical_vol <= 0:
                return 0.0

            # Calculate vol ratio (key indicator of mispricing)
            vol_ratio = iv / historical_vol

            # Higher ratio = potentially overpriced options
            # Lower ratio = potentially underpriced options
            vol_edge = 0.0
            if vol_ratio < 0.8:
                # Underpriced - positive edge for buying
                vol_edge = min((0.8 - vol_ratio) * 2, 0.5)  # Max 0.5 edge from vol
            elif vol_ratio > 1.2:
                # Overpriced - positive edge for selling
                vol_edge = min((vol_ratio - 1.2) * 0.5, 0.5)  # Max 0.5 edge from vol
                # Only consider selling edge if that's our strategy
                if option_type == 'call' and strike > current_price * 1.05:
                    # Keep edge for OTM call selling
                    pass
                elif option_type == 'put' and strike < current_price * 0.95:
                    # Keep edge for OTM put selling
                    pass
                else:
                    # Not a selling strategy match
                    vol_edge = 0.0

            # 2. Calculate event-specific historical edge
            event_type = event['event_type']
            days_to_event = (datetime.strptime(event['event_date'], '%Y-%m-%d') - datetime.now()).days

            # Get historical performance of similar event trades
            hist_performance = self._get_historical_event_performance(
                symbol,
                event_type,
                option_type,
                days_to_event,
                days_to_expiry
            )

            # Calculate edge from historical performance
            event_edge = 0.0
            if hist_performance:
                # Use win rate and average return to calculate edge
                win_rate = hist_performance.get('win_rate', 0.5)
                avg_return = hist_performance.get('avg_return', 0)

                # Only consider significant edges
                if win_rate > 0.55 and avg_return > 0:
                    event_edge = min((win_rate - 0.5) * 2, 0.4)  # Max 0.4 edge from history

            # 3. Liquidity and execution quality edge
            bid_ask_spread = (option['ask'] - option['bid']) / option['ask'] if option['ask'] > 0 else 1.0

            # Smaller spread = better execution = higher edge
            liquidity_edge = 0.0
            if bid_ask_spread < 0.05:  # Tight spread of 5% or less
                liquidity_edge = 0.1
            elif bid_ask_spread > 0.15:  # Wide spread reduces edge
                liquidity_edge = -0.1

            # 4. Calculate moneyness edge
            # At-the-money options often have the most liquid markets and reliable pricing
            moneyness = abs(strike / current_price - 1.0)
            moneyness_edge = max(0.1 - moneyness, 0)  # Max 0.1 edge for ATM options

            # 5. Calculate time decay edge for theta strategies
            decay_edge = 0.0
            if days_to_expiry < 14 and days_to_expiry > 3:
                # Sweet spot for theta decay
                decay_edge = 0.1

            # 6. Combine all edge factors with appropriate weights
            total_edge = (
                    (vol_edge * 0.35) +  # 35% weight for vol mispricing
                    (event_edge * 0.30) +  # 30% weight for event-specific edge
                    (liquidity_edge * 0.15) +  # 15% weight for liquidity
                    (moneyness_edge * 0.10) +  # 10% weight for strike positioning
                    (decay_edge * 0.10)  # 10% weight for time decay
            )

            # Log detailed edge calculation for monitoring
            logger.debug(f"Edge calculation for {symbol} {option_type}:")
            logger.debug(f"  Vol edge: {vol_edge:.2f}, Event edge: {event_edge:.2f}")
            logger.debug(f"  Liquidity edge: {liquidity_edge:.2f}, Moneyness edge: {moneyness_edge:.2f}")
            logger.debug(f"  Decay edge: {decay_edge:.2f}, Total edge: {total_edge:.2f}")

            # Ensure edge is between 0 and 1
            return max(min(total_edge, 1.0), 0.0)

        except Exception as e:
            logger.error(f"Error calculating edge: {e}")
            return 0.0  # No edge if calculation fails


    def _calculate_expected_return(self, option: Dict[str, Any], edge: float) -> Tuple[float, float]:
        """
        Calculate expected ROI and probability of profit
        Returns (expected_roi_percent, probability_of_profit)
        """
        # Extract option properties
        price = option['price']
        iv = option['iv']
        days_to_expiry = option['days_to_expiry']

        # Calculate probability of profit based on edge
        # Edge of 0.5 should correspond to roughly 60% probability
        prob_of_profit = 0.5 + (edge * 0.2)

        # Calculate expected return
        # Higher edge and IV suggest higher potential returns
        potential_return_pct = iv * 100 * np.sqrt(days_to_expiry / 30) * edge

        # Expected ROI is probability-weighted return
        expected_roi = (prob_of_profit * potential_return_pct) - ((1 - prob_of_profit) * 100)

        return expected_roi, prob_of_profit

    def execute_trade(self, opportunity: Dict[str, Any], contracts: int = None) -> bool:
        """
        Execute a trade with comprehensive risk management controls

        Args:
            opportunity: Trading opportunity with option and analysis data
            contracts: Number of contracts to trade (if None, will be calculated)

        Returns:
            Boolean indicating if trade was successfully executed
        """
        try:
            option = opportunity['option']

            # 1. PRE-TRADE SAFETY CHECKS

            # Skip synthetic data in production
            if option.get('is_synthetic', False):
                logger.warning(f"Preventing trade execution on synthetic data for {option['symbol']}")
                return False

            # Verify option is tradable
            if not self._verify_option_tradable(option):
                logger.warning(f"Option {option['symbol']} not tradable at this time")
                return False

            # Check if market is open
            if not self._is_market_open():
                logger.warning("Market is closed - cannot execute trade")
                return False

            # 2. RISK LIMITS CHECK

            # Get portfolio and account data for risk checks
            portfolio_stats = self._get_portfolio_statistics()

            # Check overall position count
            if portfolio_stats['open_position_count'] >= portfolio_stats['max_positions']:
                logger.warning(f"Maximum positions limit reached ({portfolio_stats['open_position_count']})")
                return False

            # Check strategy allocation
            strategy = opportunity.get('strategy', 'default')
            if not self._check_strategy_allocation(strategy, option['price']):
                logger.warning(f"Maximum allocation for strategy {strategy} reached")
                return False

            # Check concentration for this underlying
            symbol = option['underlying']
            if not self._check_symbol_concentration(symbol, option['price']):
                logger.warning(f"Maximum concentration for {symbol} reached")
                return False

            # 3. POSITION SIZING

            # Use provided contracts or calculate optimal size
            if contracts is None:
                # Calculate optimal position size using Kelly criterion
                kelly_fraction = opportunity.get('kelly_fraction', 0.05)
                max_risk_amount = portfolio_stats['portfolio_value'] * kelly_fraction
                option_cost = option['price'] * 100  # Cost per contract

                # Calculate max contracts based on risk allocation
                optimal_contracts = int(max_risk_amount / option_cost) if option_cost > 0 else 0

                # Set minimum and maximum limits
                optimal_contracts = max(1, min(optimal_contracts, 10))  # At least 1, at most 10
                contracts = optimal_contracts

            # Final verification of trade size vs available capital
            total_cost = option['price'] * contracts * 100
            if total_cost > portfolio_stats['buying_power'] * 0.95:  # Leave 5% buffer
                # Reduce size to fit available capital
                max_affordable = int((portfolio_stats['buying_power'] * 0.95) / (option['price'] * 100))
                if max_affordable < 1:
                    logger.warning(f"Insufficient buying power for trade: needed ${total_cost:.2f}")
                    return False
                contracts = max_affordable
                total_cost = option['price'] * contracts * 100
                logger.info(f"Reduced position size to {contracts} contracts due to capital constraints")

            # 4. EXECUTION STRATEGY

            # Determine maximum acceptable slippage
            max_slippage_pct = 0.05  # 5% maximum slippage

            # Calculate limit price with slippage buffer
            limit_price = option['ask'] * (1 + max_slippage_pct / 2)

            # If bid-ask spread is too wide, use a more conservative limit
            bid_ask_spread = (option['ask'] - option['bid']) / option['ask'] if option['ask'] > 0 else 1.0
            if bid_ask_spread > 0.1:  # >10% spread
                # Use a more conservative limit (closer to ask)
                limit_price = option['ask'] * 1.01  # Only 1% above ask
                logger.info(f"Wide spread detected ({bid_ask_spread:.1%}), using conservative limit")

            # Set a maximum price ceiling for catastrophic error prevention
            price_ceiling = option['ask'] * 1.15  # Absolute maximum 15% above ask

            # 5. EXECUTE ORDER

            logger.info(f"Executing trade: {contracts} contracts of {option['symbol']} at limit ${limit_price:.2f}")

            # Create order object
            order = LimitOrderRequest(
                symbol=option['symbol'],
                qty=contracts,
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
                extended_hours=False
            )

            # Submit order
            order_result = self.trading_client.submit_order(order)
            order_id = order_result.id

            logger.info(f"Order {order_id} submitted for {option['symbol']}")

            # 6. MONITOR EXECUTION

            # Wait for fill with timeout
            filled = False
            max_wait_seconds = 60  # Maximum wait time
            start_time = datetime.now()

            while (datetime.now() - start_time).total_seconds() < max_wait_seconds:
                # Check order status
                time.sleep(5)  # Check every 5 seconds

                try:
                    order_status = self.trading_client.get_order_by_id(order_id)

                    if order_status.status == OrderStatus.FILLED:
                        filled = True

                        # Extract actual execution details
                        filled_qty = int(order_status.filled_qty)
                        filled_price = float(order_status.filled_avg_price)

                        # Check if execution price is reasonable
                        if filled_price > price_ceiling:
                            logger.warning(f"Order filled at unusually high price: ${filled_price:.2f}")

                        logger.info(f"Order {order_id} filled: {filled_qty} contracts at ${filled_price:.2f}")
                        break

                    elif order_status.status in [OrderStatus.REJECTED, OrderStatus.CANCELED]:
                        logger.error(f"Order {order_id} {order_status.status}: {order_status.reject_reason}")
                        return False

                    # If still pending after half the wait time, modify order to improve fill chances
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > max_wait_seconds / 2 and order_status.status == OrderStatus.NEW:
                        # Increase limit price slightly (by 1%)
                        new_limit = limit_price * 1.01
                        if new_limit <= price_ceiling:
                            try:
                                self.trading_client.replace_order(
                                    order_id=order_id,
                                    qty=contracts,
                                    limit_price=new_limit
                                )
                                logger.info(f"Modified order {order_id} limit to ${new_limit:.2f}")
                                limit_price = new_limit
                            except Exception as e:
                                logger.warning(f"Could not modify order: {e}")

                except Exception as e:
                    logger.error(f"Error checking order status: {e}")
                    # Continue trying

            # If not filled, cancel the order
            if not filled:
                try:
                    self.trading_client.cancel_order_by_id(order_id)
                    logger.warning(f"Cancelled unfilled order {order_id} after {max_wait_seconds} seconds")
                except Exception:
                    pass
                return False

            # 7. POST-TRADE PROCESSING

            # Use actual fill price and quantity
            actual_price = filled_price
            actual_qty = filled_qty
            total_cost = actual_price * actual_qty * 100

            # Update account
            self.buying_power -= total_cost

            # Record trade in internal tracking
            position = {
                'symbol': option['symbol'],
                'underlying': option['underlying'],
                'type': option['option_type'],
                'strike': option['strike'],
                'expiration': option['expiration'],
                'days_to_expiry': option['days_to_expiry'],
                'entry_price': actual_price,
                'contracts': actual_qty,
                'value': total_cost,
                'entry_time': datetime.now().isoformat(),
                'event': opportunity['event'],
                'expected_roi': opportunity['expected_roi'],
                'profit_probability': opportunity['profit_probability'],
                'strategy': opportunity.get('strategy', 'default'),
                'order_id': order_id,
                'status': 'open'
            }

            # Add to positions list
            self.positions.append(position)

            # 8. LOG TRADE TO DATABASE

            # Create trade record for database
            trade_data = {
                'option': option,
                'order_id': order_id,
                'entry_price': actual_price,
                'contracts': actual_qty,
                'entry_time': datetime.now().isoformat(),
                'total_cost': total_cost,
                'expected_roi': opportunity['expected_roi']
            }

            # Add to database
            self.db.log_trade(trade_data)

            # 9. SEND NOTIFICATIONS

            # Create notification
            notification = {
                'type': 'trade_notification',
                'trade_data': trade_data,
                'action': 'entry'
            }

            # Send to notification system
            if hasattr(self, 'message_queue'):
                self.message_queue.put(notification)

            logger.info(f"Trade successfully executed and recorded: {option['symbol']}")
            return True

        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
            return False

    def _verify_option_tradable(self, option: Dict[str, Any]) -> bool:
        """Verify that an option is currently tradable"""
        try:
            # Skip if no meaningful price
            if option['ask'] <= 0 or option['price'] <= 0:
                return False

            # Check expiration (no 0DTE trading)
            if option['days_to_expiry'] <= 0:
                return False

            # Check if there is sufficient liquidity
            if option.get('volume', 0) < 10:  # Minimal volume
                return False

            # Check for wide spreads
            spread_pct = (option['ask'] - option['bid']) / option['ask'] if option['ask'] > 0 else 1.0
            if spread_pct > 0.2:  # >20% spread indicates poor liquidity
                return False

            return True
        except Exception as e:
            logger.error(f"Error verifying option tradability: {e}")
            return False

    def _is_market_open(self) -> bool:
        """Check if the market is currently open for trading"""
        try:
            # Get the clock from Alpaca
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False  # Assume closed on errors

    def _get_portfolio_statistics(self) -> Dict[str, Any]:
        """Get current portfolio statistics for risk management"""
        try:
            # Get account information
            account = self.get_account()

            # Get open positions
            positions = self.get_positions()

            # Calculate statistics
            stats = {
                'portfolio_value': float(account['portfolio_value']),
                'buying_power': float(account['buying_power']),
                'cash': float(account['cash']),
                'open_position_count': len(positions),
                'max_positions': 10,  # Default limit

                # Risk limits from config
                'max_strategy_allocation': 0.20,  # 20% per strategy
                'max_symbol_concentration': 0.15,  # 15% per symbol

                # Strategy allocations
                'strategy_allocations': self._calculate_strategy_allocations(positions),

                # Symbol concentrations
                'symbol_concentrations': self._calculate_symbol_concentrations(positions)
            }

            return stats
        except Exception as e:
            logger.error(f"Error getting portfolio statistics: {e}")
            # Return conservative defaults
            return {
                'portfolio_value': self.buying_power,
                'buying_power': self.buying_power * 0.95,
                'cash': self.buying_power * 0.95,
                'open_position_count': len(self.positions),
                'max_positions': 5,  # More conservative limit on error
                'max_strategy_allocation': 0.15,
                'max_symbol_concentration': 0.10,
                'strategy_allocations': {},
                'symbol_concentrations': {}
            }

    def _calculate_strategy_allocations(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate current allocation by strategy"""
        allocations = {}

        for position in positions:
            strategy = position.get('strategy', 'default')
            value = position.get('value', 0.0)

            if strategy not in allocations:
                allocations[strategy] = 0.0

            allocations[strategy] += value

        return allocations

    def _calculate_symbol_concentrations(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate current concentration by underlying symbol"""
        concentrations = {}

        for position in positions:
            underlying = position.get('underlying', '')
            value = position.get('value', 0.0)

            if underlying not in concentrations:
                concentrations[underlying] = 0.0

            concentrations[underlying] += value

        return concentrations

    def _check_strategy_allocation(self, strategy: str, option_price: float, contracts: int = 1) -> bool:
        """Check if a trade would exceed strategy allocation limits"""
        try:
            # Get portfolio statistics
            stats = self._get_portfolio_statistics()

            # Get current allocation for this strategy
            current_allocation = stats['strategy_allocations'].get(strategy, 0.0)

            # Calculate new allocation
            new_position_value = option_price * contracts * 100
            new_allocation = current_allocation + new_position_value

            # Calculate maximum allowed allocation
            max_allocation = stats['portfolio_value'] * stats['max_strategy_allocation']

            # Check if new allocation is within limits
            return new_allocation <= max_allocation

        except Exception as e:
            logger.error(f"Error checking strategy allocation: {e}")
            return False  # Conservative approach

    def _check_symbol_concentration(self, symbol: str, option_price: float, contracts: int = 1) -> bool:
        """Check if a trade would exceed symbol concentration limits"""
        try:
            # Get portfolio statistics
            stats = self._get_portfolio_statistics()

            # Get current concentration for this symbol
            current_concentration = stats['symbol_concentrations'].get(symbol, 0.0)

            # Calculate new concentration
            new_position_value = option_price * contracts * 100
            new_concentration = current_concentration + new_position_value

            # Calculate maximum allowed concentration
            max_concentration = stats['portfolio_value'] * stats['max_symbol_concentration']

            # Check if new concentration is within limits
            return new_concentration <= max_concentration

        except Exception as e:
            logger.error(f"Error checking symbol concentration: {e}")
            return False  # Conservative approach

    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current open positions from Alpaca"""
        try:
            # Get positions from Alpaca
            api_positions = self.trading_client.get_all_positions()

            # Update our internal position tracking
            updated_positions = []

            for position in api_positions:
                symbol = position.symbol

                # Only process option positions (symbol format: O:AAPL230505C00165000)
                if not symbol.startswith('O:'):
                    continue

                # Extract option details
                qty = int(position.qty)
                avg_entry_price = float(position.avg_entry_price)
                market_value = float(position.market_value)
                current_price = market_value / (qty * 100)

                # Find matching internal position
                internal_pos = next((p for p in self.positions if p['symbol'] == symbol), None)

                if internal_pos:
                    # Update with latest data
                    internal_pos['current_price'] = current_price
                    internal_pos['market_value'] = market_value
                    updated_positions.append(internal_pos)
                else:
                    # Create new position record if not in our tracking
                    new_pos = {
                        'symbol': symbol,
                        'underlying': symbol.split(':')[1].split('2')[0],  # Extract ticker
                        'contracts': qty,
                        'entry_price': avg_entry_price,
                        'current_price': current_price,
                        'market_value': market_value,
                        'status': 'open'
                    }
                    updated_positions.append(new_pos)

            # Update internal positions list
            self.positions = updated_positions
            return updated_positions

        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return [p for p in self.positions if p.get('status', 'open') == 'open']

    def get_account(self) -> Dict[str, Any]:
        """Get account information from Alpaca"""
        try:
            account = self.trading_client.get_account()
            return {
                'portfolio_value': float(account.portfolio_value),
                'buying_power': float(account.buying_power),
                'cash': float(account.cash)
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {
                'portfolio_value': 25000.0,
                'buying_power': self.buying_power,
                'cash': self.buying_power
            }

    def monitor_positions(self) -> List[Dict[str, Any]]:
        """
        Monitor existing positions and execute exit strategies
        """
        positions = self.get_positions()
        if not positions:
            return []

        closed_positions = []

        for position in positions:
            if position.get('status', 'open') != 'open':
                continue

            # Check if we should exit the position
            exit_signal, exit_price, exit_reason = self._check_exit_signals(position)

            if exit_signal:
                # Execute exit
                success = self._exit_position(position, exit_price, exit_reason)

                if success:
                    position['status'] = 'closed'
                    position['exit_price'] = exit_price
                    position['exit_time'] = datetime.now().isoformat()
                    position['exit_reason'] = exit_reason

                    # Calculate P&L
                    entry_cost = position['entry_price'] * position['contracts'] * 100
                    exit_value = exit_price * position['contracts'] * 100
                    position['pnl'] = exit_value - entry_cost
                    position['pnl_pct'] = (exit_value / entry_cost - 1) * 100

                    # Update buying power
                    self.buying_power += exit_value

                    logger.info(
                        f"Closed position: {position['symbol']} - P&L: ${position['pnl']:.2f} ({position['pnl_pct']:.2f}%)")

                    closed_positions.append(position)

        return closed_positions

    def _check_exit_signals(self, position: Dict[str, Any]) -> Tuple[bool, float, str]:
        """
        Check if we should exit a position based on real-time data
        """
        symbol = position['symbol']

        try:
            # Get real-time data for this option
            url = f"https://data.alpaca.markets/v2/stocks/{symbol}/quotes/latest"
            headers = {
                "Apca-Api-Key-Id": self.api_key,
                "Apca-Api-Secret-Key": self.api_secret
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                logger.warning(f"Unable to get quote for {symbol}: {response.text}")
                # If we can't get live data, use simulation
                return self._check_exit_signals_fallback(position)

            quote = response.json().get('quote', {})
            current_price = float(quote.get('bid_price', 0))  # Use bid for conservative exits

            if current_price <= 0:
                logger.warning(f"Invalid price for {symbol}: {current_price}")
                return self._check_exit_signals_fallback(position)

            # Calculate days held
            entry_time = datetime.fromisoformat(position['entry_time'])
            days_held = (datetime.now() - entry_time).total_seconds() / 86400

            # Update position with current price
            position['current_price'] = current_price

            # Calculate P&L
            entry_cost = position['entry_price'] * position['contracts'] * 100
            current_value = current_price * position['contracts'] * 100
            unrealized_pnl_pct = (current_value / entry_cost - 1) * 100

            # Check expiration approach
            days_to_expiry = position.get('days_to_expiry', 30) - days_held
            if days_to_expiry <= 1:
                return True, current_price, "approaching_expiration"

            # Check profit target (50% of expected ROI or 20% absolute)
            profit_target = position.get('expected_roi', 40) * 0.5
            profit_target = max(profit_target, 20)  # At least 20%

            if unrealized_pnl_pct >= profit_target:
                return True, current_price, "profit_target_reached"

            # Check stop loss (-100% of expected ROI or -50% absolute)
            stop_loss = -position.get('expected_roi', 40)
            stop_loss = min(stop_loss, -50)  # At most -50%

            if unrealized_pnl_pct <= stop_loss:
                return True, current_price, "stop_loss_triggered"

            # Time-based exit (held more than half time to expiry)
            if days_held > position.get('days_to_expiry', 30) * 0.6:
                # Exit if we're profitable
                if unrealized_pnl_pct > 0:
                    return True, current_price, "time_exit_profitable"

                # Also exit if we've lost more than 30%
                if unrealized_pnl_pct < -30:
                    return True, current_price, "time_exit_with_loss"

            # Event passed exit
            if 'event' in position:
                event_date = datetime.strptime(position['event']['event_date'], '%Y-%m-%d')
                if datetime.now() > event_date + timedelta(days=1):
                    return True, current_price, "event_passed"

            # No exit signal
            return False, current_price, ""

        except Exception as e:
            logger.error(f"Error checking exit signals: {e}")
            return self._check_exit_signals_fallback(position)

    def _check_exit_signals_fallback(self, position: Dict[str, Any]) -> Tuple[bool, float, str]:
        """Fallback exit check when real-time data unavailable"""
        # Get current price (estimated)
        current_price = position.get('current_price', position['entry_price'] * 0.9)

        # Calculate days held
        entry_time = datetime.fromisoformat(position['entry_time'])
        days_held = (datetime.now() - entry_time).total_seconds() / 86400

        # Emergency exit if held too long
        if days_held > 5:  # Conservative emergency exit
            return True, current_price, "emergency_exit_no_data"

        return False, current_price, ""

    def _exit_position(self, position: Dict[str, Any], exit_price: float, exit_reason: str) -> bool:
        """
        Exit a position using a real order
        """
        try:
            symbol = position['symbol']
            qty = position['contracts']

            # Place exit order
            order = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )

            order_result = self.trading_client.submit_order(order)
            order_id = order_result.id

            # Wait for fill
            filled = False
            max_attempts = 10
            for _ in range(max_attempts):
                time.sleep(2)
                order_status = self.trading_client.get_order_by_id(order_id)
                if order_status.status == OrderStatus.FILLED:
                    filled = True
                    # Use actual fill price if available
                    if hasattr(order_status, 'filled_avg_price') and order_status.filled_avg_price:
                        exit_price = float(order_status.filled_avg_price)
                    break
                elif order_status.status in [OrderStatus.REJECTED, OrderStatus.CANCELED]:
                    logger.error(f"Exit order {order_id} {order_status.status}: {order_status.reject_reason}")
                    return False

            if not filled:
                logger.error(f"Exit order {order_id} not filled after {max_attempts * 2} seconds")
                return False

            logger.info(f"Exit order {order_id} filled: {qty} contracts of {symbol} at ${exit_price}")
            return True

        except Exception as e:
            logger.error(f"Error exiting position: {e}")
            return False

    def monitor_strategy_performance(self):
        """
        Monitor strategy performance metrics and adapt parameters dynamically
        Executed periodically to ensure strategy remains effective
        """
        try:
            logger.info("Performing strategy performance monitoring")

            # 1. GATHER PERFORMANCE DATA

            # Get recent performance metrics
            recent_trades = self._get_recent_trades(days=30)
            if not recent_trades:
                logger.info("Not enough recent trades to analyze performance")
                return

            # Calculate current performance metrics
            performance_metrics = self._calculate_performance_metrics(recent_trades)

            # Load historical strategy metrics for comparison
            historical_metrics = self._load_historical_metrics()

            # 2. DETECT PERFORMANCE CHANGES

            # Check for performance degradation
            degradation = self._detect_performance_degradation(performance_metrics, historical_metrics)
            if degradation['detected']:
                logger.warning(f"Strategy performance degradation detected: {degradation['metric']} "
                               f"({degradation['current']:.2f} vs {degradation['baseline']:.2f})")

                # Adjust strategy parameters based on degradation
                self._adjust_strategy_parameters(degradation)

                # Log the adjustment
                logger.info(f"Adjusted strategy parameters due to {degradation['metric']} degradation")

            # 3. DETECT MARKET REGIME CHANGES

            # Check market conditions
            regime_change = self._detect_market_regime_change()
            if regime_change['detected']:
                logger.warning(f"Market regime change detected: {regime_change['description']}")

                # Recalibrate strategy for new regime
                self._recalibrate_for_market_regime(regime_change['type'])

                # Log the recalibration
                logger.info(f"Recalibrated strategy for {regime_change['type']} market regime")

            # 4. CHECK CIRCUIT BREAKERS

            # Verify if critical metrics are beyond safety thresholds
            breakers_triggered = self._check_performance_circuit_breakers(performance_metrics)
            if breakers_triggered:
                logger.critical(f"Performance circuit breaker triggered: {breakers_triggered}")

                # Take defensive action
                self._apply_defensive_measures(breakers_triggered)

                # Send high-priority notification
                self._send_circuit_breaker_alert(breakers_triggered, performance_metrics)

            # 5. UPDATE STRATEGY PARAMETERS

            # Make routine parameter adjustments based on recent performance
            self._update_strategy_parameters(performance_metrics)

            # 6. PERSIST METRICS AND CHANGES

            # Save current metrics for future comparison
            self._save_performance_metrics(performance_metrics)

            # Log complete performance report
            logger.info(f"Strategy monitoring complete - Sharpe: {performance_metrics['sharpe_ratio']:.2f}, "
                        f"Win Rate: {performance_metrics['win_rate']:.1f}%, "
                        f"Profit Factor: {performance_metrics['profit_factor']:.2f}")

        except Exception as e:
            logger.error(f"Error in strategy performance monitoring: {e}", exc_info=True)

    def _get_recent_trades(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get recent closed trades for performance analysis"""
        try:
            # Query database for recent closed trades
            trades = self.db.execute_query(
                "SELECT * FROM trades WHERE status = 'CLOSED' AND exit_time >= NOW() - INTERVAL %s DAY",
                (days,)
            )
            return trades
        except Exception as e:
            logger.error(f"Error retrieving recent trades: {e}")
            return []

    def _calculate_performance_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive performance metrics from trade data"""
        if not trades:
            return self._get_default_metrics()

        try:
            # Basic trade statistics
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl', 0) <= 0]

            total_trades = len(trades)
            win_count = len(winning_trades)

            # Win rate
            win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

            # Profit metrics
            gross_profits = sum(t.get('pnl', 0) for t in winning_trades)
            gross_losses = abs(sum(t.get('pnl', 0) for t in losing_trades))
            net_profit = gross_profits - gross_losses

            # Risk metrics
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
            avg_win = gross_profits / win_count if win_count > 0 else 0
            avg_loss = gross_losses / len(losing_trades) if losing_trades else 0
            win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')

            # Get daily P&L values for Sharpe calculation
            daily_pnl = {}
            for trade in trades:
                exit_date = trade.get('exit_time', '').split('T')[0]
                if exit_date:
                    daily_pnl[exit_date] = daily_pnl.get(exit_date, 0) + trade.get('pnl', 0)

            daily_returns = list(daily_pnl.values())

            # Calculate Sharpe ratio if we have enough data
            sharpe_ratio = 0
            if len(daily_returns) > 5:
                avg_return = sum(daily_returns) / len(daily_returns)
                std_dev = (sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns)) ** 0.5
                sharpe_ratio = avg_return / std_dev if std_dev > 0 else 0
                # Annualize
                sharpe_ratio *= (252 ** 0.5)

            # Calculate max drawdown
            cumulative_pnl = 0
            peak = 0
            drawdowns = []

            for trade in sorted(trades, key=lambda x: x.get('exit_time', '')):
                cumulative_pnl += trade.get('pnl', 0)
                if cumulative_pnl > peak:
                    peak = cumulative_pnl
                if peak > 0:
                    drawdown = (peak - cumulative_pnl) / peak
                    drawdowns.append(drawdown)

            max_drawdown = max(drawdowns) if drawdowns else 0

            # Strategy metrics by type
            strategy_performance = {}
            for trade in trades:
                strategy = trade.get('strategy', 'default')
                if strategy not in strategy_performance:
                    strategy_performance[strategy] = {
                        'count': 0,
                        'wins': 0,
                        'total_pnl': 0
                    }

                strategy_performance[strategy]['count'] += 1
                if trade.get('pnl', 0) > 0:
                    strategy_performance[strategy]['wins'] += 1
                strategy_performance[strategy]['total_pnl'] += trade.get('pnl', 0)

            # Calculate win rates by strategy
            for strategy in strategy_performance:
                count = strategy_performance[strategy]['count']
                if count > 0:
                    strategy_performance[strategy]['win_rate'] = (
                            strategy_performance[strategy]['wins'] / count * 100
                    )
                else:
                    strategy_performance[strategy]['win_rate'] = 0

            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'net_profit': net_profit,
                'profit_factor': profit_factor,
                'win_loss_ratio': win_loss_ratio,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'strategy_performance': strategy_performance,
                'calculation_time': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._get_default_metrics()

    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when not enough data is available"""
        return {
            'total_trades': 0,
            'win_rate': 0,
            'net_profit': 0,
            'profit_factor': 0,
            'win_loss_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'strategy_performance': {},
            'calculation_time': datetime.now().isoformat()
        }

    def _load_historical_metrics(self) -> Dict[str, Any]:
        """Load historical performance metrics for comparison"""
        try:
            # Check if metrics file exists
            metrics_file = os.path.join("data", "strategy_metrics.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading historical metrics: {e}")
            return {}

    def _save_performance_metrics(self, metrics: Dict[str, Any]):
        """Save current performance metrics for future reference"""
        try:
            # Ensure directory exists
            os.makedirs("data", exist_ok=True)

            metrics_file = os.path.join("data", "strategy_metrics.json")

            # Load existing metrics if any
            existing_metrics = {}
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    existing_metrics = json.load(f)

            # Add timestamp for this metrics snapshot
            timestamp = datetime.now().strftime('%Y-%m-%d')
            if 'history' not in existing_metrics:
                existing_metrics['history'] = {}

            # Store current metrics in historical record
            existing_metrics['history'][timestamp] = metrics

            # Keep only last 90 days of metrics
            cutoff_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            existing_metrics['history'] = {
                k: v for k, v in existing_metrics['history'].items() if k >= cutoff_date
            }

            # Update current metrics
            existing_metrics['current'] = metrics

            # Save back to file
            with open(metrics_file, 'w') as f:
                json.dump(existing_metrics, f, default=str)

        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")

    def _detect_performance_degradation(self, current: Dict[str, Any],
                                        historical: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect significant performance degradation from baseline

        Returns:
            Dictionary with degradation details
        """
        degradation = {
            'detected': False,
            'metric': None,
            'current': 0,
            'baseline': 0,
            'percent_change': 0
        }

        # If no historical data, can't detect degradation
        if not historical or 'current' not in historical:
            return degradation

        baseline = historical['current']

        # Check win rate (needs at least 10 trades)
        if current['total_trades'] >= 10 and baseline.get('win_rate', 0) > 0:
            win_rate_change = (current['win_rate'] - baseline['win_rate']) / baseline['win_rate']
            if win_rate_change < -0.15:  # Win rate dropped by >15%
                degradation['detected'] = True
                degradation['metric'] = 'win_rate'
                degradation['current'] = current['win_rate']
                degradation['baseline'] = baseline['win_rate']
                degradation['percent_change'] = win_rate_change * 100
                return degradation

        # Check profit factor (if we have enough data)
        if current['total_trades'] >= 10 and baseline.get('profit_factor', 0) > 1.2:
            pf_change = (current['profit_factor'] - baseline['profit_factor']) / baseline['profit_factor']
            if pf_change < -0.20:  # Profit factor dropped by >20%
                degradation['detected'] = True
                degradation['metric'] = 'profit_factor'
                degradation['current'] = current['profit_factor']
                degradation['baseline'] = baseline['profit_factor']
                degradation['percent_change'] = pf_change * 100
                return degradation

        # Check Sharpe ratio (if we have enough data)
        if baseline.get('sharpe_ratio', 0) > 0.5:
            sharpe_change = current['sharpe_ratio'] - baseline['sharpe_ratio']
            if sharpe_change < -0.5:  # Absolute drop of 0.5 in Sharpe
                degradation['detected'] = True
                degradation['metric'] = 'sharpe_ratio'
                degradation['current'] = current['sharpe_ratio']
                degradation['baseline'] = baseline['sharpe_ratio']
                degradation['percent_change'] = (sharpe_change / baseline['sharpe_ratio']) * 100 if baseline[
                                                                                                        'sharpe_ratio'] > 0 else 0
                return degradation

        # Check max drawdown (if we have enough data)
        if current['max_drawdown'] > 0 and baseline.get('max_drawdown', 1) > 0:
            dd_change = current['max_drawdown'] / baseline['max_drawdown']
            if dd_change > 1.5 and current['max_drawdown'] > 0.15:  # 50% worse drawdown & above 15%
                degradation['detected'] = True
                degradation['metric'] = 'max_drawdown'
                degradation['current'] = current['max_drawdown']
                degradation['baseline'] = baseline['max_drawdown']
                degradation['percent_change'] = (dd_change - 1) * 100
                return degradation

        return degradation

    def _detect_market_regime_change(self) -> Dict[str, Any]:
        """
        Detect changes in market regime that could affect strategy performance

        Returns:
            Dictionary with regime change details
        """
        regime_change = {
            'detected': False,
            'type': None,
            'description': ''
        }

        try:
            # Get market indicators
            vix = self._get_market_volatility()
            market_trend = self._get_market_trend()
            rate_environment = self._get_interest_rate_environment()

            # Check for volatility regime change
            if vix > 30 and market_trend['recent_change'] < -0.05:
                regime_change['detected'] = True
                regime_change['type'] = 'high_volatility'
                regime_change['description'] = f"High volatility regime (VIX: {vix:.1f})"
                return regime_change

            # Check for strong directional regime
            if abs(market_trend['normalized_slope']) > 2.0:
                direction = 'bullish' if market_trend['normalized_slope'] > 0 else 'bearish'
                regime_change['detected'] = True
                regime_change['type'] = f'strong_{direction}'
                regime_change['description'] = f"Strong {direction} trend detected"
                return regime_change

            # Check for rate environment changes
            if rate_environment['significant_change']:
                regime_change['detected'] = True
                regime_change['type'] = 'rate_environment_change'
                regime_change['description'] = f"Interest rate environment change: {rate_environment['description']}"
                return regime_change

            # Check for correlation breakdown
            correlation_change = self._check_correlation_breakdown()
            if correlation_change['detected']:
                regime_change['detected'] = True
                regime_change['type'] = 'correlation_breakdown'
                regime_change['description'] = correlation_change['description']
                return regime_change

            return regime_change

        except Exception as e:
            logger.error(f"Error detecting market regime change: {e}")
            return regime_change

    def _get_market_volatility(self) -> float:
        """Get current market volatility (VIX equivalent)"""
        try:
            # In a real implementation, this would fetch VIX data from an API
            # For this example, we'll simulate it
            spy_data = self._get_historical_price_data('SPY', days=30)
            if not spy_data:
                return 20.0  # Default value

            # Calculate historical volatility
            returns = []
            for i in range(1, len(spy_data)):
                returns.append(spy_data[i] / spy_data[i - 1] - 1)

            # Annualize volatility (standard deviation of returns)
            vol = np.std(returns) * np.sqrt(252) * 100

            return vol
        except Exception:
            return 20.0  # Default value on error

    def _get_market_trend(self) -> Dict[str, Any]:
        """Analyze market trend characteristics"""
        try:
            # Get SPY historical data
            spy_data = self._get_historical_price_data('SPY', days=60)
            if not spy_data or len(spy_data) < 20:
                return {
                    'direction': 'neutral',
                    'strength': 0,
                    'normalized_slope': 0,
                    'recent_change': 0
                }

            # Calculate simple linear regression
            x = np.arange(len(spy_data))
            y = np.array(spy_data)
            n = len(x)

            # Simple linear regression slope calculation
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            slope = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

            # Normalize slope by price level and days
            normalized_slope = slope / y_mean * 100

            # Calculate recent change (last 5 days)
            recent_change = (spy_data[-1] / spy_data[-6] - 1) if len(spy_data) >= 6 else 0

            # Determine direction and strength
            if normalized_slope > 0.1:
                direction = 'bullish'
                strength = normalized_slope / 0.1  # Normalize so 0.1% daily change = 1.0 strength
            elif normalized_slope < -0.1:
                direction = 'bearish'
                strength = abs(normalized_slope) / 0.1
            else:
                direction = 'neutral'
                strength = 0

            return {
                'direction': direction,
                'strength': strength,
                'normalized_slope': normalized_slope,
                'recent_change': recent_change
            }

        except Exception:
            return {
                'direction': 'neutral',
                'strength': 0,
                'normalized_slope': 0,
                'recent_change': 0
            }

    def _get_interest_rate_environment(self) -> Dict[str, Any]:
        """Analyze interest rate environment"""
        # In a real implementation, this would fetch rate data from an API
        # For simplicity, we'll return a simulated result
        return {
            'significant_change': False,
            'direction': 'stable',
            'description': 'No significant rate changes'
        }

    def _check_correlation_breakdown(self) -> Dict[str, Any]:
        """Check for breakdowns in normal market correlations"""
        # In a real implementation, this would analyze correlations
        # For simplicity, we'll return a simulated result
        return {
            'detected': False,
            'description': 'Normal market correlations maintained'
        }

    def _get_historical_price_data(self, symbol: str, days: int = 30) -> List[float]:
        """Get historical price data for analysis"""
        try:
            # In a real implementation, this would fetch from market data API
            # For this implementation, use Alpaca API
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Request historical data from Alpaca
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )

            bars = self.data_client.get_stock_bars(request_params)

            if symbol in bars:
                return [bar.close for bar in bars[symbol]]
            return []

        except Exception as e:
            logger.error(f"Error getting historical price data: {e}")
            return []

    def _check_performance_circuit_breakers(self, metrics: Dict[str, Any]) -> str:
        """
        Check if any performance metrics have breached critical thresholds

        Returns:
            String describing breaker triggered, or empty string if none
        """
        # Check for excessive drawdown
        if metrics['max_drawdown'] > 0.20:  # >20% drawdown
            return "excessive_drawdown"

        # Check for extremely poor win rate
        if metrics['total_trades'] >= 20 and metrics['win_rate'] < 40:
            return "low_win_rate"

        # Check for negative profit factor with meaningful sample
        if metrics['total_trades'] >= 15 and metrics['profit_factor'] < 0.8:
            return "negative_expectancy"

        # Check for excessive average loss
        if metrics['avg_loss'] > 0 and metrics['avg_win'] > 0:
            if metrics['win_loss_ratio'] < 0.5 and metrics['total_trades'] > 10:
                return "poor_reward_risk"

        return ""  # No breakers triggered

    def _adjust_strategy_parameters(self, degradation: Dict[str, Any]):
        """Adjust strategy parameters based on performance degradation"""
        metric = degradation['metric']

        if metric == 'win_rate':
            # Tighten entry criteria to improve quality
            self.min_edge_threshold = min(0.3, self.min_edge_threshold + 0.05)
            self.min_probability = min(0.6, self.min_probability + 0.05)

        elif metric == 'profit_factor':
            # Adjust position sizing and profit targets
            self.position_size_factor = max(0.5, self.position_size_factor * 0.8)
            self.profit_target_factor = max(1.2, self.profit_target_factor * 1.1)

        elif metric == 'sharpe_ratio':
            # Reduce overall risk and increase diversification
            self.max_strategy_allocation = max(0.15, self.max_strategy_allocation * 0.8)

        elif metric == 'max_drawdown':
            # Implement stricter risk management
            self.stop_loss_factor = min(0.75, self.stop_loss_factor * 0.9)
            self.max_symbol_concentration = max(0.1, self.max_symbol_concentration * 0.75)

        # Store adjustment in parameters history
        self._log_parameter_adjustment(metric, degradation['percent_change'])

    def _recalibrate_for_market_regime(self, regime_type: str):
        """Recalibrate strategy parameters for the current market regime"""
        if regime_type == 'high_volatility':
            # Reduce position sizes, tighten stops, increase profit targets
            self.position_size_factor = max(0.5, self.position_size_factor * 0.7)
            self.stop_loss_factor = min(0.7, self.stop_loss_factor * 0.8)
            self.profit_target_factor = min(2.0, self.profit_target_factor * 1.2)

        elif regime_type == 'strong_bullish':
            # Focus more on call options, longer durations
            self.call_bias_factor = min(1.5, self.call_bias_factor * 1.2)
            self.put_bias_factor = max(0.5, self.put_bias_factor * 0.8)
            self.duration_preference = 'longer'

        elif regime_type == 'strong_bearish':
            # Focus more on put options, shorter durations
            self.put_bias_factor = min(1.5, self.put_bias_factor * 1.2)
            self.call_bias_factor = max(0.5, self.call_bias_factor * 0.8)
            self.duration_preference = 'shorter'

        elif regime_type == 'rate_environment_change':
            # Adjust for interest rate impact on options pricing
            # This would be more complex in reality
            self.min_edge_threshold = min(0.3, self.min_edge_threshold + 0.03)

        elif regime_type == 'correlation_breakdown':
            # Reduce overall exposure and increase diversification
            self.max_positions = max(5, self.max_positions - 2)
            self.max_strategy_allocation = max(0.15, self.max_strategy_allocation * 0.8)

        # Store recalibration in parameters history
        self._log_parameter_adjustment('market_regime', regime_type)

    def _apply_defensive_measures(self, breaker_type: str):
        """Apply defensive measures when circuit breaker is triggered"""
        if breaker_type == 'excessive_drawdown':
            # Significant reduction in exposure
            self.position_size_factor = max(0.3, self.position_size_factor * 0.5)
            self.max_positions = max(3, self.max_positions - 3)
            self.trading_paused = True  # Consider pausing new trades temporarily

        elif breaker_type == 'low_win_rate':
            # Major adjustment to entry criteria
            self.min_edge_threshold = min(0.35, self.min_edge_threshold + 0.1)
            self.min_probability = min(0.65, self.min_probability + 0.1)
            self.position_size_factor = max(0.4, self.position_size_factor * 0.6)

        elif breaker_type == 'negative_expectancy':
            # Most aggressive defensive posture
            self.position_size_factor = max(0.25, self.position_size_factor * 0.4)
            self.trading_paused = True  # Pause new trades
            self.max_positions = max(2, self.max_positions - 5)

        elif breaker_type == 'poor_reward_risk':
            # Adjust risk-reward parameters
            self.stop_loss_factor = min(0.6, self.stop_loss_factor * 0.7)
            self.profit_target_factor = min(2.5, self.profit_target_factor * 1.3)

        # Set recovery mode flag
        self.recovery_mode = True
        self.recovery_start_time = datetime.now().isoformat()

        # Log defensive action
        logger.warning(f"Applied defensive measures due to {breaker_type} circuit breaker")
        self._log_parameter_adjustment('circuit_breaker', breaker_type)

    def _send_circuit_breaker_alert(self, breaker_type: str, metrics: Dict[str, Any]):
        """Send high-priority alert when circuit breaker is triggered"""
        message = (
            f"CRITICAL ALERT: Performance circuit breaker triggered\n"
            f"Type: {breaker_type}\n"
            f"Metrics:\n"
            f"- Win Rate: {metrics['win_rate']:.1f}%\n"
            f"- Profit Factor: {metrics['profit_factor']:.2f}\n"
            f"- Max Drawdown: {metrics['max_drawdown']:.1f}%\n"
            f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n\n"
            f"Defensive measures have been automatically applied."
        )

        # Send high-priority notification
        if hasattr(self, 'message_queue'):
            self.message_queue.put({
                'type': 'notification',
                'message': message,
                'level': 'critical'
            })

    def _update_strategy_parameters(self, metrics: Dict[str, Any]):
        """Make routine adjustments to strategy parameters based on performance"""
        # Only make adjustments with sufficient data
        if metrics['total_trades'] < 10:
            return

        # Slight adjustments to position sizing based on Sharpe ratio
        if metrics['sharpe_ratio'] > 1.5:
            # Increase position size slightly if doing well
            self.position_size_factor = min(1.2, self.position_size_factor * 1.05)
        elif metrics['sharpe_ratio'] < 0.8:
            # Decrease position size slightly if underperforming
            self.position_size_factor = max(0.5, self.position_size_factor * 0.95)

        # Adjust strategy allocations based on strategy performance
        if 'strategy_performance' in metrics and metrics['strategy_performance']:
            self._adjust_strategy_allocations(metrics['strategy_performance'])

        # If in recovery mode, check if we can exit
        if hasattr(self, 'recovery_mode') and self.recovery_mode:
            recovery_duration = (datetime.now() - datetime.fromisoformat(self.recovery_start_time)).days

            # Exit recovery mode after 14 days if metrics are acceptable
            if recovery_duration > 14 and metrics['sharpe_ratio'] > 0.8 and metrics['win_rate'] > 50:
                self.recovery_mode = False
                self.trading_paused = False
                logger.info("Exiting recovery mode - performance has stabilized")

    def _adjust_strategy_allocations(self, strategy_performance: Dict[str, Dict[str, Any]]):
        """Adjust allocation limits based on strategy performance"""
        # Find best and worst performing strategies
        sorted_strategies = sorted(
            strategy_performance.items(),
            key=lambda x: (x[1].get('win_rate', 0), x[1].get('total_pnl', 0)),
            reverse=True
        )

        # Only make adjustments with enough data per strategy
        strategies_with_data = [(s, data) for s, data in sorted_strategies if data['count'] >= 5]

        if len(strategies_with_data) < 2:
            return

        # Get best and worst performing strategies
        best_strategy, best_data = strategies_with_data[0]
        worst_strategy, worst_data = strategies_with_data[-1]

        # Make modest adjustments to allocation factors
        if not hasattr(self, 'strategy_allocation_factors'):
            self.strategy_allocation_factors = {}

        # Initialize factors if needed
        for strategy in strategy_performance:
            if strategy not in self.strategy_allocation_factors:
                self.strategy_allocation_factors[strategy] = 1.0

        # Adjust factors
        self.strategy_allocation_factors[best_strategy] = min(
            1.5, self.strategy_allocation_factors[best_strategy] * 1.05
        )

        self.strategy_allocation_factors[worst_strategy] = max(
            0.5, self.strategy_allocation_factors[worst_strategy] * 0.95
        )

        logger.debug(f"Adjusted strategy allocations: {best_strategy}: "
                     f"{self.strategy_allocation_factors[best_strategy]:.2f}, "
                     f"{worst_strategy}: {self.strategy_allocation_factors[worst_strategy]:.2f}")

    def _log_parameter_adjustment(self, reason: str, details: Any):
        """Log parameter adjustments for tracking and analysis"""
        try:
            params_file = os.path.join("data", "parameter_adjustments.json")

            # Create structure if file doesn't exist
            if not os.path.exists(params_file):
                adjustments = {
                    'history': []
                }
            else:
                with open(params_file, 'r') as f:
                    adjustments = json.load(f)

            # Create record of adjustment
            adjustment = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'details': details,
                'parameters': {
                    # Record current parameter values
                    'position_size_factor': getattr(self, 'position_size_factor', 1.0),
                    'min_edge_threshold': getattr(self, 'min_edge_threshold', 0.2),
                    'min_probability': getattr(self, 'min_probability', 0.55),
                    'max_positions': getattr(self, 'max_positions', 10),
                    'profit_target_factor': getattr(self, 'profit_target_factor', 1.0),
                    'stop_loss_factor': getattr(self, 'stop_loss_factor', 1.0),
                    'max_strategy_allocation': getattr(self, 'max_strategy_allocation', 0.2),
                    'recovery_mode': getattr(self, 'recovery_mode', False)
                }
            }

            # Add to history
            adjustments['history'].append(adjustment)

            # Save to file
            with open(params_file, 'w') as f:
                json.dump(adjustments, f, default=str)

        except Exception as e:
            logger.error(f"Error logging parameter adjustment: {e}")
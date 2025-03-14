"""
Multi-Provider Options Data Service

Combines Alpaca and Polygon free tiers to provide comprehensive options data
with efficient caching and rate limiting.
"""

import os
import json
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest

# Polygon imports
from polygon import RESTClient

logger = logging.getLogger(__name__)


class CachedData:
    """Container for cached data with timestamp"""

    def __init__(self, data, timestamp=None):
        self.data = data
        self.timestamp = timestamp or time.time()

    def age_seconds(self):
        """Get age of cached data in seconds"""
        return time.time() - self.timestamp

    def is_valid(self, max_age_seconds):
        """Check if cached data is still valid"""
        return self.age_seconds() < max_age_seconds


class RateLimit:
    """Rate limiter for API calls"""

    def __init__(self, calls_per_minute):
        self.calls_per_minute = calls_per_minute
        self.interval = 60.0 / calls_per_minute if calls_per_minute > 0 else 0
        self.last_call_time = 0
        self.call_count = 0

    def wait_if_needed(self):
        """Wait if necessary to respect rate limit"""
        if self.calls_per_minute <= 0:
            return

        now = time.time()
        time_since_last = now - self.last_call_time

        if time_since_last < self.interval:
            sleep_time = self.interval - time_since_last
            time.sleep(sleep_time)

        self.last_call_time = time.time()
        self.call_count += 1


class MultiProviderOptionsData:
    """
    Combined options data provider using both Alpaca and Polygon APIs.

    This class optimizes data access by:
    1. Using aggressive caching to minimize API calls
    2. Implementing rate limiting for free tier APIs
    3. Providing fallbacks between providers
    4. Handling errors gracefully
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the options data provider

        Args:
            config: Configuration dictionary with API keys
        """
        # Extract API keys from config
        self.alpaca_key = config.get('alpaca', {}).get('api_key', '')
        self.alpaca_secret = config.get('alpaca', {}).get('api_secret', '')
        self.polygon_key = config.get('polygon', {}).get('api_key', '')

        # Initialize clients
        self._init_clients()

        # Set up caching
        self.cache_dir = os.path.join("data", "options_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.memory_cache = {}

        # Default TTL values (in seconds)
        self.option_chain_ttl = 24 * 60 * 60  # 24 hours for EOD options data
        self.price_ttl = 60  # 60 seconds for prices

        # Set up rate limiting
        self.polygon_rate_limit = RateLimit(5)  # 5 calls per minute for free tier
        self.alpaca_rate_limit = RateLimit(200)  # 200 calls per minute for free tier

        # Provider preference order
        self.provider_preference = config.get('provider_preference', ['polygon', 'alpaca'])

        logger.info("Multi-provider options data service initialized")

    def _init_clients(self):
        """Initialize API clients"""
        # Init Alpaca clients
        try:
            self.alpaca_trading = TradingClient(self.alpaca_key, self.alpaca_secret, paper=True)
            self.alpaca_data = StockHistoricalDataClient(self.alpaca_key, self.alpaca_secret)
            self.alpaca_available = True
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            self.alpaca_available = False

        # Init Polygon client
        try:
            self.polygon_client = RESTClient(self.polygon_key)
            self.polygon_available = True
        except Exception as e:
            logger.error(f"Failed to initialize Polygon client: {e}")
            self.polygon_available = False

    def get_option_chain(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get option chain for a symbol, combining data from multiple providers

        Args:
            symbol: The underlying symbol

        Returns:
            List of option contracts
        """
        # Try to get from cache first
        cached_data = self._get_cached_options(symbol)
        if cached_data:
            logger.info(f"Using cached option chain for {symbol}")
            return cached_data

        options = []
        errors = []

        # Try providers in preference order
        for provider in self.provider_preference:
            if provider == 'polygon' and self.polygon_available:
                try:
                    options = self._get_options_from_polygon(symbol)
                    if options:
                        break
                except Exception as e:
                    errors.append(f"Polygon error: {e}")

            elif provider == 'alpaca' and self.alpaca_available:
                try:
                    options = self._get_options_from_alpaca(symbol)
                    if options:
                        break
                except Exception as e:
                    errors.append(f"Alpaca error: {e}")

        # If no data from any provider, log errors
        if not options and errors:
            logger.warning(f"Failed to get options for {symbol}: {'; '.join(errors)}")

        # If options were found, cache them
        if options:
            self._cache_options(symbol, options)

        return options

    def _get_options_from_polygon(self, symbol: str) -> List[Dict[str, Any]]:
        """Get options from Polygon API"""
        logger.debug(f"Fetching options from Polygon for {symbol}")
        options = []

        try:
            # Get current price for filtering strikes
            current_price = self.get_current_price(symbol)
            if not current_price or current_price <= 0:
                logger.warning(f"Invalid price for {symbol}, using default filtering")
                current_price = 100  # Default for filtering

            # Apply rate limiting
            self.polygon_rate_limit.wait_if_needed()

            # Use v3 reference endpoint for options contracts
            url = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={symbol}&apiKey={self.polygon_key}"

            # Add expiration and strike filters to minimize data
            min_strike = current_price * 0.8
            max_strike = current_price * 1.2
            today = datetime.now().date()
            min_expiry = (today + timedelta(days=7)).isoformat()
            max_expiry = (today + timedelta(days=60)).isoformat()

            params = {
                "expiration_date.gte": min_expiry,
                "expiration_date.lte": max_expiry,
                "strike_price.gte": min_strike,
                "strike_price.lte": max_strike
            }

            response = requests.get(url, params=params, timeout=15)

            if response.status_code != 200:
                logger.warning(f"Polygon API error: {response.status_code} - {response.text}")
                return []

            data = response.json()
            results = data.get('results', [])

            # Process results into standardized format
            for contract in results:
                # Basic fields all providers should have
                option = {
                    'symbol': contract.get('ticker'),
                    'underlying': symbol,
                    'expiration': contract.get('expiration_date'),
                    'strike': float(contract.get('strike_price', 0)),
                    'option_type': 'call' if contract.get('contract_type') == 'call' else 'put',
                    'contract_size': float(contract.get('contract_size', 100)),
                    'source': 'polygon'
                }

                # Get additional data for this contract
                self._enrich_polygon_option(option)

                # Only add if we got price data
                if 'price' in option and option['price'] > 0:
                    options.append(option)

            logger.info(f"Got {len(options)} contracts from Polygon for {symbol}")
            return options

        except Exception as e:
            logger.error(f"Error getting Polygon options: {e}")
            raise

    def _enrich_polygon_option(self, option: Dict[str, Any]):
        """Add additional data to an option from Polygon"""
        try:
            # Get price data for this option
            self.polygon_rate_limit.wait_if_needed()

            # Use last trade endpoint for pricing
            url = f"https://api.polygon.io/v2/last/trade/{option['symbol']}?apiKey={self.polygon_key}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if 'results' in data:
                    trade = data['results']
                    price = float(trade.get('p', 0))

                    # Add pricing data
                    option['price'] = price
                    option['last_price'] = price
                    option['bid'] = price * 0.95  # Estimate for free tier
                    option['ask'] = price * 1.05  # Estimate for free tier

                    # Add trading data
                    option['volume'] = int(trade.get('v', 0))
                    option['last_updated'] = trade.get('t', 0)

                    # Calculate days to expiry
                    exp_date = datetime.fromisoformat(option['expiration']).date()
                    today = datetime.now().date()
                    option['days_to_expiry'] = (exp_date - today).days

                    # Estimate Greeks for free tier
                    self._estimate_option_greeks(option)
            else:
                logger.warning(f"Failed to get price for {option['symbol']}: {response.status_code}")

        except Exception as e:
            logger.warning(f"Error enriching option {option['symbol']}: {e}")

    def _get_options_from_alpaca(self, symbol: str) -> List[Dict[str, Any]]:
        """Get options from Alpaca API using v1beta1 endpoint"""
        logger.debug(f"Fetching options from Alpaca for {symbol}")
        options = []

        try:
            # Get current price for filtering strikes
            current_price = self.get_current_price(symbol)
            if not current_price or current_price <= 0:
                logger.warning(f"Invalid price for {symbol}, using default filtering")
                current_price = 100  # Default for filtering

            # Apply rate limiting
            self.alpaca_rate_limit.wait_if_needed()

            # Use v1beta1 endpoint (works with free tier)
            url = f"https://data.alpaca.markets/v1beta1/options/{symbol}/chain"
            headers = {
                "Apca-Api-Key-Id": self.alpaca_key,
                "Apca-Api-Secret-Key": self.alpaca_secret
            }

            # Get current date and a date 60 days in the future
            today = datetime.now().date()
            future_date = today + timedelta(days=60)

            # Define strike range - more focused around current price
            min_strike = current_price * 0.8
            max_strike = current_price * 1.2

            # Parameters for filtering options
            params = {
                "underlyingSymbol": symbol,
                "expirationDateGte": today.isoformat(),
                "expirationDateLte": future_date.isoformat(),
                "strikePriceGte": min_strike,
                "strikePriceLte": max_strike
            }

            response = requests.get(url, headers=headers, params=params, timeout=15)

            if response.status_code != 200:
                logger.warning(f"Alpaca API error: {response.status_code} - {response.text}")
                return []

            data = response.json()
            contracts = data.get('options', [])

            # Process contracts into standardized format
            for contract in contracts:
                exp_date = datetime.fromisoformat(contract.get('expirationDate')).date()
                days_to_expiry = (exp_date - today).days

                # Skip options that are too close or too far
                if days_to_expiry < 7 or days_to_expiry > 60:
                    continue

                # Basic fields
                option = {
                    'symbol': contract.get('symbol'),
                    'underlying': symbol,
                    'expiration': contract.get('expirationDate'),
                    'strike': float(contract.get('strikePrice', 0)),
                    'option_type': 'call' if contract.get('type') == 'call' else 'put',
                    'days_to_expiry': days_to_expiry,
                    'source': 'alpaca'
                }

                # Add price data
                bid = float(contract.get('bid', 0))
                ask = float(contract.get('ask', 0))
                option['bid'] = bid
                option['ask'] = ask
                option['price'] = ask  # Conservative price
                option['volume'] = int(contract.get('volume', 0))
                option['open_interest'] = int(contract.get('openInterest', 0))

                # Only add if we have pricing data
                if bid > 0 and ask > 0:
                    # Add Greeks if available, or estimate
                    if 'delta' in contract:
                        option['delta'] = float(contract.get('delta', 0))
                        option['gamma'] = float(contract.get('gamma', 0))
                        option['theta'] = float(contract.get('theta', 0))
                        option['vega'] = float(contract.get('vega', 0))
                    else:
                        self._estimate_option_greeks(option)

                    options.append(option)

            logger.info(f"Got {len(options)} contracts from Alpaca for {symbol}")
            return options

        except Exception as e:
            logger.error(f"Error getting Alpaca options: {e}")
            raise

    def _estimate_option_greeks(self, option: Dict[str, Any]):
        """Provide simple estimates for option Greeks when not available"""
        try:
            # These are very simplified estimates for illustration
            current_price = self.get_current_price(option['underlying'])
            strike = option['strike']
            days = option['days_to_expiry']
            is_call = option['option_type'] == 'call'

            # Simple delta estimate based on moneyness
            moneyness = current_price / strike

            if is_call:
                if moneyness > 1.10:  # Deep ITM call
                    option['delta'] = 0.9
                elif moneyness > 1.05:  # ITM call
                    option['delta'] = 0.7
                elif moneyness > 0.95:  # ATM call
                    option['delta'] = 0.5
                elif moneyness > 0.9:  # OTM call
                    option['delta'] = 0.3
                else:  # Deep OTM call
                    option['delta'] = 0.1
            else:
                if moneyness < 0.9:  # Deep ITM put
                    option['delta'] = -0.9
                elif moneyness < 0.95:  # ITM put
                    option['delta'] = -0.7
                elif moneyness < 1.05:  # ATM put
                    option['delta'] = -0.5
                elif moneyness < 1.1:  # OTM put
                    option['delta'] = -0.3
                else:  # Deep OTM put
                    option['delta'] = -0.1

            # Simple estimates for other Greeks
            option['gamma'] = 0.05 if 0.95 < moneyness < 1.05 else 0.02
            option['theta'] = -0.02 * option['price'] * (1 / max(days, 1))
            option['vega'] = 0.1 * option['price'] * (min(days, 30) / 30)

        except Exception as e:
            logger.warning(f"Error estimating Greeks for {option['symbol']}: {e}")
            # Set default values
            option['delta'] = 0.5 if option['option_type'] == 'call' else -0.5
            option['gamma'] = 0.03
            option['theta'] = -0.01
            option['vega'] = 0.1

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a symbol

        Args:
            symbol: The ticker symbol

        Returns:
            Current price or 0 if not available
        """
        # Check cache first
        cache_key = f"{symbol}_price"
        if cache_key in self.memory_cache:
            cached_data = self.memory_cache[cache_key]
            if cached_data.is_valid(self.price_ttl):
                return cached_data.data

        price = 0.0

        # Try Alpaca first for real-time quotes
        if self.alpaca_available:
            try:
                self.alpaca_rate_limit.wait_if_needed()
                request = StockQuotesRequest(symbol_or_symbols=symbol)
                quotes = self.alpaca_data.get_stock_latest_quote(request)

                if symbol in quotes:
                    price = float(quotes[symbol].ask_price)
                    self.memory_cache[cache_key] = CachedData(price)
                    return price
            except Exception as e:
                logger.warning(f"Error getting Alpaca quote: {e}")

        # Fall back to Polygon
        if self.polygon_available and price <= 0:
            try:
                self.polygon_rate_limit.wait_if_needed()
                url = f"https://api.polygon.io/v2/last/trade/{symbol}?apiKey={self.polygon_key}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if 'results' in data:
                        price = float(data['results']['p'])
                        self.memory_cache[cache_key] = CachedData(price)
                        return price
            except Exception as e:
                logger.warning(f"Error getting Polygon price: {e}")

        # If all else fails, try to get the price from cached options
        if price <= 0:
            price = self._get_price_from_cached_options(symbol)

        return price

    def _get_price_from_cached_options(self, symbol: str) -> float:
        """Estimate price from cached options if available"""
        cached_data = self._get_cached_options(symbol)
        if cached_data:
            # Find ATM options and estimate price
            calls = [opt for opt in cached_data if opt['option_type'] == 'call']
            if calls:
                # Sort by strike
                calls.sort(key=lambda x: x['strike'])

                # Find the middle
                mid_index = len(calls) // 2
                if mid_index < len(calls):
                    return calls[mid_index]['strike']

        return 0.0

    def _get_cached_options(self, symbol: str) -> List[Dict[str, Any]]:
        """Get cached options data if available and not expired"""
        # Check memory cache first
        cache_key = f"{symbol}_options"
        if cache_key in self.memory_cache:
            cached_data = self.memory_cache[cache_key]
            if cached_data.is_valid(self.option_chain_ttl):
                return cached_data.data

        # Check disk cache next
        try:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_options.json")
            if os.path.exists(cache_file):
                # Check if cache is fresh (< ttl)
                file_modified = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if (datetime.now() - file_modified).total_seconds() < self.option_chain_ttl:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        # Update memory cache
                        self.memory_cache[cache_key] = CachedData(data)
                        return data
        except Exception as e:
            logger.error(f"Error reading options cache: {e}")

        # No valid cache found
        return []

    def _cache_options(self, symbol: str, options: List[Dict[str, Any]]):
        """Cache options data to memory and disk"""
        if not options:
            return

        # Update memory cache
        cache_key = f"{symbol}_options"
        self.memory_cache[cache_key] = CachedData(options)

        # Update disk cache
        try:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_options.json")
            with open(cache_file, 'w') as f:
                json.dump(options, f)
        except Exception as e:
            logger.error(f"Error caching options data: {e}")

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cache for a symbol or all symbols

        Args:
            symbol: Symbol to clear cache for, or None for all
        """
        if symbol:
            # Clear specific symbol
            price_key = f"{symbol}_price"
            options_key = f"{symbol}_options"

            if price_key in self.memory_cache:
                del self.memory_cache[price_key]

            if options_key in self.memory_cache:
                del self.memory_cache[options_key]

            # Clear disk cache
            cache_file = os.path.join(self.cache_dir, f"{symbol}_options.json")
            if os.path.exists(cache_file):
                os.remove(cache_file)
        else:
            # Clear all cache
            self.memory_cache = {}

            # Clear all disk cache
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
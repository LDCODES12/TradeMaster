"""
Core options trading strategy implementation with real API integrations.
Analyzes market data and finds trading opportunities.
"""

import logging
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Alpaca APIs
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame

# Initialize logger
logger = logging.getLogger(__name__)


class PrecisionOptionsArbitrage:
    """
    Core strategy for finding and executing options arbitrage opportunities
    using real market data and events
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy with API connections"""
        # API credentials
        self.config = config
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')

        # Initialize API clients
        if not self.api_key or not self.api_secret:
            logger.error("API credentials not provided. Strategy cannot run in live mode.")
            raise ValueError("Missing API credentials")

        # Initialize Alpaca clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)

        # Finnhub for events calendar (if available)
        self.finnhub_key = config.get('finnhub', {}).get('api_key', '')

        # AlphaVantage as fallback for fundamentals
        self.alpha_key = config.get('alphavantage', {}).get('api_key', '')

        # Polygon for additional market data (if needed)
        self.polygon_key = config.get('polygon', {}).get('api_key', '')

        # Strategy state
        self.positions = []
        self.buying_power = 0.0
        self.account = None
        self.price_cache = {}  # Cache for frequently accessed prices

        # Load account info
        self._load_account()

        # Load events calendar
        self.events_calendar = self._load_events_calendar()

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

    def _load_events_calendar(self) -> List[Dict[str, Any]]:
        """Load real market events calendar from available sources"""
        events = []

        # First try Finnhub for earnings calendar (best source)
        if self.finnhub_key:
            try:
                today = datetime.now().strftime('%Y-%m-%d')
                future = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')

                url = f"https://finnhub.io/api/v1/calendar/earnings?from={today}&to={future}&token={self.finnhub_key}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    for earning in data.get('earningsCalendar', []):
                        events.append({
                            'symbol': earning['symbol'],
                            'event_type': 'earnings',
                            'event_date': earning['date'],
                            'importance': 'high' if float(earning.get('epsEstimate', 0) or 0) > 0 else 'medium'
                        })
                    logger.info(f"Loaded {len(events)} earnings events from Finnhub")
            except Exception as e:
                logger.warning(f"Error fetching earnings from Finnhub: {e}")

        # Fallback to Alpha Vantage for earnings
        if not events and self.alpha_key:
            try:
                url = f"https://www.alphavantage.co/query?function=EARNINGS_CALENDAR&horizon=3month&apikey={self.alpha_key}"
                response = requests.get(url, timeout=10)

                if response.status_code == 200:
                    # Alpha Vantage returns CSV for this endpoint
                    lines = response.text.strip().split('\n')
                    if len(lines) > 1:
                        headers = lines[0].split(',')

                        for line in lines[1:]:
                            fields = line.split(',')
                            if len(fields) >= len(headers):
                                event_data = dict(zip(headers, fields))

                                events.append({
                                    'symbol': event_data.get('symbol', ''),
                                    'event_type': 'earnings',
                                    'event_date': event_data.get('reportDate', ''),
                                    'importance': 'high'
                                })
                    logger.info(f"Loaded {len(events)} earnings events from Alpha Vantage")
            except Exception as e:
                logger.warning(f"Error fetching earnings from Alpha Vantage: {e}")

        # Check if we need to use any available Alpaca corporate actions
        try:
            # This will use the Alpaca Corporate Actions API when it's available
            # For now, we could use Alpaca news API to identify events
            pass
        except Exception as e:
            logger.warning(f"Error fetching corporate actions from Alpaca: {e}")

        if not events:
            logger.warning("No events found from any source - using fallback watchlist")
            # Fallback to a watchlist of major stocks
            watchlist = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'AMD']
            events = self._create_watchlist_events(watchlist)

        return events

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
        """Get real options chain data from Alpaca"""
        options = []

        try:
            # Alpaca options API format - convert to our internal format
            # First get stock price for strike selection
            current_price = self._get_current_price(symbol)

            # Get available expirations
            url = f"https://data.alpaca.markets/v2/stocks/{symbol}/options/expirations"
            headers = {
                "Apca-Api-Key-Id": self.api_key,
                "Apca-Api-Secret-Key": self.api_secret
            }

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                logger.error(f"Failed to get option expirations: {response.text}")
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

            # For each expiration, get near-the-money options
            for expiration in valid_expirations:
                exp_date = datetime.strptime(expiration, '%Y-%m-%d').date()
                days_to_expiry = (exp_date - today).days

                # Calculate strike price range (±15% of current price)
                min_strike = current_price * 0.85
                max_strike = current_price * 1.15

                # Get option chain for this expiration
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

                # Process each option contract
                for contract in chain_data:
                    # Get quote data for the option
                    option_symbol = contract.get('symbol')

                    quote_url = f"https://data.alpaca.markets/v2/stocks/{option_symbol}/quotes/latest"
                    quote_resp = requests.get(quote_url, headers=headers, timeout=10)

                    if quote_resp.status_code != 200:
                        # Skip if can't get quote
                        continue

                    quote = quote_resp.json().get('quote', {})

                    # Create our internal option object
                    option = {
                        'symbol': option_symbol,
                        'underlying': symbol,
                        'expiration': expiration,
                        'strike': float(contract.get('strike_price', 0)),
                        'option_type': contract.get('side', '').lower(),  # 'call' or 'put'
                        'bid': float(quote.get('bid_price', 0)),
                        'ask': float(quote.get('ask_price', 0)),
                        'price': float(quote.get('ask_price', 0)),  # Use ask for conservative pricing
                        'iv': float(contract.get('implied_volatility', 0.5)),
                        'days_to_expiry': days_to_expiry,
                        'volume': int(quote.get('volume', 0)),
                        'open_interest': int(contract.get('open_interest', 0))
                    }

                    options.append(option)

            logger.info(f"Loaded {len(options)} option contracts for {symbol}")

        except Exception as e:
            logger.error(f"Error getting options data: {e}")

        return options

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

    def analyze_opportunities(self, min_sharpe: float = 0.25) -> List[Dict[str, Any]]:
        """
        Analyze options for trading opportunities based on events
        """
        opportunities = []

        # Use events as catalysts for option trades
        for event in self.events_calendar:
            symbol = event['symbol']
            event_type = event['event_type']
            event_date = event['event_date']

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

            # Filter options based on days to expiry
            valid_options = [
                opt for opt in options_chain
                if datetime.strptime(opt['expiration'], '%Y-%m-%d') > datetime.strptime(event_date, '%Y-%m-%d')
                   and opt['days_to_expiry'] < 60  # Not too far out
            ]

            # Find the best opportunities based on event type
            if event_type == 'earnings':
                # For earnings, look for straddles or slightly OTM options
                for option in valid_options:
                    # Calculate moneyness
                    current_price = self._get_current_price(symbol)
                    moneyness = option['strike'] / current_price

                    # Filter based on moneyness (near the money)
                    if 0.9 <= moneyness <= 1.1:
                        # Calculate metrics
                        edge = self._calculate_edge(option, event)
                        expected_roi, profit_prob = self._calculate_expected_return(option, edge)
                        sharpe_ratio = expected_roi / (option['iv'] * np.sqrt(option['days_to_expiry'] / 365))

                        if sharpe_ratio >= min_sharpe:
                            opportunities.append({
                                'option': option,
                                'event': event,
                                'edge': edge,
                                'expected_roi': expected_roi,
                                'profit_probability': profit_prob,
                                'sharpe_ratio': sharpe_ratio,
                                'max_contracts': int(10000 / (option['price'] * 100))
                            })

            elif event_type in ['product_launch', 'watchlist']:
                # For other events, prefer calls as they tend to run up into events
                for option in valid_options:
                    if option['option_type'] == 'call':
                        edge = self._calculate_edge(option, event)
                        expected_roi, profit_prob = self._calculate_expected_return(option, edge)
                        sharpe_ratio = expected_roi / (option['iv'] * np.sqrt(option['days_to_expiry'] / 365))

                        if sharpe_ratio >= min_sharpe:
                            opportunities.append({
                                'option': option,
                                'event': event,
                                'edge': edge,
                                'expected_roi': expected_roi,
                                'profit_probability': profit_prob,
                                'sharpe_ratio': sharpe_ratio,
                                'max_contracts': int(10000 / (option['price'] * 100))
                            })

        # Sort opportunities by Sharpe ratio
        opportunities.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
        logger.info(f"Found {len(opportunities)} trading opportunities")
        return opportunities

    def _calculate_edge(self, option: Dict[str, Any], event: Dict[str, Any]) -> float:
        """
        Calculate the edge (advantage) for an option trade based on event
        Returns a value between 0-1 representing the estimated edge
        """
        days_to_expiry = option['days_to_expiry']
        option_type = option['option_type']
        iv = option['iv']
        event_type = event['event_type']
        importance = event.get('importance', 'medium')

        # Base edge calculation
        edge = 0.0

        # Adjust edge based on event type
        if event_type == 'earnings':
            edge += 0.2  # Earnings events often cause volatility

            # Time decay is more predictable closer to expiration
            if days_to_expiry < 14:
                edge += 0.1

            # For earnings, both calls and puts can be valuable (straddle/strangle strategy)
            edge += 0.05

        elif event_type == 'product_launch' or event_type == 'watchlist':
            edge += 0.15  # Product launches often have run-ups

            # Calls tend to perform better for product launches
            if option_type == 'call':
                edge += 0.1

        # Adjust edge based on importance
        if importance == 'high':
            edge += 0.15
        elif importance == 'medium':
            edge += 0.1

        # Adjust edge based on IV - lower IV may indicate underpriced options
        if iv < 0.4:
            edge += 0.1

        # Normalize edge to 0-1
        edge = min(max(edge, 0.0), 1.0)

        return edge

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

    def execute_trade(self, opportunity: Dict[str, Any], contracts: int = 1) -> bool:
        """
        Execute a trade based on the opportunity
        """
        option = opportunity['option']
        total_cost = option['price'] * contracts * 100

        if total_cost > self.buying_power:
            logger.warning(f"Insufficient buying power (${self.buying_power}) for trade: ${total_cost}")
            return False

        try:
            # Place real order with Alpaca
            order = MarketOrderRequest(
                symbol=option['symbol'],
                qty=contracts,
                side=OrderSide.BUY,
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
                    break
                elif order_status.status in [OrderStatus.REJECTED, OrderStatus.CANCELED]:
                    logger.error(f"Order {order_id} {order_status.status}: {order_status.reject_reason}")
                    return False

            if not filled:
                logger.error(f"Order {order_id} not filled after {max_attempts * 2} seconds")
                return False

            logger.info(f"Order {order_id} filled: {contracts} contracts of {option['symbol']}")

            # Update account
            self.buying_power -= total_cost

            # Add to positions
            position = {
                'symbol': option['symbol'],
                'underlying': option['underlying'],
                'type': option['option_type'],
                'strike': option['strike'],
                'expiration': option['expiration'],
                'days_to_expiry': option['days_to_expiry'],
                'entry_price': option['price'],
                'contracts': contracts,
                'value': total_cost,
                'entry_time': datetime.now().isoformat(),
                'event': opportunity['event'],
                'expected_roi': opportunity['expected_roi'],
                'profit_probability': opportunity['profit_probability'],
                'sharpe_ratio': opportunity['sharpe_ratio'],
                'order_id': order_id,
                'status': 'open'
            }

            self.positions.append(position)
            logger.info(f"Added position: {option['symbol']} - {contracts} contracts at ${option['price']} each")

            return True

        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False

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
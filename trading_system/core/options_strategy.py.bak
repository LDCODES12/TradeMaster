"""
Core options trading strategy implementation.
Analyzes market data and finds trading opportunities.
"""

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus

logger = logging.getLogger(__name__)


class PrecisionOptionsArbitrage:
    """
    Core strategy for finding and executing options arbitrage opportunities
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy"""
        # Configure API access
        self.api_key = config.get('api_key', '')
        self.api_secret = config.get('api_secret', '')

        if not self.api_key or not self.api_secret:
            logger.warning("API credentials not provided. Strategy will run in simulation mode.")
            self.simulation_mode = True
        else:
            self.simulation_mode = False
            # Connect to Alpaca API
            self.client = TradingClient(self.api_key, self.api_secret, paper=True)

        # Strategy state
        self.positions = []  # Current positions
        self.buying_power = 25000.0  # Default simulation value
        self.account = None

        # Load account info
        self._load_account()

        # Event calendar for catalysts
        self.events_calendar = self._load_events_calendar()

        logger.info("Precision Options Arbitrage strategy initialized")

    def _load_account(self):
        """Load account information"""
        if not self.simulation_mode:
            try:
                self.account = self.client.get_account()
                self.buying_power = float(self.account.buying_power)
                logger.info(
                    f"Account loaded: ${self.account.portfolio_value} portfolio value, ${self.buying_power} buying power")
            except Exception as e:
                logger.error(f"Error loading account: {e}")
                self.simulation_mode = True
                self.account = {'portfolio_value': 25000.0, 'buying_power': 25000.0, 'cash': 25000.0}
        else:
            # Simulation account
            self.account = {'portfolio_value': 25000.0, 'buying_power': 25000.0, 'cash': 25000.0}
            self.buying_power = 25000.0

    def _load_events_calendar(self) -> List[Dict[str, Any]]:
        """Load market events calendar for catalyst-based trading"""
        # In a real system, this would fetch from an API
        # For simulation, we'll create a dummy calendar

        today = datetime.now()

        # Create a list of simulated events
        events = [
            {
                'symbol': 'AAPL',
                'event_type': 'earnings',
                'event_date': (today + timedelta(days=5)).strftime('%Y-%m-%d'),
                'importance': 'high'
            },
            {
                'symbol': 'MSFT',
                'event_type': 'earnings',
                'event_date': (today + timedelta(days=10)).strftime('%Y-%m-%d'),
                'importance': 'high'
            },
            {
                'symbol': 'AMZN',
                'event_type': 'earnings',
                'event_date': (today + timedelta(days=15)).strftime('%Y-%m-%d'),
                'importance': 'high'
            },
            {
                'symbol': 'GOOGL',
                'event_type': 'product_launch',
                'event_date': (today + timedelta(days=7)).strftime('%Y-%m-%d'),
                'importance': 'medium'
            },
            {
                'symbol': 'TSLA',
                'event_type': 'product_launch',
                'event_date': (today + timedelta(days=20)).strftime('%Y-%m-%d'),
                'importance': 'high'
            },
            {
                'symbol': 'AMD',
                'event_type': 'earnings',
                'event_date': (today + timedelta(days=12)).strftime('%Y-%m-%d'),
                'importance': 'medium'
            },
            {
                'symbol': 'NVDA',
                'event_type': 'earnings',
                'event_date': (today + timedelta(days=18)).strftime('%Y-%m-%d'),
                'importance': 'high'
            }
        ]

        return events

    def _get_options_chain(self, symbol: str) -> List[Dict[str, Any]]:
        """Get options chain for a symbol"""
        # In a real system, this would fetch from the broker API
        # For simulation, we'll create a synthetic options chain

        # Current date and basic stock price simulation
        now = datetime.now()
        base_price = hash(symbol + now.strftime('%Y-%m-%d')) % 300 + 50  # Random price between $50-$350

        expiration_days = [7, 14, 30, 60, 90]
        strikes_pct = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]

        # Create options chain
        options = []

        for days in expiration_days:
            expiration_date = (now + timedelta(days=days)).strftime('%Y-%m-%d')

            for strike_pct in strikes_pct:
                strike_price = round(base_price * strike_pct, 1)

                # Add call option
                call_iv = 0.3 + (np.random.random() * 0.4)  # IV between 30% and 70%
                days_to_expiry = days
                call_price = self._calculate_option_price('call', base_price, strike_price, days_to_expiry / 365,
                                                          call_iv, 0.02)

                call = {
                    'symbol': f"{symbol}_{expiration_date}_{strike_price}_C",
                    'underlying': symbol,
                    'expiration': expiration_date,
                    'strike': strike_price,
                    'option_type': 'call',
                    'bid': round(call_price * 0.95, 2),
                    'ask': round(call_price * 1.05, 2),
                    'price': round(call_price, 2),
                    'iv': call_iv,
                    'days_to_expiry': days_to_expiry,
                    'volume': int(np.random.random() * 1000) + 100,
                    'open_interest': int(np.random.random() * 5000) + 500
                }
                options.append(call)

                # Add put option
                put_iv = 0.35 + (np.random.random() * 0.4)  # IV between 35% and 75%
                put_price = self._calculate_option_price('put', base_price, strike_price, days_to_expiry / 365, put_iv,
                                                         0.02)

                put = {
                    'symbol': f"{symbol}_{expiration_date}_{strike_price}_P",
                    'underlying': symbol,
                    'expiration': expiration_date,
                    'strike': strike_price,
                    'option_type': 'put',
                    'bid': round(put_price * 0.95, 2),
                    'ask': round(put_price * 1.05, 2),
                    'price': round(put_price, 2),
                    'iv': put_iv,
                    'days_to_expiry': days_to_expiry,
                    'volume': int(np.random.random() * 1000) + 100,
                    'open_interest': int(np.random.random() * 5000) + 500
                }
                options.append(put)

        return options

    def _calculate_option_price(self, option_type: str, spot: float, strike: float, time_to_expiry: float,
                                volatility: float, risk_free_rate: float) -> float:
        """
        Calculate option price using Black-Scholes model
        Very simplified for demonstration purposes
        """
        # Simplified option pricing - in a real system use proper BS model
        moneyness = spot / strike
        time_factor = np.sqrt(time_to_expiry)
        vol_factor = volatility * time_factor

        if option_type == 'call':
            # Very rough approximation
            if moneyness > 1.1:  # Deep ITM
                return max(0.01, spot - strike + (vol_factor * spot * 0.1))
            elif moneyness < 0.9:  # Deep OTM
                return max(0.01, vol_factor * spot * 0.2)
            else:  # Near the money
                return max(0.01, vol_factor * spot * 0.3 + max(0, spot - strike))
        else:  # Put
            if moneyness < 0.9:  # Deep ITM for puts
                return max(0.01, strike - spot + (vol_factor * spot * 0.1))
            elif moneyness > 1.1:  # Deep OTM for puts
                return max(0.01, vol_factor * spot * 0.2)
            else:  # Near the money
                return max(0.01, vol_factor * spot * 0.3 + max(0, strike - spot))

    def analyze_opportunities(self, min_sharpe: float = 0.25) -> List[Dict[str, Any]]:
        """
        Analyze options for trading opportunities based on events

        Args:
            min_sharpe: Minimum Sharpe ratio for considering an opportunity

        Returns:
            List of trading opportunities
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

            # Get options chain for this symbol
            options_chain = self._get_options_chain(symbol)

            # Filter options based on days to expiry
            # For earnings/events we want options that expire after the event
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
                    moneyness = option['strike'] / self._get_current_price(symbol)

                    # Filter based on moneyness (near the money)
                    if 0.9 <= moneyness <= 1.1:
                        edge = self._calculate_edge(option, event)

                        if edge > 0:
                            # Calculate expected ROI and probability of profit
                            expected_roi, profit_prob = self._calculate_expected_return(option, edge)

                            # Calculate Sharpe ratio (simplified)
                            sharpe_ratio = expected_roi / (option['iv'] * np.sqrt(option['days_to_expiry'] / 365))

                            if sharpe_ratio >= min_sharpe:
                                opportunities.append({
                                    'option': option,
                                    'event': event,
                                    'edge': edge,
                                    'expected_roi': expected_roi,
                                    'profit_probability': profit_prob,
                                    'sharpe_ratio': sharpe_ratio,
                                    'max_contracts': int(10000 / (option['price'] * 100))  # Max position sizing
                                })

            elif event_type == 'product_launch':
                # For product launches, prefer calls as they tend to run up into events
                for option in valid_options:
                    if option['option_type'] == 'call':
                        edge = self._calculate_edge(option, event)

                        if edge > 0:
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

        return opportunities

    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        # In a real system, fetch from the broker API
        # For simulation, create a realistic price
        return hash(symbol + datetime.now().strftime('%Y-%m-%d')) % 300 + 50

    def _calculate_edge(self, option: Dict[str, Any], event: Dict[str, Any]) -> float:
        """
        Calculate the edge (advantage) for an option trade based on event
        Returns a value between 0-1 representing the estimated edge
        """
        # Extract option properties
        days_to_expiry = option['days_to_expiry']
        option_type = option['option_type']
        iv = option['iv']

        # Extract event properties
        event_type = event['event_type']
        importance = event['importance']

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

        elif event_type == 'product_launch':
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

        Args:
            opportunity: Trading opportunity
            contracts: Number of contracts to trade

        Returns:
            True if trade successful, False otherwise
        """
        option = opportunity['option']

        total_cost = option['price'] * contracts * 100

        if total_cost > self.buying_power:
            logger.warning(f"Insufficient buying power (${self.buying_power}) for trade: ${total_cost}")
            return False

        try:
            if not self.simulation_mode:
                # In a real system, this would place an order with the broker
                order = MarketOrderRequest(
                    symbol=option['symbol'],
                    qty=contracts,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )

                order_result = self.client.submit_order(order)
                order_id = order_result.id

                # Wait for fill
                time.sleep(2)

                # Get order status
                order_status = self.client.get_order_by_id(order_id).status

                if order_status != OrderStatus.FILLED:
                    logger.error(f"Order {order_id} not filled: {order_status}")
                    return False

                logger.info(f"Order {order_id} filled: {contracts} contracts of {option['symbol']}")
            else:
                # Simulation mode
                order_id = f"sim-{int(time.time())}-{hash(option['symbol'])}"
                logger.info(f"Simulation order {order_id} for {contracts} contracts of {option['symbol']}")

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
        """Get current open positions"""
        return [p for p in self.positions if p.get('status', 'open') == 'open']

    def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        if not self.simulation_mode:
            try:
                account = self.client.get_account()
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
        else:
            # In simulation mode, calculate portfolio value
            position_value = sum(p['value'] for p in self.positions if p.get('status', 'open') == 'open')
            portfolio_value = self.buying_power + position_value
            return {
                'portfolio_value': portfolio_value,
                'buying_power': self.buying_power,
                'cash': self.buying_power
            }

    def monitor_positions(self) -> List[Dict[str, Any]]:
        """
        Monitor existing positions and execute exit strategies

        Returns:
            List of closed positions
        """
        if not self.positions:
            return []

        closed_positions = []

        for position in self.positions:
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
        Check if we should exit a position

        Returns:
            Tuple of (exit_signal, exit_price, exit_reason)
        """
        # Get current price (simulated)
        current_price = self._simulate_current_price(position)

        # Calculate days held
        entry_time = datetime.fromisoformat(position['entry_time'])
        days_held = (datetime.now() - entry_time).total_seconds() / 86400

        # Calculate P&L
        entry_cost = position['entry_price'] * position['contracts'] * 100
        current_value = current_price * position['contracts'] * 100
        unrealized_pnl_pct = (current_value / entry_cost - 1) * 100

        # Check expiration approach
        days_to_expiry = position['days_to_expiry'] - days_held
        if days_to_expiry <= 1:
            return True, current_price, "approaching_expiration"

        # Check profit target (50% of expected ROI)
        profit_target = position['expected_roi'] * 0.5
        if unrealized_pnl_pct >= profit_target:
            return True, current_price, "profit_target_reached"

        # Check stop loss (-100% of expected ROI)
        stop_loss = -position['expected_roi']
        if unrealized_pnl_pct <= stop_loss:
            return True, current_price, "stop_loss_triggered"

        # Time-based exit (held more than half time to expiry)
        if days_held > position['days_to_expiry'] * 0.5:
            # Exit if we're profitable
            if unrealized_pnl_pct > 0:
                return True, current_price, "time_exit_profitable"

            # Also exit if we've lost more than 30%
            if unrealized_pnl_pct < -30:
                return True, current_price, "time_exit_with_loss"

        # Event passed exit
        event_date = datetime.strptime(position['event']['event_date'], '%Y-%m-%d')
        if datetime.now() > event_date + timedelta(days=1):
            return True, current_price, "event_passed"

        # No exit signal
        return False, current_price, ""

    def _simulate_current_price(self, position: Dict[str, Any]) -> float:
        """Simulate current price for a position"""
        # Calculate how much time has passed as a fraction of days to expiry
        entry_time = datetime.fromisoformat(position['entry_time'])
        days_held = (datetime.now() - entry_time).total_seconds() / 86400
        time_fraction = days_held / position['days_to_expiry']

        # Check if event date has passed
        event_date = datetime.strptime(position['event']['event_date'], '%Y-%m-%d')
        event_passed = datetime.now() > event_date

        # Base price movement on random walk with drift based on expected ROI
        expected_return = position['expected_roi'] / 100  # Convert from percentage

        # Calculate drift component
        if event_passed:
            # After event, trend toward expected outcome
            drift = expected_return * (0.8 + np.random.random() * 0.4)
        else:
            # Before event, trend is more modest
            drift = expected_return * (0.3 + np.random.random() * 0.3) * time_fraction

        # Add random component
        random_component = np.random.normal(0, 0.2 * position['sharpe_ratio'])

        # Combine components
        price_change_pct = drift + random_component

        # Apply to entry price
        current_price = position['entry_price'] * (1 + price_change_pct)

        # Ensure price is never negative
        current_price = max(0.01, current_price)

        return round(current_price, 2)

    def _exit_position(self, position: Dict[str, Any], exit_price: float, exit_reason: str) -> bool:
        """
        Exit a position

        Returns:
            True if successful, False otherwise
        """
        if not self.simulation_mode:
            try:
                # In a real system, place a sell order with the broker
                order = MarketOrderRequest(
                    symbol=position['symbol'],
                    qty=position['contracts'],
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )

                order_result = self.client.submit_order(order)

                # Wait for fill
                time.sleep(2)

                # Check order status
                order_status = self.client.get_order_by_id(order_result.id).status

                if order_status != OrderStatus.FILLED:
                    logger.error(f"Exit order not filled: {order_status}")
                    return False

                logger.info(f"Exit order filled: {position['contracts']} contracts of {position['symbol']}")
                return True

            except Exception as e:
                logger.error(f"Error exiting position: {e}")
                return False
        else:
            # Simulation mode
            logger.info(f"Simulation exit: {position['contracts']} contracts of {position['symbol']} at ${exit_price}")
            return True
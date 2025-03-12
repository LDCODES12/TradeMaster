"""
Risk management system for controlling trading risk.
Implements position sizing, exposure limits, and circuit breakers.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Advanced risk management system that monitors and controls trading risk
    """

    def __init__(self, db_manager, config: Dict[str, Any]):
        """Initialize the risk manager"""
        self.db = db_manager
        self.config = config
        self.portfolio_value = 0.0

        # Configure risk limits
        self.max_position_size = float(config.get('max_position_size', 0.05))  # Max 5% of portfolio per position
        self.max_strategy_allocation = float(config.get('max_strategy_allocation', 0.20))  # Max 20% per strategy
        self.max_sector_allocation = float(config.get('max_sector_allocation', 0.30))  # Max 30% per sector
        self.max_daily_drawdown = float(config.get('max_daily_drawdown', 0.03))  # 3% max daily loss
        self.max_total_drawdown = float(config.get('max_total_drawdown', 0.10))  # 10% max total drawdown
        self.circuit_breakers_enabled = config.get('circuit_breakers_enabled', True) in [True, 'true', 'True', '1', 1]

        # State tracking
        self.historical_drawdown = 0.0
        self.current_var_95 = 0.02  # Default 2% VaR
        self.current_var_99 = 0.03  # Default 3% VaR

        # Initialize with baseline metrics
        self._update_portfolio_metrics()

    def _update_portfolio_metrics(self):
        """Update portfolio metrics from historical data"""
        try:
            # Get the most recent portfolio snapshot
            snapshots = self.db.get_portfolio_snapshots(days_back=1)
            if snapshots:
                latest = snapshots[-1]
                self.portfolio_value = latest['account_value']

                # Calculate rolling metrics if we have enough history
                all_snapshots = self.db.get_portfolio_snapshots(days_back=30)
                if len(all_snapshots) > 5:
                    values = [s['account_value'] for s in all_snapshots]

                    # Calculate historical drawdown
                    peak = max(values)
                    current = values[-1]
                    self.historical_drawdown = (peak - current) / peak if peak > 0 else 0

                    # Calculate daily returns for VaR
                    daily_returns = []
                    for i in range(1, len(values)):
                        daily_return = (values[i] - values[i - 1]) / values[i - 1]
                        daily_returns.append(daily_return)

                    # Calculate 95% and 99% VaR
                    if daily_returns:
                        daily_returns = sorted(daily_returns)
                        var_95_idx = int(len(daily_returns) * 0.05)
                        var_99_idx = int(len(daily_returns) * 0.01)
                        self.current_var_95 = abs(daily_returns[var_95_idx]) if var_95_idx < len(
                            daily_returns) else 0.02
                        self.current_var_99 = abs(daily_returns[var_99_idx]) if var_99_idx < len(
                            daily_returns) else 0.03
            else:
                # If no snapshots, initialize with a default value
                self.portfolio_value = 10000.0  # Default starting value
        except Exception as e:
            logger.error(f"Error updating portfolio metrics: {e}")
            # Use conservative default values
            self.portfolio_value = 10000.0
            self.current_var_95 = 0.02  # 2% daily VaR
            self.current_var_99 = 0.03  # 3% daily VaR
            self.historical_drawdown = 0.0

    def calculate_position_size(self, opportunity: Dict[str, Any]) -> int:
        """
        Calculate the optimal position size based on Kelly criterion and risk limits

        Args:
            opportunity: Trading opportunity

        Returns:
            Number of contracts to trade
        """
        option = opportunity['option']

        # Get Kelly fraction (probability of profit - probability of loss)
        prob_profit = opportunity.get('profit_probability', 0.5)
        kelly_fraction = 2 * prob_profit - 1  # Simple Kelly formula

        # Apply a cap to the Kelly fraction to be more conservative (quarter-Kelly)
        kelly_fraction = min(max(0.0, kelly_fraction), 0.25)

        # Calculate max position size based on portfolio value
        max_risk_amount = self.portfolio_value * self.max_position_size
        option_cost = option['price'] * 100  # Cost per contract

        # Calculate max contracts based on maximum risk allocation
        max_contracts_by_risk = int(max_risk_amount / option_cost) if option_cost > 0 else 0

        # Calculate Kelly-optimal contracts
        target_position = kelly_fraction * self.portfolio_value
        max_contracts_by_kelly = int(target_position / option_cost) if option_cost > 0 else 0

        # Take the more conservative of the two
        optimal_contracts = min(max_contracts_by_risk, max_contracts_by_kelly)

        # Log the decision
        logger.info(
            f"Position sizing for {option['symbol']}: "
            f"Kelly: {max_contracts_by_kelly}, Risk Limit: {max_contracts_by_risk}, "
            f"Selected: {optimal_contracts}"
        )

        return max(1, optimal_contracts)  # At least 1 contract if we decide to trade

    def check_trade_approval(self, opportunity: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a trade should be approved based on risk management rules

        Args:
            opportunity: Trading opportunity

        Returns:
            Tuple of (approved, reason)
        """
        # Get open positions
        open_positions = self.db.get_open_positions()

        # Check total position count limit
        max_positions = int(self.config.get('max_positions', 10))
        if len(open_positions) >= max_positions:
            return False, "Maximum positions limit reached"

        # Check strategy allocation limit
        event_type = opportunity['event']['event_type']
        strategy_positions = [p for p in open_positions if p['event_type'] == event_type]
        strategy_allocation = sum(p['entry_price'] * p['quantity'] * 100 for p in strategy_positions)

        new_position_cost = opportunity['option']['price'] * 100
        if strategy_allocation + new_position_cost > (self.portfolio_value * self.max_strategy_allocation):
            return False, f"Maximum allocation for strategy {event_type} reached"

        # Check symbol concentration (avoid too many positions in same underlying)
        symbol = opportunity['event']['symbol']
        symbol_positions = [p for p in open_positions if p['underlying'] == symbol]

        max_positions_per_symbol = int(self.config.get('max_positions_per_symbol', 3))
        if len(symbol_positions) >= max_positions_per_symbol:
            return False, f"Maximum positions for {symbol} reached"

        # Check VaR impact
        expected_var = new_position_cost * 0.5  # Assume 50% worst-case loss for options
        if expected_var > (self.portfolio_value * self.current_var_95 * 0.2):
            return False, "Trade exceeds risk budget based on Value-at-Risk"

        # Check Sharpe ratio minimum
        min_sharpe = float(self.config.get('min_sharpe', 0.2))
        if opportunity.get('sharpe_ratio', 0) < min_sharpe:
            return False, f"Sharpe ratio too low: {opportunity.get('sharpe_ratio', 0):.2f}"

        # All checks passed
        return True, "Trade approved"

    def check_circuit_breakers(self) -> Tuple[bool, str]:
        """
        Check if any circuit breakers have been triggered

        Returns:
            Tuple of (trading_allowed, reason)
        """
        if not self.circuit_breakers_enabled:
            return True, "Circuit breakers disabled"

        # Check daily drawdown circuit breaker
        snapshots = self.db.get_portfolio_snapshots(days_back=1)
        if len(snapshots) >= 2:
            start_value = snapshots[0]['account_value']
            current_value = snapshots[-1]['account_value']
            daily_drawdown = (start_value - current_value) / start_value if start_value > 0 else 0

            if daily_drawdown > self.max_daily_drawdown:
                logger.warning(f"Daily drawdown circuit breaker triggered: {daily_drawdown:.2%}")
                return False, f"Daily drawdown of {daily_drawdown:.2%} exceeded {self.max_daily_drawdown:.2%} limit"

        # Check total drawdown circuit breaker
        if self.historical_drawdown > self.max_total_drawdown:
            logger.warning(f"Total drawdown circuit breaker triggered: {self.historical_drawdown:.2%}")
            return False, f"Total drawdown of {self.historical_drawdown:.2%} exceeded {self.max_total_drawdown:.2%} limit"

        # All circuit breakers are clear
        return True, "Circuit breakers clear"

    def calculate_risk_metrics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics for the portfolio

        Returns:
            Dictionary of risk metrics
        """
        # Get historical data
        portfolio_snapshots = self.db.get_portfolio_snapshots(days_back=60)
        closed_trades = self.db.get_closed_positions(days_back=60)

        # Default values if not enough data
        risk_metrics = {
            'var_95': self.current_var_95,
            'var_99': self.current_var_99,
            'max_drawdown': self.historical_drawdown,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'beta': 1.0,
            'correlation_spy': 0.0,
            'avg_win_loss_ratio': 0.0,
            'win_rate': 0.0
        }

        # Calculate advanced metrics if we have enough data
        if len(portfolio_snapshots) > 10:
            # Extract daily values and calculate returns
            daily_values = [snapshot['account_value'] for snapshot in portfolio_snapshots]
            daily_returns = [(daily_values[i] - daily_values[i - 1]) / daily_values[i - 1] for i in
                             range(1, len(daily_values))]

            if daily_returns:
                # Calculate Sharpe ratio
                risk_free_rate = 0.03 / 252  # Daily risk-free rate (~3% annual)
                excess_returns = [r - risk_free_rate for r in daily_returns]
                avg_excess_return = sum(excess_returns) / len(excess_returns)
                std_dev = np.std(daily_returns)
                risk_metrics['sharpe_ratio'] = (avg_excess_return / std_dev) * np.sqrt(252) if std_dev > 0 else 0

                # Calculate Sortino ratio (using only negative returns for denominator)
                negative_returns = [r for r in excess_returns if r < 0]
                downside_dev = np.std(negative_returns) if negative_returns else 0.01
                risk_metrics['sortino_ratio'] = (avg_excess_return / downside_dev) * np.sqrt(
                    252) if downside_dev > 0 else 0

        # Calculate win rate and average win/loss ratio
        if closed_trades:
            winning_trades = [t for t in closed_trades if t['pnl'] > 0]
            losing_trades = [t for t in closed_trades if t['pnl'] < 0]

            win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
            risk_metrics['win_rate'] = win_rate * 100  # Convert to percentage

            # Calculate average profit/loss ratio
            if winning_trades and losing_trades:
                avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
                avg_loss = abs(sum(t['pnl'] for t in losing_trades) / len(losing_trades))
                risk_metrics['avg_win_loss_ratio'] = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Log and store risk metrics
        self.db.log_risk_metrics(risk_metrics)
        logger.info(
            f"Risk metrics updated: Sharpe={risk_metrics['sharpe_ratio']:.2f}, VaR95={risk_metrics['var_95']:.2%}")

        return risk_metrics
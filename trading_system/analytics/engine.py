"""
Analytics engine for performance tracking and visualization.
Provides comprehensive trading performance metrics and reports.
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class AnalyticsEngine:
    """
    Analytics engine for performance tracking and visualization
    """

    def __init__(self, db_manager, data_dir: str = "data"):
        """Initialize the analytics engine"""
        self.db = db_manager
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def generate_performance_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance summary

        Returns:
            Dictionary with performance metrics
        """
        # Get performance summary from database
        summary = self.db.get_performance_summary()

        # Get portfolio snapshots for equity curve
        snapshots = self.db.get_portfolio_snapshots(days_back=30)

        # Add equity curve data
        if snapshots:
            summary['equity_curve'] = [
                {'date': snapshot['timestamp'], 'value': snapshot['account_value']}
                for snapshot in snapshots
            ]

            # Calculate additional metrics
            values = [snapshot['account_value'] for snapshot in snapshots]
            if len(values) > 1:
                # Calculate returns
                returns = [
                    (values[i] - values[i - 1]) / values[i - 1]
                    for i in range(1, len(values))
                ]

                # Calculate drawdowns
                peak = values[0]
                drawdowns = []
                for value in values:
                    peak = max(peak, value)
                    drawdown = (peak - value) / peak if peak > 0 else 0
                    drawdowns.append(drawdown)

                # Add metrics to summary
                summary['metrics'] = {
                    'total_return': (values[-1] / values[0] - 1) * 100 if values[0] > 0 else 0,
                    'annualized_return': ((values[-1] / values[0]) ** (365 / len(values)) - 1) * 100 if values[
                                                                                                            0] > 0 and len(
                        values) > 0 else 0,
                    'volatility': np.std(returns) * np.sqrt(252) * 100 if returns else 0,
                    'max_drawdown': max(drawdowns) * 100 if drawdowns else 0,
                    'win_rate': summary['overall']['win_rate'] if 'win_rate' in summary['overall'] else 0
                }

        return summary

    def generate_risk_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive risk report

        Returns:
            Dictionary with risk metrics
        """
        # Get risk metrics from database
        risk_metrics = self.db.get_risk_metrics(days_back=30)

        # Get open positions for exposure analysis
        open_positions = self.db.get_open_positions()

        # Calculate strategy exposures
        strategy_exposures = {}
        for position in open_positions:
            strategy = position['event_type']
            if strategy not in strategy_exposures:
                strategy_exposures[strategy] = 0

            position_value = position['entry_price'] * position['quantity'] * 100
            strategy_exposures[strategy] += position_value

        # Calculate underlying exposures
        underlying_exposures = {}
        for position in open_positions:
            underlying = position['underlying']
            if underlying not in underlying_exposures:
                underlying_exposures[underlying] = 0

            position_value = position['entry_price'] * position['quantity'] * 100
            underlying_exposures[underlying] += position_value

        # Create time series of key risk metrics
        time_series = {}
        if risk_metrics:
            time_series = {
                'timestamps': [m['timestamp'] for m in risk_metrics],
                'var_95': [m['var_95'] * 100 for m in risk_metrics],  # Convert to percentage
                'sharpe_ratio': [m['sharpe_ratio'] for m in risk_metrics],
                'max_drawdown': [m['max_drawdown'] * 100 for m in risk_metrics]  # Convert to percentage
            }

        # Calculate latest metrics
        latest_metrics = risk_metrics[-1] if risk_metrics else {
            'var_95': 0,
            'var_99': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'max_drawdown': 0,
            'win_rate': 0
        }

        return {
            'latest_metrics': latest_metrics,
            'time_series': time_series,
            'strategy_exposures': strategy_exposures,
            'underlying_exposures': underlying_exposures,
            'open_positions_count': len(open_positions),
            'total_position_value': sum(p['entry_price'] * p['quantity'] * 100 for p in open_positions)
        }

    def generate_trade_analysis(self) -> Dict[str, Any]:
        """
        Generate analysis of trading performance

        Returns:
            Dictionary with trade analysis
        """
        # Get closed trades from database
        closed_trades = self.db.get_closed_positions(days_back=90)

        # Group trades by strategy
        trades_by_strategy = {}
        for trade in closed_trades:
            strategy = trade['event_type']
            if strategy not in trades_by_strategy:
                trades_by_strategy[strategy] = []
            trades_by_strategy[strategy].append(trade)

        # Calculate strategy performance
        strategy_performance = {}
        for strategy, trades in trades_by_strategy.items():
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]

            win_rate = len(winning_trades) / len(trades) if trades else 0

            avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
            avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

            total_pnl = sum(t['pnl'] for t in trades)
            avg_duration = sum((datetime.fromisoformat(t['exit_time']) - datetime.fromisoformat(
                t['entry_time'])).total_seconds() / 86400 for t in trades) / len(trades) if trades else 0

            strategy_performance[strategy] = {
                'trade_count': len(trades),
                'win_rate': win_rate * 100,  # Convert to percentage
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'avg_return': sum(t['pnl_percent'] for t in trades) / len(trades) if trades else 0,
                'total_pnl': total_pnl,
                'avg_duration_days': avg_duration
            }

        # Calculate performance by underlying
        performance_by_underlying = {}
        for trade in closed_trades:
            underlying = trade['underlying']
            if underlying not in performance_by_underlying:
                performance_by_underlying[underlying] = {
                    'trade_count': 0,
                    'win_count': 0,
                    'total_pnl': 0,
                    'avg_return': 0
                }

            performance_by_underlying[underlying]['trade_count'] += 1
            if trade['pnl'] > 0:
                performance_by_underlying[underlying]['win_count'] += 1

            performance_by_underlying[underlying]['total_pnl'] += trade['pnl']

        # Calculate win rates and avg returns for each underlying
        for underlying, perf in performance_by_underlying.items():
            perf['win_rate'] = (perf['win_count'] / perf['trade_count']) * 100 if perf['trade_count'] > 0 else 0

            # Find all trades for this underlying to calculate average return
            underlying_trades = [t for t in closed_trades if t['underlying'] == underlying]
            perf['avg_return'] = sum(t['pnl_percent'] for t in underlying_trades) / len(
                underlying_trades) if underlying_trades else 0

        # Get trade data ordered by date
        trade_timeline = [
            {
                'date': trade['exit_time'],
                'symbol': trade['symbol'],
                'strategy': trade['event_type'],
                'pnl': trade['pnl'],
                'pnl_percent': trade['pnl_percent'],
                'duration_days': (datetime.fromisoformat(trade['exit_time']) - datetime.fromisoformat(
                    trade['entry_time'])).total_seconds() / 86400
            }
            for trade in sorted(closed_trades, key=lambda x: x['exit_time']) if trade['exit_time']
        ]

        return {
            'strategy_performance': strategy_performance,
            'performance_by_underlying': performance_by_underlying,
            'trade_timeline': trade_timeline,
            'total_trades': len(closed_trades),
            'total_pnl': sum(t['pnl'] for t in closed_trades),
            'avg_return': sum(t['pnl_percent'] for t in closed_trades) / len(closed_trades) if closed_trades else 0,
            'win_rate': len([t for t in closed_trades if t['pnl'] > 0]) / len(
                closed_trades) * 100 if closed_trades else 0
        }

    def generate_daily_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive daily report

        Returns:
            Dictionary with daily report data
        """
        # Get today's date
        today = datetime.now().strftime('%Y-%m-%d')

        # Get performance summary
        performance = self.generate_performance_summary()

        # Get risk report
        risk = self.generate_risk_report()

        # Get trade analysis
        trade_analysis = self.generate_trade_analysis()

        # Get today's trades
        today_trades = self.db.execute_query(
            "SELECT * FROM trades WHERE date(entry_time) = %s OR date(exit_time) = %s",
            (today, today)
        )

        # Generate daily P&L
        daily_pnl = 0
        for trade in today_trades:
            if trade['status'] == 'CLOSED' and trade['exit_time'] and trade['exit_time'].startswith(today):
                daily_pnl += trade['pnl']

        # Generate report
        report = {
            'date': today,
            'daily_pnl': daily_pnl,
            'open_positions': risk['open_positions_count'],
            'total_position_value': risk['total_position_value'],
            'performance': performance,
            'risk': risk,
            'trade_analysis': trade_analysis,
            'today_trades': today_trades
        }

        # Save report to disk
        report_file = os.path.join(self.data_dir, f"daily_report_{today}.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, default=str)

        return report

    def generate_performance_charts(self, output_dir: str = None) -> List[str]:
        """
        Generate performance charts data for visualization

        Args:
            output_dir: Directory to save chart data (defaults to data_dir)

        Returns:
            List of file paths to generated chart data
        """
        if output_dir is None:
            output_dir = self.data_dir

        os.makedirs(output_dir, exist_ok=True)

        chart_files = []

        # Get data
        performance = self.generate_performance_summary()
        risk = self.generate_risk_report()
        trade_analysis = self.generate_trade_analysis()

        # 1. Equity curve data
        if 'equity_curve' in performance and performance['equity_curve']:
            equity_data = {
                'dates': [point['date'] for point in performance['equity_curve']],
                'values': [point['value'] for point in performance['equity_curve']]
            }

            # Save data
            equity_curve_file = os.path.join(output_dir, 'equity_curve.json')
            with open(equity_curve_file, 'w') as f:
                json.dump(equity_data, f, default=str)

            chart_files.append(equity_curve_file)

        # 2. Strategy performance data
        if trade_analysis and 'strategy_performance' in trade_analysis:
            strategy_data = {
                'strategies': list(trade_analysis['strategy_performance'].keys()),
                'win_rates': [trade_analysis['strategy_performance'][s]['win_rate'] for s in
                              trade_analysis['strategy_performance']],
                'avg_returns': [trade_analysis['strategy_performance'][s]['avg_return'] for s in
                                trade_analysis['strategy_performance']],
                'trade_counts': [trade_analysis['strategy_performance'][s]['trade_count'] for s in
                                 trade_analysis['strategy_performance']]
            }

            # Save data
            strategy_perf_file = os.path.join(output_dir, 'strategy_performance.json')
            with open(strategy_perf_file, 'w') as f:
                json.dump(strategy_data, f, default=str)

            chart_files.append(strategy_perf_file)

        # 3. Risk metrics data
        if risk and 'time_series' in risk and risk['time_series']:
            risk_data = {
                'timestamps': risk['time_series']['timestamps'],
                'var_95': risk['time_series']['var_95'],
                'sharpe_ratio': risk['time_series']['sharpe_ratio'],
                'max_drawdown': risk['time_series']['max_drawdown']
            }

            # Save data
            risk_metrics_file = os.path.join(output_dir, 'risk_metrics.json')
            with open(risk_metrics_file, 'w') as f:
                json.dump(risk_data, f, default=str)

            chart_files.append(risk_metrics_file)

        # 4. Trade P&L distribution data
        if trade_analysis and 'trade_timeline' in trade_analysis:
            pnl_values = [trade['pnl'] for trade in trade_analysis['trade_timeline']]

            pnl_data = {
                'values': pnl_values
            }

            # Save data
            pnl_dist_file = os.path.join(output_dir, 'pnl_distribution.json')
            with open(pnl_dist_file, 'w') as f:
                json.dump(pnl_data, f, default=str)

            chart_files.append(pnl_dist_file)

        return chart_files
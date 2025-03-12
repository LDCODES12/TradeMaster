"""
Command-line interface for the trading system.
Provides text-based monitoring and control.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Any
from tabulate import tabulate

logger = logging.getLogger(__name__)


class TradingSystemCLI:
    """Command-line interface for the trading system"""

    def __init__(self, trading_system):
        """Initialize the CLI"""
        self.trading_system = trading_system

    def start(self):
        """Start the CLI"""
        self._print_header()

        # Enter command loop
        self.command_loop()

    def _print_header(self):
        """Print the header banner"""
        print("\n" + "=" * 80)
        print("  Advanced Algorithmic Trading Platform".center(80))
        print("=" * 80)
        print("Type 'help' for available commands".center(80))
        print("-" * 80)

    def command_loop(self):
        """Main command loop"""
        while True:
            print("\n")
            command = input("Trading> ").strip().lower()

            if command in ['exit', 'quit', 'q']:
                print("Shutting down trading system...")
                if self.trading_system.trading_active:
                    self.trading_system.stop_trading("User requested shutdown")
                if self.trading_system.scheduler.running:
                    self.trading_system.scheduler.shutdown()
                print("Trading system shutdown complete")
                break

            elif command in ['help', '?', 'h']:
                self._show_help()

            elif command in ['status', 'stat', 's']:
                self._show_system_status()

            elif command in ['start', 'run']:
                if self.trading_system.trading_active:
                    print("Trading system is already active")
                else:
                    success = self.trading_system.start_trading()
                    if success:
                        print("Trading system started successfully")
                    else:
                        print("Failed to start trading system")

            elif command in ['stop', 'pause']:
                if not self.trading_system.trading_active:
                    print("Trading system is already inactive")
                else:
                    success = self.trading_system.stop_trading("User requested stop")
                    if success:
                        print("Trading system stopped successfully")
                    else:
                        print("Failed to stop trading system")

            elif command in ['positions', 'pos', 'p']:
                self._show_open_positions()

            elif command in ['trades', 't']:
                self._show_recent_trades()

            elif command in ['performance', 'perf']:
                self._show_performance_summary()

            elif command in ['risk', 'r']:
                self._show_risk_metrics()

            elif command in ['report', 'rep']:
                print("Generating daily report...")
                self.trading_system.generate_daily_report()
                print("Daily report generated")

            elif command in ['cycle', 'c']:
                if not self.trading_system.trading_active:
                    print("Trading system is not active. Activate it first.")
                else:
                    print("Running manual trading cycle...")
                    self.trading_system.trading_cycle()
                    print("Trading cycle completed")

            elif command in ['monitor', 'm']:
                print("Monitoring positions...")
                self.trading_system.monitor_positions()
                print("Position monitoring completed")

            elif command in ['clear', 'cls']:
                os.system('cls' if os.name == 'nt' else 'clear')
                self._print_header()

            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands")

    def _show_help(self):
        """Show help menu"""
        print("\nAvailable Commands:")

        commands = [
            ["status (s)", "Show system status"],
            ["start", "Start trading system"],
            ["stop", "Stop trading system"],
            ["positions (p)", "Show open positions"],
            ["trades (t)", "Show recent trades"],
            ["performance", "Show performance summary"],
            ["risk (r)", "Show risk metrics"],
            ["report", "Generate daily report"],
            ["cycle (c)", "Run manual trading cycle"],
            ["monitor (m)", "Monitor positions"],
            ["clear", "Clear the screen"],
            ["help (h)", "Show this help menu"],
            ["exit (q)", "Exit the system"]
        ]

        print(tabulate(commands, headers=["Command", "Description"], tablefmt="simple"))

    def _show_system_status(self):
        """Show system status"""
        status = self.trading_system.get_system_status()

        print("\nSystem Status:")
        print(f"Trading Active: {'Yes' if status['trading_active'] else 'No'}")
        print(f"System Ready: {'Yes' if status['system_ready'] else 'No'}")
        print(f"Market Open: {'Yes' if status['market_open'] else 'No'}")
        print(f"Last Run: {status['last_run_time']}")
        print(f"Open Positions: {status['open_positions']}")
        print(f"Daily P&L: ${status['daily_pnl']:.2f}")

        print("\nDaily Statistics:")
        headers = ["Metric", "Value"]
        data = [
            ["Trades Executed", status['daily_stats']['trades_executed']],
            ["Trades Exited", status['daily_stats']['trades_exited']],
            ["Opportunities Analyzed", status['daily_stats']['opportunities_analyzed']],
            ["Daily P&L", f"${status['daily_stats']['daily_pnl']:.2f}"]
        ]
        print(tabulate(data, headers=headers, tablefmt="simple"))

        print("\nScheduled Jobs:")
        headers = ["Job ID", "Next Run Time"]
        data = [[job_id, next_run] for job_id, next_run in status['scheduler_status']['jobs'].items()]
        if data:
            print(tabulate(data, headers=headers, tablefmt="simple"))
        else:
            print("No scheduled jobs")

    def _show_open_positions(self):
        """Show open positions"""
        open_positions = self.trading_system.db_manager.get_open_positions()

        if not open_positions:
            print("\nNo open positions")
            return

        print(f"\nOpen Positions ({len(open_positions)}):")

        headers = ["Symbol", "Type", "Strike", "Exp Date", "Entry Price", "Qty", "Event", "Event Date"]
        data = []

        for pos in open_positions:
            data.append([
                pos['symbol'],
                pos['trade_type'],
                f"${pos['strike_price']:.2f}",
                pos['expiration'],
                f"${pos['entry_price']:.2f}",
                pos['quantity'],
                pos['event_type'],
                pos['event_date']
            ])

        print(tabulate(data, headers=headers, tablefmt="simple"))

        # Calculate total position value
        total_value = sum(pos['entry_price'] * pos['quantity'] * 100 for pos in open_positions)
        print(f"\nTotal Position Value: ${total_value:.2f}")

    def _show_recent_trades(self):
        """Show recent closed trades"""
        closed_trades = self.trading_system.db_manager.get_closed_positions(days_back=30)

        if not closed_trades:
            print("\nNo closed trades in the last 30 days")
            return

        print(f"\nRecent Closed Trades ({len(closed_trades)}):")

        headers = ["Symbol", "Type", "Entry", "Exit", "P&L", "P&L %", "Exit Reason"]
        data = []

        for trade in closed_trades:
            data.append([
                trade['symbol'],
                trade['trade_type'],
                f"${trade['entry_price']:.2f}",
                f"${trade['exit_price']:.2f}" if trade['exit_price'] else "-",
                f"${trade['pnl']:.2f}" if trade['pnl'] is not None else "-",
                f"{trade['pnl_percent']:.2f}%" if trade['pnl_percent'] is not None else "-",
                trade['exit_reason'] if trade['exit_reason'] else "-"
            ])

        print(tabulate(data, headers=headers, tablefmt="simple"))

        # Calculate aggregate statistics
        total_pnl = sum(trade['pnl'] for trade in closed_trades if trade['pnl'] is not None)
        win_count = sum(1 for trade in closed_trades if trade['pnl'] is not None and trade['pnl'] > 0)
        win_rate = (win_count / len(closed_trades)) * 100 if closed_trades else 0

        print(f"\nTotal P&L: ${total_pnl:.2f}")
        print(f"Win Rate: {win_rate:.2f}% ({win_count}/{len(closed_trades)})")

    def _show_performance_summary(self):
        """Show performance summary"""
        summary = self.trading_system.analytics.generate_performance_summary()

        print("\nPerformance Summary:")

        if 'overall' in summary:
            overall = summary['overall']

            print("\nOverall Statistics:")
            headers = ["Metric", "Value"]
            data = [
                ["Total Trades", overall.get('total_trades', 0)],
                ["Closed Trades", overall.get('closed_trades', 0)],
                ["Open Trades", overall.get('open_trades', 0)],
                ["Winning Trades", overall.get('winning_trades', 0)],
                ["Losing Trades", overall.get('losing_trades', 0)],
                ["Win Rate", f"{overall.get('win_rate', 0):.2f}%"],
                ["Average Return", f"{overall.get('avg_return_percent', 0):.2f}%"],
                ["Total P&L", f"${overall.get('total_pnl', 0):.2f}"]
            ]
            print(tabulate(data, headers=headers, tablefmt="simple"))

        if 'by_strategy' in summary and summary['by_strategy']:
            print("\nPerformance by Strategy:")
            headers = ["Strategy", "Trades", "Win Rate", "Avg Return", "Total P&L"]
            data = []

            for strat in summary['by_strategy']:
                data.append([
                    strat['strategy'],
                    strat['total_trades'],
                    f"{strat.get('win_rate', 0):.2f}%",
                    f"{strat.get('avg_return_percent', 0):.2f}%",
                    f"${strat.get('total_pnl', 0):.2f}"
                ])

            print(tabulate(data, headers=headers, tablefmt="simple"))

        if 'metrics' in summary:
            metrics = summary['metrics']

            print("\nKey Performance Metrics:")
            headers = ["Metric", "Value"]
            data = [
                ["Total Return", f"{metrics.get('total_return', 0):.2f}%"],
                ["Annualized Return", f"{metrics.get('annualized_return', 0):.2f}%"],
                ["Volatility", f"{metrics.get('volatility', 0):.2f}%"],
                ["Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%"],
                ["Win Rate", f"{metrics.get('win_rate', 0):.2f}%"]
            ]
            print(tabulate(data, headers=headers, tablefmt="simple"))

    def _show_risk_metrics(self):
        """Show risk metrics"""
        risk_report = self.trading_system.analytics.generate_risk_report()

        print("\nRisk Management Report:")

        if 'latest_metrics' in risk_report and risk_report['latest_metrics']:
            metrics = risk_report['latest_metrics']

            print("\nLatest Risk Metrics:")
            headers = ["Metric", "Value"]
            data = [
                ["Value-at-Risk (95%)", f"{metrics.get('var_95', 0) * 100:.2f}%"],
                ["Value-at-Risk (99%)", f"{metrics.get('var_99', 0) * 100:.2f}%"],
                ["Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}"],
                ["Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}"],
                ["Maximum Drawdown", f"{metrics.get('max_drawdown', 0) * 100:.2f}%"],
                ["Win Rate", f"{metrics.get('win_rate', 0):.2f}%"]
            ]
            print(tabulate(data, headers=headers, tablefmt="simple"))

        if 'strategy_exposures' in risk_report and risk_report['strategy_exposures']:
            print("\nStrategy Exposures:")
            headers = ["Strategy", "Exposure ($)", "% of Portfolio"]
            data = []

            total_value = sum(amount for strategy, amount in risk_report['strategy_exposures'].items())

            for strategy, amount in risk_report['strategy_exposures'].items():
                percentage = (amount / total_value) * 100 if total_value > 0 else 0
                data.append([
                    strategy,
                    f"${amount:.2f}",
                    f"{percentage:.2f}%"
                ])

            print(tabulate(data, headers=headers, tablefmt="simple"))

        if 'underlying_exposures' in risk_report and risk_report['underlying_exposures']:
            print("\nUnderlying Exposures:")
            headers = ["Symbol", "Exposure ($)", "% of Portfolio"]
            data = []

            total_value = sum(amount for symbol, amount in risk_report['underlying_exposures'].items())

            for symbol, amount in risk_report['underlying_exposures'].items():
                percentage = (amount / total_value) * 100 if total_value > 0 else 0
                data.append([
                    symbol,
                    f"${amount:.2f}",
                    f"{percentage:.2f}%"
                ])

            print(tabulate(data, headers=headers, tablefmt="simple"))
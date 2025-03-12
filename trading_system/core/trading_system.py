"""
Main trading system coordinator class.
Integrates all components and manages the trading lifecycle.
"""

import logging
import threading
import queue
from datetime import datetime
import pytz
from typing import Dict, Any, Tuple, List
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Import components
from core.options_strategy import PrecisionOptionsArbitrage
from data.database import DatabaseManager
from risk.manager import RiskManager
from analytics.engine import AnalyticsEngine
from ui.notifications import NotificationSystem

logger = logging.getLogger(__name__)
EASTERN_TZ = pytz.timezone('US/Eastern')


class TradingSystem:
    """
    Main trading system that coordinates all components
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the trading system with configuration"""
        self.config = config

        # Initialize database
        self.db_manager = DatabaseManager()

        # Initialize trading strategy
        self.strategy = PrecisionOptionsArbitrage(config.get('alpaca', {}))

        # Initialize risk manager
        self.risk_manager = RiskManager(self.db_manager, config.get('risk_management', {}))

        # Initialize analytics engine
        self.analytics = AnalyticsEngine(self.db_manager)

        # Initialize notification system
        self.notifications = NotificationSystem(config.get('notifications', {}))

        # Initialize scheduler
        self.scheduler = BackgroundScheduler(timezone=EASTERN_TZ)

        # System state
        self.trading_active = False
        self.system_ready = False
        self.last_run_time = None
        self.daily_stats = {
            'trades_executed': 0,
            'trades_exited': 0,
            'opportunities_analyzed': 0,
            'daily_pnl': 0.0
        }

        # Message queue for inter-thread communication
        self.message_queue = queue.Queue()

        # Initialize system
        self._setup_system()

    def _setup_system(self):
        """Set up the trading system components and schedules"""
        # Set up scheduled tasks

        # Main trading cycle (during market hours)
        check_interval = int(self.config.get('trading_system', {}).get('check_interval_minutes', 10))
        self.scheduler.add_job(
            self.trading_cycle,
            'interval',
            minutes=check_interval,
            id='trading_cycle',
            replace_existing=True
        )

        # Position monitoring
        monitor_interval = int(self.config.get('trading_system', {}).get('monitor_interval_minutes', 5))
        self.scheduler.add_job(
            self.monitor_positions,
            'interval',
            minutes=monitor_interval,
            id='position_monitor',
            replace_existing=True
        )

        # Daily reports (after market close)
        self.scheduler.add_job(
            self.generate_daily_report,
            CronTrigger(hour=16, minute=15, timezone=EASTERN_TZ),  # 4:15 PM Eastern
            id='daily_report',
            replace_existing=True
        )

        # System health check
        self.scheduler.add_job(
            self.system_health_check,
            'interval',
            minutes=30,
            id='health_check',
            replace_existing=True
        )

        # Risk metrics update
        self.scheduler.add_job(
            self.update_risk_metrics,
            'interval',
            minutes=60,
            id='risk_metrics',
            replace_existing=True
        )

        # Start message processing thread
        threading.Thread(target=self._process_messages, daemon=True).start()

        # System is ready
        self.system_ready = True
        logger.info("Trading system initialized and ready")

    def _process_messages(self):
        """Process messages from the queue"""
        while True:
            try:
                message = self.message_queue.get(timeout=1.0)

                if message['type'] == 'notification':
                    self.notifications.send_notification(
                        message['message'],
                        level=message['level'],
                        channel=message.get('channel')
                    )
                elif message['type'] == 'trade_notification':
                    self.notifications.send_trade_notification(
                        message['trade_data'],
                        message['action']
                    )
                elif message['type'] == 'risk_alert':
                    self.notifications.send_risk_alert(
                        message['alert_type'],
                        message['message']
                    )
                elif message['type'] == 'log_trade':
                    self.db_manager.log_trade(message['trade_data'])
                elif message['type'] == 'update_trade_exit':
                    self.db_manager.update_trade_exit(
                        message['symbol'],
                        message['order_id'],
                        message['exit_data']
                    )

                self.message_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    def is_market_open(self) -> bool:
        """Check if the market is currently open"""
        try:
            current_time = datetime.now(EASTERN_TZ)

            # Check if it's a weekday
            if current_time.weekday() >= 5:  # Saturday or Sunday
                return False

            # Check if it's during trading hours
            start_time_str = self.config.get('trading_system', {}).get('trading_hours_start', '09:30')
            end_time_str = self.config.get('trading_system', {}).get('trading_hours_end', '16:00')

            start_hour, start_minute = map(int, start_time_str.split(':'))
            end_hour, end_minute = map(int, end_time_str.split(':'))

            market_open = current_time.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
            market_close = current_time.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)

            return market_open <= current_time <= market_close

        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False

    def start_trading(self):
        """Start the trading system"""
        if not self.system_ready:
            logger.error("Cannot start trading - system not ready")
            return False

        if self.trading_active:
            logger.warning("Trading system already active")
            return True

        try:
            # Start the scheduler
            if not self.scheduler.running:
                self.scheduler.start()

            # Set trading state
            self.trading_active = True

            # Log and notify
            logger.info("Trading system activated")
            self.message_queue.put({
                'type': 'notification',
                'message': "Trading system activated",
                'level': 'info'
            })

            # Perform initial position monitoring
            self.monitor_positions()

            # Run initial trading cycle if market is open
            if self.is_market_open():
                self.trading_cycle()

            return True

        except Exception as e:
            logger.error(f"Error starting trading system: {e}")
            self.message_queue.put({
                'type': 'notification',
                'message': f"Failed to start trading system: {e}",
                'level': 'error'
            })
            return False

    def stop_trading(self, reason: str = "User requested stop"):
        """Stop the trading system"""
        if not self.trading_active:
            logger.warning("Trading system already inactive")
            return True

        try:
            # Set trading state
            self.trading_active = False

            # Log and notify
            logger.info(f"Trading system deactivated: {reason}")
            self.message_queue.put({
                'type': 'notification',
                'message': f"Trading system deactivated: {reason}",
                'level': 'warning'
            })

            return True

        except Exception as e:
            logger.error(f"Error stopping trading system: {e}")
            self.message_queue.put({
                'type': 'notification',
                'message': f"Error stopping trading system: {e}",
                'level': 'error'
            })
            return False

    def trading_cycle(self):
        """Run a full trading cycle"""
        if not self.trading_active:
            logger.info("Trading cycle skipped - trading inactive")
            return

        if not self.is_market_open():
            logger.info("Trading cycle skipped - market closed")
            return

        logger.info("Starting trading cycle")
        self.last_run_time = datetime.now()

        try:
            # Check circuit breakers
            circuit_breakers_clear, reason = self.risk_manager.check_circuit_breakers()
            if not circuit_breakers_clear:
                logger.warning(f"Trading halted: {reason}")
                self.message_queue.put({
                    'type': 'risk_alert',
                    'alert_type': 'circuit_breaker',
                    'message': reason
                })
                self.stop_trading(reason=reason)
                return

            # Update portfolio metrics
            self._update_portfolio_snapshot()

            # Get open positions
            open_positions = self.db_manager.get_open_positions()
            max_positions = int(self.config.get('trading_system', {}).get('max_positions', 5))

            if len(open_positions) >= max_positions:
                logger.info(
                    f"Maximum positions reached ({len(open_positions)}/{max_positions}), skipping opportunity analysis")
                return

            # Find new opportunities
            min_sharpe = float(self.config.get('trading_system', {}).get('min_sharpe', 0.25))

            # Analyze opportunities using the strategy
            opportunities = self.strategy.analyze_opportunities(min_sharpe=min_sharpe)

            self.daily_stats['opportunities_analyzed'] += len(opportunities)

            if not opportunities:
                logger.info("No suitable opportunities found")
                return

            # Filter opportunities based on risk management
            approved_opportunities = []
            for opp in opportunities:
                approved, reason = self.risk_manager.check_trade_approval(opp)
                if approved:
                    approved_opportunities.append(opp)
                else:
                    logger.info(f"Opportunity rejected: {opp['option']['symbol']} - {reason}")

            if not approved_opportunities:
                logger.info("No opportunities approved by risk management")
                return

            # Sort by Sharpe ratio
            approved_opportunities.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

            # Execute trades
            positions_to_add = max_positions - len(open_positions)
            positions_taken = 0

            for opp in approved_opportunities[:positions_to_add]:
                # Calculate optimal position size
                contracts = self.risk_manager.calculate_position_size(opp)

                # Execute trade
                if contracts > 0:
                    logger.info(f"Executing trade: {opp['option']['symbol']} - {contracts} contract(s)")
                    success = self.strategy.execute_trade(opp, contracts=contracts)

                    if success:
                        positions_taken += 1
                        self.daily_stats['trades_executed'] += 1

                        # Store trade in database
                        trade_data = {**opp, 'contracts': contracts}
                        self.message_queue.put({
                            'type': 'log_trade',
                            'trade_data': trade_data
                        })

                        # Send notification
                        self.message_queue.put({
                            'type': 'trade_notification',
                            'trade_data': trade_data,
                            'action': 'entry'
                        })

            logger.info(f"Trading cycle completed: {positions_taken} new positions taken")

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            self.message_queue.put({
                'type': 'notification',
                'message': f"Error in trading cycle: {e}",
                'level': 'error'
            })

    def monitor_positions(self):
        """Monitor existing positions and execute exit strategies"""
        if not self.system_ready:
            return

        try:
            # Use the strategy's monitor_positions method
            closed_positions = self.strategy.monitor_positions()

            if closed_positions:
                self.daily_stats['trades_exited'] += len(closed_positions)
                logger.info(f"{len(closed_positions)} position(s) exited")

                # Update closed positions in the database
                for position in closed_positions:
                    # Update trade exit in database
                    exit_data = {
                        'exit_price': position['exit_price'],
                        'exit_time': position['exit_time'],
                        'pnl': position['pnl'],
                        'pnl_pct': position['pnl_pct'],
                        'exit_reason': position['exit_reason']
                    }

                    self.message_queue.put({
                        'type': 'update_trade_exit',
                        'symbol': position['symbol'],
                        'order_id': position['order_id'],
                        'exit_data': exit_data
                    })

                    # Send notification
                    self.message_queue.put({
                        'type': 'trade_notification',
                        'trade_data': position,
                        'action': 'exit'
                    })

                    # Update daily P&L
                    self.daily_stats['daily_pnl'] += position['pnl']

            # Update portfolio snapshot
            self._update_portfolio_snapshot()

        except Exception as e:
            logger.error(f"Error monitoring positions: {e}", exc_info=True)
            self.message_queue.put({
                'type': 'notification',
                'message': f"Error monitoring positions: {e}",
                'level': 'error'
            })

    def _update_portfolio_snapshot(self):
        """Update portfolio snapshot in the database"""
        try:
            # Get account data
            account = self.strategy.get_account()

            # Calculate position values
            open_positions = self.strategy.get_positions()
            open_positions_value = sum(p['value'] for p in open_positions)

            # Create snapshot
            snapshot = {
                'account_value': account['portfolio_value'],
                'buying_power': account['buying_power'],
                'cash': account['cash'],
                'open_positions_count': len(open_positions),
                'open_positions_value': open_positions_value,
                'daily_pnl': self.daily_stats['daily_pnl']
            }

            # Store snapshot
            self.db_manager.log_portfolio_snapshot(snapshot)

        except Exception as e:
            logger.error(f"Error updating portfolio snapshot: {e}")

    def update_risk_metrics(self):
        """Update risk metrics"""
        if not self.system_ready:
            return

        try:
            # Calculate risk metrics
            risk_metrics = self.risk_manager.calculate_risk_metrics()
            logger.info(
                f"Risk metrics updated: VaR(95)={risk_metrics['var_95']:.2%}, Sharpe={risk_metrics['sharpe_ratio']:.2f}")

        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")

    def generate_daily_report(self):
        """Generate daily performance report"""
        if not self.system_ready:
            return

        try:
            # Generate report
            report = self.analytics.generate_daily_report()

            # Send report notification
            self.notifications.send_daily_report(report)

            # Reset daily stats
            self.daily_stats = {
                'trades_executed': 0,
                'trades_exited': 0,
                'opportunities_analyzed': 0,
                'daily_pnl': 0.0
            }

            logger.info("Daily report generated and sent")

        except Exception as e:
            logger.error(f"Error generating daily report: {e}")

    def system_health_check(self):
        """Check system health and resources"""
        if not self.system_ready:
            return

        try:
            import psutil

            # Check system resources
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')

            # Log health check
            health_status = "OK"
            if cpu_percent > 90 or memory_info.percent > 90 or disk_usage.percent > 90:
                health_status = "WARNING"

            logger.info(
                f"System health check: {health_status} - CPU: {cpu_percent}%, Memory: {memory_info.percent}%, Disk: {disk_usage.percent}%")

            # Send warnings if needed
            if health_status == "WARNING":
                self.message_queue.put({
                    'type': 'notification',
                    'message': f"System resources critical: CPU {cpu_percent}%, Memory {memory_info.percent}%, Disk {disk_usage.percent}%",
                    'level': 'warning'
                })

        except Exception as e:
            logger.error(f"Error in system health check: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        # Get open positions
        open_positions = self.db_manager.get_open_positions()

        # Calculate daily P&L
        daily_pnl = self.daily_stats['daily_pnl']

        # Get risk metrics
        risk_metrics = self.db_manager.get_risk_metrics(days_back=1)
        latest_risk = risk_metrics[-1] if risk_metrics else {}

        # Get scheduler status
        jobs = {job.id: job.next_run_time for job in self.scheduler.get_jobs()}

        return {
            'trading_active': self.trading_active,
            'system_ready': self.system_ready,
            'market_open': self.is_market_open(),
            'last_run_time': self.last_run_time,
            'open_positions': len(open_positions),
            'daily_pnl': daily_pnl,
            'daily_stats': self.daily_stats,
            'risk_metrics': latest_risk,
            'scheduler_status': {
                'running': self.scheduler.running,
                'jobs': jobs
            }
        }
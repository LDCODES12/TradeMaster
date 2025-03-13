"""
Main trading system coordinator class with real market data integration.
Integrates all components and manages the trading lifecycle.
"""

import logging
import threading
import queue
import asyncio
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Any, Tuple



# Core scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Alpaca API
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame



# Initialize logger and timezone
logger = logging.getLogger(__name__)
EASTERN_TZ = pytz.timezone('US/Eastern')


class TradingSystem:
    """
    Main trading system that coordinates all components with live market data
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the trading system with configuration"""
        self.config = config

        # Extract Alpaca credentials
        alpaca_config = config.get('alpaca', {})
        self.api_key = alpaca_config.get('api_key', '')
        self.api_secret = alpaca_config.get('api_secret', '')

        # Validate API credentials
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials are required")

        # Initialize database
        from data.database import DatabaseManager
        self.db_manager = DatabaseManager()

        # Initialize trading clients
        self.trading_client = TradingClient(self.api_key, self.api_secret, paper=True)
        self.data_client = StockHistoricalDataClient(self.api_key, self.api_secret)

        # Create a clean API configuration object
        api_config = {
            'alpaca': {
                'api_key': self.api_key,
                'api_secret': self.api_secret
            },
            'finnhub': {'api_key': config.get('finnhub', {}).get('api_key', '')},
            'alphavantage': {'api_key': config.get('alphavantage', {}).get('api_key', '')},
            'polygon': {'api_key': config.get('polygon', {}).get('api_key', '')}
        }

        # Initialize trading strategy
        from core.options_strategy import PrecisionOptionsArbitrage
        self.strategy = PrecisionOptionsArbitrage(api_config)

        # Initialize risk manager
        from risk.manager import RiskManager
        self.risk_manager = RiskManager(self.db_manager, config.get('risk_management', {}))

        # Initialize analytics engine
        from analytics.engine import AnalyticsEngine
        self.analytics = AnalyticsEngine(self.db_manager)

        # Initialize notification system
        from ui.notifications import NotificationSystem
        self.notifications = NotificationSystem(config.get('notifications', {}))

        # Initialize scheduler
        self.scheduler = BackgroundScheduler(timezone=EASTERN_TZ)

        # System state
        self.trading_active = False
        self.system_ready = False
        self.last_run_time = None
        self.market_data_stream = None
        self.stream_thread = None
        self.last_quotes = {}  # Store most recent quotes

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

        self.market_data_stream_manager = None


    def _setup_system(self):
        """Set up the trading system components and schedules"""
        # Set up scheduled tasks

        self.scheduler.add_job(
            self.strategy.monitor_strategy_performance,
            CronTrigger(hour=17, minute=30, timezone=EASTERN_TZ),  # 5:30 PM Eastern
            id='strategy_monitor',
            replace_existing=True
        )

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

    async def _handle_trade(self, trade):
        """Asynchronous trade update handler"""
        logger.debug(f"Trade update received: {trade}")
        # Implement any trade-specific processing here
        # Can access self attributes and methods

    async def _handle_quote(self, quote):
        """Asynchronous quote update handler"""
        logger.debug(f"Quote update received: {quote}")

        # Store latest quote data
        self.last_quotes[quote.symbol] = {
            'bid': quote.bid_price,
            'ask': quote.ask_price,
            'time': datetime.now(),
            'spread': quote.ask_price - quote.bid_price
        }

        # Check for any impact on our positions
        self._check_for_price_alerts(quote.symbol, quote)

    async def _handle_bar(self, bar):
        """Asynchronous bar update handler"""
        logger.debug(f"Bar update received: {bar}")
        # Implement any bar-related processing here

    def _setup_market_data_stream(self):
        """Set up real-time market data streaming with Alpaca"""
        if hasattr(self,
                   'market_data_stream_manager') and self.market_data_stream_manager and self.market_data_stream_manager.is_running():
            logger.info("Market data stream already running")
            return

        try:
            # Import the StreamManager here to avoid circular imports
            from utils.market_data_stream import MarketDataStreamManager

            # Initialize the stream manager
            self.market_data_stream_manager = MarketDataStreamManager(
                self.api_key,
                self.api_secret,
            )

            # Find symbols to track based on open positions and watchlist
            symbols_to_track = set()

            # Add symbols from open positions
            open_positions = self.db_manager.get_open_positions()
            for position in open_positions:
                underlying = position.get('underlying', '')
                if underlying:
                    symbols_to_track.add(underlying)

            # Add symbols from event calendar
            if hasattr(self.strategy, 'events_calendar'):
                for event in self.strategy.events_calendar:
                    symbols_to_track.add(event['symbol'])

            symbols_to_track = list(symbols_to_track)

            if not symbols_to_track:
                # Default watchlist if no specific symbols
                symbols_to_track = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'AMZN', 'GOOGL']

            logger.info(f"Setting up market data stream for {len(symbols_to_track)} symbols")

            # Set up the handlers
            self.market_data_stream_manager.setup(
                trade_handler=self._handle_trade,
                quote_handler=self._handle_quote,
                bar_handler=self._handle_bar,
                symbols=symbols_to_track
            )

            # Start the stream
            success = self.market_data_stream_manager.start()
            if success:
                logger.info("Market data streaming started successfully")
            else:
                logger.warning("Failed to start market data stream")

        except Exception as e:
            logger.error(f"Failed to set up market data stream: {e}")
            self.market_data_stream_manager = None

    def _cleanup_market_data_stream(self):
        """Clean up market data stream resources"""
        if hasattr(self, 'market_data_stream_manager') and self.market_data_stream_manager:
            try:
                self.market_data_stream_manager.stop()
                logger.info("Market data stream stopped and cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up market data stream: {e}")
            finally:
                self.market_data_stream_manager = None

    async def _check_for_price_alerts(self, symbol: str, quote: Any):
        """Check if a price update should trigger alerts for open positions"""
        try:
            # Extract current price based on how it's provided
            if hasattr(quote, 'bid_price'):
                current_price = float(quote.bid_price)  # Original Alpaca stream object
            elif isinstance(quote, dict) and 'bp' in quote:
                current_price = float(quote.get('bp', 0))  # Msgpack unpacked dict
            else:
                # Can't determine price
                return

            if current_price <= 0:
                return

            # Check all positions with this underlying
            open_positions = self.db_manager.get_open_positions()
            for position in open_positions:
                if position.get('underlying') == symbol:
                    # Calculate unrealized P&L
                    entry_price = position.get('entry_price', 0)
                    contracts = position.get('quantity', 0)

                    if entry_price and contracts:
                        entry_value = entry_price * contracts * 100
                        # For options we'd need the actual option price, but this is a rough estimate
                        current_value = entry_value * (1 + (current_price / position.get('strike', 100) - 1))
                        pnl_pct = (current_value / entry_value - 1) * 100

                        # Check for significant price movements
                        if pnl_pct > 20:  # Profitable position
                            self.message_queue.put({
                                'type': 'notification',
                                'message': f"Position {position['symbol']} is up {pnl_pct:.1f}% - consider taking profit",
                                'level': 'info'
                            })
                        elif pnl_pct < -15:  # Losing position
                            self.message_queue.put({
                                'type': 'notification',
                                'message': f"Position {position['symbol']} is down {abs(pnl_pct):.1f}% - monitor closely",
                                'level': 'warning'
                            })
        except Exception as e:
            logger.error(f"Error checking price alerts: {e}")

    def is_market_open(self) -> bool:
        """Check if the market is currently open using Alpaca API"""
        try:
            # Get the clock from Alpaca
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market hours via API: {e}")

            # Fallback to time-based check
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

            # Start market data streaming
            self._setup_market_data_stream()

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

            # Clean up streaming connections
            self._cleanup_market_data_stream()

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
            open_positions_value = sum(p.get('market_value', 0) for p in open_positions)

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

            # Check Alpaca API connectivity
            try:
                clock = self.trading_client.get_clock()
                logger.info(f"Alpaca API check: OK (Market is {'open' if clock.is_open else 'closed'})")
            except Exception as e:
                logger.error(f"Alpaca API check failed: {e}")
                self.message_queue.put({
                    'type': 'notification',
                    'message': f"Alpaca API connectivity issue: {e}",
                    'level': 'error'
                })

            # Check market data stream
            if self.trading_active:
                stream_running = (hasattr(self, 'market_data_stream_manager') and
                                  self.market_data_stream_manager and
                                  self.market_data_stream_manager.is_running())

                if not stream_running:
                    logger.warning("Market data stream not running - attempting to restart")
                    self._setup_market_data_stream()

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
        jobs = {}
        for job in self.scheduler.get_jobs():
            jobs[job.id] = str(job.next_run_time) if hasattr(job, 'next_run_time') else 'Unknown'

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
            },
            'market_data_stream': 'active' if self.market_data_stream else 'inactive'
        }
"""
Notification system for alerts and reporting.
Handles various notification channels like console, email, etc.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class NotificationSystem:
    """
    Notification system for alerts and reporting
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the notification system"""
        self.config = config
        self.enabled = config.get('enabled', True) in [True, 'true', 'True', '1', 1]

        # Set up notification channels
        self.channels = {}

        # Terminal notifications (always enabled)
        self.channels['terminal'] = self._send_terminal_notification

        # Check if email notifications are enabled and properly configured
        if self.enabled and 'email' in config and config.get('email', {}).get('enabled', False):
            try:
                # Only import if email is enabled to avoid unnecessary dependencies
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                import smtplib

                self.email_config = config.get('email', {})
                self.channels['email'] = self._send_email_notification
                logger.info("Email notifications enabled")
            except ImportError:
                logger.warning("Email notifications requested but dependencies not installed")

        # Check if Slack notifications are enabled
        if self.enabled and 'slack' in config and config.get('slack', {}).get('enabled', False):
            try:
                # Only import if Slack is enabled to avoid unnecessary dependencies
                import requests

                self.slack_config = config.get('slack', {})
                self.channels['slack'] = self._send_slack_notification
                logger.info("Slack notifications enabled")
            except ImportError:
                logger.warning("Slack notifications requested but dependencies not installed")

    def send_notification(self, message: str, level: str = 'info', channel: str = None):
        """
        Send a notification

        Args:
            message: Notification message
            level: Severity level ('info', 'warning', 'error', 'critical')
            channel: Specific channel to use (defaults to all enabled channels)
        """
        if not self.enabled:
            return

        # Format the message with timestamp and level
        formatted_message = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level.upper()}] {message}"

        # Send to specific channel or all channels
        if channel and channel in self.channels:
            self.channels[channel](formatted_message, level)
        else:
            # Send to all enabled channels
            for channel_name, send_func in self.channels.items():
                send_func(formatted_message, level)

    def _send_terminal_notification(self, message: str, level: str):
        """Send a notification to the terminal"""
        # Use different colors based on level
        if level.lower() == 'error' or level.lower() == 'critical':
            print(f"\033[91m{message}\033[0m")  # Red
        elif level.lower() == 'warning':
            print(f"\033[93m{message}\033[0m")  # Yellow
        else:
            print(f"\033[94m{message}\033[0m")  # Blue

    def _send_email_notification(self, message: str, level: str):
        """Send a notification via email"""
        try:
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            import smtplib

            # Get email configuration
            smtp_host = self.email_config.get('smtp_host', 'smtp.gmail.com')
            smtp_port = int(self.email_config.get('smtp_port', 587))
            smtp_user = self.email_config.get('smtp_user', '')
            smtp_password = self.email_config.get('smtp_password', '')
            from_addr = self.email_config.get('from', 'trading-system@example.com')
            to_addrs = self.email_config.get('to', [])

            if not smtp_user or not smtp_password or not to_addrs:
                logger.warning("Email notification failed: Missing configuration")
                return

            # Create message
            msg = MIMEMultipart()
            msg['From'] = from_addr
            msg['To'] = ', '.join(to_addrs) if isinstance(to_addrs, list) else to_addrs
            msg['Subject'] = f"Trading System {level.upper()}: {message[:50]}..."

            msg.attach(MIMEText(message, 'plain'))

            # Send the message
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

            logger.info(f"Email notification sent to {to_addrs}")
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    def _send_slack_notification(self, message: str, level: str):
        """Send a notification via Slack webhook"""
        try:
            import requests

            webhook_url = self.slack_config.get('webhook_url', '')

            if not webhook_url:
                logger.warning("Slack notification failed: Missing webhook URL")
                return

            # Add emoji based on level
            if level.lower() == 'error' or level.lower() == 'critical':
                emoji = ':red_circle:'
            elif level.lower() == 'warning':
                emoji = ':warning:'
            else:
                emoji = ':large_blue_circle:'

            payload = {
                'text': f"{emoji} {message}"
            }

            response = requests.post(webhook_url, json=payload)

            if response.status_code != 200:
                logger.warning(f"Slack notification failed with status {response.status_code}: {response.text}")
            else:
                logger.info("Slack notification sent")
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def send_trade_notification(self, trade_data: Dict[str, Any], action: str):
        """
        Send a notification about a trade

        Args:
            trade_data: Trade data dictionary
            action: Action performed ('entry', 'exit')
        """
        if not self.enabled:
            return

        if action == 'entry':
            symbol = trade_data.get('symbol', '')
            if not symbol and 'option' in trade_data:
                symbol = trade_data['option'].get('symbol', '')

            contracts = trade_data.get('contracts', 0)
            entry_price = trade_data.get('entry_price', 0)
            if not entry_price and 'option' in trade_data:
                entry_price = trade_data['option'].get('price', 0)

            event_type = ''
            event_symbol = ''
            if 'event' in trade_data:
                event_type = trade_data['event'].get('event_type', '')
                event_symbol = trade_data['event'].get('symbol', '')

            expected_roi = trade_data.get('expected_roi', 0)

            message = (
                f"NEW TRADE: {contracts} contract(s) of {symbol} "
                f"at ${entry_price:.2f}. "
                f"Based on {event_type} for {event_symbol}. "
                f"Expected ROI: {expected_roi:.2f}%"
            )
            self.send_notification(message, level='info')
        elif action == 'exit':
            symbol = trade_data.get('symbol', '')
            exit_price = trade_data.get('exit_price', 0)
            pnl = trade_data.get('pnl', 0)
            pnl_pct = trade_data.get('pnl_pct', 0)
            exit_reason = trade_data.get('exit_reason', 'unknown')

            message = (
                f"TRADE CLOSED: {symbol} at ${exit_price:.2f}. "
                f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%). "
                f"Reason: {exit_reason}"
            )
            level = 'info' if pnl >= 0 else 'warning'
            self.send_notification(message, level=level)

    def send_risk_alert(self, alert_type: str, message: str):
        """
        Send a risk management alert

        Args:
            alert_type: Type of alert ('circuit_breaker', 'position_limit', 'drawdown')
            message: Alert message
        """
        if not self.enabled:
            return

        full_message = f"RISK ALERT ({alert_type}): {message}"
        self.send_notification(full_message, level='warning')

    def send_system_status(self, status: Dict[str, Any]):
        """
        Send system status notification

        Args:
            status: System status dictionary
        """
        if not self.enabled:
            return

        message = (
            f"SYSTEM STATUS: Trading {'ACTIVE' if status['trading_active'] else 'INACTIVE'}. "
            f"Positions: {status['open_positions']}. "
            f"Daily P&L: ${status['daily_pnl']:.2f}"
        )
        self.send_notification(message, level='info')

    def send_daily_report(self, report: Dict[str, Any]):
        """
        Send daily performance report

        Args:
            report: Daily report dictionary
        """
        if not self.enabled:
            return

        message = (
            f"DAILY REPORT ({report['date']})\n\n"
            f"Daily P&L: ${report['daily_pnl']:.2f}\n"
            f"Open Positions: {report['open_positions']}\n"
            f"Position Value: ${report['total_position_value']:.2f}\n\n"
        )

        # Add performance metrics if available
        if 'trade_analysis' in report:
            trade_analysis = report['trade_analysis']
            message += (
                f"Performance Metrics:\n"
                f"- Win Rate: {trade_analysis.get('win_rate', 0):.2f}%\n"
                f"- Avg Return: {trade_analysis.get('avg_return', 0):.2f}%\n"
                f"- Total Trades: {trade_analysis.get('total_trades', 0)}\n\n"
            )

        # Add risk metrics if available
        if 'risk' in report and 'latest_metrics' in report['risk']:
            risk_metrics = report['risk']['latest_metrics']
            message += (
                f"Risk Metrics:\n"
                f"- VaR (95%): {risk_metrics.get('var_95', 0) * 100:.2f}%\n"
                f"- Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}\n"
            )

        self.send_notification(message, level='info')
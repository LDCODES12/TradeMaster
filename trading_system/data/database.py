"""
Database manager for storing and retrieving trading data.
Uses SQLite for simplicity and portability.
"""

import sqlite3
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)
DB_PATH = "data/trading_system.db"


class DatabaseManager:
    """Handles all database operations for the trading system"""

    def __init__(self, db_path: str = DB_PATH):
        """Initialize the database manager"""
        self.db_path = db_path

        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self._initialize_database()

    def _initialize_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create trades table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            underlying TEXT NOT NULL,
            trade_type TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            quantity INTEGER NOT NULL,
            entry_time TIMESTAMP NOT NULL,
            exit_time TIMESTAMP,
            expiration TEXT NOT NULL,
            strike_price REAL NOT NULL,
            days_to_expiry INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            event_date TEXT NOT NULL,
            expected_roi REAL NOT NULL,
            profit_probability REAL NOT NULL,
            sharpe_ratio REAL NOT NULL,
            status TEXT NOT NULL,
            pnl REAL,
            pnl_percent REAL,
            exit_reason TEXT,
            order_id TEXT NOT NULL
        )
        ''')

        # Create portfolio_snapshots table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            account_value REAL NOT NULL,
            buying_power REAL NOT NULL,
            cash REAL NOT NULL,
            open_positions_count INTEGER NOT NULL,
            open_positions_value REAL NOT NULL,
            daily_pnl REAL,
            daily_pnl_percent REAL
        )
        ''')

        # Create risk_metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            var_95 REAL NOT NULL,
            var_99 REAL NOT NULL,
            max_drawdown REAL NOT NULL,
            sharpe_ratio REAL NOT NULL,
            sortino_ratio REAL NOT NULL,
            beta REAL NOT NULL,
            correlation_spy REAL NOT NULL,
            avg_win_loss_ratio REAL NOT NULL,
            win_rate REAL NOT NULL
        )
        ''')

        # Create system_logs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP NOT NULL,
            level TEXT NOT NULL,
            component TEXT NOT NULL,
            message TEXT NOT NULL
        )
        ''')

        conn.commit()
        conn.close()

        logger.info("Database initialized successfully")

    def log_trade(self, trade_data: Dict[str, Any]):
        """Log a trade to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO trades (
            symbol, underlying, trade_type, entry_price, quantity, entry_time,
            expiration, strike_price, days_to_expiry, event_type, event_date,
            expected_roi, profit_probability, sharpe_ratio, status, order_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data['option']['symbol'],
            trade_data['event']['symbol'],
            trade_data['option']['option_type'],
            trade_data['option']['price'],
            trade_data['contracts'],
            datetime.now(),
            trade_data['option']['expiration'],
            trade_data['option']['strike'],
            trade_data['option']['days_to_expiry'],
            trade_data['event']['event_type'],
            trade_data['event']['event_date'],
            trade_data['expected_roi'],
            trade_data['profit_probability'],
            trade_data.get('sharpe_ratio', 0.0),
            'OPEN',
            trade_data['order_id'] if 'order_id' in trade_data else f"order-{hash(str(trade_data))}"
        ))

        conn.commit()
        conn.close()
        logger.info(f"Trade logged to database: {trade_data['option']['symbol']}")

    def update_trade_exit(self, symbol: str, order_id: str, exit_data: Dict[str, Any]):
        """Update a trade with exit information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        UPDATE trades SET
            exit_price = ?,
            exit_time = ?,
            status = ?,
            pnl = ?,
            pnl_percent = ?,
            exit_reason = ?
        WHERE symbol = ? AND order_id = ? AND status = 'OPEN'
        ''', (
            exit_data['exit_price'],
            exit_data['exit_time'],
            'CLOSED',
            exit_data['pnl'],
            exit_data['pnl_pct'],
            exit_data['exit_reason'],
            symbol,
            order_id
        ))

        conn.commit()
        conn.close()
        logger.info(f"Trade exit updated in database: {symbol}")

    def log_portfolio_snapshot(self, snapshot_data: Dict[str, Any]):
        """Log a portfolio snapshot to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO portfolio_snapshots (
            timestamp, account_value, buying_power, cash, 
            open_positions_count, open_positions_value, 
            daily_pnl, daily_pnl_percent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            snapshot_data['account_value'],
            snapshot_data['buying_power'],
            snapshot_data['cash'],
            snapshot_data['open_positions_count'],
            snapshot_data['open_positions_value'],
            snapshot_data.get('daily_pnl', 0.0),
            snapshot_data.get('daily_pnl_percent', 0.0)
        ))

        conn.commit()
        conn.close()

    def log_risk_metrics(self, risk_data: Dict[str, Any]):
        """Log risk metrics to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO risk_metrics (
            timestamp, var_95, var_99, max_drawdown, sharpe_ratio,
            sortino_ratio, beta, correlation_spy, avg_win_loss_ratio, win_rate
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            risk_data['var_95'],
            risk_data['var_99'],
            risk_data['max_drawdown'],
            risk_data['sharpe_ratio'],
            risk_data['sortino_ratio'],
            risk_data['beta'],
            risk_data['correlation_spy'],
            risk_data['avg_win_loss_ratio'],
            risk_data['win_rate']
        ))

        conn.commit()
        conn.close()

    def log_system_event(self, level: str, component: str, message: str):
        """Log a system event to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
        INSERT INTO system_logs (
            timestamp, level, component, message
        ) VALUES (?, ?, ?, ?)
        ''', (
            datetime.now(),
            level,
            component,
            message
        ))

        conn.commit()
        conn.close()

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions from the database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('''
        SELECT * FROM trades WHERE status = 'OPEN'
        ''')

        positions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return positions

    def get_closed_positions(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get closed positions from the last N days"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        cursor.execute('''
        SELECT * FROM trades 
        WHERE status = 'CLOSED' AND exit_time >= ?
        ORDER BY exit_time DESC
        ''', (start_date,))

        positions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return positions

    def get_portfolio_snapshots(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get portfolio snapshots from the last N days"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        cursor.execute('''
        SELECT * FROM portfolio_snapshots 
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
        ''', (start_date,))

        snapshots = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return snapshots

    def get_risk_metrics(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get risk metrics from the last N days"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        cursor.execute('''
        SELECT * FROM risk_metrics 
        WHERE timestamp >= ?
        ORDER BY timestamp ASC
        ''', (start_date,))

        metrics = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of trading performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get overall statistics
        cursor.execute('''
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed_trades,
            SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
            AVG(CASE WHEN pnl IS NOT NULL THEN pnl_percent ELSE NULL END) as avg_return_percent,
            SUM(CASE WHEN pnl IS NOT NULL THEN pnl ELSE 0 END) as total_pnl
        FROM trades
        ''')

        overall_stats = dict(zip([column[0] for column in cursor.description], cursor.fetchone()))

        # Get performance by strategy
        cursor.execute('''
        SELECT 
            event_type as strategy,
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            AVG(CASE WHEN pnl IS NOT NULL THEN pnl_percent ELSE NULL END) as avg_return_percent,
            SUM(CASE WHEN pnl IS NOT NULL THEN pnl ELSE 0 END) as total_pnl
        FROM trades
        WHERE status = 'CLOSED'
        GROUP BY event_type
        ''')

        strategy_stats = [dict(zip([column[0] for column in cursor.description], row)) for row in cursor.fetchall()]

        # Get recent performance
        cursor.execute('''
        SELECT 
            date(exit_time) as date,
            SUM(pnl) as daily_pnl
        FROM trades
        WHERE status = 'CLOSED' AND exit_time >= date('now', '-30 days')
        GROUP BY date(exit_time)
        ORDER BY date ASC
        ''')

        daily_pnl = [dict(zip([column[0] for column in cursor.description], row)) for row in cursor.fetchall()]

        conn.close()

        # Calculate win rate and other derived metrics
        if overall_stats['closed_trades'] > 0:
            overall_stats['win_rate'] = (overall_stats['winning_trades'] / overall_stats['closed_trades']) * 100
        else:
            overall_stats['win_rate'] = 0

        # Add strategy win rates
        for strategy in strategy_stats:
            if strategy['total_trades'] > 0:
                strategy['win_rate'] = (strategy['winning_trades'] / strategy['total_trades']) * 100
            else:
                strategy['win_rate'] = 0

        return {
            'overall': overall_stats,
            'by_strategy': strategy_stats,
            'daily_pnl': daily_pnl
        }

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a custom query and return results"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return results
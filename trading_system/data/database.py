"""
PostgreSQL database manager for the trading system.
Optimized for production use with connection pooling.
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Import PostgreSQL libraries
import psycopg2
import psycopg2.extras
from psycopg2 import pool

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager optimized for PostgreSQL"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the database manager

        Args:
            config: PostgreSQL configuration with keys:
                   - postgres_host: PostgreSQL host
                   - postgres_port: PostgreSQL port
                   - postgres_user: PostgreSQL username
                   - postgres_password: PostgreSQL password
                   - postgres_dbname: PostgreSQL database name
                   - pool_min: Minimum connections in pool (default: 1)
                   - pool_max: Maximum connections in pool (default: 10)
        """
        # Handle case when config is None or not provided
        if config is None:
            # Try to load from config.ini
            try:
                from config.settings import ConfigManager
                config_manager = ConfigManager('config.ini')
                db_config = config_manager.get_config().get('database', {})
                self.config = db_config
                logger.info("Loaded database config from config.ini")
            except Exception as e:
                logger.warning(f"Unable to load from config.ini: {e}")
                # Use system username as default
                import getpass
                username = getpass.getuser()
                self.config = {
                    'postgres_host': 'localhost',
                    'postgres_port': 5432,
                    'postgres_user': username,  # Use system username
                    'postgres_password': '',
                    'postgres_dbname': 'trading_system',
                    'pool_min': 1,
                    'pool_max': 10
                }
                logger.info(f"Using default config with username {username}")
        else:
            self.config = config

        # Get PostgreSQL connection parameters
        self.host = self.config.get('postgres_host', 'localhost')
        self.port = self.config.get('postgres_port', 5432)
        self.user = self.config.get('postgres_user', '')
        self.password = self.config.get('postgres_password', '')
        self.dbname = self.config.get('postgres_dbname', 'trading_system')

        # If no user specified, use system username
        if not self.user:
            import getpass
            self.user = getpass.getuser()
            logger.info(f"No username provided, using system user: {self.user}")

        # Connection pool settings
        self.pool_min = int(self.config.get('pool_min', 1))
        self.pool_max = int(self.config.get('pool_max', 10))

        # Initialize connection pool
        try:
            self.connection_pool = pool.ThreadedConnectionPool(
                self.pool_min,
                self.pool_max,
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.dbname
            )

            # Initialize database
            self._initialize_database()

            logger.info(f"PostgreSQL database initialized at {self.host}:{self.port}/{self.dbname}")
        except Exception as e:
            # Error details
            error_msg = f"Database connection error: {e}"
            logger.error(error_msg)

            # Print diagnostic information
            logger.error(f"Connection details: host={self.host}, port={self.port}, user={self.user}, dbname={self.dbname}")

            # Try to check if PostgreSQL is running
            try:
                import subprocess
                result = subprocess.run(['pg_isready'], capture_output=True, text=True)
                logger.error(f"PostgreSQL server status: {result.stdout.strip()}")
            except:
                pass

            # Suggest solutions
            logger.error("Possible solutions:")
            logger.error("1. Check if PostgreSQL is running with 'brew services list'")
            logger.error("2. Create the user if it doesn't exist: createuser -s postgres")
            logger.error("3. Update config.ini with your system username")

            raise

    def _get_connection(self):
        """Get a connection from the pool"""
        return self.connection_pool.getconn()

    def _release_connection(self, conn):
        """Release a connection back to the pool"""
        self.connection_pool.putconn(conn)

    def _initialize_database(self):
        """Create database tables if they don't exist"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Create trades table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
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
                id SERIAL PRIMARY KEY,
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
                id SERIAL PRIMARY KEY,
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
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                level TEXT NOT NULL,
                component TEXT NOT NULL,
                message TEXT NOT NULL
            )
            ''')

            # Create indexes for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS trades_symbol_idx ON trades(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS trades_status_idx ON trades(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS trades_entry_time_idx ON trades(entry_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS snapshots_timestamp_idx ON portfolio_snapshots(timestamp)')

            conn.commit()
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._release_connection(conn)

    def log_trade(self, trade_data: Dict[str, Any]):
        """Log a trade to the database"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Extract option data
            option = trade_data.get('option', {})
            event = trade_data.get('event', {})

            # Insert the trade
            cursor.execute('''
            INSERT INTO trades (
                symbol, underlying, trade_type, entry_price, quantity, entry_time,
                expiration, strike_price, days_to_expiry, event_type, event_date,
                expected_roi, profit_probability, sharpe_ratio, status, order_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                option.get('symbol', ''),
                event.get('symbol', ''),
                option.get('option_type', ''),
                option.get('price', 0.0),
                trade_data.get('contracts', 0),
                datetime.now(),
                option.get('expiration', ''),
                option.get('strike', 0.0),
                option.get('days_to_expiry', 0),
                event.get('event_type', ''),
                event.get('event_date', ''),
                trade_data.get('expected_roi', 0.0),
                trade_data.get('profit_probability', 0.0),
                trade_data.get('sharpe_ratio', 0.0),
                'OPEN',
                trade_data.get('order_id', f"order-{hash(str(trade_data))}")
            ))

            conn.commit()
            logger.info(f"Trade logged to database: {option.get('symbol', '')}")
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self._release_connection(conn)

    def update_trade_exit(self, symbol: str, order_id: str, exit_data: Dict[str, Any]):
        """Update a trade with exit information"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Update the trade
            cursor.execute('''
            UPDATE trades SET
                exit_price = %s,
                exit_time = %s,
                status = %s,
                pnl = %s,
                pnl_percent = %s,
                exit_reason = %s
            WHERE symbol = %s AND order_id = %s AND status = 'OPEN'
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
            logger.info(f"Trade exit updated in database: {symbol}")
        except Exception as e:
            logger.error(f"Error updating trade exit: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self._release_connection(conn)

    def log_portfolio_snapshot(self, snapshot_data: Dict[str, Any]):
        """Log a portfolio snapshot to the database"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Insert the snapshot
            cursor.execute('''
            INSERT INTO portfolio_snapshots (
                timestamp, account_value, buying_power, cash, 
                open_positions_count, open_positions_value, 
                daily_pnl, daily_pnl_percent
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
        except Exception as e:
            logger.error(f"Error logging portfolio snapshot: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self._release_connection(conn)

    def log_risk_metrics(self, risk_data: Dict[str, Any]):
        """Log risk metrics to the database"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Insert the risk metrics
            cursor.execute('''
            INSERT INTO risk_metrics (
                timestamp, var_95, var_99, max_drawdown, sharpe_ratio,
                sortino_ratio, beta, correlation_spy, avg_win_loss_ratio, win_rate
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
        except Exception as e:
            logger.error(f"Error logging risk metrics: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self._release_connection(conn)

    def log_system_event(self, level: str, component: str, message: str):
        """Log a system event to the database"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Insert the system log
            cursor.execute('''
            INSERT INTO system_logs (
                timestamp, level, component, message
            ) VALUES (%s, %s, %s, %s)
            ''', (
                datetime.now(),
                level,
                component,
                message
            ))

            conn.commit()
        except Exception as e:
            logger.error(f"Error logging system event: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                self._release_connection(conn)

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions from the database"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # Get open positions
            cursor.execute('SELECT * FROM trades WHERE status = %s', ('OPEN',))
            positions = [dict(row) for row in cursor.fetchall()]
            return positions
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []
        finally:
            if conn:
                self._release_connection(conn)

    def get_closed_positions(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get closed positions from the last N days"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # Calculate start date
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            # Get closed positions
            cursor.execute('''
            SELECT * FROM trades 
            WHERE status = %s AND exit_time >= %s
            ORDER BY exit_time DESC
            ''', ('CLOSED', start_date))

            positions = [dict(row) for row in cursor.fetchall()]
            return positions
        except Exception as e:
            logger.error(f"Error getting closed positions: {e}")
            return []
        finally:
            if conn:
                self._release_connection(conn)

    def get_portfolio_snapshots(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get portfolio snapshots from the last N days"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # Calculate start date
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            # Get portfolio snapshots
            cursor.execute('''
            SELECT * FROM portfolio_snapshots 
            WHERE timestamp >= %s
            ORDER BY timestamp ASC
            ''', (start_date,))

            snapshots = [dict(row) for row in cursor.fetchall()]
            return snapshots
        except Exception as e:
            logger.error(f"Error getting portfolio snapshots: {e}")
            return []
        finally:
            if conn:
                self._release_connection(conn)

    def get_risk_metrics(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get risk metrics from the last N days"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # Calculate start date
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            # Get risk metrics
            cursor.execute('''
            SELECT * FROM risk_metrics 
            WHERE timestamp >= %s
            ORDER BY timestamp ASC
            ''', (start_date,))

            metrics = [dict(row) for row in cursor.fetchall()]
            return metrics
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return []
        finally:
            if conn:
                self._release_connection(conn)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of trading performance"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # Get overall statistics
            cursor.execute('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed_trades,
                SUM(CASE WHEN status = 'OPEN' THEN 1 ELSE 0 END) as open_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                AVG(CASE WHEN pnl IS NOT NULL THEN pnl_percent ELSE NULL END) as avg_return_percent,
                COALESCE(SUM(CASE WHEN pnl IS NOT NULL THEN pnl ELSE 0 END), 0) as total_pnl
            FROM trades
            ''')

            overall_stats = dict(cursor.fetchone())

            # Get performance by strategy
            cursor.execute('''
            SELECT 
                event_type as strategy,
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                AVG(CASE WHEN pnl IS NOT NULL THEN pnl_percent ELSE NULL END) as avg_return_percent,
                COALESCE(SUM(CASE WHEN pnl IS NOT NULL THEN pnl ELSE 0 END), 0) as total_pnl
            FROM trades
            WHERE status = 'CLOSED'
            GROUP BY event_type
            ''')

            strategy_stats = [dict(row) for row in cursor.fetchall()]

            # Get recent performance
            cursor.execute('''
            SELECT 
                DATE(exit_time) as date,
                SUM(pnl) as daily_pnl
            FROM trades
            WHERE status = 'CLOSED' AND exit_time >= CURRENT_DATE - INTERVAL '30 days'
            GROUP BY DATE(exit_time)
            ORDER BY date ASC
            ''')

            daily_pnl = [dict(row) for row in cursor.fetchall()]

            # Calculate win rate and other derived metrics
            if overall_stats['closed_trades'] is not None and overall_stats['closed_trades'] > 0:
                overall_stats['win_rate'] = (overall_stats['winning_trades'] / overall_stats['closed_trades']) * 100
            else:
                overall_stats['win_rate'] = 0

            # Add strategy win rates
            for strategy in strategy_stats:
                if strategy['total_trades'] is not None and strategy['total_trades'] > 0:
                    strategy['win_rate'] = (strategy['winning_trades'] / strategy['total_trades']) * 100
                else:
                    strategy['win_rate'] = 0

            # Return complete summary
            return {
                'overall': overall_stats,
                'by_strategy': strategy_stats,
                'daily_pnl': daily_pnl
            }
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            # Return a minimal valid structure in case of error
            return {
                'overall': {
                    'total_trades': 0,
                    'closed_trades': 0,
                    'open_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'avg_return_percent': 0,
                    'total_pnl': 0,
                    'win_rate': 0
                },
                'by_strategy': [],
                'daily_pnl': []
            }
        finally:
            if conn:
                self._release_connection(conn)

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute a custom query and return results"""
        conn = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # Execute the query
            cursor.execute(query, params)

            # Get results
            results = [dict(row) for row in cursor.fetchall()]
            return results
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []
        finally:
            if conn:
                self._release_connection(conn)

    def import_from_sqlite(self, sqlite_path: str):
        """
        Import data from SQLite database

        Args:
            sqlite_path: Path to SQLite database file
        """
        try:
            import sqlite3

            # Connect to SQLite
            sqlite_conn = sqlite3.connect(sqlite_path)
            sqlite_conn.row_factory = sqlite3.Row

            # Tables to import
            tables = ['trades', 'portfolio_snapshots', 'risk_metrics', 'system_logs']

            # Get PostgreSQL connection
            pg_conn = self._get_connection()
            pg_cursor = pg_conn.cursor()

            # Process each table
            for table in tables:
                logger.info(f"Importing {table}...")

                # Get SQLite data
                sqlite_cursor = sqlite_conn.cursor()
                sqlite_cursor.execute(f"SELECT * FROM {table}")
                rows = sqlite_cursor.fetchall()

                if not rows:
                    logger.info(f"No data in {table} to import")
                    continue

                # Get column names
                columns = [column[0] for column in sqlite_cursor.description]

                # Create INSERT statement for PostgreSQL
                placeholders = ', '.join(['%s'] * len(columns))
                columns_str = ', '.join(columns)
                insert_sql = f"INSERT INTO {table} ({columns_str}) VALUES ({placeholders})"

                # Import in batches
                batch_size = 1000
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i+batch_size]
                    batch_data = [tuple(row) for row in batch]
                    pg_cursor.executemany(insert_sql, batch_data)
                    pg_conn.commit()

                logger.info(f"Imported {len(rows)} rows into {table}")

            logger.info("Import completed successfully")

        except Exception as e:
            logger.error(f"Import error: {e}")
            raise
        finally:
            if 'sqlite_conn' in locals() and sqlite_conn:
                sqlite_conn.close()
            if 'pg_conn' in locals() and pg_conn:
                self._release_connection(pg_conn)
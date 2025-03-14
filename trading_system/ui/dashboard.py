"""
Streamlit-based web dashboard for the trading system.
Provides visual monitoring and interactive control.
"""

import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any
from utils.sentiment_dashboard import add_sentiment_dashboard


logger = logging.getLogger(__name__)


def create_dashboard(trading_system):
    """Create a Streamlit dashboard for the trading system"""
    st.set_page_config(
        page_title="Algorithmic Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar controls
    st.sidebar.title("Trading Controls")

    # Trading system status
    status = trading_system.get_system_status()

    # Status indicators
    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("System", "Active" if status['system_ready'] else "Inactive")
    col2.metric("Trading", "Active" if status['trading_active'] else "Inactive")
    col3.metric("Market", "Open" if status['market_open'] else "Closed")

    # Trading controls - With session state persistence
    if 'trading_active' not in st.session_state:
        st.session_state.trading_active = status['trading_active']

    if st.session_state.trading_active:
        if st.sidebar.button("Stop Trading", key="stop_trading"):
            trading_system.stop_trading("User requested stop via dashboard")
            st.session_state.trading_active = False
            st.sidebar.success("Trading stopped successfully")
    else:
        if st.sidebar.button("Start Trading", key="start_trading"):
            trading_system.start_trading()
            st.session_state.trading_active = True
            st.sidebar.success("Trading started successfully")

    # Display the current state (useful for debugging)
    st.sidebar.text(f"Session trading state: {'Active' if st.session_state.trading_active else 'Inactive'}")


    # Manual actions
    st.sidebar.subheader("Manual Actions")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Trading Cycle", key="trading_cycle"):
        trading_system.trading_cycle()
        st.sidebar.success("Trading cycle executed")

    if col2.button("Monitor Positions", key="monitor_positions"):
        trading_system.monitor_positions()
        st.sidebar.success("Positions monitored")

    if st.sidebar.button("Generate Report", key="generate_report"):
        trading_system.generate_daily_report()
        st.sidebar.success("Daily report generated")

    # ADD YOUR CODE HERE - right after the existing buttons
    if st.sidebar.button("Test API Connections", key="test_apis"):
        with st.spinner("Testing API connections..."):
            results = trading_system.strategy.test_api_connections()

        # Display results in a nice table
        st.subheader("API Connection Test Results")
        for api, result in results.items():
            if result['status'] == 'success':
                st.success(f"**{api.upper()}**: {result.get('details', 'Connected')}")
            else:
                st.error(f"**{api.upper()}**: {result.get('error', 'Failed to connect')}")

    # Dashboard selection
    st.sidebar.subheader("Dashboard Views")
    dashboard_view = st.sidebar.radio(
        "Select Dashboard",
        ["Overview", "Positions", "Performance", "Risk Analysis", "Sentiment Analysis", "System Status"]
    )

    # Main content area
    st.title("Algorithmic Trading Platform")

    # Show selected dashboard
    if dashboard_view == "Overview":
        show_overview_dashboard(trading_system)
    elif dashboard_view == "Positions":
        show_positions_dashboard(trading_system)
    elif dashboard_view == "Performance":
        show_performance_dashboard(trading_system)
    elif dashboard_view == "Risk Analysis":
        show_risk_dashboard(trading_system)
    elif dashboard_view == "System Status":
        show_system_dashboard(trading_system)
    elif dashboard_view == "Sentiment Analysis":
        add_sentiment_dashboard(trading_system)


def show_overview_dashboard(trading_system):
    """Show overview dashboard"""
    st.header("Trading Dashboard Overview")

    # Get data
    status = trading_system.get_system_status()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Open Positions", status['open_positions'])
    col2.metric("Daily P&L", f"${status['daily_pnl']:.2f}")
    col3.metric("Trades Today", status['daily_stats']['trades_executed'])
    col4.metric("Win Rate", f"{status['risk_metrics'].get('win_rate', 0):.1f}%" if 'risk_metrics' in status and status[
        'risk_metrics'] else "N/A")

    # Split the screen
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Portfolio Performance")

        # Get equity curve data
        performance = trading_system.analytics.generate_performance_summary()

        if 'equity_curve' in performance and performance['equity_curve']:
            equity_df = pd.DataFrame(performance['equity_curve'])
            equity_df['date'] = pd.to_datetime(equity_df['date'])

            fig = px.line(
                equity_df,
                x='date',
                y='value',
                title='Portfolio Equity Curve',
                labels={'date': 'Date', 'value': 'Portfolio Value ($)'}
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to display equity curve")

    with col2:
        st.subheader("Recent Activity")

        # Get recent trades
        recent_trades = trading_system.db_manager.get_closed_positions(days_back=7)

        if recent_trades:
            for trade in recent_trades[:5]:  # Show latest 5 trades
                with st.container():
                    col1, col2 = st.columns([1, 1])
                    col1.write(f"**{trade['symbol']}** ({trade['trade_type']})")

                    if trade['pnl'] is not None:
                        color = "green" if trade['pnl'] > 0 else "red"
                        col2.markdown(
                            f"<span style='color:{color};'>${trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%)</span>",
                            unsafe_allow_html=True)

                    st.text(f"Exit: {trade['exit_reason']} on {trade['exit_time']}")
                    st.divider()
        else:
            st.info("No recent trades to display")

    # Strategy performance
    st.subheader("Strategy Performance")

    trade_analysis = trading_system.analytics.generate_trade_analysis()

    if 'strategy_performance' in trade_analysis and trade_analysis['strategy_performance']:
        strategy_data = []

        for strategy, metrics in trade_analysis['strategy_performance'].items():
            strategy_data.append({
                'strategy': strategy,
                'win_rate': metrics['win_rate'],
                'avg_return': metrics['avg_return'],
                'trade_count': metrics['trade_count'],
                'total_pnl': metrics['total_pnl']
            })

        strategy_df = pd.DataFrame(strategy_data)

        fig = px.bar(
            strategy_df,
            x='strategy',
            y='total_pnl',
            color='win_rate',
            text='trade_count',
            title='Strategy P&L and Win Rate',
            labels={
                'strategy': 'Strategy',
                'total_pnl': 'Total P&L ($)',
                'win_rate': 'Win Rate (%)'
            },
            color_continuous_scale='RdYlGn'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to display strategy performance")


def show_positions_dashboard(trading_system):
    """Show positions dashboard"""
    st.header("Trading Positions")

    # Get open positions
    open_positions = trading_system.db_manager.get_open_positions()

    st.subheader(f"Open Positions ({len(open_positions)})")

    if open_positions:
        # Create DataFrame
        pos_df = pd.DataFrame(open_positions)

        # Format columns
        cols_to_display = ['symbol', 'trade_type', 'strike_price', 'expiration',
                           'entry_price', 'quantity', 'event_type', 'event_date', 'entry_time']

        # Calculate days held
        pos_df['days_held'] = (datetime.now() - pd.to_datetime(pos_df['entry_time'])).dt.days

        # Add days to expiry (if available)
        if 'days_to_expiry' in pos_df.columns:
            pos_df['days_remaining'] = pos_df['days_to_expiry'] - pos_df['days_held']

        # Format currency columns
        for col in ['entry_price', 'strike_price']:
            if col in pos_df.columns:
                pos_df[col] = pos_df[col].apply(lambda x: f"${x:.2f}")

        # Display the table
        st.dataframe(pos_df[cols_to_display + ['days_held']], use_container_width=True)

        # Calculate position allocation
        st.subheader("Position Allocation")

        # Group by underlying and strategy
        underlying_allocation = pos_df.groupby('underlying')['quantity'].sum()
        strategy_allocation = pos_df.groupby('event_type')['quantity'].sum()

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                names=underlying_allocation.index,
                values=underlying_allocation.values,
                title='Allocation by Underlying'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.pie(
                names=strategy_allocation.index,
                values=strategy_allocation.values,
                title='Allocation by Strategy'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No open positions")

    # Show recent closed positions
    st.subheader("Recent Closed Positions")

    closed_positions = trading_system.db_manager.get_closed_positions(days_back=30)

    if closed_positions:
        # Create DataFrame
        closed_df = pd.DataFrame(closed_positions)

        # Format columns
        cols_to_display = ['symbol', 'trade_type', 'entry_price', 'exit_price',
                           'pnl', 'pnl_percent', 'exit_time', 'exit_reason']

        # Format currency columns
        for col in ['entry_price', 'exit_price', 'pnl']:
            if col in closed_df.columns:
                closed_df[col] = closed_df[col].apply(lambda x: f"${x:.2f}" if x is not None else "-")

        if 'pnl_percent' in closed_df.columns:
            closed_df['pnl_percent'] = closed_df['pnl_percent'].apply(lambda x: f"{x:.2f}%" if x is not None else "-")

        # Display the table
        st.dataframe(closed_df[cols_to_display], use_container_width=True)

        # Show P&L distribution
        st.subheader("P&L Distribution")

        # Extract numeric P&L for histogram
        pnl_values = [trade['pnl'] for trade in closed_positions if trade['pnl'] is not None]

        if pnl_values:
            fig = px.histogram(
                x=pnl_values,
                nbins=20,
                title='P&L Distribution',
                labels={'x': 'P&L ($)'},
                color_discrete_sequence=['blue']
            )

            fig.add_vline(x=0, line_dash="dash", line_color="red")

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed positions in the last 30 days")


def show_performance_dashboard(trading_system):
    """Show performance dashboard"""
    st.header("Trading Performance Analysis")

    # Get performance data
    performance = trading_system.analytics.generate_performance_summary()
    trade_analysis = trading_system.analytics.generate_trade_analysis()

    # Summary metrics
    if 'metrics' in performance:
        metrics = performance['metrics']

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Return", f"{metrics.get('total_return', 0):.2f}%")
        col2.metric("Annualized Return", f"{metrics.get('annualized_return', 0):.2f}%")
        col3.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.2f}%")
        col4.metric("Win Rate", f"{metrics.get('win_rate', 0):.2f}%")

    # Equity curve
    st.subheader("Equity Curve")

    if 'equity_curve' in performance and performance['equity_curve']:
        equity_df = pd.DataFrame(performance['equity_curve'])
        equity_df['date'] = pd.to_datetime(equity_df['date'])

        fig = px.line(
            equity_df,
            x='date',
            y='value',
            title='Portfolio Equity Curve',
            labels={'date': 'Date', 'value': 'Portfolio Value ($)'}
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to display equity curve")

    # Performance metrics by strategy
    st.subheader("Performance by Strategy")

    if 'by_strategy' in performance and performance['by_strategy']:
        strategy_df = pd.DataFrame(performance['by_strategy'])

        if not strategy_df.empty:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=strategy_df['strategy'],
                y=strategy_df['win_rate'],
                name='Win Rate (%)',
                yaxis='y',
                offsetgroup=1,
                marker_color='green'
            ))

            fig.add_trace(go.Bar(
                x=strategy_df['strategy'],
                y=strategy_df['avg_return_percent'],
                name='Avg Return (%)',
                yaxis='y2',
                offsetgroup=2,
                marker_color='blue'
            ))

            fig.update_layout(
                title='Strategy Performance Metrics',
                xaxis=dict(title='Strategy'),
                yaxis=dict(title='Win Rate (%)', side='left', showgrid=False),
                yaxis2=dict(title='Avg Return (%)', side='right', overlaying='y', showgrid=False),
                barmode='group',
                legend=dict(x=0.01, y=0.99)
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to display strategy performance")
    else:
        st.info("Not enough data to display strategy performance")

    # Trade timeline
    st.subheader("Trade Timeline")

    if 'trade_timeline' in trade_analysis and trade_analysis['trade_timeline']:
        timeline_df = pd.DataFrame(trade_analysis['trade_timeline'])
        timeline_df['date'] = pd.to_datetime(timeline_df['date'])

        fig = px.scatter(
            timeline_df,
            x='date',
            y='pnl',
            size='duration_days',
            color='strategy',
            title='Trade P&L Timeline',
            hover_data=['symbol', 'pnl_percent'],
            labels={
                'date': 'Date',
                'pnl': 'P&L ($)',
                'duration_days': 'Duration (days)',
                'strategy': 'Strategy'
            }
        )

        fig.add_hline(y=0, line_dash="dash", line_color="red")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to display trade timeline")

    # Performance by underlying
    st.subheader("Performance by Underlying")

    if 'performance_by_underlying' in trade_analysis and trade_analysis['performance_by_underlying']:
        underlying_data = []

        for underlying, perf in trade_analysis['performance_by_underlying'].items():
            underlying_data.append({
                'underlying': underlying,
                'win_rate': perf['win_rate'],
                'avg_return': perf['avg_return'],
                'trade_count': perf['trade_count'],
                'total_pnl': perf['total_pnl']
            })

        underlying_df = pd.DataFrame(underlying_data)

        if not underlying_df.empty:
            fig = px.scatter(
                underlying_df,
                x='win_rate',
                y='avg_return',
                size='trade_count',
                color='total_pnl',
                hover_name='underlying',
                title='Underlying Performance Analysis',
                labels={
                    'win_rate': 'Win Rate (%)',
                    'avg_return': 'Average Return (%)',
                    'trade_count': 'Number of Trades',
                    'total_pnl': 'Total P&L ($)'
                },
                color_continuous_scale='RdYlGn'
            )

            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.add_vline(x=50, line_dash="dash", line_color="gray")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to display underlying performance")
    else:
        st.info("Not enough data to display underlying performance")


def show_risk_dashboard(trading_system):
    """Show risk analysis dashboard"""
    st.header("Risk Analysis Dashboard")

    # Get risk data
    risk_report = trading_system.analytics.generate_risk_report()

    # Risk metrics
    if 'latest_metrics' in risk_report and risk_report['latest_metrics']:
        metrics = risk_report['latest_metrics']

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("VaR (95%)", f"{metrics.get('var_95', 0) * 100:.2f}%")
        col2.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
        col3.metric("Sortino Ratio", f"{metrics.get('sortino_ratio', 0):.2f}")
        col4.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0) * 100:.2f}%")

    # Risk metrics over time
    st.subheader("Risk Metrics Over Time")

    if 'time_series' in risk_report and risk_report['time_series']:
        ts_df = pd.DataFrame({
            'timestamp': pd.to_datetime(risk_report['time_series']['timestamps']),
            'var_95': risk_report['time_series']['var_95'],
            'sharpe_ratio': risk_report['time_series']['sharpe_ratio'],
            'max_drawdown': risk_report['time_series']['max_drawdown']
        })

        col1, col2 = st.columns(2)

        with col1:
            fig = px.line(
                ts_df,
                x='timestamp',
                y='var_95',
                title='Value-at-Risk (95%) Over Time',
                labels={
                    'timestamp': 'Date',
                    'var_95': 'VaR 95% (%)'
                }
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.line(
                ts_df,
                x='timestamp',
                y='sharpe_ratio',
                title='Sharpe Ratio Over Time',
                labels={
                    'timestamp': 'Date',
                    'sharpe_ratio': 'Sharpe Ratio'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to display risk metrics over time")

    # Portfolio allocation
    st.subheader("Portfolio Risk Allocation")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Strategy Exposures")

        if 'strategy_exposures' in risk_report and risk_report['strategy_exposures']:
            strategy_data = [
                {'strategy': strategy, 'exposure': amount}
                for strategy, amount in risk_report['strategy_exposures'].items()
            ]

            strategy_df = pd.DataFrame(strategy_data)

            if not strategy_df.empty:
                fig = px.pie(
                    strategy_df,
                    names='strategy',
                    values='exposure',
                    title='Exposure by Strategy',
                    labels={'exposure': 'Exposure ($)'}
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strategy exposure data available")
        else:
            st.info("No strategy exposure data available")

    with col2:
        st.subheader("Underlying Exposures")

        if 'underlying_exposures' in risk_report and risk_report['underlying_exposures']:
            underlying_data = [
                {'underlying': underlying, 'exposure': amount}
                for underlying, amount in risk_report['underlying_exposures'].items()
            ]

            underlying_df = pd.DataFrame(underlying_data)

            if not underlying_df.empty:
                fig = px.pie(
                    underlying_df,
                    names='underlying',
                    values='exposure',
                    title='Exposure by Underlying',
                    labels={'exposure': 'Exposure ($)'}
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No underlying exposure data available")
        else:
            st.info("No underlying exposure data available")

    # Treemap of all exposures
    st.subheader("Detailed Risk Allocation")

    if ('strategy_exposures' in risk_report and risk_report['strategy_exposures'] and
            'underlying_exposures' in risk_report and risk_report['underlying_exposures']):

        # Combine strategy and underlying exposures
        exposures = [
            {'category': 'Strategy', 'name': strategy, 'value': amount}
            for strategy, amount in risk_report['strategy_exposures'].items()
        ]

        exposures.extend([
            {'category': 'Underlying', 'name': underlying, 'value': amount}
            for underlying, amount in risk_report['underlying_exposures'].items()
        ])

        exposure_df = pd.DataFrame(exposures)

        if not exposure_df.empty:
            fig = px.treemap(
                exposure_df,
                path=['category', 'name'],
                values='value',
                title='Portfolio Risk Allocation',
                color='value',
                color_continuous_scale='Viridis'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to display detailed risk allocation")
    else:
        st.info("Not enough data to display detailed risk allocation")


def show_system_dashboard(trading_system):
    """Show system status dashboard"""
    st.header("System Status Dashboard")

    # Get system status
    status = trading_system.get_system_status()

    # System state
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Trading Status", "Active" if status['trading_active'] else "Inactive")
    col2.metric("Market Status", "Open" if status['market_open'] else "Closed")
    col3.metric("Open Positions", status['open_positions'])
    col4.metric("Last Run", status['last_run_time'].strftime("%H:%M:%S") if status['last_run_time'] else "Never")

    # Daily stats
    st.subheader("Daily Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Trades Executed", status['daily_stats']['trades_executed'])
    col2.metric("Trades Exited", status['daily_stats']['trades_exited'])
    col3.metric("Opportunities Analyzed", status['daily_stats']['opportunities_analyzed'])
    col4.metric("Daily P&L", f"${status['daily_stats']['daily_pnl']:.2f}")

    # Scheduled jobs
    st.subheader("Scheduled Jobs")

    job_data = [
        {"Job ID": job_id, "Next Run Time": next_run}
        for job_id, next_run in status['scheduler_status']['jobs'].items()
    ]

    if job_data:
        st.table(pd.DataFrame(job_data))
    else:
        st.info("No scheduled jobs")

    # System logs
    st.subheader("Recent System Logs")

    logs = trading_system.db_manager.execute_query(
        "SELECT * FROM system_logs ORDER BY timestamp DESC LIMIT 50"
    )

    if logs:
        log_df = pd.DataFrame(logs)
        log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])

        st.dataframe(log_df[['timestamp', 'level', 'component', 'message']], use_container_width=True)
    else:
        st.info("No system logs available")

    # System health
    st.subheader("System Health")

    try:
        import psutil

        # Get system resource usage
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        disk_usage = psutil.disk_usage('/')

        col1, col2, col3 = st.columns(3)

        col1.metric("CPU Usage", f"{cpu_percent}%")
        col2.metric("Memory Usage", f"{memory_info.percent}%")
        col3.metric("Disk Usage", f"{disk_usage.percent}%")
    except ImportError:
        st.warning("System health metrics require the psutil package")
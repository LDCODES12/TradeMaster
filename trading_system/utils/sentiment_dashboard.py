"""
Sentiment Analysis Dashboard Components for Trading System

Adds sentiment analysis visualization to the Streamlit dashboard.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional


def add_sentiment_dashboard(trading_system):
    """
    Add sentiment analysis dashboard to the trading system UI

    This function should be called from the main dashboard file.

    Args:
        trading_system: The main trading system instance
    """
    st.title("Market Sentiment Analysis")

    # Initialize sentiment analyzer if not already done
    if not hasattr(trading_system, 'sentiment_analyzer'):
        polygon_key = trading_system.config.get('polygon', {}).get('api_key', '')
        if polygon_key:
            from trading_system.utils.sentiment_analyzer import NewsSentimentAnalyzer
            trading_system.sentiment_analyzer = NewsSentimentAnalyzer(polygon_key)
        else:
            st.error("Polygon API key not found in configuration. Sentiment analysis is not available.")
            return

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Sentiment Overview", "Symbol Analysis", "Market Trends"])

    with tab1:
        show_sentiment_overview(trading_system)

    with tab2:
        show_symbol_sentiment_analysis(trading_system)

    with tab3:
        show_market_sentiment_trends(trading_system)


def show_sentiment_overview(trading_system):
    """Show sentiment overview for portfolio and watchlist"""
    st.header("Portfolio & Watchlist Sentiment")

    # Get portfolio and watchlist symbols
    symbols = get_portfolio_and_watchlist_symbols(trading_system)

    if not symbols:
        st.info("No symbols found in portfolio or watchlist")
        return

    # Show sentiment analysis in progress indicator
    with st.spinner("Analyzing sentiment for your portfolio..."):
        # Get sentiment for all symbols
        sentiment_scores = trading_system.sentiment_analyzer.get_multi_ticker_sentiment(symbols)

    # Create dataframe for visualization
    sentiment_data = []
    for symbol, score in sentiment_scores.items():
        # Determine sentiment category
        if score > 0.3:
            category = "Very Positive"
        elif score > 0.1:
            category = "Positive"
        elif score > -0.1:
            category = "Neutral"
        elif score > -0.3:
            category = "Negative"
        else:
            category = "Very Negative"

        sentiment_data.append({
            "Symbol": symbol,
            "Sentiment Score": score,
            "Category": category
        })

    df = pd.DataFrame(sentiment_data)

    # Show sentiment distribution chart
    st.subheader("Sentiment Distribution")

    fig = px.bar(
        df,
        x="Symbol",
        y="Sentiment Score",
        color="Category",
        color_discrete_map={
            "Very Positive": "#2E8B57",
            "Positive": "#90EE90",
            "Neutral": "#D3D3D3",
            "Negative": "#FFA07A",
            "Very Negative": "#DC143C"
        },
        title="Sentiment Score by Symbol"
    )

    # Add a horizontal line at y=0
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(df) - 0.5,
        y0=0,
        y1=0,
        line=dict(color="black", width=1, dash="dash")
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Sentiment Score (-1 to +1)",
        legend_title="Sentiment Category"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show sentiment table with details
    st.subheader("Portfolio Sentiment Details")

    # Add in-position column if available
    df_display = df.copy()
    if hasattr(trading_system, 'db_manager'):
        try:
            open_positions = trading_system.db_manager.get_open_positions()
            position_symbols = set([pos.get('underlying', '') for pos in open_positions])
            df_display["In Position"] = df_display["Symbol"].apply(lambda x: "✓" if x in position_symbols else "")
        except Exception as e:
            st.warning(f"Could not retrieve position data: {e}")

    # Format for display
    df_display["Sentiment Score"] = df_display["Sentiment Score"].apply(lambda x: f"{x:.2f}")

    # Show the table
    st.dataframe(df_display, hide_index=True, use_container_width=True)

    # Show highest sentiment opportunities
    st.subheader("Top Sentiment Opportunities")

    # Filter to strongest sentiment (positive or negative)
    top_opportunities = df.copy()
    top_opportunities["Abs_Score"] = top_opportunities["Sentiment Score"].abs()
    top_opportunities = top_opportunities.sort_values("Abs_Score", ascending=False).head(3)

    # Show top opportunities
    for _, row in top_opportunities.iterrows():
        symbol = row["Symbol"]
        score = row["Sentiment Score"]
        category = row["Category"]

        # Create expandable section for each opportunity
        with st.expander(f"{symbol}: {category} ({score:.2f})"):
            # Show recent news for this symbol
            try:
                articles = trading_system.sentiment_analyzer.get_top_articles(
                    symbol,
                    sentiment="positive" if score > 0 else "negative",
                    limit=3
                )

                if articles:
                    for article in articles:
                        st.markdown(f"**{article.get('title', 'No title')}**")
                        st.markdown(f"*{article.get('published_utc', '')}*")
                        st.markdown(article.get('description', 'No description available'))

                        if 'insights' in article:
                            for insight in article.get('insights', []):
                                if insight.get('ticker') == symbol:
                                    st.markdown(f"**Sentiment:** {insight.get('sentiment', '')}")
                                    st.markdown(f"**Reasoning:** {insight.get('sentiment_reasoning', '')}")

                        st.markdown(f"[Read full article]({article.get('article_url', '#')})")
                        st.divider()
                else:
                    st.info(f"No recent news found for {symbol}")

            except Exception as e:
                st.error(f"Error retrieving news: {e}")


def show_symbol_sentiment_analysis(trading_system):
    """Show detailed sentiment analysis for a specific symbol"""
    st.header("Symbol Sentiment Analysis")

    # Get portfolio and watchlist symbols
    symbols = get_portfolio_and_watchlist_symbols(trading_system)

    if not symbols:
        symbols = ["SPY", "AAPL", "MSFT", "GOOGL", "AMZN"]  # Default symbols

    # Allow user to select a symbol
    selected_symbol = st.selectbox("Select Symbol", symbols)

    if selected_symbol:
        # Time range selector
        time_range = st.slider(
            "Analysis Period (Days)",
            min_value=7,
            max_value=90,
            value=30
        )

        # Show sentiment trend
        st.subheader(f"Sentiment Trend for {selected_symbol}")

        with st.spinner(f"Analyzing sentiment trend for {selected_symbol}..."):
            try:
                # Get sentiment trend data
                sentiment_df = trading_system.sentiment_analyzer.get_sentiment_trend(
                    selected_symbol,
                    days_back=time_range
                )

                if sentiment_df.empty:
                    st.info(f"No sentiment data available for {selected_symbol}")
                    return

                # Make sure date is in datetime format
                sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

                # Create a plotly figure
                fig = go.Figure()

                # Add sentiment counts
                fig.add_trace(go.Scatter(
                    x=sentiment_df['date'],
                    y=sentiment_df['positive'],
                    mode='lines',
                    name='Positive',
                    line=dict(color='green', width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=sentiment_df['date'],
                    y=sentiment_df['negative'],
                    mode='lines',
                    name='Negative',
                    line=dict(color='red', width=2)
                ))

                fig.add_trace(go.Scatter(
                    x=sentiment_df['date'],
                    y=sentiment_df['neutral'],
                    mode='lines',
                    name='Neutral',
                    line=dict(color='gray', width=2, dash='dot')
                ))

                # Update layout
                fig.update_layout(
                    title=f"Sentiment Trend for {selected_symbol}",
                    xaxis_title="Date",
                    yaxis_title="Article Count",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show sentiment score trend
                fig2 = go.Figure()

                fig2.add_trace(go.Scatter(
                    x=sentiment_df['date'],
                    y=sentiment_df['score'],
                    mode='lines',
                    name='Sentiment Score',
                    line=dict(color='blue', width=2)
                ))

                # Add a horizontal line at y=0
                fig2.add_shape(
                    type="line",
                    x0=sentiment_df['date'].min(),
                    x1=sentiment_df['date'].max(),
                    y0=0,
                    y1=0,
                    line=dict(color="black", width=1, dash="dash")
                )

                # Update layout
                fig2.update_layout(
                    title=f"Sentiment Score Trend for {selected_symbol}",
                    xaxis_title="Date",
                    yaxis_title="Sentiment Score (-1 to +1)",
                    yaxis=dict(range=[-1, 1])
                )

                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Error calculating sentiment trend: {e}")

        # Show recent news with sentiment
        st.subheader(f"Recent News for {selected_symbol}")

        with st.spinner(f"Retrieving recent news for {selected_symbol}..."):
            try:
                # Get recent news articles
                articles = trading_system.sentiment_analyzer.get_ticker_news(
                    selected_symbol,
                    days_back=time_range,
                    limit=10
                )

                if not articles:
                    st.info(f"No recent news found for {selected_symbol}")
                    return

                # Display recent news with sentiment
                for article in articles[:5]:  # Show top 5 articles
                    # Create a container for the article
                    with st.container():
                        # Title with published date
                        pub_date = article.get('published_utc', '').split('T')[0]
                        st.markdown(f"### {article.get('title', 'No title')}")
                        st.markdown(f"*Published: {pub_date}*")

                        # Description
                        if 'description' in article:
                            st.markdown(article['description'])

                        # Display sentiment insights
                        for insight in article.get('insights', []):
                            if insight.get('ticker') == selected_symbol:
                                sentiment = insight.get('sentiment', '')
                                sentiment_color = {
                                    'positive': 'green',
                                    'negative': 'red',
                                    'neutral': 'gray'
                                }.get(sentiment, 'black')

                                st.markdown(
                                    f"**Sentiment:** <span style='color:{sentiment_color};'>{sentiment.title()}</span>",
                                    unsafe_allow_html=True)
                                st.markdown(f"**Reasoning:** {insight.get('sentiment_reasoning', '')}")

                        # Link to full article
                        st.markdown(f"[Read full article]({article.get('article_url', '#')})")
                        st.divider()

            except Exception as e:
                st.error(f"Error retrieving news: {e}")


def show_market_sentiment_trends(trading_system):
    """Show market-wide sentiment trends"""
    st.header("Market Sentiment Trends")

    # Define major market indices and sectors
    market_indices = {
        "Market Indices": ["SPY", "QQQ", "DIA", "IWM"],
        "Sectors": ["XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE"],
        "Tech Leaders": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
    }

    # Let user choose a category
    category = st.selectbox("Select Category", list(market_indices.keys()))

    if category:
        symbols = market_indices[category]

        # Time range selector
        time_range = st.slider(
            "Analysis Period (Days)",
            min_value=7,
            max_value=30,
            value=7,
            key="market_trend_slider"
        )

        # Show sentiment analysis in progress indicator
        with st.spinner(f"Analyzing sentiment for {category}..."):
            # Get sentiment for all symbols
            sentiment_scores = trading_system.sentiment_analyzer.get_multi_ticker_sentiment(
                symbols,
                days_back=time_range
            )

        # Create dataframe for visualization
        sentiment_data = []
        for symbol, score in sentiment_scores.items():
            # Determine sentiment category
            if score > 0.3:
                category = "Very Positive"
            elif score > 0.1:
                category = "Positive"
            elif score > -0.1:
                category = "Neutral"
            elif score > -0.3:
                category = "Negative"
            else:
                category = "Very Negative"

            sentiment_data.append({
                "Symbol": symbol,
                "Sentiment Score": score,
                "Category": category
            })

        df = pd.DataFrame(sentiment_data)

        # Show sentiment heatmap
        st.subheader(f"Sentiment Heatmap - {category}")

        fig = px.treemap(
            df,
            path=['Category', 'Symbol'],
            values=df['Sentiment Score'].abs(),  # Size by absolute value
            color='Sentiment Score',
            color_continuous_scale=['#DC143C', '#FFA07A', '#D3D3D3', '#90EE90', '#2E8B57'],
            range_color=[-1, 1],
            hover_data=['Sentiment Score']
        )

        fig.update_layout(
            title=f"Sentiment Heatmap - {category}",
            coloraxis_colorbar=dict(
                title="Sentiment",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show average sentiment by symbol
        st.subheader(f"Average Sentiment - {category}")

        # Sort by sentiment score
        df_sorted = df.sort_values("Sentiment Score", ascending=False)

        fig = px.bar(
            df_sorted,
            x="Symbol",
            y="Sentiment Score",
            color="Sentiment Score",
            color_continuous_scale=['#DC143C', '#FFA07A', '#D3D3D3', '#90EE90', '#2E8B57'],
            range_color=[-1, 1],
            title=f"Average Sentiment Score - {category}"
        )

        # Add a horizontal line at y=0
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(df) - 0.5,
            y0=0,
            y1=0,
            line=dict(color="black", width=1, dash="dash")
        )

        fig.update_layout(
            xaxis_title="",
            yaxis_title="Sentiment Score (-1 to +1)"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show detailed table
        st.subheader("Detailed Sentiment Data")

        # Format for display
        df_display = df.copy()
        df_display["Sentiment Score"] = df_display["Sentiment Score"].apply(lambda x: f"{x:.2f}")

        # Show the table
        st.dataframe(df_display, hide_index=True, use_container_width=True)


def get_portfolio_and_watchlist_symbols(trading_system):
    """Get combined list of portfolio and watchlist symbols"""
    symbols = set()

    # Get symbols from open positions
    if hasattr(trading_system, 'db_manager'):
        try:
            open_positions = trading_system.db_manager.get_open_positions()
            for position in open_positions:
                underlying = position.get('underlying', '')
                if underlying:
                    symbols.add(underlying)
        except Exception as e:
            st.warning(f"Could not retrieve position data: {e}")

    # Add symbols from watchlist
    if hasattr(trading_system, 'config') and 'watchlist' in trading_system.config:
        watchlist = trading_system.config['watchlist']
        for symbol in watchlist:
            symbols.add(symbol)

    # If no symbols found, add some defaults
    if not symbols:
        symbols = {"SPY", "AAPL", "MSFT", "GOOGL", "AMZN"}

    return list(symbols)


def register_dashboard_components(trading_system):
    """
    Register sentiment dashboard components with the trading system

    This function should be called during system initialization

    Args:
        trading_system: The main trading system instance
    """
    if not hasattr(trading_system, 'dashboard_views'):
        trading_system.dashboard_views = {}

    # Register the sentiment dashboard
    trading_system.dashboard_views['Sentiment Analysis'] = lambda: add_sentiment_dashboard(trading_system)
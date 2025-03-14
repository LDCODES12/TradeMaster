"""
News Sentiment Analysis for Trading Strategy Enhancement

Provides sentiment analysis of stock news from Polygon.io API
with efficient caching and rate limiting.
"""

import os
import json
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Polygon imports
from polygon import RESTClient

logger = logging.getLogger(__name__)


class SentimentCache:
    """Efficient caching for sentiment data"""

    def __init__(self, cache_dir="data/sentiment_cache"):
        """Initialize the cache"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.memory_cache = {}

    def get(self, key: str, max_age_seconds: int = 86400) -> Optional[Dict[str, Any]]:
        """
        Get data from cache if it exists and is not too old

        Args:
            key: Cache key
            max_age_seconds: Maximum age in seconds

        Returns:
            Cached data or None if not found or expired
        """
        # Check memory cache first
        if key in self.memory_cache:
            cached_time, cached_data = self.memory_cache[key]
            if time.time() - cached_time < max_age_seconds:
                return cached_data

        # Check disk cache
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            if os.path.exists(cache_file):
                file_modified = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if (datetime.now() - file_modified).total_seconds() < max_age_seconds:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        # Update memory cache
                        self.memory_cache[key] = (time.time(), data)
                        return data
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")

        return None

    def set(self, key: str, data: Dict[str, Any]):
        """
        Store data in cache

        Args:
            key: Cache key
            data: Data to cache
        """
        # Update memory cache
        self.memory_cache[key] = (time.time(), data)

        # Update disk cache
        try:
            cache_file = os.path.join(self.cache_dir, f"{key}.json")
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")


class NewsSentimentAnalyzer:
    """
    News sentiment analysis using Polygon.io API.

    This class provides sentiment analysis for trading strategies,
    optimized for the Polygon free tier API limits.
    """

    def __init__(self, api_key: str, calls_per_minute: int = 5):
        """
        Initialize the sentiment analyzer

        Args:
            api_key: Polygon.io API key
            calls_per_minute: API rate limit (5 for free tier)
        """
        self.api_key = api_key
        self.client = RESTClient(api_key)
        self.calls_per_minute = calls_per_minute
        self.calls_interval = 60.0 / calls_per_minute if calls_per_minute > 0 else 0
        self.last_call_time = 0

        # Set up cache
        self.cache = SentimentCache()

        # Default TTLs
        self.ticker_news_ttl = 12 * 60 * 60  # 12 hours for ticker news
        self.sentiment_trend_ttl = 24 * 60 * 60  # 24 hours for trends

        logger.info(f"News sentiment analyzer initialized with {calls_per_minute} calls/minute limit")

    def _rate_limited_call(self, func, *args, **kwargs):
        """Execute API call with rate limiting"""
        if self.calls_per_minute <= 0:
            return func(*args, **kwargs)

        now = time.time()
        time_since_last = now - self.last_call_time

        if time_since_last < self.calls_interval:
            sleep_time = self.calls_interval - time_since_last
            time.sleep(sleep_time)

        self.last_call_time = time.time()
        return func(*args, **kwargs)

    def get_ticker_news(self, ticker: str, days_back: int = 7, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent news for a ticker

        Args:
            ticker: Stock symbol
            days_back: Days of history to retrieve
            limit: Maximum number of articles

        Returns:
            List of news articles with sentiment
        """
        # Normalize ticker
        ticker = ticker.upper()

        # Create cache key
        cache_key = f"{ticker}_news_{days_back}_{limit}"

        # Check cache first
        cached_data = self.cache.get(cache_key, self.ticker_news_ttl)
        if cached_data:
            logger.info(f"Using cached news for {ticker}")
            return cached_data

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        try:
            logger.info(f"Fetching news for {ticker} from {start_date.date()} to {end_date.date()}")

            # Make API call with rate limiting
            news_articles = self._rate_limited_call(
                self.client.list_ticker_news,
                ticker,
                published_utc=f"gte:{start_date.strftime('%Y-%m-%d')}",
                order="desc",
                limit=limit
            )

            # Convert to serializable format
            news_data = []
            for article in news_articles:
                article_data = {
                    'id': article.id,
                    'title': article.title,
                    'published_utc': article.published_utc,
                    'article_url': article.article_url,
                    'tickers': article.tickers,
                    'insights': []
                }

                # Add description if available
                if hasattr(article, 'description'):
                    article_data['description'] = article.description

                # Add insights if available
                if hasattr(article, 'insights') and article.insights:
                    for insight in article.insights:
                        insight_data = {
                            'ticker': insight.ticker,
                            'sentiment': insight.sentiment,
                            'sentiment_reasoning': insight.sentiment_reasoning
                        }
                        article_data['insights'].append(insight_data)

                news_data.append(article_data)

            # Cache the results
            self.cache.set(cache_key, news_data)

            logger.info(f"Retrieved {len(news_data)} news articles for {ticker}")
            return news_data

        except Exception as e:
            logger.error(f"Error getting news for {ticker}: {e}")
            return []

    def get_sentiment_counts(self, ticker: str, days_back: int = 7) -> Dict[str, int]:
        """
        Get aggregated sentiment counts for a ticker

        Args:
            ticker: Stock symbol
            days_back: Days of history to analyze

        Returns:
            Dict with counts for each sentiment
        """
        # Normalize ticker
        ticker = ticker.upper()

        # Get news articles
        articles = self.get_ticker_news(ticker, days_back)

        # Count sentiments
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}

        for article in articles:
            for insight in article.get('insights', []):
                if insight.get('ticker') == ticker and insight.get('sentiment') in sentiment_counts:
                    sentiment_counts[insight.get('sentiment')] += 1

        return sentiment_counts

    def get_sentiment_score(self, ticker: str, days_back: int = 7) -> float:
        """
        Calculate a normalized sentiment score

        Args:
            ticker: Stock symbol
            days_back: Days of history to analyze

        Returns:
            Score from -1.0 (most negative) to 1.0 (most positive)
        """
        sentiment_counts = self.get_sentiment_counts(ticker, days_back)

        # Calculate weighted score
        positive = sentiment_counts["positive"]
        negative = sentiment_counts["negative"]
        total = sum(sentiment_counts.values())

        if total == 0:
            return 0.0

        # Calculate score as weighted average
        score = (positive - negative) / total

        return score

    def get_sentiment_trend(self, ticker: str, days_back: int = 30) -> pd.DataFrame:
        """
        Get daily sentiment trend for a ticker

        Args:
            ticker: Stock symbol
            days_back: Days of history to analyze

        Returns:
            DataFrame with daily sentiment counts
        """
        # Normalize ticker
        ticker = ticker.upper()

        # Create cache key
        cache_key = f"{ticker}_sentiment_trend_{days_back}"

        # Check cache first
        cached_data = self.cache.get(cache_key, self.sentiment_trend_ttl)
        if cached_data:
            logger.info(f"Using cached sentiment trend for {ticker}")
            # Convert back to DataFrame
            return pd.DataFrame(cached_data)

        try:
            logger.info(f"Calculating sentiment trend for {ticker} over {days_back} days")

            # Define date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Get all news for the period in one call to minimize API usage
            all_news = self.get_ticker_news(ticker, days_back, limit=1000)

            # Group by day
            sentiment_trend = []
            for day in pd.date_range(start=start_date, end=end_date):
                day_str = day.strftime("%Y-%m-%d")

                # Initialize counts
                daily_sentiment = {
                    'date': day_str,
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0,
                    'total': 0
                }

                # Count articles for this day
                for article in all_news:
                    pub_date = article.get('published_utc', '').split('T')[0]
                    if pub_date == day_str:
                        daily_sentiment['total'] += 1

                        # Count sentiments
                        for insight in article.get('insights', []):
                            if insight.get('ticker') == ticker and insight.get('sentiment') in daily_sentiment:
                                daily_sentiment[insight.get('sentiment')] += 1

                sentiment_trend.append(daily_sentiment)

            # Convert to DataFrame
            df = pd.DataFrame(sentiment_trend)

            # Calculate sentiment score
            df['score'] = df.apply(
                lambda row: (row['positive'] - row['negative']) / row['total'] if row['total'] > 0 else 0,
                axis=1
            )

            # Cache trend data
            self.cache.set(cache_key, df.to_dict('records'))

            logger.info(f"Calculated sentiment trend for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error calculating sentiment trend: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['date', 'positive', 'negative', 'neutral', 'total', 'score'])

    def plot_sentiment_trend(self, ticker: str, days_back: int = 30, save_path: Optional[str] = None):
        """
        Plot sentiment trend for visualization

        Args:
            ticker: Stock symbol
            days_back: Days of history to analyze
            save_path: Optional path to save the plot
        """
        try:
            # Get sentiment trend data
            df = self.get_sentiment_trend(ticker, days_back)

            if df.empty:
                logger.warning(f"No sentiment data to plot for {ticker}")
                return

            # Convert date column to datetime if it's not already
            df['date'] = pd.to_datetime(df['date'])

            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

            # Plot sentiment counts
            ax1.plot(df['date'], df['positive'], label='Positive', color='green')
            ax1.plot(df['date'], df['negative'], label='Negative', color='red')
            ax1.plot(df['date'], df['neutral'], label='Neutral', color='gray', linestyle='--')
            ax1.set_title(f'News Sentiment Trend for {ticker}')
            ax1.set_ylabel('Article Count')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot sentiment score
            ax2.plot(df['date'], df['score'], label='Sentiment Score', color='blue')
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_ylim(-1.1, 1.1)  # Set y-limits for the score
            ax2.set_ylabel('Score (-1 to +1)')
            ax2.set_xlabel('Date')
            ax2.grid(True, alpha=0.3)

            # Format x-axis
            ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            fig.autofmt_xdate()  # Rotate date labels

            plt.tight_layout()

            # Save if requested
            if save_path:
                plt.savefig(save_path)
                logger.info(f"Sentiment trend plot saved to {save_path}")

            plt.show()

        except Exception as e:
            logger.error(f"Error plotting sentiment trend: {e}")

    def get_top_articles(self, ticker: str, sentiment: str = None, days_back: int = 7, limit: int = 5) -> List[
        Dict[str, Any]]:
        """
        Get top articles for a ticker, optionally filtered by sentiment

        Args:
            ticker: Stock symbol
            sentiment: Optional filter ('positive', 'negative', 'neutral')
            days_back: Days of history to analyze
            limit: Maximum number of articles to return

        Returns:
            List of relevant articles
        """
        # Normalize ticker
        ticker = ticker.upper()

        # Get news
        articles = self.get_ticker_news(ticker, days_back)

        # Filter by sentiment if requested
        if sentiment:
            filtered_articles = []
            for article in articles:
                for insight in article.get('insights', []):
                    if insight.get('ticker') == ticker and insight.get('sentiment') == sentiment:
                        # Add reasoning to article for easy access
                        article['sentiment_reason'] = insight.get('sentiment_reasoning')
                        filtered_articles.append(article)
                        break
            articles = filtered_articles

        # Sort by date (newest first) and limit
        sorted_articles = sorted(
            articles,
            key=lambda x: x.get('published_utc', ''),
            reverse=True
        )

        return sorted_articles[:limit]

    def get_multi_ticker_sentiment(self, tickers: List[str], days_back: int = 7) -> Dict[str, float]:
        """
        Get sentiment scores for multiple tickers

        Args:
            tickers: List of stock symbols
            days_back: Days of history to analyze

        Returns:
            Dictionary mapping tickers to sentiment scores
        """
        results = {}

        for ticker in tickers:
            try:
                results[ticker] = self.get_sentiment_score(ticker, days_back)
            except Exception as e:
                logger.error(f"Error getting sentiment for {ticker}: {e}")
                results[ticker] = 0.0

            # Respect rate limits
            if self.calls_per_minute > 0:
                time.sleep(60.0 / self.calls_per_minute)

        return results
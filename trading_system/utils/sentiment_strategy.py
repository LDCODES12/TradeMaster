"""
Sentiment-Enhanced Options Trading Strategy

Integrates news sentiment analysis with options trading strategy
to identify high-probability opportunities.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class SentimentEnhancedStrategy:
    """
    Options trading strategy enhanced with news sentiment analysis.

    This strategy combines options data from multiple providers with
    news sentiment analysis to identify high-probability trades.
    """

    def __init__(self, options_provider, sentiment_analyzer, config: Dict[str, Any] = None):
        """
        Initialize the strategy

        Args:
            options_provider: MultiProviderOptionsData instance
            sentiment_analyzer: NewsSentimentAnalyzer instance
            config: Strategy configuration
        """
        self.options_provider = options_provider
        self.sentiment_analyzer = sentiment_analyzer
        self.config = config or {}

        # Strategy parameters (with defaults)
        self.min_sentiment_score = self.config.get('min_sentiment_score', 0.2)
        self.max_negative_score = self.config.get('max_negative_score', -0.3)
        self.min_probability = self.config.get('min_probability', 0.55)
        self.min_edge = self.config.get('min_edge', 0.2)
        self.min_sharpe = self.config.get('min_sharpe', 0.25)

        # Options selection parameters
        self.min_days_to_expiry = self.config.get('min_days_to_expiry', 7)
        self.max_days_to_expiry = self.config.get('max_days_to_expiry', 45)
        self.min_otm_percent = self.config.get('min_otm_percent', -0.05)  # Allow 5% ITM
        self.max_otm_percent = self.config.get('max_otm_percent', 0.15)  # Up to 15% OTM

        # Opportunity scoring weights
        self.sentiment_weight = self.config.get('sentiment_weight', 0.3)
        self.probability_weight = self.config.get('probability_weight', 0.4)
        self.sharpe_weight = self.config.get('sharpe_weight', 0.3)

        logger.info("Sentiment-enhanced options strategy initialized")

    def analyze_opportunities(self, watchlist: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze trading opportunities for a watchlist of symbols

        Args:
            watchlist: List of ticker symbols to analyze

        Returns:
            List of trading opportunities
        """
        opportunities = []

        # Get sentiment for all symbols in one batch to minimize API calls
        sentiment_scores = self.sentiment_analyzer.get_multi_ticker_sentiment(watchlist)

        logger.info(f"Analyzing {len(watchlist)} symbols for opportunities")

        for symbol in watchlist:
            # Get sentiment score
            sentiment_score = sentiment_scores.get(symbol, 0)

            # Skip if sentiment is not strong enough
            if abs(sentiment_score) < 0.1:
                logger.debug(f"Skipping {symbol} - neutral sentiment ({sentiment_score:.2f})")
                continue

            # Get options data
            options = self.options_provider.get_option_chain(symbol)
            if not options:
                logger.warning(f"No options data available for {symbol}")
                continue

            # Get current price
            current_price = self.options_provider.get_current_price(symbol)
            if current_price <= 0:
                logger.warning(f"Invalid price for {symbol}")
                continue

            # Find opportunities based on sentiment direction
            symbol_opportunities = []

            if sentiment_score >= self.min_sentiment_score:
                # Positive sentiment - look for calls
                call_options = [opt for opt in options if opt['option_type'] == 'call']
                symbol_opportunities.extend(self._analyze_bullish_options(call_options, current_price, sentiment_score))

            elif sentiment_score <= self.max_negative_score:
                # Negative sentiment - look for puts
                put_options = [opt for opt in options if opt['option_type'] == 'put']
                symbol_opportunities.extend(self._analyze_bearish_options(put_options, current_price, sentiment_score))

            # Add valid opportunities to the list
            opportunities.extend(symbol_opportunities)

        # Sort by opportunity score
        opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)

        logger.info(f"Found {len(opportunities)} total opportunities")
        return opportunities

    def _analyze_bullish_options(self, call_options: List[Dict[str, Any]],
                                 current_price: float,
                                 sentiment_score: float) -> List[Dict[str, Any]]:
        """
        Analyze call options for bullish opportunities

        Args:
            call_options: List of call option contracts
            current_price: Current price of the underlying
            sentiment_score: Sentiment score for the underlying

        Returns:
            List of bullish opportunities
        """
        opportunities = []

        # Filter options by days to expiry and moneyness
        filtered_options = self._filter_options_by_criteria(call_options, current_price)

        for option in filtered_options:
            # Calculate option metrics
            otm_percent = (option['strike'] - current_price) / current_price

            # Calculate adjusted probability based on sentiment
            base_probability = self._calculate_base_probability(option, 'bullish')
            sentiment_boost = sentiment_score * 0.1  # Max 10% boost from sentiment
            adjusted_probability = min(0.95, base_probability + sentiment_boost)

            # Calculate edge
            edge = self._calculate_edge(option, adjusted_probability)

            # Skip options with insufficient edge
            if edge < self.min_edge:
                continue

            # Calculate expected return
            expected_roi, risk_reward = self._calculate_expected_return(option, adjusted_probability)

            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(option, expected_roi)

            # Skip options with insufficient Sharpe
            if sharpe_ratio < self.min_sharpe:
                continue

            # Calculate overall opportunity score
            opportunity_score = (
                    sentiment_score * self.sentiment_weight +
                    adjusted_probability * self.probability_weight +
                    sharpe_ratio * self.sharpe_weight
            )

            # Create opportunity object
            opportunity = {
                'option': option,
                'strategy_type': 'bullish_call',
                'current_price': current_price,
                'sentiment_score': sentiment_score,
                'base_probability': base_probability,
                'adjusted_probability': adjusted_probability,
                'edge': edge,
                'expected_roi': expected_roi,
                'risk_reward': risk_reward,
                'sharpe_ratio': sharpe_ratio,
                'opportunity_score': opportunity_score,
                'otm_percent': otm_percent,
                'analysis_time': datetime.now().isoformat()
            }

            opportunities.append(opportunity)

        return opportunities

    def _analyze_bearish_options(self, put_options: List[Dict[str, Any]],
                                 current_price: float,
                                 sentiment_score: float) -> List[Dict[str, Any]]:
        """
        Analyze put options for bearish opportunities

        Args:
            put_options: List of put option contracts
            current_price: Current price of the underlying
            sentiment_score: Sentiment score for the underlying

        Returns:
            List of bearish opportunities
        """
        opportunities = []

        # Filter options by days to expiry and moneyness
        filtered_options = self._filter_options_by_criteria(put_options, current_price)

        for option in filtered_options:
            # Calculate option metrics
            otm_percent = (current_price - option['strike']) / current_price

            # Calculate adjusted probability based on sentiment
            # For puts, we use the absolute value of negative sentiment
            base_probability = self._calculate_base_probability(option, 'bearish')
            sentiment_boost = abs(sentiment_score) * 0.1  # Max 10% boost from sentiment
            adjusted_probability = min(0.95, base_probability + sentiment_boost)

            # Calculate edge
            edge = self._calculate_edge(option, adjusted_probability)

            # Skip options with insufficient edge
            if edge < self.min_edge:
                continue

            # Calculate expected return
            expected_roi, risk_reward = self._calculate_expected_return(option, adjusted_probability)

            # Calculate Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(option, expected_roi)

            # Skip options with insufficient Sharpe
            if sharpe_ratio < self.min_sharpe:
                continue

            # Calculate overall opportunity score
            opportunity_score = (
                    abs(sentiment_score) * self.sentiment_weight +  # Use absolute value for negative sentiment
                    adjusted_probability * self.probability_weight +
                    sharpe_ratio * self.sharpe_weight
            )

            # Create opportunity object
            opportunity = {
                'option': option,
                'strategy_type': 'bearish_put',
                'current_price': current_price,
                'sentiment_score': sentiment_score,
                'base_probability': base_probability,
                'adjusted_probability': adjusted_probability,
                'edge': edge,
                'expected_roi': expected_roi,
                'risk_reward': risk_reward,
                'sharpe_ratio': sharpe_ratio,
                'opportunity_score': opportunity_score,
                'otm_percent': otm_percent,
                'analysis_time': datetime.now().isoformat()
            }

            opportunities.append(opportunity)

        return opportunities

    def _filter_options_by_criteria(self, options: List[Dict[str, Any]], current_price: float) -> List[Dict[str, Any]]:
        """
        Filter options by days to expiry and moneyness

        Args:
            options: List of option contracts
            current_price: Current price of the underlying

        Returns:
            Filtered list of options
        """
        filtered_options = []

        for option in options:
            # Check days to expiry
            if option.get('days_to_expiry', 0) < self.min_days_to_expiry:
                continue

            if option.get('days_to_expiry', 0) > self.max_days_to_expiry:
                continue

            # Check moneyness
            is_call = option['option_type'] == 'call'
            strike = option['strike']

            if is_call:
                otm_percent = (strike - current_price) / current_price
            else:
                otm_percent = (current_price - strike) / current_price

            if otm_percent < self.min_otm_percent:
                continue

            if otm_percent > self.max_otm_percent:
                continue

            # Check for sufficient liquidity
            bid = option.get('bid', 0)
            ask = option.get('ask', 0)

            if bid <= 0 or ask <= 0:
                continue

            # Check for reasonable spread
            spread_percent = (ask - bid) / ask if ask > 0 else 1.0
            if spread_percent > 0.15:  # More than 15% spread
                continue

            # Add to filtered list
            filtered_options.append(option)

        return filtered_options

    def _calculate_base_probability(self, option: Dict[str, Any], direction: str) -> float:
        """
        Calculate base probability of profit

        Args:
            option: Option contract
            direction: Trade direction ('bullish' or 'bearish')

        Returns:
            Probability value (0-1)
        """
        # If we have delta, use it to estimate probability
        if 'delta' in option:
            delta = abs(option['delta'])

            # Delta is a rough approximation of ITM probability
            if direction == 'bullish' and option['option_type'] == 'call':
                # For calls, delta directly approximates probability
                return min(0.95, max(0.05, delta))
            elif direction == 'bearish' and option['option_type'] == 'put':
                # For puts, abs(delta) approximates probability
                return min(0.95, max(0.05, delta))

        # Fallback to estimating from other factors
        days_to_expiry = option.get('days_to_expiry', 30)
        is_call = option['option_type'] == 'call'

        # Start with a base probability
        base_prob = 0.5

        # Adjust for option type and direction alignment
        if (direction == 'bullish' and is_call) or (direction == 'bearish' and not is_call):
            base_prob += 0.05  # Small boost for aligned direction
        else:
            base_prob -= 0.05  # Small penalty for misaligned direction

        # Adjust for days to expiry (more time = higher probability)
        time_factor = min(0.1, days_to_expiry / 300)  # Max 0.1 adjustment
        base_prob += time_factor

        # Ensure probability is within bounds
        return min(0.95, max(0.05, base_prob))

    def _calculate_edge(self, option: Dict[str, Any], probability: float) -> float:
        """
        Calculate trading edge

        Args:
            option: Option contract
            probability: Probability of profit

        Returns:
            Edge value (0-1)
        """
        # Calculate based on probability vs break-even
        edge = (probability - 0.5) * 2  # Scale to 0-1 range

        # Adjust for time decay
        if 'theta' in option:
            theta = abs(option['theta'])
            price = option['price']

            # Normalize theta as percentage of price
            theta_percent = theta / price if price > 0 else 0

            # Adjust edge based on theta (higher theta = lower edge for long options)
            edge -= min(0.2, theta_percent * 5)  # Max 0.2 reduction

        # Adjust for implied volatility vs historical
        if 'iv' in option:
            iv = option['iv']
            # This is simplified - in production you'd compare to historical vol
            iv_factor = min(0.2, max(-0.2, 0.5 - iv))  # Range: -0.2 to 0.2
            edge += iv_factor

        # Ensure edge is within bounds
        return min(1.0, max(0.0, edge))

    def _calculate_expected_return(self, option: Dict[str, Any], probability: float) -> Tuple[float, float]:
        """
        Calculate expected return and risk-reward ratio

        Args:
            option: Option contract
            probability: Probability of profit

        Returns:
            Tuple of (expected_roi_percent, risk_reward_ratio)
        """
        price = option['price']

        # Estimate potential profit and loss
        # This is simplified - in production you'd use more sophisticated models
        potential_profit = price * 2  # Assume 100% upside
        potential_loss = price  # Assume full loss

        # Calculate expected value
        expected_value = (probability * potential_profit) - ((1 - probability) * potential_loss)

        # Calculate ROI
        expected_roi = (expected_value / price) * 100

        # Calculate risk-reward ratio
        risk_reward = potential_profit / potential_loss if potential_loss > 0 else float('inf')

        return expected_roi, risk_reward

    def _calculate_sharpe_ratio(self, option: Dict[str, Any], expected_roi: float) -> float:
        """
        Calculate Sharpe ratio for risk-adjusted returns

        Args:
            option: Option contract
            expected_roi: Expected return on investment

        Returns:
            Sharpe ratio
        """
        # Estimate volatility from option data
        if 'iv' in option:
            # Use implied volatility if available
            volatility = option['iv'] * 100  # Convert to percentage
        else:
            # Default volatility estimate
            volatility = 30  # 30% annualized

        # Adjust for days to expiry
        days_to_expiry = option.get('days_to_expiry', 30)
        time_factor = days_to_expiry / 365  # Fraction of a year

        # Annualize expected return
        annualized_return = expected_roi * (365 / days_to_expiry)

        # Calculate Sharpe (risk-free rate assumed to be 0 for simplicity)
        if volatility > 0:
            sharpe = annualized_return / (volatility * time_factor ** 0.5)
        else:
            sharpe = 0

        # Cap at reasonable bounds
        return max(0, min(10, sharpe))

    def calculate_position_size(self, opportunity: Dict[str, Any], account_value: float) -> int:
        """
        Calculate optimal position size using Kelly criterion

        Args:
            opportunity: Trading opportunity
            account_value: Total account value

        Returns:
            Number of contracts to trade
        """
        option = opportunity['option']
        probability = opportunity['adjusted_probability']

        # Kelly fraction (adjusted for safety)
        kelly = (2 * probability - 1)  # Full Kelly
        kelly_fraction = kelly * 0.25  # Quarter Kelly for safety

        # Cap at sensible maximum
        max_fraction = 0.05  # No more than 5% of account on one trade
        fraction = min(max_fraction, max(0, kelly_fraction))

        # Calculate position value
        position_value = account_value * fraction

        # Calculate contracts
        option_price = option['price'] * 100  # Cost per contract
        contracts = int(position_value / option_price) if option_price > 0 else 0

        # Ensure at least 1 contract if any are recommended
        contracts = max(1, contracts) if contracts > 0 else 0

        return contracts

    def get_sentiment_articles(self, symbol: str, opportunity: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get relevant sentiment articles for a trading opportunity

        Args:
            symbol: Ticker symbol
            opportunity: Trading opportunity

        Returns:
            List of relevant news articles with sentiment
        """
        # Determine which sentiment to look for
        sentiment_score = opportunity['sentiment_score']

        if sentiment_score > 0:
            sentiment_filter = 'positive'
        elif sentiment_score < 0:
            sentiment_filter = 'negative'
        else:
            sentiment_filter = None

        # Get relevant articles
        articles = self.sentiment_analyzer.get_top_articles(
            symbol,
            sentiment=sentiment_filter,
            days_back=7,
            limit=3
        )

        return articles
"""
NLP Sentiment Analysis Module
===============================
Advanced NLP-based sentiment analysis for cryptocurrency markets.

Data Sources:
- Twitter: Crypto influencers and trending topics
- Reddit: r/cryptocurrency, r/bitcoin, r/ethereum
- News: CryptoPanic, CoinTelegraph, CoinDesk
- Events: Economic calendar, protocol upgrades, regulatory news

Capabilities:
- Real-time sentiment scoring (-1 to +1)
- Entity extraction (coins, projects, people)
- Event detection (launches, hacks, regulations)
- Sentiment momentum tracking
- Whale alert integration

Author: DEMIR AI PRO
Version: 8.0
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import deque, Counter
import logging
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SentimentData:
    """Sentiment data point"""
    source: str
    text: str
    score: float
    entities: List[str]
    timestamp: str
    author: Optional[str] = None
    engagement: Optional[int] = None


@dataclass
class EventData:
    """Detected event"""
    event_type: str
    title: str
    description: str
    impact: str  # 'HIGH', 'MEDIUM', 'LOW'
    sentiment: float
    timestamp: str
    source: str


class NLPSentimentAnalyzer:
    """
    Core NLP Sentiment Analyzer
    
    Uses lexicon-based and rule-based sentiment analysis.
    Can be extended with ML models (BERT, FinBERT) for production.
    """
    
    # Crypto-specific sentiment lexicon
    POSITIVE_WORDS = [
        'bullish', 'moon', 'pump', 'rally', 'breakout', 'adoption',
        'milestone', 'upgrade', 'partnership', 'institutional',
        'ath', 'surge', 'gains', 'growth', 'innovation', 'buy',
        'long', 'hodl', 'accumulate', 'support', 'breakthrough'
    ]
    
    NEGATIVE_WORDS = [
        'bearish', 'dump', 'crash', 'fud', 'scam', 'hack', 'rug',
        'fraud', 'investigation', 'ban', 'regulation', 'lawsuit',
        'collapse', 'losses', 'decline', 'sell', 'short', 'resistance',
        'risk', 'warning', 'concern', 'vulnerability'
    ]
    
    AMPLIFIERS = ['very', 'extremely', 'super', 'massive', 'huge', 'enormous']
    NEGATIONS = ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither']
    
    def __init__(self):
        self.history = deque(maxlen=1000)
        
    def analyze_text(self, text: str) -> Dict[str, any]:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment score and breakdown
        """
        try:
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            
            positive_count = 0
            negative_count = 0
            
            i = 0
            while i < len(words):
                word = words[i]
                multiplier = 1.0
                
                # Check for amplifiers
                if i > 0 and words[i-1] in self.AMPLIFIERS:
                    multiplier = 1.5
                
                # Check for negations
                negation = False
                if i > 0 and words[i-1] in self.NEGATIONS:
                    negation = True
                
                # Count sentiment words
                if word in self.POSITIVE_WORDS:
                    if negation:
                        negative_count += multiplier
                    else:
                        positive_count += multiplier
                elif word in self.NEGATIVE_WORDS:
                    if negation:
                        positive_count += multiplier
                    else:
                        negative_count += multiplier
                
                i += 1
            
            # Calculate sentiment score (-1 to +1)
            total = positive_count + negative_count
            if total > 0:
                score = (positive_count - negative_count) / total
            else:
                score = 0.0
            
            # Classify sentiment
            if score > 0.3:
                classification = "POSITIVE"
            elif score < -0.3:
                classification = "NEGATIVE"
            else:
                classification = "NEUTRAL"
            
            return {
                'score': score,
                'classification': classification,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'word_count': len(words),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Text sentiment analysis error: {e}")
            return {
                'score': 0.0,
                'classification': 'NEUTRAL',
                'positive_count': 0,
                'negative_count': 0,
                'word_count': 0,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract cryptocurrency entities from text
        
        Returns:
            List of detected coins/tokens
        """
        # Common crypto symbols and names
        crypto_entities = [
            'btc', 'bitcoin', 'eth', 'ethereum', 'usdt', 'tether',
            'bnb', 'binance', 'xrp', 'ripple', 'ada', 'cardano',
            'sol', 'solana', 'doge', 'dogecoin', 'avax', 'avalanche',
            'matic', 'polygon', 'dot', 'polkadot', 'link', 'chainlink'
        ]
        
        text_lower = text.lower()
        detected = []
        
        for entity in crypto_entities:
            if re.search(r'\b' + entity + r'\b', text_lower):
                detected.append(entity.upper())
        
        # Also look for $ symbols (e.g., $BTC)
        dollar_symbols = re.findall(r'\$([A-Z]{2,10})', text)
        detected.extend(dollar_symbols)
        
        return list(set(detected))
    
    def aggregate_sentiment(
        self,
        sentiments: List[SentimentData],
        window_minutes: int = 60
    ) -> Dict[str, any]:
        """
        Aggregate multiple sentiment data points
        
        Args:
            sentiments: List of SentimentData objects
            window_minutes: Time window for aggregation
            
        Returns:
            Aggregated sentiment metrics
        """
        try:
            if not sentiments:
                return self._empty_aggregate()
            
            # Filter by time window
            cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
            recent = [
                s for s in sentiments
                if datetime.fromisoformat(s.timestamp) >= cutoff
            ]
            
            if not recent:
                return self._empty_aggregate()
            
            # Calculate weighted average (weight by engagement if available)
            total_score = 0
            total_weight = 0
            
            for s in recent:
                weight = s.engagement if s.engagement else 1
                total_score += s.score * weight
                total_weight += weight
            
            avg_score = total_score / total_weight if total_weight > 0 else 0
            
            # Count by classification
            positive = sum(1 for s in recent if s.score > 0.3)
            negative = sum(1 for s in recent if s.score < -0.3)
            neutral = len(recent) - positive - negative
            
            # Most mentioned entities
            all_entities = []
            for s in recent:
                all_entities.extend(s.entities)
            entity_counts = Counter(all_entities)
            top_entities = entity_counts.most_common(10)
            
            # Sentiment by source
            source_sentiment = {}
            for source in set(s.source for s in recent):
                source_data = [s for s in recent if s.source == source]
                source_avg = np.mean([s.score for s in source_data])
                source_sentiment[source] = source_avg
            
            return {
                'avg_score': avg_score,
                'sample_size': len(recent),
                'positive_count': positive,
                'negative_count': negative,
                'neutral_count': neutral,
                'positive_pct': (positive / len(recent)) * 100 if recent else 0,
                'negative_pct': (negative / len(recent)) * 100 if recent else 0,
                'top_entities': top_entities,
                'source_sentiment': source_sentiment,
                'window_minutes': window_minutes,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Sentiment aggregation error: {e}")
            return self._empty_aggregate()
    
    def _empty_aggregate(self) -> Dict:
        return {
            'avg_score': 0.0,
            'sample_size': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'positive_pct': 0,
            'negative_pct': 0,
            'top_entities': [],
            'source_sentiment': {},
            'window_minutes': 0,
            'timestamp': datetime.utcnow().isoformat()
        }


class EventDetector:
    """
    Cryptocurrency Event Detection
    
    Detects and classifies events:
    - Protocol upgrades (hard forks, network updates)
    - Security incidents (hacks, exploits)
    - Regulatory news (bans, approvals, lawsuits)
    - Market events (listings, delistings, large transfers)
    - Partnerships and integrations
    """
    
    EVENT_KEYWORDS = {
        'UPGRADE': ['upgrade', 'fork', 'update', 'launch', 'mainnet', 'testnet'],
        'SECURITY': ['hack', 'exploit', 'vulnerability', 'breach', 'attack', 'scam'],
        'REGULATORY': ['sec', 'regulation', 'ban', 'lawsuit', 'investigation', 'compliance'],
        'MARKET': ['listing', 'delisting', 'whale', 'transfer', 'burn', 'mint'],
        'PARTNERSHIP': ['partnership', 'collaboration', 'integration', 'adoption']
    }
    
    def __init__(self):
        self.detected_events = deque(maxlen=100)
        
    def detect_event(self, text: str, source: str = "unknown") -> Optional[EventData]:
        """
        Detect if text describes a significant event
        
        Args:
            text: Text to analyze
            source: Source of the text
            
        Returns:
            EventData if event detected, None otherwise
        """
        try:
            text_lower = text.lower()
            
            # Check for event keywords
            detected_type = None
            max_matches = 0
            
            for event_type, keywords in self.EVENT_KEYWORDS.items():
                matches = sum(1 for kw in keywords if kw in text_lower)
                if matches > max_matches:
                    max_matches = matches
                    detected_type = event_type
            
            if not detected_type or max_matches == 0:
                return None
            
            # Analyze sentiment of event
            analyzer = NLPSentimentAnalyzer()
            sentiment_result = analyzer.analyze_text(text)
            
            # Determine impact based on event type and sentiment
            if detected_type in ['SECURITY', 'REGULATORY']:
                impact = 'HIGH'
            elif detected_type in ['MARKET', 'PARTNERSHIP']:
                impact = 'MEDIUM'
            else:
                impact = 'LOW'
            
            event = EventData(
                event_type=detected_type,
                title=text[:100],  # First 100 chars as title
                description=text,
                impact=impact,
                sentiment=sentiment_result['score'],
                timestamp=datetime.utcnow().isoformat(),
                source=source
            )
            
            self.detected_events.append(event)
            logger.info(f"Event detected: {event.event_type} (impact: {event.impact})")
            
            return event
            
        except Exception as e:
            logger.error(f"Event detection error: {e}")
            return None
    
    def get_recent_events(
        self,
        hours: int = 24,
        event_type: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Get recent detected events
        
        Args:
            hours: Time window in hours
            event_type: Filter by specific event type
            
        Returns:
            List of recent events
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        recent = [
            asdict(event) for event in self.detected_events
            if datetime.fromisoformat(event.timestamp) >= cutoff
        ]
        
        if event_type:
            recent = [e for e in recent if e['event_type'] == event_type]
        
        return recent


class SocialMediaMonitor:
    """
    Social Media Sentiment Monitor
    
    Monitors Twitter, Reddit, and other social platforms for crypto sentiment.
    Tracks trending topics, influencer opinions, and community sentiment.
    
    Note: This is a framework. Real implementation requires API keys and
    actual integration with Twitter API v2, Reddit API, etc.
    """
    
    def __init__(self, update_interval: int = 300):
        """
        Args:
            update_interval: How often to fetch new data (seconds)
        """
        self.update_interval = update_interval
        self.analyzer = NLPSentimentAnalyzer()
        self.event_detector = EventDetector()
        self.sentiment_cache = deque(maxlen=500)
        
    async def fetch_twitter_sentiment(
        self,
        query: str = "bitcoin OR ethereum OR crypto",
        max_results: int = 100
    ) -> List[SentimentData]:
        """
        Fetch and analyze Twitter sentiment
        
        Note: Requires Twitter API v2 credentials in environment
        
        Args:
            query: Search query
            max_results: Max tweets to fetch
            
        Returns:
            List of SentimentData objects
        """
        # Production implementation would use Twitter API v2
        # For now, return framework structure
        logger.warning("Twitter integration requires API credentials")
        return []
    
    async def fetch_reddit_sentiment(
        self,
        subreddits: List[str] = ['cryptocurrency', 'bitcoin', 'ethereum'],
        limit: int = 100
    ) -> List[SentimentData]:
        """
        Fetch and analyze Reddit sentiment
        
        Note: Requires Reddit API credentials (PRAW)
        
        Args:
            subreddits: List of subreddit names
            limit: Max posts to fetch per subreddit
            
        Returns:
            List of SentimentData objects
        """
        # Production implementation would use PRAW
        logger.warning("Reddit integration requires API credentials")
        return []
    
    async def fetch_news_sentiment(
        self,
        sources: List[str] = ['cryptopanic', 'cointelegraph', 'coindesk']
    ) -> List[SentimentData]:
        """
        Fetch and analyze crypto news sentiment
        
        Note: Requires news API credentials
        
        Args:
            sources: List of news sources
            
        Returns:
            List of SentimentData objects
        """
        # Production implementation would use news APIs
        logger.warning("News integration requires API credentials")
        return []
    
    def process_text_batch(
        self,
        texts: List[Tuple[str, str]],  # [(text, source), ...]
        detect_events: bool = True
    ) -> Dict[str, any]:
        """
        Process batch of texts for sentiment analysis
        
        Args:
            texts: List of (text, source) tuples
            detect_events: Whether to detect events
            
        Returns:
            Aggregated sentiment results
        """
        try:
            sentiments = []
            events = []
            
            for text, source in texts:
                # Analyze sentiment
                sentiment_result = self.analyzer.analyze_text(text)
                entities = self.analyzer.extract_entities(text)
                
                sentiment_data = SentimentData(
                    source=source,
                    text=text,
                    score=sentiment_result['score'],
                    entities=entities,
                    timestamp=datetime.utcnow().isoformat()
                )
                
                sentiments.append(sentiment_data)
                self.sentiment_cache.append(sentiment_data)
                
                # Detect events
                if detect_events:
                    event = self.event_detector.detect_event(text, source)
                    if event:
                        events.append(asdict(event))
            
            # Aggregate results
            aggregated = self.analyzer.aggregate_sentiment(sentiments)
            aggregated['detected_events'] = events
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return self.analyzer._empty_aggregate()
    
    def get_sentiment_trend(
        self,
        minutes: int = 60,
        interval: int = 10
    ) -> Dict[str, List[float]]:
        """
        Get sentiment trend over time
        
        Args:
            minutes: Total time window
            interval: Interval size in minutes
            
        Returns:
            Dictionary with timestamps and scores
        """
        try:
            cutoff = datetime.utcnow() - timedelta(minutes=minutes)
            recent = [
                s for s in self.sentiment_cache
                if datetime.fromisoformat(s.timestamp) >= cutoff
            ]
            
            # Create time buckets
            num_buckets = minutes // interval
            buckets = {i: [] for i in range(num_buckets)}
            
            for sentiment in recent:
                time_diff = datetime.utcnow() - datetime.fromisoformat(sentiment.timestamp)
                bucket_idx = min(int(time_diff.total_seconds() / 60 / interval), num_buckets - 1)
                buckets[bucket_idx].append(sentiment.score)
            
            # Calculate average per bucket
            trend_scores = []
            trend_timestamps = []
            
            for i in range(num_buckets):
                if buckets[i]:
                    avg_score = np.mean(buckets[i])
                    trend_scores.append(avg_score)
                    bucket_time = datetime.utcnow() - timedelta(minutes=(num_buckets - i) * interval)
                    trend_timestamps.append(bucket_time.isoformat())
            
            return {
                'timestamps': trend_timestamps,
                'scores': trend_scores,
                'interval_minutes': interval,
                'window_minutes': minutes
            }
            
        except Exception as e:
            logger.error(f"Sentiment trend calculation error: {e}")
            return {
                'timestamps': [],
                'scores': [],
                'interval_minutes': interval,
                'window_minutes': minutes
            }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of NLP Sentiment Analysis
    """
    
    print("\n=== NLP SENTIMENT ANALYZER ===")
    analyzer = NLPSentimentAnalyzer()
    
    # Test texts
    texts = [
        "Bitcoin is extremely bullish! Moon soon! 🚀",
        "Major hack detected, FUD spreading across market",
        "Ethereum upgrade successful, institutional adoption growing",
        "Massive sell-off, bearish sentiment everywhere"
    ]
    
    for text in texts:
        result = analyzer.analyze_text(text)
        entities = analyzer.extract_entities(text)
        print(f"\nText: {text}")
        print(f"Sentiment: {result['classification']} (score: {result['score']:.2f})")
        print(f"Entities: {entities}")
    
    print("\n=== EVENT DETECTOR ===")
    detector = EventDetector()
    
    event_texts = [
        "Major Bitcoin hard fork scheduled for next week",
        "SEC approves first spot Bitcoin ETF",
        "Binance exchange hacked, $100M stolen"
    ]
    
    for text in event_texts:
        event = detector.detect_event(text, "news")
        if event:
            print(f"\nEvent: {event.event_type}")
            print(f"Impact: {event.impact}")
            print(f"Sentiment: {event.sentiment:.2f}")
    
    print("\n=== SOCIAL MEDIA MONITOR ===")
    monitor = SocialMediaMonitor()
    
    # Simulate batch processing
    batch = [(text, "twitter") for text in texts]
    results = monitor.process_text_batch(batch, detect_events=True)
    
    print(f"Average Sentiment: {results['avg_score']:.2f}")
    print(f"Positive: {results['positive_pct']:.1f}%")
    print(f"Negative: {results['negative_pct']:.1f}%")
    print(f"Events Detected: {len(results['detected_events'])}")

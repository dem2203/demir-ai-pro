"""
Sentiment Analysis Layer

Multi-source sentiment analysis:
- Fear & Greed Index
- Funding rates
- Order book imbalance
- Social sentiment
- Market regime
"""

from .analyzer import SentimentAnalyzer

__all__ = ['SentimentAnalyzer']

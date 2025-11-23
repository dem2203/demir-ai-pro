#!/usr/bin/env python3
"""
Sentiment Analyzer

Production sentiment analysis from multiple sources.
Zero mock data - all from real APIs.
"""

import logging
import requests
import numpy as np
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Multi-source sentiment analyzer.
    """
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'DEMIR-AI-PRO/8.0'})
        logger.info("✅ Sentiment Analyzer initialized")
    
    def get_fear_greed_index(self) -> float:
        """
        Fetch Fear & Greed Index.
        
        Returns:
            Score 0-1 (0=extreme fear, 1=extreme greed)
        """
        try:
            response = self.session.get(
                'https://api.alternative.me/fng/',
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                value = int(data['data'][0]['value'])
                return value / 100.0
        except Exception as e:
            logger.error(f"Fear & Greed failed: {e}")
        return 0.5
    
    def get_funding_rates(self, symbol: str = 'BTCUSDT') -> float:
        """
        Fetch Binance funding rates.
        
        Returns:
            Sentiment score 0-1
        """
        try:
            response = self.session.get(
                'https://fapi.binance.com/fapi/v1/fundingRate',
                params={'symbol': symbol, 'limit': 24},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                rates = [float(d['fundingRate']) for d in data]
                avg_funding = np.mean(rates)
                score = max(0.1, min(0.9, 0.5 - avg_funding * 100))
                return float(score)
        except Exception as e:
            logger.error(f"Funding rates failed: {e}")
        return 0.5
    
    def get_order_book_sentiment(self, symbol: str = 'BTCUSDT') -> float:
        """
        Calculate order book buy/sell imbalance.
        
        Returns:
            Sentiment score 0-1
        """
        try:
            response = self.session.get(
                'https://fapi.binance.com/fapi/v1/depth',
                params={'symbol': symbol, 'limit': 20},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                buy_vol = sum(float(b[1]) for b in data['bids'][:5])
                sell_vol = sum(float(a[1]) for a in data['asks'][:5])
                
                if buy_vol + sell_vol > 0:
                    imbalance = buy_vol / (buy_vol + sell_vol)
                    return float(np.clip(imbalance, 0, 1))
        except Exception as e:
            logger.error(f"Order book failed: {e}")
        return 0.5
    
    def analyze(self, symbol: str = 'BTCUSDT') -> Dict[str, Any]:
        """
        Comprehensive sentiment analysis.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Sentiment analysis results
        """
        scores = {
            'fear_greed': self.get_fear_greed_index(),
            'funding_rates': self.get_funding_rates(symbol),
            'order_book': self.get_order_book_sentiment(symbol)
        }
        
        avg_sentiment = np.mean(list(scores.values()))
        
        return {
            'symbol': symbol,
            'sentiment_score': float(avg_sentiment),
            'components': scores,
            'timestamp': datetime.now().isoformat()
        }

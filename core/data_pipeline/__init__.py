"""
Data Pipeline Module

Real-time data fetching, processing, and caching.
Production-grade data pipeline with failover.
"""

from .fetcher import DataFetcher
from .processor import DataProcessor

__all__ = ['DataFetcher', 'DataProcessor']

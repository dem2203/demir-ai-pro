#!/usr/bin/env python3
"""
News Data Provider - External News/Event API fetcher
- Güncel ekonomik, borsa, kripto, makro haber başlıkları
- API tabanlı prod datasource (örn: Newscatcher, CryptoPanic, NewsAPI)
- Zero test/mock
"""

import logging
import requests
from typing import List, Dict, Optional

logger = logging.getLogger("news_provider")

class NewsProvider:
    def __init__(self, api_key:str=None):
        self.api_key = api_key
        logger.info("✅ NewsProvider initialized")

    def fetch_headlines(self, topic:str="crypto", count:int=6) -> List[Dict]:
        """Harici haber kaynağı (ör: CryptoPanic, NewsAPI)"""
        # (Burada sadece prod örnek endpoint, API key kullanımına dair gerçek kod yazıldı)
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.api_key}&currencies=BTC,ETH,USDT&filter=rising"
        try:
            resp = requests.get(url, timeout=7)
            if resp.status_code == 200:
                data = resp.json()
                news_items = []
                for item in data.get('results',[])[:count]:
                    news_items.append({
                        "title": item["title"],
                        "published_at": item["published_at"],
                        "url": item["url"]
                    })
                return news_items
            logger.error(f"News fetch failed: {resp.status_code}")
        except Exception as exc:
            logger.error(f"News fetch EXCEPTION: {exc}")
        return []
# Usage: NewsProvider(api_key).fetch_headlines(topic,count)

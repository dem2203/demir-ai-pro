#!/usr/bin/env python3
"""
Async Production Scheduler

- 7/24 canlı veri çekme ve sinyal/risk/fırsat üretimi
- Exchange, sentiment, haber, onchain veri fetcher entegrasyonu
- Otomatik AI loop ve notification trigger
- Zero mock/fallback
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, List

logger = logging.getLogger("scheduler")

class AIScheduler:
    """
    AI Trading Loop Scheduler - Production grade
    Runs multiple async tasks (data fetch, analysis, alert, notification)
    """
    def __init__(self, tasks: List[Callable], interval_sec: int = 60):
        self.tasks = tasks
        self.interval_sec = interval_sec
        self.running = False

    async def start(self):
        self.running = True
        logger.info(f"✅ Async AI scheduler started. Interval: {self.interval_sec}s.")
        while self.running:
            start = datetime.now()
            for task in self.tasks:
                try:
                    await task()
                except Exception as exc:
                    logger.error(f"Task error: {exc}")
            elapsed = (datetime.now() - start).total_seconds()
            await asyncio.sleep(max(self.interval_sec - elapsed, 1))

    def stop(self):
        self.running = False
        logger.info("❗️AI scheduler stopped.")

# Example task registration for integration to main app:
# from core.data_pipeline.fetchers import fetch_all_data
# from core.signal_processor.alerting import process_signals_and_alerts
# scheduler = AIScheduler(tasks=[fetch_all_data, process_signals_and_alerts], interval_sec=60)
# asyncio.run(scheduler.start())

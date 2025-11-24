"""
Pure Technical Strategy Backtest
=================================
Backtesting for sentiment-free trading strategy.

Author: DEMIR AI PRO  
Version: 8.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from backtesting import BacktestEngine, PerformanceAnalyzer
from core.signal_processor.enhanced_aggregator import EnhancedSignalAggregator
from core.risk_manager.dynamic_sizing import DynamicPositionSizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PureTechnicalStrategy:
    """Pure Technical + Microstructure Strategy"""
    
    def __init__(self, confidence_threshold=0.7, max_risk_pct=0.02):
        self.aggregator = EnhancedSignalAggregator(confidence_threshold=confidence_threshold)
        self.position_sizer = DynamicPositionSizer(max_risk_pct=max_risk_pct)
        self.confidence_threshold = confidence_threshold
    
    def generate_signals(self, df, orderbook_history=None):
        """Generate trading signals"""
        signals = []
        min_periods = 50
        
        for i in range(len(df)):
            if i < min_periods:
                signals.append(0)
                continue
            
            hist_df = df.iloc[:i+1]
            orderbook = orderbook_history[i] if orderbook_history and i < len(orderbook_history) else None
            
            try:
                signal_data = self.aggregator.generate_signal(hist_df, orderbook, hist_df['close'].iloc[-1])
                signals.append(signal_data.signal if signal_data.confidence >= self.confidence_threshold else 0)
            except Exception as e:
                logger.error(f"Error at {i}: {e}")
                signals.append(0)
        
        return pd.Series(signals, index=df.index)


def generate_sample_data(num_candles=1000, seed=42):
    """Generate sample OHLCV data"""
    np.random.seed(seed)
    returns = np.random.randn(num_candles) * 0.02
    trend = np.linspace(0, 0.5, num_candles)
    prices = 100 * np.exp(np.cumsum(returns) + trend)
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.randn(num_candles) * 0.005),
        'high': prices * (1 + abs(np.random.randn(num_candles)) * 0.01),
        'low': prices * (1 - abs(np.random.randn(num_candles)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000, 10000, num_candles)
    })
    
    start_date = datetime(2024, 1, 1)
    df.index = [start_date + timedelta(hours=i) for i in range(num_candles)]
    return df


def run_backtest(df, initial_capital=10000, commission_pct=0.001):
    """Run backtest"""
    logger.info("="*60)
    logger.info("PURE TECHNICAL STRATEGY BACKTEST")
    logger.info("="*60)
    
    strategy = PureTechnicalStrategy(confidence_threshold=0.7, max_risk_pct=0.02)
    engine = BacktestEngine(initial_capital=initial_capital, commission_pct=commission_pct, slippage_pct=0.0005)
    
    logger.info(f"\nParameters:")
    logger.info(f"  Period: {df.index[0]} to {df.index[-1]}")
    logger.info(f"  Candles: {len(df)}")
    logger.info(f"  Capital: ${initial_capital:,.0f}")
    
    logger.info("\nGenerating signals...")
    df['signal'] = strategy.generate_signals(df)
    
    logger.info("\nRunning backtest...")
    results = engine.run(df, lambda d, **p: d['signal'], {})
    
    logger.info("\n"+"="*60)
    logger.info("RESULTS")
    logger.info("="*60)
    logger.info(f"\nTotal Return: {results.total_return:.2%}")
    logger.info(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown: {results.max_drawdown:.2%}")
    logger.info(f"Win Rate: {results.win_rate:.2%}")
    logger.info(f"Total Trades: {results.total_trades}")
    
    return results


if __name__ == "__main__":
    logger.info("Generating data...")
    df = generate_sample_data(2000, 42)
    run_backtest(df, 10000, 0.001)

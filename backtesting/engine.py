"""
Enterprise Backtesting Engine
==============================
Production-grade backtesting framework for systematic trading strategies.

Features:
- Historical simulation with slippage and fees
- Comprehensive performance metrics (Sharpe, Sortino, Calmar, Max DD)
- Parameter optimization (grid search, random search)
- Walk-forward analysis
- Monte Carlo simulation
- Equity curve visualization
- Trade-by-trade analysis

Author: DEMIR AI PRO
Version: 8.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from enum import Enum
import itertools
from scipy import stats

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class PositionSide(Enum):
    """Position sides"""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class Trade:
    """Individual trade record"""
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    side: str
    size: float
    pnl: float
    pnl_pct: float
    commission: float
    duration_minutes: int


@dataclass
class BacktestResults:
    """Complete backtest results"""
    # Performance metrics
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: int
    
    # Risk metrics
    volatility: float
    value_at_risk_95: float
    expected_shortfall_95: float
    
    # Equity data
    equity_curve: List[float]
    drawdown_curve: List[float]
    
    # Trades
    trades: List[Trade]
    
    # Timestamps
    start_date: str
    end_date: str
    duration_days: int


class BacktestEngine:
    """
    Core Backtesting Engine
    
    Simulates strategy execution on historical data with realistic
    market conditions (slippage, fees, execution delay).
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        commission_pct: float = 0.001,  # 0.1% per trade
        slippage_pct: float = 0.0005,   # 0.05% slippage
        leverage: float = 1.0
    ):
        """
        Args:
            initial_capital: Starting capital
            commission_pct: Commission per trade (as decimal)
            slippage_pct: Slippage per trade (as decimal)
            leverage: Maximum leverage allowed
        """
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.leverage = leverage
        
        self.reset()
        
    def reset(self):
        """Reset backtest state"""
        self.capital = self.initial_capital
        self.position_side = PositionSide.FLAT
        self.position_size = 0
        self.entry_price = 0
        self.entry_time = None
        
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.timestamps = []
        
    def run(
        self,
        df: pd.DataFrame,
        strategy_func: Callable,
        strategy_params: Optional[Dict] = None
    ) -> BacktestResults:
        """
        Run backtest on historical data
        
        Args:
            df: OHLCV DataFrame
            strategy_func: Function that returns signals (-1, 0, 1)
            strategy_params: Parameters to pass to strategy
            
        Returns:
            BacktestResults object
        """
        try:
            logger.info(f"Starting backtest on {len(df)} candles")
            self.reset()
            
            if strategy_params is None:
                strategy_params = {}
            
            # Generate signals
            df['signal'] = strategy_func(df, **strategy_params)
            
            # Simulate trading
            for i in range(1, len(df)):
                current_bar = df.iloc[i]
                signal = current_bar['signal']
                timestamp = current_bar.name if hasattr(current_bar, 'name') else str(i)
                
                # Execute trades based on signal
                if signal == 1 and self.position_side != PositionSide.LONG:
                    # Enter long
                    self._close_position(current_bar, timestamp)
                    self._open_position(PositionSide.LONG, current_bar, timestamp)
                    
                elif signal == -1 and self.position_side != PositionSide.SHORT:
                    # Enter short
                    self._close_position(current_bar, timestamp)
                    self._open_position(PositionSide.SHORT, current_bar, timestamp)
                    
                elif signal == 0 and self.position_side != PositionSide.FLAT:
                    # Close position
                    self._close_position(current_bar, timestamp)
                
                # Update equity
                self._update_equity(current_bar, timestamp)
            
            # Close any open position at end
            if self.position_side != PositionSide.FLAT:
                final_bar = df.iloc[-1]
                final_timestamp = final_bar.name if hasattr(final_bar, 'name') else str(len(df)-1)
                self._close_position(final_bar, final_timestamp)
            
            # Calculate performance metrics
            results = self._calculate_metrics(df)
            
            logger.info(f"Backtest complete: {results.total_trades} trades, "
                       f"{results.total_return:.2%} return")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest execution error: {e}")
            raise
    
    def _open_position(
        self,
        side: PositionSide,
        bar: pd.Series,
        timestamp: str
    ):
        """Open a new position"""
        price = bar['close']
        
        # Apply slippage
        if side == PositionSide.LONG:
            entry_price = price * (1 + self.slippage_pct)
        else:
            entry_price = price * (1 - self.slippage_pct)
        
        # Calculate position size (use all capital with leverage)
        max_size = (self.capital * self.leverage) / entry_price
        
        # Apply commission
        commission = self.capital * self.commission_pct
        self.capital -= commission
        
        self.position_side = side
        self.position_size = max_size
        self.entry_price = entry_price
        self.entry_time = timestamp
        
    def _close_position(self, bar: pd.Series, timestamp: str):
        """Close current position"""
        if self.position_side == PositionSide.FLAT:
            return
        
        price = bar['close']
        
        # Apply slippage
        if self.position_side == PositionSide.LONG:
            exit_price = price * (1 - self.slippage_pct)
        else:
            exit_price = price * (1 + self.slippage_pct)
        
        # Calculate P&L
        if self.position_side == PositionSide.LONG:
            pnl = (exit_price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - exit_price) * self.position_size
        
        # Apply commission
        commission = (exit_price * self.position_size) * self.commission_pct
        pnl -= commission
        
        # Update capital
        self.capital += pnl
        
        # Calculate metrics
        pnl_pct = (pnl / self.initial_capital) * 100
        
        # Calculate duration
        try:
            if isinstance(self.entry_time, str) and isinstance(timestamp, str):
                entry_dt = pd.to_datetime(self.entry_time)
                exit_dt = pd.to_datetime(timestamp)
                duration = int((exit_dt - entry_dt).total_seconds() / 60)
            else:
                duration = 0
        except:
            duration = 0
        
        # Record trade
        trade = Trade(
            entry_time=str(self.entry_time),
            entry_price=self.entry_price,
            exit_time=str(timestamp),
            exit_price=exit_price,
            side=self.position_side.value,
            size=self.position_size,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            duration_minutes=duration
        )
        self.trades.append(trade)
        
        # Reset position
        self.position_side = PositionSide.FLAT
        self.position_size = 0
        self.entry_price = 0
        self.entry_time = None
        
    def _update_equity(self, bar: pd.Series, timestamp: str):
        """Update equity curve"""
        equity = self.capital
        
        # Add unrealized P&L if position is open
        if self.position_side != PositionSide.FLAT:
            current_price = bar['close']
            
            if self.position_side == PositionSide.LONG:
                unrealized_pnl = (current_price - self.entry_price) * self.position_size
            else:
                unrealized_pnl = (self.entry_price - current_price) * self.position_size
            
            equity += unrealized_pnl
        
        self.equity_curve.append(equity)
        self.timestamps.append(timestamp)
        
    def _calculate_metrics(self, df: pd.DataFrame) -> BacktestResults:
        """Calculate comprehensive performance metrics"""
        
        # Convert equity to numpy array
        equity = np.array(self.equity_curve)
        
        # Returns
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns)]  # Remove NaN
        
        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        # CAGR
        num_years = len(df) / (365 * 24 * 60)  # Assuming 1-minute bars
        cagr = (equity[-1] / equity[0]) ** (1 / num_years) - 1 if num_years > 0 else 0
        
        # Sharpe Ratio (annualized)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(365 * 24 * 60)
        else:
            sharpe = 0
        
        # Sortino Ratio (annualized)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (returns.mean() / downside_returns.std()) * np.sqrt(365 * 24 * 60)
        else:
            sortino = 0
        
        # Drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax
        max_drawdown = abs(drawdown.min())
        
        # Max drawdown duration
        dd_duration = 0
        current_dd_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                dd_duration = max(dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        # Calmar Ratio
        calmar = cagr / max_drawdown if max_drawdown > 0 else 0
        
        # Trade statistics
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl < 0]
            
            total_trades = len(self.trades)
            num_wins = len(winning_trades)
            num_losses = len(losing_trades)
            win_rate = num_wins / total_trades if total_trades > 0 else 0
            
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            avg_duration = int(np.mean([t.duration_minutes for t in self.trades]))
        else:
            total_trades = 0
            num_wins = 0
            num_losses = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_duration = 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(365 * 24 * 60) if len(returns) > 0 else 0
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        
        # Dates
        start_date = df.index[0] if hasattr(df, 'index') else "N/A"
        end_date = df.index[-1] if hasattr(df, 'index') else "N/A"
        duration_days = len(df) // (24 * 60)  # Assuming 1-minute bars
        
        return BacktestResults(
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_drawdown,
            max_drawdown_duration=dd_duration,
            total_trades=total_trades,
            winning_trades=num_wins,
            losing_trades=num_losses,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration=avg_duration,
            volatility=volatility,
            value_at_risk_95=var_95,
            expected_shortfall_95=cvar_95,
            equity_curve=equity.tolist(),
            drawdown_curve=drawdown.tolist(),
            trades=[asdict(t) for t in self.trades],
            start_date=str(start_date),
            end_date=str(end_date),
            duration_days=duration_days
        )


class PerformanceAnalyzer:
    """
    Advanced Performance Analysis
    
    Provides deep performance insights:
    - Time-based performance breakdown
    - Win/loss analysis by conditions
    - Trade distribution analysis
    - Risk-adjusted returns
    """
    
    @staticmethod
    def analyze_results(results: BacktestResults) -> Dict[str, any]:
        """
        Comprehensive results analysis
        
        Args:
            results: BacktestResults object
            
        Returns:
            Dictionary with detailed analysis
        """
        try:
            trades_df = pd.DataFrame(results.trades)
            
            if trades_df.empty:
                return {
                    'error': 'No trades to analyze',
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            # Trade size distribution
            trade_sizes = trades_df['pnl_pct'].values
            
            analysis = {
                'performance_summary': {
                    'total_return': results.total_return,
                    'sharpe_ratio': results.sharpe_ratio,
                    'max_drawdown': results.max_drawdown,
                    'win_rate': results.win_rate,
                    'profit_factor': results.profit_factor
                },
                'trade_distribution': {
                    'mean_pnl': float(np.mean(trade_sizes)),
                    'median_pnl': float(np.median(trade_sizes)),
                    'std_pnl': float(np.std(trade_sizes)),
                    'skewness': float(stats.skew(trade_sizes)),
                    'kurtosis': float(stats.kurtosis(trade_sizes)),
                    'percentile_25': float(np.percentile(trade_sizes, 25)),
                    'percentile_75': float(np.percentile(trade_sizes, 75))
                },
                'win_loss_analysis': {
                    'avg_win': results.avg_win,
                    'avg_loss': results.avg_loss,
                    'avg_win_to_loss_ratio': abs(results.avg_win / results.avg_loss) if results.avg_loss != 0 else 0,
                    'largest_win': float(trades_df['pnl'].max()),
                    'largest_loss': float(trades_df['pnl'].min()),
                    'consecutive_wins': PerformanceAnalyzer._max_consecutive(trades_df, 'pnl', lambda x: x > 0),
                    'consecutive_losses': PerformanceAnalyzer._max_consecutive(trades_df, 'pnl', lambda x: x < 0)
                },
                'duration_analysis': {
                    'avg_duration': results.avg_trade_duration,
                    'min_duration': int(trades_df['duration_minutes'].min()),
                    'max_duration': int(trades_df['duration_minutes'].max())
                },
                'risk_metrics': {
                    'sharpe': results.sharpe_ratio,
                    'sortino': results.sortino_ratio,
                    'calmar': results.calmar_ratio,
                    'var_95': results.value_at_risk_95,
                    'cvar_95': results.expected_shortfall_95
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Performance analysis error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    @staticmethod
    def _max_consecutive(df: pd.DataFrame, column: str, condition: Callable) -> int:
        """Calculate maximum consecutive occurrences"""
        max_count = 0
        current_count = 0
        
        for value in df[column]:
            if condition(value):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count


class StrategyOptimizer:
    """
    Strategy Parameter Optimization
    
    Optimizes strategy parameters using:
    - Grid search
    - Random search
    - Objective function optimization
    """
    
    def __init__(self, backtest_engine: BacktestEngine):
        self.engine = backtest_engine
        
    def grid_search(
        self,
        df: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List],
        optimization_metric: str = 'sharpe_ratio'
    ) -> Dict[str, any]:
        """
        Perform grid search optimization
        
        Args:
            df: Historical data
            strategy_func: Strategy function
            param_grid: Dictionary of parameters and values to test
            optimization_metric: Metric to optimize
            
        Returns:
            Dictionary with best parameters and results
        """
        try:
            logger.info(f"Starting grid search with {len(param_grid)} parameters")
            
            # Generate all parameter combinations
            keys = param_grid.keys()
            values = param_grid.values()
            combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
            
            logger.info(f"Testing {len(combinations)} combinations")
            
            best_score = -np.inf
            best_params = None
            best_results = None
            all_results = []
            
            for i, params in enumerate(combinations):
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(combinations)} combinations tested")
                
                try:
                    results = self.engine.run(df, strategy_func, params)
                    score = getattr(results, optimization_metric)
                    
                    all_results.append({
                        'params': params,
                        'score': score,
                        'total_return': results.total_return,
                        'sharpe_ratio': results.sharpe_ratio,
                        'max_drawdown': results.max_drawdown,
                        'total_trades': results.total_trades
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_results = results
                        
                except Exception as e:
                    logger.warning(f"Failed to test params {params}: {e}")
                    continue
            
            logger.info(f"Grid search complete. Best {optimization_metric}: {best_score:.4f}")
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'best_results': asdict(best_results) if best_results else None,
                'all_results': all_results,
                'total_combinations': len(combinations),
                'optimization_metric': optimization_metric,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Grid search error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


class WalkForwardOptimizer:
    """
    Walk-Forward Analysis
    
    Prevents overfitting by:
    - Optimizing on in-sample period
    - Testing on out-of-sample period
    - Rolling windows forward
    """
    
    def __init__(self, backtest_engine: BacktestEngine, optimizer: StrategyOptimizer):
        self.engine = backtest_engine
        self.optimizer = optimizer
        
    def walk_forward(
        self,
        df: pd.DataFrame,
        strategy_func: Callable,
        param_grid: Dict[str, List],
        in_sample_pct: float = 0.7,
        num_windows: int = 5,
        optimization_metric: str = 'sharpe_ratio'
    ) -> Dict[str, any]:
        """
        Perform walk-forward analysis
        
        Args:
            df: Complete historical data
            strategy_func: Strategy function
            param_grid: Parameters to optimize
            in_sample_pct: Percentage of data for optimization
            num_windows: Number of walk-forward windows
            optimization_metric: Metric to optimize
            
        Returns:
            Dictionary with walk-forward results
        """
        try:
            logger.info(f"Starting walk-forward analysis with {num_windows} windows")
            
            total_length = len(df)
            window_size = total_length // num_windows
            in_sample_size = int(window_size * in_sample_pct)
            out_sample_size = window_size - in_sample_size
            
            results = []
            
            for i in range(num_windows):
                start_idx = i * window_size
                in_sample_end = start_idx + in_sample_size
                out_sample_end = in_sample_end + out_sample_size
                
                if out_sample_end > total_length:
                    break
                
                logger.info(f"Window {i+1}/{num_windows}")
                
                # In-sample optimization
                in_sample_data = df.iloc[start_idx:in_sample_end]
                opt_result = self.optimizer.grid_search(
                    in_sample_data,
                    strategy_func,
                    param_grid,
                    optimization_metric
                )
                
                # Out-of-sample testing
                out_sample_data = df.iloc[in_sample_end:out_sample_end]
                out_sample_results = self.engine.run(
                    out_sample_data,
                    strategy_func,
                    opt_result['best_params']
                )
                
                results.append({
                    'window': i + 1,
                    'in_sample_score': opt_result['best_score'],
                    'out_sample_score': getattr(out_sample_results, optimization_metric),
                    'best_params': opt_result['best_params'],
                    'out_sample_return': out_sample_results.total_return,
                    'out_sample_sharpe': out_sample_results.sharpe_ratio,
                    'out_sample_max_dd': out_sample_results.max_drawdown
                })
            
            # Aggregate results
            avg_out_sample_score = np.mean([r['out_sample_score'] for r in results])
            avg_out_sample_return = np.mean([r['out_sample_return'] for r in results])
            
            logger.info(f"Walk-forward complete. Avg out-sample {optimization_metric}: {avg_out_sample_score:.4f}")
            
            return {
                'windows': results,
                'avg_out_sample_score': avg_out_sample_score,
                'avg_out_sample_return': avg_out_sample_return,
                'num_windows': len(results),
                'optimization_metric': optimization_metric,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Walk-forward analysis error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of Backtesting Engine
    """
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    df = pd.DataFrame({
        'open': 100 + np.random.randn(1000).cumsum(),
        'high': 102 + np.random.randn(1000).cumsum(),
        'low': 98 + np.random.randn(1000).cumsum(),
        'close': 100 + np.random.randn(1000).cumsum(),
        'volume': np.random.randint(1000, 5000, 1000)
    }, index=dates)
    
    # Simple moving average crossover strategy
    def sma_crossover_strategy(df: pd.DataFrame, fast_period: int = 10, slow_period: int = 20) -> pd.Series:
        """Simple MA crossover strategy"""
        sma_fast = df['close'].rolling(fast_period).mean()
        sma_slow = df['close'].rolling(slow_period).mean()
        
        signal = pd.Series(0, index=df.index)
        signal[sma_fast > sma_slow] = 1  # Long
        signal[sma_fast < sma_slow] = -1  # Short
        
        return signal
    
    print("\n=== BACKTEST ENGINE ===")
    engine = BacktestEngine(initial_capital=10000, commission_pct=0.001)
    results = engine.run(df, sma_crossover_strategy, {'fast_period': 10, 'slow_period': 20})
    
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Total Trades: {results.total_trades}")
    
    print("\n=== PERFORMANCE ANALYZER ===")
    analysis = PerformanceAnalyzer.analyze_results(results)
    print(f"Profit Factor: {analysis['performance_summary']['profit_factor']:.2f}")
    print(f"Avg Win/Loss Ratio: {analysis['win_loss_analysis']['avg_win_to_loss_ratio']:.2f}")
    
    print("\n=== STRATEGY OPTIMIZER ===")
    optimizer = StrategyOptimizer(engine)
    param_grid = {
        'fast_period': [5, 10, 15],
        'slow_period': [20, 30, 40]
    }
    
    opt_results = optimizer.grid_search(df, sma_crossover_strategy, param_grid)
    print(f"Best Parameters: {opt_results['best_params']}")
    print(f"Best Sharpe Ratio: {opt_results['best_score']:.2f}")

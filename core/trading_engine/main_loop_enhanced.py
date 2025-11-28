"""
Main Trading Loop (Phase 3.5: Enhanced with Logging & Resilience)
=================================================================
Full production trading loop with:
- Real-time Binance data feed
- Enhanced signal aggregation
- Dynamic position sizing
- Order execution (paper/live)
- Database trade logging
- Advanced error recovery
- Live dashboard broadcasting
- Telegram alerts
- Zero mock/fallback

Author: DEMIR AI PRO
Version: 8.0
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
import pandas as pd

from config import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    DATABASE_URL,
    TELEGRAM_TOKEN,
    TELEGRAM_CHAT_ID,
    ADVISORY_MODE_ENABLED,
    DEBUG_MODE,
)

# Import core modules with graceful error handling
try:
    from core.signal_processor.enhanced_aggregator import EnhancedSignalAggregator
    from core.risk_manager.dynamic_sizing import DynamicPositionSizer
    from core.signal_processor.layers.sentiment.emergency_events import (
        EmergencyEventDetector,
        EmergencyActionHandler
    )
except ImportError as e:
    logging.error(f"❌ Core module import failed: {e}")

try:
    from integrations.binance.market_data import BinanceMarketData
    from core.trading_engine.order_router import OrderRouter
    from integrations.notifications.telegram_alert import TelegramAlert
except ImportError as e:
    logging.error(f"❌ Integration module import failed: {e}")

# Phase 3.5 imports
try:
    from database.trade_logger import TradeLogger
    from core.monitoring.resilience_manager import get_resilience_manager, CircuitBreakerConfig
    from api.dashboard_api import broadcast_trade_update, broadcast_pnl_update, broadcast_performance_update
except ImportError as e:
    logging.error(f"❌ Phase 3.5 module import failed: {e}")

logger = logging.getLogger("trading_engine.main_loop_enhanced")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class EnhancedTradingEngine:
    """Production-grade trading engine with full monitoring & resilience"""
    
    def __init__(
        self,
        symbol: str,
        account_balance: float,
        mode: str = "PAPER"  # PAPER or LIVE
    ):
        self.symbol = symbol.upper()
        self.account_balance = account_balance
        self.mode = mode
        
        # Core components
        try:
            self.aggregator = EnhancedSignalAggregator()
            self.position_sizer = DynamicPositionSizer()
            self.emergency_detector = EmergencyEventDetector()
            self.emergency_handler = EmergencyActionHandler(self.emergency_detector)
            self.market_data = BinanceMarketData(symbol)
        except Exception as e:
            logger.error(f"❌ Failed to initialize core components: {e}")
            raise
        
        # Execution & Notifications
        try:
            self.order_router = OrderRouter(mode=mode)
            self.telegram = TelegramAlert()
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize execution components: {e}")
        
        # Phase 3.5: Logging & Resilience
        try:
            self.trade_logger = TradeLogger()
            self.resilience = get_resilience_manager()
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize Phase 3.5 components: {e}")
        
        # State
        self.current_position = None
        self.current_trade_id: Optional[int] = None
        self.running = True
        self.total_pnl = 0.0
        
        logger.info(
            f"🚀 EnhancedTradingEngine initialized: {self.symbol} | "
            f"Mode: {mode} | Balance: ${account_balance:,.2f}"
        )
    
    async def trading_loop(self, poll_interval: int = 60):
        """
        Main trading loop with resilience
        
        Args:
            poll_interval: Seconds between iterations (default: 60)
        """
        logger.info("▶️  Starting enhanced trading loop...")
        
        # Send startup notification
        try:
            await self.telegram.send_alert(
                f"🚀 *DEMIR AI PRO v8.0 Enhanced Started*\n"
                f"Symbol: `{self.symbol}`\n"
                f"Mode: `{self.mode}`\n"
                f"Balance: `${self.account_balance:,.2f}`\n"
                f"Features: Database Logging + Error Recovery + Live Dashboard"
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to send startup notification: {e}")
        
        while self.running:
            try:
                await self._execute_trading_iteration(poll_interval)
            except Exception as e:
                logger.error(f"❌ Trading iteration error: {e}", exc_info=True)
                
                # Send error notification
                try:
                    await self.telegram.send_alert(f"❌ *Trading Loop Error*\n{str(e)[:200]}")
                except Exception as telegram_error:
                    logger.warning(f"⚠️  Failed to send error notification: {telegram_error}")
                
                await asyncio.sleep(poll_interval // 2)
    
    async def _execute_trading_iteration(self, poll_interval: int):
        """Single trading iteration with resilience"""
        
        # =================
        # 1. DATA FETCH (with circuit breaker)
        # =================
        
        try:
            ohlcv = await self.resilience.safe_call(
                self.market_data.fetch_ohlcv,
                circuit_name="binance_ohlcv",
                circuit_config=CircuitBreakerConfig(failure_threshold=3, timeout=30),
                max_retries=2,
                interval="1m",
                limit=150
            )
            
            orderbook = await self.resilience.safe_call(
                self.market_data.fetch_orderbook_snapshot,
                circuit_name="binance_orderbook",
                circuit_config=CircuitBreakerConfig(failure_threshold=3, timeout=30),
                max_retries=2,
                symbol=self.symbol,
                limit=20
            )
        except Exception as e:
            logger.error(f"❌ Data fetch error: {e}")
            await asyncio.sleep(poll_interval)
            return
        
        if not ohlcv or not orderbook or not orderbook.get("bids") or not orderbook.get("asks"):
            logger.warning("⚠️  Data fetch incomplete - skipping iteration")
            await asyncio.sleep(poll_interval)
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv)
        df = df[["open", "high", "low", "close", "volume"]]
        current_price = df["close"].iloc[-1]
        
        # =================
        # 2. EMERGENCY CHECK
        # =================
        
        try:
            if self.emergency_handler.is_trading_halted():
                logger.error("🚨 TRADING HALTED - Emergency active!")
                
                # Close any open positions
                if self.current_position:
                    await self._emergency_close_position(current_price)
                
                await asyncio.sleep(poll_interval * 2)
                return
        except Exception as e:
            logger.warning(f"⚠️  Emergency check failed: {e}")
        
        # =================
        # 3. SIGNAL GENERATION
        # =================
        
        try:
            signal_data = self.aggregator.generate_signal(df, orderbook, current_price)
            
            logger.info(
                f"📊 Signal: {signal_data.signal} | "
                f"Confidence: {signal_data.confidence:.0%} | "
                f"Regime: {signal_data.regime} | "
                f"Reason: {signal_data.reason}"
            )
        except Exception as e:
            logger.error(f"❌ Signal generation error: {e}")
            await asyncio.sleep(poll_interval)
            return
        
        # =================
        # 4. POSITION SIZING
        # =================
        
        try:
            atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
            
            pos_result = self.position_sizer.calculate_position(
                account_balance=self.account_balance,
                entry_price=current_price,
                stop_loss_price=signal_data.stop_loss,
                atr=atr,
                regime=signal_data.regime
            )
            
            logger.info(
                f"💰 Position size: {pos_result.size:.6f} {self.symbol} "
                f"(${pos_result.size_usd:,.0f}) | {pos_result.reason}"
            )
        except Exception as e:
            logger.error(f"❌ Position sizing error: {e}")
            await asyncio.sleep(poll_interval)
            return
        
        # =================
        # 5. EXECUTION LOGIC
        # =================
        
        try:
            await self._execute_trading_logic(signal_data, pos_result, current_price)
        except Exception as e:
            logger.error(f"❌ Trading logic error: {e}")
            await asyncio.sleep(poll_interval)
            return
        
        # =================
        # 6. UPDATE POSITION P&L & LOG
        # =================
        
        try:
            if self.current_position:
                self.order_router.update_unrealized_pnl(self.symbol, current_price)
                position = self.order_router.get_position(self.symbol)
                unrealized_pnl = position.unrealized_pnl if position else 0.0
                
                logger.info(f"💵 Unrealized P&L: ${unrealized_pnl:,.2f}")
                
                # Log position snapshot to database
                if position:
                    await self.trade_logger.log_position_snapshot(
                        symbol=self.symbol,
                        side=position.side,
                        entry_price=position.entry_price,
                        current_price=current_price,
                        quantity=position.size,
                        unrealized_pnl=unrealized_pnl,
                        stop_loss=position.stop_loss,
                        take_profit=position.take_profit
                    )
                
                # Broadcast to dashboard
                await broadcast_pnl_update({
                    "total_pnl": self.total_pnl + unrealized_pnl,
                    "unrealized_pnl": unrealized_pnl,
                    "pnl_percent": ((self.total_pnl + unrealized_pnl) / self.account_balance * 100)
                })
        except Exception as e:
            logger.warning(f"⚠️  Position logging error: {e}")
        
        # Wait for next iteration
        await asyncio.sleep(poll_interval)
    
    async def _execute_trading_logic(self, signal_data, pos_result, current_price):
        """Execute trading decisions with database logging"""
        
        # Open LONG position
        if signal_data.signal == "LONG" and not self.current_position:
            if signal_data.confidence >= 0.65:
                logger.info(f"✅ Opening LONG position")
                
                try:
                    result = await self.order_router.execute_order(
                        symbol=self.symbol,
                        side="BUY",
                        size=pos_result.size,
                        price=current_price,
                        stop_loss=signal_data.stop_loss,
                        take_profit=signal_data.take_profit
                    )
                    
                    if result.success:
                        self.current_position = "LONG"
                        
                        # Log to database
                        self.current_trade_id = await self.trade_logger.log_trade_open(
                            symbol=self.symbol,
                            side="LONG",
                            order_type="MARKET",
                            entry_price=result.filled_price,
                            quantity=result.filled_size,
                            commission=result.commission,
                            stop_loss=signal_data.stop_loss,
                            take_profit=signal_data.take_profit,
                            signal_confidence=signal_data.confidence,
                            regime=signal_data.regime,
                            order_id=result.order_id,
                            metadata={"reason": signal_data.reason}
                        )
                        
                        # Send notifications
                        await self.telegram.send_trade_alert(
                            action="OPEN",
                            symbol=self.symbol,
                            size=result.filled_size,
                            price=result.filled_price,
                            stop_loss=signal_data.stop_loss,
                            take_profit=signal_data.take_profit
                        )
                        
                        # Broadcast to dashboard
                        await broadcast_trade_update({
                            "action": "OPEN",
                            "symbol": self.symbol,
                            "side": "LONG",
                            "price": result.filled_price,
                            "size": result.filled_size
                        })
                    else:
                        logger.error(f"❌ Order failed: {result.error_message}")
                except Exception as e:
                    logger.error(f"❌ Open LONG error: {e}")
        
        # Close LONG position
        elif signal_data.signal == "SHORT" and self.current_position == "LONG":
            logger.info(f"✅ Closing LONG position")
            
            try:
                result = await self.order_router.execute_order(
                    symbol=self.symbol,
                    side="SELL",
                    size=pos_result.size,
                    price=current_price
                )
                
                if result.success:
                    # Get final P&L
                    position = self.order_router.get_position(self.symbol)
                    pnl = position.realized_pnl if position else 0.0
                    pnl_percent = (pnl / (position.entry_price * position.size) * 100) if position else 0.0
                    
                    self.total_pnl += pnl
                    self.current_position = None
                    
                    # Log to database
                    if self.current_trade_id:
                        await self.trade_logger.log_trade_close(
                            trade_id=self.current_trade_id,
                            exit_price=result.filled_price,
                            pnl=pnl,
                            pnl_percent=pnl_percent
                        )
                        self.current_trade_id = None
                    
                    # Send notifications
                    await self.telegram.send_trade_alert(
                        action="CLOSE",
                        symbol=self.symbol,
                        size=result.filled_size,
                        price=result.filled_price,
                        pnl=pnl
                    )
                    
                    # Broadcast to dashboard
                    await broadcast_trade_update({
                        "action": "CLOSE",
                        "symbol": self.symbol,
                        "price": result.filled_price,
                        "pnl": pnl
                    })
                    
                    # Update performance metrics
                    summary = await self.trade_logger.get_performance_summary()
                    await broadcast_performance_update(summary)
                    
                    logger.info(f"💰 Position closed | Trade P&L: ${pnl:,.2f} | Total P&L: ${self.total_pnl:,.2f}")
                else:
                    logger.error(f"❌ Close order failed: {result.error_message}")
            except Exception as e:
                logger.error(f"❌ Close LONG error: {e}")
        
        # Check stop loss / take profit hits
        elif self.current_position:
            try:
                await self._check_exit_conditions(current_price)
            except Exception as e:
                logger.error(f"❌ Exit condition check error: {e}")
    
    async def _check_exit_conditions(self, current_price: float):
        """Check if stop loss or take profit hit"""
        position = self.order_router.get_position(self.symbol)
        if not position:
            return
        
        # Stop loss hit
        if position.stop_loss and current_price <= position.stop_loss:
            logger.warning(f"🛑 Stop loss hit at ${current_price:,.2f}")
            
            try:
                result = await self.order_router.execute_order(
                    symbol=self.symbol,
                    side="SELL",
                    size=position.size,
                    price=current_price
                )
                
                if result.success:
                    pnl = position.realized_pnl
                    pnl_percent = (pnl / (position.entry_price * position.size) * 100)
                    self.total_pnl += pnl
                    self.current_position = None
                    
                    # Log to database
                    if self.current_trade_id:
                        await self.trade_logger.log_trade_close(
                            trade_id=self.current_trade_id,
                            exit_price=result.filled_price,
                            pnl=pnl,
                            pnl_percent=pnl_percent
                        )
                        self.current_trade_id = None
                    
                    await self.telegram.send_alert(
                        f"🛑 *Stop Loss Hit*\n"
                        f"Symbol: `{self.symbol}`\n"
                        f"Exit: `${current_price:,.2f}`\n"
                        f"P&L: `${pnl:,.2f}`"
                    )
            except Exception as e:
                logger.error(f"❌ Stop loss close error: {e}")
        
        # Take profit hit
        elif position.take_profit and current_price >= position.take_profit:
            logger.info(f"🎯 Take profit hit at ${current_price:,.2f}")
            
            try:
                result = await self.order_router.execute_order(
                    symbol=self.symbol,
                    side="SELL",
                    size=position.size,
                    price=current_price
                )
                
                if result.success:
                    pnl = position.realized_pnl
                    pnl_percent = (pnl / (position.entry_price * position.size) * 100)
                    self.total_pnl += pnl
                    self.current_position = None
                    
                    # Log to database
                    if self.current_trade_id:
                        await self.trade_logger.log_trade_close(
                            trade_id=self.current_trade_id,
                            exit_price=result.filled_price,
                            pnl=pnl,
                            pnl_percent=pnl_percent
                        )
                        self.current_trade_id = None
                    
                    await self.telegram.send_alert(
                        f"🎯 *Take Profit Hit*\n"
                        f"Symbol: `{self.symbol}`\n"
                        f"Exit: `${current_price:,.2f}`\n"
                        f"P&L: `${pnl:,.2f}`"
                    )
            except Exception as e:
                logger.error(f"❌ Take profit close error: {e}")
    
    async def _emergency_close_position(self, current_price: float):
        """Emergency position closure with logging"""
        position = self.order_router.get_position(self.symbol)
        if not position:
            return
        
        logger.error(f"🚨 EMERGENCY CLOSE - Closing position at ${current_price:,.2f}")
        
        try:
            result = await self.order_router.execute_order(
                symbol=self.symbol,
                side="SELL",
                size=position.size,
                price=current_price
            )
            
            if result.success:
                pnl = position.realized_pnl
                pnl_percent = (pnl / (position.entry_price * position.size) * 100)
                self.total_pnl += pnl
                self.current_position = None
                
                # Log to database
                if self.current_trade_id:
                    await self.trade_logger.log_trade_close(
                        trade_id=self.current_trade_id,
                        exit_price=result.filled_price,
                        pnl=pnl,
                        pnl_percent=pnl_percent
                    )
                    self.current_trade_id = None
                
                await self.telegram.send_emergency_alert(
                    event_type="FORCED_CLOSE",
                    title="Emergency Position Closure",
                    description=f"Position closed due to emergency event. P&L: ${pnl:,.2f}"
                )
        except Exception as e:
            logger.error(f"❌ Emergency close error: {e}")
    
    async def stop(self):
        """Graceful shutdown with cleanup"""
        logger.info("🛑 Stopping enhanced trading engine...")
        self.running = False
        
        # Close any open positions
        if self.current_position:
            try:
                ohlcv = await self.market_data.fetch_ohlcv(interval="1m", limit=1)
                if ohlcv:
                    current_price = ohlcv[0]["close"]
                    await self._emergency_close_position(current_price)
            except Exception as e:
                logger.warning(f"⚠️  Failed to close position during shutdown: {e}")
        
        # Cleanup
        try:
            await self.order_router.close()
            await self.trade_logger.close()
        except Exception as e:
            logger.warning(f"⚠️  Cleanup error: {e}")
        
        # Send final report
        try:
            summary = await self.trade_logger.get_performance_summary()
            
            await self.telegram.send_alert(
                f"🛑 *Enhanced Trading Engine Stopped*\n\n"
                f"Total P&L: `${self.total_pnl:,.2f}`\n"
                f"Final Balance: `${self.account_balance + self.total_pnl:,.2f}`\n\n"
                f"Total Trades: `{summary.get('total_trades', 0)}`\n"
                f"Win Rate: `{summary.get('win_rate', 0):.1f}%`\n"
                f"Profit Factor: `{summary.get('profit_factor', 0):.2f}`"
            )
        except Exception as e:
            logger.warning(f"⚠️  Failed to send final report: {e}")
        
        logger.info("✅ Enhanced trading engine stopped gracefully")


async def main():
    """Main entry point"""
    import os
    
    ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000"))
    SYMBOL = os.getenv("DEFAULT_SYMBOL", "BTCUSDT")
    MODE = os.getenv("TRADING_MODE", "PAPER")
    
    engine = EnhancedTradingEngine(
        symbol=SYMBOL,
        account_balance=ACCOUNT_BALANCE,
        mode=MODE
    )
    
    try:
        await engine.trading_loop(poll_interval=60)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Keyboard interrupt received")
        await engine.stop()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())

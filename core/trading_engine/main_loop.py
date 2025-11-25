"""
Main Trading Loop (Phase 3: Full Execution Integration)
=======================================================
Full async, production-ready, live trading loop with:
- Real-time Binance data feed
- EnhancedSignalAggregator + DynamicPositionSizer
- OrderRouter (paper/live execution)
- Telegram alerts for all events
- Emergency event protection
- Zero fallback/mock - pure production

Author: DEMIR AI PRO
Version: 8.0
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
import pandas as pd

from core.signal_processor.enhanced_aggregator import EnhancedSignalAggregator
from core.risk_manager.dynamic_sizing import DynamicPositionSizer
from core.signal_processor.layers.sentiment.emergency_events import (
    EmergencyEventDetector,
    EmergencyActionHandler
)
from integrations.binance.market_data import BinanceMarketData
from core.trading_engine.order_router import OrderRouter
from integrations.notifications.telegram_alert import TelegramAlert

logger = logging.getLogger("trading_engine.main_loop")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class TradingEngine:
    """Production-grade trading engine with full execution"""
    
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
        self.aggregator = EnhancedSignalAggregator()
        self.position_sizer = DynamicPositionSizer()
        self.emergency_detector = EmergencyEventDetector()
        self.emergency_handler = EmergencyActionHandler(self.emergency_detector)
        self.market_data = BinanceMarketData(symbol)
        
        # Execution & Notifications
        self.order_router = OrderRouter(mode=mode)
        self.telegram = TelegramAlert()
        
        # State
        self.current_position = None
        self.running = True
        self.total_pnl = 0.0
        
        logger.info(f"🚀 TradingEngine initialized: {self.symbol} | Mode: {mode} | Balance: ${account_balance:,.2f}")
    
    async def trading_loop(self, poll_interval: int = 60):
        """
        Main trading loop
        
        Args:
            poll_interval: Seconds between iterations (default: 60)
        """
        logger.info("▶️  Starting trading loop...")
        
        # Send startup notification
        await self.telegram.send_alert(
            f"🚀 *DEMIR AI PRO v8.0 Started*\n"
            f"Symbol: `{self.symbol}`\n"
            f"Mode: `{self.mode}`\n"
            f"Balance: `${self.account_balance:,.2f}`"
        )
        
        while self.running:
            try:
                # =================
                # 1. DATA FETCH
                # =================
                ohlcv = await self.market_data.fetch_ohlcv(interval="1m", limit=150)
                orderbook = await self.market_data.fetch_orderbook_snapshot(self.symbol, limit=20)
                
                if not ohlcv or not orderbook['bids'] or not orderbook['asks']:
                    logger.warning("⚠️  Data fetch incomplete - skipping iteration")
                    await asyncio.sleep(poll_interval)
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(ohlcv)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                current_price = df['close'].iloc[-1]
                
                # =================
                # 2. EMERGENCY CHECK
                # =================
                if self.emergency_handler.is_trading_halted():
                    logger.error("🚨 TRADING HALTED - Emergency active!")
                    
                    # Close any open positions
                    if self.current_position:
                        await self._emergency_close_position(current_price)
                    
                    await asyncio.sleep(poll_interval * 2)
                    continue
                
                # =================
                # 3. SIGNAL GENERATION
                # =================
                signal_data = self.aggregator.generate_signal(df, orderbook, current_price)
                
                logger.info(
                    f"📊 Signal: {signal_data.signal} | "
                    f"Confidence: {signal_data.confidence:.0%} | "
                    f"Regime: {signal_data.regime} | "
                    f"Reason: {signal_data.reason}"
                )
                
                # Send signal notification (non-critical, silent)
                await self.telegram.send_alert(
                    f"📊 Signal: `{signal_data.signal}` | "
                    f"Conf: `{signal_data.confidence:.0%}` | "
                    f"Regime: `{signal_data.regime}`",
                    silent=True
                )
                
                # =================
                # 4. POSITION SIZING
                # =================
                atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
                
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
                
                # =================
                # 5. EXECUTION LOGIC
                # =================
                await self._execute_trading_logic(
                    signal_data,
                    pos_result,
                    current_price
                )
                
                # =================
                # 6. UPDATE POSITION P&L
                # =================
                if self.current_position:
                    self.order_router.update_unrealized_pnl(self.symbol, current_price)
                    unrealized_pnl = self.order_router.get_total_pnl()
                    
                    logger.info(f"💵 Unrealized P&L: ${unrealized_pnl:,.2f}")
                
                # Wait for next iteration
                await asyncio.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}", exc_info=True)
                
                # Send error notification
                await self.telegram.send_alert(
                    f"❌ *Trading Loop Error*\n{str(e)[:200]}"
                )
                
                await asyncio.sleep(poll_interval // 2)
    
    async def _execute_trading_logic(self, signal_data, pos_result, current_price):
        """
        Execute trading decisions based on signals
        
        Args:
            signal_data: Signal from aggregator
            pos_result: Position sizing result
            current_price: Current market price
        """
        # Open LONG position
        if signal_data.signal == "LONG" and not self.current_position:
            if signal_data.confidence >= 0.65:  # Minimum confidence threshold
                logger.info(f"✅ Opening LONG position")
                
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
                    
                    # Send trade notification
                    await self.telegram.send_trade_alert(
                        action="OPEN",
                        symbol=self.symbol,
                        size=result.filled_size,
                        price=result.filled_price,
                        stop_loss=signal_data.stop_loss,
                        take_profit=signal_data.take_profit
                    )
                else:
                    logger.error(f"❌ Order failed: {result.error_message}")
        
        # Close LONG position
        elif signal_data.signal == "SHORT" and self.current_position == "LONG":
            logger.info(f"✅ Closing LONG position")
            
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
                
                self.total_pnl += pnl
                self.current_position = None
                
                # Send trade notification with P&L
                await self.telegram.send_trade_alert(
                    action="CLOSE",
                    symbol=self.symbol,
                    size=result.filled_size,
                    price=result.filled_price,
                    pnl=pnl
                )
                
                logger.info(f"💰 Position closed | Trade P&L: ${pnl:,.2f} | Total P&L: ${self.total_pnl:,.2f}")
            else:
                logger.error(f"❌ Close order failed: {result.error_message}")
        
        # Check stop loss / take profit hits
        elif self.current_position:
            await self._check_exit_conditions(current_price)
    
    async def _check_exit_conditions(self, current_price: float):
        """
        Check if stop loss or take profit hit
        
        Args:
            current_price: Current market price
        """
        position = self.order_router.get_position(self.symbol)
        if not position:
            return
        
        # Stop loss hit
        if position.stop_loss and current_price <= position.stop_loss:
            logger.warning(f"🛑 Stop loss hit at ${current_price:,.2f}")
            
            result = await self.order_router.execute_order(
                symbol=self.symbol,
                side="SELL",
                size=position.size,
                price=current_price
            )
            
            if result.success:
                pnl = position.realized_pnl
                self.total_pnl += pnl
                self.current_position = None
                
                await self.telegram.send_alert(
                    f"🛑 *Stop Loss Hit*\n"
                    f"Symbol: `{self.symbol}`\n"
                    f"Exit: `${current_price:,.2f}`\n"
                    f"P&L: `${pnl:,.2f}`"
                )
        
        # Take profit hit
        elif position.take_profit and current_price >= position.take_profit:
            logger.info(f"🎯 Take profit hit at ${current_price:,.2f}")
            
            result = await self.order_router.execute_order(
                symbol=self.symbol,
                side="SELL",
                size=position.size,
                price=current_price
            )
            
            if result.success:
                pnl = position.realized_pnl
                self.total_pnl += pnl
                self.current_position = None
                
                await self.telegram.send_alert(
                    f"🎯 *Take Profit Hit*\n"
                    f"Symbol: `{self.symbol}`\n"
                    f"Exit: `${current_price:,.2f}`\n"
                    f"P&L: `${pnl:,.2f}`"
                )
    
    async def _emergency_close_position(self, current_price: float):
        """
        Emergency position closure
        
        Args:
            current_price: Current market price
        """
        position = self.order_router.get_position(self.symbol)
        if not position:
            return
        
        logger.error(f"🚨 EMERGENCY CLOSE - Closing position at ${current_price:,.2f}")
        
        result = await self.order_router.execute_order(
            symbol=self.symbol,
            side="SELL",
            size=position.size,
            price=current_price
        )
        
        if result.success:
            pnl = position.realized_pnl
            self.total_pnl += pnl
            self.current_position = None
            
            await self.telegram.send_emergency_alert(
                event_type="FORCED_CLOSE",
                title="Emergency Position Closure",
                description=f"Position closed due to emergency event. P&L: ${pnl:,.2f}"
            )
    
    async def stop(self):
        """Graceful shutdown"""
        logger.info("🛑 Stopping trading engine...")
        self.running = False
        
        # Close any open positions
        if self.current_position:
            # Get current price for final close
            ohlcv = await self.market_data.fetch_ohlcv(interval="1m", limit=1)
            if ohlcv:
                current_price = ohlcv[0]['close']
                await self._emergency_close_position(current_price)
        
        # Cleanup
        await self.order_router.close()
        
        # Send final report
        await self.telegram.send_alert(
            f"🛑 *Trading Engine Stopped*\n"
            f"Total P&L: `${self.total_pnl:,.2f}`\n"
            f"Final Balance: `${self.account_balance + self.total_pnl:,.2f}`"
        )
        
        logger.info("✅ Trading engine stopped gracefully")


async def main():
    """Main entry point"""
    ACCOUNT_BALANCE = 10000.0
    SYMBOL = "BTCUSDT"
    MODE = "PAPER"  # Change to "LIVE" for real trading
    
    engine = TradingEngine(
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

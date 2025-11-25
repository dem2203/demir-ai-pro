# DEMIR AI PRO v8.0 - Phase 3 Deployment Guide

## 🚀 Production Deployment Checklist

### Prerequisites

- [x] GitHub repository updated with Phase 3 code
- [x] Railway account connected to GitHub
- [ ] Binance API keys (with trading permissions)
- [ ] Telegram bot configured
- [ ] PostgreSQL database ready

---

## Step 1: Railway Environment Variables

### Navigate to Railway Dashboard

1. Go to your project: `demir-ai-pro`
2. Click **Variables** tab
3. Add the following:

### Required Variables

```bash
# Binance API
BINANCE_API_KEY=<your_key>
BINANCE_SECRET_KEY=<your_secret>

# Telegram
TELEGRAM_BOT_TOKEN=<your_bot_token>
TELEGRAM_CHAT_ID=<your_chat_id>

# Database (auto-provided by Railway PostgreSQL addon)
DATABASE_URL=<auto_generated>

# Trading Config
TRADING_MODE=PAPER  # Start with PAPER!
DEFAULT_SYMBOL=BTCUSDT
ACCOUNT_BALANCE=10000
POLL_INTERVAL=60

# Risk Management
MAX_POSITION_SIZE_PCT=5
MAX_LEVERAGE=3
MIN_CONFIDENCE=0.65

# System
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### How to Get Telegram Credentials

**Bot Token:**
1. Open Telegram, search for `@BotFather`
2. Send `/newbot`
3. Follow instructions, copy token

**Chat ID:**
1. Search for `@userinfobot` on Telegram
2. Send `/start`
3. Copy your chat ID

---

## Step 2: Paper Trading Test (48 Hours)

### Deploy to Railway

```bash
# Railway will auto-deploy from main branch
# Or manually trigger:
railway up
```

### Monitor Paper Trading

1. **Railway Logs:**
   - Dashboard → Deployments → Logs
   - Watch for trading signals and executions

2. **Telegram Alerts:**
   - You'll receive:
     - Startup notification
     - Hourly performance updates
     - Trade execution alerts
     - Emergency notifications

3. **Expected Behavior:**
   - System fetches BTCUSDT data every 60s
   - Generates signals (LONG/SHORT/NEUTRAL)
   - Executes paper trades when confidence ≥ 65%
   - Reports P&L hourly

### Local Testing (Optional)

```bash
# Set up local environment
cp .env.example .env
# Fill in your credentials in .env

# Run paper trading test
python tests/paper_trading_test.py --duration 48 --symbol BTCUSDT

# Or run main loop directly
python core/trading_engine/main_loop.py
```

---

## Step 3: Performance Evaluation

### After 48 Hours, Check:

**✅ Success Criteria:**
- [ ] No critical errors in logs
- [ ] Signals generated regularly
- [ ] Trades executed successfully
- [ ] P&L tracking works
- [ ] Stop loss / Take profit triggered correctly
- [ ] Telegram alerts received
- [ ] Win rate ≥ 50%
- [ ] Profit factor ≥ 1.0
- [ ] Max drawdown ≤ 10%

**❌ Red Flags:**
- Frequent API errors
- Missing data
- Execution failures
- Unrealistic P&L swings
- Alert system failures

### Review Logs

```bash
# Railway: Dashboard → Logs
# Look for:
- "Trading loop error" (should be rare)
- "Signal: LONG/SHORT" (regular)
- "Position opened/closed" (when confident)
- "P&L: $X" (after each trade)
```

---

## Step 4: Go Live (After Successful Paper Test)

### ⚠️ CRITICAL SAFETY CHECKS

1. **Review Paper Trading Results:**
   - Positive ROI?
   - Acceptable drawdown?
   - No system errors?

2. **Binance Account:**
   - API keys have trading permissions?
   - Sufficient balance?
      - Recommended: Start with $1,000-$5,000
   - IP whitelist configured (optional security)

3. **Risk Limits:**
   - `MAX_POSITION_SIZE_PCT=5` (max 5% per trade)
   - `MAX_LEVERAGE=3` (conservative)
   - `MIN_CONFIDENCE=0.65` (only high-confidence trades)

### Switch to Live Mode

**Railway Dashboard:**

1. Navigate to Variables
2. Change: `TRADING_MODE=LIVE`
3. Update: `ACCOUNT_BALANCE=<your_actual_balance>`
4. Save changes
5. Railway auto-redeploys

### First Live Trade Monitoring

**Watch closely for first 24 hours:**

- Monitor Telegram alerts in real-time
- Check Binance account for actual orders
- Verify stop losses are placed
- Confirm P&L matches expectations

**Emergency Stop:**

If something goes wrong:

1. **Immediate:** Railway Dashboard → Stop deployment
2. **Binance:** Cancel all open orders manually
3. **Review:** Check logs for error cause
4. **Fix:** Update code, redeploy to paper mode

---

## Step 5: Ongoing Monitoring

### Daily Tasks

- [ ] Review Telegram hourly reports
- [ ] Check Railway logs for errors
- [ ] Monitor Binance account balance
- [ ] Verify positions match system state

### Weekly Tasks

- [ ] Review performance metrics
- [ ] Analyze win rate trends
- [ ] Check for code updates
- [ ] Backup trade history

### Monthly Tasks

- [ ] Deep performance analysis
- [ ] Parameter optimization (if needed)
- [ ] System upgrades
- [ ] Security audit

---

## Troubleshooting

### Common Issues

**1. "Telegram not configured"**
- Solution: Check `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in Railway variables

**2. "Binance API error"**
- Solution: Verify API keys, check IP whitelist, ensure trading permissions

**3. "Data fetch incomplete"**
- Solution: Binance API might be rate-limited. System auto-retries.

**4. "Position size too large"**
- Solution: Reduce `MAX_POSITION_SIZE_PCT` or check `ACCOUNT_BALANCE`

**5. No trades executing**
- Check: `MIN_CONFIDENCE` might be too high
- Check: Market conditions might not meet entry criteria
- Check: Signals being generated but confidence < 65%

### Emergency Contacts

- Railway Support: https://railway.app/help
- Binance API: https://www.binance.com/en/support

---

## Performance Optimization (Advanced)

### After 1 Month of Live Trading

**If results are good:**
- Consider increasing `MAX_POSITION_SIZE_PCT` to 7-10%
- Test multi-symbol trading (future enhancement)
- Optimize `MIN_CONFIDENCE` threshold

**If results need improvement:**
- Review losing trades for patterns
- Adjust regime detection parameters
- Fine-tune indicator weights
- Run backtesting with new parameters

---

## System Architecture (Production)

```
┌───────────────────────────┐
│   Railway (Cloud)          │
│                            │
│  ┌────────────────────┐  │
│  │ TradingEngine      │  │
│  │ (main_loop.py)     │  │
│  └────────┬───────────┘  │
│           │                │
└───────────┼────────────────┘
            │
    ┌───────┼────────┐
    │       │        │
┌───┴───┐ ┌─┴───┐ ┌──┴───┐
│Binance│ │ TG  │ │ PgSQL│
│  API  │ │ Bot │ │   DB  │
└───────┘ └─────┘ └───────┘
  Real       Alerts   State
  Data                Storage
```

---

## Success! 🎉

You now have a production-grade AI trading system running 24/7 on Railway.

**Next Steps:**
- Monitor performance for 1 week
- Fine-tune parameters based on results
- Consider adding more symbols (future)
- Implement advanced ML models (Phase 4)

**Good luck and trade responsibly!**

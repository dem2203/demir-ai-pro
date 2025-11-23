# DEMIR AI PRO v8.0

**Enterprise-Grade AI Crypto Trading Bot**

🛡️ **Zero Mock Data** | 🚀 **Production Ready** | 🧠 **Multi-Layer AI** | 📊 **Real-Time Analysis**

---

## 🎯 Overview

DEMIR AI PRO is a professional cryptocurrency trading bot powered by multi-layer artificial intelligence. Built with enterprise-grade standards, zero tolerance for mock data, and production-ready deployment.

### ✨ Key Features

- ✅ **100% Real Data** - Zero mock, fallback, or test data tolerance
- ✅ **Multi-Layer AI** - Sentiment + ML + Technical analysis
- ✅ **Production Validated** - Strict data validators at every layer
- ✅ **Modular Architecture** - Clean separation of concerns
- ✅ **Enterprise Standards** - Production-grade error handling
- ✅ **Real-Time Signals** - Live market data from Binance Futures
- ✅ **Professional UI** - Turkish pro trader dashboard
- ✅ **Railway Deployment** - One-click cloud deployment

---

## 📚 Table of Contents

1. [Architecture](#architecture)
2. [Tech Stack](#tech-stack)
3. [Quick Start](#quick-start)
4. [Deployment](#deployment)
5. [Configuration](#configuration)
6. [API Documentation](#api-documentation)
7. [Contributing](#contributing)

---

## 🏛️ Architecture

### Modular Design

```
demir-ai-pro/
├── core/                      # Core business logic
│   ├── ai_engine/            # AI/ML ensemble
│   ├── signal_processor/     # Signal generation
│   ├── risk_manager/         # Risk management
│   └── data_pipeline/        # Data fetching
│
├── integrations/           # External APIs
│   ├── binance/             # Binance Futures
│   └── telegram/            # Notifications
│
├── database/               # Data persistence
│   ├── connection.py        # PostgreSQL pooling
│   ├── models.py            # Table schemas
│   └── validators.py        # Data validation
│
├── api/                    # FastAPI routes
│   ├── health.py            # Health checks
│   ├── prices.py            # Price data
│   ├── signals.py           # Signal data
│   └── status.py            # System status
│
├── config/                 # Configuration
│   ├── settings.py          # Environment config
│   └── validation.py        # Config validation
│
├── monitoring/             # Health monitoring
│   └── health_monitor.py    # System metrics
│
├── ui/                     # Dashboard
│   └── dashboard.html       # Turkish pro UI
│
└── main.py                 # Application entry
```

### Data Flow

```
Binance API → Data Pipeline → AI Engine → Signal Processor → Validator → Database
                                        │
                                        ↓
                                   API Routes → Dashboard UI
                                        │
                                        ↓
                                  Telegram Bot
```

---

## 🛠️ Tech Stack

### Backend
- **Python 3.11+** - Modern Python with type hints
- **FastAPI** - High-performance async API framework
- **PostgreSQL 15+** - Production database with TimescaleDB
- **Redis** - In-memory caching (optional but recommended)

### AI/ML
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **TensorFlow/Keras** - LSTM models
- **XGBoost** - Gradient boosting
- **scikit-learn** - ML utilities

### Exchange Integration
- **CCXT** - Unified exchange API
- **WebSockets** - Real-time data streams

### Deployment
- **Docker** - Containerization
- **Railway.app** - Cloud platform
- **Uvicorn** - ASGI server

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- PostgreSQL 15+
- Binance API keys

### Installation

```bash
# Clone repository
git clone https://github.com/dem2203/demir-ai-pro.git
cd demir-ai-pro

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
# Edit .env with your API keys
```

### Configuration

Edit `.env` file:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/demir_ai_pro

# Binance
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Telegram (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Application
ENVIRONMENT=development
DEBUG=true
API_PORT=8000
```

### Run Application

```bash
# Start application
python main.py

# Access dashboard
open http://localhost:8000

# API documentation
open http://localhost:8000/docs
```

---

## ☁️ Deployment

### Railway Deployment (Recommended)

1. **Install Railway CLI**
```bash
npm install -g @railway/cli
```

2. **Login to Railway**
```bash
railway login
```

3. **Create Project**
```bash
railway init
```

4. **Add PostgreSQL**
```bash
railway add postgresql
```

5. **Set Environment Variables**
```bash
railway variables set BINANCE_API_KEY=your_key
railway variables set BINANCE_API_SECRET=your_secret
```

6. **Deploy**
```bash
railway up
```

### Docker Deployment

```bash
# Build image
docker build -t demir-ai-pro .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  -e BINANCE_API_KEY=... \
  -e BINANCE_API_SECRET=... \
  demir-ai-pro
```

---

## 📊 API Documentation

### Health Check

```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-23T23:00:00Z",
  "service": "DEMIR AI PRO"
}
```

### Get Prices

```bash
GET /api/prices
```

Response:
```json
{
  "prices": {
    "BTCUSDT": {
      "price": 97234.50,
      "volume": 1234567890,
      "change24h": 2.5
    }
  }
}
```

### Get Latest Signals

```bash
GET /api/signals/latest?limit=10
```

Response:
```json
{
  "signals": [
    {
      "symbol": "BTCUSDT",
      "direction": "LONG",
      "entry_price": 97234.50,
      "take_profit_1": 98500.00,
      "stop_loss": 96500.00,
      "confidence": 0.85,
      "timestamp": "2025-11-23T23:00:00Z"
    }
  ]
}
```

---

## 🛡️ Production Standards

### Zero Tolerance Rules

1. ❌ **NO MOCK DATA** - All data from real APIs
2. ❌ **NO FALLBACK** - No fallback to fake data
3. ❌ **NO TEST DATA** - No hardcoded test values
4. ❌ **NO PLACEHOLDERS** - No "TODO" in production

### Validation Layers

1. **Configuration Validation** - Startup checks
2. **Data Validation** - Real-time verification
3. **Signal Validation** - Multi-layer checks
4. **Price Validation** - Exchange verification

---

## 💻 Development

### Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=.

# Lint
flake8 .

# Format
black .
```

### Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📜 License

This project is proprietary and confidential.

---

## 📞 Support

For support or questions, please open an issue on GitHub.

---

**Built with professional standards. Zero compromises.**

🔥 **DEMIR AI PRO v8.0** - Enterprise-Grade AI Trading

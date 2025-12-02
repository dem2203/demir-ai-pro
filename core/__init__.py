"""
DEMIR AI PRO - Core Business Logic

Modular core engine with:
- AI/ML ensemble models
- Signal generation and validation
- Risk management
- Data pipeline
- Trading engine (background task)

Graceful degradation: Missing modules won't break the system
"""

# Trading Engine (always available)
try:
    from .trading_engine import TradingEngine, get_engine
    TRADING_ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Trading engine import failed: {e}")
    TRADING_ENGINE_AVAILABLE = False
    TradingEngine = None
    get_engine = None

# Technical Analysis Engine with backward compatibility
try:
    from .technical_analysis import ProfessionalTAEngine, get_ta_engine
    # Alias for backward compatibility
    TechnicalAnalyzer = ProfessionalTAEngine
    TECHNICAL_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Technical analysis import failed: {e}")
    TECHNICAL_ANALYSIS_AVAILABLE = False
    TechnicalAnalyzer = None
    ProfessionalTAEngine = None
    get_ta_engine = None

# Optional AI modules (graceful degradation)
try:
    from .ai_engine import AIEngine
except ImportError:
    AIEngine = None

try:
    from .signal_processor import SignalProcessor
except ImportError:
    SignalProcessor = None

try:
    from .risk_manager import RiskManager
except ImportError:
    RiskManager = None

try:
    from .data_pipeline import DataPipeline
except ImportError:
    DataPipeline = None

__all__ = [
    'AIEngine',
    'SignalProcessor',
    'RiskManager',
    'DataPipeline',
    'TradingEngine',
    'get_engine',
    'TechnicalAnalyzer',
    'ProfessionalTAEngine',
    'get_ta_engine',
    'TRADING_ENGINE_AVAILABLE',
    'TECHNICAL_ANALYSIS_AVAILABLE',
]

"""AI Engine Module v11.0

Production AI prediction engine with:
- LSTM (Time-series RNN)
- XGBoost (Gradient boosting)
- Random Forest (Ensemble trees)
- Gradient Boosting (Boosted trees)

✅ Pure AI predictions
✅ Real-time 24/7 engine
✅ Feature engineering
✅ Ensemble voting
"""

# Core exports - prediction engine and model trainer
try:
    from .prediction_engine import get_prediction_engine, PredictionEngine
    PREDICTION_ENGINE_AVAILABLE = True
except ImportError as e:
    PREDICTION_ENGINE_AVAILABLE = False
    print(f"Prediction engine import failed: {e}")

try:
    from .model_trainer import get_model_trainer, ModelTrainer
    MODEL_TRAINER_AVAILABLE = True
except ImportError:
    MODEL_TRAINER_AVAILABLE = False

try:
    from .feature_engineering import get_feature_engineer, FeatureEngineer
    FEATURE_ENGINEER_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEER_AVAILABLE = False

# Optional: Ensemble model if exists
try:
    from .ensemble import EnsembleModel
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

__all__ = [
    'get_prediction_engine',
    'PredictionEngine',
    'get_model_trainer',
    'ModelTrainer',
    'get_feature_engineer',
    'FeatureEngineer',
    'PREDICTION_ENGINE_AVAILABLE',
    'MODEL_TRAINER_AVAILABLE',
    'FEATURE_ENGINEER_AVAILABLE'
]

if ENSEMBLE_AVAILABLE:
    __all__.append('EnsembleModel')

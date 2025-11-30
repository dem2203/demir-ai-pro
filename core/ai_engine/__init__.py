"""
AI Engine Module v9.1

Multi-layer ML ensemble for trading signal generation.

Real ML Models:
- LSTM (Time-series RNN)
- XGBoost (Gradient boosting)
- Random Forest (Ensemble trees)
- Gradient Boosting (Boosted trees)

Features:
- Real model training (model_trainer)
- 24/7 prediction engine (prediction_engine)
- Feature engineering
- Ensemble voting
"""

from .ensemble import EnsembleModel
from .lstm_model import LSTMPredictor
from .xgboost_model import XGBoostPredictor

# v9.1 NEW: Export prediction engine and model trainer
try:
    from .prediction_engine import get_prediction_engine, PredictionEngine
    PREDICTION_ENGINE_AVAILABLE = True
except ImportError:
    PREDICTION_ENGINE_AVAILABLE = False

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

__all__ = [
    'EnsembleModel',
    'LSTMPredictor',
    'XGBoostPredictor',
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

#!/usr/bin/env python3
"""
DEMIR AI PRO v8.0 - AI Prediction Database Models

Database tables for AI/ML system:
- AI predictions storage
- Model performance tracking
- Feature importance history
- Training metadata
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class AIPrediction(Base):
    """
    Store AI prediction results for tracking and analysis
    """
    __tablename__ = 'ai_predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Ensemble prediction
    ensemble_direction = Column(String(10), nullable=False)  # BUY, SELL, NEUTRAL
    ensemble_confidence = Column(Float, nullable=False)
    prob_buy = Column(Float, nullable=False)
    prob_neutral = Column(Float, nullable=False)
    prob_sell = Column(Float, nullable=False)
    
    # Individual model predictions
    lstm_direction = Column(String(10))
    lstm_confidence = Column(Float)
    xgboost_direction = Column(String(10))
    xgboost_confidence = Column(Float)
    rf_direction = Column(String(10))
    rf_confidence = Column(Float)
    gb_direction = Column(String(10))
    gb_confidence = Column(Float)
    
    # Metadata
    agreement_score = Column(Float)
    price_at_prediction = Column(Float)
    
    # Outcome tracking (filled after time passes)
    actual_direction = Column(String(10))  # Actual price movement
    actual_change_1h = Column(Float)  # % change after 1 hour
    actual_change_4h = Column(Float)  # % change after 4 hours
    actual_change_24h = Column(Float)  # % change after 24 hours
    prediction_correct = Column(Boolean)  # Was ensemble prediction correct?
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<AIPrediction {self.symbol} {self.ensemble_direction} @ {self.timestamp}>"


class ModelPerformance(Base):
    """
    Track ML model performance metrics over time
    """
    __tablename__ = 'model_performance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False, index=True)  # lstm, xgboost, rf, gb, ensemble
    evaluation_date = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Performance metrics
    accuracy = Column(Float)  # Overall accuracy
    precision = Column(Float)  # Precision score
    recall = Column(Float)  # Recall score
    f1_score = Column(Float)  # F1 score
    win_rate = Column(Float)  # Trading win rate
    
    # Evaluation window
    evaluation_period_days = Column(Integer, default=7)
    total_predictions = Column(Integer)
    correct_predictions = Column(Integer)
    
    # Symbol-specific (optional)
    symbol = Column(String(20), index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelPerformance {self.model_name} acc={self.accuracy:.2f}>"


class FeatureImportance(Base):
    """
    Store feature importance scores from trained models
    """
    __tablename__ = 'feature_importance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False, index=True)
    feature_name = Column(String(100), nullable=False, index=True)
    importance_score = Column(Float, nullable=False)
    
    # Training metadata
    training_date = Column(DateTime, default=datetime.utcnow, index=True)
    symbol = Column(String(20))  # If model is symbol-specific
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<FeatureImportance {self.model_name}.{self.feature_name}={self.importance_score:.4f}>"


class ModelTraining(Base):
    """
    Track model training sessions and metadata
    """
    __tablename__ = 'model_training'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(50), nullable=False, index=True)
    model_version = Column(String(20), nullable=False)
    
    # Training details
    training_start = Column(DateTime, nullable=False)
    training_end = Column(DateTime, nullable=False)
    training_duration_seconds = Column(Integer)
    
    # Dataset info
    training_samples = Column(Integer)
    validation_samples = Column(Integer)
    test_samples = Column(Integer)
    
    # Hyperparameters (stored as JSON)
    hyperparameters = Column(JSON)
    
    # Results
    train_accuracy = Column(Float)
    validation_accuracy = Column(Float)
    test_accuracy = Column(Float)
    
    # Model file path
    model_file_path = Column(String(255))
    
    # Status
    is_production = Column(Boolean, default=False)  # Is this the active production model?
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<ModelTraining {self.model_name} v{self.model_version}>"


def create_ai_tables(engine):
    """
    Create all AI-related tables in database
    
    Usage:
        from database import get_engine
        from database.models_ai import create_ai_tables
        
        engine = get_engine()
        create_ai_tables(engine)
    """
    Base.metadata.create_all(engine)
    print("✅ AI database tables created successfully")


if __name__ == "__main__":
    # Test: Print table schemas
    from sqlalchemy import create_engine
    from sqlalchemy.schema import CreateTable
    
    print("AI Database Table Schemas:\n")
    
    for table in [AIPrediction, ModelPerformance, FeatureImportance, ModelTraining]:
        print(f"--- {table.__tablename__} ---")
        print(CreateTable(table.__table__).compile(dialect=create_engine('postgresql://').dialect))
        print("\n")

#!/usr/bin/env python3
"""DEMIR AI PRO v10.1 - PURE AI Prediction Engine

Real AI predictions with:
- LSTM time-series forecasting (TRAINED MODELS ONLY)
- XGBoost gradient boosting (TRAINED MODELS ONLY)
- Random Forest ensemble (TRAINED MODELS ONLY)
- Gradient Boosting classifier (TRAINED MODELS ONLY)
- NO FALLBACK - Model not ready = explicit message
- Auto-training on first prediction

❌ NO MOCK DATA
❌ NO FALLBACK PREDICTIONS
✅ 100% Pure AI
"""

import logging
import os
import asyncio
import json
import time
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np
import pytz
import joblib

logger = logging.getLogger(__name__)

class PredictionDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

@dataclass
class ModelPrediction:
    direction: PredictionDirection
    confidence: float
    probability: float
    execution_time_ms: float
    model_loaded: bool = False

@dataclass
class EnsemblePrediction:
    direction: PredictionDirection
    confidence: float
    probabilities: Dict[str, float]

@dataclass
class AIPrediction:
    symbol: str
    timestamp: str
    ensemble_prediction: EnsemblePrediction
    model_predictions: Dict[str, ModelPrediction]
    agreement_score: float
    execution_time_ms: float
    version: str
    models_ready: bool

@dataclass
class PerformanceMetrics:
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    avg_execution_time_ms: float
    last_prediction_time: Optional[str]
    uptime_hours: float
    models_loaded: Dict[str, bool]

class PredictionEngine:
    """Pure AI prediction engine - NO FALLBACK"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.models_loaded: Dict[str, bool] = {}
        self.is_running: bool = False
        self.last_predictions: Dict[str, AIPrediction] = {}
        self.telegram_notifier: Optional[Any] = None
        self.last_hourly_update: Optional[datetime] = None
        self.models_dir = Path("models/saved")
        self.strong_buy_threshold: float = 0.75
        self.strong_sell_threshold: float = 0.75
        self.start_time: Optional[datetime] = None
        self.total_predictions: int = 0
        self.successful_predictions: int = 0
        self.failed_predictions: int = 0
        self.execution_times: List[float] = []
        self.prediction_interval: int = 300
        self.hourly_update_interval: int = 3600
        self.version: str = "10.1"
        self.training_started: bool = False
        logger.info("PredictionEngine initialized (PURE AI)", version=self.version)
    
    async def start(self) -> None:
        try:
            logger.info("Starting PURE AI Prediction Engine", version=self.version)
            self.start_time = datetime.now(pytz.UTC)
            await self._load_models()
            await self._init_telegram()
            
            # If no models loaded, start training immediately
            if not any(self.models_loaded.values()):
                logger.warning("No models found - starting immediate training")
                asyncio.create_task(self._immediate_training())
            
            self.is_running = True
            asyncio.create_task(self._prediction_loop())
            asyncio.create_task(self._hourly_status_loop())
            asyncio.create_task(self._metrics_reporter_loop())
            asyncio.create_task(self._auto_training_loop())
            
            logger.info("PURE AI Prediction Engine started", models_loaded=self.models_loaded)
        except Exception as e:
            logger.error(f"Prediction engine startup failed: {e}")
    
    async def stop(self) -> None:
        logger.info("Stopping AI Prediction Engine")
        self.is_running = False
        metrics = self.get_performance_metrics()
        logger.info("Final performance metrics", extra=asdict(metrics))
    
    async def _load_models(self) -> None:
        try:
            logger.info("Loading ML models", models_dir=str(self.models_dir))
            if not self.models_dir.exists():
                logger.warning("Models directory not found - will train on first prediction")
                self.models_loaded = {'lstm': False, 'xgboost': False, 'random_forest': False, 'gradient_boosting': False}
                return
            
            symbol = 'BTCUSDT'
            
            # XGBoost
            xgb_files = list(self.models_dir.glob(f"xgboost_{symbol}_*.pkl"))
            if xgb_files:
                latest_xgb = sorted(xgb_files)[-1]
                self.models['xgboost'] = joblib.load(latest_xgb)
                self.models_loaded['xgboost'] = True
                logger.info(f"XGBoost loaded: {latest_xgb}")
            else:
                self.models_loaded['xgboost'] = False
            
            # Random Forest
            rf_files = list(self.models_dir.glob(f"random_forest_{symbol}_*.pkl"))
            if rf_files:
                latest_rf = sorted(rf_files)[-1]
                self.models['random_forest'] = joblib.load(latest_rf)
                self.models_loaded['random_forest'] = True
                logger.info(f"Random Forest loaded: {latest_rf}")
            else:
                self.models_loaded['random_forest'] = False
            
            # Gradient Boosting
            gb_files = list(self.models_dir.glob(f"gradient_boosting_{symbol}_*.pkl"))
            if gb_files:
                latest_gb = sorted(gb_files)[-1]
                self.models['gradient_boosting'] = joblib.load(latest_gb)
                self.models_loaded['gradient_boosting'] = True
                logger.info(f"Gradient Boosting loaded: {latest_gb}")
            else:
                self.models_loaded['gradient_boosting'] = False
            
            # LSTM
            try:
                from tensorflow import keras
                lstm_files = list(self.models_dir.glob(f"lstm_{symbol}_*.h5"))
                if lstm_files:
                    latest_lstm = sorted(lstm_files)[-1]
                    self.models['lstm'] = keras.models.load_model(latest_lstm)
                    self.models_loaded['lstm'] = True
                    logger.info(f"LSTM loaded: {latest_lstm}")
                else:
                    self.models_loaded['lstm'] = False
            except ImportError:
                logger.warning("TensorFlow not available - LSTM skipped")
                self.models_loaded['lstm'] = False
            
            loaded_count = sum(self.models_loaded.values())
            logger.info(f"ML models loaded: {loaded_count}/4", models=self.models_loaded)
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            self.models_loaded = {'lstm': False, 'xgboost': False, 'random_forest': False, 'gradient_boosting': False}
    
    async def _immediate_training(self) -> None:
        """Start training immediately if no models exist"""
        if self.training_started:
            return
        self.training_started = True
        
        try:
            logger.info("Starting immediate model training...")
            from core.ai_engine.model_trainer import get_model_trainer
            trainer = get_model_trainer()
            await trainer.train_all_models()
            await self._load_models()
            logger.info("Immediate training completed - models ready")
        except Exception as e:
            logger.error(f"Immediate training failed: {e}")
    
    async def _init_telegram(self) -> None:
        try:
            from integrations.telegram_notifier import TelegramNotifier
            telegram_token = os.getenv('TELEGRAM_TOKEN')
            telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
            if telegram_token and telegram_chat_id:
                self.telegram_notifier = TelegramNotifier(token=telegram_token, chat_id=telegram_chat_id)
                startup_msg = f"🧠 PURE AI v{self.version} Started\n✅ Real ML Models Only\n🤖 Models loaded: {sum(self.models_loaded.values())}/4\n⏰ {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}"
                await self.telegram_notifier.send_message(startup_msg)
                logger.info("Telegram ready")
        except Exception as e:
            logger.error(f"Telegram init error: {e}")
    
    async def _prediction_loop(self) -> None:
        logger.info("Starting prediction loop")
        while self.is_running:
            try:
                symbols = await self._get_monitored_coins()
                for symbol in symbols:
                    prediction = await self.predict(symbol)
                    if prediction and prediction.models_ready:
                        await self._check_and_alert(symbol, prediction)
                        self.last_predictions[symbol] = prediction
                await asyncio.sleep(self.prediction_interval)
            except Exception as e:
                logger.error(f"Prediction loop error: {e}")
                await asyncio.sleep(60)
    
    async def _hourly_status_loop(self) -> None:
        logger.info("Starting hourly status")
        while self.is_running:
            try:
                now = datetime.now(pytz.UTC)
                if self.last_hourly_update is None or (now - self.last_hourly_update).seconds >= 3600:
                    await self._send_hourly_status()
                    self.last_hourly_update = now
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Hourly status error: {e}")
    
    async def _metrics_reporter_loop(self) -> None:
        while self.is_running:
            try:
                await asyncio.sleep(900)
                metrics = self.get_performance_metrics()
                logger.info("Performance metrics", extra=asdict(metrics))
            except Exception as e:
                logger.error(f"Metrics reporter error: {e}")
    
    async def _auto_training_loop(self) -> None:
        await asyncio.sleep(3600)
        while self.is_running:
            try:
                from core.ai_engine.model_trainer import get_model_trainer
                trainer = get_model_trainer()
                if trainer._should_retrain():
                    logger.info("Starting scheduled training")
                    await trainer.train_all_models()
                    await self._load_models()
                    logger.info("Scheduled training completed")
                await asyncio.sleep(21600)
            except Exception as e:
                logger.error(f"Auto-training error: {e}")
                await asyncio.sleep(3600)
    
    async def predict(self, symbol: str) -> Optional[AIPrediction]:
        """Generate PURE AI prediction - NO FALLBACK"""
        start_time = time.time()
        try:
            self.total_predictions += 1
            
            # Check if models ready
            if not any(self.models_loaded.values()):
                logger.warning(f"No models ready for {symbol} - training in progress")
                self.failed_predictions += 1
                return None
            
            # Get technical analysis
            from core.technical_analysis import TechnicalAnalyzer
            analyzer = TechnicalAnalyzer()
            analysis = analyzer.analyze(symbol)
            if not analysis:
                self.failed_predictions += 1
                return None
            
            # Extract features
            from core.ai_engine.feature_engineering import get_feature_engineer
            fe = get_feature_engineer()
            features = fe.extract_features(analysis)
            
            # Get predictions from LOADED models only
            predictions: Dict[str, ModelPrediction] = {}
            
            if self.models_loaded.get('lstm'):
                predictions['lstm'] = await self._predict_lstm_pure(symbol, features)
            if self.models_loaded.get('xgboost'):
                predictions['xgboost'] = await self._predict_xgboost_pure(symbol, features)
            if self.models_loaded.get('random_forest'):
                predictions['random_forest'] = await self._predict_rf_pure(symbol, features)
            if self.models_loaded.get('gradient_boosting'):
                predictions['gradient_boosting'] = await self._predict_gb_pure(symbol, features)
            
            if not predictions:
                logger.warning(f"No model predictions available for {symbol}")
                self.failed_predictions += 1
                return None
            
            ensemble = self._calculate_ensemble(predictions)
            agreement = self._calculate_agreement(predictions)
            execution_time_ms = (time.time() - start_time) * 1000
            self.execution_times.append(execution_time_ms)
            
            if len(self.execution_times) > 1000:
                self.execution_times = self.execution_times[-1000:]
            
            result = AIPrediction(
                symbol=symbol,
                timestamp=datetime.now(pytz.UTC).isoformat(),
                ensemble_prediction=ensemble,
                model_predictions=predictions,
                agreement_score=agreement,
                execution_time_ms=execution_time_ms,
                version=self.version,
                models_ready=True
            )
            
            self.successful_predictions += 1
            logger.info(f"PURE AI prediction for {symbol}", direction=ensemble.direction.value, confidence=ensemble.confidence, models_used=len(predictions))
            return result
        except Exception as e:
            self.failed_predictions += 1
            logger.error(f"Prediction error for {symbol}: {e}")
            return None
    
    async def _predict_lstm_pure(self, symbol: str, features: Dict[str, float]) -> ModelPrediction:
        """LSTM prediction - TRAINED MODEL ONLY"""
        start_time = time.time()
        try:
            feature_vector = np.array(list(features.values())).reshape(1, 1, -1)
            prediction_prob = self.models['lstm'].predict(feature_vector, verbose=0)[0][0]
            
            if prediction_prob > 0.6:
                direction = PredictionDirection.BUY
                confidence = float(prediction_prob)
            elif prediction_prob < 0.4:
                direction = PredictionDirection.SELL
                confidence = float(1 - prediction_prob)
            else:
                direction = PredictionDirection.NEUTRAL
                confidence = 0.5
            
            execution_time_ms = (time.time() - start_time) * 1000
            return ModelPrediction(direction=direction, confidence=confidence, probability=float(prediction_prob), execution_time_ms=execution_time_ms, model_loaded=True)
        except Exception as e:
            logger.error(f"LSTM prediction error: {e}")
            raise
    
    async def _predict_xgboost_pure(self, symbol: str, features: Dict[str, float]) -> ModelPrediction:
        """XGBoost prediction - TRAINED MODEL ONLY"""
        start_time = time.time()
        try:
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            prediction = self.models['xgboost'].predict(feature_vector)[0]
            prediction_prob = self.models['xgboost'].predict_proba(feature_vector)[0]
            
            if prediction == 1:
                direction = PredictionDirection.BUY
                confidence = float(prediction_prob[1])
            else:
                direction = PredictionDirection.SELL
                confidence = float(prediction_prob[0])
            
            execution_time_ms = (time.time() - start_time) * 1000
            return ModelPrediction(direction=direction, confidence=confidence, probability=float(prediction_prob[1]), execution_time_ms=execution_time_ms, model_loaded=True)
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            raise
    
    async def _predict_rf_pure(self, symbol: str, features: Dict[str, float]) -> ModelPrediction:
        """Random Forest prediction - TRAINED MODEL ONLY"""
        start_time = time.time()
        try:
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            prediction = self.models['random_forest'].predict(feature_vector)[0]
            prediction_prob = self.models['random_forest'].predict_proba(feature_vector)[0]
            
            if prediction == 1:
                direction = PredictionDirection.BUY
                confidence = float(prediction_prob[1])
            else:
                direction = PredictionDirection.SELL
                confidence = float(prediction_prob[0])
            
            execution_time_ms = (time.time() - start_time) * 1000
            return ModelPrediction(direction=direction, confidence=confidence, probability=float(prediction_prob[1]), execution_time_ms=execution_time_ms, model_loaded=True)
        except Exception as e:
            logger.error(f"Random Forest prediction error: {e}")
            raise
    
    async def _predict_gb_pure(self, symbol: str, features: Dict[str, float]) -> ModelPrediction:
        """Gradient Boosting prediction - TRAINED MODEL ONLY"""
        start_time = time.time()
        try:
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            prediction = self.models['gradient_boosting'].predict(feature_vector)[0]
            prediction_prob = self.models['gradient_boosting'].predict_proba(feature_vector)[0]
            
            if prediction == 1:
                direction = PredictionDirection.BUY
                confidence = float(prediction_prob[1])
            else:
                direction = PredictionDirection.SELL
                confidence = float(prediction_prob[0])
            
            execution_time_ms = (time.time() - start_time) * 1000
            return ModelPrediction(direction=direction, confidence=confidence, probability=float(prediction_prob[1]), execution_time_ms=execution_time_ms, model_loaded=True)
        except Exception as e:
            logger.error(f"Gradient Boosting prediction error: {e}")
            raise
    
    def _calculate_ensemble(self, predictions: Dict[str, ModelPrediction]) -> EnsemblePrediction:
        weights = {'lstm': 0.30, 'xgboost': 0.30, 'random_forest': 0.20, 'gradient_boosting': 0.20}
        total_weight = sum(weights[k] for k in predictions.keys())
        weights = {k: weights[k]/total_weight for k in predictions.keys()}
        
        direction_scores = {'BUY': 0.0, 'NEUTRAL': 0.0, 'SELL': 0.0}
        for model_name, prediction in predictions.items():
            weight = weights[model_name]
            direction_key = prediction.direction.value
            direction_scores[direction_key] += weight * prediction.confidence
        
        total = sum(direction_scores.values())
        probabilities = {k: v/total for k, v in direction_scores.items()} if total > 0 else {'BUY': 0.33, 'NEUTRAL': 0.34, 'SELL': 0.33}
        final_direction = PredictionDirection(max(probabilities, key=probabilities.get))
        final_confidence = probabilities[final_direction.value]
        
        return EnsemblePrediction(direction=final_direction, confidence=final_confidence, probabilities=probabilities)
    
    def _calculate_agreement(self, predictions: Dict[str, ModelPrediction]) -> float:
        directions = [p.direction.value for p in predictions.values()]
        if not directions:
            return 0.0
        from collections import Counter
        counts = Counter(directions)
        most_common_count = counts.most_common(1)[0][1]
        return most_common_count / len(directions)
    
    async def _check_and_alert(self, symbol: str, prediction: AIPrediction) -> None:
        try:
            if not self.telegram_notifier:
                return
            ensemble = prediction.ensemble_prediction
            if (ensemble.direction == PredictionDirection.BUY and ensemble.confidence >= self.strong_buy_threshold) or \
               (ensemble.direction == PredictionDirection.SELL and ensemble.confidence >= self.strong_sell_threshold):
                # Send alert via telegram_ultra if available
                pass
        except Exception as e:
            logger.error(f"Alert error: {e}")
    
    async def _send_hourly_status(self) -> None:
        try:
            if not self.telegram_notifier:
                return
            from integrations.binance_client import get_binance_client
            binance = get_binance_client()
            btc_price = await binance.get_current_price('BTCUSDT')
            eth_price = await binance.get_current_price('ETHUSDT')
            ltc_price = await binance.get_current_price('LTCUSDT')
            metrics = self.get_performance_metrics()
            message = f"🔔 HOURLY STATUS\n\n🔸 BTC: ${btc_price:,.2f}\n🔹 ETH: ${eth_price:,.2f}\n🟦 LTC: ${ltc_price:,.2f}\n\n🧠 PURE AI v{self.version}\n✅ Models: {sum(metrics.models_loaded.values())}/4\n📊 Predictions: {metrics.total_predictions}\n⏱️ Avg: {metrics.avg_execution_time_ms:.1f}ms\n⏰ {datetime.now(pytz.UTC).strftime('%H:%M UTC')}"
            await self.telegram_notifier.send_message(message)
        except Exception as e:
            logger.error(f"Hourly status error: {e}")
    
    async def _get_monitored_coins(self) -> List[str]:
        try:
            from api.coin_manager import get_monitored_coins
            return get_monitored_coins() or ['BTCUSDT', 'ETHUSDT', 'LTCUSDT']
        except:
            return ['BTCUSDT', 'ETHUSDT', 'LTCUSDT']
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        uptime_hours = 0.0
        if self.start_time:
            uptime_hours = (datetime.now(pytz.UTC) - self.start_time).total_seconds() / 3600
        avg_exec_time = float(np.mean(self.execution_times)) if self.execution_times else 0.0
        last_pred_time = None
        if self.last_predictions:
            last_symbol = list(self.last_predictions.keys())[-1]
            last_pred_time = self.last_predictions[last_symbol].timestamp
        return PerformanceMetrics(total_predictions=self.total_predictions, successful_predictions=self.successful_predictions, failed_predictions=self.failed_predictions, avg_execution_time_ms=avg_exec_time, last_prediction_time=last_pred_time, uptime_hours=uptime_hours, models_loaded=self.models_loaded)

_prediction_engine: Optional[PredictionEngine] = None

def get_prediction_engine() -> PredictionEngine:
    global _prediction_engine
    if _prediction_engine is None:
        _prediction_engine = PredictionEngine()
    return _prediction_engine

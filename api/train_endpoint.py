#!/usr/bin/env python3
"""🏋️ DEMIR AI PRO v11.0 - Training Endpoint

Manual model training trigger.
"""

import logging
import asyncio
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)

training_in_progress = False
training_start_time = None

async def start_training():
    """Start model training"""
    global training_in_progress, training_start_time
    
    if training_in_progress:
        return {
            "success": False,
            "error": "Training already in progress",
            "started_at": training_start_time.isoformat() if training_start_time else None
        }
    
    try:
        training_in_progress = True
        training_start_time = datetime.now(pytz.UTC)
        
        logger.info("🏋️ Manual training started")
        
        from core.ai_engine.model_trainer import get_model_trainer
        trainer = get_model_trainer()
        
        # Background task
        asyncio.create_task(_train_task(trainer))
        
        return {
            "success": True,
            "message": "Training started",
            "started_at": training_start_time.isoformat(),
            "estimated_minutes": 10
        }
        
    except Exception as e:
        training_in_progress = False
        logger.error(f"❌ Training start error: {e}")
        return {"success": False, "error": str(e)}

async def _train_task(trainer):
    """Background training"""
    global training_in_progress
    try:
        logger.info("🏋️ Training models...")
        await trainer.train_all_models()
        
        from core.ai_engine.prediction_engine import get_prediction_engine
        pred_engine = get_prediction_engine()
        await pred_engine._load_models()
        
        logger.info("✅ Training complete")
        
        # Telegram notification
        try:
            from integrations.telegram_ultra import get_telegram_ultra
            import os
            token = os.getenv('TELEGRAM_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            if token and chat_id:
                tg = get_telegram_ultra(token, chat_id)
                await tg.send_text(
                    "🏋️ TRAINING COMPLETE\n"
                    "✅ All models trained\n"
                    f"⏰ {datetime.now(pytz.UTC).strftime('%H:%M UTC')}"
                )
        except:
            pass
            
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
    finally:
        training_in_progress = False

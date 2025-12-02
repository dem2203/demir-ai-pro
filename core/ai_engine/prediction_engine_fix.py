#!/usr/bin/env python3
# QUICK FIX SCRIPT - Replace problematic logging lines
import re

# Read original file
with open('core/ai_engine/prediction_engine.py', 'r') as f:
    content = f.read()

# Replace problematic lines
replacements = [
    ('logger.info("PredictionEngine initialized (PURE AI)", version=self.version)',
     'logger.info("🧠 PredictionEngine v%s initialized (PURE AI)", self.version)'),
    ('logger.info("Starting PURE AI Prediction Engine", version=self.version)',
     'logger.info("🚀 Starting PURE AI Prediction Engine v%s", self.version)'),
    ('logger.info("PURE AI Prediction Engine started", models_loaded=self.models_loaded)',
     'logger.info("✅ PURE AI Prediction Engine started | Models: %d/4", sum(self.models_loaded.values()))'),
    ('logger.info(f"PURE AI prediction for {symbol}", direction=ensemble.direction.value, confidence=ensemble.confidence, models_used=len(predictions))',
     'logger.info("🎯 AI: %s | %s | Conf: %.1f%% | Models: %d", symbol, ensemble.direction.value, ensemble.confidence*100, len(predictions))'),
]

for old, new in replacements:
    content = content.replace(old, new)

# Write back
with open('core/ai_engine/prediction_engine.py', 'w') as f:
    f.write(content)

print('✅ Fixed AI prediction engine logging')

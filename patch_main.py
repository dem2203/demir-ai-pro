#!/usr/bin/env python3
"""Patch to add admin routes to main.py

Add these lines after: app.state.app_state = app_state

Insert:
# ====================================================================
# INCLUDE ROUTERS
# ====================================================================

try:
    from routes.admin_routes import router as admin_router
    app.include_router(admin_router)
    logger.info("✅ Admin routes loaded")
except Exception as e:
    logger.warning(f"⚠️  Admin routes not loaded: {e}")
"""

# Manual steps:
# 1. Open main.py
# 2. Find line: app.state.app_state = app_state
# 3. Add above code after that line
# 4. Before the MIDDLEWARE section

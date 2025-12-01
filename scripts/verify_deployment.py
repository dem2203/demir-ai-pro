#!/usr/bin/env python3
"""DEMIR AI PRO - Deployment Verification Script

This script verifies that Railway deployed v10.1 correctly.

Usage:
    python scripts/verify_deployment.py https://your-railway-url.railway.app
"""

import sys
import requests
import json
from typing import Dict, Any

def verify_deployment(base_url: str) -> bool:
    """Verify v10.1 deployment"""
    print("\n" + "="*60)
    print("🔍 DEMIR AI PRO - DEPLOYMENT VERIFICATION")
    print("="*60 + "\n")
    
    success = True
    
    # Test 1: Health endpoint
    print("🟢 Test 1: Health Check")
    try:
        resp = requests.get(f"{base_url}/health", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            version = data.get('version', 'UNKNOWN')
            print(f"   ✅ Status: {resp.status_code}")
            print(f"   ✅ Version: {version}")
            
            if version != "10.1":
                print(f"   ❌ ERROR: Expected v10.1, got {version}")
                print(f"   ⚠️  Railway is still running OLD code!")
                success = False
            else:
                print(f"   ✅ Correct version deployed!")
            
            # Check services
            services = data.get('services', {})
            print(f"\n   Services:")
            for service, status in services.items():
                icon = "✅" if status else "❌"
                print(f"   {icon} {service}: {status}")
            
            # Check prediction engine
            pred_engine = data.get('prediction_engine', {})
            if pred_engine:
                print(f"\n   Prediction Engine:")
                print(f"   ✅ Running: {pred_engine.get('running', False)}")
                models = pred_engine.get('models_loaded', 0)
                print(f"   🧠 Models: {models}/4")
                if models == 0:
                    print(f"   ⚠️  Models training in progress (wait 5-10 min)")
                elif models == 4:
                    print(f"   ✅ All models ready - PURE AI active!")
        else:
            print(f"   ❌ Health check failed: {resp.status_code}")
            success = False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        success = False
    
    # Test 2: Dashboard
    print(f"\n🟢 Test 2: Dashboard")
    try:
        resp = requests.get(base_url, timeout=10)
        if resp.status_code == 200:
            print(f"   ✅ Dashboard accessible: {resp.status_code}")
            if "v10.1" in resp.text or "ULTRA" in resp.text:
                print(f"   ✅ v10.1 dashboard loaded")
            else:
                print(f"   ⚠️  Dashboard may be old version")
        else:
            print(f"   ❌ Dashboard failed: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ Dashboard error: {e}")
    
    # Test 3: AI Status endpoint
    print(f"\n🟢 Test 3: AI Status API")
    try:
        resp = requests.get(f"{base_url}/api/ai/status", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"   ✅ AI Status endpoint exists")
            ai_data = data.get('data', {})
            print(f"   ✅ Running: {ai_data.get('running', False)}")
            print(f"   ✅ Version: {ai_data.get('version', 'UNKNOWN')}")
            models = ai_data.get('models_loaded', {})
            print(f"   🧠 Models: {models}")
        elif resp.status_code == 404:
            print(f"   ❌ AI Status endpoint NOT FOUND")
            print(f"   ⚠️  This means Railway deployed OLD code (v7.0)")
            success = False
        else:
            print(f"   ❌ AI Status failed: {resp.status_code}")
    except Exception as e:
        print(f"   ❌ AI Status error: {e}")
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ DEPLOYMENT SUCCESSFUL - v10.1 is LIVE!")
        print("\n💡 Next steps:")
        print("   1. Wait 5-10 minutes for model training")
        print("   2. Refresh dashboard")
        print("   3. Models should show 4/4")
        print("   4. PURE AI predictions will start automatically")
    else:
        print("❌ DEPLOYMENT FAILED - Still running v7.0")
        print("\n🔧 Fix:")
        print("   1. Go to Railway dashboard")
        print("   2. Settings -> Clear Build Cache")
        print("   3. Trigger manual redeploy")
        print("   4. Wait 2-3 minutes")
        print("   5. Run this script again")
    print("="*60 + "\n")
    
    return success

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Usage: python scripts/verify_deployment.py https://your-railway-url.railway.app")
        sys.exit(1)
    
    url = sys.argv[1].rstrip('/')
    success = verify_deployment(url)
    sys.exit(0 if success else 1)

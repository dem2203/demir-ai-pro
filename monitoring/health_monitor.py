#!/usr/bin/env python3
"""
System Health Monitor

Monitors system health and performance metrics.
"""

import logging
import psutil
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class HealthMonitor:
    """
    System health monitoring service.
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        logger.info("✅ Health Monitor initialized")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get current system metrics.
        
        Returns:
            Dict with CPU, memory, disk metrics
        """
        try:
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_percent': psutil.disk_usage('/').percent,
                'disk_used_gb': psutil.disk_usage('/').used / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def get_uptime(self) -> float:
        """
        Get uptime in seconds.
        
        Returns:
            Uptime in seconds
        """
        return (datetime.now() - self.start_time).total_seconds()
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status.
        
        Returns:
            Health status dict
        """
        metrics = self.get_system_metrics()
        uptime = self.get_uptime()
        
        # Determine overall status
        status = 'healthy'
        if metrics.get('cpu_percent', 0) > 90:
            status = 'warning'
        if metrics.get('memory_percent', 0) > 90:
            status = 'warning'
        if metrics.get('disk_percent', 0) > 90:
            status = 'critical'
        
        return {
            'status': status,
            'uptime_seconds': uptime,
            'uptime_hours': uptime / 3600,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }

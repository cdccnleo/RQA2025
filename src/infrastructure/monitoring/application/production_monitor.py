#!/usr/bin/env python3
"""
简化版 ProductionMonitor，提供基本的健康检查与生命周期接口，
用于兼容既有单元测试环境。
"""

from datetime import datetime
from typing import Dict, Any


class ProductionMonitor:
    """生产环境监控器占位实现"""

    def __init__(self) -> None:
        self.status = "idle"

    def start(self) -> None:
        """启动监控"""
        self.status = "running"

    def stop(self) -> None:
        """停止监控"""
        self.status = "stopped"

    def monitor_system(self) -> Dict[str, Any]:
        """执行一次系统监控，返回基本状态信息"""
        return {
            "status": self.status,
            "checked_at": datetime.now().isoformat(),
        }

    def health_check(self) -> Dict[str, Any]:
        """健康检查接口"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
        }


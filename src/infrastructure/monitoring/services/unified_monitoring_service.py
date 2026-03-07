"""
unified_monitoring 模块

提供 unified_monitoring 相关功能和接口。
"""

import logging

from .continuous_monitoring_service import ContinuousMonitoringSystem
from datetime import datetime
from typing import Dict, Any, Optional
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025 基础设施层统一监控接口

统一监控服务接口，为各层提供一致的监控服务访问。

作者: RQA2025 Team
创建时间: 2025年9月28日
"""

logger = logging.getLogger(__name__)


class UnifiedMonitoring:
    """统一监控服务接口"""

    def __init__(self):
        """初始化统一监控服务"""
        self._monitoring_system = None
        self._initialized = False

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化监控系统

        Args:
            config: 配置参数

        Returns:
            bool: 初始化是否成功
        """
        try:
            self._monitoring_system = ContinuousMonitoringSystem()
            self._initialized = True
            logger.info("统一监控服务初始化成功")
            return True
        except Exception as e:
            logger.error(f"统一监控服务初始化失败: {e}")
            self._initialized = False
            return False

    def start_monitoring(self) -> bool:
        """启动监控

        Returns:
            bool: 启动是否成功
        """
        if not self._initialized or not self._monitoring_system:
            logger.warning("监控系统未初始化")
            return False

        try:
            self._monitoring_system.start_monitoring()
            logger.info("监控服务启动成功")
            return True
        except Exception as e:
            logger.error(f"启动监控服务失败: {e}")
            return False

    def stop_monitoring(self) -> bool:
        """停止监控

        Returns:
            bool: 停止是否成功
        """
        if not self._monitoring_system:
            return True

        try:
            self._monitoring_system.stop_monitoring()
            logger.info("监控服务停止成功")
            return True
        except Exception as e:
            logger.error(f"停止监控服务失败: {e}")
            return False

    def get_monitoring_report(self) -> Dict[str, Any]:
        """获取监控报告

        Returns:
            Dict[str, Any]: 监控报告数据
        """
        if not self._initialized or not self._monitoring_system:
            return {
                "status": "error",
                "message": "监控系统未初始化",
                "timestamp": datetime.now().isoformat()
            }

        try:
            return self._monitoring_system.get_monitoring_report()
        except Exception as e:
            logger.error(f"获取监控报告失败: {e}")
            return {
                "status": "error",
                "message": f"获取监控报告失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def get_status(self) -> Dict[str, Any]:
        """获取监控服务状态

        Returns:
            Dict[str, Any]: 状态信息
        """
        return {
            "service_name": "unified_monitoring",
            "initialized": self._initialized,
            "running": self._monitoring_system.monitoring_active if self._monitoring_system else False,
            "timestamp": datetime.now().isoformat()
        }

    def health_check(self) -> Dict[str, Any]:
        """健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        status = self.get_status()
        is_healthy = status.get("initialized", False) and status.get("running", False)

        return {
            "service": "unified_monitoring",
            "healthy": is_healthy,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }

    def __del__(self):
        """析构函数"""
        self.stop_monitoring()

# 便捷函数


def get_unified_monitoring() -> UnifiedMonitoring:
    """获取统一监控服务实例

    Returns:
        UnifiedMonitoring: 统一监控服务实例
    """
    monitoring = UnifiedMonitoring()
    monitoring.initialize()
    return monitoring


def create_monitoring_service(config: Optional[Dict[str, Any]] = None) -> UnifiedMonitoring:
    """创建监控服务

    Args:
        config: 配置参数

    Returns:
        UnifiedMonitoring: 配置后的监控服务实例
    """
    monitoring = UnifiedMonitoring()
    monitoring.initialize(config)
    return monitoring

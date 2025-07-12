#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
灾备系统状态监控模块
实时监控主备节点状态和同步情况
"""

import time
import threading
from typing import Dict, Any
from dataclasses import dataclass
from src.infrastructure.error import ErrorHandler
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class NodeStatus:
    """节点状态数据类"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    service_status: Dict[str, bool]
    last_heartbeat: float
    is_primary: bool

class DisasterMonitor:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化灾备监控器
        :param config: 监控配置
        """
        self.config = config
        self.error_handler = ErrorHandler()
        self.monitoring_interval = config.get("interval", 5)
        self.running = False
        self.thread = None
        self.node_status = {
            "primary": None,
            "secondary": None
        }
        self.sync_status = {
            "last_sync_time": 0,
            "sync_lag": 0,
            "queue_size": 0
        }
        self.alert_rules = config.get("alert_rules", {})
        self.alert_history = []

    def start(self):
        """启动监控"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        self.thread.start()
        logger.info("Disaster monitor started")

    def stop(self):
        """停止监控"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("Disaster monitor stopped")

    def _monitor_loop(self):
        """监控主循环"""
        while self.running:
            try:
                # 1. 收集节点状态
                self._collect_node_status()

                # 2. 检查同步状态
                self._check_sync_status()

                # 3. 执行健康检查
                self._perform_health_checks()

                # 4. 检查告警条件
                self._check_alerts()

                time.sleep(self.monitoring_interval)

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self.error_handler.handle(e)
                time.sleep(1)

    def _collect_node_status(self):
        """收集节点状态信息"""
        # 实现主节点状态收集
        primary_status = NodeStatus(
            cpu_usage=self._get_cpu_usage("primary"),
            memory_usage=self._get_memory_usage("primary"),
            disk_usage=self._get_disk_usage("primary"),
            service_status=self._get_service_status("primary"),
            last_heartbeat=time.time(),
            is_primary=True
        )
        self.node_status["primary"] = primary_status

        # 实现备用节点状态收集
        secondary_status = NodeStatus(
            cpu_usage=self._get_cpu_usage("secondary"),
            memory_usage=self._get_memory_usage("secondary"),
            disk_usage=self._get_disk_usage("secondary"),
            service_status=self._get_service_status("secondary"),
            last_heartbeat=time.time(),
            is_primary=False
        )
        self.node_status["secondary"] = secondary_status

    def _get_cpu_usage(self, node: str) -> float:
        """获取节点CPU使用率"""
        # 实现实际CPU使用率获取逻辑
        return 0.0

    def _get_memory_usage(self, node: str) -> float:
        """获取节点内存使用率"""
        # 实现实际内存使用率获取逻辑
        return 0.0

    def _get_disk_usage(self, node: str) -> float:
        """获取节点磁盘使用率"""
        # 实现实际磁盘使用率获取逻辑
        return 0.0

    def _get_service_status(self, node: str) -> Dict[str, bool]:
        """获取节点服务状态"""
        # 实现实际服务状态检查逻辑
        return {}

    def _check_sync_status(self):
        """检查数据同步状态"""
        # 实现同步状态检查逻辑
        self.sync_status = {
            "last_sync_time": time.time(),
            "sync_lag": 0,
            "queue_size": 0
        }

    def _perform_health_checks(self):
        """执行健康检查"""
        # 检查主节点健康状况
        primary_healthy = self._is_node_healthy(self.node_status["primary"])

        # 检查备用节点健康状况
        secondary_healthy = self._is_node_healthy(self.node_status["secondary"])

        # 记录健康状态
        self.health_status = {
            "primary": primary_healthy,
            "secondary": secondary_healthy,
            "timestamp": time.time()
        }

    def _is_node_healthy(self, node_status: NodeStatus) -> bool:
        """检查节点是否健康"""
        if not node_status:
            return False

        # 检查资源使用率
        if (node_status.cpu_usage > self.alert_rules.get("cpu_threshold", 90) or
            node_status.memory_usage > self.alert_rules.get("memory_threshold", 90) or
            node_status.disk_usage > self.alert_rules.get("disk_threshold", 90)):
            return False

        # 检查关键服务状态
        for service, required in self.alert_rules.get("critical_services", {}).items():
            if required and not node_status.service_status.get(service, False):
                return False

        # 检查心跳超时
        if time.time() - node_status.last_heartbeat > self.alert_rules.get("heartbeat_timeout", 30):
            return False

        return True

    def _check_alerts(self):
        """检查并触发告警"""
        # 检查主节点告警
        if not self.health_status["primary"]:
            self._trigger_alert(
                "PRIMARY_NODE_UNHEALTHY",
                f"Primary node is unhealthy. CPU: {self.node_status['primary'].cpu_usage}%",
                severity="critical"
            )

        # 检查备用节点告警
        if not self.health_status["secondary"]:
            self._trigger_alert(
                "SECONDARY_NODE_UNHEALTHY",
                f"Secondary node is unhealthy. Memory: {self.node_status['secondary'].memory_usage}%",
                severity="warning"
            )

        # 检查同步延迟告警
        if self.sync_status["sync_lag"] > self.alert_rules.get("max_sync_lag", 10):
            self._trigger_alert(
                "SYNC_LAG_TOO_HIGH",
                f"Sync lag is too high: {self.sync_status['sync_lag']}s",
                severity="warning"
            )

    def _trigger_alert(self, alert_type: str, message: str, severity: str = "warning"):
        """触发告警"""
        alert = {
            "type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": time.time()
        }
        self.alert_history.append(alert)
        logger.warning(f"ALERT: {alert_type} - {message}")

        # 实现实际告警通知逻辑
        self.error_handler.handle(
            Exception(f"{severity.upper()} ALERT: {message}"),
            level=severity
        )

    def get_status(self) -> Dict[str, Any]:
        """获取当前监控状态"""
        return {
            "node_status": {
                "primary": self._serialize_node_status(self.node_status["primary"]),
                "secondary": self._serialize_node_status(self.node_status["secondary"])
            },
            "sync_status": self.sync_status,
            "health_status": self.health_status,
            "last_alert": self.alert_history[-1] if self.alert_history else None
        }

    def _serialize_node_status(self, status: NodeStatus) -> Dict[str, Any]:
        """序列化节点状态"""
        if not status:
            return {}

        return {
            "cpu_usage": status.cpu_usage,
            "memory_usage": status.memory_usage,
            "disk_usage": status.disk_usage,
            "service_status": status.service_status,
            "last_heartbeat": status.last_heartbeat,
            "is_primary": status.is_primary
        }

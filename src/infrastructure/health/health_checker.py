#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
服务健康检查模块
负责监控系统各服务的健康状态
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.config.config_manager import ConfigManager

logger = get_logger(__name__)

@dataclass
class HealthStatus:
    """服务健康状态"""
    service: str
    status: str  # UP, DOWN, DEGRADED
    timestamp: float
    details: Dict[str, Any]
    last_check: float

class HealthChecker:
    def __init__(self, config: Dict[str, Any], config_manager: Optional[ConfigManager] = None):
        """
        初始化健康检查器
        :param config: 系统配置
        :param config_manager: 可选的配置管理器实例，用于测试时注入mock对象
        """
        self.config = config
        
        # 测试钩子：允许注入mock的ConfigManager
        if config_manager is not None:
            self.config_manager = config_manager
        else:
            self.config_manager = ConfigManager(config)
            
        self.health_status: Dict[str, HealthStatus] = {}
        self.checkers = {
            'database': self._check_database,
            'redis': self._check_redis,
            'trading_engine': self._check_trading_engine,
            'risk_system': self._check_risk_system,
            'data_service': self._check_data_service
        }
        self.running = False
        self.check_thread = None

        # 加载健康检查配置
        self.health_config = self.config_manager.get_config('health_check', {})
        self.check_interval = self.health_config.get('interval', 10)
        self.services_to_check = self.health_config.get('services', list(self.checkers.keys()))

    def start(self) -> None:
        """
        启动健康检查
        """
        if self.running:
            return

        self.running = True
        self.check_thread = threading.Thread(
            target=self._check_loop,
            daemon=True
        )
        self.check_thread.start()
        logger.info("健康检查服务已启动")

    def stop(self) -> None:
        """
        停止健康检查
        """
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5)
        logger.info("健康检查服务已停止")

    def get_status(self, service: Optional[str] = None) -> Dict[str, HealthStatus]:
        """
        获取健康状态
        :param service: 服务名称，None表示获取所有
        :return: 健康状态字典
        """
        if service:
            status = self.health_status.get(service)
            return {service: status} if status else {}
        return self.health_status.copy()

    def is_healthy(self, service: str) -> bool:
        """
        检查服务是否健康
        :param service: 服务名称
        :return: 是否健康
        """
        status = self.health_status.get(service)
        return status is not None and status.status == "UP"

    async def health_endpoint(self) -> Dict[str, Any]:
        """
        健康检查端点
        :return: 健康状态响应
        """
        try:
            # 执行健康检查
            self._perform_checks()
            
            # 构建响应
            response = {
                "status": "healthy",
                "timestamp": time.time(),
                "services": {}
            }
            
            all_healthy = True
            for service, status in self.health_status.items():
                response["services"][service] = {
                    "status": status.status,
                    "timestamp": status.timestamp,
                    "details": status.details
                }
                if status.status != "UP":
                    all_healthy = False
            
            if not all_healthy:
                response["status"] = "unhealthy"
            
            return response
            
        except Exception as e:
            logger.error(f"健康检查端点出错: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }

    async def ready_endpoint(self) -> Dict[str, Any]:
        """
        就绪检查端点
        :return: 就绪状态响应
        """
        try:
            # 检查关键服务是否就绪
            critical_services = ['database', 'trading_engine']
            ready_services = []
            
            for service in critical_services:
                if self.is_healthy(service):
                    ready_services.append(service)
            
            is_ready = len(ready_services) == len(critical_services)
            
            return {
                "ready": is_ready,
                "timestamp": time.time(),
                "ready_services": ready_services,
                "required_services": critical_services
            }
            
        except Exception as e:
            logger.error(f"就绪检查端点出错: {str(e)}")
            return {
                "ready": False,
                "error": str(e),
                "timestamp": time.time()
            }

    def _check_loop(self) -> None:
        """
        健康检查主循环
        """
        logger.info(f"健康检查循环启动，间隔: {self.check_interval}秒")

        while self.running:
            try:
                self._perform_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"健康检查循环出错: {str(e)}")
                time.sleep(1)

    def _perform_checks(self) -> None:
        """
        执行所有健康检查
        """
        for service in self.services_to_check:
            if service in self.checkers:
                try:
                    status, details = self.checkers[service]()
                    self.health_status[service] = HealthStatus(
                        service=service,
                        status=status,
                        timestamp=time.time(),
                        details=details,
                        last_check=time.time()
                    )
                except Exception as e:
                    logger.error(f"检查服务 {service} 健康状态失败: {str(e)}")
                    self.health_status[service] = HealthStatus(
                        service=service,
                        status="DOWN",
                        timestamp=time.time(),
                        details={"error": str(e)},
                        last_check=time.time()
                    )

    def _check_database(self) -> tuple[str, Dict[str, Any]]:
        """
        检查数据库健康状态
        :return: (状态, 详情)
        """
        # 实际项目中应实现真实检查逻辑
        db_config = self.config_manager.get_config('database')
        if not db_config:
            return "DOWN", {"error": "Database config missing"}

        # 模拟检查
        return "UP", {
            "connection_time": 5.2,
            "query_latency": 12.5,
            "active_connections": 8
        }

    def _check_redis(self) -> tuple[str, Dict[str, Any]]:
        """
        检查Redis健康状态
        :return: (状态, 详情)
        """
        # 模拟检查
        return "UP", {
            "used_memory": "1.2GB",
            "ops_per_sec": 12500,
            "connected_clients": 3
        }

    def _check_trading_engine(self) -> tuple[str, Dict[str, Any]]:
        """
        检查交易引擎健康状态
        :return: (状态, 详情)
        """
        # 模拟检查
        return "UP", {
            "order_rate": "1500/s",
            "latency": "45ms",
            "queue_depth": 12
        }

    def _check_risk_system(self) -> tuple[str, Dict[str, Any]]:
        """
        检查风控系统健康状态
        :return: (状态, 详情)
        """
        # 模拟检查
        return "UP", {
            "check_rate": "8000/s",
            "avg_latency": "3ms",
            "rejected_orders": 5
        }

    def _check_data_service(self) -> tuple[str, Dict[str, Any]]:
        """
        检查数据服务健康状态
        :return: (状态, 详情)
        """
        # 模拟检查
        return "UP", {
            "data_freshness": "2s",
            "cache_hit_rate": 0.85,
            "api_latency": "120ms"
        }

    def get_health_report(self) -> Dict[str, Any]:
        """
        获取健康报告
        :return: 健康报告字典
        """
        report = {
            "timestamp": time.time(),
            "overall_status": "UP",
            "services": {},
            "summary": {
                "total_services": len(self.health_status),
                "healthy_services": 0,
                "unhealthy_services": 0
            }
        }

        healthy_count = 0
        for service, status in self.health_status.items():
            report["services"][service] = {
                "status": status.status,
                "timestamp": status.timestamp,
                "details": status.details,
                "last_check": status.last_check
            }
            if status.status == "UP":
                healthy_count += 1

        report["summary"]["healthy_services"] = healthy_count
        report["summary"]["unhealthy_services"] = len(self.health_status) - healthy_count

        if healthy_count < len(self.health_status):
            report["overall_status"] = "DEGRADED"
        if healthy_count == 0:
            report["overall_status"] = "DOWN"

        return report

    def register_custom_check(self, service: str, checker: Any) -> None:
        """
        注册自定义健康检查
        :param service: 服务名称
        :param checker: 检查函数
        """
        self.checkers[service] = checker
        if service not in self.services_to_check:
            self.services_to_check.append(service)

    def trigger_manual_check(self, service: Optional[str] = None) -> None:
        """
        触发手动健康检查
        :param service: 服务名称，None表示检查所有
        """
        if service:
            if service in self.checkers:
                self._perform_checks()
        else:
            self._perform_checks()

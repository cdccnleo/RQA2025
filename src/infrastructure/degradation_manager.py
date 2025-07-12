#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
服务降级管理器
负责监控系统服务状态，在异常情况下自动降级服务
"""

import threading
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from src.infrastructure.utils.logger import get_logger
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.health.health_checker import HealthChecker
from src.infrastructure.error.circuit_breaker import CircuitBreaker

logger = get_logger(__name__)

@dataclass
class ServiceLevel:
    """服务级别定义"""
    name: str
    priority: int  # 1-10，数字越大优先级越高
    core: bool     # 是否核心服务
    current_level: int  # 当前降级级别
    max_level: int     # 最大允许降级级别

@dataclass
class DegradationRule:
    """降级规则定义"""
    condition: str  # 触发条件
    action: str     # 执行动作
    level: int      # 降级级别
    cooldown: int   # 冷却时间(秒)
    priority: int = 1   # 规则优先级
    enabled: bool = True   # 是否启用

class DegradationManager:
    def __init__(
        self, 
        config: Dict[str, Any],
        config_manager: Optional[ConfigManager] = None,
        health_checker: Optional[HealthChecker] = None,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        """
        初始化降级管理器
        :param config: 系统配置
        :param config_manager: 可选的配置管理器实例，用于测试时注入mock对象
        :param health_checker: 可选的健康检查器实例，用于测试时注入mock对象
        :param circuit_breaker: 可选的熔断器实例，用于测试时注入mock对象
        """
        self.config = config
        
        # 测试钩子：允许注入mock的依赖
        self.config_manager = config_manager or ConfigManager(config)
        self.health_checker = health_checker or HealthChecker(config)
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        
        self.services: Dict[str, ServiceLevel] = {}
        self.rules: List[DegradationRule] = []
        self.lock = threading.Lock()
        self.running = False

        # 加载降级配置
        self._load_config()

    def start(self) -> None:
        """
        启动降级监控
        """
        if self.running:
            return

        self.running = True
        monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True
        )
        monitor_thread.start()
        logger.info("服务降级监控已启动")

    def stop(self) -> None:
        """
        停止降级监控
        """
        self.running = False
        logger.info("服务降级监控已停止")

    def register_service(self, name: str, priority: int, core: bool, max_level: int = 3) -> None:
        """
        注册服务
        :param name: 服务名称
        :param priority: 优先级(1-10)
        :param core: 是否核心服务
        :param max_level: 最大允许降级级别
        """
        with self.lock:
            self.services[name] = ServiceLevel(
                name=name,
                priority=priority,
                core=core,
                current_level=0,
                max_level=max_level
            )
        logger.info(f"注册服务: {name}, 优先级: {priority}, 核心: {core}")

    def add_rule(self, condition: str, action: str, level: int, cooldown: int = 60) -> None:
        """
        添加降级规则
        :param condition: 触发条件表达式
        :param action: 执行动作
        :param level: 降级级别
        :param cooldown: 冷却时间(秒)
        """
        with self.lock:
            self.rules.append(DegradationRule(
                condition=condition,
                action=action,
                level=level,
                cooldown=cooldown
            ))
        logger.info(f"添加降级规则: {condition} -> {action} (级别: {level})")

    def degrade_service(self, name: str, level: int) -> bool:
        """
        降级指定服务
        :param name: 服务名称
        :param level: 降级级别
        :return: 是否成功
        """
        with self.lock:
            if name not in self.services:
                logger.warning(f"未知服务: {name}")
                return False

            if level > self.services[name].max_level:
                logger.warning(f"服务 {name} 不允许降级到级别 {level}")
                return False

            self.services[name].current_level = level
            logger.warning(f"服务 {name} 降级到级别 {level}")
            return True

    def restore_service(self, name: str) -> bool:
        """
        恢复指定服务
        :param name: 服务名称
        :return: 是否成功
        """
        with self.lock:
            if name not in self.services:
                logger.warning(f"未知服务: {name}")
                return False

            if self.services[name].current_level == 0:
                return True

            self.services[name].current_level = 0
            logger.info(f"服务 {name} 已恢复")
            return True

    def get_service_level(self, name: str) -> Optional[int]:
        """
        获取服务当前降级级别
        :param name: 服务名称
        :return: 降级级别(0表示不降级)
        """
        with self.lock:
            if name not in self.services:
                return None
            return self.services[name].current_level

    def is_degraded(self, name: str) -> bool:
        """
        检查服务是否被降级
        :param name: 服务名称
        :return: 是否被降级
        """
        with self.lock:
            if name not in self.services:
                return False
            return self.services[name].current_level > 0

    def auto_degrade(self) -> None:
        """
        自动执行降级检查
        """
        # 获取系统状态
        health_status = self.health_checker.get_status()
        breaker_status = {"state": self.circuit_breaker.state} if self.circuit_breaker else {}

        # 评估所有规则
        for rule in self.rules:
            try:
                # 简化版条件评估 - 实际项目应使用更复杂的表达式解析
                should_trigger = self._evaluate_condition(rule.condition, health_status or {}, breaker_status)
                if should_trigger:
                    self._execute_action(rule.action, rule.level)
                    time.sleep(0.1)  # 防止密集触发
            except Exception as e:
                logger.error(f"评估降级规则出错: {str(e)}")

    def _load_config(self) -> None:
        """加载降级配置"""
        degradation_config = self.config_manager.get_config('degradation', {})

        # 加载服务定义
        for service in degradation_config.get('services', []):
            self.register_service(
                name=service['name'],
                priority=service['priority'],
                core=service['core'],
                max_level=service.get('max_level', 3)
            )

        # 加载降级规则
        for rule in degradation_config.get('rules', []):
            self.add_rule(
                condition=rule['condition'],
                action=rule['action'],
                level=rule['level'],
                cooldown=rule.get('cooldown', 60)
            )

    def _evaluate_condition(self, condition: str, health: Dict, breaker: Dict) -> bool:
        """
        评估触发条件
        :param condition: 条件表达式
        :param health: 健康状态
        :param breaker: 熔断状态
        :return: 是否满足条件
        """
        # 简化的条件评估 - 实际项目应实现更完整的表达式解析
        if "health.database.status == 'DOWN'" in condition:
            return health.get('database', {}).get('status') == 'DOWN'
        elif "breaker.trading_engine.state == 'OPEN'" in condition:
            return breaker.get('trading_engine', {}).get('state') == 'OPEN'
        elif "health.redis.used_memory > '2GB'" in condition:
            return health.get('redis', {}).get('used_memory', '0') > '2GB'
        return False

    def _execute_action(self, action: str, level: int) -> None:
        """
        执行降级动作
        :param action: 动作指令
        :param level: 降级级别
        """
        # 简化的动作执行 - 实际项目应实现更完整的指令解析
        if action.startswith("degrade:"):
            service = action.split(":")[1].strip()
            self.degrade_service(service, level)
        elif action == "disable_non_core":
            self._degrade_non_core_services(level)

    def _degrade_non_core_services(self, level: int) -> None:
        """
        降级所有非核心服务
        :param level: 降级级别
        """
        with self.lock:
            for name, service in self.services.items():
                if not service.core and service.current_level < level:
                    service.current_level = level
                    logger.warning(f"非核心服务 {name} 降级到级别 {level}")

    def _monitor_loop(self) -> None:
        """
        降级监控循环
        """
        logger.info("降级监控循环启动")
        while self.running:
            try:
                self.auto_degrade()
                time.sleep(10)  # 每10秒检查一次
            except Exception as e:
                logger.error(f"降级监控循环出错: {str(e)}")
                time.sleep(30)

    def get_status_report(self) -> Dict[str, Any]:
        """
        获取降级状态报告
        :return: 状态报告字典
        """
        report = {
            "timestamp": time.time(),
            "services": [],
            "degraded_services": [],
            "rules": len(self.rules)
        }

        with self.lock:
            for name, service in self.services.items():
                service_info = {
                    "name": name,
                    "priority": service.priority,
                    "core": service.core,
                    "current_level": service.current_level,
                    "max_level": service.max_level
                }
                report["services"].append(service_info)
                if service.current_level > 0:
                    report["degraded_services"].append(service_info)

        return report

    def force_degrade_all(self, level: int) -> None:
        """
        强制降级所有服务(测试用)
        :param level: 降级级别
        """
        with self.lock:
            for name in self.services:
                if level <= self.services[name].max_level:
                    self.services[name].current_level = level
        logger.warning(f"强制所有服务降级到级别 {level}")

    def force_restore_all(self) -> None:
        """
        强制恢复所有服务(测试用)
        """
        with self.lock:
            for name in self.services:
                self.services[name].current_level = 0
        logger.info("强制所有服务恢复")

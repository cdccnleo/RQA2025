#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略服务依赖注入配置
Strategy Service Dependency Injection Configuration

配置策略服务相关的依赖注入和组件注册。
"""

from typing import Dict, Any
import logging
from strategy.core.container import DependencyContainer, Lifecycle
from .strategy_service import UnifiedStrategyService
from ..backtest.backtest_service import BacktestService
from ..backtest.backtest_engine import BacktestEngine
from ..backtest.backtest_persistence import BacktestPersistence
from ...optimization.strategy.optimization_service import OptimizationService
from ..optimization.parameter_optimizer import ParameterOptimizer
from ..optimization.walk_forward_optimizer import WalkForwardOptimizer
from ..monitoring.monitoring_service import MonitoringService
from ..monitoring.alert_service import AlertService
from ..lifecycle.strategy_lifecycle_manager import StrategyLifecycleManager

logger = logging.getLogger(__name__)


class StrategyServiceDependencyConfig:

    """
    策略服务依赖注入配置
    Strategy Service Dependency Injection Configuration

    配置所有策略服务组件的依赖注入关系。
    """

    def __init__(self, container: DependencyContainer):
        """
        初始化配置

        Args:
            container: 依赖注入容器
        """
        self.container = container
        self.configured_services = set()

        logger.info("策略服务依赖注入配置初始化完成")

    def configure_all(self):
        """
        配置所有策略服务组件
        """
        try:
            logger.info("开始配置策略服务依赖注入...")

            # 配置核心服务
            self._configure_core_services()

            # 配置回测服务
            self._configure_backtest_services()

            # 配置优化服务
            self._configure_optimization_services()

            # 配置监控服务
            self._configure_monitoring_services()

            # 配置生命周期管理
            self._configure_lifecycle_services()

            # 配置服务间依赖关系
            self._configure_service_dependencies()

            logger.info("策略服务依赖注入配置完成")

        except Exception as e:
            logger.error(f"配置策略服务依赖注入失败: {e}")
            raise

    def _configure_core_services(self):
        """配置核心服务"""
        # 注册策略服务
        self.container.register(
            name="strategy_service",
            service_class=UnifiedStrategyService,
            lifecycle=Lifecycle.SINGLETON,
            description="统一策略服务"
        )

        # 注册策略工厂
        from ..strategies.strategy_factory import StrategyFactory
        self.container.register(
            name="strategy_factory",
            service_class=StrategyFactory,
            lifecycle=Lifecycle.SINGLETON,
            description="策略工厂"
        )

        logger.info("核心服务配置完成")

    def _configure_backtest_services(self):
        """配置回测服务"""
        # 注册回测引擎
        self.container.register(
            name="backtest_engine",
            service_class=BacktestEngine,
            lifecycle=Lifecycle.SINGLETON,
            description="回测引擎"
        )

        # 注册回测持久化
        self.container.register(
            name="backtest_persistence",
            service_class=BacktestPersistence,
            lifecycle=Lifecycle.SINGLETON,
            description="回测持久化",
            parameters={
                "storage_path": "./data / backtest"
            }
        )

        # 注册回测服务
        self.container.register(
            name="backtest_service",
            service_class=BacktestService,
            lifecycle=Lifecycle.SINGLETON,
            description="回测服务",
            dependencies=["strategy_service", "backtest_engine", "backtest_persistence"]
        )

        logger.info("回测服务配置完成")

    def _configure_optimization_services(self):
        """配置优化服务"""
        # 注册参数优化器
        self.container.register(
            name="parameter_optimizer",
            service_class=ParameterOptimizer,
            lifecycle=Lifecycle.SINGLETON,
            description="参数优化器"
        )

        # 注册步进优化器
        self.container.register(
            name="walk_forward_optimizer",
            service_class=WalkForwardOptimizer,
            lifecycle=Lifecycle.SINGLETON,
            description="步进优化器",
            dependencies=["backtest_service"]
        )

        # 注册优化服务
        self.container.register(
            name="optimization_service",
            service_class=OptimizationService,
            lifecycle=Lifecycle.SINGLETON,
            description="优化服务",
            dependencies=["strategy_service", "parameter_optimizer", "walk_forward_optimizer"]
        )

        logger.info("优化服务配置完成")

    def _configure_monitoring_services(self):
        """配置监控服务"""
        # 注册监控服务
        self.container.register(
            name="monitoring_service",
            service_class=MonitoringService,
            lifecycle=Lifecycle.SINGLETON,
            description="监控服务"
        )

        # 注册告警服务
        self.container.register(
            name="alert_service",
            service_class=AlertService,
            lifecycle=Lifecycle.SINGLETON,
            description="告警服务"
        )

        logger.info("监控服务配置完成")

    def _configure_lifecycle_services(self):
        """配置生命周期管理服务"""
        # 注册生命周期管理器
        self.container.register(
            name="lifecycle_manager",
            service_class=StrategyLifecycleManager,
            lifecycle=Lifecycle.SINGLETON,
            description="策略生命周期管理器",
            dependencies=["strategy_service", "strategy_factory", "backtest_persistence"]
        )

        logger.info("生命周期管理服务配置完成")

    def _configure_service_dependencies(self):
        """配置服务间依赖关系"""
        try:
            # 为策略服务注册依赖
            if self.container.is_registered("strategy_service"):
                strategy_service = self.container.resolve("strategy_service")

                # 注册回测服务
                if self.container.is_registered("backtest_service"):
                    backtest_service = self.container.resolve("backtest_service")
                    strategy_service.register_backtest_service(backtest_service)

                # 注册优化服务
                if self.container.is_registered("optimization_service"):
                    optimization_service = self.container.resolve("optimization_service")
                    strategy_service.register_optimization_service(optimization_service)

                # 注册监控服务
                if self.container.is_registered("monitoring_service"):
                    monitoring_service = self.container.resolve("monitoring_service")
                    strategy_service.register_monitoring_service(monitoring_service)

            logger.info("服务间依赖关系配置完成")

        except Exception as e:
            logger.error(f"配置服务间依赖关系失败: {e}")

    def get_service_status(self) -> Dict[str, Any]:
        """
        获取服务状态

        Returns:
            Dict[str, Any]: 服务状态信息
        """
        status = {
            "total_services": len(self.configured_services),
            "configured_services": list(self.configured_services),
            "container_health": self.container.is_healthy(),
            "service_health": {}
        }

        # 检查各服务的健康状态
        service_names = [
            "strategy_service", "backtest_service", "optimization_service",
            "monitoring_service", "alert_service", "lifecycle_manager"
        ]

        for service_name in service_names:
            try:
                if self.container.is_registered(service_name):
                    service_instance = self.container.resolve(service_name)
                    status["service_health"][service_name] = "healthy"
                else:
                    status["service_health"][service_name] = "not_registered"
            except Exception as e:
                status["service_health"][service_name] = f"error: {e}"

        return status

    def initialize_services(self):
        """
        初始化所有服务
        """
        try:
            logger.info("开始初始化策略服务...")

            # 初始化核心服务
            if self.container.is_registered("strategy_service"):
                strategy_service = self.container.resolve("strategy_service")
                # 这里可以添加额外的初始化逻辑

            # 初始化回测服务
            if self.container.is_registered("backtest_service"):
                backtest_service = self.container.resolve("backtest_service")
                # 这里可以添加回测服务的初始化逻辑

            # 初始化监控服务
            if self.container.is_registered("monitoring_service"):
                monitoring_service = self.container.resolve("monitoring_service")
                # 这里可以添加监控服务的初始化逻辑

            logger.info("策略服务初始化完成")

        except Exception as e:
            logger.error(f"策略服务初始化失败: {e}")
            raise

    def shutdown_services(self):
        """
        关闭所有服务
        """
        try:
            logger.info("开始关闭策略服务...")

            # 关闭监控服务
            if self.container.is_registered("monitoring_service"):
                monitoring_service = self.container.resolve("monitoring_service")
                # 这里可以添加监控服务的关闭逻辑

            # 关闭回测服务
            if self.container.is_registered("backtest_service"):
                backtest_service = self.container.resolve("backtest_service")
                # 这里可以添加回测服务的关闭逻辑

            logger.info("策略服务关闭完成")

        except Exception as e:
            logger.error(f"策略服务关闭失败: {e}")


def configure_strategy_services(container: DependencyContainer) -> StrategyServiceDependencyConfig:
    """
    配置策略服务依赖注入

    Args:
        container: 依赖注入容器

    Returns:
        StrategyServiceDependencyConfig: 配置实例
    """
    config = StrategyServiceDependencyConfig(container)
    config.configure_all()
    return config


# 便捷函数

def get_strategy_service(container: DependencyContainer) -> UnifiedStrategyService:
    """
    获取策略服务实例

    Args:
        container: 依赖注入容器

    Returns:
        UnifiedStrategyService: 策略服务实例
    """
    return container.resolve("strategy_service")


def get_backtest_service(container: DependencyContainer) -> BacktestService:
    """
    获取回测服务实例

    Args:
        container: 依赖注入容器

    Returns:
        BacktestService: 回测服务实例
    """
    return container.resolve("backtest_service")


def get_optimization_service(container: DependencyContainer) -> OptimizationService:
    """
    获取优化服务实例

    Args:
        container: 依赖注入容器

    Returns:
        OptimizationService: 优化服务实例
    """
    return container.resolve("optimization_service")


def get_monitoring_service(container: DependencyContainer) -> MonitoringService:
    """
    获取监控服务实例

    Args:
        container: 依赖注入容器

    Returns:
        MonitoringService: 监控服务实例
    """
    return container.resolve("monitoring_service")


def get_lifecycle_manager(container: DependencyContainer) -> StrategyLifecycleManager:
    """
    获取生命周期管理器实例

    Args:
        container: 依赖注入容器

    Returns:
        StrategyLifecycleManager: 生命周期管理器实例
    """
    return container.resolve("lifecycle_manager")


# 导出
__all__ = [
    'StrategyServiceDependencyConfig',
    'configure_strategy_services',
    'get_strategy_service',
    'get_backtest_service',
    'get_optimization_service',
    'get_monitoring_service',
    'get_lifecycle_manager'
]

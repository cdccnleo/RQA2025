#!/usr/bin/env python3
"""
RQA2025 量化交易系统架构启动器

基于21层架构设计，实现分层启动和管理：
- 核心业务层：策略层、交易层、风险控制层、特征层
- 核心支撑层：数据管理层、机器学习层、基础设施层、流处理层
- 辅助支撑层：核心服务层、监控层、优化层、网关层、适配器层、自动化层、弹性层、测试层、工具层

职责：
1. 协调各架构层级的启动顺序
2. 管理层间依赖关系
3. 提供统一的启动接口
4. 实现优雅关闭机制
"""

import os
import sys
import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ArchitectureLayer(Enum):
    """架构层级枚举"""
    # 核心业务层 (价值创造)
    STRATEGY = "strategy_layer"           # 策略层
    TRADING = "trading_layer"             # 交易层
    RISK_CONTROL = "risk_control_layer"   # 风险控制层
    FEATURE = "feature_layer"             # 特征层

    # 核心支撑层 (技术赋能)
    DATA_MANAGEMENT = "data_management_layer"  # 数据管理层
    MACHINE_LEARNING = "ml_layer"              # 机器学习层
    INFRASTRUCTURE = "infrastructure_layer"    # 基础设施层
    STREAMING = "streaming_layer"              # 流处理层

    # 辅助支撑层 (专业支撑)
    CORE_SERVICES = "core_services_layer"      # 核心服务层
    MONITORING = "monitoring_layer"            # 监控层
    OPTIMIZATION = "optimization_layer"        # 优化层
    GATEWAY = "gateway_layer"                  # 网关层
    ADAPTER = "adapter_layer"                  # 适配器层
    AUTOMATION = "automation_layer"            # 自动化层
    RESILIENCE = "resilience_layer"            # 弹性层
    TESTING = "testing_layer"                  # 测试层
    UTILS = "utils_layer"                      # 工具层

@dataclass
class LayerStatus:
    """层级状态"""
    layer: ArchitectureLayer
    status: str  # 'pending', 'starting', 'running', 'failed', 'stopped'
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    dependencies: List[ArchitectureLayer] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class ArchitectureStartupManager:
    """
    架构启动管理器

    负责：
    1. 定义层级启动顺序和依赖关系
    2. 协调各层级初始化
    3. 监控启动状态和健康检查
    4. 实现优雅关闭
    """

    def __init__(self):
        self.layer_status: Dict[ArchitectureLayer, LayerStatus] = {}
        self.layer_instances: Dict[ArchitectureLayer, Any] = {}
        self.startup_order = self._define_startup_order()
        self.logger = logging.getLogger(__name__)

    def _define_startup_order(self) -> List[ArchitectureLayer]:
        """
        定义架构层级启动顺序

        启动顺序基于依赖关系：
        1. 基础设施层优先启动（提供基础服务）
        2. 核心支撑层其次（提供技术能力）
        3. 核心业务层（实现业务价值）
        4. 辅助支撑层（提供支撑服务）
        """
        return [
            # 1. 基础设施层 - 基础服务
            ArchitectureLayer.INFRASTRUCTURE,
            ArchitectureLayer.UTILS,

            # 2. 核心支撑层 - 技术赋能
            ArchitectureLayer.CORE_SERVICES,
            ArchitectureLayer.DATA_MANAGEMENT,
            ArchitectureLayer.STREAMING,
            ArchitectureLayer.MACHINE_LEARNING,

            # 3. 核心业务层 - 价值创造
            ArchitectureLayer.FEATURE,
            ArchitectureLayer.STRATEGY,
            ArchitectureLayer.RISK_CONTROL,
            ArchitectureLayer.TRADING,

            # 4. 辅助支撑层 - 专业支撑
            ArchitectureLayer.ADAPTER,
            ArchitectureLayer.RESILIENCE,
            ArchitectureLayer.TESTING,
            ArchitectureLayer.AUTOMATION,
            ArchitectureLayer.OPTIMIZATION,
            ArchitectureLayer.MONITORING,
            ArchitectureLayer.GATEWAY,
        ]

    def _define_layer_dependencies(self) -> Dict[ArchitectureLayer, List[ArchitectureLayer]]:
        """定义层级依赖关系"""
        return {
            # 基础设施层依赖
            ArchitectureLayer.INFRASTRUCTURE: [],
            ArchitectureLayer.UTILS: [],

            # 核心支撑层依赖基础设施
            ArchitectureLayer.CORE_SERVICES: [ArchitectureLayer.INFRASTRUCTURE],
            ArchitectureLayer.DATA_MANAGEMENT: [ArchitectureLayer.INFRASTRUCTURE, ArchitectureLayer.CORE_SERVICES],
            ArchitectureLayer.STREAMING: [ArchitectureLayer.INFRASTRUCTURE, ArchitectureLayer.DATA_MANAGEMENT],
            ArchitectureLayer.MACHINE_LEARNING: [ArchitectureLayer.INFRASTRUCTURE, ArchitectureLayer.DATA_MANAGEMENT],

            # 核心业务层依赖支撑层
            ArchitectureLayer.FEATURE: [ArchitectureLayer.DATA_MANAGEMENT, ArchitectureLayer.MACHINE_LEARNING],
            ArchitectureLayer.STRATEGY: [ArchitectureLayer.FEATURE, ArchitectureLayer.MACHINE_LEARNING],
            ArchitectureLayer.RISK_CONTROL: [ArchitectureLayer.DATA_MANAGEMENT, ArchitectureLayer.STREAMING],
            ArchitectureLayer.TRADING: [ArchitectureLayer.STRATEGY, ArchitectureLayer.RISK_CONTROL, ArchitectureLayer.STREAMING],

            # 辅助支撑层依赖核心层
            ArchitectureLayer.ADAPTER: [ArchitectureLayer.INFRASTRUCTURE],
            ArchitectureLayer.RESILIENCE: [ArchitectureLayer.INFRASTRUCTURE, ArchitectureLayer.CORE_SERVICES],
            ArchitectureLayer.TESTING: [],
            ArchitectureLayer.AUTOMATION: [ArchitectureLayer.INFRASTRUCTURE, ArchitectureLayer.MONITORING],
            ArchitectureLayer.OPTIMIZATION: [ArchitectureLayer.INFRASTRUCTURE, ArchitectureLayer.MONITORING],
            ArchitectureLayer.MONITORING: [ArchitectureLayer.INFRASTRUCTURE],
            ArchitectureLayer.GATEWAY: [ArchitectureLayer.CORE_SERVICES, ArchitectureLayer.MONITORING],
        }

    def initialize_layer_status(self):
        """初始化层级状态"""
        dependencies = self._define_layer_dependencies()

        for layer in ArchitectureLayer:
            self.layer_status[layer] = LayerStatus(
                layer=layer,
                status='pending',
                dependencies=dependencies.get(layer, [])
            )

    def check_dependencies(self, layer: ArchitectureLayer) -> bool:
        """检查层级依赖是否满足"""
        status = self.layer_status[layer]

        for dep in status.dependencies:
            dep_status = self.layer_status[dep]
            if dep_status.status not in ['running']:
                return False

        return True

    def start_layer(self, layer: ArchitectureLayer) -> bool:
        """
        启动指定层级

        返回值：
        - True: 启动成功
        - False: 启动失败
        """
        try:
            self.logger.info(f"启动架构层级: {layer.value}")

            # 检查依赖
            if not self.check_dependencies(layer):
                self.logger.error(f"层级 {layer.value} 依赖未满足")
                self._update_layer_status(layer, 'failed', "依赖层级未就绪")
                return False

            # 更新状态为启动中
            self._update_layer_status(layer, 'starting')

            # 根据层级调用对应的启动逻辑
            instance = self._create_layer_instance(layer)
            if instance:
                # 调用层级启动方法
                if hasattr(instance, 'start') and callable(instance.start):
                    result = instance.start()
                    if result:
                        self.layer_instances[layer] = instance
                        self._update_layer_status(layer, 'running')
                        self.logger.info(f"层级 {layer.value} 启动成功")
                        return True
                    else:
                        self._update_layer_status(layer, 'failed', "启动方法返回失败")
                        return False
                else:
                    # 如果没有start方法，直接标记为运行中
                    self.layer_instances[layer] = instance
                    self._update_layer_status(layer, 'running')
                    self.logger.info(f"层级 {layer.value} 初始化成功")
                    return True
            else:
                self._update_layer_status(layer, 'failed', "实例创建失败")
                return False

        except Exception as e:
            self.logger.error(f"启动层级 {layer.value} 时发生错误: {e}")
            self._update_layer_status(layer, 'failed', str(e))
            return False

    def _create_layer_instance(self, layer: ArchitectureLayer) -> Optional[Any]:
        """创建层级实例"""
        try:
            if layer == ArchitectureLayer.INFRASTRUCTURE:
                # 基础设施层 - 导入现有的基础设施管理器
                from scripts.start_production import DatabaseManager, RedisCacheManager
                # 这里应该有一个基础设施管理器，但暂时返回None表示跳过
                return None

            elif layer == ArchitectureLayer.DATA_MANAGEMENT:
                # 数据管理层 - 数据采集和存储
                from scripts.start_production import QuantDataStorage, QuantDataCollector
                storage = QuantDataStorage()
                collector = QuantDataCollector()
                return {'storage': storage, 'collector': collector}

            elif layer == ArchitectureLayer.CORE_SERVICES:
                # 核心服务层 - 事件总线和依赖注入
                # 这里应该导入相关组件，暂时返回占位符
                return {'status': 'placeholder'}

            elif layer == ArchitectureLayer.GATEWAY:
                # 网关层 - Web API服务
                # 这里应该导入FastAPI应用，暂时返回占位符
                return {'status': 'placeholder'}

            elif layer == ArchitectureLayer.MONITORING:
                # 监控层 - 系统监控
                # 这里应该导入监控组件，暂时返回占位符
                return {'status': 'placeholder'}

            else:
                # 其他层级暂时返回占位符
                return {'status': 'placeholder', 'layer': layer.value}

        except ImportError as e:
            self.logger.warning(f"无法导入层级 {layer.value} 的组件: {e}")
            return None
        except Exception as e:
            self.logger.error(f"创建层级 {layer.value} 实例时发生错误: {e}")
            return None

    def _update_layer_status(self, layer: ArchitectureLayer, status: str, error_message: str = None):
        """更新层级状态"""
        layer_status = self.layer_status[layer]
        layer_status.status = status

        if status == 'starting':
            layer_status.start_time = time.time()
        elif status in ['running', 'failed', 'stopped']:
            layer_status.end_time = time.time()

        if error_message:
            layer_status.error_message = error_message

    def start_system(self) -> bool:
        """
        启动整个系统

        按照定义的顺序启动各个架构层级
        """
        self.logger.info("开始启动RQA2025量化交易系统...")

        # 初始化状态
        self.initialize_layer_status()

        # 按顺序启动各层级
        success_count = 0
        total_count = len(self.startup_order)

        for layer in self.startup_order:
            self.logger.info(f"[{success_count + 1}/{total_count}] 启动 {layer.value}...")

            if self.start_layer(layer):
                success_count += 1
            else:
                self.logger.error(f"层级 {layer.value} 启动失败，停止系统启动")
                return False

        self.logger.info(f"系统启动完成: {success_count}/{total_count} 个层级成功启动")
        return success_count == total_count

    def stop_system(self):
        """停止整个系统"""
        self.logger.info("开始停止RQA2025量化交易系统...")

        # 按相反顺序停止各层级
        for layer in reversed(self.startup_order):
            try:
                instance = self.layer_instances.get(layer)
                if instance and hasattr(instance, 'stop') and callable(instance.stop):
                    instance.stop()
                    self.logger.info(f"层级 {layer.value} 已停止")
                else:
                    self.logger.debug(f"层级 {layer.value} 无需停止")

                self._update_layer_status(layer, 'stopped')

            except Exception as e:
                self.logger.error(f"停止层级 {layer.value} 时发生错误: {e}")

        self.logger.info("系统停止完成")

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status_summary = {
            'total_layers': len(ArchitectureLayer),
            'running_layers': 0,
            'failed_layers': 0,
            'pending_layers': 0,
            'layer_details': {}
        }

        for layer, status in self.layer_status.items():
            status_summary['layer_details'][layer.value] = {
                'status': status.status,
                'start_time': status.start_time,
                'end_time': status.end_time,
                'error_message': status.error_message,
                'dependencies': [dep.value for dep in status.dependencies]
            }

            if status.status == 'running':
                status_summary['running_layers'] += 1
            elif status.status == 'failed':
                status_summary['failed_layers'] += 1
            elif status.status == 'pending':
                status_summary['pending_layers'] += 1

        return status_summary

    def health_check(self) -> Dict[str, Any]:
        """系统健康检查"""
        health_status = {
            'overall_status': 'healthy',
            'timestamp': time.time(),
            'layer_health': {}
        }

        # 检查各层级的健康状态
        for layer, instance in self.layer_instances.items():
            try:
                if instance and hasattr(instance, 'health_check'):
                    health = instance.health_check()
                    health_status['layer_health'][layer.value] = health

                    # 如果有层级不健康，整个系统标记为不健康
                    if health.get('status') != 'healthy':
                        health_status['overall_status'] = 'unhealthy'
                else:
                    health_status['layer_health'][layer.value] = {'status': 'unknown'}

            except Exception as e:
                health_status['layer_health'][layer.value] = {
                    'status': 'error',
                    'error': str(e)
                }
                health_status['overall_status'] = 'unhealthy'

        return health_status

# 全局架构启动管理器实例
architecture_manager = ArchitectureStartupManager()

def start_architecture_system() -> bool:
    """
    启动整个架构系统

    这是一个便捷函数，用于启动完整的RQA2025系统
    """
    return architecture_manager.start_system()

def stop_architecture_system():
    """停止整个架构系统"""
    architecture_manager.stop_system()

def get_architecture_status() -> Dict[str, Any]:
    """获取架构系统状态"""
    return architecture_manager.get_system_status()

def check_architecture_health() -> Dict[str, Any]:
    """检查架构系统健康状态"""
    return architecture_manager.health_check()

if __name__ == "__main__":
    # 测试启动
    print("RQA2025 架构启动器测试")

    success = start_architecture_system()
    if success:
        print("✅ 系统启动成功")

        # 显示状态
        status = get_architecture_status()
        print(f"运行层级: {status['running_layers']}/{status['total_layers']}")

        # 健康检查
        health = check_architecture_health()
        print(f"系统健康状态: {health['overall_status']}")

        # 停止系统
        stop_architecture_system()
        print("✅ 系统已停止")

    else:
        print("❌ 系统启动失败")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
业务流程集成模块
实现业务流程与各层服务的集成，提供统一的业务流程管理接口
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict

from src.core.constants import (
    DEFAULT_TIMEOUT, SECONDS_PER_MINUTE, MAX_RETRIES
)

from .foundation.base import BaseComponent
from .foundation.exceptions.core_exceptions import IntegrationException
from .orchestration.orchestrator_refactored import BusinessProcessOrchestrator
from .architecture_layers import (
    CoreServicesLayer,
    InfrastructureLayer,
    DataManagementLayer,
    FeatureProcessingLayer,
    ModelInferenceLayer,
    StrategyDecisionLayer,
    RiskComplianceLayer,
    TradingExecutionLayer,
    MonitoringFeedbackLayer
)

logger = logging.getLogger(__name__)


class IntegrationStatus(Enum):

    """集成状态枚举"""
    DISCONNECTED = "disconnected"      # 未连接
    CONNECTING = "connecting"          # 连接中
    CONNECTED = "connected"            # 已连接
    INTEGRATING = "integrating"        # 集成中
    INTEGRATED = "integrated"          # 已集成
    ERROR = "error"                    # 错误状态
    DISCONNECTING = "disconnecting"    # 断开连接中


@dataclass
class IntegrationConfig:

    """集成配置"""
    integration_id: str
    name: str
    description: str = ""
    version: str = "1.0.0"
    enabled: bool = True
    auto_reconnect: bool = True
    reconnect_interval: int = DEFAULT_TIMEOUT  # 秒
    max_reconnect_attempts: int = 5
    timeout: int = SECONDS_PER_MINUTE  # 秒
    retry_count: int = 3
    health_check_interval: int = SECONDS_PER_MINUTE  # 秒

    # 业务流程配置
    process_configs: List[str] = None
    layer_integrations: List[str] = None

    def __post_init__(self):

        if self.process_configs is None:
            self.process_configs = []
        if self.layer_integrations is None:
            self.layer_integrations = []


@dataclass
class IntegrationMetrics:

    """集成指标"""
    connection_count: int = 0
    disconnection_count: int = 0
    integration_count: int = 0
    error_count: int = 0
    last_connection_time: Optional[float] = None
    last_disconnection_time: Optional[float] = None
    last_integration_time: Optional[float] = None
    last_error_time: Optional[float] = None
    total_uptime: float = 0.0
    total_downtime: float = 0.0

    @property
    def availability(self) -> float:
        """可用性百分比"""
        total_time = self.total_uptime + self.total_downtime
        return (self.total_uptime / total_time * 100) if total_time > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """错误率"""
        total_operations = self.integration_count + self.error_count
        return (self.error_count / total_operations * 100) if total_operations > 0 else 0.0


class LayerIntegrationManager:

    """架构层集成管理器"""

    def __init__(self):

        self.layers = {}
        self.integrations = {}
        self.health_monitors = {}
        self._initialize_layers()

    def _initialize_layers(self):
        """初始化架构层"""
        try:
            # 创建核心服务层
            core_services = CoreServicesLayer()
            self.layers['core_services'] = core_services

            # 创建基础设施层
            infrastructure = InfrastructureLayer(core_services)
            self.layers['infrastructure'] = infrastructure

            # 创建数据管理层
            data_management = DataManagementLayer(infrastructure)
            self.layers['data_management'] = data_management

            # 创建特征处理层
            feature_processing = FeatureProcessingLayer(data_management)
            self.layers['feature_processing'] = feature_processing

            # 创建模型推理层
            model_inference = ModelInferenceLayer(feature_processing)
            self.layers['model_inference'] = model_inference

            # 创建策略决策层
            strategy_decision = StrategyDecisionLayer(model_inference)
            self.layers['strategy_decision'] = strategy_decision

            # 创建风控合规层
            risk_compliance = RiskComplianceLayer(strategy_decision)
            self.layers['risk_compliance'] = risk_compliance

            # 创建交易执行层
            trading_execution = TradingExecutionLayer(risk_compliance)
            self.layers['trading_execution'] = trading_execution

            # 创建监控反馈层
            monitoring_feedback = MonitoringFeedbackLayer(trading_execution)
            self.layers['monitoring_feedback'] = monitoring_feedback

            logger.info("架构层初始化完成")

        except Exception as e:
            logger.error(f"架构层初始化失败: {str(e)}")
            raise IntegrationException(f"架构层初始化失败: {str(e)}")

    def get_layer(self, layer_name: str) -> Optional[Any]:
        """获取指定层"""
        return self.layers.get(layer_name)

    def get_all_layers(self) -> Dict[str, Any]:
        """获取所有层"""
        return self.layers.copy()

    def check_layer_health(self, layer_name: str) -> bool:
        """检查指定层健康状态"""
        layer = self.layers.get(layer_name)
        if layer:
            try:
                return layer.is_healthy()
            except Exception as e:
                logger.error(f"检查层健康状态失败 {layer_name}: {str(e)}")
                return False
        return False

    def check_all_layers_health(self) -> Dict[str, bool]:
        """检查所有层健康状态"""
        health_status = {}
        for layer_name, layer in self.layers.items():
            health_status[layer_name] = self.check_layer_health(layer_name)
        return health_status


class ProcessIntegrationManager:

    """业务流程集成管理器"""

    def __init__(self, orchestrator: BusinessProcessOrchestrator):

        self.orchestrator = orchestrator
        self.integrated_processes = {}
        self.process_handlers = {}
        self.integration_metrics = defaultdict(IntegrationMetrics)
        self._initialize_process_handlers()

    def _initialize_process_handlers(self):
        """初始化流程处理器"""
        # 数据收集流程处理器
        self.process_handlers['data_collection'] = self._handle_data_collection
        # 特征提取流程处理器
        self.process_handlers['feature_extraction'] = self._handle_feature_extraction
        # 模型预测流程处理器
        self.process_handlers['model_prediction'] = self._handle_model_prediction
        # 策略决策流程处理器
        self.process_handlers['strategy_decision'] = self._handle_strategy_decision
        # 风控检查流程处理器
        self.process_handlers['risk_check'] = self._handle_risk_check
        # 交易执行流程处理器
        self.process_handlers['trading_execution'] = self._handle_trading_execution
        # 监控反馈流程处理器
        self.process_handlers['monitoring_feedback'] = self._handle_monitoring_feedback

    def integrate_process(self, process_id: str, process_config: dict) -> bool:
        """集成业务流程"""
        try:
            if process_id in self.integrated_processes:
                logger.warning(f"流程 {process_id} 已经集成")
                return True

            # 记录集成状态
            self.integrated_processes[process_id] = {
                'config': process_config,
                'status': 'integrated',
                'integrated_time': time.time(),
                'metrics': IntegrationMetrics()
            }

            logger.info(f"流程 {process_id} 集成成功")
            return True

        except Exception as e:
            logger.error(f"流程 {process_id} 集成失败: {str(e)}")
            return False

    def _get_process_handler(self, process_config: dict) -> Callable:
        """获取流程处理器"""
        process_type = process_config.get('type', 'default')
        return self.process_handlers.get(process_type, self._handle_default_process)

    def _handle_data_collection(self, context: dict) -> dict:
        """处理数据收集流程"""
        try:
            # 实现数据收集逻辑
            symbols = context.get('symbols', [])

            # 调用数据管理层
            data_layer = self.orchestrator.get_layer('data_management')
            if data_layer:
                result = data_layer.collect_market_data(symbols)
                return {'status': 'success', 'data': result}
            else:
                return {'status': 'error', 'message': '数据管理层不可用'}

        except Exception as e:
            logger.error(f"数据收集流程处理失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _handle_feature_extraction(self, context: dict) -> dict:
        """处理特征提取流程"""
        try:
            # 实现特征提取逻辑
            data = context.get('data', {})

            # 调用特征处理层
            feature_layer = self.orchestrator.get_layer('feature_processing')
            if feature_layer:
                result = feature_layer.extract_features(data)
                return {'status': 'success', 'features': result}
            else:
                return {'status': 'error', 'message': '特征处理层不可用'}

        except Exception as e:
            logger.error(f"特征提取流程处理失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _handle_model_prediction(self, context: dict) -> dict:
        """处理模型预测流程"""
        try:
            # 实现模型预测逻辑
            features = context.get('features', {})

            # 调用模型推理层
            model_layer = self.orchestrator.get_layer('model_inference')
            if model_layer:
                result = model_layer.predict(features)
                return {'status': 'success', 'predictions': result}
            else:
                return {'status': 'error', 'message': '模型推理层不可用'}

        except Exception as e:
            logger.error(f"模型预测流程处理失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _handle_strategy_decision(self, context: dict) -> dict:
        """处理策略决策流程"""
        try:
            # 实现策略决策逻辑
            predictions = context.get('predictions', {})

            # 调用策略决策层
            strategy_layer = self.orchestrator.get_layer('strategy_decision')
            if strategy_layer:
                result = strategy_layer.make_decision(predictions)
                return {'status': 'success', 'decision': result}
            else:
                return {'status': 'error', 'message': '策略决策层不可用'}

        except Exception as e:
            logger.error(f"策略决策流程处理失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _handle_risk_check(self, context: dict) -> dict:
        """处理风控检查流程"""
        try:
            # 实现风控检查逻辑
            decision = context.get('decision', {})

            # 调用风控合规层
            risk_layer = self.orchestrator.get_layer('risk_compliance')
            if risk_layer:
                result = risk_layer.check_risk(decision)
                return {'status': 'success', 'risk_result': result}
            else:
                return {'status': 'error', 'message': '风控合规层不可用'}

        except Exception as e:
            logger.error(f"风控检查流程处理失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _handle_trading_execution(self, context: dict) -> dict:
        """处理交易执行流程"""
        try:
            # 实现交易执行逻辑
            risk_result = context.get('risk_result', {})

            # 调用交易执行层
            trading_layer = self.orchestrator.get_layer('trading_execution')
            if trading_layer:
                result = trading_layer.generate_orders(risk_result)
                return {'status': 'success', 'orders': result}
            else:
                return {'status': 'error', 'message': '交易执行层不可用'}

        except Exception as e:
            logger.error(f"交易执行流程处理失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _handle_monitoring_feedback(self, context: dict) -> dict:
        """处理监控反馈流程"""
        try:
            # 实现监控反馈逻辑
            execution_result = context.get('execution_result', {})

            # 调用监控反馈层
            monitoring_layer = self.orchestrator.get_layer('monitoring_feedback')
            if monitoring_layer:
                result = monitoring_layer.update_performance_metrics(execution_result)
                return {'status': 'success', 'metrics': result}
            else:
                return {'status': 'error', 'message': '监控反馈层不可用'}

        except Exception as e:
            logger.error(f"监控反馈流程处理失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}

    def _handle_default_process(self, context: dict) -> dict:
        """处理默认流程"""
        logger.warning(f"未找到流程处理器，使用默认处理器")
        return {'status': 'warning', 'message': '使用默认流程处理器'}


class BusinessProcessIntegration(BaseComponent):

    """业务流程集成主类"""

    def __init__(self, config_dir: str = "config / integration", max_processes: int = MAX_RETRIES):

        super().__init__("BusinessProcessIntegration")
        self.config_dir = config_dir
        self.max_processes = max_processes

        # 集成状态
        self.integration_status = IntegrationStatus.DISCONNECTED
        self.integration_configs = {}
        self.integration_metrics = {}

        # 管理器
        self.orchestrator = None
        self.layer_manager = None
        self.process_manager = None

        # 监控和重连
        self.health_monitor_thread = None
        self.reconnect_thread = None
        self.running = False

        # 初始化锁
        self._lock = threading.RLock()

        logger.info("业务流程集成模块初始化完成")

    def initialize(self) -> bool:
        """初始化集成模块"""
        try:
            with self._lock:
                if self.integration_status != IntegrationStatus.DISCONNECTED:
                    logger.warning("集成模块已经初始化")
                    return True

                # 更新状态
                self.integration_status = IntegrationStatus.CONNECTING

                # 初始化业务流程编排器
                self.orchestrator = BusinessProcessOrchestrator(
                    config_dir="config / processes",
                    max_instances=self.max_processes
                )

                if not self.orchestrator.initialize():
                    raise IntegrationException("业务流程编排器初始化失败")

                # 初始化架构层集成管理器
                self.layer_manager = LayerIntegrationManager()

                # 初始化业务流程集成管理器
                self.process_manager = ProcessIntegrationManager(self.orchestrator)

                # 加载集成配置
                self._load_integration_configs()

                # 启动健康监控
                self._start_health_monitoring()

                # 更新状态
                self.integration_status = IntegrationStatus.INTEGRATED

                logger.info("业务流程集成模块初始化成功")
                return True

        except Exception as e:
            logger.error(f"业务流程集成模块初始化失败: {str(e)}")
            self.integration_status = IntegrationStatus.ERROR
            return False

    def shutdown(self) -> bool:
        """关闭集成模块"""
        try:
            with self._lock:
                if self.integration_status == IntegrationStatus.DISCONNECTED:
                    return True

                # 更新状态
                self.integration_status = IntegrationStatus.DISCONNECTING

                # 停止健康监控
                self._stop_health_monitoring()

                # 关闭业务流程编排器
                if self.orchestrator:
                    self.orchestrator.shutdown()

                # 清理资源
                self.orchestrator = None
                self.layer_manager = None
                self.process_manager = None

                # 更新状态
                self.integration_status = IntegrationStatus.DISCONNECTED

                logger.info("业务流程集成模块关闭成功")
                return True

        except Exception as e:
            logger.error(f"业务流程集成模块关闭失败: {str(e)}")
            return False

    def _load_integration_configs(self):
        """加载集成配置"""
        try:
            # 这里应该从配置文件加载集成配置
            # 暂时使用默认配置

            default_configs = [
                {
                    'integration_id': 'trading_cycle',
                    'name': '交易周期集成',
                    'description': '完整的交易周期流程集成',
                    'version': '1.0.0',
                    'enabled': True,
                    'process_configs': ['data_collection', 'feature_extraction', 'model_prediction', 'strategy_decision', 'risk_check', 'trading_execution', 'monitoring_feedback'],
                    'layer_integrations': ['core_services', 'infrastructure', 'data_management', 'feature_processing', 'model_inference', 'strategy_decision', 'risk_compliance', 'trading_execution', 'monitoring_feedback']
                }
            ]

            for config in default_configs:
                integration_config = IntegrationConfig(**config)
                self.integration_configs[integration_config.integration_id] = integration_config
                self.integration_metrics[integration_config.integration_id] = IntegrationMetrics()

            logger.info(f"加载了 {len(self.integration_configs)} 个集成配置")

        except Exception as e:
            logger.error(f"加载集成配置失败: {str(e)}")

    def _start_health_monitoring(self):
        """启动健康监控"""
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            return

        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_worker,
            daemon=True,
            name="HealthMonitor"
        )
        self.health_monitor_thread.start()
        self.running = True

    def _stop_health_monitoring(self):
        """停止健康监控"""
        self.running = False
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5)

    def _health_monitor_worker(self):
        """健康监控工作线程"""
        while self.running:
            try:
                # 检查架构层健康状态
                if self.layer_manager:
                    layer_health = self.layer_manager.check_all_layers_health()
                    for layer_name, is_healthy in layer_health.items():
                        if not is_healthy:
                            logger.warning(f"架构层 {layer_name} 健康状态异常")

                # 检查业务流程编排器健康状态
                if self.orchestrator:
                    orchestrator_health = self.orchestrator.health_check()
                    if not orchestrator_health:
                        logger.warning("业务流程编排器健康状态异常")

                # 更新集成指标
                self._update_integration_metrics()

                # 等待下次检查
                time.sleep(SECONDS_PER_MINUTE)  # 每分钟检查一次

            except Exception as e:
                logger.error(f"健康监控异常: {str(e)}")
                time.sleep(DEFAULT_TIMEOUT)  # 异常时30秒后重试

    def _update_integration_metrics(self):
        """更新集成指标"""
        try:
            current_time = time.time()

            for integration_id, metrics in self.integration_metrics.items():
                if self.integration_status == IntegrationStatus.INTEGRATED:
                    metrics.total_uptime += SECONDS_PER_MINUTE  # 假设每分钟更新一次
                else:
                    metrics.total_downtime += SECONDS_PER_MINUTE

                metrics.last_connection_time = current_time

        except Exception as e:
            logger.error(f"更新集成指标失败: {str(e)}")

    def get_integration_status(self) -> IntegrationStatus:
        """获取集成状态"""
        return self.integration_status

    def get_integration_metrics(self) -> Dict[str, IntegrationMetrics]:
        """获取集成指标"""
        return self.integration_metrics.copy()

    def get_layer_manager(self) -> Optional[LayerIntegrationManager]:
        """获取架构层集成管理器"""
        return self.layer_manager

    def get_process_manager(self) -> Optional[ProcessIntegrationManager]:
        """获取业务流程集成管理器"""
        return self.process_manager

    def get_orchestrator(self) -> Optional[BusinessProcessOrchestrator]:
        """获取业务流程编排器"""
        return self.orchestrator

    def start_trading_cycle(self, symbols: List[str], strategy_config: dict, process_id: str = None) -> str:
        """启动交易周期"""
        if not self.orchestrator:
            raise IntegrationException("业务流程编排器未初始化")

        return self.orchestrator.start_trading_cycle(symbols, strategy_config, process_id)

    def get_process_status(self, process_id: str) -> Optional[dict]:
        """获取流程状态"""
        if not self.orchestrator:
            return None

        return self.orchestrator.get_process(process_id)

    def get_all_processes(self) -> List[dict]:
        """获取所有流程"""
        if not self.orchestrator:
            return []

        return self.orchestrator.get_running_processes()

    def health_check(self) -> bool:
        """健康检查"""
        try:
            # 检查集成状态
            if self.integration_status != IntegrationStatus.INTEGRATED:
                return False

            # 检查业务流程编排器
            if self.orchestrator and not self.orchestrator.health_check():
                return False

            # 检查架构层
            if self.layer_manager:
                layer_health = self.layer_manager.check_all_layers_health()
                if not all(layer_health.values()):
                    return False

            return True

        except Exception as e:
            logger.error(f"健康检查失败: {str(e)}")
            return False

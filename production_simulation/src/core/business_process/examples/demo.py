#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
业务流程编排演示脚本 - 重构版
展示完整的业务流程驱动架构集成

重构说明：
- 将大类BusinessProcessDemo拆分为多个职责单一的组件
- 提高代码可维护性和可扩展性
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入相关模块 - 使用基础实现以确保兼容性
class BusinessProcessOrchestrator:
    """基础业务流程编排器实现"""
    def __init__(self, config_dir="config/processes", max_instances=10): pass
    def initialize(self): pass
    def shutdown(self): pass
    def start_trading_cycle(self, symbols=None, strategy_config=None):
        return f"demo_process_{hash(str(symbols) + str(strategy_config)) % 1000}"
    def pause_process(self, pid): pass
    def get_current_state(self): return type('State', (), {'value': 'running'})()
    def get_running_processes(self): return []

class EventBus:
    """基础事件总线实现"""
    def __init__(self, max_workers=10, enable_async=True, **kwargs): pass
    def initialize(self): return True
    def shutdown(self): pass
    def subscribe(self, event_type, handler, priority=0): pass
    def get_event_statistics(self): return {}

class EventPriority:
    """事件优先级枚举"""
    HIGH = 1
    NORMAL = 0

class ServiceContainer:
    """基础服务容器实现"""
    def __init__(self, name): pass

class SystemIntegrationManager:
    """基础系统集成管理器实现"""
    pass

class BusinessProcessIntegration:
    """基础业务流程集成实现"""
    def __init__(self, config_dir="config/integration", max_processes=5): pass
    def initialize(self): pass
    def shutdown(self): pass

class DependencyContainer:
    """基础依赖注入容器实现"""
    def __init__(self, enable_health_monitoring=True, **kwargs): pass
    def initialize(self): return True
    def shutdown(self): pass
    def get_component(self): return None

class EventType:
    """事件类型枚举"""
    PROCESS_STARTED = 'process_started'
    PROCESS_COMPLETED = 'process_completed'
    PROCESS_ERROR = 'process_error'
    DATA_COLLECTED = 'data_collected'
    FEATURES_EXTRACTED = 'features_extracted'
    MODEL_PREDICTION_READY = 'model_prediction_ready'
    STRATEGY_DECISION_READY = 'strategy_decision_ready'
    RISK_CHECK_COMPLETED = 'risk_check_completed'
    EXECUTION_COMPLETED = 'execution_completed'
    MONITORING_FEEDBACK = 'monitoring_feedback'


@dataclass
class DemoConfig:
    """演示配置"""
    symbols: List[str] = None
    strategy_name: str = "demo_strategy"
    max_processes: int = 5
    process_timeout: int = 300
    enable_monitoring: bool = True

    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]


class DemoMetrics:
    """演示指标管理"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置指标"""
        self.metrics = {
            'total_processes': 0,
            'completed_processes': 0,
            'failed_processes': 0,
            'active_processes': 0,
            'start_time': None,
            'end_time': None
        }

    def update_metric(self, key: str, value: Any):
        """更新指标"""
        self.metrics[key] = value

    def increment_metric(self, key: str, amount: int = 1):
        """增加指标值"""
        if key in self.metrics and isinstance(self.metrics[key], (int, float)):
            self.metrics[key] += amount

    def get_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        return self.metrics.copy()


class ComponentInitializer(ABC):
    """组件初始化器基类"""

    @abstractmethod
    def initialize(self) -> bool:
        """初始化组件"""
        pass

    @abstractmethod
    def get_component(self):
        """获取组件实例"""
        pass


class EventBusInitializer(ComponentInitializer):
    """事件总线初始化器"""

    def __init__(self):
        self.event_bus: Optional[EventBus] = None

    def initialize(self) -> bool:
        """初始化事件总线"""
        try:
            logger.info("📡 初始化事件总线...")
            self.event_bus = EventBus(
                max_workers=10,
                enable_async=True,
                enable_persistence=True,
                enable_retry=True,
                enable_monitoring=True
            )
            self.event_bus.initialize()
            return True
        except Exception as e:
            logger.error(f"❌ 事件总线初始化失败: {e}")
            return False

    def get_component(self):
        """获取事件总线实例"""
        return self.event_bus


class DependencyContainerInitializer(ComponentInitializer):
    """依赖注入容器初始化器"""

    def __init__(self):
        self.container: Optional[DependencyContainer] = None

    def initialize(self) -> bool:
        """初始化依赖注入容器"""
        try:
            logger.info("🔧 初始化依赖注入容器...")
            self.container = DependencyContainer(
                enable_health_monitoring=True,
                enable_service_discovery=True
            )
            self.container.initialize()
            return True
        except Exception as e:
            logger.error(f"❌ 依赖注入容器初始化失败: {e}")
            return False

    def get_component(self):
        """获取容器实例"""
        return self.container


class EventHandlerSetup:
    """事件处理器设置"""

    def __init__(self, event_bus: EventBus, orchestrator: BusinessProcessOrchestrator):
        self.event_bus = event_bus
        self.orchestrator = orchestrator

    def setup_process_event_handlers(self):
        """设置业务流程事件处理器"""
        # 数据收集事件处理器
        self.event_bus.add_event_handler(
            EventType.DATA_COLLECTION_STARTED,
            self._handle_data_collection_started
        )
        # 特征提取事件处理器
        self.event_bus.add_event_handler(
            EventType.FEATURE_EXTRACTION_COMPLETED,
            self._handle_feature_extraction_completed
        )
        # 模型预测事件处理器
        self.event_bus.add_event_handler(
            EventType.MODEL_PREDICTION_COMPLETED,
            self._handle_model_prediction_completed
        )
        # 策略决策事件处理器
        self.event_bus.add_event_handler(
            EventType.STRATEGY_DECISION_MADE,
            self._handle_strategy_decision_made
        )
        # 风控检查事件处理器
        self.event_bus.add_event_handler(
            EventType.RISK_CHECK_COMPLETED,
            self._handle_risk_check_completed
        )
        # 交易执行事件处理器
        self.event_bus.add_event_handler(
            EventType.TRADE_EXECUTION_COMPLETED,
            self._handle_trade_execution_completed
        )
        # 监控反馈事件处理器
        self.event_bus.add_event_handler(
            EventType.MONITORING_FEEDBACK_RECEIVED,
            self._handle_monitoring_feedback_received
        )

    def _handle_data_collection_started(self, event):
        """处理数据收集开始事件"""
        logger.info(f"📊 数据收集开始: {event.data}")

    def _handle_feature_extraction_completed(self, event):
        """处理特征提取完成事件"""
        logger.info(f"🔍 特征提取完成: {event.data}")

    def _handle_model_prediction_completed(self, event):
        """处理模型预测完成事件"""
        logger.info(f"🤖 模型预测完成: {event.data}")

    def _handle_strategy_decision_made(self, event):
        """处理策略决策事件"""
        logger.info(f"🎯 策略决策完成: {event.data}")

    def _handle_risk_check_completed(self, event):
        """处理风控检查完成事件"""
        logger.info(f"⚠️ 风控检查完成: {event.data}")

    def _handle_trade_execution_completed(self, event):
        """处理交易执行完成事件"""
        logger.info(f"💰 交易执行完成: {event.data}")

    def _handle_monitoring_feedback_received(self, event):
        """处理监控反馈事件"""
        logger.info(f"📈 监控反馈接收: {event.data}")


class TradingCycleManager:
    """交易周期管理器"""

    def __init__(self, config: DemoConfig, orchestrator: BusinessProcessOrchestrator,
                 event_bus: EventBus, metrics: DemoMetrics):
        self.config = config
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.metrics = metrics
        self.is_running = False

    def start_trading_cycle(self) -> bool:
        """启动交易周期"""
        if not self.orchestrator:
            logger.error("❌ 编排器未初始化，无法启动交易周期")
            return False

        event_bus = self.event_bus_initializer.get_component()
        if not event_bus:
            logger.error("❌ 事件总线未初始化，无法启动交易周期")
            return False

        # 初始化交易周期管理器
        self.trading_cycle_manager = TradingCycleManager(
            self.config, self.orchestrator, event_bus, self.metrics
        )

        # 启动交易周期
        success = self.trading_cycle_manager.start_trading_cycle()
        if success:
            self.is_running = True
        return success


class DemoInitializer:
    """演示初始化器 - 负责所有组件的初始化"""

    def __init__(self, config: DemoConfig):
        self.config = config
        self.event_bus_initializer = EventBusInitializer()
        self.container_initializer = DependencyContainerInitializer()
        self.orchestrator: Optional[BusinessProcessOrchestrator] = None
        self.service_container: Optional[ServiceContainer] = None
        self.integration: Optional[BusinessProcessIntegration] = None

    def initialize_all_components(self) -> bool:
        """初始化所有组件"""
        try:
            logger.info("🚀 初始化业务流程演示环境...")

            # 按顺序初始化各个组件
            if not self.event_bus_initializer.initialize():
                return False

            if not self.container_initializer.initialize():
                return False

            if not self._initialize_service_container():
                return False

            if not self._initialize_orchestrator():
                return False

            if not self._initialize_integration():
                return False

            logger.info("✅ 业务流程演示环境初始化完成")
            return True

        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            self._cleanup_components()
            return False

    def _initialize_service_container(self) -> bool:
        """初始化服务容器"""
        try:
            logger.info("🏗️ 初始化服务容器...")
            self.service_container = ServiceContainer("config / services")
            return True
        except Exception as e:
            logger.error(f"❌ 服务容器初始化失败: {e}")
            return False

    def _initialize_orchestrator(self) -> bool:
        """初始化业务流程编排器"""
        try:
            logger.info("🎭 初始化业务流程编排器...")
            self.orchestrator = BusinessProcessOrchestrator(
                config_dir="config / processes",
                max_instances=self.config.max_processes
            )
            self.orchestrator.initialize()
            return True
        except Exception as e:
            logger.error(f"❌ 业务流程编排器初始化失败: {e}")
            return False

    def _initialize_integration(self) -> bool:
        """初始化业务流程集成"""
        try:
            logger.info("🔗 初始化业务流程集成...")
            self.integration = BusinessProcessIntegration(
                config_dir="config / integration",
                max_processes=self.config.max_processes
            )
            self.integration.initialize()
            return True
        except Exception as e:
            logger.error(f"❌ 业务流程集成初始化失败: {e}")
            return False

    def _cleanup_components(self) -> None:
        """清理已初始化的组件"""
        try:
            event_bus = self.event_bus_initializer.get_component()
            if event_bus:
                event_bus.shutdown()

            container = self.container_initializer.get_component()
            if container:
                container.shutdown()

            if self.orchestrator:
                self.orchestrator.shutdown()

            if self.integration:
                self.integration.shutdown()
        except Exception as e:
            logger.warning(f"组件清理时出错: {e}")

    def get_components(self) -> Dict[str, Any]:
        """获取所有已初始化的组件"""
        return {
            'event_bus': self.event_bus_initializer.get_component(),
            'container': self.container_initializer.get_component(),
            'orchestrator': self.orchestrator,
            'service_container': self.service_container,
            'integration': self.integration
        }


class DemoMonitor:
    """演示监控器 - 负责状态监控和指标管理"""

    def __init__(self, metrics: DemoMetrics, orchestrator: BusinessProcessOrchestrator,
                 event_bus: EventBus):
        self.metrics = metrics
        self.orchestrator = orchestrator
        self.event_bus = event_bus

    def get_demo_status(self) -> Dict[str, Any]:
        """获取演示状态"""
        metrics = self.metrics.get_summary()
        status = {
            'is_running': True,  # 监控器本身表示运行中
            'active_processes': metrics.get('active_processes', 0),
            'total_processes': metrics.get('total_processes', 0),
            'completed_processes': metrics.get('completed_processes', 0),
            'failed_processes': metrics.get('failed_processes', 0),
            'process_list': []  # 需要从外部传入
        }

        if self.orchestrator:
            status['orchestrator_status'] = self.orchestrator.get_current_state().value
            status['running_processes'] = len(self.orchestrator.get_running_processes())

        if self.event_bus:
            status['event_bus_stats'] = self.event_bus.get_event_statistics()

        return status

    def get_demo_metrics(self) -> Dict[str, Any]:
        """获取演示指标"""
        metrics = self.metrics.get_summary()

        if metrics.get('start_time') and metrics.get('end_time'):
            metrics_copy = metrics.copy()
            metrics_copy['total_duration'] = metrics['end_time'] - metrics['start_time']
            return metrics_copy
        elif metrics.get('start_time'):
            metrics_copy = metrics.copy()
            metrics_copy['current_duration'] = time.time() - metrics['start_time']
            return metrics_copy

        if metrics['total_processes'] > 0:
            metrics['success_rate'] = (metrics['completed_processes'] /
                                       metrics['total_processes']) * 100
            metrics['failure_rate'] = (metrics['failed_processes'] /
                                       metrics['total_processes']) * 100

        return metrics


class DemoEventHandler:
    """演示事件处理器 - 负责所有事件处理逻辑"""

    def __init__(self, event_bus: EventBus, orchestrator: BusinessProcessOrchestrator,
                 metrics: DemoMetrics):
        self.event_bus = event_bus
        self.orchestrator = orchestrator
        self.metrics = metrics

    def setup_event_handlers(self) -> None:
        """设置所有事件处理器"""
        if not self.event_bus:
            return

        # 业务流程事件
        self.event_bus.subscribe(EventType.PROCESS_STARTED, self._on_process_started)
        self.event_bus.subscribe(EventType.PROCESS_COMPLETED, self._on_process_completed)
        self.event_bus.subscribe(EventType.PROCESS_ERROR, self._on_process_error)

        # 数据处理事件
        self.event_bus.subscribe(EventType.DATA_COLLECTED, self._on_data_collected)
        self.event_bus.subscribe(EventType.FEATURES_EXTRACTED, self._on_features_extracted)
        self.event_bus.subscribe(EventType.MODEL_PREDICTION_READY, self._on_model_prediction_ready)

        # 策略和决策事件
        self.event_bus.subscribe(EventType.STRATEGY_DECISION_READY, self._on_strategy_decision_ready)
        self.event_bus.subscribe(EventType.RISK_CHECK_COMPLETED, self._on_risk_check_completed)
        self.event_bus.subscribe(EventType.EXECUTION_COMPLETED, self._on_execution_completed)

        # 监控事件
        self.event_bus.subscribe(EventType.MONITORING_FEEDBACK, self._on_monitoring_feedback_received)

    # 事件处理器方法
    def _on_process_started(self, event):
        """进程启动事件处理"""
        logger.info(f"🚀 进程启动: {event.data.get('process_id', 'unknown')}")

    def _on_process_completed(self, event):
        """进程完成事件处理"""
        process_id = event.data.get('process_id', 'unknown')
        logger.info(f"✅ 进程完成: {process_id}")
        self.metrics.increment_metric('completed_processes')
        self.metrics.increment_metric('active_processes', -1)

    def _on_process_error(self, event):
        """进程错误事件处理"""
        process_id = event.data.get('process_id', 'unknown')
        error_msg = event.data.get('error_message', 'unknown error')
        logger.error(f"❌ 进程错误: {process_id} - {error_msg}")
        self.metrics.increment_metric('failed_processes')
        self.metrics.increment_metric('active_processes', -1)

    def _on_data_collected(self, event):
        """数据采集完成事件处理"""
        symbol = event.data.get('symbol', 'unknown')
        logger.info(f"📊 数据采集完成: {symbol}")

    def _on_features_extracted(self, event):
        """特征提取完成事件处理"""
        symbol = event.data.get('symbol', 'unknown')
        logger.info(f"🔍 特征提取完成: {symbol}")

    def _on_model_prediction_ready(self, event):
        """模型预测完成事件处理"""
        symbol = event.data.get('symbol', 'unknown')
        logger.info(f"🤖 模型预测完成: {symbol}")

    def _on_strategy_decision_ready(self, event):
        """策略决策完成事件处理"""
        symbol = event.data.get('symbol', 'unknown')
        logger.info(f"🎯 策略决策完成: {symbol}")

    def _on_risk_check_completed(self, event):
        """风险检查完成事件处理"""
        symbol = event.data.get('symbol', 'unknown')
        passed = event.data.get('passed', False)
        status = "通过" if passed else "拒绝"
        logger.info(f"🛡️ 风险检查完成: {symbol} - {status}")

    def _on_execution_completed(self, event):
        """交易执行完成事件处理"""
        symbol = event.data.get('symbol', 'unknown')
        logger.info(f"💹 交易执行完成: {symbol}")

    def _on_monitoring_feedback_received(self, event):
        """监控反馈事件处理"""
        logger.info(f"📈 监控反馈接收: {event.data}")


class DemoRunner:
    """演示运行器 - 负责演示的启动、运行和停止"""

    def __init__(self, config: DemoConfig, initializer: DemoInitializer,
                 event_handler: DemoEventHandler, metrics: DemoMetrics):
        self.config = config
        self.initializer = initializer
        self.event_handler = event_handler
        self.metrics = metrics
        self.demo_processes: List[str] = []
        self.is_running = False

    def start_demo(self) -> bool:
        """启动演示"""
        try:
            logger.info("🎬 启动业务流程演示...")

            # 获取组件
            components = self.initializer.get_components()
            orchestrator = components['orchestrator']

            if not orchestrator:
                logger.error("❌ 编排器未初始化，无法启动演示")
                return False

            # 设置事件处理器
            event_bus = components['event_bus']
            if event_bus:
                self.event_handler.setup_event_handlers()

            # 启动交易周期
            self.is_running = True
            self.metrics.update_metric('start_time', time.time())
            self.metrics.reset()

            # 为每个交易标的启动交易周期
            for symbol in self.config.symbols:
                process_id = self._start_trading_cycle(symbol, orchestrator)
                if process_id:
                    self.demo_processes.append(process_id)
                    self.metrics.increment_metric('total_processes')
                    self.metrics.increment_metric('active_processes')

            if not self.demo_processes:
                logger.error("❌ 没有成功启动任何交易周期")
                return False

            logger.info(f"✅ 演示启动成功，共启动 {len(self.demo_processes)} 个交易周期")
            return True

        except Exception as e:
            logger.error(f"❌ 启动演示失败: {e}")
            self.is_running = False
            return False

    def stop_demo(self) -> bool:
        """停止演示"""
        try:
            logger.info("🛑 停止业务流程演示...")
            self.is_running = False
            self.metrics.update_metric('end_time', time.time())

            # 获取组件
            components = self.initializer.get_components()
            orchestrator = components['orchestrator']
            integration = components['integration']
            event_bus = components['event_bus']
            container = components['container']

            # 停止所有进程
            if orchestrator:
                for process_id in self.demo_processes:
                    try:
                        orchestrator.pause_process(process_id)
                        logger.info(f"⏸️ 暂停进程: {process_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ 暂停进程失败 {process_id}: {e}")

            # 关闭服务
            if integration:
                integration.shutdown()

            if orchestrator:
                orchestrator.shutdown()

            # 关闭事件总线和容器
            if event_bus:
                event_bus.shutdown()

            if container:
                container.shutdown()

            logger.info("✅ 演示停止完成")
            return True

        except Exception as e:
            logger.error(f"❌ 停止演示失败: {e}")
            return False

    def _start_trading_cycle(self, symbol: str, orchestrator: BusinessProcessOrchestrator) -> Optional[str]:
        """启动单个交易周期"""
        try:
            strategy_config = {
                "name": f"{self.config.strategy_name}_{symbol}",
                "symbol": symbol,
                "parameters": {
                    "lookback_period": 20,
                    "threshold": 0.02,
                    "max_position": 1000
                }
            }

            process_id = orchestrator.start_trading_cycle(
                symbols=[symbol],
                strategy_config=strategy_config
            )

            return process_id

        except Exception as e:
            logger.error(f"❌ 启动交易周期失败 {symbol}: {e}")
            return None


class BusinessProcessDemo:
    """
    业务流程演示类 - 进一步重构版

    使用组合模式，将不同职责完全分离到专用组件中：
    - DemoInitializer: 初始化管理
    - DemoRunner: 运行控制
    - DemoMonitor: 状态监控
    - DemoEventHandler: 事件处理
    - DemoMetrics: 指标管理
    """

    def __init__(self, config: DemoConfig):
        self.config = config

        # 初始化专用组件
        self.metrics = DemoMetrics()
        self.initializer = DemoInitializer(config)

        # 其他组件将在初始化后创建
        self.event_handler: Optional[DemoEventHandler] = None
        self.runner: Optional[DemoRunner] = None
        self.monitor: Optional[DemoMonitor] = None

    def initialize(self) -> bool:
        """初始化演示环境 - 进一步重构版"""
        try:
            # 初始化所有组件
            if not self.initializer.initialize_all_components():
                return False

            # 获取组件引用
            components = self.initializer.get_components()

            # 创建专用组件
            self.event_handler = DemoEventHandler(
                components['event_bus'],
                components['orchestrator'],
                self.metrics
            )

            self.runner = DemoRunner(
                self.config,
                self.initializer,
                self.event_handler,
                self.metrics
            )

            self.monitor = DemoMonitor(
                self.metrics,
                components['orchestrator'],
                components['event_bus']
            )

            logger.info("✅ 业务流程演示环境初始化完成")
            return True

        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            return False

    def start_demo(self) -> bool:
        """启动演示 - 委派给DemoRunner"""
        if not self.runner:
            logger.error("❌ Runner未初始化")
            return False
        return self.runner.start_demo()

    def stop_demo(self) -> bool:
        """停止演示 - 委派给DemoRunner"""
        if not self.runner:
            logger.error("❌ Runner未初始化")
            return False
        return self.runner.stop_demo()

    def get_demo_status(self) -> Dict[str, Any]:
        """获取演示状态 - 委派给DemoMonitor"""
        if not self.monitor:
            return {'error': 'Monitor未初始化'}
        status = self.monitor.get_demo_status()
        status['process_list'] = self.runner.demo_processes.copy() if self.runner else []
        return status

    def get_demo_metrics(self) -> Dict[str, Any]:
        """获取演示指标 - 委派给DemoMonitor"""
        if not self.monitor:
            return {'error': 'Monitor未初始化'}
        return self.monitor.get_demo_metrics()


def create_demo_config() -> DemoConfig:
    """创建演示配置"""
    return DemoConfig(
        symbols=["AAPL", "GOOGL", "MSFT"],
        strategy_name="demo_strategy",
        max_processes=3,
        process_timeout=180,
        enable_monitoring=True
    )


def create_demo_instance(config: DemoConfig) -> BusinessProcessDemo:
    """创建演示实例"""
    return BusinessProcessDemo(config)


def initialize_demo(demo: BusinessProcessDemo) -> bool:
    """初始化演示环境"""
    if not demo.initialize():
        logger.error("❌ 演示环境初始化失败")
        return False
    return True


def start_demo_execution(demo: BusinessProcessDemo) -> bool:
    """启动演示执行"""
    if not demo.start_demo():
        logger.error("❌ 演示启动失败")
        return False
    return True


def monitor_demo_status(demo: BusinessProcessDemo) -> None:
    """监控演示状态"""
    try:
        while demo.is_running and demo.demo_metrics['active_processes'] > 0:
            status = demo.get_demo_status()
            metrics = demo.get_demo_metrics()

            logger.info(f"📊 演示状态: 活跃进程 {status['active_processes']}, "
                        f"完成 {metrics['completed_processes']}, "
                        f"失败 {metrics['failed_processes']}")

            time.sleep(10)  # 每10秒检查一次状态

    except KeyboardInterrupt:
        logger.info("⏹️ 收到中断信号，正在停止演示...")


def stop_demo_execution(demo: BusinessProcessDemo) -> None:
    """停止演示执行"""
    demo.stop_demo()


def report_demo_results(demo: BusinessProcessDemo) -> None:
    """报告演示结果"""
    final_metrics = demo.get_demo_metrics()
    logger.info("🎉 演示完成！")
    logger.info(f"📈 总进程数: {final_metrics['total_processes']}")
    logger.info(f"✅ 成功完成: {final_metrics['completed_processes']}")
    logger.info(f"❌ 失败数量: {final_metrics['failed_processes']}")
    if 'success_rate' in final_metrics:
        logger.info(f"📊 成功率: {final_metrics['success_rate']:.1f}%")


def run_business_process_demo():
    """运行业务流程演示 - 重构版"""
    try:
        # 1. 创建演示配置
        config = create_demo_config()

        # 2. 创建演示实例
        demo = create_demo_instance(config)

        # 3. 初始化演示环境
        if not initialize_demo(demo):
            return False

        # 4. 启动演示
        if not start_demo_execution(demo):
            return False

        # 5. 监控演示状态
        monitor_demo_status(demo)

        # 6. 停止演示
        stop_demo_execution(demo)

        # 7. 输出最终结果
        report_demo_results(demo)

        return True

    except Exception as e:
        logger.error(f"❌ 演示运行失败: {e}")
        return False


if __name__ == "__main__":
    run_business_process_demo()

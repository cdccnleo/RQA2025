#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
业务流程编排演示脚本
展示完整的业务流程驱动架构集成
"""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from ..orchestration.business_process_orchestrator import (
    BusinessProcessOrchestrator,
    EventType
)
from .event_bus import EventBus, EventPriority
from .container import DependencyContainer
from .service_container import ServiceContainer
from .business_process_integration import BusinessProcessIntegration

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


class BusinessProcessDemo:

    """业务流程演示类"""

    def __init__(self, config: DemoConfig):

        self.config = config
        self.orchestrator: Optional[BusinessProcessOrchestrator] = None
        self.event_bus: Optional[EventBus] = None
        self.container: Optional[DependencyContainer] = None
        self.service_container: Optional[ServiceContainer] = None
        self.integration: Optional[BusinessProcessIntegration] = None

        # 演示状态
        self.is_running = False
        self.demo_processes: List[str] = []
        self.demo_metrics = {
            'total_processes': 0,
            'completed_processes': 0,
            'failed_processes': 0,
            'active_processes': 0,
            'start_time': None,
            'end_time': None
        }

    def initialize(self) -> bool:
        """初始化演示环境"""
        try:
            logger.info("🚀 初始化业务流程演示环境...")

            # 1. 初始化事件总线
            logger.info("📡 初始化事件总线...")
            self.event_bus = EventBus(
                max_workers=10,
                enable_async=True,
                enable_persistence=True,
                enable_retry=True,
                enable_monitoring=True
            )
            self.event_bus.initialize()

            # 2. 初始化依赖注入容器
            logger.info("🔧 初始化依赖注入容器...")
            self.container = DependencyContainer(
                enable_health_monitoring=True,
                enable_service_discovery=True
            )
            self.container.initialize()

            # 3. 初始化服务容器
            logger.info("🏗️ 初始化服务容器...")
            self.service_container = ServiceContainer("config / services")

            # 4. 初始化业务流程编排器
            logger.info("🎭 初始化业务流程编排器...")
            self.orchestrator = BusinessProcessOrchestrator(
                config_dir="config / processes",
                max_instances=self.config.max_processes
            )
            self.orchestrator.initialize()

            # 5. 初始化业务流程集成
            logger.info("🔗 初始化业务流程集成...")
            self.integration = BusinessProcessIntegration(
                config_dir="config / integration",
                max_processes=self.config.max_processes
            )
            self.integration.initialize()

            # 6. 设置事件处理器
            self._setup_event_handlers()

            logger.info("✅ 业务流程演示环境初始化完成")
            return True

        except Exception as e:
            logger.error(f"❌ 初始化失败: {e}")
            # 清理已初始化的组件
            self.event_bus = None
            self.container = None
            self.service_container = None
            self.orchestrator = None
            self.integration = None
            return False

    def _setup_event_handlers(self):
        """设置事件处理器"""
        if not self.event_bus:
            return

        # 监听业务流程事件
        self.event_bus.subscribe(
            EventType.PROCESS_STARTED,
            self._on_process_started,
            priority=EventPriority.HIGH
        )

        self.event_bus.subscribe(
            EventType.PROCESS_COMPLETED,
            self._on_process_completed,
            priority=EventPriority.HIGH
        )

        self.event_bus.subscribe(
            EventType.PROCESS_ERROR,
            self._on_process_error,
            priority=EventPriority.HIGH
        )

        # 监听业务层事件
        self.event_bus.subscribe(
            EventType.DATA_COLLECTED,
            self._on_data_collected,
            priority=EventPriority.NORMAL
        )

        self.event_bus.subscribe(
            EventType.FEATURES_EXTRACTED,
            self._on_features_extracted,
            priority=EventPriority.NORMAL
        )

        self.event_bus.subscribe(
            EventType.MODEL_PREDICTION_READY,
            self._on_model_prediction_ready,
            priority=EventPriority.NORMAL
        )

        self.event_bus.subscribe(
            EventType.STRATEGY_DECISION_READY,
            self._on_strategy_decision_ready,
            priority=EventPriority.NORMAL
        )

        self.event_bus.subscribe(
            EventType.RISK_CHECK_COMPLETED,
            self._on_risk_check_completed,
            priority=EventPriority.NORMAL
        )

        self.event_bus.subscribe(
            EventType.EXECUTION_COMPLETED,
            self._on_execution_completed,
            priority=EventPriority.NORMAL
        )

        logger.info("📡 事件处理器设置完成")

    def start_demo(self) -> bool:
        """开始演示"""
        try:
            if not self.orchestrator:
                logger.error("❌ 编排器未初始化")
                return False

            logger.info("🎬 开始业务流程演示...")
            self.is_running = True
            self.demo_metrics['start_time'] = time.time()

            # 为每个股票启动一个交易周期
            for symbol in self.config.symbols:
                if len(self.demo_processes) >= self.config.max_processes:
                    logger.warning(f"⚠️ 达到最大进程数限制: {self.config.max_processes}")
                    break

                process_id = self._start_trading_cycle(symbol)
                if process_id:
                    self.demo_processes.append(process_id)
                    self.demo_metrics['total_processes'] += 1
                    self.demo_metrics['active_processes'] += 1

                    logger.info(f"📈 启动交易周期: {symbol} -> {process_id}")

                    # 添加延迟，避免同时启动过多进程
                    time.sleep(1)

            logger.info(f"✅ 演示启动完成，共启动 {len(self.demo_processes)} 个进程")
            return True

        except Exception as e:
            logger.error(f"❌ 启动演示失败: {e}")
            return False

    def _start_trading_cycle(self, symbol: str) -> Optional[str]:
        """启动交易周期"""
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

            process_id = self.orchestrator.start_trading_cycle(
                symbols=[symbol],
                strategy_config=strategy_config
            )

            return process_id

        except Exception as e:
            logger.error(f"❌ 启动交易周期失败 {symbol}: {e}")
            return None

    def stop_demo(self) -> bool:
        """停止演示"""
        try:
            logger.info("🛑 停止业务流程演示...")
            self.is_running = False
            self.demo_metrics['end_time'] = time.time()

            # 停止所有进程
            if self.orchestrator:
                for process_id in self.demo_processes:
                    try:
                        self.orchestrator.pause_process(process_id)
                        logger.info(f"⏸️ 暂停进程: {process_id}")
                    except Exception as e:
                        logger.warning(f"⚠️ 暂停进程失败 {process_id}: {e}")

            # 关闭服务
            if self.integration:
                self.integration.shutdown()

            if self.orchestrator:
                self.orchestrator.shutdown()

            if self.event_bus:
                self.event_bus.shutdown()

            if self.container:
                self.container.shutdown()

            logger.info("✅ 演示停止完成")
            return True

        except Exception as e:
            logger.error(f"❌ 停止演示失败: {e}")
            return False

    def get_demo_status(self) -> Dict[str, Any]:
        """获取演示状态"""
        status = {
            'is_running': self.is_running,
            'active_processes': self.demo_metrics['active_processes'],
            'total_processes': self.demo_metrics['total_processes'],
            'completed_processes': self.demo_metrics['completed_processes'],
            'failed_processes': self.demo_metrics['failed_processes'],
            'process_list': self.demo_processes.copy()
        }

        if self.orchestrator:
            status['orchestrator_status'] = self.orchestrator.get_current_state().value
            status['running_processes'] = len(self.orchestrator.get_running_processes())

        if self.event_bus:
            status['event_bus_stats'] = self.event_bus.get_event_statistics()

        return status

    def get_demo_metrics(self) -> Dict[str, Any]:
        """获取演示指标"""
        metrics = self.demo_metrics.copy()

        if metrics['start_time'] and metrics['end_time']:
            metrics['total_duration'] = metrics['end_time'] - metrics['start_time']
        elif metrics['start_time']:
            metrics['current_duration'] = time.time() - metrics['start_time']

        if metrics['total_processes'] > 0:
            metrics['success_rate'] = (metrics['completed_processes'] /
                                       metrics['total_processes']) * 100
            metrics['failure_rate'] = (metrics['failed_processes'] /
                                       metrics['total_processes']) * 100

        return metrics

    # 事件处理器

    def _on_process_started(self, event):
        """进程启动事件处理"""
        logger.info(f"🚀 进程启动: {event.data.get('process_id', 'unknown')}")

    def _on_process_completed(self, event):
        """进程完成事件处理"""
        process_id = event.data.get('process_id', 'unknown')
        logger.info(f"✅ 进程完成: {process_id}")
        self.demo_metrics['completed_processes'] += 1
        self.demo_metrics['active_processes'] -= 1

        if process_id in self.demo_processes:
            self.demo_processes.remove(process_id)

    def _on_process_error(self, event):
        """进程错误事件处理"""
        process_id = event.data.get('process_id', 'unknown')
        error_msg = event.data.get('error_message', 'unknown error')
        logger.error(f"❌ 进程错误: {process_id} - {error_msg}")
        self.demo_metrics['failed_processes'] += 1
        self.demo_metrics['active_processes'] -= 1

        if process_id in self.demo_processes:
            self.demo_processes.remove(process_id)

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


def run_business_process_demo():
    """运行业务流程演示"""
    try:
        # 创建演示配置
        config = DemoConfig(
            symbols=["AAPL", "GOOGL", "MSFT"],
            strategy_name="demo_strategy",
            max_processes=3,
            process_timeout=180,
            enable_monitoring=True
        )

        # 创建演示实例
        demo = BusinessProcessDemo(config)

        # 初始化演示环境
        if not demo.initialize():
            logger.error("❌ 演示环境初始化失败")
            return False

        # 启动演示
        if not demo.start_demo():
            logger.error("❌ 演示启动失败")
            return False

        # 监控演示状态
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

        # 停止演示
        demo.stop_demo()

        # 输出最终结果
        final_metrics = demo.get_demo_metrics()
        logger.info("🎉 演示完成！")
        logger.info(f"📈 总进程数: {final_metrics['total_processes']}")
        logger.info(f"✅ 成功完成: {final_metrics['completed_processes']}")
        logger.info(f"❌ 失败数量: {final_metrics['failed_processes']}")
        if 'success_rate' in final_metrics:
            logger.info(f"📊 成功率: {final_metrics['success_rate']:.1f}%")

        return True

    except Exception as e:
        logger.error(f"❌ 演示运行失败: {e}")
        return False


if __name__ == "__main__":
    run_business_process_demo()

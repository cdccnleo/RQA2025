#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
业务流程编排器实际业务集成示例
展示如何将业务流程编排器与现有的业务层集成，实现完整的量化交易流程
"""

import asyncio
import time
from typing import Dict, List, Any
import logging

# 导入业务流程编排器
from src.core.business_process_orchestrator import (
    BusinessProcessOrchestrator,
    BusinessProcessState,
    EventType
)

# 导入业务层组件
from src.data.data_manager import DataManager
from src.features.feature_manager import FeatureManager
from src.models.model_manager import ModelManager
from src.trading.risk.china.risk_controller import ChinaRiskController
from src.trading.execution.execution_engine import ExecutionEngine

# 导入事件总线
from src.core.event_bus import EventBus

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BusinessProcessIntegrationDemo:
    """业务流程集成演示类"""

    def __init__(self):
        """初始化演示类"""
        self.orchestrator = None
        self.event_bus = None
        self.data_manager = None
        self.feature_manager = None
        self.model_manager = None
        self.risk_controller = None
        self.execution_engine = None

    async def initialize(self):
        """初始化所有组件"""
        logger.info("开始初始化业务流程集成演示...")

        try:
            # 1. 初始化事件总线
            self.event_bus = EventBus()
            logger.info("✅ 事件总线初始化完成")

            # 2. 初始化业务流程编排器
            self.orchestrator = BusinessProcessOrchestrator(
                config_dir="config/processes",
                max_instances=50
            )

            # 设置事件总线
            self.orchestrator._event_bus = self.event_bus

            # 初始化编排器
            success = self.orchestrator.initialize()
            if not success:
                raise RuntimeError("业务流程编排器初始化失败")
            logger.info("✅ 业务流程编排器初始化完成")

            # 3. 初始化业务层组件
            await self._initialize_business_layers()

            # 4. 设置事件处理器
            self._setup_event_handlers()

            logger.info("✅ 所有组件初始化完成")
            return True

        except Exception as e:
            logger.error(f"初始化失败: {e}")
            return False

    async def _initialize_business_layers(self):
        """初始化业务层组件"""
        try:
            # 数据层
            self.data_manager = DataManager()
            logger.info("✅ 数据管理器初始化完成")

            # 特征层
            self.feature_manager = FeatureManager()
            logger.info("✅ 特征管理器初始化完成")

            # 模型层
            self.model_manager = ModelManager()
            logger.info("✅ 模型管理器初始化完成")

            # 风控层
            self.risk_controller = ChinaRiskController({})
            logger.info("✅ 风控控制器初始化完成")

            # 交易执行层
            self.execution_engine = ExecutionEngine()
            logger.info("✅ 执行引擎初始化完成")

        except Exception as e:
            logger.error(f"业务层初始化失败: {e}")
            raise

    def _setup_event_handlers(self):
        """设置事件处理器"""
        # 订阅业务流程编排器的事件
        self.event_bus.subscribe(EventType.DATA_COLLECTED, self._handle_data_collected)
        self.event_bus.subscribe(EventType.FEATURES_EXTRACTED, self._handle_features_extracted)
        self.event_bus.subscribe(EventType.MODEL_PREDICTION_READY,
                                 self._handle_model_prediction_ready)
        self.event_bus.subscribe(EventType.SIGNALS_GENERATED, self._handle_signals_generated)
        self.event_bus.subscribe(EventType.RISK_CHECK_COMPLETED, self._handle_risk_check_completed)
        self.event_bus.subscribe(EventType.ORDERS_GENERATED, self._handle_orders_generated)
        self.event_bus.subscribe(EventType.EXECUTION_COMPLETED, self._handle_execution_completed)

        logger.info("✅ 事件处理器设置完成")

    async def run_trading_cycle(self, symbols: List[str], strategy_config: Dict[str, Any]):
        """运行完整的交易周期"""
        logger.info(f"开始运行交易周期，标的: {symbols}")

        try:
            # 1. 启动交易周期流程
            process_id = self.orchestrator.start_trading_cycle(
                symbols=symbols,
                strategy_config=strategy_config
            )

            logger.info(f"交易周期流程已启动，流程ID: {process_id}")

            # 2. 等待流程完成
            await self._wait_for_process_completion(process_id)

            # 3. 获取流程结果
            result = self._get_process_result(process_id)

            logger.info(f"交易周期完成，结果: {result}")
            return result

        except Exception as e:
            logger.error(f"交易周期运行失败: {e}")
            raise

    async def _wait_for_process_completion(self, process_id: str, timeout: int = 300):
        """等待流程完成"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            # 检查流程状态
            process = self.orchestrator._process_monitor.get_process(process_id)
            if not process:
                logger.warning(f"流程 {process_id} 不存在")
                return

            if process.status == BusinessProcessState.COMPLETED:
                logger.info(f"流程 {process_id} 已完成")
                return
            elif process.status == BusinessProcessState.ERROR:
                logger.error(f"流程 {process_id} 出错: {process.error_message}")
                return

            # 等待1秒后再次检查
            await asyncio.sleep(1)

        logger.warning(f"流程 {process_id} 超时")

    def _get_process_result(self, process_id: str) -> Dict[str, Any]:
        """获取流程结果"""
        process = self.orchestrator._process_monitor.get_process(process_id)
        if not process:
            return {"status": "error", "message": "流程不存在"}

        return {
            "status": process.status.value,
            "progress": process.progress,
            "current_step": process.current_step,
            "error_message": process.error_message,
            "context": process.context
        }

    # 事件处理器方法
    async def _handle_data_collected(self, event: Dict[str, Any]):
        """处理数据收集完成事件"""
        logger.info(f"收到数据收集完成事件: {event}")

        try:
            # 模拟数据收集完成后的处理
            symbols = event.get('data', {}).get('symbols', [])

            # 发布特征提取开始事件
            self.event_bus.publish(EventType.FEATURE_EXTRACTION_STARTED, {
                'symbols': symbols,
                'timestamp': time.time()
            })

            # 模拟特征提取
            await asyncio.sleep(2)

            # 发布特征提取完成事件
            self.event_bus.publish(EventType.FEATURES_EXTRACTED, {
                'symbols': symbols,
                'features': {'technical': True, 'fundamental': True},
                'timestamp': time.time()
            })

        except Exception as e:
            logger.error(f"处理数据收集完成事件失败: {e}")

    async def _handle_features_extracted(self, event: Dict[str, Any]):
        """处理特征提取完成事件"""
        logger.info(f"收到特征提取完成事件: {event}")

        try:
            # 发布模型预测开始事件
            self.event_bus.publish(EventType.MODEL_PREDICTION_STARTED, {
                'timestamp': time.time()
            })

            # 模拟模型预测
            await asyncio.sleep(3)

            # 发布模型预测完成事件
            self.event_bus.publish(EventType.MODEL_PREDICTION_READY, {
                'predictions': {'signal': 1, 'confidence': 0.8},
                'timestamp': time.time()
            })

        except Exception as e:
            logger.error(f"处理特征提取完成事件失败: {e}")

    async def _handle_model_prediction_ready(self, event: Dict[str, Any]):
        """处理模型预测完成事件"""
        logger.info(f"收到模型预测完成事件: {event}")

        try:
            # 发布策略决策开始事件
            self.event_bus.publish(EventType.STRATEGY_DECISION_STARTED, {
                'timestamp': time.time()
            })

            # 模拟策略决策
            await asyncio.sleep(1)

            # 发布信号生成完成事件
            self.event_bus.publish(EventType.SIGNALS_GENERATED, {
                'signals': {'action': 'buy', 'quantity': 1000},
                'timestamp': time.time()
            })

        except Exception as e:
            logger.error(f"处理模型预测完成事件失败: {e}")

    async def _handle_signals_generated(self, event: Dict[str, Any]):
        """处理信号生成完成事件"""
        logger.info(f"收到信号生成完成事件: {event}")

        try:
            # 发布风控检查开始事件
            self.event_bus.publish(EventType.RISK_CHECK_STARTED, {
                'timestamp': time.time()
            })

            # 模拟风控检查
            await asyncio.sleep(1)

            # 发布风控检查完成事件
            self.event_bus.publish(EventType.RISK_CHECK_COMPLETED, {
                'passed': True,
                'risk_score': 0.2,
                'timestamp': time.time()
            })

        except Exception as e:
            logger.error(f"处理信号生成完成事件失败: {e}")

    async def _handle_risk_check_completed(self, event: Dict[str, Any]):
        """处理风控检查完成事件"""
        logger.info(f"收到风控检查完成事件: {event}")

        if event.get('data', {}).get('passed', False):
            try:
                # 发布订单生成开始事件
                self.event_bus.publish(EventType.ORDER_GENERATION_STARTED, {
                    'timestamp': time.time()
                })

                # 模拟订单生成
                await asyncio.sleep(1)

                # 发布订单生成完成事件
                self.event_bus.publish(EventType.ORDERS_GENERATED, {
                    'orders': [{'symbol': '000001.SZ', 'action': 'buy', 'quantity': 1000}],
                    'timestamp': time.time()
                })

            except Exception as e:
                logger.error(f"处理风控检查完成事件失败: {e}")
        else:
            logger.warning("风控检查未通过，跳过订单生成")

    async def _handle_orders_generated(self, event: Dict[str, Any]):
        """处理订单生成完成事件"""
        logger.info(f"收到订单生成完成事件: {event}")

        try:
            # 发布执行开始事件
            self.event_bus.publish(EventType.EXECUTION_STARTED, {
                'timestamp': time.time()
            })

            # 模拟订单执行
            await asyncio.sleep(2)

            # 发布执行完成事件
            self.event_bus.publish(EventType.EXECUTION_COMPLETED, {
                'executed_orders': [{'order_id': '001', 'status': 'filled'}],
                'timestamp': time.time()
            })

        except Exception as e:
            logger.error(f"处理订单生成完成事件失败: {e}")

    async def _handle_execution_completed(self, event: Dict[str, Any]):
        """处理执行完成事件"""
        logger.info(f"收到执行完成事件: {event}")

        try:
            # 发布交易周期完成事件
            self.event_bus.publish(EventType.TRADING_CYCLE_COMPLETED, {
                'summary': '交易周期执行完成',
                'timestamp': time.time()
            })

        except Exception as e:
            logger.error(f"处理执行完成事件失败: {e}")

    async def run_demo(self):
        """运行演示"""
        logger.info("开始运行业务流程集成演示...")

        try:
            # 初始化
            success = await self.initialize()
            if not success:
                logger.error("初始化失败")
                return

            # 配置交易策略
            strategy_config = {
                'strategy_name': 'ML_Strategy',
                'parameters': {
                    'lookback_period': 20,
                    'prediction_horizon': 5,
                    'risk_tolerance': 0.1
                }
            }

            # 运行交易周期
            symbols = ['000001.SZ', '600000.SH']
            result = await self.run_trading_cycle(symbols, strategy_config)

            logger.info(f"演示完成，结果: {result}")

        except Exception as e:
            logger.error(f"演示运行失败: {e}")
        finally:
            # 清理资源
            if self.orchestrator:
                self.orchestrator.shutdown()


async def main():
    """主函数"""
    demo = BusinessProcessIntegrationDemo()
    await demo.run_demo()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())

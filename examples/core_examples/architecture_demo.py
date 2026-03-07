from .orchestration.business_process_orchestrator import BusinessProcessOrchestrator, BusinessProcessState
import json
from typing import List
import logging
import time
CONST_60 = 60

CONST_60 = 60

CONST_05 = 5

CONST_30 = 30

CONST_60 = 60

CONST_60 = 60

CONST_30 = 30

CONST_05 = 5

CONST_60 = 60


CONST_60 = 60

# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
业务流程驱动架构演示程序
展示完整的业务流程及依赖关系
"""


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ArchitectureDemo:

    """架构演示类"""

    def __init__(self):

        self.orchestrator = BusinessProcessOrchestrator()
        self.demo_results = {}

    def run_complete_trading_cycle(self, symbols: List[str], strategy_config: dict = None):
        """运行完整的交易周期"""
        logger.info("=" * 60)
        logger.info("开始业务流程驱动架构演示")
        logger.info("=" * 60)

        # 设置默认策略配置
        if strategy_config is None:
            strategy_config = {
                'strategy_name': 'ml_momentum',
                'risk_tolerance': 'medium',
                'position_size': 0.1,
                'max_drawdown': 0.05
            }

        # 启动交易周期
        logger.info(f"启动交易周期，股票代码: {symbols}")
        logger.info(f"策略配置: {strategy_config}")

        start_time = time.time()

        try:
            # 启动交易周期
            self.orchestrator.start_trading_cycle(symbols, strategy_config)

            # 等待交易周期完成
            self._wait_for_completion()

            # 收集结果
            self._collect_results(start_time)

            # 显示结果
            self._display_results()

        except Exception as e:
            logger.error(f"交易周期执行失败: {e}")
            self._handle_error(e)

    def _wait_for_completion(self, timeout: int = 30):
        """等待交易周期完成"""
        logger.info("等待交易周期完成...")

        start_time = time.time()
        while True:
            current_state = self.orchestrator.get_current_state()

            if current_state == BusinessProcessState.COMPLETED:
                logger.info("交易周期已完成")
                break
            elif current_state == BusinessProcessState.ERROR:
                logger.error("交易周期执行出错")
                break
            elif time.time() - start_time > timeout:
                logger.warning("交易周期超时")
                break

            time.sleep(0.1)

    def _collect_results(self, start_time: float):
        """收集执行结果"""
        end_time = time.time()
        execution_time = end_time - start_time

        # 获取状态历史
        state_history = self.orchestrator.get_state_history()

        # 获取事件历史
        event_history = self.orchestrator.get_event_history()

        # 获取当前状态
        current_state = self.orchestrator.get_current_state()

        # 收集各层状态
        layer_status = {
            'core_services': self.orchestrator.core_services.get_status(),
            'infrastructure': self.orchestrator.infrastructure.get_status(),
            'data_management': self.orchestrator.data_management.get_status(),
            'feature_processing': self.orchestrator.feature_processing.get_status(),
            'model_inference': self.orchestrator.model_inference.get_status(),
            'strategy_decision': self.orchestrator.strategy_decision.get_status(),
            'risk_compliance': self.orchestrator.risk_compliance.get_status(),
            'trading_execution': self.orchestrator.trading_execution.get_status(),
            'monitoring_feedback': self.orchestrator.monitoring_feedback.get_status()
        }

        self.demo_results = {
            'execution_time': execution_time,
            'current_state': current_state.value,
            'state_history': state_history,
            'event_history': event_history,
            'layer_status': layer_status,
            'success': current_state == BusinessProcessState.COMPLETED
        }

    def _display_results(self):
        """显示执行结果"""
        logger.info("=" * 60)
        logger.info("业务流程驱动架构演示结果")
        logger.info("=" * 60)

        results = self.demo_results

        # 基本信息
        logger.info(f"执行时间: {results['execution_time']:.2f} 秒")
        logger.info(f"最终状态: {results['current_state']}")
        logger.info(f"执行成功: {results['success']}")

        # 状态转换历史
        logger.info("\n状态转换历史:")
        for i, transition in enumerate(results['state_history']):
            from_state = transition['from_state'].value
            to_state = transition['to_state'].value
            timestamp = transition['timestamp']
            logger.info(f"  {i + 1}. {from_state} -> {to_state} (时间: {timestamp:.2f})")

        # 事件历史
        logger.info(f"\n事件历史 (共 {len(results['event_history'])} 个事件):")
        for i, event in enumerate(results['event_history'][-10:]):  # 只显示最后10个事件
            event_type = event['type'].value
            timestamp = event['timestamp']
            logger.info(f"  {i + 1}. {event_type} (时间: {timestamp:.2f})")

        # 各层状态
        logger.info("\n各层状态:")
        for layer_name, status in results['layer_status'].items():
            health = status.get('health', 'unknown')
            logger.info(f"  {layer_name}: {health}")

        # 性能指标
        if results['success']:
            logger.info("\n性能指标:")
            logger.info(f"  总执行时间: {results['execution_time']:.2f} 秒")
            logger.info(f"  状态转换次数: {len(results['state_history'])}")
            logger.info(f"  事件数量: {len(results['event_history'])}")

    def _handle_error(self, error: Exception):
        """处理错误"""
        logger.error(f"演示执行失败: {error}")

        # 获取错误状态
        current_state = self.orchestrator.get_current_state()
        state_history = self.orchestrator.get_state_history()

        logger.info(f"错误时的状态: {current_state.value}")
        logger.info(f"状态转换历史: {len(state_history)} 次")

    def run_performance_test(self, symbols: List[str], iterations: int = 5):
        """运行性能测试"""
        logger.info("=" * 60)
        logger.info("开始性能测试")
        logger.info("=" * 60)

        results = []

        for i in range(iterations):
            logger.info(f"执行第 {i + 1}/{iterations} 次测试")

            # 重置编排器
            self.orchestrator.reset()

            # 运行交易周期
            start_time = time.time()
            self.orchestrator.start_trading_cycle(symbols, {})

            # 等待完成
            self._wait_for_completion()

            end_time = time.time()
            execution_time = end_time - start_time

            current_state = self.orchestrator.get_current_state()
            success = current_state == BusinessProcessState.COMPLETED

            results.append({
                'iteration': i + 1,
                'execution_time': execution_time,
                'success': success,
                'state_count': len(self.orchestrator.get_state_history()),
                'event_count': len(self.orchestrator.get_event_history())
            })

            logger.info(f"  执行时间: {execution_time:.2f} 秒")
            logger.info(f"  执行成功: {success}")

        # 分析性能结果
        self._analyze_performance_results(results)

    def _analyze_performance_results(self, results: List[dict]):
        """分析性能测试结果"""
        logger.info("\n" + "=" * 60)
        logger.info("性能测试结果分析")
        logger.info("=" * 60)

        # 计算统计信息
        execution_times = [r['execution_time'] for r in results]
        success_count = sum(1 for r in results if r['success'])
        state_counts = [r['state_count'] for r in results]
        event_counts = [r['event_count'] for r in results]

        logger.info(f"测试次数: {len(results)}")
        logger.info(f"成功次数: {success_count}")
        logger.info(f"成功率: {success_count / len(results) * 100:.1f}%")

        logger.info(f"\n执行时间统计:")
        logger.info(f"  平均时间: {sum(execution_times) / len(execution_times):.2f} 秒")
        logger.info(f"  最短时间: {min(execution_times):.2f} 秒")
        logger.info(f"  最长时间: {max(execution_times):.2f} 秒")

        logger.info(f"\n状态转换统计:")
        logger.info(f"  平均状态数: {sum(state_counts) / len(state_counts):.1f}")
        logger.info(f"  最少状态数: {min(state_counts)}")
        logger.info(f"  最多状态数: {max(state_counts)}")

        logger.info(f"\n事件统计:")
        logger.info(f"  平均事件数: {sum(event_counts) / len(event_counts):.1f}")
        logger.info(f"  最少事件数: {min(event_counts)}")
        logger.info(f"  最多事件数: {max(event_counts)}")

    def export_results(self, filename: str = None):
        """导出结果到文件"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"architecture_demo_results_{timestamp}.json"

        try:
            with open(filename, 'w', encoding='utf - 8') as f:
                json.dump(self.demo_results, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"结果已导出到: {filename}")
        except Exception as e:
            logger.error(f"导出结果失败: {e}")

    def get_architecture_summary(self):
        """获取架构摘要"""
        logger.info("=" * 60)
        logger.info("业务流程驱动架构摘要")
        logger.info("=" * 60)

        logger.info("架构层次:")
        logger.info("  1. 监控反馈层 (Monitoring & Feedback Layer)")
        logger.info("  2. 交易执行层 (Trading Execution Layer)")
        logger.info("  3. 风控合规层 (Risk & Compliance Layer)")
        logger.info("  4. 策略决策层 (Strategy Decision Layer)")
        logger.info("  5. 模型推理层 (Model Inference Layer)")
        logger.info("  6. 特征处理层 (Feature Processing Layer)")
        logger.info("  7. 数据管理层 (Data Management Layer)")
        logger.info("  8. 基础设施层 (Infrastructure Layer)")
        logger.info("  9. 核心服务层 (Core Services Layer)")

        logger.info("\n核心特性:")
        logger.info("  - 基于业务流程的单向依赖关系")
        logger.info("  - 事件驱动的松耦合架构")
        logger.info("  - 状态机管理业务流程")
        logger.info("  - 分层接口标准化")
        logger.info("  - 实时监控和反馈机制")

        logger.info("\n业务流程:")
        logger.info("  数据采集 → 特征工程 → 模型预测 → 策略决策 → 风控检查 → 交易执行 → 监控反馈")


def main():
    """主函数"""
    # 创建演示实例
    demo = ArchitectureDemo()

    # 显示架构摘要
    demo.get_architecture_summary()

    # 运行完整交易周期演示
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    demo.run_complete_trading_cycle(symbols)

    # 运行性能测试
    demo.run_performance_test(symbols, iterations=3)

    # 导出结果
    demo.export_results()


if __name__ == "__main__":
    main()

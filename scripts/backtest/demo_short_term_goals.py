#!/usr/bin/env python3
"""
短期目标实现演示脚本
展示分布式回测和自动策略生成功能
"""

import logging
import pandas as pd
import numpy as np

from src.backtest.distributed_engine import DistributedBacktestEngine
from src.backtest.auto_strategy_generator import AutoStrategyGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_auto_strategy_generation():
    """演示自动策略生成"""
    print("🤖 自动策略生成演示")

    # 创建样本数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    price_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)

    data = {'AAPL': price_data}
    returns = price_data['close'].pct_change().dropna()

    # 生成策略
    generator = AutoStrategyGenerator()
    strategies = generator.generate_strategies(data, returns)

    print(f"生成策略数量: {len(strategies)}")

    # 显示最佳策略
    best_strategies = generator.get_best_strategies(count=3)
    for i, strategy in enumerate(best_strategies):
        print(f"策略 {i+1}: {strategy.strategy_name}")
        print(f"  夏普比率: {strategy.performance_metrics['sharpe_ratio']:.3f}")

    return strategies


def demo_distributed_backtest():
    """演示分布式回测"""
    print("\n⚡ 分布式回测演示")

    # 创建引擎
    engine = DistributedBacktestEngine()

    # 提交多个任务
    task_ids = []
    for i in range(3):
        task_id = engine.submit_backtest(
            strategy_config={'name': f'strategy_{i}', 'type': 'ma'},
            data_config={'symbol': 'AAPL', 'start_date': '2023-01-01'},
            backtest_config={'initial_capital': 100000}
        )
        task_ids.append(task_id)
        print(f"提交任务 {i+1}: {task_id}")

    # 获取系统统计
    stats = engine.get_system_stats()
    print(f"系统统计: {stats}")

    # 关闭引擎
    engine.shutdown()


def main():
    """主函数"""
    print("🚀 短期目标实现演示")
    print("="*50)

    # 演示自动策略生成
    strategies = demo_auto_strategy_generation()

    # 演示分布式回测
    demo_distributed_backtest()

    print("\n✅ 短期目标实现完成！")
    print("- 分布式回测系统: 支持大规模并行回测")
    print("- 自动策略生成: 自动发现和生成策略")
    print("- 系统集成: 完整的策略生成到回测流程")


if __name__ == "__main__":
    main()

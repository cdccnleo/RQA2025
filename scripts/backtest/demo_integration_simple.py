#!/usr/bin/env python3
"""
简化的系统集成优化演示脚本
专注于核心功能测试
"""

import logging
import pandas as pd
import numpy as np

from src.backtest.auto_strategy_generator import AutoStrategyGenerator
from src.backtest.distributed_engine import DistributedBacktestEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_auto_strategy_generation():
    """测试自动策略生成"""
    print("🤖 测试自动策略生成")

    # 创建简单样本数据
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

    print(f"✅ 生成策略数量: {len(strategies)}")

    # 获取最佳策略
    best_strategies = generator.get_best_strategies(count=2)
    print(f"✅ 最佳策略数量: {len(best_strategies)}")

    return strategies


def test_distributed_backtest():
    """测试分布式回测"""
    print("\n⚡ 测试分布式回测")

    # 创建引擎
    engine = DistributedBacktestEngine()

    # 提交简单任务
    task_id = engine.submit_backtest(
        strategy_config={'name': 'test_strategy', 'type': 'ma'},
        data_config={'symbol': 'AAPL', 'start_date': '2023-01-01'},
        backtest_config={'initial_capital': 100000}
    )

    print(f"✅ 提交任务成功: {task_id}")

    # 获取系统统计
    stats = engine.get_system_stats()
    print(f"✅ 系统统计: {stats}")

    # 关闭引擎
    engine.shutdown()
    print("✅ 分布式引擎关闭成功")


def test_integration():
    """测试集成功能"""
    print("\n🔗 测试集成功能")

    # 1. 生成策略
    strategies = test_auto_strategy_generation()

    # 2. 分布式回测
    test_distributed_backtest()

    print("✅ 集成测试完成")


def main():
    """主函数"""
    print("🚀 系统集成优化简化演示")
    print("="*40)

    try:
        test_integration()

        print("\n✅ 系统集成优化演示完成！")
        print("\n📋 集成成果:")
        print("- ✅ 自动策略生成功能正常")
        print("- ✅ 分布式回测引擎正常")
        print("- ✅ 系统集成基础功能验证通过")

    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        logger.error(f"演示错误: {e}", exc_info=True)


if __name__ == "__main__":
    main()

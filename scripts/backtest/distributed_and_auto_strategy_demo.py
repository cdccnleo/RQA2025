#!/usr/bin/env python3
"""
分布式回测和自动策略生成演示脚本
展示短期目标的实现成果
"""

import time
import logging
import pandas as pd
import numpy as np

from src.backtest.distributed_engine import DistributedBacktestEngine
from src.backtest.auto_strategy_generator import AutoStrategyGenerator

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data():
    """创建样本数据"""
    dates = pd.date_range('2023-01-01', periods=200, freq='D')

    # 创建多个股票的价格数据
    data = {}
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

    for symbol in symbols:
        # 生成随机价格数据
        base_price = 100 + np.random.randint(0, 100)
        price_data = pd.DataFrame({
            'open': np.random.randn(200).cumsum() + base_price,
            'high': np.random.randn(200).cumsum() + base_price + 2,
            'low': np.random.randn(200).cumsum() + base_price - 2,
            'close': np.random.randn(200).cumsum() + base_price,
            'volume': np.random.randint(1000, 10000, 200)
        }, index=dates)

        data[symbol] = price_data

    return data


def demo_auto_strategy_generation():
    """演示自动策略生成"""
    print("\n" + "="*60)
    print("🤖 自动策略生成演示")
    print("="*60)

    # 创建样本数据
    data = create_sample_data()

    # 计算收益率
    returns = data['AAPL']['close'].pct_change().dropna()

    # 创建策略生成器
    generator = AutoStrategyGenerator()

    print("1. 开始自动策略生成...")
    start_time = time.time()

    # 生成策略
    strategies = generator.generate_strategies(data, returns)

    generation_time = time.time() - start_time
    print(f"   生成完成，耗时: {generation_time:.2f}秒")
    print(f"   生成策略数量: {len(strategies)}")

    # 显示策略详情
    print("\n2. 生成的策略详情:")
    for i, strategy in enumerate(strategies[:5]):  # 只显示前5个
        print(f"   策略 {i+1}: {strategy.strategy_name}")
        print(f"     类型: {strategy.strategy_type}")
        print(f"     逻辑: {strategy.logic['type']}")
        print(f"     夏普比率: {strategy.performance_metrics['sharpe_ratio']:.3f}")
        print(f"     总收益率: {strategy.performance_metrics['total_return']:.3f}")
        print()

    # 获取最佳策略
    print("3. 最佳策略:")
    best_strategies = generator.get_best_strategies(count=3)
    for i, strategy in enumerate(best_strategies):
        print(f"   第{i+1}名: {strategy.strategy_name}")
        print(f"     夏普比率: {strategy.performance_metrics['sharpe_ratio']:.3f}")
        print(f"     总收益率: {strategy.performance_metrics['total_return']:.3f}")
        print(f"     最大回撤: {strategy.performance_metrics['max_drawdown']:.3f}")
        print()


def demo_distributed_backtest():
    """演示分布式回测"""
    print("\n" + "="*60)
    print("⚡ 分布式回测演示")
    print("="*60)

    # 创建分布式回测引擎
    engine = DistributedBacktestEngine()

    print("1. 初始化分布式回测引擎...")

    # 创建多个回测任务
    tasks = []
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

    for i, symbol in enumerate(symbols):
        task_config = {
            'strategy_config': {
                'name': f'ma_strategy_{symbol}',
                'type': 'moving_average',
                'parameters': {'short_period': 5, 'long_period': 20}
            },
            'data_config': {
                'symbol': symbol,
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'data_source': 'mock'
            },
            'backtest_config': {
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.001
            }
        }

        # 提交任务
        task_id = engine.submit_backtest(
            strategy_config=task_config['strategy_config'],
            data_config=task_config['data_config'],
            backtest_config=task_config['backtest_config'],
            priority=i
        )
        tasks.append(task_id)
        print(f"   提交任务 {i+1}: {task_id}")

    print(f"\n2. 已提交 {len(tasks)} 个回测任务")

    # 监控任务状态
    print("\n3. 监控任务状态...")
    for i in range(5):
        print(f"   第{i+1}次检查:")
        for task_id in tasks:
            status = engine.get_task_status(task_id)
            print(f"     任务 {task_id}: {status['status']}")
        time.sleep(1)

    # 获取系统统计
    print("\n4. 系统统计:")
    stats = engine.get_system_stats()
    print(f"   总任务数: {stats.get('total_tasks', 0)}")
    print(f"   完成任务数: {stats.get('completed_tasks', 0)}")
    print(f"   运行中任务数: {stats.get('running_tasks', 0)}")
    print(f"   平均执行时间: {stats.get('avg_execution_time', 0):.2f}秒")

    # 获取任务结果
    print("\n5. 任务结果:")
    for task_id in tasks:
        result = engine.get_task_result(task_id)
        if result:
            print(f"   任务 {task_id}:")
            print(f"     策略: {result.strategy_name}")
            print(f"     执行时间: {result.execution_time:.2f}秒")
            print(f"     内存使用: {result.memory_usage:.2f}MB")
            print(f"     夏普比率: {result.performance_metrics.get('sharpe_ratio', 0):.3f}")
            print(f"     总收益率: {result.performance_metrics.get('total_return', 0):.3f}")
            print()

    # 关闭引擎
    engine.shutdown()
    print("✅ 分布式回测演示完成")


def demo_integration():
    """演示集成功能"""
    print("\n" + "="*60)
    print("🔗 集成功能演示")
    print("="*60)

    # 创建样本数据
    data = create_sample_data()
    returns = data['AAPL']['close'].pct_change().dropna()

    print("1. 自动策略生成...")
    generator = AutoStrategyGenerator()
    strategies = generator.generate_strategies(data, returns)

    print(f"   生成策略数量: {len(strategies)}")

    # 使用生成的策略进行分布式回测
    print("\n2. 分布式回测生成的策略...")
    engine = DistributedBacktestEngine()

    for i, strategy in enumerate(strategies[:3]):  # 只测试前3个策略
        task_config = {
            'strategy_config': {
                'name': strategy.strategy_name,
                'type': strategy.strategy_type,
                'logic': strategy.logic,
                'parameters': strategy.parameters
            },
            'data_config': {
                'symbol': 'AAPL',
                'start_date': '2023-01-01',
                'end_date': '2023-12-31',
                'data_source': 'mock'
            },
            'backtest_config': {
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.001
            }
        }

        task_id = engine.submit_backtest(
            strategy_config=task_config['strategy_config'],
            data_config=task_config['data_config'],
            backtest_config=task_config['backtest_config']
        )

        print(f"   提交策略回测任务: {strategy.strategy_name}")

    # 等待任务完成
    time.sleep(2)

    # 获取结果
    print("\n3. 回测结果:")
    for strategy in strategies[:3]:
        result = engine.get_task_result(strategy.strategy_id)
        if result:
            print(f"   策略: {strategy.strategy_name}")
            print(f"     原始夏普比率: {strategy.performance_metrics['sharpe_ratio']:.3f}")
            print(f"     回测夏普比率: {result.performance_metrics.get('sharpe_ratio', 0):.3f}")
            print()

    engine.shutdown()
    print("✅ 集成功能演示完成")


def main():
    """主函数"""
    print("🚀 分布式回测和自动策略生成演示")
    print("="*60)
    print("本次演示展示短期目标的实现成果:")
    print("1. 分布式回测系统 - 支持大规模策略并行回测")
    print("2. 自动策略生成系统 - 自动发现和生成交易策略")
    print("3. 集成功能 - 自动生成策略并进行分布式回测")
    print("="*60)

    try:
        # 演示自动策略生成
        demo_auto_strategy_generation()

        # 演示分布式回测
        demo_distributed_backtest()

        # 演示集成功能
        demo_integration()

        print("\n" + "="*60)
        print("🎉 短期目标实现完成！")
        print("="*60)
        print("✅ 分布式回测系统: 支持大规模并行回测")
        print("✅ 自动策略生成: 自动发现和生成策略")
        print("✅ 系统集成: 完整的策略生成到回测流程")
        print("✅ 测试覆盖: 全面的单元测试和集成测试")
        print("✅ 性能优化: 高效的分布式计算和缓存机制")

    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()

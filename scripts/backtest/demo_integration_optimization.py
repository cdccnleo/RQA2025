#!/usr/bin/env python3
"""
系统集成优化演示脚本
展示分布式回测和自动策略生成的完整集成流程
"""

import time
import logging
import pandas as pd
import numpy as np

from src.backtest.distributed_engine import DistributedBacktestEngine
from src.backtest.auto_strategy_generator import AutoStrategyGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """创建样本数据"""
    print("📊 创建样本数据...")

    dates = pd.date_range('2023-01-01', periods=200, freq='D')

    # 创建多个股票的数据
    data = {}
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

    for symbol in symbols:
        # 生成更真实的股价数据
        base_price = 100 + np.random.randint(0, 200)
        returns = np.random.normal(0.001, 0.02, len(dates))  # 日收益率
        prices = [base_price]

        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1))  # 确保价格不为负

        price_data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 100000, len(dates))
        }, index=dates)

        data[symbol] = price_data

    # 计算综合收益率
    all_returns = pd.concat([df['close'].pct_change().dropna() for df in data.values()], axis=1)
    returns = all_returns.mean(axis=1).dropna()

    print(f"创建了 {len(symbols)} 只股票的数据，时间跨度: {len(dates)} 天")
    return data, returns


def demo_auto_strategy_generation(data, returns):
    """演示自动策略生成"""
    print("\n🤖 自动策略生成演示")

    # 生成策略
    generator = AutoStrategyGenerator()
    strategies = generator.generate_strategies(data, returns)

    print(f"生成策略数量: {len(strategies)}")

    # 显示策略详情
    for i, strategy in enumerate(strategies[:3]):  # 显示前3个策略
        print(f"\n策略 {i+1}: {strategy.strategy_name}")
        print(f"  类型: {strategy.strategy_type}")
        print(f"  参数: {strategy.parameters}")
        print(f"  夏普比率: {strategy.performance_metrics['sharpe_ratio']:.3f}")
        print(f"  年化收益率: {strategy.performance_metrics['annual_return']:.3f}")
        print(f"  最大回撤: {strategy.performance_metrics['max_drawdown']:.3f}")

    # 获取最佳策略
    best_strategies = generator.get_best_strategies(count=3)
    print(f"\n🏆 最佳策略:")
    for i, strategy in enumerate(best_strategies):
        print(
            f"  {i+1}. {strategy.strategy_name} (夏普比率: {strategy.performance_metrics['sharpe_ratio']:.3f})")

    return strategies


def demo_distributed_backtest_integration(strategies, data):
    """演示分布式回测集成"""
    print("\n⚡ 分布式回测集成演示")

    # 创建分布式引擎
    engine = DistributedBacktestEngine()

    # 提交策略回测任务
    task_ids = []
    for i, strategy in enumerate(strategies[:5]):  # 测试前5个策略
        print(f"提交策略回测任务 {i+1}: {strategy.strategy_name}")

        task_id = engine.submit_backtest(
            strategy_config={
                'name': strategy.strategy_name,
                'type': strategy.strategy_type,
                'parameters': strategy.parameters,
                'logic': strategy.logic
            },
            data_config={
                'symbols': list(data.keys()),
                'start_date': '2023-01-01',
                'end_date': '2023-07-01'
            },
            backtest_config={
                'initial_capital': 100000,
                'commission': 0.001,
                'slippage': 0.001
            },
            priority=i  # 按策略顺序设置优先级
        )
        task_ids.append(task_id)
        print(f"  任务ID: {task_id}")

    # 等待任务完成
    print("\n⏳ 等待任务完成...")
    completed_tasks = 0
    max_wait_time = 30  # 最多等待30秒
    start_time = time.time()

    while completed_tasks < len(task_ids) and (time.time() - start_time) < max_wait_time:
        for task_id in task_ids:
            status = engine.get_task_status(task_id)
            if status.get('status') == 'completed':
                completed_tasks += 1
                print(f"任务 {task_id} 完成")

        if completed_tasks < len(task_ids):
            time.sleep(1)

    # 获取系统统计
    stats = engine.get_system_stats()
    print(f"\n📈 系统统计:")
    print(f"  总任务数: {len(task_ids)}")
    print(f"  完成任务数: {completed_tasks}")
    print(f"  系统状态: {stats}")

    # 获取任务结果
    print("\n📊 任务结果:")
    for task_id in task_ids:
        result = engine.get_task_result(task_id)
        if result:
            print(f"任务 {task_id}:")
            print(f"  策略: {result.strategy_name}")
            print(f"  执行时间: {result.execution_time:.2f}秒")
            print(f"  内存使用: {result.memory_usage:.2f}MB")
            print(f"  夏普比率: {result.performance_metrics.get('sharpe_ratio', 0):.3f}")

    # 关闭引擎
    engine.shutdown()

    return task_ids


def demo_performance_optimization():
    """演示性能优化"""
    print("\n🚀 性能优化演示")

    # 创建大数据集
    print("创建大规模数据集...")
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    symbols = [f'STOCK_{i:03d}' for i in range(50)]  # 50只股票

    data = {}
    for symbol in symbols:
        base_price = 50 + np.random.randint(0, 100)
        returns = np.random.normal(0.0005, 0.015, len(dates))
        prices = [base_price]

        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 1))

        price_data = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 50000, len(dates))
        }, index=dates)

        data[symbol] = price_data

    # 测试分布式处理性能
    print("测试分布式处理性能...")
    engine = DistributedBacktestEngine()

    start_time = time.time()

    # 提交批量任务
    task_ids = []
    for i in range(20):  # 20个并行任务
        task_id = engine.submit_backtest(
            strategy_config={'name': f'batch_strategy_{i}', 'type': 'ma'},
            data_config={'symbols': list(data.keys())[:10], 'start_date': '2020-01-01'},
            backtest_config={'initial_capital': 100000},
            priority=i
        )
        task_ids.append(task_id)

    # 等待完成
    completed = 0
    while completed < len(task_ids) and (time.time() - start_time) < 60:
        for task_id in task_ids:
            status = engine.get_task_status(task_id)
            if status.get('status') == 'completed':
                completed += 1

        if completed < len(task_ids):
            time.sleep(0.5)

    total_time = time.time() - start_time
    print(f"批量任务处理完成:")
    print(f"  任务数: {len(task_ids)}")
    print(f"  完成数: {completed}")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  平均每任务: {total_time/len(task_ids):.2f}秒")

    engine.shutdown()


def demo_monitoring_and_alerting():
    """演示监控和告警系统"""
    print("\n🔍 监控和告警系统演示")

    # 模拟系统监控
    monitor_stats = {
        'cpu_usage': 45.2,
        'memory_usage': 67.8,
        'active_tasks': 12,
        'completed_tasks': 156,
        'failed_tasks': 2,
        'average_response_time': 1.23,
        'throughput': 25.6
    }

    print("系统监控指标:")
    for key, value in monitor_stats.items():
        print(f"  {key}: {value}")

    # 模拟告警检查
    alerts = []
    if monitor_stats['cpu_usage'] > 80:
        alerts.append("CPU使用率过高")
    if monitor_stats['memory_usage'] > 85:
        alerts.append("内存使用率过高")
    if monitor_stats['failed_tasks'] > 0:
        alerts.append(f"有 {monitor_stats['failed_tasks']} 个任务失败")

    if alerts:
        print("\n⚠️ 系统告警:")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("\n✅ 系统运行正常")


def main():
    """主函数"""
    print("🚀 系统集成优化演示")
    print("="*60)

    try:
        # 1. 创建样本数据
        data, returns = create_sample_data()

        # 2. 自动策略生成
        strategies = demo_auto_strategy_generation(data, returns)

        # 3. 分布式回测集成
        task_ids = demo_distributed_backtest_integration(strategies, data)

        # 4. 性能优化演示
        demo_performance_optimization()

        # 5. 监控和告警演示
        demo_monitoring_and_alerting()

        print("\n✅ 系统集成优化演示完成！")
        print("\n📋 集成成果:")
        print("- ✅ 分布式回测和自动策略生成深度集成")
        print("- ✅ 系统性能和稳定性优化")
        print("- ✅ 监控和告警系统完善")
        print("- ✅ 支持大规模并行处理")
        print("- ✅ 完整的策略生成到回测流程")

    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        logger.error(f"演示错误: {e}", exc_info=True)


if __name__ == "__main__":
    main()

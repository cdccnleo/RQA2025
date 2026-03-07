#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分布式回测引擎使用示例

演示如何使用分布式回测引擎进行大规模策略回测。
"""

from src.backtest.distributed_engine import DistributedBacktestEngine
import sys
import os
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def simple_strategy(data):
    """简单策略：买入所有股票"""
    signals = {}
    for symbol in data.keys():
        signals[symbol] = 1.0  # 买入信号
    return signals


def momentum_strategy(data):
    """动量策略：基于价格动量的策略"""
    signals = {}
    for symbol, df in data.items():
        if len(df) > 20:
            # 计算20日动量
            momentum = (df['close'].iloc[-1] / df['close'].iloc[-20]) - 1
            if momentum > 0.05:  # 动量大于5%
                signals[symbol] = 1.0  # 买入
            elif momentum < -0.05:  # 动量小于-5%
                signals[symbol] = -1.0  # 卖出
            else:
                signals[symbol] = 0.0  # 持有
        else:
            signals[symbol] = 0.0
    return signals


def main():
    """主函数：演示分布式回测引擎的使用"""
    print("🚀 启动分布式回测引擎示例")

    # 1. 初始化分布式回测引擎
    config = {
        'max_workers': 4,  # 使用4个worker
        'cache_dir': 'cache/distributed_backtest'
    }

    engine = DistributedBacktestEngine(config)
    print(f"✅ 分布式回测引擎初始化完成，使用 {config['max_workers']} 个worker")

    try:
        # 2. 准备多个回测任务
        tasks = []

        # 任务1：简单策略
        tasks.append({
            'name': 'simple_strategy',
            'strategy_config': {'name': 'simple_strategy', 'type': 'simple'},
            'data_config': {
                'symbols': ['000001.SZ', '000002.SZ', '000858.SZ'],
                'start_date': '2023-01-01',
                'end_date': '2023-06-30'
            },
            'backtest_config': {
                'initial_capital': 1000000,
                'commission_rate': 0.0003,
                'slippage_rate': 0.0001
            }
        })

        # 任务2：动量策略
        tasks.append({
            'name': 'momentum_strategy',
            'strategy_config': {'name': 'momentum_strategy', 'type': 'momentum'},
            'data_config': {
                'symbols': ['000001.SZ', '000002.SZ', '000858.SZ', '000300.SH'],
                'start_date': '2023-01-01',
                'end_date': '2023-06-30'
            },
            'backtest_config': {
                'initial_capital': 2000000,
                'commission_rate': 0.0003,
                'slippage_rate': 0.0001
            }
        })

        # 任务3：大规模回测
        tasks.append({
            'name': 'large_scale_backtest',
            'strategy_config': {'name': 'simple_strategy', 'type': 'simple'},
            'data_config': {
                'symbols': [f'00000{i}.SZ' for i in range(1, 11)],  # 10只股票
                'start_date': '2023-01-01',
                'end_date': '2023-12-31'
            },
            'backtest_config': {
                'initial_capital': 5000000,
                'commission_rate': 0.0003,
                'slippage_rate': 0.0001
            }
        })

        print(f"📋 准备提交 {len(tasks)} 个回测任务")

        # 3. 提交任务
        task_ids = []
        for i, task in enumerate(tasks, 1):
            task_id = engine.submit_backtest(
                task['strategy_config'],
                task['data_config'],
                task['backtest_config'],
                priority=i  # 优先级递增
            )
            task_ids.append(task_id)
            print(f"✅ 任务 {i} ({task['name']}) 已提交，ID: {task_id}")

        # 4. 监控任务状态
        print("\n📊 监控任务执行状态...")
        for i, task_id in enumerate(task_ids, 1):
            status = engine.get_task_status(task_id)
            print(f"任务 {i} 状态: {status['status']}")

        # 5. 等待任务完成并获取结果
        print("\n⏳ 等待任务完成...")
        time.sleep(10)  # 等待任务执行

        # 6. 获取系统统计
        stats = engine.get_system_stats()
        print(f"\n📈 系统统计:")
        print(f"  运行时间: {stats['uptime_seconds']:.2f} 秒")
        print(f"  内存使用: {stats['memory_usage_percent']:.1f}%")
        print(f"  CPU使用: {stats['cpu_usage_percent']:.1f}%")
        print(f"  磁盘使用: {stats['disk_usage_percent']:.1f}%")

        # 7. 获取任务结果
        print(f"\n📋 任务结果:")
        for i, task_id in enumerate(task_ids, 1):
            result = engine.get_task_result(task_id)
            if result:
                print(f"任务 {i} 结果:")
                print(f"  策略名称: {result.strategy_name}")
                print(f"  执行时间: {result.execution_time:.2f} 秒")
                print(f"  内存使用: {result.memory_usage:.1f} MB")
                if result.performance_metrics:
                    print(f"  性能指标: {result.performance_metrics}")
            else:
                print(f"任务 {i} 结果尚未完成")

        print("\n✅ 分布式回测示例完成！")

    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 8. 关闭引擎
        print("\n🔄 关闭分布式回测引擎...")
        engine.shutdown()
        print("✅ 分布式回测引擎已关闭")


if __name__ == "__main__":
    main()

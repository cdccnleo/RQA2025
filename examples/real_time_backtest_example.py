#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实时回测引擎使用示例

演示如何使用实时回测引擎进行实时策略回测。
"""

from src.backtest.real_time_engine import RealTimeBacktestEngine
import sys
import os
import time
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def simple_momentum_strategy(data, state):
    """简单动量策略"""
    signals = {}

    # 模拟策略逻辑
    if data.symbol == '000001.SZ':
        # 基于价格变化的简单信号
        price = data.data.get('price', 100.0)
        if price > 105.0:
            signals[data.symbol] = 1.0  # 买入
        elif price < 95.0:
            signals[data.symbol] = -1.0  # 卖出
        else:
            signals[data.symbol] = 0.0  # 持有

    return signals


def risk_management_strategy(data, state):
    """风险管理策略"""
    signals = {}

    # 检查投资组合风险
    if state.portfolio_value < 900000:  # 投资组合价值过低
        # 减少持仓
        for symbol in state.positions:
            signals[symbol] = -0.5  # 减仓50%

    return signals


def main():
    """主函数：演示实时回测引擎的使用"""
    print("🚀 启动实时回测引擎示例")

    # 1. 初始化实时回测引擎
    config = {
        'max_workers': 2,
        'cache_dir': 'cache/real_time_backtest'
    }

    engine = RealTimeBacktestEngine(config)
    print(f"✅ 实时回测引擎初始化完成")

    try:
        # 2. 添加策略
        engine.add_strategy('momentum', simple_momentum_strategy)
        engine.add_strategy('risk_management', risk_management_strategy)
        print("✅ 策略添加完成")

        # 3. 启动引擎
        initial_capital = 1000000.0
        engine.start(initial_capital)
        print(f"✅ 实时回测引擎已启动，初始资金: {initial_capital}")

        # 4. 模拟实时数据流
        print("📊 开始模拟实时数据流...")

        # 模拟市场数据
        market_data = [
            {
                'timestamp': datetime.now(),
                'symbol': '000001.SZ',
                'type': 'market',
                'source': 'simulation',
                'price': 100.0,
                'volume': 1000
            },
            {
                'timestamp': datetime.now() + timedelta(seconds=1),
                'symbol': '000001.SZ',
                'type': 'market',
                'source': 'simulation',
                'price': 105.0,
                'volume': 1200
            },
            {
                'timestamp': datetime.now() + timedelta(seconds=2),
                'symbol': '000001.SZ',
                'type': 'market',
                'source': 'simulation',
                'price': 110.0,
                'volume': 1500
            },
            {
                'timestamp': datetime.now() + timedelta(seconds=3),
                'symbol': '000001.SZ',
                'type': 'market',
                'source': 'simulation',
                'price': 95.0,
                'volume': 800
            }
        ]

        # 处理数据
        for i, data in enumerate(market_data):
            print(f"📈 处理数据 {i+1}/{len(market_data)}: {data['symbol']} @ {data['price']}")

            # 处理数据
            engine.process_data(data)

            # 获取当前状态
            current_state = engine.get_current_state()
            if current_state:
                print(f"   💰 投资组合价值: {current_state.portfolio_value:,.2f}")
                print(f"   💵 现金: {current_state.cash:,.2f}")
                print(f"   📊 持仓数量: {len(current_state.positions)}")
                print(f"   📈 交易数量: {len(current_state.trades)}")

            # 获取指标
            metrics = engine.get_metrics()
            if metrics:
                print(f"   📊 指标: {metrics}")

            # 模拟实时延迟
            time.sleep(0.5)

        print("✅ 数据流处理完成")

        # 5. 获取最终结果
        final_state = engine.get_current_state()
        if final_state:
            print("\n📊 最终结果:")
            print(f"   投资组合价值: {final_state.portfolio_value:,.2f}")
            print(f"   现金: {final_state.cash:,.2f}")
            print(f"   持仓: {final_state.positions}")
            print(f"   交易记录: {len(final_state.trades)} 笔")
            print(f"   指标: {final_state.metrics}")

        # 6. 停止引擎
        engine.stop()
        print("✅ 实时回测引擎已停止")

    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        engine.stop()

    print("🎉 实时回测引擎示例完成")


if __name__ == "__main__":
    main()

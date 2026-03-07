#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级实时回测引擎示例

演示增量计算、策略执行和性能监控功能。
"""

from src.backtest.real_time_engine import RealTimeBacktestEngine
import sys
import os
import time
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def momentum_strategy(data, state):
    """动量策略"""
    signals = {}

    if data.symbol == '000001.SZ':
        price = data.data.get('price', 100.0)

        # 简单的动量策略：价格突破时买入，跌破时卖出
        if price > 105.0:
            signals[data.symbol] = 100.0  # 买入100股
        elif price < 95.0:
            signals[data.symbol] = -100.0  # 卖出100股

    return signals


def mean_reversion_strategy(data, state):
    """均值回归策略"""
    signals = {}

    if data.symbol == '000002.SZ':
        price = data.data.get('price', 100.0)

        # 均值回归策略：价格偏离均值时反向操作
        if price > 110.0:
            signals[data.symbol] = -50.0  # 卖出50股
        elif price < 90.0:
            signals[data.symbol] = 50.0   # 买入50股

    return signals


def risk_management_strategy(data, state):
    """风险管理策略"""
    signals = {}

    # 检查投资组合风险
    if state.portfolio_value < 950000:  # 投资组合价值过低
        # 减少所有持仓
        for symbol in state.positions:
            signals[symbol] = -state.positions[symbol] * 0.5  # 减仓50%

    return signals


def main():
    """主函数：演示高级实时回测功能"""
    print("🚀 启动高级实时回测引擎示例")

    # 1. 初始化实时回测引擎
    config = {
        'max_workers': 4,
        'cache_dir': 'cache/real_time_backtest_advanced'
    }

    engine = RealTimeBacktestEngine(config)
    print(f"✅ 实时回测引擎初始化完成")

    try:
        # 2. 添加多个策略
        engine.add_strategy('momentum', momentum_strategy)
        engine.add_strategy('mean_reversion', mean_reversion_strategy)
        engine.add_strategy('risk_management', risk_management_strategy)
        print("✅ 策略添加完成")

        # 3. 启动引擎
        initial_capital = 1000000.0
        engine.start(initial_capital)
        print(f"✅ 实时回测引擎已启动，初始资金: {initial_capital}")

        # 4. 模拟多股票实时数据流
        print("📊 开始模拟多股票实时数据流...")

        # 模拟多股票市场数据
        market_data = [
            # 时间序列1
            {
                'timestamp': datetime.now(),
                'symbol': '000001.SZ',
                'type': 'market',
                'source': 'simulation',
                'price': 100.0,
                'volume': 1000
            },
            {
                'timestamp': datetime.now(),
                'symbol': '000002.SZ',
                'type': 'market',
                'source': 'simulation',
                'price': 100.0,
                'volume': 800
            },
            # 时间序列2 - 价格上涨
            {
                'timestamp': datetime.now() + timedelta(seconds=1),
                'symbol': '000001.SZ',
                'type': 'market',
                'source': 'simulation',
                'price': 106.0,
                'volume': 1200
            },
            {
                'timestamp': datetime.now() + timedelta(seconds=1),
                'symbol': '000002.SZ',
                'type': 'market',
                'source': 'simulation',
                'price': 112.0,
                'volume': 1000
            },
            # 时间序列3 - 价格回调
            {
                'timestamp': datetime.now() + timedelta(seconds=2),
                'symbol': '000001.SZ',
                'type': 'market',
                'source': 'simulation',
                'price': 94.0,
                'volume': 800
            },
            {
                'timestamp': datetime.now() + timedelta(seconds=2),
                'symbol': '000002.SZ',
                'type': 'market',
                'source': 'simulation',
                'price': 88.0,
                'volume': 600
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

                # 显示持仓详情
                if current_state.positions:
                    print(f"   📋 持仓详情:")
                    for symbol, quantity in current_state.positions.items():
                        print(f"      {symbol}: {quantity} 股")

                # 显示最新交易
                if current_state.trades:
                    latest_trade = current_state.trades[-1]
                    print(f"   🔄 最新交易: {latest_trade['action']} {latest_trade['symbol']} "
                          f"{latest_trade['quantity']} 股 @ {latest_trade['price']}")

            # 获取指标
            metrics = engine.get_metrics()
            if metrics:
                print(f"   📊 指标: {metrics}")

            # 模拟实时延迟
            time.sleep(0.3)

        print("✅ 数据流处理完成")

        # 5. 获取最终结果
        final_state = engine.get_current_state()
        if final_state:
            print("\n📊 最终结果:")
            print(f"   投资组合价值: {final_state.portfolio_value:,.2f}")
            print(f"   现金: {final_state.cash:,.2f}")
            print(f"   持仓: {final_state.positions}")
            print(f"   交易记录: {len(final_state.trades)} 笔")

            # 显示交易历史
            if final_state.trades:
                print(f"   📋 交易历史:")
                for i, trade in enumerate(final_state.trades[-5:]):  # 显示最后5笔交易
                    print(f"     {i+1}. {trade['action']} {trade['symbol']} "
                          f"{trade['quantity']} 股 @ {trade['price']} "
                          f"(价值: {trade['value']:,.2f})")

            print(f"   指标: {final_state.metrics}")

        # 6. 停止引擎
        engine.stop()
        print("✅ 实时回测引擎已停止")

    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        engine.stop()

    print("🎉 高级实时回测引擎示例完成")


if __name__ == "__main__":
    main()

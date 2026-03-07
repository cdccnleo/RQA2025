#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实时回测系统第二阶段功能示例

演示动态策略管理、缓存优化、参数管理和高级监控功能。
"""

import time
import logging
from datetime import datetime
import random

from src.backtest.real_time_engine import RealTimeBacktestEngine

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def momentum_strategy(data, state):
    """动量策略"""
    price = data.data.get("price", 0)
    volume = data.data.get("volume", 0)

    # 基于价格和成交量的动量信号
    if price > 100 and volume > 500:
        return {data.symbol: 100}
    elif price < 80:
        return {data.symbol: -50}
    return {}


def value_strategy(data, state):
    """价值策略"""
    price = data.data.get("price", 0)

    # 基于价格的价值信号
    if price < 50:
        return {data.symbol: 200}
    elif price > 120:
        return {data.symbol: -100}
    return {}


def mean_reversion_strategy(data, state):
    """均值回归策略"""
    price = data.data.get("price", 0)

    # 简化的均值回归逻辑
    if price < 90:
        return {data.symbol: 150}
    elif price > 110:
        return {data.symbol: -150}
    return {}


def main():
    """主函数"""
    logger.info("开始实时回测系统第二阶段功能演示")

    # 初始化实时回测引擎
    engine = RealTimeBacktestEngine()

    # 添加策略
    engine.add_strategy("momentum", momentum_strategy)
    engine.add_strategy("value", value_strategy)
    engine.add_strategy("mean_reversion", mean_reversion_strategy)

    # 第二阶段功能：动态策略管理
    logger.info("=== 动态策略管理功能 ===")

    # 设置策略参数
    engine.update_strategy_parameters("momentum", {"threshold": 100, "volume_threshold": 500})
    engine.update_strategy_parameters("value", {"buy_threshold": 50, "sell_threshold": 120})
    engine.update_strategy_parameters("mean_reversion", {"lower_bound": 90, "upper_bound": 110})

    # 设置风险限制
    engine.incremental_engine.dynamic_manager.strategies["momentum"]["config"].risk_limits = {
        "max_drawdown": 0.1,
        "var_95": 0.05
    }

    # 启动引擎
    engine.start(initial_capital=1000000.0)

    # 第二阶段功能：设置告警阈值
    engine.set_alert_threshold("portfolio_value", 500000.0)
    engine.set_alert_threshold("cash", 100000.0)

    logger.info("引擎已启动，开始处理模拟数据...")

    # 模拟实时数据流
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA"]
    base_prices = {"AAPL": 150, "GOOGL": 2800, "MSFT": 300, "TSLA": 800}

    for i in range(20):
        # 生成模拟市场数据
        for symbol in symbols:
            # 模拟价格波动
            base_price = base_prices[symbol]
            price_change = random.uniform(-0.05, 0.05)
            current_price = base_price * (1 + price_change)
            volume = random.randint(100, 2000)

            # 更新基础价格
            base_prices[symbol] = current_price

            # 创建市场数据
            market_data = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "type": "market",
                "price": current_price,
                "volume": volume,
                "source": "simulator"
            }

            # 处理数据
            engine.process_data(market_data)

            # 获取当前状态
            current_state = engine.get_current_state()
            if current_state:
                logger.info(f"时间: {current_state.timestamp}, "
                            f"投资组合价值: {current_state.portfolio_value:.2f}, "
                            f"现金: {current_state.cash:.2f}, "
                            f"持仓数量: {len(current_state.positions)}")

        # 第二阶段功能：动态策略调整
        if i == 10:
            logger.info("=== 动态策略调整 ===")

            # 禁用价值策略
            engine.disable_strategy("value")
            logger.info("价值策略已禁用")

            # 更新动量策略参数
            engine.update_strategy_parameters(
                "momentum", {"threshold": 120, "volume_threshold": 800})
            logger.info("动量策略参数已更新")

            # 启用均值回归策略
            engine.enable_strategy("mean_reversion")
            logger.info("均值回归策略已启用")

        # 获取指标
        metrics = engine.get_metrics()
        if metrics and "current" in metrics:
            current_metrics = metrics["current"]
            logger.info(f"当前指标 - 投资组合价值: {current_metrics.get('portfolio_value', 0):.2f}, "
                        f"现金: {current_metrics.get('cash', 0):.2f}")

        time.sleep(0.1)  # 模拟实时数据间隔

    # 第二阶段功能：性能分析
    logger.info("=== 性能分析 ===")

    # 获取所有策略的性能指标
    for strategy_id, strategy_info in engine.incremental_engine.dynamic_manager.strategies.items():
        config = strategy_info["config"]
        performance = config.performance_metrics
        logger.info(f"策略 {strategy_id}:")
        logger.info(f"  启用状态: {config.enabled}")
        logger.info(f"  参数: {config.parameters}")
        logger.info(f"  风险限制: {config.risk_limits}")
        logger.info(f"  性能指标: {performance}")

    # 获取缓存统计
    cache_manager = engine.incremental_engine.cache_manager
    logger.info(f"缓存统计:")
    logger.info(f"  数据缓存条目数: {len(cache_manager.data_cache)}")
    logger.info(f"  指标缓存条目数: {len(cache_manager.metrics_cache)}")

    # 停止引擎
    engine.stop()
    logger.info("引擎已停止")

    # 最终状态报告
    final_state = engine.get_current_state()
    if final_state:
        logger.info("=== 最终状态报告 ===")
        logger.info(f"最终投资组合价值: {final_state.portfolio_value:.2f}")
        logger.info(f"最终现金: {final_state.cash:.2f}")
        logger.info(f"持仓: {final_state.positions}")
        logger.info(f"总交易次数: {len(final_state.trades)}")

        # 计算收益率
        initial_capital = 1000000.0
        total_return = (final_state.portfolio_value - initial_capital) / initial_capital
        logger.info(f"总收益率: {total_return:.4f} ({total_return*100:.2f}%)")

    logger.info("✓ 实时回测系统第二阶段功能演示完成")


if __name__ == "__main__":
    main()

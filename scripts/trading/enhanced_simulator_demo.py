#!/usr/bin/env python3
"""
增强策略模拟器演示脚本
"""

from src.trading.strategy_workspace import (
    AutomaticStrategyGenerator, StrategySimulator,
    StrategyConfig, SimulationConfig, SimulationMode, StrategyTemplate, MarketType
)
import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_sample_market_data(days: int = 252) -> pd.DataFrame:
    """生成样本市场数据"""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, days)
    prices = 100 * np.exp(np.cumsum(returns))
    volumes = np.random.lognormal(10, 0.5, days)

    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
        'close': prices,
        'volume': volumes
    })

    data.set_index('date', inplace=True)
    return data


def demo_backtest_simulation():
    """演示回测模拟"""
    logger.info("=== 回测模拟演示 ===")

    generator = AutomaticStrategyGenerator()
    config = StrategyConfig(
        template=StrategyTemplate.MOVING_AVERAGE,
        market_type=MarketType.A_SHARE,
        symbols=["000001.SZ"],
        timeframes=["1d"],
        risk_level="medium",
        target_return=0.15,
        max_drawdown=0.1
    )

    strategy, initial_params = generator.generate_strategy(config)
    market_data = generate_sample_market_data(100)
    simulator = StrategySimulator()

    sim_config = SimulationConfig(
        mode=SimulationMode.BACKTEST,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 4, 30),
        initial_capital=100000.0,
        commission_rate=0.0003,
        slippage=0.0001,
        risk_free_rate=0.03
    )

    result = simulator.simulate(strategy, market_data, sim_config)

    logger.info(f"回测模拟结果:")
    logger.info(f"总收益率: {result.total_return:.2%}")
    logger.info(f"年化收益率: {result.annualized_return:.2%}")
    logger.info(f"夏普比率: {result.sharpe_ratio:.2f}")
    logger.info(f"最大回撤: {result.max_drawdown:.2%}")

    return result


def demo_monte_carlo_simulation():
    """演示蒙特卡洛模拟"""
    logger.info("=== 蒙特卡洛模拟演示 ===")

    generator = AutomaticStrategyGenerator()
    config = StrategyConfig(
        template=StrategyTemplate.RSI,
        market_type=MarketType.A_SHARE,
        symbols=["000001.SZ"],
        timeframes=["1d"],
        risk_level="medium",
        target_return=0.15,
        max_drawdown=0.1
    )

    strategy, initial_params = generator.generate_strategy(config)
    market_data = generate_sample_market_data(100)
    simulator = StrategySimulator()

    sim_config = SimulationConfig(
        mode=SimulationMode.MONTE_CARLO,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 4, 30),
        initial_capital=100000.0,
        commission_rate=0.0003,
        slippage=0.0001,
        risk_free_rate=0.03,
        monte_carlo_iterations=20  # 减少迭代次数
    )

    result = simulator.simulate(strategy, market_data, sim_config)

    logger.info(f"蒙特卡洛模拟结果:")
    logger.info(f"平均收益率: {result.total_return:.2%}")
    logger.info(f"平均夏普比率: {result.sharpe_ratio:.2f}")

    return result


def demo_stress_test_simulation():
    """演示压力测试模拟"""
    logger.info("=== 压力测试模拟演示 ===")

    generator = AutomaticStrategyGenerator()
    config = StrategyConfig(
        template=StrategyTemplate.MACHINE_LEARNING,
        market_type=MarketType.A_SHARE,
        symbols=["000001.SZ"],
        timeframes=["1d"],
        risk_level="medium",
        target_return=0.15,
        max_drawdown=0.1
    )

    strategy, initial_params = generator.generate_strategy(config)
    market_data = generate_sample_market_data(100)
    simulator = StrategySimulator()

    stress_scenarios = [
        {"name": "价格冲击-10%", "price_shock": -0.1},
        {"name": "价格冲击+10%", "price_shock": 0.1}
    ]

    sim_config = SimulationConfig(
        mode=SimulationMode.STRESS_TEST,
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 4, 30),
        initial_capital=100000.0,
        commission_rate=0.0003,
        slippage=0.0001,
        risk_free_rate=0.03,
        stress_test_scenarios=stress_scenarios
    )

    result = simulator.simulate(strategy, market_data, sim_config)

    logger.info(f"压力测试模拟结果:")
    logger.info(f"最坏情况收益率: {result.total_return:.2%}")
    logger.info(f"最坏情况夏普比率: {result.sharpe_ratio:.2f}")

    return result


def main():
    """主函数"""
    logger.info("开始增强策略模拟器演示")

    try:
        demo_backtest_simulation()
        print()

        demo_monte_carlo_simulation()
        print()

        demo_stress_test_simulation()
        print()

        logger.info("增强策略模拟器演示完成")

    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()

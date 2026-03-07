#!/usr/bin/env python3
"""
增强策略优化器演示脚本

展示新增的优化算法功能：
- 粒子群优化
- 模拟退火优化
- 多目标优化
- 集成优化
"""

from src.trading.strategy_workspace.visual_editor import VisualStrategyEditor
from src.trading.strategy_workspace import (
    AutomaticStrategyGenerator, StrategyOptimizer, StrategyConfig, OptimizationConfig,
    OptimizationMethod, StrategyTemplate, MarketType
)
import sys
import os
import logging
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_sample_market_data(days: int = 252) -> pd.DataFrame:
    """生成样本市场数据"""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

    # 生成价格数据
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, days)  # 日收益率
    prices = 100 * np.exp(np.cumsum(returns))

    # 生成成交量数据
    volumes = np.random.lognormal(10, 0.5, days)

    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
        'close': prices,
        'volume': volumes
    })

    return data


def objective_function(strategy: VisualStrategyEditor) -> float:
    """目标函数 - 模拟策略评估"""
    # 这里应该运行策略模拟，暂时返回模拟的夏普比率
    return np.random.uniform(0.5, 2.0)


def demo_particle_swarm_optimization():
    """演示粒子群优化"""
    logger.info("=== 粒子群优化演示 ===")

    # 创建策略生成器
    generator = AutomaticStrategyGenerator()

    # 生成策略
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

    # 创建优化器
    optimizer = StrategyOptimizer()

    # 配置粒子群优化
    opt_config = OptimizationConfig(
        method=OptimizationMethod.PARTICLE_SWARM,
        max_iterations=50,
        particle_count=20,
        cognitive_weight=2.0,
        social_weight=2.0,
        inertia_weight=0.7
    )

    # 运行优化
    result = optimizer.optimize(strategy, objective_function, opt_config)

    logger.info(f"粒子群优化结果:")
    logger.info(f"最佳参数: {result.best_params}")
    logger.info(f"最佳得分: {result.best_score:.4f}")
    logger.info(f"评估次数: {len(result.all_results)}")

    return result


def demo_simulated_annealing():
    """演示模拟退火优化"""
    logger.info("=== 模拟退火优化演示 ===")

    # 创建策略生成器
    generator = AutomaticStrategyGenerator()

    # 生成策略
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

    # 创建优化器
    optimizer = StrategyOptimizer()

    # 配置模拟退火优化
    opt_config = OptimizationConfig(
        method=OptimizationMethod.SIMULATED_ANNEALING,
        max_iterations=100,
        initial_temperature=100.0,
        cooling_rate=0.95
    )

    # 运行优化
    result = optimizer.optimize(strategy, objective_function, opt_config)

    logger.info(f"模拟退火优化结果:")
    logger.info(f"最佳参数: {result.best_params}")
    logger.info(f"最佳得分: {result.best_score:.4f}")
    logger.info(f"评估次数: {len(result.all_results)}")

    return result


def demo_multi_objective_optimization():
    """演示多目标优化"""
    logger.info("=== 多目标优化演示 ===")

    # 创建策略生成器
    generator = AutomaticStrategyGenerator()

    # 生成策略
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

    # 创建优化器
    optimizer = StrategyOptimizer()

    # 配置多目标优化
    opt_config = OptimizationConfig(
        method=OptimizationMethod.MULTI_OBJECTIVE,
        max_iterations=50,
        population_size=30,
        objectives=["sharpe_ratio", "max_drawdown", "total_return"],
        weights=[0.4, 0.3, 0.3]
    )

    # 运行优化
    result = optimizer.optimize(strategy, objective_function, opt_config)

    logger.info(f"多目标优化结果:")
    logger.info(f"最佳参数: {result.best_params}")
    logger.info(f"最佳得分: {result.best_score:.4f}")
    logger.info(f"评估次数: {len(result.all_results)}")

    return result


def demo_ensemble_optimization():
    """演示集成优化"""
    logger.info("=== 集成优化演示 ===")

    # 创建策略生成器
    generator = AutomaticStrategyGenerator()

    # 生成策略
    config = StrategyConfig(
        template=StrategyTemplate.DEEP_LEARNING,
        market_type=MarketType.A_SHARE,
        symbols=["000001.SZ"],
        timeframes=["1d"],
        risk_level="medium",
        target_return=0.15,
        max_drawdown=0.1
    )

    strategy, initial_params = generator.generate_strategy(config)

    # 创建优化器
    optimizer = StrategyOptimizer()

    # 配置集成优化
    opt_config = OptimizationConfig(
        method=OptimizationMethod.ENSEMBLE,
        max_iterations=60,
        population_size=20,
        ensemble_methods=[
            OptimizationMethod.GENETIC,
            OptimizationMethod.PARTICLE_SWARM,
            OptimizationMethod.SIMULATED_ANNEALING
        ]
    )

    # 运行优化
    result = optimizer.optimize(strategy, objective_function, opt_config)

    logger.info(f"集成优化结果:")
    logger.info(f"最佳参数: {result.best_params}")
    logger.info(f"最佳得分: {result.best_score:.4f}")
    logger.info(f"评估次数: {len(result.all_results)}")

    return result


def demo_optimization_comparison():
    """演示不同优化算法的比较"""
    logger.info("=== 优化算法比较演示 ===")

    # 创建策略生成器
    generator = AutomaticStrategyGenerator()

    # 生成策略
    config = StrategyConfig(
        template=StrategyTemplate.STATISTICAL_ARBITRAGE,
        market_type=MarketType.A_SHARE,
        symbols=["000001.SZ"],
        timeframes=["1d"],
        risk_level="medium",
        target_return=0.15,
        max_drawdown=0.1
    )

    strategy, initial_params = generator.generate_strategy(config)

    # 创建优化器
    optimizer = StrategyOptimizer()

    # 定义要比较的优化方法
    optimization_methods = [
        (OptimizationMethod.GRID_SEARCH, "网格搜索"),
        (OptimizationMethod.GENETIC, "遗传算法"),
        (OptimizationMethod.PARTICLE_SWARM, "粒子群优化"),
        (OptimizationMethod.SIMULATED_ANNEALING, "模拟退火"),
        (OptimizationMethod.ENSEMBLE, "集成优化")
    ]

    results = {}

    for method, name in optimization_methods:
        logger.info(f"运行{name}优化...")

        try:
            opt_config = OptimizationConfig(
                method=method,
                max_iterations=30,
                population_size=20
            )

            result = optimizer.optimize(strategy, objective_function, opt_config)
            results[name] = result

            logger.info(f"{name} - 最佳得分: {result.best_score:.4f}")

        except Exception as e:
            logger.error(f"{name}优化失败: {e}")
            continue

    # 比较结果
    logger.info("\n=== 优化算法比较结果 ===")
    for name, result in results.items():
        logger.info(f"{name}: 得分={result.best_score:.4f}, 评估次数={len(result.all_results)}")

    return results


def main():
    """主函数"""
    logger.info("开始增强策略优化器演示")

    try:
        # 演示各种优化算法
        demo_particle_swarm_optimization()
        print()

        demo_simulated_annealing()
        print()

        demo_multi_objective_optimization()
        print()

        demo_ensemble_optimization()
        print()

        demo_optimization_comparison()
        print()

        logger.info("增强策略优化器演示完成")

    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()

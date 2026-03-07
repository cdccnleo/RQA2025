"""
机器学习集成模块演示脚本

演示策略推荐和自动优化功能
"""

from src.utils.logger import get_logger
from src.trading.ml_integration import (
    SimilarityAnalyzer,
    RecommendationEngine,
    AutoOptimizer,
    HyperparameterTuner,
    MultiObjectiveOptimizer,
    OptimizationEngine
)
import sys
import os
import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


logger = get_logger(__name__)


def generate_sample_strategy_data():
    """生成示例策略数据"""
    strategies = {}

    # 生成多个策略的示例数据
    for i in range(10):
        strategy_id = f"strategy_{i+1}"

        # 生成收益数据
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        returns = np.random.normal(0.001, 0.02, len(dates))
        returns_series = pd.Series(returns, index=dates)

        # 计算风险指标
        volatility = returns_series.std() * np.sqrt(252)
        max_drawdown = (returns_series.cumsum() - returns_series.cumsum().expanding().max()).min()
        sharpe_ratio = returns_series.mean() / returns_series.std() if returns_series.std() > 0 else 0

        # 生成交易数据
        trades_data = pd.DataFrame({
            'date': dates[::5],  # 每5天一次交易
            'volume': np.random.uniform(1000, 10000, len(dates[::5])),
            'pnl': np.random.normal(100, 500, len(dates[::5])),
            'hold_time': np.random.randint(1, 10, len(dates[::5]))
        })
        trades_data.set_index('date', inplace=True)

        strategy_data = {
            'id': strategy_id,
            'name': f'策略{i+1}',
            'category': np.random.choice(['momentum', 'mean_reversion', 'arbitrage']),
            'risk_level': np.random.choice(['low', 'medium', 'high']),
            'returns': returns_series,
            'risk_metrics': {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'var_95': returns_series.quantile(0.05),
                'cvar_95': returns_series[returns_series <= returns_series.quantile(0.05)].mean()
            },
            'trades': trades_data,
            'description': f'这是一个示例策略{i+1}的描述'
        }

        strategies[strategy_id] = strategy_data

    return strategies


def demo_similarity_analysis():
    """演示相似度分析功能"""
    logger.info("=== 相似度分析演示 ===")

    # 生成示例数据
    strategies = generate_sample_strategy_data()

    # 创建相似度分析器
    analyzer = SimilarityAnalyzer()

    # 计算两个策略的相似度
    strategy1 = strategies['strategy_1']
    strategy2 = strategies['strategy_2']

    # 计算综合相似度
    similarity = analyzer.calculate_comprehensive_similarity(strategy1, strategy2)
    logger.info(f"策略1和策略2的综合相似度: {similarity:.4f}")

    # 查找相似策略
    target_strategy = strategies['strategy_1']
    candidate_strategies = [strategies[f'strategy_{i}'] for i in range(2, 11)]

    similar_strategies = analyzer.find_similar_strategies(
        target_strategy, candidate_strategies, top_k=3
    )

    logger.info("与策略1最相似的策略:")
    for strategy_id, similarity_score in similar_strategies:
        logger.info(f"  {strategy_id}: {similarity_score:.4f}")


def demo_strategy_recommendation():
    """演示策略推荐功能"""
    logger.info("=== 策略推荐演示 ===")

    # 生成示例数据
    strategies = generate_sample_strategy_data()

    # 创建推荐引擎
    engine = RecommendationEngine()

    # 添加策略数据
    for strategy_id, strategy_data in strategies.items():
        engine.add_strategy_data(strategy_id, strategy_data)

    # 设置用户偏好
    user_preferences = {
        'risk_preference': 'medium',
        'return_preference': 'moderate',
        'strategy_type_preference': ['momentum', 'mean_reversion']
    }

    # 获取推荐
    recommendations = engine.get_recommendations(
        user_id='user_001',
        target_strategy_id='strategy_1',
        top_k=3,
        method='hybrid'
    )

    logger.info("推荐策略:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec['strategy_info']['name']}")
        logger.info(f"     得分: {rec['score']:.4f}")
        logger.info(f"     相似度: {rec['similarity']:.4f}")
        logger.info(f"     推荐理由: {rec['reasons']}")


def demo_auto_optimization():
    """演示自动优化功能"""
    logger.info("=== 自动优化演示 ===")

    # 创建自动优化器
    optimizer = AutoOptimizer(optimization_method='bayesian', max_iterations=50)

    # 定义目标函数（示例：最大化夏普比率）
    def objective_function(params):
        """示例目标函数"""
        # 模拟策略性能计算
        sharpe_ratio = params.get('param1', 0) * 0.5 + params.get('param2', 0) * 0.3
        return sharpe_ratio

    # 设置参数边界
    parameter_bounds = {
        'param1': (0.1, 1.0),
        'param2': (0.01, 0.5),
        'param3': (10, 100)
    }

    # 设置目标函数和参数边界
    optimizer.set_objective_function(objective_function)
    optimizer.set_parameter_bounds(parameter_bounds)

    # 执行优化
    best_params, best_score = optimizer.optimize()

    logger.info(f"最优参数: {best_params}")
    logger.info(f"最优得分: {best_score:.4f}")


def demo_hyperparameter_tuning():
    """演示超参数调优功能"""
    logger.info("=== 超参数调优演示 ===")

    # 创建超参数调优器
    tuner = HyperparameterTuner(optimization_method='random', max_iterations=30)

    # 定义参数空间
    param_space = {
        'learning_rate': [0.01, 0.1, 0.5],
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200]
    }

    # 生成示例数据
    X = pd.DataFrame(np.random.randn(100, 5))
    y = pd.Series(np.random.randn(100))

    # 执行调优
    best_params, best_score = tuner.tune_hyperparameters(
        estimator=None,  # 实际应用中需要真实的估计器
        param_space=param_space,
        X=X,
        y=y
    )

    logger.info(f"最优超参数: {best_params}")
    logger.info(f"最优得分: {best_score:.4f}")


def demo_multi_objective_optimization():
    """演示多目标优化功能"""
    logger.info("=== 多目标优化演示 ===")

    # 创建多目标优化器
    optimizer = MultiObjectiveOptimizer(optimization_method='weighted_sum')

    # 定义目标函数
    def objective1(params):
        """收益目标"""
        return params.get('param1', 0) * 0.6 + params.get('param2', 0) * 0.4

    def objective2(params):
        """风险目标（负值）"""
        return -(params.get('param1', 0) * 0.3 + params.get('param2', 0) * 0.7)

    # 添加目标函数
    optimizer.add_objective(objective1)
    optimizer.add_objective(objective2)

    # 定义参数边界
    parameter_bounds = {
        'param1': (0.1, 1.0),
        'param2': (0.01, 0.5)
    }

    # 执行优化
    solutions = optimizer.optimize(parameter_bounds)

    logger.info(f"找到 {len(solutions)} 个帕累托最优解")
    for i, solution in enumerate(solutions, 1):
        logger.info(f"  解{i}: 参数={solution['parameters']}, 目标值={solution['objective_values']}")


def demo_optimization_engine():
    """演示优化引擎功能"""
    logger.info("=== 优化引擎演示 ===")

    # 创建优化引擎
    engine = OptimizationEngine()

    # 定义策略函数
    def strategy_function(params):
        """示例策略函数"""
        return params.get('param1', 0) * 0.5 + params.get('param2', 0) * 0.3

    # 定义参数边界
    parameter_bounds = {
        'param1': (0.1, 1.0),
        'param2': (0.01, 0.5)
    }

    # 执行单目标优化
    result = engine.optimize_strategy_parameters(
        strategy_function, parameter_bounds, 'single_objective'
    )

    logger.info(f"优化结果: {result}")

    # 生成示例收益数据
    returns_data = pd.DataFrame({
        'asset1': np.random.normal(0.001, 0.02, 252),
        'asset2': np.random.normal(0.001, 0.015, 252),
        'asset3': np.random.normal(0.001, 0.025, 252)
    })

    # 执行投资组合优化
    portfolio_result = engine.optimize_portfolio_weights(
        returns_data, 'mean_variance'
    )

    logger.info(f"投资组合优化结果: {portfolio_result}")


def main():
    """主函数"""
    logger.info("开始机器学习集成模块演示")

    try:
        # 演示相似度分析
        demo_similarity_analysis()

        # 演示策略推荐
        demo_strategy_recommendation()

        # 演示自动优化
        demo_auto_optimization()

        # 演示超参数调优
        demo_hyperparameter_tuning()

        # 演示多目标优化
        demo_multi_objective_optimization()

        # 演示优化引擎
        demo_optimization_engine()

        logger.info("机器学习集成模块演示完成")

    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()

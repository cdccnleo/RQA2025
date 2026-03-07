"""
自动策略生成演示脚本

展示策略工作台的完整功能，包括：
- 自动策略生成
- 策略优化
- 策略模拟
- 策略分析
- 策略存储
"""

from src.utils.logger import get_logger
from src.trading.strategy_workspace import (
    AutomaticStrategyGenerator, StrategyConfig, StrategyTemplate, MarketType,
    StrategyOptimizer, OptimizationMethod, OptimizationConfig,
    StrategySimulator, SimulationMode, SimulationConfig,
    StrategyAnalyzer, StrategyStore
)
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


logger = get_logger(__name__)


def generate_sample_market_data(symbol: str = "000001.SZ", days: int = 252) -> pd.DataFrame:
    """生成示例市场数据"""
    np.random.seed(42)

    # 生成日期序列
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # 生成价格数据
    initial_price = 10.0
    returns = np.random.normal(0.0005, 0.02, len(dates))  # 日收益率
    prices = [initial_price]

    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(new_price)

    # 创建DataFrame
    data = pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)

    data.index.name = symbol
    return data


def objective_function(strategy) -> float:
    """目标函数 - 用于策略优化"""
    # 这里应该实现真正的策略评估逻辑
    # 简化实现，返回一个随机分数
    return np.random.uniform(0.5, 1.5)


def demo_automatic_strategy_generation():
    """演示自动策略生成"""
    logger.info("=== 开始自动策略生成演示 ===")

    # 1. 创建策略生成器
    generator = AutomaticStrategyGenerator()

    # 2. 配置策略参数
    config = StrategyConfig(
        template=StrategyTemplate.MOVING_AVERAGE,
        market_type=MarketType.A_SHARE,
        symbols=["000001.SZ"],
        timeframes=["1d"],
        risk_level="medium",
        target_return=0.15,
        max_drawdown=0.2
    )

    # 3. 生成策略
    logger.info("正在生成策略...")
    strategy, parameters = generator.generate_strategy(config)

    logger.info(f"策略生成成功！")
    logger.info(f"参数: {parameters}")

    # 4. 可视化策略
    logger.info("策略结构:")
    strategy.visualize()

    return strategy, parameters


def demo_strategy_optimization(strategy, parameters):
    """演示策略优化"""
    logger.info("\n=== 开始策略优化演示 ===")

    # 1. 创建优化器
    optimizer = StrategyOptimizer()

    # 2. 配置优化参数
    opt_config = OptimizationConfig(
        method=OptimizationMethod.GENETIC,
        max_iterations=20,
        population_size=10,
        elite_size=2,
        mutation_rate=0.1,
        crossover_rate=0.7
    )

    # 3. 执行优化
    logger.info("正在优化策略参数...")
    result = optimizer.optimize(strategy, objective_function, opt_config)

    logger.info(f"优化完成！")
    logger.info(f"最佳参数: {result.best_params}")
    logger.info(f"最佳分数: {result.best_score:.4f}")

    return result


def demo_strategy_simulation(strategy, parameters):
    """演示策略模拟"""
    logger.info("\n=== 开始策略模拟演示 ===")

    # 1. 生成市场数据
    logger.info("生成市场数据...")
    market_data = generate_sample_market_data()

    # 2. 创建模拟器
    simulator = StrategySimulator()

    # 3. 配置模拟参数
    sim_config = SimulationConfig(
        mode=SimulationMode.BACKTEST,
        start_date=market_data.index[0],
        end_date=market_data.index[-1],
        initial_capital=100000.0,
        commission_rate=0.0003,
        slippage=0.0001,
        risk_free_rate=0.03
    )

    # 4. 执行模拟
    logger.info("正在执行策略模拟...")
    result = simulator.simulate(strategy, market_data, sim_config)

    logger.info(f"模拟完成！")
    logger.info(f"总收益率: {result.total_return:.2%}")
    logger.info(f"年化收益率: {result.annualized_return:.2%}")
    logger.info(f"夏普比率: {result.sharpe_ratio:.3f}")
    logger.info(f"最大回撤: {result.max_drawdown:.2%}")
    logger.info(f"胜率: {result.win_rate:.2%}")

    return result


def demo_strategy_analysis(result):
    """演示策略分析"""
    logger.info("\n=== 开始策略分析演示 ===")

    # 1. 创建分析器
    analyzer = StrategyAnalyzer()

    # 2. 分析绩效
    logger.info("分析策略绩效...")
    performance = analyzer.analyze_performance(result)

    logger.info("绩效指标:")
    logger.info(f"  Sortino比率: {performance.sortino_ratio:.3f}")
    logger.info(f"  Calmar比率: {performance.calmar_ratio:.3f}")
    logger.info(f"  VaR(95%): {performance.var_95:.4f}")
    logger.info(f"  CVaR(95%): {performance.cvar_95:.4f}")

    # 3. 分析风险
    logger.info("分析策略风险...")
    risk = analyzer.analyze_risk(result)

    logger.info("风险指标:")
    logger.info(f"  波动率: {risk.volatility:.2%}")
    logger.info(f"  下行偏差: {risk.downside_deviation:.2%}")
    logger.info(f"  Beta: {risk.beta:.3f}")
    logger.info(f"  Alpha: {risk.alpha:.4f}")

    # 4. 分析交易
    logger.info("分析交易记录...")
    trade_analysis = analyzer.analyze_trades(result)

    logger.info("交易分析:")
    logger.info(f"  总交易次数: {trade_analysis.total_trades}")
    logger.info(f"  盈利交易: {trade_analysis.winning_trades}")
    logger.info(f"  亏损交易: {trade_analysis.losing_trades}")
    logger.info(f"  平均盈利: {trade_analysis.avg_win:.2f}")
    logger.info(f"  平均亏损: {trade_analysis.avg_loss:.2f}")
    logger.info(f"  交易频率: {trade_analysis.trade_frequency:.2f} 次/天")

    # 5. 生成报告
    logger.info("生成绩效报告...")
    report = analyzer.generate_performance_report(result)

    logger.info("绩效报告摘要:")
    for key, value in report['summary'].items():
        logger.info(f"  {key}: {value}")

    return report


def demo_strategy_storage(strategy, parameters, result):
    """演示策略存储"""
    logger.info("\n=== 开始策略存储演示 ===")

    # 1. 创建存储组件
    store = StrategyStore()

    # 2. 创建策略
    logger.info("创建策略记录...")
    strategy_id = store.create_strategy(
        name="自动生成移动平均策略",
        description="基于移动平均线的自动生成策略",
        author="AI Assistant",
        market_type="a_share",
        risk_level="medium",
        tags=["移动平均", "自动生成", "A股"]
    )

    # 3. 保存策略
    logger.info("保存策略版本...")
    version_id = store.save_strategy(
        strategy_id=strategy_id,
        strategy=strategy,
        parameters=parameters,
        version_notes="初始版本"
    )

    # 4. 保存模拟结果
    logger.info("保存模拟结果...")
    result_id = store.save_simulation_result(strategy_id, version_id, result)

    # 5. 列出策略
    logger.info("列出所有策略...")
    strategies = store.list_strategies()
    logger.info(f"总策略数: {len(strategies)}")

    for strategy_info in strategies:
        logger.info(f"  - {strategy_info['name']} ({strategy_info['strategy_id']})")

    # 6. 搜索策略
    logger.info("搜索策略...")
    search_results = store.search_strategies(
        query="移动平均",
        tags=["自动生成"],
        market_type="a_share"
    )

    logger.info(f"搜索结果: {len(search_results)} 个策略")

    # 7. 获取统计信息
    logger.info("获取统计信息...")
    stats = store.get_strategy_statistics()

    logger.info("策略统计:")
    logger.info(f"  总策略数: {stats['total_strategies']}")
    logger.info(f"  按状态分布: {stats['by_status']}")
    logger.info(f"  按市场类型分布: {stats['by_market_type']}")
    logger.info(f"  按风险等级分布: {stats['by_risk_level']}")

    return strategy_id, version_id, result_id


def demo_complete_workflow():
    """演示完整工作流程"""
    logger.info("=== 自动策略生成完整工作流程演示 ===")

    try:
        # 1. 自动策略生成
        strategy, parameters = demo_automatic_strategy_generation()

        # 2. 策略优化
        opt_result = demo_strategy_optimization(strategy, parameters)

        # 3. 策略模拟
        sim_result = demo_strategy_simulation(strategy, parameters)

        # 4. 策略分析
        analysis_report = demo_strategy_analysis(sim_result)

        # 5. 策略存储
        strategy_id, version_id, result_id = demo_strategy_storage(strategy, parameters, sim_result)

        logger.info("\n=== 演示完成 ===")
        logger.info(f"策略ID: {strategy_id}")
        logger.info(f"版本ID: {version_id}")
        logger.info(f"结果ID: {result_id}")

        return {
            'strategy_id': strategy_id,
            'version_id': version_id,
            'result_id': result_id,
            'strategy': strategy,
            'parameters': parameters,
            'optimization_result': opt_result,
            'simulation_result': sim_result,
            'analysis_report': analysis_report
        }

    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


def main():
    """主函数"""
    try:
        # 设置日志级别
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 运行完整演示
        results = demo_complete_workflow()

        logger.info("自动策略生成演示成功完成！")

        # 返回结果供后续使用
        return results

    except Exception as e:
        logger.error(f"演示失败: {e}")
        raise


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
增强策略分析器演示脚本

展示策略分析器的新功能：
- 高级风险分析
- 交易行为分析
- 市场微观结构分析
- 策略归因分析
- 实时监控指标
"""

from src.trading.strategy_workspace.strategy_generator import AutomaticStrategyGenerator, StrategyConfig, StrategyTemplate, MarketType
from src.trading.strategy_workspace.simulator import StrategySimulator, SimulationConfig, SimulationMode
from src.trading.strategy_workspace.analyzer import StrategyAnalyzer
import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_data():
    """创建示例数据"""
    logger.info("创建示例市场数据...")

    # 生成模拟市场数据
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)

    # 生成价格数据
    price_data = []
    current_price = 100.0

    for date in dates:
        # 添加趋势和随机波动
        daily_return = 0.0005 + np.random.normal(0, 0.02)  # 年化约12%收益，20%波动率
        current_price *= (1 + daily_return)
        price_data.append({
            'date': date,
            'open': current_price * (1 + np.random.normal(0, 0.005)),
            'high': current_price * (1 + abs(np.random.normal(0, 0.01))),
            'low': current_price * (1 - abs(np.random.normal(0, 0.01))),
            'close': current_price,
            'volume': np.random.randint(1000000, 5000000)
        })

    market_data = pd.DataFrame(price_data)
    market_data.set_index('date', inplace=True)

    logger.info(f"创建了 {len(market_data)} 天的市场数据")
    return market_data


def demo_advanced_risk_analysis():
    """演示高级风险分析"""
    logger.info("=== 演示高级风险分析 ===")

    try:
        # 创建分析器
        analyzer = StrategyAnalyzer()

        # 创建示例策略和模拟结果
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
        strategy, _ = generator.generate_strategy(config)

        simulator = StrategySimulator()
        market_data = create_sample_data()

        config = SimulationConfig(
            mode=SimulationMode.BACKTEST,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            commission_rate=0.001,
            slippage=0.0005
        )

        result = simulator.simulate(strategy, market_data, config)

        # 执行高级风险分析
        advanced_risk = analyzer.analyze_advanced_risk(result)

        logger.info("高级风险分析结果:")
        logger.info(f"  99% VaR: {advanced_risk.var_99:.4f}")
        logger.info(f"  99% CVaR: {advanced_risk.cvar_99:.4f}")
        logger.info(f"  期望损失: {advanced_risk.expected_shortfall:.4f}")
        logger.info(f"  尾部风险: {advanced_risk.tail_risk:.4f}")
        logger.info(f"  风险等级: {advanced_risk.risk_level.value}")

        logger.info("压力测试结果:")
        for scenario, result in advanced_risk.stress_test_results.items():
            logger.info(f"  {scenario}: {result:.4f}")

        logger.info("场景分析结果:")
        for scenario, result in advanced_risk.scenario_analysis.items():
            logger.info(f"  {scenario}: {result:.4f}")

        return True

    except Exception as e:
        logger.error(f"高级风险分析演示失败: {e}")
        return False


def demo_trade_behavior_analysis():
    """演示交易行为分析"""
    logger.info("=== 演示交易行为分析 ===")

    try:
        # 创建分析器
        analyzer = StrategyAnalyzer()

        # 创建示例策略和模拟结果
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
        strategy, _ = generator.generate_strategy(config)

        simulator = StrategySimulator()
        market_data = create_sample_data()

        config = SimulationConfig(
            mode=SimulationMode.BACKTEST,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            commission_rate=0.001,
            slippage=0.0005
        )

        result = simulator.simulate(strategy, market_data, config)

        # 执行交易行为分析
        trade_behavior = analyzer.analyze_trade_behavior(result)

        logger.info("交易行为分析结果:")
        logger.info(f"  市场冲击: {trade_behavior.market_impact:.6f}")

        logger.info("交易模式:")
        for pattern, value in trade_behavior.trade_patterns.items():
            if isinstance(value, dict):
                logger.info(f"  {pattern}:")
                for k, v in value.items():
                    logger.info(f"    {k}: {v}")
            else:
                logger.info(f"  {pattern}: {value}")

        logger.info(f"  异常交易数量: {len(trade_behavior.anomaly_trades)}")

        logger.info("交易聚类:")
        for cluster, value in trade_behavior.trade_clustering.items():
            if isinstance(value, dict):
                logger.info(f"  {cluster}:")
                for k, v in value.items():
                    logger.info(f"    {k}: {v}")
            else:
                logger.info(f"  {cluster}: {value}")

        logger.info("滑点分析:")
        for metric, value in trade_behavior.slippage_analysis.items():
            logger.info(f"  {metric}: {value:.4f}")

        logger.info("执行质量:")
        for metric, value in trade_behavior.execution_quality.items():
            logger.info(f"  {metric}: {value:.4f}")

        return True

    except Exception as e:
        logger.error(f"交易行为分析演示失败: {e}")
        return False


def demo_market_microstructure_analysis():
    """演示市场微观结构分析"""
    logger.info("=== 演示市场微观结构分析 ===")

    try:
        # 创建分析器
        analyzer = StrategyAnalyzer()

        # 创建示例策略和模拟结果
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
        strategy, _ = generator.generate_strategy(config)

        simulator = StrategySimulator()
        market_data = create_sample_data()

        config = SimulationConfig(
            mode=SimulationMode.BACKTEST,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            commission_rate=0.001,
            slippage=0.0005
        )

        result = simulator.simulate(strategy, market_data, config)

        # 执行市场微观结构分析
        microstructure = analyzer.analyze_market_microstructure(result)

        logger.info("市场微观结构分析结果:")
        logger.info(f"  买卖价差: {microstructure.bid_ask_spread:.4f}")
        logger.info(f"  订单流不平衡: {microstructure.order_flow_imbalance:.4f}")
        logger.info(f"  价格冲击: {microstructure.price_impact:.4f}")
        logger.info(f"  市场效率: {microstructure.market_efficiency:.2f}")

        logger.info("市场深度:")
        for level, depth in microstructure.market_depth.items():
            logger.info(f"  {level}: {depth:,.0f}")

        logger.info("流动性指标:")
        for metric, value in microstructure.liquidity_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return True

    except Exception as e:
        logger.error(f"市场微观结构分析演示失败: {e}")
        return False


def demo_strategy_attribution_analysis():
    """演示策略归因分析"""
    logger.info("=== 演示策略归因分析 ===")

    try:
        # 创建分析器
        analyzer = StrategyAnalyzer()

        # 创建示例策略和模拟结果
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
        strategy, _ = generator.generate_strategy(config)

        simulator = StrategySimulator()
        market_data = create_sample_data()

        config = SimulationConfig(
            mode=SimulationMode.BACKTEST,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            commission_rate=0.001,
            slippage=0.0005
        )

        result = simulator.simulate(strategy, market_data, config)

        # 执行策略归因分析
        attribution = analyzer.analyze_strategy_attribution(result)

        logger.info("策略归因分析结果:")

        logger.info("因子贡献:")
        for factor, contribution in attribution.factor_contributions.items():
            logger.info(f"  {factor}: {contribution:.2f}")

        logger.info("行业配置:")
        for sector, allocation in attribution.sector_allocations.items():
            logger.info(f"  {sector}: {allocation:.2f}")

        logger.info("风格分析:")
        for style, weight in attribution.style_analysis.items():
            logger.info(f"  {style}: {weight:.2f}")

        logger.info("风险分解:")
        for risk_type, weight in attribution.risk_decomposition.items():
            logger.info(f"  {risk_type}: {weight:.2f}")

        logger.info("绩效归因:")
        for component, contribution in attribution.performance_attribution.items():
            logger.info(f"  {component}: {contribution:.2f}")

        return True

    except Exception as e:
        logger.error(f"策略归因分析演示失败: {e}")
        return False


def demo_real_time_monitoring():
    """演示实时监控指标"""
    logger.info("=== 演示实时监控指标 ===")

    try:
        # 创建分析器
        analyzer = StrategyAnalyzer()

        # 创建示例策略和模拟结果
        generator = AutomaticStrategyGenerator()
        config = StrategyConfig(
            template=StrategyTemplate.MEAN_REVERSION_ENHANCED,
            market_type=MarketType.A_SHARE,
            symbols=["000001.SZ"],
            timeframes=["1d"],
            risk_level="medium",
            target_return=0.15,
            max_drawdown=0.1
        )
        strategy, _ = generator.generate_strategy(config)

        simulator = StrategySimulator()
        market_data = create_sample_data()

        config = SimulationConfig(
            mode=SimulationMode.BACKTEST,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            commission_rate=0.001,
            slippage=0.0005
        )

        result = simulator.simulate(strategy, market_data, config)

        # 获取实时监控指标
        monitoring = analyzer.get_real_time_monitoring_metrics(result)

        logger.info("实时监控指标:")
        logger.info(f"  当前回撤: {monitoring.current_drawdown:.2%}")
        logger.info(f"  滚动夏普比率: {monitoring.rolling_sharpe:.3f}")
        logger.info(f"  滚动波动率: {monitoring.rolling_volatility:.2%}")
        logger.info(f"  持仓集中度: {monitoring.position_concentration:.2%}")

        logger.info("敞口指标:")
        for exposure, value in monitoring.exposure_metrics.items():
            logger.info(f"  {exposure}: {value:.2f}")

        logger.info("风险警报:")
        if monitoring.risk_alerts:
            for alert in monitoring.risk_alerts:
                logger.info(f"  ⚠️ {alert}")
        else:
            logger.info("  ✅ 无风险警报")

        return True

    except Exception as e:
        logger.error(f"实时监控指标演示失败: {e}")
        return False


def demo_comprehensive_analysis():
    """演示综合分析"""
    logger.info("=== 演示综合分析 ===")

    try:
        # 创建分析器
        analyzer = StrategyAnalyzer()

        # 创建示例策略和模拟结果
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
        strategy, _ = generator.generate_strategy(config)

        simulator = StrategySimulator()
        market_data = create_sample_data()

        config = SimulationConfig(
            mode=SimulationMode.BACKTEST,
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            initial_capital=100000,
            commission_rate=0.001,
            slippage=0.0005
        )

        result = simulator.simulate(strategy, market_data, config)

        # 执行所有分析
        logger.info("执行综合分析...")

        # 基础分析
        performance = analyzer.analyze_performance(result)
        risk = analyzer.analyze_risk(result)
        trades = analyzer.analyze_trades(result)

        # 高级分析
        advanced_risk = analyzer.analyze_advanced_risk(result)
        trade_behavior = analyzer.analyze_trade_behavior(result)
        microstructure = analyzer.analyze_market_microstructure(result)
        attribution = analyzer.analyze_strategy_attribution(result)
        monitoring = analyzer.get_real_time_monitoring_metrics(result)

        # 生成综合报告
        report = analyzer.generate_performance_report(result)

        logger.info("综合分析完成!")
        logger.info(f"策略总收益率: {performance.total_return:.2%}")
        logger.info(f"年化收益率: {performance.annualized_return:.2%}")
        logger.info(f"夏普比率: {performance.sharpe_ratio:.3f}")
        logger.info(f"最大回撤: {performance.max_drawdown:.2%}")
        logger.info(f"胜率: {performance.win_rate:.2%}")
        logger.info(f"风险等级: {advanced_risk.risk_level.value}")
        logger.info(f"总交易次数: {trades.total_trades}")

        return True

    except Exception as e:
        logger.error(f"综合分析演示失败: {e}")
        return False


def main():
    """主函数"""
    logger.info("开始增强策略分析器演示...")

    # 运行各个演示
    demos = [
        ("高级风险分析", demo_advanced_risk_analysis),
        ("交易行为分析", demo_trade_behavior_analysis),
        ("市场微观结构分析", demo_market_microstructure_analysis),
        ("策略归因分析", demo_strategy_attribution_analysis),
        ("实时监控指标", demo_real_time_monitoring),
        ("综合分析", demo_comprehensive_analysis)
    ]

    success_count = 0
    total_count = len(demos)

    for demo_name, demo_func in demos:
        logger.info(f"\n{'='*50}")
        logger.info(f"运行演示: {demo_name}")
        logger.info(f"{'='*50}")

        try:
            if demo_func():
                success_count += 1
                logger.info(f"✅ {demo_name} 演示成功")
            else:
                logger.error(f"❌ {demo_name} 演示失败")
        except Exception as e:
            logger.error(f"❌ {demo_name} 演示异常: {e}")

    logger.info(f"\n{'='*50}")
    logger.info("演示总结")
    logger.info(f"{'='*50}")
    logger.info(f"成功: {success_count}/{total_count}")
    logger.info(f"成功率: {success_count/total_count*100:.1f}%")

    if success_count == total_count:
        logger.info("🎉 所有演示都成功完成!")
    else:
        logger.warning("⚠️ 部分演示失败，请检查错误信息")

    logger.info("增强策略分析器演示结束")


if __name__ == "__main__":
    main()

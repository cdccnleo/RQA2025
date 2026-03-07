#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级分析功能使用示例
"""

import numpy as np
from datetime import datetime, timedelta
import logging

from src.backtest.advanced_analytics import (
    AdvancedAnalyticsEngine,
    FactorData
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("开始高级分析功能演示")

    # 初始化引擎
    engine = AdvancedAnalyticsEngine()

    # 添加因子数据
    for i in range(20):
        for symbol in ['000001.SZ', '000002.SZ']:
            for factor in ['momentum', 'value', 'size']:
                factor_data = FactorData(
                    timestamp=datetime.now() - timedelta(days=20-i),
                    symbol=symbol,
                    factor_name=factor,
                    factor_value=0.5 + np.random.normal(0, 0.1)
                )
                engine.add_factor_data(factor_data)

    # 计算因子暴露度
    for symbol in ['000001.SZ', '000002.SZ']:
        for factor in ['momentum', 'value', 'size']:
            exposure = engine.multi_factor_analyzer.calculate_factor_exposure(symbol, factor)
            logger.info(f"{symbol} - {factor}: {exposure:.4f}")

    # 计算自定义指标
    returns = [0.01, 0.02, -0.015, 0.025, 0.02]
    portfolio_values = [100, 110, 105, 95, 100, 115]

    metrics = engine.calculate_custom_metrics(returns, portfolio_values)
    logger.info("自定义指标:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")

    logger.info("✓ 高级分析功能演示完成")


if __name__ == "__main__":
    main()

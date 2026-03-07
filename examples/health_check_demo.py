#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQA2025 健康检查模块演示
"""

import asyncio
import logging
from src.infrastructure.health import get_enhanced_health_checker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """主函数"""
    logger.info("开始健康检查模块演示...")

    # 创建增强健康检查器
    checker = get_enhanced_health_checker()

    # 执行健康检查
    result = await checker.perform_health_check("demo", "liveness")
    logger.info(f"健康检查结果: {result.status}")

    # 获取性能报告
    report = checker.get_performance_report()
    logger.info(f"性能报告: {len(report.get('metrics_summary', {}))} 个指标")

    # 获取告警摘要
    alerts = checker.get_alert_summary()
    logger.info(f"活跃告警: {alerts.get('active_alerts_count', 0)} 个")

    logger.info("演示完成!")

if __name__ == "__main__":
    asyncio.run(main())

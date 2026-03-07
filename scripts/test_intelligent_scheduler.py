#!/usr/bin/env python3
"""
智能调度器测试脚本
验证P1阶段智能调度功能
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_market_monitor():
    """测试市场状态监控器"""
    logger.info("=== 测试市场状态监控器 ===")

    try:
        from src.core.orchestration.market_adaptive_monitor import get_market_adaptive_monitor

        monitor = get_market_adaptive_monitor()
        regime_analysis = await monitor.get_current_regime()

        logger.info(f"当前市场状态: {regime_analysis.current_regime.value}")
        logger.info(f"置信度: {regime_analysis.confidence:.2f}")
        logger.info(f"推荐行动: {regime_analysis.recommended_actions}")

        # 获取统计信息
        stats = monitor.get_regime_statistics()
        logger.info(f"市场状态统计: {stats}")

        return True

    except Exception as e:
        logger.error(f"市场状态监控器测试失败: {e}")
        return False


async def test_data_priority_manager():
    """测试数据优先级管理器"""
    logger.info("=== 测试数据优先级管理器 ===")

    try:
        from src.core.orchestration.data_priority_manager import get_data_priority_manager

        manager = get_data_priority_manager()

        # 测试不同数据源的优先级
        test_sources = [
            '000001',  # 上证指数 - 应该识别为核心指数
            '000858',  # 五粮液 - 核心股票
            '600036',  # 招商银行 - 核心股票
            'sh000001',  # 上证指数代码
            'macro_gdp',  # 宏观数据
            'news_finance'  # 新闻数据
        ]

        for source_id in test_sources:
            priority = manager.get_data_priority(source_id)
            score = manager.calculate_task_priority_score(source_id)
            logger.info(f"数据源 {source_id}: 优先级={priority.priority_level}, 得分={score}")

        return True

    except Exception as e:
        logger.error(f"数据优先级管理器测试失败: {e}")
        return False


async def test_incremental_strategy():
    """测试增量采集策略"""
    logger.info("=== 测试增量采集策略 ===")

    try:
        from src.core.orchestration.incremental_collection_strategy import get_incremental_collection_strategy

        strategy = get_incremental_collection_strategy()

        # 测试不同数据源的采集策略
        test_sources = [
            ('000001', 'stock'),  # 核心股票
            ('sh000001', 'index'),  # 主要指数
            ('normal_stock', 'stock'),  # 普通股票
            ('macro_data', 'macro')  # 宏观数据
        ]

        for source_id, data_type in test_sources:
            window = strategy.determine_collection_strategy(source_id, data_type)
            logger.info(f"数据源 {source_id} 采集策略: 模式={window.mode}, 优先级={window.priority}")
            logger.info(f"  时间窗口: {window.start_date} 至 {window.end_date}")

        return True

    except Exception as e:
        logger.error(f"增量采集策略测试失败: {e}")
        return False


async def test_intelligent_scheduler():
    """测试智能调度器"""
    logger.info("=== 测试智能调度器 ===")

    try:
        from src.core.orchestration.business_process.service_scheduler import get_data_collection_scheduler

        scheduler = get_data_collection_scheduler()

        # 获取智能调度状态
        status = await scheduler.get_intelligent_scheduling_status()

        logger.info("智能调度器状态:")
        logger.info(f"  市场状态: {status['market_regime']['current']}")
        logger.info(f"  置信度: {status['market_regime']['confidence']}")
        logger.info(f"  并发任务数: {status['scheduler_adjustments']['max_concurrent_tasks']}")
        logger.info(f"  检查间隔: {status['scheduler_adjustments']['check_interval']}秒")
        logger.info(f"  优先级管理: {status['priority_manager']['active']}")

        return True

    except Exception as e:
        logger.error(f"智能调度器测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    logger.info("开始智能调度器功能测试")
    logger.info("=" * 60)

    test_results = []

    # 测试各个组件
    test_functions = [
        test_market_monitor,
        test_data_priority_manager,
        test_incremental_strategy,
        test_intelligent_scheduler
    ]

    for test_func in test_functions:
        try:
            result = await test_func()
            test_results.append(result)
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"{test_func.__name__}: {status}")
        except Exception as e:
            logger.error(f"{test_func.__name__} 执行异常: {e}")
            test_results.append(False)

        logger.info("-" * 40)

    # 汇总结果
    passed = sum(test_results)
    total = len(test_results)

    logger.info("=" * 60)
    logger.info(f"测试完成: {passed}/{total} 通过")

    if passed == total:
        logger.info("🎉 所有测试通过！P1阶段智能调度功能正常")
    else:
        logger.warning(f"⚠️  {total - passed} 个测试失败，需要检查")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
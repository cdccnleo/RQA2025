#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中期优化执行脚本
演示中期优化建议的实现
"""

from src.core.optimizations.optimization_implementer import (
    OptimizationImplementer, OptimizationPhase
)
import sys
import logging
from pathlib import Path
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    logger.info("开始执行中期优化建议实现")

    # 创建优化实现器
    implementer = OptimizationImplementer()

    # 先执行短期优化任务（如果未完成）
    logger.info("=" * 50)
    logger.info("检查并执行短期优化任务")
    logger.info("=" * 50)

    short_term_summary = implementer.get_optimization_summary()
    if short_term_summary['phase_summary']['short_term']['completed'] < 5:
        logger.info("执行短期优化任务...")
        short_term_results = implementer.execute_optimizations(OptimizationPhase.SHORT_TERM)
        logger.info(
            f"短期优化完成: {short_term_results['completed_tasks']}/{short_term_results['total_tasks']} 成功")
    else:
        logger.info("短期优化任务已完成")

    # 显示优化摘要
    summary = implementer.get_optimization_summary()
    logger.info(f"优化摘要: {summary}")

    # 执行中期优化
    logger.info("=" * 50)
    logger.info("执行中期优化任务")
    logger.info("=" * 50)

    medium_term_results = implementer.execute_optimizations(OptimizationPhase.MEDIUM_TERM)
    logger.info(
        f"中期优化完成: {medium_term_results['completed_tasks']}/{medium_term_results['total_tasks']} 成功")

    # 显示任务结果
    for result in medium_term_results['results']:
        if result['status'] == 'completed':
            logger.info(f"✓ {result['task_id']}: {result.get('results', {})}")
        else:
            logger.error(f"✗ {result['task_id']}: {result.get('error', '未知错误')}")

    # 显示分布式支持详情
    logger.info("=" * 50)
    logger.info("分布式支持详情")
    logger.info("=" * 50)

    distributed_support = implementer.distributed_support
    if distributed_support:
        services = distributed_support.discover_services()
        logger.info(f"发现的服务: {services}")

        for node_id, node in distributed_support.nodes.items():
            logger.info(f"节点 {node_id}: {node.host}:{node.port} - {node.status}")

    # 显示多级缓存详情
    logger.info("=" * 50)
    logger.info("多级缓存详情")
    logger.info("=" * 50)

    multi_level_cache = implementer.multi_level_cache
    if multi_level_cache:
        cache_stats = multi_level_cache.get_cache_stats()
        logger.info(f"缓存统计: {cache_stats}")

        # 测试缓存功能
        test_key = "performance_test"
        test_value = {"data": "test", "timestamp": time.time()}

        # 设置缓存
        multi_level_cache.set(test_key, test_value, "L1")
        logger.info(f"设置缓存: {test_key} -> L1")

        # 获取缓存
        cached_value = multi_level_cache.get(test_key)
        logger.info(f"获取缓存: {test_key} -> {cached_value is not None}")

        # 更新统计
        updated_stats = multi_level_cache.get_cache_stats()
        logger.info(f"更新后的缓存统计: {updated_stats}")

    # 显示监控增强详情
    logger.info("=" * 50)
    logger.info("监控增强详情")
    logger.info("=" * 50)

    monitoring_enhancer = implementer.monitoring_enhancer
    if monitoring_enhancer:
        # 添加更多指标
        monitoring_enhancer.add_metric("response_time", 125.5, "application")
        monitoring_enhancer.add_metric("throughput", 1500.0, "application")

        metrics_summary = monitoring_enhancer.get_metrics_summary()
        logger.info(f"指标摘要: {metrics_summary}")

        active_alerts = monitoring_enhancer.get_active_alerts()
        logger.info(f"活跃告警: {len(active_alerts)}")

        dashboards = monitoring_enhancer.dashboards
        logger.info(f"仪表板: {list(dashboards.keys())}")

    # 显示性能调优详情
    logger.info("=" * 50)
    logger.info("性能调优详情")
    logger.info("=" * 50)

    performance_tuner = implementer.performance_tuner
    if performance_tuner:
        performance_summary = performance_tuner.get_performance_summary()
        logger.info(f"性能摘要: {performance_summary}")

        if performance_tuner.performance_history:
            latest_analysis = performance_tuner.performance_history[-1]["analysis"]
            logger.info(f"最新性能分析: {latest_analysis}")

    # 显示最终摘要
    logger.info("=" * 50)
    logger.info("中期优化执行完成")
    logger.info("=" * 50)

    final_summary = implementer.get_optimization_summary()
    logger.info(f"总任务数: {final_summary['total_tasks']}")
    logger.info(f"已完成: {final_summary['completed_tasks']}")
    logger.info(f"失败: {final_summary['failed_tasks']}")
    logger.info(f"完成率: {final_summary['completion_rate']:.2%}")

    # 显示各阶段完成情况
    for phase, stats in final_summary['phase_summary'].items():
        completion_rate = stats['completed'] / stats['total'] if stats['total'] > 0 else 0
        logger.info(f"{phase}: {stats['completed']}/{stats['total']} ({completion_rate:.2%})")

    logger.info("中期优化建议实现演示完成")


if __name__ == "__main__":
    main()

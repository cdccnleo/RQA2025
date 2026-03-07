#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
优化执行脚本
演示短期、中期和长期优化建议的实现
"""

from src.core.optimizations.optimization_implementer import (
    OptimizationImplementer, OptimizationPhase
)
import sys
import logging
from pathlib import Path

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
    logger.info("开始执行优化建议实现")

    # 创建优化实现器
    implementer = OptimizationImplementer()

    # 显示优化摘要
    summary = implementer.get_optimization_summary()
    logger.info(f"优化摘要: {summary}")

    # 执行短期优化
    logger.info("=" * 50)
    logger.info("执行短期优化任务")
    logger.info("=" * 50)

    short_term_results = implementer.execute_optimizations(OptimizationPhase.SHORT_TERM)
    logger.info(
        f"短期优化完成: {short_term_results['completed_tasks']}/{short_term_results['total_tasks']} 成功")

    # 显示任务结果
    for result in short_term_results['results']:
        if result['status'] == 'completed':
            logger.info(f"✓ {result['task_id']}: {result.get('results', {})}")
        else:
            logger.error(f"✗ {result['task_id']}: {result.get('error', '未知错误')}")

    # 显示中期优化任务（不执行，因为依赖未满足）
    logger.info("=" * 50)
    logger.info("中期优化任务（待执行）")
    logger.info("=" * 50)

    medium_term_tasks = [task for task in implementer.tasks.values()
                         if task.phase == OptimizationPhase.MEDIUM_TERM]

    for task in medium_term_tasks:
        status = "✓" if task.status == "completed" else "⏳"
        logger.info(f"{status} {task.task_id}: {task.name} - {task.description}")
        if task.dependencies:
            logger.info(f"    依赖: {', '.join(task.dependencies)}")

    # 显示长期优化任务
    logger.info("=" * 50)
    logger.info("长期优化任务（待执行）")
    logger.info("=" * 50)

    long_term_tasks = [task for task in implementer.tasks.values()
                       if task.phase == OptimizationPhase.LONG_TERM]

    for task in long_term_tasks:
        status = "✓" if task.status == "completed" else "⏳"
        logger.info(f"{status} {task.task_id}: {task.name} - {task.description}")
        if task.dependencies:
            logger.info(f"    依赖: {', '.join(task.dependencies)}")

    # 显示最终摘要
    logger.info("=" * 50)
    logger.info("优化执行完成")
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

    logger.info("优化建议实现演示完成")


if __name__ == "__main__":
    main()

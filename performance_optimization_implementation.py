#!/usr/bin/env python3
"""
RQA2025性能优化实施脚本

执行数据库优化、缓存优化、异步处理优化等性能提升措施。
"""

import asyncio
import logging
import sys
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
import psycopg2
import redis
import aiohttp
import json
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.database.optimization import DatabaseOptimizer
from src.infrastructure.cache.advanced_cache import CacheManager
from src.core.async_processing.optimized_async import AsyncProcessingManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceOptimizer:
    """性能优化执行器"""

    def __init__(self):
        self.db_optimizer = None
        self.cache_manager = None
        self.async_manager = None
        self.baseline_metrics = {}
        self.optimized_metrics = {}
        self.optimization_results = {}

    async def initialize(self):
        """初始化优化器"""
        logger.info("初始化性能优化器...")

        # 初始化各个优化模块
        self.db_optimizer = DatabaseOptimizer(
            connection_string=os.getenv("DATABASE_URL", "postgresql://localhost/rqa2025")
        )

        self.cache_manager = CacheManager()
        await self.cache_manager.initialize()

        self.async_manager = AsyncProcessingManager()
        await self.async_manager.initialize()

        logger.info("性能优化器初始化完成")

    async def run_baseline_tests(self) -> Dict[str, Any]:
        """运行基准测试"""
        logger.info("运行性能基准测试...")

        baseline = {
            'timestamp': datetime.utcnow().isoformat(),
            'database': await self._test_database_performance(),
            'cache': await self._test_cache_performance(),
            'async_processing': await self._test_async_performance()
        }

        self.baseline_metrics = baseline
        logger.info(f"基准测试完成: {json.dumps(baseline, indent=2, default=str)}")
        return baseline

    async def _test_database_performance(self) -> Dict[str, Any]:
        """测试数据库性能"""
        if not self.db_optimizer:
            return {'error': 'Database optimizer not initialized'}

        try:
            # 测试查询性能
            test_query = """
            SELECT COUNT(*) as total_orders,
                   AVG(price) as avg_price,
                   MAX(created_at) as latest_order
            FROM orders
            WHERE created_at >= NOW() - INTERVAL '24 hours'
            """

            analysis = await self.db_optimizer.analyze_query_performance(test_query)

            # 获取表统计
            report = await self.db_optimizer.get_performance_report()

            return {
                'query_analysis': analysis,
                'table_stats': report,
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"数据库性能测试失败: {e}")
            return {'error': str(e), 'status': 'failed'}

    async def _test_cache_performance(self) -> Dict[str, Any]:
        """测试缓存性能"""
        if not self.cache_manager:
            return {'error': 'Cache manager not initialized'}

        try:
            # 缓存性能测试
            test_data = {'key': f'test_{int(time.time())}', 'value': 'test_data'}

            # 测试写入性能
            start_time = time.time()
            await self.cache_manager.cache.set(test_data['key'], test_data, ttl=60)
            write_time = time.time() - start_time

            # 测试读取性能
            start_time = time.time()
            result = await self.cache_manager.cache.get(test_data['key'])
            read_time = time.time() - start_time

            # 获取缓存统计
            stats = self.cache_manager.get_performance_stats()

            return {
                'write_time': write_time,
                'read_time': read_time,
                'cache_stats': stats,
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"缓存性能测试失败: {e}")
            return {'error': str(e), 'status': 'failed'}

    async def _test_async_performance(self) -> Dict[str, Any]:
        """测试异步处理性能"""
        if not self.async_manager:
            return {'error': 'Async manager not initialized'}

        try:
            # 异步处理性能测试
            async def dummy_task(n: int):
                await asyncio.sleep(0.01)  # 10ms模拟任务
                return n * 2

            # 批量提交任务
            start_time = time.time()
            tasks = [dummy_task(i) for i in range(100)]
            results = await self.async_manager.coroutine_pool.submit_many(tasks)
            total_time = time.time() - start_time

            # 获取异步处理统计
            stats = await self.async_manager.get_processing_stats()

            return {
                'total_time': total_time,
                'tasks_count': len(results),
                'avg_time_per_task': total_time / len(results),
                'async_stats': stats,
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"异步处理性能测试失败: {e}")
            return {'error': str(e), 'status': 'failed'}

    async def apply_database_optimizations(self) -> Dict[str, Any]:
        """应用数据库优化"""
        logger.info("开始数据库优化...")

        if not self.db_optimizer:
            return {'error': 'Database optimizer not initialized'}

        results = {
            'indexes_created': [],
            'stats_updated': False,
            'partitioning': {},
            'errors': []
        }

        try:
            # 创建优化的索引
            logger.info("创建数据库索引...")
            index_results = await self.db_optimizer.create_optimized_indexes()
            results['indexes_created'] = index_results

            # 更新统计信息
            logger.info("更新表统计信息...")
            stats_result = await self.db_optimizer.optimize_table_statistics()
            results['stats_updated'] = stats_result

            # 检查分区设置
            logger.info("检查表分区配置...")
            partitioning_result = await self.db_optimizer.setup_partitioning()
            results['partitioning'] = partitioning_result

        except Exception as e:
            logger.error(f"数据库优化失败: {e}")
            results['errors'].append(str(e))

        logger.info(f"数据库优化完成: {results}")
        return results

    async def apply_cache_optimizations(self) -> Dict[str, Any]:
        """应用缓存优化"""
        logger.info("开始缓存优化...")

        if not self.cache_manager:
            return {'error': 'Cache manager not initialized'}

        results = {
            'warmup_results': {},
            'cache_stats': {},
            'errors': []
        }

        try:
            # 预热热门数据
            logger.info("预热缓存数据...")
            warmup_results = await self.cache_manager.warmup_popular_data()
            results['warmup_results'] = warmup_results

            # 获取缓存统计
            cache_stats = self.cache_manager.get_performance_stats()
            results['cache_stats'] = cache_stats

        except Exception as e:
            logger.error(f"缓存优化失败: {e}")
            results['errors'].append(str(e))

        logger.info(f"缓存优化完成: {results}")
        return results

    async def apply_async_optimizations(self) -> Dict[str, Any]:
        """应用异步处理优化"""
        logger.info("开始异步处理优化...")

        if not self.async_manager:
            return {'error': 'Async manager not initialized'}

        results = {
            'task_queue_stats': {},
            'coroutine_pool_stats': {},
            'test_results': {},
            'errors': []
        }

        try:
            # 测试异步任务处理
            logger.info("测试异步任务处理...")
            task_id = await self.async_manager.submit_market_data_update(['AAPL', 'GOOG', 'MSFT'])
            results['test_task_id'] = task_id

            # 等待任务完成
            await asyncio.sleep(2)
            task_status = await self.async_manager.task_queue.get_task_status(task_id)
            results['test_results'] = task_status

            # 获取异步处理统计
            stats = await self.async_manager.get_processing_stats()
            results['task_queue_stats'] = stats['task_queue']
            results['coroutine_pool_stats'] = stats['coroutine_pool']

        except Exception as e:
            logger.error(f"异步处理优化失败: {e}")
            results['errors'].append(str(e))

        logger.info(f"异步处理优化完成: {results}")
        return results

    async def run_optimized_tests(self) -> Dict[str, Any]:
        """运行优化后的测试"""
        logger.info("运行优化后的性能测试...")

        optimized = {
            'timestamp': datetime.utcnow().isoformat(),
            'database': await self._test_database_performance(),
            'cache': await self._test_cache_performance(),
            'async_processing': await self._test_async_performance()
        }

        self.optimized_metrics = optimized
        logger.info(f"优化后测试完成: {json.dumps(optimized, indent=2, default=str)}")
        return optimized

    def calculate_improvements(self) -> Dict[str, Any]:
        """计算性能提升"""
        improvements = {
            'database': self._calculate_metric_improvements('database'),
            'cache': self._calculate_metric_improvements('cache'),
            'async_processing': self._calculate_metric_improvements('async_processing'),
            'overall_score': 0.0
        }

        # 计算总体得分
        scores = []
        for category in ['database', 'cache', 'async_processing']:
            if 'improvement_percentage' in improvements[category]:
                scores.append(improvements[category]['improvement_percentage'])

        if scores:
            improvements['overall_score'] = sum(scores) / len(scores)

        return improvements

    def _calculate_metric_improvements(self, category: str) -> Dict[str, Any]:
        """计算单个类别的性能提升"""
        baseline = self.baseline_metrics.get(category, {})
        optimized = self.optimized_metrics.get(category, {})

        if not baseline or not optimized or baseline.get('status') == 'failed' or optimized.get('status') == 'failed':
            return {'error': 'Insufficient data for comparison'}

        improvements = {}

        # 数据库性能提升计算
        if category == 'database':
            baseline_time = baseline.get('query_analysis', {}).get('execution_time', 0)
            optimized_time = optimized.get('query_analysis', {}).get('execution_time', 0)

            if baseline_time > 0 and optimized_time > 0:
                improvement = (baseline_time - optimized_time) / baseline_time * 100
                improvements['query_time_improvement'] = improvement
                improvements['improvement_percentage'] = improvement

        # 缓存性能提升计算
        elif category == 'cache':
            baseline_read = baseline.get('read_time', 0)
            optimized_read = optimized.get('read_time', 0)

            if baseline_read > 0 and optimized_read > 0:
                improvement = (baseline_read - optimized_read) / baseline_read * 100
                improvements['cache_read_improvement'] = improvement
                improvements['improvement_percentage'] = improvement

        # 异步处理性能提升计算
        elif category == 'async_processing':
            baseline_total = baseline.get('total_time', 0)
            optimized_total = optimized.get('total_time', 0)

            if baseline_total > 0 and optimized_total > 0:
                improvement = (baseline_total - optimized_total) / baseline_total * 100
                improvements['async_processing_improvement'] = improvement
                improvements['improvement_percentage'] = improvement

        return improvements

    async def generate_report(self) -> Dict[str, Any]:
        """生成优化报告"""
        logger.info("生成性能优化报告...")

        report = {
            'execution_time': datetime.utcnow().isoformat(),
            'baseline_metrics': self.baseline_metrics,
            'optimized_metrics': self.optimized_metrics,
            'optimization_results': self.optimization_results,
            'improvements': self.calculate_improvements(),
            'recommendations': self._generate_recommendations()
        }

        # 保存报告到文件
        report_file = f"performance_optimization_report_{int(time.time())}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        logger.info(f"性能优化报告已保存到: {report_file}")
        return report

    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []

        improvements = self.calculate_improvements()

        # 数据库优化建议
        db_improvement = improvements.get('database', {}).get('improvement_percentage', 0)
        if db_improvement < 20:
            recommendations.append("数据库性能提升有限，建议进一步优化索引和查询结构")

        # 缓存优化建议
        cache_improvement = improvements.get('cache', {}).get('improvement_percentage', 0)
        if cache_improvement < 30:
            recommendations.append("缓存性能有待提升，建议优化缓存策略和预热机制")

        # 异步处理优化建议
        async_improvement = improvements.get('async_processing', {}).get('improvement_percentage', 0)
        if async_improvement < 25:
            recommendations.append("异步处理性能需要改进，建议优化并发控制和任务调度")

        # 总体建议
        overall_score = improvements.get('overall_score', 0)
        if overall_score > 50:
            recommendations.append("🎉 性能优化效果显著，系统性能大幅提升")
        elif overall_score > 20:
            recommendations.append("✅ 性能优化效果良好，建议持续监控和微调")
        else:
            recommendations.append("⚠️ 性能优化效果有限，建议深入分析瓶颈点")

        return recommendations


async def main():
    """主函数"""
    logger.info("🚀 开始RQA2025性能优化实施")

    optimizer = PerformanceOptimizer()

    try:
        # 初始化
        await optimizer.initialize()

        # 运行基准测试
        logger.info("📊 第一阶段: 运行基准性能测试")
        baseline = await optimizer.run_baseline_tests()

        # 应用优化措施
        logger.info("🔧 第二阶段: 应用性能优化措施")

        # 数据库优化
        db_results = await optimizer.apply_database_optimizations()
        optimizer.optimization_results['database'] = db_results

        # 缓存优化
        cache_results = await optimizer.apply_cache_optimizations()
        optimizer.optimization_results['cache'] = cache_results

        # 异步处理优化
        async_results = await optimizer.apply_async_optimizations()
        optimizer.optimization_results['async'] = async_results

        # 等待优化生效
        logger.info("⏳ 等待优化措施生效...")
        await asyncio.sleep(5)

        # 运行优化后的测试
        logger.info("📈 第三阶段: 运行优化后性能测试")
        optimized = await optimizer.run_optimized_tests()

        # 生成报告
        logger.info("📋 第四阶段: 生成优化报告")
        report = await optimizer.generate_report()

        # 输出总结
        improvements = report['improvements']
        logger.info("🎯 性能优化完成总结:")
        logger.info(".2f"        logger.info(".2f"        logger.info(".2f"        logger.info(".2f"
        for rec in report['recommendations']:
            logger.info(f"💡 {rec}")

    except Exception as e:
        logger.error(f"性能优化执行失败: {e}")
        raise
    finally:
        # 清理资源
        if optimizer.async_manager:
            await optimizer.async_manager.shutdown()

    logger.info("✅ RQA2025性能优化实施完成")


if __name__ == "__main__":
    # 设置环境变量
    os.environ.setdefault('DATABASE_URL', 'postgresql://localhost/rqa2025')

    # 运行优化
    asyncio.run(main())

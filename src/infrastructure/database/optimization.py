"""
RQA2025数据库性能优化模块

实现数据库查询优化、索引管理、连接池配置等性能优化功能。
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """数据库性能优化器"""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def analyze_query_performance(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """分析查询性能"""
        def _execute_explain():
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # 执行EXPLAIN ANALYZE
                    explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"
                    cursor.execute(explain_query, params or ())

                    result = cursor.fetchall()
                    return result[0]['QUERY PLAN'][0] if result else {}

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, _execute_explain
            )

            analysis = {
                'execution_time': result.get('Execution Time', 0),
                'planning_time': result.get('Planning Time', 0),
                'total_cost': result.get('Total Cost', 0),
                'actual_rows': result.get('Actual Rows', 0),
                'buffers': result.get('Buffers', {}),
                'recommendations': []
            }

            # 生成优化建议
            if analysis['execution_time'] > 1000:  # 超过1秒
                analysis['recommendations'].append("查询执行时间过长，考虑添加索引")

            if analysis['total_cost'] > 10000:
                analysis['recommendations'].append("查询成本过高，可能需要优化查询结构")

            if analysis['actual_rows'] > 100000:
                analysis['recommendations'].append("返回行数过多，考虑添加分页或过滤条件")

            return analysis

        except Exception as e:
            logger.error(f"查询性能分析失败: {e}")
            return {'error': str(e)}

    async def create_optimized_indexes(self) -> List[str]:
        """创建优化的索引"""
        indexes_created = []

        index_definitions = [
            # 订单表索引
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_symbol_time
            ON orders(symbol, created_at DESC)
            WHERE status IN ('pending', 'filled');
            """,

            # 市场数据表索引
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_market_data_timestamp
            ON market_data(timestamp DESC, symbol);
            """,

            # 投资组合表索引
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_portfolio_user_active
            ON portfolios(user_id, created_at DESC)
            WHERE status = 'active';
            """,

            # 交易记录表复合索引
            """
            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_symbol_price_time
            ON trades(symbol, price, executed_at DESC);
            """
        ]

        def _create_indexes():
            created = []
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    for index_sql in index_definitions:
                        try:
                            cursor.execute(index_sql)
                            conn.commit()
                            created.append("Index created successfully")
                            logger.info("数据库索引创建成功")
                        except Exception as e:
                            logger.warning(f"索引创建失败: {e}")
                            conn.rollback()
            return created

        try:
            indexes_created = await asyncio.get_event_loop().run_in_executor(
                self.executor, _create_indexes
            )
        except Exception as e:
            logger.error(f"批量创建索引失败: {e}")

        return indexes_created

    async def optimize_table_statistics(self) -> bool:
        """优化表统计信息"""
        def _update_stats():
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    # 更新关键表的统计信息
                    tables = ['orders', 'market_data', 'portfolios', 'trades', 'users']

                    for table in tables:
                        try:
                            cursor.execute(f"ANALYZE {table};")
                            logger.info(f"表 {table} 统计信息已更新")
                        except Exception as e:
                            logger.warning(f"更新表 {table} 统计信息失败: {e}")

                    conn.commit()
                    return True

        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, _update_stats
            )
        except Exception as e:
            logger.error(f"表统计信息优化失败: {e}")
            return False

    async def setup_partitioning(self) -> Dict[str, Any]:
        """设置表分区（按时间分区示例）"""
        partition_setup = {
            'orders': {
                'partitioned': False,
                'strategy': 'monthly',
                'retention_days': 365
            },
            'market_data': {
                'partitioned': False,
                'strategy': 'daily',
                'retention_days': 90
            },
            'trades': {
                'partitioned': False,
                'strategy': 'monthly',
                'retention_days': 730
            }
        }

        # 这里实现实际的分区创建逻辑
        # 由于分区创建比较复杂，这里提供示例

        def _create_partitions():
            results = {}
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    # 示例：为orders表创建月份分区
                    try:
                        # 检查是否已分区
                        cursor.execute("""
                            SELECT count(*)
                            FROM pg_inherits
                            WHERE inhparent = (
                                SELECT oid FROM pg_class
                                WHERE relname = 'orders'
                            );
                        """)

                        partition_count = cursor.fetchone()[0]

                        if partition_count == 0:
                            # 创建分区表（这里只是示例，实际需要更复杂的逻辑）
                            logger.info("表分区设置需要手动执行复杂DDL操作")
                            results['orders'] = '需要手动分区'
                        else:
                            results['orders'] = f'已有 {partition_count} 个分区'

                    except Exception as e:
                        logger.error(f"分区检查失败: {e}")
                        results['orders'] = f'错误: {e}'

            return results

        try:
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor, _create_partitions
            )
            return results
        except Exception as e:
            logger.error(f"分区设置失败: {e}")
            return {'error': str(e)}

    async def get_performance_report(self) -> Dict[str, Any]:
        """生成数据库性能报告"""
        def _generate_report():
            report = {
                'timestamp': datetime.utcnow().isoformat(),
                'slow_queries': [],
                'index_usage': [],
                'table_sizes': [],
                'connection_stats': {}
            }

            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # 获取慢查询
                    try:
                        cursor.execute("""
                            SELECT query, total_time, calls, mean_time
                            FROM pg_stat_statements
                            WHERE mean_time > 100  -- 平均执行时间超过100ms
                            ORDER BY mean_time DESC
                            LIMIT 10;
                        """)
                        report['slow_queries'] = [dict(row) for row in cursor.fetchall()]
                    except Exception as e:
                        logger.warning(f"获取慢查询统计失败: {e}")

                    # 获取表大小
                    try:
                        cursor.execute("""
                            SELECT schemaname, tablename,
                                   pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                            FROM pg_tables
                            WHERE schemaname = 'public'
                            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                            LIMIT 10;
                        """)
                        report['table_sizes'] = [dict(row) for row in cursor.fetchall()]
                    except Exception as e:
                        logger.warning(f"获取表大小统计失败: {e}")

                    # 获取连接统计
                    try:
                        cursor.execute("""
                            SELECT count(*) as active_connections,
                                   count(*) filter (where state = 'idle') as idle_connections,
                                   count(*) filter (where state = 'active') as active_connections
                            FROM pg_stat_activity
                            WHERE datname = current_database();
                        """)
                        report['connection_stats'] = dict(cursor.fetchone())
                    except Exception as e:
                        logger.warning(f"获取连接统计失败: {e}")

            return report

        try:
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, _generate_report
            )
        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {'error': str(e)}


class QueryOptimizer:
    """查询优化器"""

    def __init__(self, database_optimizer: DatabaseOptimizer):
        self.db_optimizer = database_optimizer

    async def optimize_select_query(self, query: str, params: Optional[tuple] = None) -> Dict[str, Any]:
        """优化SELECT查询"""
        # 分析当前查询性能
        analysis = await self.db_optimizer.analyze_query_performance(query, params)

        # 生成优化建议
        optimizations = {
            'original_query': query,
            'analysis': analysis,
            'suggested_improvements': [],
            'optimized_query': query
        }

        # 检查是否需要添加WHERE条件
        if 'WHERE' not in query.upper() and analysis.get('actual_rows', 0) > 10000:
            optimizations['suggested_improvements'].append(
                "考虑添加WHERE条件来减少返回行数"
            )

        # 检查是否需要索引
        if analysis.get('execution_time', 0) > 500:
            optimizations['suggested_improvements'].append(
                "执行时间较长，考虑添加相关字段的索引"
            )

        # 检查是否需要分页
        if analysis.get('actual_rows', 0) > 1000 and 'LIMIT' not in query.upper():
            optimizations['suggested_improvements'].append(
                "返回数据较多，考虑使用分页查询"
            )

            # 生成优化的分页查询
            if 'ORDER BY' in query.upper():
                optimized = f"{query} LIMIT 100"
            else:
                optimized = f"{query} ORDER BY id LIMIT 100"

            optimizations['optimized_query'] = optimized

        return optimizations

    async def batch_optimize_queries(self, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量优化查询"""
        results = []

        for query_info in queries:
            query = query_info.get('query', '')
            params = query_info.get('params')

            if query:
                optimization = await self.optimize_select_query(query, params)
                results.append({
                    'original': query_info,
                    'optimization': optimization
                })

        return results


# 连接池配置优化
DATABASE_CONFIG = {
    'pool_size': 20,          # 连接池大小
    'max_overflow': 30,       # 最大溢出连接
    'pool_timeout': 30,       # 连接超时时间
    'pool_recycle': 3600,    # 连接回收时间 (1小时)
    'echo': False,            # 生产环境关闭SQL日志
    'pool_pre_ping': True,    # 连接前检查连接是否可用
}

# 数据库参数优化
DB_PARAMETERS = {
    # 内存配置
    'shared_buffers': '256MB',          # 共享缓冲区
    'effective_cache_size': '1GB',      # 有效缓存大小
    'work_mem': '4MB',                  # 工作内存
    'maintenance_work_mem': '64MB',     # 维护工作内存

    # 检查点配置
    'checkpoint_completion_target': '0.9',  # 检查点完成目标
    'wal_buffers': '16MB',               # WAL缓冲区
    'checkpoint_segments': '32',         # 检查点段数

    # 连接配置
    'max_connections': '200',            # 最大连接数
    'tcp_keepalives_idle': '60',         # TCP保活空闲时间
    'tcp_keepalives_interval': '10',     # TCP保活间隔

    # 查询优化
    'random_page_cost': '1.1',          # 随机页面成本
    'effective_io_concurrency': '200',  # 有效IO并发
}

"""
query_optimizer.py

数据库查询优化组件 - QueryOptimizer

提供SQL查询优化功能，包括：
- 查询计划分析
- 索引优化建议
- 批量查询优化
- 查询缓存
- 慢查询检测

特性：
- 自动查询重写优化
- 智能索引推荐
- 查询结果缓存
- 批量操作优化
- 性能监控和告警

作者: RQA2025 Team
日期: 2026-02-13
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from functools import wraps

import numpy as np

# 配置日志
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """查询类型枚举"""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    UPSERT = "UPSERT"
    BATCH = "BATCH"


class OptimizationStrategy(Enum):
    """优化策略枚举"""
    INDEX = "index"              # 索引优化
    REWRITE = "rewrite"          # 查询重写
    CACHE = "cache"              # 缓存优化
    BATCH = "batch"              # 批处理优化
    PREFETCH = "prefetch"        # 预取优化


@dataclass
class QueryPlan:
    """查询计划"""
    query: str
    query_type: QueryType
    estimated_cost: float = 0.0
    estimated_rows: int = 0
    index_usage: List[str] = field(default_factory=list)
    table_scans: List[str] = field(default_factory=list)
    join_operations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class IndexSuggestion:
    """索引建议"""
    table: str
    columns: List[str]
    index_type: str = "btree"
    priority: str = "medium"  # high, medium, low
    reason: str = ""
    estimated_improvement: float = 0.0


@dataclass
class QueryStats:
    """查询统计信息"""
    query_hash: str
    query_pattern: str
    execution_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    max_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    last_executed: float = 0.0
    is_slow_query: bool = False


@dataclass
class QueryOptimizerConfig:
    """查询优化器配置"""
    # 缓存配置
    enable_query_cache: bool = True
    query_cache_ttl: int = 300  # 5分钟
    query_cache_max_size: int = 1000
    
    # 慢查询检测
    slow_query_threshold_ms: float = 100.0  # 100ms
    slow_query_log_enabled: bool = True
    
    # 批量优化
    batch_size: int = 1000
    enable_batch_optimization: bool = True
    
    # 索引建议
    enable_index_suggestions: bool = True
    min_query_count_for_suggestion: int = 10
    
    # 查询重写
    enable_query_rewrite: bool = True
    
    # 性能监控
    enable_performance_monitoring: bool = True
    stats_window_size: int = 1000


class QueryCache:
    """查询缓存 - 缓存查询结果"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    def _generate_key(self, query: str, params: Optional[Tuple] = None) -> str:
        """生成缓存键"""
        key_data = {
            'query': query,
            'params': params or ()
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    async def get(self, query: str, params: Optional[Tuple] = None) -> Optional[Any]:
        """获取缓存结果"""
        async with self._lock:
            key = self._generate_key(query, params)
            
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # 检查是否过期
            if time.time() - entry['created_at'] > self.ttl:
                del self._cache[key]
                del self._access_times[key]
                return None
            
            self._access_times[key] = time.time()
            return entry['result']
    
    async def set(self, query: str, result: Any, params: Optional[Tuple] = None) -> None:
        """设置缓存结果"""
        async with self._lock:
            # 清理过期项
            await self._cleanup_expired()
            
            # 如果缓存已满，移除最久未访问的项
            if len(self._cache) >= self.max_size:
                await self._evict_oldest()
            
            key = self._generate_key(query, params)
            self._cache[key] = {
                'result': result,
                'created_at': time.time()
            }
            self._access_times[key] = time.time()
    
    async def invalidate(self, query: str, params: Optional[Tuple] = None) -> None:
        """使缓存项失效"""
        async with self._lock:
            key = self._generate_key(query, params)
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
    
    async def clear(self) -> None:
        """清空缓存"""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    async def _cleanup_expired(self) -> None:
        """清理过期项"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry['created_at'] > self.ttl
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._access_times[key]
    
    async def _evict_oldest(self) -> None:
        """移除最久未访问的项"""
        if self._access_times:
            oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'ttl': self.ttl
        }


class QueryAnalyzer:
    """查询分析器 - 分析SQL查询"""
    
    def __init__(self):
        self._query_patterns: Dict[str, QueryStats] = {}
        self._table_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'query_count': 0,
            'scan_count': 0,
            'index_usage': defaultdict(int)
        })
    
    def analyze_query(self, query: str) -> QueryPlan:
        """分析查询并生成查询计划"""
        query_type = self._detect_query_type(query)
        
        plan = QueryPlan(
            query=query,
            query_type=query_type
        )
        
        # 分析表扫描
        plan.table_scans = self._detect_table_scans(query)
        
        # 分析索引使用
        plan.index_usage = self._detect_index_usage(query)
        
        # 分析JOIN操作
        plan.join_operations = self._detect_joins(query)
        
        # 生成警告
        plan.warnings = self._generate_warnings(query, plan)
        
        # 估算成本
        plan.estimated_cost = self._estimate_cost(plan)
        
        return plan
    
    def _detect_query_type(self, query: str) -> QueryType:
        """检测查询类型"""
        upper_query = query.strip().upper()
        
        if upper_query.startswith('SELECT'):
            return QueryType.SELECT
        elif upper_query.startswith('INSERT'):
            return QueryType.INSERT
        elif upper_query.startswith('UPDATE'):
            return QueryType.UPDATE
        elif upper_query.startswith('DELETE'):
            return QueryType.DELETE
        else:
            return QueryType.SELECT
    
    def _detect_table_scans(self, query: str) -> List[str]:
        """检测表扫描"""
        scans = []
        
        # 提取FROM子句中的表
        from_pattern = r'FROM\s+(\w+)'
        from_matches = re.findall(from_pattern, query, re.IGNORECASE)
        scans.extend(from_matches)
        
        # 提取JOIN子句中的表
        join_pattern = r'JOIN\s+(\w+)'
        join_matches = re.findall(join_pattern, query, re.IGNORECASE)
        scans.extend(join_matches)
        
        return list(set(scans))
    
    def _detect_index_usage(self, query: str) -> List[str]:
        """检测索引使用"""
        indexes = []
        
        # 检测WHERE子句中的列
        where_pattern = r'WHERE\s+(.+?)(?:ORDER|GROUP|LIMIT|$)'
        where_match = re.search(where_pattern, query, re.IGNORECASE)
        
        if where_match:
            where_clause = where_match.group(1)
            # 提取列名
            column_pattern = r'(\w+)\s*[=<>]'
            columns = re.findall(column_pattern, where_clause)
            indexes.extend(columns)
        
        return indexes
    
    def _detect_joins(self, query: str) -> List[str]:
        """检测JOIN操作"""
        joins = []
        
        join_pattern = r'(\w+)\s+JOIN\s+(\w+)\s+ON\s+(.+?)(?:\s+(?:LEFT|RIGHT|INNER|JOIN|WHERE|ORDER|GROUP|LIMIT)|$)'
        matches = re.findall(join_pattern, query, re.IGNORECASE)
        
        for match in matches:
            joins.append(f"{match[0]} JOIN {match[1]} ON {match[2]}")
        
        return joins
    
    def _generate_warnings(self, query: str, plan: QueryPlan) -> List[str]:
        """生成查询警告"""
        warnings = []
        
        # 检查SELECT *
        if re.search(r'SELECT\s+\*', query, re.IGNORECASE):
            warnings.append("使用SELECT *可能影响性能，建议指定具体列")
        
        # 检查缺少WHERE的UPDATE/DELETE
        if plan.query_type in [QueryType.UPDATE, QueryType.DELETE]:
            if not re.search(r'WHERE\s+', query, re.IGNORECASE):
                warnings.append(f"{plan.query_type.value}语句缺少WHERE子句，可能影响所有行")
        
        # 检查子查询
        if re.search(r'\(\s*SELECT', query, re.IGNORECASE):
            warnings.append("使用子查询，考虑改为JOIN以提高性能")
        
        # 检查LIKE '%xxx%'
        if re.search(r"LIKE\s+'%[^']+%'", query, re.IGNORECASE):
            warnings.append("使用前导通配符的LIKE查询无法使用索引")
        
        return warnings
    
    def _estimate_cost(self, plan: QueryPlan) -> float:
        """估算查询成本"""
        cost = 0.0
        
        # 表扫描成本
        cost += len(plan.table_scans) * 100
        
        # JOIN成本
        cost += len(plan.join_operations) * 50
        
        # 警告成本
        cost += len(plan.warnings) * 10
        
        return cost
    
    def generate_index_suggestions(self, query: str, plan: QueryPlan) -> List[IndexSuggestion]:
        """生成索引建议"""
        suggestions = []
        
        # 为WHERE子句中的列建议索引
        for table in plan.table_scans:
            # 提取该表的WHERE条件列
            where_pattern = rf'WHERE\s+(.+?)(?:ORDER|GROUP|LIMIT|$)'
            where_match = re.search(where_pattern, query, re.IGNORECASE)
            
            if where_match:
                where_clause = where_match.group(1)
                # 提取该表的列
                column_pattern = rf'{table}\.(\w+)\s*[=<>]'
                columns = re.findall(column_pattern, where_clause, re.IGNORECASE)
                
                if columns:
                    suggestions.append(IndexSuggestion(
                        table=table,
                        columns=columns,
                        priority="high" if plan.query_type == QueryType.SELECT else "medium",
                        reason=f"WHERE子句中频繁使用的列: {', '.join(columns)}"
                    ))
        
        return suggestions
    
    def record_execution(self, query: str, execution_time_ms: float) -> None:
        """记录查询执行"""
        query_pattern = self._normalize_query(query)
        query_hash = hashlib.md5(query_pattern.encode()).hexdigest()
        
        if query_hash not in self._query_patterns:
            self._query_patterns[query_hash] = QueryStats(
                query_hash=query_hash,
                query_pattern=query_pattern
            )
        
        stats = self._query_patterns[query_hash]
        stats.execution_count += 1
        stats.total_time_ms += execution_time_ms
        stats.avg_time_ms = stats.total_time_ms / stats.execution_count
        stats.max_time_ms = max(stats.max_time_ms, execution_time_ms)
        stats.min_time_ms = min(stats.min_time_ms, execution_time_ms)
        stats.last_executed = time.time()
    
    def _normalize_query(self, query: str) -> str:
        """标准化查询（用于模式匹配）"""
        # 移除多余空白
        normalized = ' '.join(query.split())
        # 替换具体值为占位符
        normalized = re.sub(r"'[^']*'", "'?'", normalized)
        normalized = re.sub(r'\b\d+\b', '?', normalized)
        return normalized
    
    def get_slow_queries(self, threshold_ms: float) -> List[QueryStats]:
        """获取慢查询列表"""
        slow_queries = []
        
        for stats in self._query_patterns.values():
            if stats.avg_time_ms > threshold_ms:
                stats.is_slow_query = True
                slow_queries.append(stats)
        
        # 按平均时间排序
        slow_queries.sort(key=lambda x: x.avg_time_ms, reverse=True)
        return slow_queries
    
    def get_stats(self) -> Dict[str, Any]:
        """获取分析器统计"""
        return {
            'total_patterns': len(self._query_patterns),
            'table_stats': dict(self._table_stats)
        }


class QueryOptimizer:
    """
    数据库查询优化器
    
    提供SQL查询优化功能，包括查询计划分析、索引建议、
    查询缓存和批量操作优化。
    
    Attributes:
        config: 优化器配置
        cache: 查询缓存
        analyzer: 查询分析器
        
    Example:
        >>> config = QueryOptimizerConfig()
        >>> optimizer = QueryOptimizer(config)
        >>> 
        >>> # 分析查询
        >>> plan = optimizer.analyze("SELECT * FROM users WHERE age > 18")
        >>> 
        >>> # 获取索引建议
        >>> suggestions = optimizer.get_index_suggestions("SELECT * FROM users WHERE age > 18")
        >>> 
        >>> # 使用装饰器优化函数
        >>> @optimizer.cache_query(ttl=300)
        >>> async def get_user(user_id: int):
        ...     return await db.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
    """
    
    def __init__(self, config: Optional[QueryOptimizerConfig] = None):
        """
        初始化查询优化器
        
        Args:
            config: 优化器配置，如果为None则使用默认配置
        """
        self.config = config or QueryOptimizerConfig()
        
        # 初始化组件
        self.cache = QueryCache(
            max_size=self.config.query_cache_max_size,
            ttl=self.config.query_cache_ttl
        )
        self.analyzer = QueryAnalyzer()
        
        # 统计信息
        self._total_queries = 0
        self._optimized_queries = 0
        self._cached_queries = 0
        
        logger.info("QueryOptimizer initialized")
    
    def analyze(self, query: str) -> QueryPlan:
        """
        分析查询
        
        Args:
            query: SQL查询语句
            
        Returns:
            查询计划
        """
        return self.analyzer.analyze_query(query)
    
    def get_index_suggestions(self, query: str) -> List[IndexSuggestion]:
        """
        获取索引建议
        
        Args:
            query: SQL查询语句
            
        Returns:
            索引建议列表
        """
        if not self.config.enable_index_suggestions:
            return []
        
        plan = self.analyze(query)
        return self.analyzer.generate_index_suggestions(query, plan)
    
    def optimize_query(self, query: str) -> str:
        """
        优化查询（重写）
        
        Args:
            query: 原始SQL查询
            
        Returns:
            优化后的查询
        """
        if not self.config.enable_query_rewrite:
            return query
        
        optimized = query
        
        # 优化1: 将SELECT *改为具体列（如果可能）
        # 注意: 这需要知道表结构，这里只做简单示例
        
        # 优化2: 添加LIMIT（如果没有）
        if re.search(r'SELECT', optimized, re.IGNORECASE):
            if not re.search(r'LIMIT\s+\d+', optimized, re.IGNORECASE):
                # 对于没有LIMIT的SELECT查询，添加一个默认LIMIT
                # 实际使用时需要更谨慎
                pass
        
        # 优化3: 转换子查询为JOIN
        # 这是一个复杂的优化，需要解析SQL结构
        
        return optimized
    
    async def execute_with_cache(
        self,
        query: str,
        fetch_func: Callable,
        params: Optional[Tuple] = None,
        cache_ttl: Optional[int] = None
    ) -> Any:
        """
        使用缓存执行查询
        
        Args:
            query: SQL查询
            fetch_func: 实际执行查询的函数
            params: 查询参数
            cache_ttl: 缓存时间（秒）
            
        Returns:
            查询结果
        """
        if not self.config.enable_query_cache:
            return await fetch_func()
        
        # 尝试从缓存获取
        cached_result = await self.cache.get(query, params)
        if cached_result is not None:
            self._cached_queries += 1
            logger.debug(f"Query cache hit: {query[:50]}...")
            return cached_result
        
        # 执行查询
        start_time = time.time()
        result = await fetch_func()
        execution_time = (time.time() - start_time) * 1000
        
        # 记录执行统计
        self.analyzer.record_execution(query, execution_time)
        self._total_queries += 1
        
        # 检查是否为慢查询
        if execution_time > self.config.slow_query_threshold_ms:
            if self.config.slow_query_log_enabled:
                logger.warning(f"Slow query detected ({execution_time:.2f}ms): {query[:100]}...")
        
        # 缓存结果
        ttl = cache_ttl or self.config.query_cache_ttl
        await self.cache.set(query, result, params)
        
        return result
    
    def cache_query(self, ttl: Optional[int] = None) -> Callable:
        """
        查询缓存装饰器
        
        Args:
            ttl: 缓存时间（秒）
            
        Returns:
            装饰器函数
            
        Example:
            >>> @optimizer.cache_query(ttl=600)
            ... async def get_user(user_id: int):
            ...     return await db.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # 生成缓存键
                key_data = {
                    'func': func.__qualname__,
                    'args': args,
                    'kwargs': kwargs
                }
                key_str = json.dumps(key_data, sort_keys=True, default=str)
                cache_key = hashlib.md5(key_str.encode()).hexdigest()
                
                # 尝试从缓存获取
                cached = await self.cache.get(cache_key)
                if cached is not None:
                    return cached
                
                # 执行函数
                result = await func(*args, **kwargs)
                
                # 缓存结果
                cache_ttl = ttl or self.config.query_cache_ttl
                await self.cache.set(cache_key, result)
                
                return result
            
            return wrapper
        return decorator
    
    def batch_optimize(
        self,
        queries: List[str],
        batch_size: Optional[int] = None
    ) -> List[str]:
        """
        批量优化查询
        
        Args:
            queries: 查询列表
            batch_size: 批大小
            
        Returns:
            优化后的查询列表
        """
        if not self.config.enable_batch_optimization:
            return queries
        
        batch_size = batch_size or self.config.batch_size
        optimized_queries = []
        
        for query in queries:
            optimized = self.optimize_query(query)
            optimized_queries.append(optimized)
        
        self._optimized_queries += len(queries)
        
        return optimized_queries
    
    def create_batch_insert(
        self,
        table: str,
        columns: List[str],
        values_list: List[List[Any]]
    ) -> str:
        """
        创建批量INSERT语句
        
        Args:
            table: 表名
            columns: 列名列表
            values_list: 值列表的列表
            
        Returns:
            批量INSERT SQL语句
        """
        if not values_list:
            return ""
        
        columns_str = ', '.join(columns)
        
        # 构建值部分
        value_placeholders = []
        for i, values in enumerate(values_list):
            placeholders = ', '.join([f'${j+1}' for j in range(len(values))])
            value_placeholders.append(f'({placeholders})')
        
        values_str = ', '.join(value_placeholders)
        
        query = f"INSERT INTO {table} ({columns_str}) VALUES {values_str}"
        
        return query
    
    def get_slow_queries(self) -> List[QueryStats]:
        """
        获取慢查询列表
        
        Returns:
            慢查询统计列表
        """
        return self.analyzer.get_slow_queries(self.config.slow_query_threshold_ms)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取优化器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'total_queries': self._total_queries,
            'optimized_queries': self._optimized_queries,
            'cached_queries': self._cached_queries,
            'cache_stats': self.cache.get_stats(),
            'analyzer_stats': self.analyzer.get_stats(),
            'slow_queries': len(self.get_slow_queries())
        }
    
    async def clear_cache(self) -> None:
        """清空查询缓存"""
        await self.cache.clear()
        logger.info("Query cache cleared")


# 全局优化器实例（单例模式）
_global_optimizer: Optional[QueryOptimizer] = None
_global_optimizer_lock = asyncio.Lock()


async def get_global_optimizer(config: Optional[QueryOptimizerConfig] = None) -> QueryOptimizer:
    """
    获取全局查询优化器实例
    
    Args:
        config: 优化器配置，仅在第一次调用时使用
        
    Returns:
        全局优化器实例
    """
    global _global_optimizer
    
    if _global_optimizer is None:
        async with _global_optimizer_lock:
            if _global_optimizer is None:
                _global_optimizer = QueryOptimizer(config)
    
    return _global_optimizer


async def clear_global_optimizer() -> None:
    """清除全局优化器实例"""
    global _global_optimizer
    
    async with _global_optimizer_lock:
        if _global_optimizer:
            await _global_optimizer.clear_cache()
            _global_optimizer = None

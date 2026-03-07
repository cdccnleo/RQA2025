"""
统一查询接口（Unified Query Interface）

本模块提供跨存储的统一查询接口，支持多种数据源的查询聚合和数据分析。

重要说明：
    本模块的QueryResult用于统一查询接口层，支持高级数据分析。
    如需底层数据库操作，请使用 src.infrastructure.utils.interfaces.database_interfaces.QueryResult
    
    两者区别：
    - unified_query.QueryResult: 高级接口，pd.DataFrame格式，支持跨存储查询
    - database_interfaces.QueryResult: 轻量级，List[Dict]格式，用于数据库适配器

使用场景：
    ✅ 跨存储统一查询（InfluxDB + Parquet混合）
    ✅ 需要数据分析和转换
    ✅ 业务层数据接口
    ✅ 需要查询追踪和来源标识
    
    ❌ 不适用于底层数据库直接操作
    ❌ 不适用于需要避免pandas依赖的场景

架构层次：
    业务层 → [本模块] → 数据库适配器层 → 数据库
"""

import logging

import asyncio
import hashlib
import pandas as pd
import threading
import time

# from src.data.adapters.miniqmt.data_cache import ParquetStorage  # 暂时注释，避免循环导入
# from src.infrastructure.database.influxdb_adapter import InfluxDBAdapter  # 暂时注释，避免循环导入
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
"""
基础设施层 - 缓存系统组件

unified_query 模块

缓存系统相关的文件
提供缓存系统相关的功能实现。
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一查询接口
支持InfluxDB + arquet混合存储的跨存储查询和数据聚合
"""

logger = logging.getLogger(__name__)

# 统一查询常量


class UnifiedQueryConstants:
    """统一查询相关常量"""

    # 默认超时配置 (秒)
    DEFAULT_QUERY_TIMEOUT = 30
    MAX_QUERY_TIMEOUT = 300
    MIN_QUERY_TIMEOUT = 5

    # 默认执行时间
    DEFAULT_EXECUTION_TIME = 0.0

    # 默认记录数
    DEFAULT_RECORD_COUNT = 0

    # 并发查询配置
    DEFAULT_MAX_CONCURRENT_QUERIES = 10
    MAX_CONCURRENT_QUERIES = 50

    # 缓存配置 (秒)
    DEFAULT_CACHE_TTL = 300  # 5分钟
    CACHE_TTL_SECONDS = 300
    QUERY_CACHE_SIZE = 1000

    # 时间窗口配置 (分钟)
    DEFAULT_TIME_WINDOW_MINUTES = 5

    # 查询ID长度
    QUERY_ID_LENGTH = 8

    # 批量大小配置
    DEFAULT_BATCH_SIZE = 1000
    MAX_BATCH_SIZE = 10000
    MIN_BATCH_SIZE = 100

    # 重试配置
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0

    # 性能阈值 (秒)
    PERFORMANCE_WARNING_THRESHOLD = 10.0
    PERFORMANCE_CRITICAL_THRESHOLD = 30.0

    # 异步处理配置
    DEFAULT_MAX_ASYNC_WORKERS = 20
    ASYNC_TIMEOUT_BUFFER = 5.0  # 异步超时缓冲(秒)
    ASYNC_BATCH_SIZE = 1000  # 异步批处理大小


class QueryType(Enum):
    """查询类型"""

    REALTIME = "realtime"  # 实时查询
    HISTORICAL = "historical"  # 历史查询
    AGGREGATED = "aggregated"  # 聚合查询
    CROSS_STORAGE = "cross_storage"  # 跨存储查询
    SELECT = "select"  # SELECT查询
    INSERT = "insert"  # INSERT查询
    UPDATE = "update"  # UPDATE查询
    DELETE = "delete"  # DELETE查询


class StorageType(Enum):
    """存储类型"""

    INFLUXDB = "influxdb"
    PARQUET = "parquet"
    REDIS = "redis"
    HYBRID = "hybrid"


@dataclass
class QueryRequest:
    """查询请求"""

    query_id: str
    query_type: QueryType
    data_type: str
    symbols: List[str]
    start_time: datetime
    end_time: datetime
    storage_preference: Optional[StorageType] = None
    aggregation: Optional[Dict[str, Any]] = None
    filters: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    timeout: int = UnifiedQueryConstants.DEFAULT_QUERY_TIMEOUT


@dataclass
class QueryResult:
    """
    统一查询结果（高级接口层）
    
    用途：
        跨存储查询的统一返回格式，支持复杂数据分析和聚合。
        用于业务层的主要数据接口，支持查询追踪和来源标识。
    
    注意：
        ⚠️ 本类与 database_interfaces.QueryResult 不同！
        - 本类用于统一查询接口层（高层抽象）
        - 数据格式为 pd.DataFrame，支持数据分析
        - 包含query_id用于查询追踪
        - 包含data_source标识数据来源
        - 如需底层数据库操作，使用 database_interfaces.QueryResult
    
    使用示例：
        >>> result = QueryResult(
        ...     query_id="abc123",
        ...     success=True,
        ...     data=pd.DataFrame([{"id": 1}]),
        ...     data_source="influxdb",
        ...     record_count=1,
        ...     execution_time=0.5
        ... )
    
    Attributes:
        query_id: 查询唯一标识符（必需）
        success: 查询是否成功（必需）
        data: 查询结果数据（pd.DataFrame格式）
        error_message: 错误信息（如果失败）
        execution_time: 执行时间（秒）
        data_source: 数据来源标识（如"influxdb", "parquet"等）
        record_count: 记录数（注意：不是row_count）
    """

    query_id: str
    success: bool
    data: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    execution_time: float = UnifiedQueryConstants.DEFAULT_EXECUTION_TIME
    data_source: Optional[str] = None
    record_count: int = UnifiedQueryConstants.DEFAULT_RECORD_COUNT


class UnifiedQueryInterface:
    """统一查询接口"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化统一查询接口

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.query_timeout = self.config.get(
            "query_timeout", UnifiedQueryConstants.DEFAULT_QUERY_TIMEOUT)
        self.max_concurrent_queries = self.config.get(
            "max_concurrent_queries", UnifiedQueryConstants.DEFAULT_MAX_CONCURRENT_QUERIES
        )
        
        # 初始化子组件
        self._init_components()
        
        # 存储适配器
        self.storage_adapters = self._initialize_storage_adapters()
        self._adapters = self.storage_adapters  # 为测试兼容性添加别名
        
        # 查询统计
        self._query_stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_execution_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 为了向后兼容，保留原有属性
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.cache_ttl = self.config.get("cache_ttl", UnifiedQueryConstants.DEFAULT_CACHE_TTL)
        self.query_cache: Dict[str, Tuple[QueryResult, float]] = {}
        self.cache_lock = threading.RLock()
        self.query_executor = ThreadPoolExecutor(max_workers=self.max_concurrent_queries)
        self.async_executor = ThreadPoolExecutor(
            max_workers=self.config.get("max_async_workers",
                                   UnifiedQueryConstants.DEFAULT_MAX_ASYNC_WORKERS)
        )

        logger.info("统一查询接口已初始化")
    
    def _init_components(self) -> None:
        """初始化子组件"""
        try:
            from .query_cache_manager import QueryCacheManager
            from .query_executor import QueryExecutor
            from .query_validator import QueryValidator
            
            self._cache_manager = QueryCacheManager(
                cache_enabled=self.config.get("cache_enabled", True),
                cache_ttl=self.config.get("cache_ttl", UnifiedQueryConstants.DEFAULT_CACHE_TTL)
            )
            self._validator = QueryValidator()
            # 执行器需要在storage_adapters初始化后创建
            self._executor = None
            self.COMPONENTS_AVAILABLE = True
            
        except ImportError as e:
            logger.warning(f"无法导入查询组件，使用兼容模式: {e}")
            self._cache_manager = None
            self._validator = None
            self._executor = None
            self.COMPONENTS_AVAILABLE = False

    def query_data(self, request: QueryRequest) -> QueryResult:
        """
        执行数据查询 - 同步版本

        Args:
            request: 查询请求

        Returns:
            QueryResult: 查询结果
        """
        # 使用异步执行器的同步调用
        future = self.async_executor.submit(self._execute_query_async, request)
        return future.result(timeout=self.query_timeout)

    async def query_data_async(self, request: QueryRequest) -> QueryResult:
        """
        执行数据查询 - 异步版本

        Args:
            request: 查询请求

        Returns:
            QueryResult: 查询结果
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.async_executor,
            self._execute_query_async,
            request
        )

    def _execute_query_async(self, request: QueryRequest) -> QueryResult:
        """
        异步查询执行核心逻辑

        Args:
            request: 查询请求

        Returns:
            QueryResult: 查询结果
        """
        start_time = time.time()

        try:
            # 检查缓存
            if self.cache_enabled:
                cached_result = self._get_cached_result(request)
                if cached_result:
                    logger.info(f"查询缓存命中: {request.query_id}")
                    return cached_result

            # 根据查询类型选择执行策略
            if request.query_type == QueryType.REALTIME:
                result = self._execute_realtime_query(request)
            elif request.query_type == QueryType.HISTORICAL:
                result = self._execute_historical_query(request)
            elif request.query_type == QueryType.AGGREGATED:
                result = self._execute_aggregated_query(request)
            elif request.query_type == QueryType.CROSS_STORAGE:
                result = self._execute_cross_storage_query(request)
            else:
                raise ValueError(f"不支持的查询类型: {request.query_type}")

            # 计算执行时间
            result.execution_time = time.time() - start_time

            # 缓存结果
            if self.cache_enabled and result.success:
                self._cache_result(request, result)

            logger.info(f"查询完成: {request.query_id}, 耗时: {result.execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"查询执行失败: {request.query_id}, 错误: {e}")
            return QueryResult(
                query_id=request.query_id,
                success=False,
                data=pd.DataFrame(),
                error_message=str(e),
                execution_time=time.time() - start_time,
                record_count=0,
            )

    def query_realtime_data(
        self,
        symbols: List[str],
        data_type: str = "tick",
        storage_preference: Optional[StorageType] = None,
    ):
        """
        查询实时数据

        Args:
            symbols: 股票代码列表
            data_type: 数据类型
            storage_preference: 存储偏好

        Returns:
            QueryResult: 查询结果
        """
        request = QueryRequest(
            query_id=f"realtime_{int(time.time())}",
            query_type=QueryType.REALTIME,
            data_type=data_type,
            symbols=symbols,
            start_time=datetime.now() - timedelta(minutes=UnifiedQueryConstants.DEFAULT_TIME_WINDOW_MINUTES),
            end_time=datetime.now(),
            storage_preference=storage_preference or StorageType.INFLUXDB,
        )

        return self.query_data(request)

    def query_historical_data(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        data_type: str = "kline",
        storage_preference: Optional[StorageType] = None,
    ):
        """
        查询历史数据

        Args:
            symbols: 股票代码列表
            start_time: 开始时间
            end_time: 结束时间
            data_type: 数据类型
            storage_preference: 存储偏好

        Returns:
            QueryResult: 查询结果
        """
        request = QueryRequest(
            query_id=f"historical_{int(time.time())}",
            query_type=QueryType.HISTORICAL,
            data_type=data_type,
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            storage_preference=storage_preference or StorageType.PARQUET,
        )

        return self.query_data(request)

    def query_aggregated_data(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        aggregation: Dict[str, Any],
        data_type: str = "kline",
    ):
        """
        查询聚合数据

        Args:
            symbols: 股票代码列表
            start_time: 开始时间
            end_time: 结束时间
            aggregation: 聚合配置
            data_type: 数据类型

        Returns:
            QueryResult: 查询结果
        """
        request = QueryRequest(
            query_id=f"aggregated_{int(time.time())}",
            query_type=QueryType.AGGREGATED,
            data_type=data_type,
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
            aggregation=aggregation,
        )

        return self.query_data(request)

    def query_cross_storage_data(
        self,
        symbols: List[str],
        start_time: datetime,
        end_time: datetime,
        data_type: str = "kline",
    ):
        """
        跨存储查询数据

        Args:
            symbols: 股票代码列表
            start_time: 开始时间
            end_time: 结束时间
            data_type: 数据类型

        Returns:
            QueryResult: 查询结果
        """
        request = QueryRequest(
            query_id=f"cross_storage_{int(time.time())}",
            query_type=QueryType.CROSS_STORAGE,
            data_type=data_type,
            symbols=symbols,
            start_time=start_time,
            end_time=end_time,
        )

        return self.query_data(request)

    def _execute_realtime_query(self, request: QueryRequest) -> QueryResult:
        """执行实时查询"""
        try:
            # 优先使用InfluxDB进行实时查询
            adapter = self.storage_adapters.get(StorageType.INFLUXDB)
            if not adapter:
                return QueryResult(
                    query_id=request.query_id,
                    success=False,
                    data=pd.DataFrame(),
                    record_count=0,
                    error_message="未找到可用的数据源适配器",
                )

            data = adapter.query_realtime(
                symbols=request.symbols,
                data_type=request.data_type,
                limit=request.limit,
            )

            return QueryResult(
                query_id=request.query_id,
                success=True,
                data=data,
                data_source="influxdb",
                record_count=len(data) if data is not None else 0,
            )

        except Exception as e:
            logger.error(f"实时查询失败: {e}")
            return QueryResult(
                query_id=request.query_id,
                success=False,
                data=pd.DataFrame(),
                record_count=0,
                error_message=str(e),
            )

    def _execute_historical_query(self, request: QueryRequest) -> QueryResult:
        """执行历史查询"""
        try:
            # 优先使用Parquet进行历史查询
            adapter = self.storage_adapters.get(StorageType.PARQUET)
            if not adapter:
                raise RuntimeError("Parquet适配器未初始化")

            data = adapter.query_historical(
                symbols=request.symbols,
                data_type=request.data_type,
                start_time=request.start_time,
                end_time=request.end_time,
                filters=request.filters,
            )

            return QueryResult(
                query_id=request.query_id,
                success=True,
                data=data,
                data_source="parquet",
                record_count=len(data) if data is not None else 0,
            )

        except Exception as e:
            logger.error(f"历史查询失败: {e}")
            return QueryResult(
                query_id=request.query_id,
                success=False,
                data=pd.DataFrame(),
                record_count=0,
                error_message=str(e),
            )

    def _execute_aggregated_query(self, request: QueryRequest) -> QueryResult:
        """执行聚合查询"""
        try:
            # 根据聚合配置选择最佳存储
            if request.aggregation.get("real_time", False):
                adapter = self.storage_adapters.get(StorageType.INFLUXDB)
                data_source = "influxdb"
            else:
                adapter = self.storage_adapters.get(StorageType.PARQUET)
                data_source = "parquet"

            if not adapter:
                raise RuntimeError(f"{data_source}适配器未初始化")

            data = adapter.query_aggregated(
                symbols=request.symbols,
                data_type=request.data_type,
                start_time=request.start_time,
                end_time=request.end_time,
                aggregation=request.aggregation,
            )
            return QueryResult(
                query_id=request.query_id,
                success=True,
                data=data,
                data_source=data_source,
                record_count=len(data) if data is not None else 0,
            )
        except Exception as e:
            logger.error(f"聚合查询失败: {e}")
            return QueryResult(
                query_id=request.query_id,
                success=False,
                data=pd.DataFrame(),
                record_count=0,
                error_message=str(e),
            )

    def _execute_cross_storage_query(self, request: QueryRequest) -> QueryResult:
        """执行跨存储查询"""
        try:
            # 并行查询多个存储
            futures = []

            with ThreadPoolExecutor(max_workers=len(self.storage_adapters)) as executor:
                for storage_type, adapter in self.storage_adapters.items():
                    future = executor.submit(self._query_single_storage,
                                             adapter, storage_type, request)
                    futures.append((storage_type, future))

            # 收集结果
            results = []
            for storage_type, future in futures:
                try:
                    result = future.result(timeout=request.timeout)
                    if result is not None:
                        results.append((storage_type, result))
                except Exception as e:
                    logger.warning(f"存储 {storage_type} 查询失败: {e}")

            # 合并结果
            if results:
                merged_data = self._merge_cross_storage_results(results)
                return QueryResult(
                    query_id=request.query_id,
                    success=True,
                    data=merged_data,
                    data_source="cross_storage",
                    record_count=len(merged_data) if merged_data is not None else 0,
                )
            else:
                return QueryResult(
                    query_id=request.query_id,
                    success=False,
                    data=pd.DataFrame(),
                    record_count=0,
                    error_message="所有存储查询都失败",
                )
        except Exception as e:
            logger.error(f"跨存储查询失败: {e}")
            return QueryResult(
                query_id=request.query_id,
                success=False,
                data=pd.DataFrame(),
                record_count=0,
                error_message=str(e),
            )

    def _query_single_storage(self, adapter, storage_type: StorageType, request: QueryRequest):
        """查询单个存储"""
        try:
            if storage_type == StorageType.INFLUXDB:
                return adapter.query_realtime(
                    symbols=request.symbols,
                    data_type=request.data_type,
                    limit=request.limit,
                )
            elif storage_type == StorageType.PARQUET:
                return adapter.query_historical(
                    symbols=request.symbols,
                    data_type=request.data_type,
                    start_time=request.start_time,
                    end_time=request.end_time,
                    filters=request.filters,
                )

            else:
                logger.warning(f"不支持的存储类型: {storage_type}")
                return None

        except Exception as e:
            logger.error(f"查询存储 {storage_type} 失败: {e}")
            return None

    def _merge_cross_storage_results(self, results: List[Tuple[StorageType, pd.DataFrame]]) -> pd.DataFrame:
        """合并跨存储查询结果"""
        if not results:
            return pd.DataFrame()

        # 按时间戳排序并去重
        merged_data = []
        for storage_type, data in results:
            if data is not None and not data.empty:
                # 添加数据源标识
                data["data_source"] = storage_type.value
                merged_data.append(data)

        if not merged_data:
            return pd.DataFrame()

        # 合并数据
        combined_df = pd.concat(merged_data, ignore_index=True)

        # 按时间戳排序
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.sort_values("timestamp")

        # 去重（保留最新的数据）
        if "timestamp" in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
        return combined_df

    def _get_cached_result(self, request: QueryRequest) -> Optional[QueryResult]:
        """获取缓存的查询结果"""
        if not self.cache_enabled:
            return None
        if self._cache_manager:
            return self._cache_manager.get_cached_result(request)
        
        cache_key = self._generate_cache_key(request)

        with self.cache_lock:
            if cache_key in self.query_cache:
                result, timestamp = self.query_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return result
                del self.query_cache[cache_key]

        return None

    def _cache_result(self, request: QueryRequest, result: QueryResult):
        """缓存查询结果"""
        if not self.cache_enabled:
            return
        if self._cache_manager:
            try:
                self._cache_manager.cache_result(request, result)
                self._cache_manager.cleanup_expired_cache()
                return
            except Exception as exc:  # pragma: no cover - 记录降级
                logger.warning(f"缓存管理器写入失败，回退到内置缓存: {exc}")

        cache_key = self._generate_cache_key(request)

        with self.cache_lock:
            self.query_cache[cache_key] = (result, time.time())
            self._cleanup_expired_cache()

    def _generate_cache_key(self, request: QueryRequest) -> str:
        """生成缓存键"""
        key_parts = [
            request.query_type.value,
            request.data_type,
            ",".join(sorted(request.symbols)),
            request.start_time.isoformat(),
            request.end_time.isoformat(),
        ]
        if request.aggregation:
            key_parts.append(str(request.aggregation))

        if request.filters:
            key_parts.append(str(request.filters))

        return hashlib.sha256("|".join(key_parts).encode()).hexdigest()

    def _cleanup_expired_cache(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = []

        for key, (result, timestamp) in self.query_cache.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self.query_cache[key]

    def _initialize_storage_adapters(self) -> Dict[StorageType, Any]:
        """初始化存储适配器"""
        adapters = {}
        try:
            # 初始化InfluxDB适配器
            # 跨层级导入：infrastructure层组件
            # adapters[StorageType.INFLUXDB] = InfluxDBAdapter()

            # 初始化Parquet适配器
            # 跨层级导入：adapters层组件
            # adapters[StorageType.PARQUET] = ParquetStorage()

            # 暂时注释掉，避免未定义的名称错误

            logger.info("存储适配器初始化完成")
            
            # 在storage_adapters初始化后创建执行器
            if self.COMPONENTS_AVAILABLE and hasattr(self, '_executor') and self._executor is None:
                try:
                    from .query_executor import QueryExecutor
                    self._executor = QueryExecutor(adapters, self.max_concurrent_queries)
                except ImportError:
                    pass

        except Exception as e:
            logger.error(f"存储适配器初始化失败: {e}")

        return adapters

    def get_query_statistics(self) -> Dict[str, Any]:
        """获取查询统计信息"""
        # 使用缓存管理器组件获取统计
        if self._cache_manager:
            cache_stats = self._cache_manager.get_cache_statistics()
            return {
                **cache_stats,
                "max_concurrent_queries": self.max_concurrent_queries,
                "query_timeout": self.query_timeout,
            }
        
        # 回退到原有方法
        with self.cache_lock:
            cache_size = len(self.query_cache)
            cache_hit_rate = self._calculate_cache_hit_rate()

        return {
            "cache_size": cache_size,
            "cache_hit_rate": cache_hit_rate,
            "max_concurrent_queries": self.max_concurrent_queries,
            "query_timeout": self.query_timeout,
        }

    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        # 简化实现，实际项目中需要更复杂的统计
        return 0.0

    async def query_data_async(self, request: QueryRequest) -> QueryResult:
        """
        异步执行数据查询 (性能优化版)

        Args:
            request: 查询请求

        Returns:
            QueryResult: 查询结果
        """
        loop = asyncio.get_event_loop()

        try:
            # 在线程池中执行查询，避免阻塞事件循环
            result = await loop.run_in_executor(self.async_executor, self.query_data, request)
            return result

        except Exception as e:
            logger.error(f"异步查询执行失败: {request.query_id}, 错误: {e}")
            return QueryResult(
                query_id=request.query_id,
                success=False,
                data=pd.DataFrame(),
                record_count=0,
                error_message=str(e),
                execution_time=0.0,
            )

    async def query_multiple_data_async(self, requests: List[QueryRequest]) -> List[QueryResult]:
        """
        并发执行多个查询请求 (性能优化版)

        Args:
            requests: 查询请求列表

        Returns:
            List[QueryResult]: 查询结果列表
        """
        # 验证请求
        if not self._validate_requests(requests):
            return []

        try:
            # 智能分组请求
            grouped_requests = self._group_requests_by_type(requests)

            # 创建并发任务
            all_tasks = self._create_concurrent_tasks(grouped_requests)

            # 执行并发查询
            raw_results = await self._execute_concurrent_queries(all_tasks)

            # 处理查询结果
            processed_results = self._process_query_results(raw_results, grouped_requests)

            # 记录成功信息
            self._log_query_success(processed_results, requests, grouped_requests)

            return processed_results

        except Exception as e:
            # 处理查询异常
            return self._handle_query_exception(e, requests)

    def _validate_requests(self, requests: List[QueryRequest]) -> bool:
        """验证请求列表"""
        return bool(requests)

    def _create_concurrent_tasks(self, grouped_requests: Dict[str, List[QueryRequest]]) -> List:
        """创建并发任务"""
        all_tasks = []
        for group in grouped_requests.values():
            # 每个组内的查询并发执行
            group_tasks = [self.query_data_async(request) for request in group]
            all_tasks.extend(group_tasks)
        return all_tasks

    async def _execute_concurrent_queries(self, all_tasks: List) -> List:
        """执行并发查询"""
        # 并发执行所有查询，设置超时保护
        return await asyncio.gather(*all_tasks, return_exceptions=True)

    def _process_query_results(
        self,
        raw_results: List,
        grouped_requests: Dict[str, List[QueryRequest]]
    ) -> List[QueryResult]:
        """处理查询结果"""
        processed_results: List[QueryResult] = []

        if isinstance(raw_results, (str, bytes)):
            iterable_results = [raw_results]
        else:
            try:
                iterable_results = list(raw_results)
            except TypeError:
                iterable_results = [raw_results]

        task_index = 0
        for group_requests in grouped_requests.values():
            for request in group_requests:
                try:
                    result = iterable_results[task_index]
                except IndexError:
                    result = RuntimeError("结果数量不足")
                finally:
                    task_index += 1

                if isinstance(result, QueryResult):
                    processed_results.append(result)
                else:
                    processed_results.append(
                        QueryResult(
                            query_id=request.query_id,
                            success=False,
                            error_message=str(result),
                            execution_time=0.0,
                            record_count=0,
                        )
                    )
        return processed_results

    def _log_query_success(
        self,
        processed_results: List[QueryResult],
        requests: List[QueryRequest],
        grouped_requests: Dict[str, List[QueryRequest]]
    ):
        """记录查询成功信息"""
        logger.info(
            f"批量异步查询完成: {len(processed_results)}/{len(requests)} 成功 (分组数: {len(grouped_requests)})")

    def _handle_query_exception(self, e: Exception, requests: List[QueryRequest]) -> List[QueryResult]:
        """处理查询异常"""
        logger.error(f"批量异步查询失败: {e}")
        if not requests:
            return []
        return [
            QueryResult(
                query_id=request.query_id,
                success=False,
                error_message=str(e),
                execution_time=0.0,
                record_count=0,
            )
            for request in requests
        ]

    def _group_requests_by_type(self, requests: List[QueryRequest]) -> Dict[str, List[QueryRequest]]:
        """按查询类型智能分组，提高缓存命中率"""
        groups = {
            "realtime": [],
            "historical": [],
            "aggregated": [],
            "cross_storage": []
        }

        for request in requests:
            if request.query_type == QueryType.REALTIME:
                groups["realtime"].append(request)
            elif request.query_type == QueryType.HISTORICAL:
                groups["historical"].append(request)
            elif request.query_type == QueryType.AGGREGATED:
                groups["aggregated"].append(request)
            elif request.query_type == QueryType.CROSS_STORAGE:
                groups["cross_storage"].append(request)

        # 移除空组
        return {k: v for k, v in groups.items() if v}

    async def query_realtime_data_async(
        self, data_type: str, symbols: List[str], storage_preference: Optional[StorageType] = None
    ) -> QueryResult:
        """
        异步查询实时数据 (性能优化版)

        Args:
            data_type: 数据类型
            symbols: 交易对列表
            storage_preference: 存储偏好

        Returns:
            QueryResult: 查询结果
        """
        request = QueryRequest(
            query_id=f"realtime_async_{int(time.time())}",
            query_type=QueryType.REALTIME,
            data_type=data_type,
            symbols=symbols,
            start_time=datetime.now() - timedelta(minutes=UnifiedQueryConstants.DEFAULT_TIME_WINDOW_MINUTES),
            end_time=datetime.now(),
            storage_preference=storage_preference or StorageType.INFLUXDB,
        )

        return await self.query_data_async(request)


    def register_adapter(self, storage_type: StorageType, adapter: Any) -> None:
        """
        注册存储适配器
        
        Args:
            storage_type: 存储类型
            adapter: 适配器实例
        """
        if not isinstance(storage_type, StorageType):
            logger.warning(f"无法注册未知存储类型适配器: {storage_type}")
            return
        self._adapters[storage_type] = adapter
        self.storage_adapters[storage_type] = adapter
        logger.info(f"已注册适配器: {storage_type.value}")

    def unregister_adapter(self, storage_type: StorageType) -> None:
        """
        取消注册存储适配器
        
        Args:
            storage_type: 存储类型
        """
        removed = False
        if storage_type in self._adapters:
            del self._adapters[storage_type]
            removed = True
        if storage_type in self.storage_adapters:
            del self.storage_adapters[storage_type]
            removed = True
        if removed:
            logger.info(f"已取消注册适配器: {getattr(storage_type, 'value', storage_type)}")
        else:
            logger.warning(f"未找到适配器可取消: {storage_type}")

    def get_registered_adapters(self) -> List[StorageType]:
        """
        获取已注册的适配器列表
        
        Returns:
            已注册的存储类型列表
        """
        return list(self._adapters.keys())

    def get_query_stats(self) -> Dict[str, Any]:
        """
        获取查询统计信息
        
        Returns:
            查询统计字典
        """
        stats = self._query_stats.copy()
        if stats['total_queries'] > 0:
            stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_queries']
        else:
            stats['avg_execution_time'] = 0.0
        
        if stats['cache_hits'] + stats['cache_misses'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['cache_hit_rate'] = 0.0
        
        stats['active_connections'] = len(self._adapters)
        return stats

    def validate_query_request(self, request: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        验证查询请求
        
        Args:
            request: 查询请求字典
            
        Returns:
            (是否有效, 错误信息)
        """
        # 检查必需字段
        required_fields = ['query_type', 'storage_type']
        for field in required_fields:
            if field not in request:
                return False, f"缺少必需字段: {field}"
        
        # 验证查询类型
        query_type = request.get('query_type')
        valid_query_types = ['realtime', 'historical', 'aggregated', 'cross_storage']
        if query_type not in valid_query_types:
            return False, f"无效的查询类型: {query_type}"
        
        # 验证存储类型
        storage_type = request.get('storage_type')
        valid_storage_types = ['influxdb', 'parquet', 'redis', 'hybrid']
        if storage_type not in valid_storage_types:
            return False, f"无效的存储类型: {storage_type}"
        
        return True, None

    def shutdown(self):
        """关闭查询接口"""
        self.query_executor.shutdown(wait=True)
        self.async_executor.shutdown(wait=True)
        logger.info("统一查询接口已关闭")

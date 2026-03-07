"""
postgresql_loader.py

PostgreSQL数据加载器模块

提供从PostgreSQL数据库加载股票数据的功能，符合数据管理层架构设计。
支持连接池、自动重试、批量查询等功能。

作者: RQA2025 Team
日期: 2026-02-13
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .base_loader import BaseDataLoader, DataLoaderConfig, LoadResult

# 配置日志
logger = logging.getLogger(__name__)


class PostgreSQLDataLoader(BaseDataLoader):
    """
    PostgreSQL数据加载器
    
    从PostgreSQL数据库加载股票历史数据，支持：
    - 股票日线数据加载
    - 多股票批量加载
    - 日期范围查询
    - 连接池管理
    
    Attributes:
        config: 加载器配置
        _connection_pool: 数据库连接池
        
    Example:
        >>> config = DataLoaderConfig(source_type="postgresql")
        >>> loader = PostgreSQLDataLoader(config)
        >>> 
        >>> # 加载单只股票数据
        >>> result = loader.load_stock_data("002837", "2025-01-01", "2025-12-31")
        >>> if result.success:
        ...     df = result.data
        ...     print(f"加载了 {result.row_count} 条记录")
        >>> 
        >>> # 批量加载多只股票
        >>> symbols = ["002837", "688702"]
        >>> results = loader.load_multiple_stocks(symbols, "2025-01-01", "2025-12-31")
    """
    
    def __init__(self, config: Optional[DataLoaderConfig] = None):
        """
        初始化PostgreSQL数据加载器
        
        Args:
            config: 加载器配置，如果为None则使用默认配置
        """
        super().__init__(config)
        
        # 连接池（延迟初始化）
        self._connection_pool = None
        self._db_config = None
        
        # 初始化数据库配置
        self._init_db_config()
        
        logger.info("PostgreSQLDataLoader 初始化完成")
    
    def _init_db_config(self):
        """初始化数据库配置"""
        # 从环境变量获取数据库配置
        # 注意：在Docker环境中，使用'postgres'作为主机名，而不是'localhost'
        self._db_config = {
            'host': os.getenv('RQA_DB_HOST', 
                            os.getenv('DB_HOST', 
                                    os.getenv('POSTGRES_HOST', 'postgres'))),  # 默认使用'postgres'而不是'localhost'
            'port': os.getenv('RQA_DB_PORT', 
                            os.getenv('DB_PORT', 
                                    os.getenv('POSTGRES_PORT', '5432'))),
            'database': os.getenv('RQA_DB_NAME', 
                                os.getenv('DB_NAME', 
                                        os.getenv('POSTGRES_DB', 'rqa2025_prod'))),
            'user': os.getenv('RQA_DB_USER', 
                            os.getenv('DB_USER', 
                                    os.getenv('POSTGRES_USER', 'rqa2025_admin'))),
            'password': os.getenv('RQA_DB_PASSWORD', 
                                os.getenv('DB_PASSWORD', 
                                        os.getenv('POSTGRES_PASSWORD', 'SecurePass123!'))),
        }
        
        logger.debug(f"数据库配置: host={self._db_config['host']}, "
                    f"database={self._db_config['database']}")
    
    def _get_connection_pool(self):
        """获取数据库连接池（延迟初始化）"""
        if self._connection_pool is None:
            try:
                import psycopg2
                from psycopg2 import pool
                from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
                
                # 创建连接池
                self._connection_pool = psycopg2.pool.SimpleConnectionPool(
                    1,  # 最小连接数
                    10,  # 最大连接数
                    **self._db_config
                )
                
                logger.info("数据库连接池初始化成功")
                
            except ImportError:
                logger.error("psycopg2未安装，无法连接PostgreSQL")
                raise
            except Exception as e:
                logger.error(f"数据库连接池初始化失败: {e}")
                raise
        
        return self._connection_pool
    
    def _get_connection(self):
        """从连接池获取连接"""
        pool = self._get_connection_pool()
        conn = pool.getconn()
        conn.set_isolation_level(0)  # ISOLATION_LEVEL_AUTOCOMMIT
        return conn
    
    def _return_connection(self, conn):
        """归还连接到连接池"""
        if self._connection_pool and conn:
            self._connection_pool.putconn(conn)
    
    def validate_connection(self) -> bool:
        """
        验证数据库连接
        
        Returns:
            连接是否有效
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            self._return_connection(conn)
            
            logger.debug("数据库连接验证成功")
            return True
            
        except Exception as e:
            logger.error(f"数据库连接验证失败: {e}")
            return False
    
    def load(self, query: str, params: Optional[Dict[str, Any]] = None) -> LoadResult:
        """
        执行SQL查询并加载数据
        
        Args:
            query: SQL查询语句
            params: 查询参数（字典格式）
            
        Returns:
            加载结果
        """
        import time
        
        start_time = time.time()
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # 执行查询
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # 获取列名
            columns = [desc[0] for desc in cursor.description]
            
            # 获取数据
            rows = cursor.fetchall()
            
            # 创建DataFrame
            df = pd.DataFrame(rows, columns=columns)
            
            cursor.close()
            self._return_connection(conn)
            
            load_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"SQL查询成功: {len(df)} 条记录, 耗时 {load_time_ms:.2f}ms")
            
            return LoadResult(
                data=df,
                success=True,
                message=f"成功加载 {len(df)} 条记录",
                row_count=len(df),
                load_time_ms=load_time_ms
            )
            
        except Exception as e:
            load_time_ms = (time.time() - start_time) * 1000
            logger.error(f"SQL查询失败: {e}")
            
            return LoadResult(
                data=None,
                success=False,
                message=f"查询失败: {str(e)}",
                row_count=0,
                load_time_ms=load_time_ms
            )
    
    def load_stock_data(self, symbol: str, start_date: str, end_date: str) -> LoadResult:
        """
        加载单只股票的历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            加载结果
        """
        query = """
            SELECT
                date as trade_date,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                amount
            FROM akshare_stock_data
            WHERE symbol = %s
              AND date BETWEEN %s AND %s
            ORDER BY date ASC
        """

        # 使用元组作为参数（位置参数），与SQL中的%s占位符匹配
        params = (symbol, start_date, end_date)

        result = self.load(query, params)
        
        if result.success:
            logger.info(f"股票 {symbol} 数据加载成功: {result.row_count} 条记录")
        else:
            logger.warning(f"股票 {symbol} 数据加载失败: {result.message}")
        
        return result
    
    def load_multiple_stocks(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, LoadResult]:
        """
        批量加载多只股票的历史数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            每个股票的加载结果字典
        """
        results = {}
        
        for symbol in symbols:
            results[symbol] = self.load_stock_data(symbol, start_date, end_date)
        
        success_count = sum(1 for r in results.values() if r.success)
        logger.info(f"批量加载完成: {success_count}/{len(symbols)} 只股票成功")
        
        return results
    
    def close(self):
        """关闭数据加载器，释放资源"""
        if self._connection_pool:
            try:
                self._connection_pool.closeall()
                logger.info("数据库连接池已关闭")
            except Exception as e:
                logger.warning(f"关闭连接池时出错: {e}")
            finally:
                self._connection_pool = None
    
    # ==================== 增强功能：自动重试机制 ====================
    
    def load_with_retry(self, query: str, params: Optional[Dict[str, Any]] = None,
                       max_retries: int = 3, backoff_factor: float = 2.0) -> LoadResult:
        """
        带重试的加载方法
        
        使用指数退避策略进行自动重试，提高数据加载的可靠性。
        
        Args:
            query: SQL查询语句
            params: 查询参数
            max_retries: 最大重试次数（默认3次）
            backoff_factor: 退避因子（默认2.0，即每次重试等待时间翻倍）
            
        Returns:
            加载结果
        """
        for attempt in range(max_retries):
            result = self.load(query, params)
            
            if result.success:
                if attempt > 0:
                    logger.info(f"查询在尝试 {attempt + 1} 次后成功")
                return result
            
            # 如果不是最后一次尝试，则等待后重试
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                logger.warning(
                    f"查询失败，{wait_time:.1f}秒后重试 "
                    f"(尝试 {attempt + 1}/{max_retries}): {result.message}"
                )
                time.sleep(wait_time)
            else:
                logger.error(f"查询在 {max_retries} 次尝试后仍然失败: {result.message}")
        
        return result
    
    def load_stock_data_with_retry(self, symbol: str, start_date: str, end_date: str,
                                   max_retries: int = 3) -> LoadResult:
        """
        带重试的股票数据加载方法
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            max_retries: 最大重试次数
            
        Returns:
            加载结果
        """
        query = """
            SELECT
                date as trade_date,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                amount
            FROM akshare_stock_data
            WHERE symbol = %s
              AND date BETWEEN %s AND %s
            ORDER BY date ASC
        """

        # 使用元组作为参数（位置参数），与SQL中的%s占位符匹配
        params = (symbol, start_date, end_date)

        return self.load_with_retry(query, params, max_retries=max_retries)
    
    # ==================== 增强功能：缓存集成 ====================
    
    def _generate_cache_key(self, query: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        生成缓存键
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            缓存键字符串
        """
        import hashlib
        
        # 组合查询和参数
        key_content = f"{query}_{str(params)}"
        
        # 使用MD5生成固定长度的缓存键
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def load_with_cache(self, query: str, params: Optional[Dict[str, Any]] = None,
                       cache_ttl: int = 300, use_memory_cache: bool = True) -> LoadResult:
        """
        带缓存的加载方法
        
        集成内存缓存，减少数据库查询次数，提高响应速度。
        
        Args:
            query: SQL查询语句
            params: 查询参数
            cache_ttl: 缓存过期时间（秒，默认300秒=5分钟）
            use_memory_cache: 是否使用内存缓存
            
        Returns:
            加载结果
        """
        # 如果不使用缓存，直接加载
        if not use_memory_cache:
            return self.load(query, params)
        
        # 生成缓存键
        cache_key = self._generate_cache_key(query, params)
        
        # 检查内存缓存
        if hasattr(self, '_memory_cache') and cache_key in self._memory_cache:
            cached_entry = self._memory_cache[cache_key]
            
            # 检查缓存是否过期
            if time.time() - cached_entry['timestamp'] < cache_ttl:
                logger.debug(f"内存缓存命中: {cache_key[:8]}...")
                cached_data = cached_entry['data']
                return LoadResult(
                    data=cached_data,
                    success=True,
                    message="从内存缓存加载",
                    row_count=len(cached_data) if hasattr(cached_data, '__len__') else 0,
                    load_time_ms=0,
                    metadata={'cached': True, 'cache_type': 'memory'}
                )
            else:
                # 缓存过期，删除
                del self._memory_cache[cache_key]
        
        # 加载数据
        result = self.load(query, params)
        
        # 如果加载成功，存入缓存
        if result.success and result.data is not None:
            if not hasattr(self, '_memory_cache'):
                self._memory_cache = {}
            
            # 限制缓存大小（最多1000条）
            if len(self._memory_cache) >= 1000:
                # 删除最旧的缓存条目
                oldest_key = min(self._memory_cache.keys(), 
                               key=lambda k: self._memory_cache[k]['timestamp'])
                del self._memory_cache[oldest_key]
            
            self._memory_cache[cache_key] = {
                'data': result.data,
                'timestamp': time.time()
            }
            logger.debug(f"数据已缓存: {cache_key[:8]}...")
        
        return result
    
    def load_stock_data_with_cache(self, symbol: str, start_date: str, end_date: str,
                                   cache_ttl: int = 300) -> LoadResult:
        """
        带缓存的股票数据加载方法
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            cache_ttl: 缓存过期时间（秒）
            
        Returns:
            加载结果
        """
        query = """
            SELECT
                date as trade_date,
                open_price,
                high_price,
                low_price,
                close_price,
                volume,
                amount
            FROM akshare_stock_data
            WHERE symbol = %s
              AND date BETWEEN %s AND %s
            ORDER BY date ASC
        """

        # 使用元组作为参数（位置参数），与SQL中的%s占位符匹配
        params = (symbol, start_date, end_date)

        return self.load_with_cache(query, params, cache_ttl=cache_ttl)
    
    def clear_cache(self):
        """清除内存缓存"""
        if hasattr(self, '_memory_cache'):
            cache_size = len(self._memory_cache)
            self._memory_cache.clear()
            logger.info(f"已清除 {cache_size} 条内存缓存")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        if not hasattr(self, '_memory_cache'):
            return {
                'cache_enabled': False,
                'cache_size': 0,
                'cache_entries': 0
            }
        
        return {
            'cache_enabled': True,
            'cache_size': len(self._memory_cache),
            'cache_entries': list(self._memory_cache.keys())[:10]  # 只显示前10个
        }
    
    # ==================== 增强功能：数据质量检查 ====================
    
    def check_data_quality(self, data: pd.DataFrame, data_source: str = "postgresql") -> Dict[str, Any]:
        """
        检查数据质量
        
        使用 DataQualityMonitor 进行数据质量检查，包括完整性、准确性、一致性等。
        
        Args:
            data: 要检查的数据
            data_source: 数据源名称
            
        Returns:
            质量检查结果字典
        """
        try:
            from ..quality import DataQualityMonitor
            
            # 创建质量监控器
            quality_monitor = DataQualityMonitor()
            
            # 执行质量检查
            quality_report = quality_monitor.check_quality(data, data_source)
            
            logger.info(
                f"数据质量检查完成: 数据源={data_source}, "
                f"总分={quality_report.overall_score:.2f}, "
                f"等级={quality_report.quality_level.value}"
            )
            
            return {
                'success': True,
                'overall_score': quality_report.overall_score,
                'quality_level': quality_report.quality_level.value,
                'metrics': {
                    name: {
                        'value': metric.value,
                        'threshold': metric.threshold,
                        'status': metric.status
                    }
                    for name, metric in quality_report.metrics.items()
                },
                'anomalies': quality_report.anomalies,
                'recommendations': quality_report.recommendations,
                'alert_level': quality_report.alert_level.value if quality_report.alert_level else None
            }
            
        except ImportError as e:
            logger.warning(f"数据质量监控模块不可用: {e}")
            return {
                'success': False,
                'error': f"质量监控模块不可用: {e}",
                'overall_score': 0.0
            }
        except Exception as e:
            logger.error(f"数据质量检查失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'overall_score': 0.0
            }
    
    def load_with_quality_check(self, query: str, params: Optional[Dict[str, Any]] = None,
                               quality_threshold: float = 0.7) -> LoadResult:
        """
        带质量检查的加载方法
        
        加载数据后自动进行质量检查，如果质量得分低于阈值则记录警告。
        
        Args:
            query: SQL查询语句
            params: 查询参数
            quality_threshold: 质量阈值（默认0.7）
            
        Returns:
            加载结果（包含质量检查信息）
        """
        # 加载数据
        result = self.load(query, params)
        
        # 如果加载成功，进行质量检查
        if result.success and result.data is not None:
            quality_result = self.check_data_quality(result.data)
            
            if quality_result['success']:
                overall_score = quality_result['overall_score']
                
                # 将质量信息添加到结果元数据
                result.metadata['quality_check'] = True
                result.metadata['quality_score'] = overall_score
                result.metadata['quality_level'] = quality_result.get('quality_level')
                
                # 如果质量得分低于阈值，记录警告
                if overall_score < quality_threshold:
                    logger.warning(
                        f"数据质量得分低于阈值: {overall_score:.2f} < {quality_threshold}"
                    )
                    result.metadata['quality_warning'] = True
                else:
                    result.metadata['quality_warning'] = False
            else:
                result.metadata['quality_check'] = False
                result.metadata['quality_error'] = quality_result.get('error')
        
        return result
    
    # ==================== 增强功能：性能监控 ====================
    
    def record_performance_metric(self, metric_name: str, value: float, unit: str = "ms",
                                 metadata: Optional[Dict[str, Any]] = None):
        """
        记录性能指标
        
        Args:
            metric_name: 指标名称
            value: 指标值
            unit: 单位（默认ms）
            metadata: 额外元数据
        """
        try:
            from ..monitoring import PerformanceMonitor, PerformanceMetric
            
            # 创建性能监控器（如果不存在）
            if not hasattr(self, '_performance_monitor'):
                self._performance_monitor = PerformanceMonitor()
            
            # 创建性能指标
            metric = PerformanceMetric(
                name=metric_name,
                value=value,
                unit=unit,
                metadata=metadata or {}
            )
            
            # 记录指标
            self._performance_monitor.record_metric(metric)
            
            logger.debug(f"性能指标记录: {metric_name}={value}{unit}")
            
        except ImportError as e:
            logger.debug(f"性能监控模块不可用: {e}")
        except Exception as e:
            logger.warning(f"记录性能指标失败: {e}")
    
    def load_with_monitoring(self, query: str, params: Optional[Dict[str, Any]] = None) -> LoadResult:
        """
        带性能监控的加载方法
        
        记录加载操作的性能指标，包括加载时间、成功率等。
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            加载结果
        """
        import time
        
        start_time = time.time()
        
        # 执行加载
        result = self.load(query, params)
        
        # 计算加载时间
        load_time_ms = (time.time() - start_time) * 1000
        
        # 记录性能指标
        self.record_performance_metric(
            metric_name="postgresql_load_time",
            value=load_time_ms,
            unit="ms",
            metadata={
                'success': result.success,
                'row_count': result.row_count,
                'query_hash': self._generate_cache_key(query, params)[:8]
            }
        )
        
        # 记录成功率
        self.record_performance_metric(
            metric_name="postgresql_load_success",
            value=1.0 if result.success else 0.0,
            unit="rate",
            metadata={'query_hash': self._generate_cache_key(query, params)[:8]}
        )
        
        # 将性能信息添加到结果元数据
        result.metadata['load_time_ms'] = load_time_ms
        result.metadata['monitored'] = True
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            性能统计信息字典
        """
        try:
            if hasattr(self, '_performance_monitor'):
                return self._performance_monitor.get_stats()
            else:
                return {
                    'monitoring_enabled': False,
                    'message': '性能监控未启用'
                }
        except Exception as e:
            logger.warning(f"获取性能统计失败: {e}")
            return {
                'monitoring_enabled': False,
                'error': str(e)
            }
    
    # ==================== 增强功能：综合加载方法 ====================
    
    def load_enhanced(self, query: str, params: Optional[Dict[str, Any]] = None,
                     use_retry: bool = True, use_cache: bool = True,
                     use_quality_check: bool = True, use_monitoring: bool = True,
                     max_retries: int = 3, cache_ttl: int = 300,
                     quality_threshold: float = 0.7) -> LoadResult:
        """
        增强版加载方法（集成所有功能）
        
        集成自动重试、缓存、质量检查和性能监控的完整加载方法。
        
        Args:
            query: SQL查询语句
            params: 查询参数
            use_retry: 是否使用自动重试
            use_cache: 是否使用缓存
            use_quality_check: 是否进行质量检查
            use_monitoring: 是否进行性能监控
            max_retries: 最大重试次数
            cache_ttl: 缓存过期时间（秒）
            quality_threshold: 质量阈值
            
        Returns:
            加载结果
        """
        import time
        
        start_time = time.time()
        
        # 选择加载方法
        if use_retry and use_cache:
            # 先尝试从缓存加载
            result = self.load_with_cache(query, params, cache_ttl=cache_ttl)
            # 如果缓存未命中且失败，使用重试
            if not result.success and not result.metadata.get('cached'):
                result = self.load_with_retry(query, params, max_retries=max_retries)
                # 如果重试成功，更新缓存
                if result.success:
                    cache_key = self._generate_cache_key(query, params)
                    if not hasattr(self, '_memory_cache'):
                        self._memory_cache = {}
                    self._memory_cache[cache_key] = {
                        'data': result.data,
                        'timestamp': time.time()
                    }
        elif use_retry:
            result = self.load_with_retry(query, params, max_retries=max_retries)
        elif use_cache:
            result = self.load_with_cache(query, params, cache_ttl=cache_ttl)
        else:
            result = self.load(query, params)
        
        # 性能监控
        if use_monitoring:
            load_time_ms = (time.time() - start_time) * 1000
            self.record_performance_metric(
                metric_name="postgresql_enhanced_load_time",
                value=load_time_ms,
                unit="ms",
                metadata={
                    'success': result.success,
                    'row_count': result.row_count,
                    'use_retry': use_retry,
                    'use_cache': use_cache,
                    'use_quality_check': use_quality_check
                }
            )
            result.metadata['load_time_ms'] = load_time_ms
            result.metadata['monitored'] = True
        
        # 质量检查
        if use_quality_check and result.success and result.data is not None:
            quality_result = self.check_data_quality(result.data)
            if quality_result['success']:
                result.metadata['quality_check'] = True
                result.metadata['quality_score'] = quality_result['overall_score']
                result.metadata['quality_level'] = quality_result.get('quality_level')
                if quality_result['overall_score'] < quality_threshold:
                    result.metadata['quality_warning'] = True
                    logger.warning(f"数据质量得分较低: {quality_result['overall_score']:.2f}")
        
        return result
    
    # ==================== 第三阶段：统一接口与数据湖集成 ====================
    
    def load_to_data_lake(self, query: str, dataset_name: str,
                         params: Optional[Dict[str, Any]] = None,
                         partition_key: Optional[str] = None,
                         custom_metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        加载数据到数据湖
        
        从 PostgreSQL 加载数据并存储到数据湖，支持分区管理和元数据记录。
        
        Args:
            query: SQL查询语句
            dataset_name: 数据集名称
            params: 查询参数
            partition_key: 分区键（可选）
            custom_metadata: 自定义元数据（可选）
            
        Returns:
            存储的文件路径，失败返回 None
        """
        try:
            from ..lake.data_lake_manager import DataLakeManager, LakeConfig
            
            # 加载数据
            result = self.load(query, params)
            
            if not result.success or result.data is None:
                logger.error(f"加载数据失败，无法存储到数据湖: {result.message}")
                return None
            
            # 创建数据湖管理器（如果不存在）
            if not hasattr(self, '_data_lake_manager'):
                lake_config = LakeConfig(
                    base_path="data_lake/postgresql",
                    compression="parquet",
                    metadata_enabled=True
                )
                self._data_lake_manager = DataLakeManager(lake_config)
            
            # 构建元数据
            metadata = {
                'source': 'postgresql',
                'query': query[:200],  # 限制长度
                'timestamp': datetime.now().isoformat(),
                'row_count': result.row_count,
                'load_time_ms': result.load_time_ms
            }
            
            # 合并自定义元数据
            if custom_metadata:
                metadata.update(custom_metadata)
            
            # 存储到数据湖
            file_path = self._data_lake_manager.store_data(
                data=result.data,
                dataset_name=dataset_name,
                partition_key=partition_key,
                metadata=metadata
            )
            
            logger.info(f"数据已存储到数据湖: {file_path}")
            return file_path
            
        except ImportError as e:
            logger.warning(f"数据湖管理器不可用: {e}")
            return None
        except Exception as e:
            logger.error(f"存储到数据湖失败: {e}")
            return None
    
    def load_from_data_lake(self, dataset_name: str,
                           partition_filter: Optional[Dict[str, Any]] = None,
                           date_range: Optional[tuple] = None) -> Optional[pd.DataFrame]:
        """
        从数据湖加载数据
        
        Args:
            dataset_name: 数据集名称
            partition_filter: 分区过滤条件
            date_range: 日期范围
            
        Returns:
            数据DataFrame，失败返回 None
        """
        try:
            from ..lake.data_lake_manager import DataLakeManager, LakeConfig
            
            # 创建数据湖管理器（如果不存在）
            if not hasattr(self, '_data_lake_manager'):
                lake_config = LakeConfig(base_path="data_lake/postgresql")
                self._data_lake_manager = DataLakeManager(lake_config)
            
            # 从数据湖加载
            df = self._data_lake_manager.load_data(
                dataset_name=dataset_name,
                partition_filter=partition_filter,
                date_range=date_range
            )
            
            if df.empty:
                logger.warning(f"数据湖中没有找到数据: {dataset_name}")
                return None
            
            logger.info(f"从数据湖加载数据成功: {dataset_name}, {len(df)} 条记录")
            return df
            
        except ImportError as e:
            logger.warning(f"数据湖管理器不可用: {e}")
            return None
        except Exception as e:
            logger.error(f"从数据湖加载失败: {e}")
            return None
    
    def sync_to_data_lake(self, query: str, dataset_name: str,
                         params: Optional[Dict[str, Any]] = None,
                         sync_interval: int = 3600) -> bool:
        """
        同步数据到数据湖（带缓存检查）
        
        如果数据湖中的数据比缓存新，则跳过同步。
        
        Args:
            query: SQL查询语句
            dataset_name: 数据集名称
            params: 查询参数
            sync_interval: 最小同步间隔（秒，默认1小时）
            
        Returns:
            同步是否成功
        """
        try:
            # 检查上次同步时间
            cache_key = f"sync_{dataset_name}"
            if hasattr(self, '_memory_cache') and cache_key in self._memory_cache:
                last_sync = self._memory_cache[cache_key].get('timestamp', 0)
                if time.time() - last_sync < sync_interval:
                    logger.debug(f"跳过同步，上次同步在 {sync_interval} 秒内")
                    return True
            
            # 执行同步
            file_path = self.load_to_data_lake(query, dataset_name, params)
            
            if file_path:
                # 更新同步时间
                if not hasattr(self, '_memory_cache'):
                    self._memory_cache = {}
                self._memory_cache[cache_key] = {
                    'timestamp': time.time(),
                    'file_path': file_path
                }
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"同步到数据湖失败: {e}")
            return False
    
    def get_data_lake_info(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取数据湖信息
        
        Args:
            dataset_name: 数据集名称（可选，如果为None则返回所有数据集）
            
        Returns:
            数据湖信息字典
        """
        try:
            from ..lake.data_lake_manager import DataLakeManager, LakeConfig
            
            # 创建数据湖管理器（如果不存在）
            if not hasattr(self, '_data_lake_manager'):
                lake_config = LakeConfig(base_path="data_lake/postgresql")
                self._data_lake_manager = DataLakeManager(lake_config)
            
            if dataset_name:
                # 获取特定数据集信息
                info = self._data_lake_manager.get_dataset_info(dataset_name)
                return {
                    'dataset_name': dataset_name,
                    'info': info
                }
            else:
                # 获取所有数据集列表
                datasets = self._data_lake_manager.list_datasets()
                return {
                    'datasets': datasets,
                    'count': len(datasets)
                }
            
        except ImportError as e:
            logger.warning(f"数据湖管理器不可用: {e}")
            return {'error': str(e)}
        except Exception as e:
            logger.error(f"获取数据湖信息失败: {e}")
            return {'error': str(e)}
    
    # ==================== 统一接口方法 ====================
    
    def load_data(self, source: str = "postgresql", symbol: Optional[str] = None,
                  start_date: Optional[str] = None, end_date: Optional[str] = None,
                  use_data_lake: bool = True, **kwargs) -> LoadResult:
        """
        统一数据加载接口
        
        符合数据管理层架构设计的标准加载接口，支持从 PostgreSQL 或数据湖加载数据。
        
        Args:
            source: 数据源类型（默认 postgresql）
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            use_data_lake: 是否优先从数据湖加载
            **kwargs: 其他参数
            
        Returns:
            加载结果
        """
        # 如果指定了股票代码，使用股票数据加载
        if symbol and start_date and end_date:
            dataset_name = f"stock_{symbol}"
            
            # 优先从数据湖加载
            if use_data_lake:
                try:
                    df = self.load_from_data_lake(
                        dataset_name=dataset_name,
                        date_range=(pd.to_datetime(start_date), pd.to_datetime(end_date))
                    )
                    
                    if df is not None and not df.empty:
                        logger.info(f"从数据湖加载数据: {symbol}")
                        return LoadResult(
                            data=df,
                            success=True,
                            message="从数据湖加载成功",
                            row_count=len(df),
                            load_time_ms=0,
                            metadata={'source': 'data_lake', 'cached': True}
                        )
                except Exception as e:
                    logger.debug(f"从数据湖加载失败，尝试从数据库加载: {e}")
            
            # 从数据库加载
            return self.load_stock_data(symbol, start_date, end_date)
        
        # 通用查询加载
        query = kwargs.get('query')
        params = kwargs.get('params')
        
        if query:
            return self.load_enhanced(query, params, **kwargs)
        
        return LoadResult(
            data=None,
            success=False,
            message="缺少必要的加载参数",
            row_count=0,
            load_time_ms=0
        )
    
    def get_loader_metadata(self) -> Dict[str, Any]:
        """
        获取加载器元数据
        
        符合数据管理层架构设计的标准元数据接口。
        
        Returns:
            加载器元数据字典
        """
        return {
            'loader_type': 'PostgreSQLDataLoader',
            'source_type': 'postgresql',
            'version': '2.0',
            'capabilities': [
                'database_connection',
                'connection_pool',
                'retry_mechanism',
                'memory_cache',
                'quality_check',
                'performance_monitoring',
                'data_lake_integration'
            ],
            'config': {
                'host': self._db_config.get('host') if hasattr(self, '_db_config') else 'unknown',
                'database': self._db_config.get('database') if hasattr(self, '_db_config') else 'unknown'
            },
            'stats': {
                'cache_enabled': hasattr(self, '_memory_cache'),
                'data_lake_enabled': hasattr(self, '_data_lake_manager'),
                'performance_monitoring_enabled': hasattr(self, '_performance_monitor')
            }
        }


# 全局加载器实例（单例模式）
_postgresql_loader: Optional[PostgreSQLDataLoader] = None


def get_postgresql_loader(config: Optional[DataLoaderConfig] = None) -> PostgreSQLDataLoader:
    """
    获取PostgreSQL数据加载器实例（单例模式）
    
    Args:
        config: 加载器配置，如果为None则使用默认配置
        
    Returns:
        PostgreSQLDataLoader实例
    """
    global _postgresql_loader
    
    if _postgresql_loader is None:
        _postgresql_loader = PostgreSQLDataLoader(config)
    
    return _postgresql_loader


def close_postgresql_loader():
    """关闭全局PostgreSQL数据加载器实例"""
    global _postgresql_loader
    
    if _postgresql_loader:
        _postgresql_loader.close()
        _postgresql_loader = None
        logger.info("PostgreSQL数据加载器已关闭")

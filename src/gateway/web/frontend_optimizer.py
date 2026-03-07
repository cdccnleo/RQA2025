"""
frontend_optimizer.py

前端性能优化组件 - FrontendOptimizer

提供前端性能优化功能，包括：
- API响应压缩和缓存
- 数据分页和虚拟滚动支持
- 响应数据精简
- WebSocket连接优化
- 静态资源缓存策略

特性：
- 智能数据分页
- 响应数据字段筛选
- Gzip/Brotli压缩
- ETag缓存验证
- 批量API请求合并

作者: RQA2025 Team
日期: 2026-02-13
"""

import gzip
import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from functools import wraps

import numpy as np

# 配置日志
logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """压缩类型枚举"""
    NONE = "none"
    GZIP = "gzip"
    BROTLI = "brotli"


class CacheControl(Enum):
    """缓存控制策略"""
    NO_CACHE = "no-cache"
    NO_STORE = "no-store"
    PRIVATE = "private"
    PUBLIC = "public"
    MAX_AGE = "max-age"


@dataclass
class PaginationConfig:
    """分页配置"""
    default_page_size: int = 20
    max_page_size: int = 100
    min_page_size: int = 5
    enable_cursor_pagination: bool = True
    cursor_ttl_seconds: int = 300


@dataclass
class ResponseCacheConfig:
    """响应缓存配置"""
    enabled: bool = True
    default_ttl_seconds: int = 60
    max_cache_size: int = 1000
    compression_threshold_bytes: int = 1024  # 1KB
    compression_type: CompressionType = CompressionType.GZIP


@dataclass
class DataFilterConfig:
    """数据过滤配置"""
    enabled: bool = True
    max_depth: int = 3
    default_fields: Optional[List[str]] = None
    exclude_fields: Optional[List[str]] = None


@dataclass
class FrontendOptimizerConfig:
    """前端优化器配置"""
    pagination: PaginationConfig = field(default_factory=PaginationConfig)
    response_cache: ResponseCacheConfig = field(default_factory=ResponseCacheConfig)
    data_filter: DataFilterConfig = field(default_factory=DataFilterConfig)
    enable_batch_requests: bool = True
    enable_etags: bool = True
    enable_response_compression: bool = True


@dataclass
class PaginatedResponse:
    """分页响应"""
    data: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool
    cursor: Optional[str] = None


@dataclass
class BatchRequest:
    """批量请求"""
    request_id: str
    method: str
    endpoint: str
    params: Optional[Dict[str, Any]] = None
    body: Optional[Dict[str, Any]] = None


@dataclass
class BatchResponse:
    """批量响应"""
    request_id: str
    status: int
    data: Any
    error: Optional[str] = None
    cached: bool = False


class ResponseCache:
    """响应缓存 - 缓存API响应"""
    
    def __init__(self, config: ResponseCacheConfig):
        self.config = config
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._etag_map: Dict[str, str] = {}  # URL -> ETag
        self._access_times: Dict[str, float] = {}
    
    def _generate_key(self, endpoint: str, params: Optional[Dict] = None) -> str:
        """生成缓存键"""
        key_data = {
            'endpoint': endpoint,
            'params': params or {}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _generate_etag(self, data: Any) -> str:
        """生成ETag"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return f'"{hashlib.md5(data_str.encode()).hexdigest()}"'
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Tuple[Any, Optional[str]]]:
        """获取缓存响应"""
        if not self.config.enabled:
            return None
        
        key = self._generate_key(endpoint, params)
        
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        # 检查是否过期
        if time.time() - entry['created_at'] > self.config.default_ttl_seconds:
            del self._cache[key]
            if key in self._etag_map:
                del self._etag_map[key]
            if key in self._access_times:
                del self._access_times[key]
            return None
        
        self._access_times[key] = time.time()
        etag = self._etag_map.get(key)
        
        return entry['data'], etag
    
    def set(self, endpoint: str, data: Any, params: Optional[Dict] = None) -> str:
        """设置缓存响应"""
        if not self.config.enabled:
            return ""
        
        # 清理过期项
        self._cleanup_expired()
        
        # 如果缓存已满，移除最久未访问的项
        if len(self._cache) >= self.config.max_cache_size:
            self._evict_oldest()
        
        key = self._generate_key(endpoint, params)
        etag = self._generate_etag(data)
        
        self._cache[key] = {
            'data': data,
            'created_at': time.time(),
            'endpoint': endpoint,
            'params': params
        }
        self._etag_map[key] = etag
        self._access_times[key] = time.time()
        
        return etag
    
    def validate_etag(self, endpoint: str, params: Optional[Dict], client_etag: str) -> bool:
        """验证ETag是否匹配"""
        key = self._generate_key(endpoint, params)
        cached_etag = self._etag_map.get(key)
        
        if cached_etag and cached_etag == client_etag:
            return True
        
        return False
    
    def invalidate(self, endpoint_pattern: Optional[str] = None) -> int:
        """使缓存项失效"""
        if endpoint_pattern is None:
            count = len(self._cache)
            self._cache.clear()
            self._etag_map.clear()
            self._access_times.clear()
            return count
        
        count = 0
        keys_to_remove = []
        
        for key, entry in self._cache.items():
            if endpoint_pattern in entry.get('endpoint', ''):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self._cache[key]
            if key in self._etag_map:
                del self._etag_map[key]
            if key in self._access_times:
                del self._access_times[key]
            count += 1
        
        return count
    
    def _cleanup_expired(self) -> None:
        """清理过期项"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry['created_at'] > self.config.default_ttl_seconds
        ]
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._etag_map:
                del self._etag_map[key]
            if key in self._access_times:
                del self._access_times[key]
    
    def _evict_oldest(self) -> None:
        """移除最久未访问的项"""
        if self._access_times:
            oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            del self._cache[oldest_key]
            if oldest_key in self._etag_map:
                del self._etag_map[oldest_key]
            del self._access_times[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            'size': len(self._cache),
            'max_size': self.config.max_cache_size,
            'ttl_seconds': self.config.default_ttl_seconds,
            'compression_enabled': self.config.compression_type != CompressionType.NONE
        }


class DataPaginator:
    """数据分页器"""
    
    def __init__(self, config: PaginationConfig):
        self.config = config
        self._cursors: Dict[str, Dict[str, Any]] = {}
    
    def paginate(
        self,
        data: List[Any],
        page: int = 1,
        page_size: Optional[int] = None,
        use_cursor: bool = False
    ) -> PaginatedResponse:
        """
        对数据进行分页
        
        Args:
            data: 完整数据列表
            page: 页码（从1开始）
            page_size: 每页大小
            use_cursor: 是否使用游标分页
            
        Returns:
            分页响应
        """
        # 验证并调整分页参数
        page_size = self._validate_page_size(page_size)
        page = max(1, page)
        
        total = len(data)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        
        # 确保页码不超出范围
        page = min(page, total_pages)
        
        # 计算切片范围
        start_index = (page - 1) * page_size
        end_index = min(start_index + page_size, total)
        
        # 获取当前页数据
        page_data = data[start_index:end_index]
        
        # 生成游标（如果使用）
        cursor = None
        if use_cursor and self.config.enable_cursor_pagination and end_index < total:
            cursor = self._generate_cursor(data, end_index)
        
        return PaginatedResponse(
            data=page_data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=end_index < total,
            has_prev=page > 1,
            cursor=cursor
        )
    
    def paginate_with_cursor(
        self,
        data: List[Any],
        cursor: Optional[str] = None,
        page_size: Optional[int] = None
    ) -> PaginatedResponse:
        """
        使用游标分页
        
        Args:
            data: 完整数据列表
            cursor: 游标
            page_size: 每页大小
            
        Returns:
            分页响应
        """
        page_size = self._validate_page_size(page_size)
        
        # 从游标获取起始位置
        start_index = 0
        if cursor and cursor in self._cursors:
            cursor_data = self._cursors[cursor]
            if time.time() - cursor_data['created_at'] < self.config.cursor_ttl_seconds:
                start_index = cursor_data['index']
            else:
                # 游标过期，移除
                del self._cursors[cursor]
        
        total = len(data)
        end_index = min(start_index + page_size, total)
        
        page_data = data[start_index:end_index]
        
        # 生成下一页游标
        next_cursor = None
        if end_index < total:
            next_cursor = self._generate_cursor(data, end_index)
        
        # 计算页码（用于兼容性）
        page = (start_index // page_size) + 1
        total_pages = (total + page_size - 1) // page_size if total > 0 else 1
        
        return PaginatedResponse(
            data=page_data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=end_index < total,
            has_prev=start_index > 0,
            cursor=next_cursor
        )
    
    def _validate_page_size(self, page_size: Optional[int]) -> int:
        """验证并调整页大小"""
        if page_size is None:
            return self.config.default_page_size
        
        return max(self.config.min_page_size, min(page_size, self.config.max_page_size))
    
    def _generate_cursor(self, data: List[Any], index: int) -> str:
        """生成游标"""
        cursor_data = {
            'index': index,
            'timestamp': time.time(),
            'hash': hashlib.md5(str(data[index:index+1]).encode()).hexdigest()[:8]
        }
        cursor_str = json.dumps(cursor_data)
        cursor = hashlib.md5(cursor_str.encode()).hexdigest()
        
        # 存储游标
        self._cursors[cursor] = {
            'index': index,
            'created_at': time.time()
        }
        
        # 清理过期游标
        self._cleanup_expired_cursors()
        
        return cursor
    
    def _cleanup_expired_cursors(self) -> None:
        """清理过期游标"""
        current_time = time.time()
        expired_cursors = [
            cursor for cursor, data in self._cursors.items()
            if current_time - data['created_at'] > self.config.cursor_ttl_seconds
        ]
        
        for cursor in expired_cursors:
            del self._cursors[cursor]


class DataFilter:
    """数据过滤器 - 筛选响应字段"""
    
    def __init__(self, config: DataFilterConfig):
        self.config = config
    
    def filter_fields(
        self,
        data: Any,
        fields: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        max_depth: Optional[int] = None
    ) -> Any:
        """
        过滤数据字段
        
        Args:
            data: 原始数据
            fields: 要包含的字段列表
            exclude: 要排除的字段列表
            max_depth: 最大递归深度
            
        Returns:
            过滤后的数据
        """
        if not self.config.enabled:
            return data
        
        max_depth = max_depth or self.config.max_depth
        
        # 使用默认配置
        if fields is None:
            fields = self.config.default_fields
        if exclude is None:
            exclude = self.config.exclude_fields
        
        return self._filter_recursive(data, fields, exclude, 0, max_depth)
    
    def _filter_recursive(
        self,
        data: Any,
        fields: Optional[List[str]],
        exclude: Optional[List[str]],
        depth: int,
        max_depth: int
    ) -> Any:
        """递归过滤数据"""
        # 检查深度
        if depth >= max_depth:
            return data
        
        # 处理字典
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # 检查排除列表
                if exclude and key in exclude:
                    continue
                
                # 检查包含列表
                if fields and key not in fields:
                    continue
                
                # 递归处理值
                result[key] = self._filter_recursive(value, fields, exclude, depth + 1, max_depth)
            
            return result
        
        # 处理列表
        if isinstance(data, list):
            return [
                self._filter_recursive(item, fields, exclude, depth + 1, max_depth)
                for item in data
            ]
        
        # 基本类型，直接返回
        return data


class ResponseCompressor:
    """响应压缩器"""
    
    def __init__(self, config: ResponseCacheConfig):
        self.config = config
    
    def compress(self, data: Any, compression_type: Optional[CompressionType] = None) -> Tuple[bytes, CompressionType]:
        """
        压缩数据
        
        Args:
            data: 要压缩的数据
            compression_type: 压缩类型
            
        Returns:
            (压缩后的数据, 实际使用的压缩类型)
        """
        compression_type = compression_type or self.config.compression_type
        
        # 序列化数据
        if isinstance(data, (dict, list)):
            data_bytes = json.dumps(data, default=str).encode('utf-8')
        elif isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode('utf-8')
        
        # 检查是否需要压缩
        if len(data_bytes) < self.config.compression_threshold_bytes:
            return data_bytes, CompressionType.NONE
        
        # 压缩
        if compression_type == CompressionType.GZIP:
            try:
                compressed = gzip.compress(data_bytes, compresslevel=6)
                return compressed, CompressionType.GZIP
            except Exception as e:
                logger.warning(f"Gzip压缩失败: {e}")
                return data_bytes, CompressionType.NONE
        
        # Brotli压缩（如果可用）
        if compression_type == CompressionType.BROTLI:
            try:
                import brotli
                compressed = brotli.compress(data_bytes)
                return compressed, CompressionType.BROTLI
            except ImportError:
                logger.debug("Brotli不可用，使用Gzip")
                try:
                    compressed = gzip.compress(data_bytes, compresslevel=6)
                    return compressed, CompressionType.GZIP
                except Exception as e:
                    logger.warning(f"Gzip压缩失败: {e}")
                    return data_bytes, CompressionType.NONE
        
        return data_bytes, CompressionType.NONE
    
    def decompress(self, data: bytes, compression_type: CompressionType) -> bytes:
        """
        解压数据
        
        Args:
            data: 压缩的数据
            compression_type: 压缩类型
            
        Returns:
            解压后的数据
        """
        if compression_type == CompressionType.NONE:
            return data
        
        if compression_type == CompressionType.GZIP:
            try:
                return gzip.decompress(data)
            except Exception as e:
                logger.warning(f"Gzip解压失败: {e}")
                return data
        
        if compression_type == CompressionType.BROTLI:
            try:
                import brotli
                return brotli.decompress(data)
            except ImportError:
                logger.warning("Brotli不可用")
                return data
        
        return data


class FrontendOptimizer:
    """
    前端性能优化器
    
    提供前端性能优化功能，包括响应缓存、数据分页、
    字段过滤和响应压缩。
    
    Attributes:
        config: 优化器配置
        cache: 响应缓存
        paginator: 数据分页器
        filter: 数据过滤器
        compressor: 响应压缩器
        
    Example:
        >>> config = FrontendOptimizerConfig()
        >>> optimizer = FrontendOptimizer(config)
        >>> 
        >>> # 缓存响应
        >>> etag = optimizer.cache_response("/api/users", users_data)
        >>> 
        >>> # 分页数据
        >>> paginated = optimizer.paginate(users, page=1, page_size=20)
        >>> 
        >>> # 过滤字段
        >>> filtered = optimizer.filter_fields(user_data, fields=["id", "name", "email"])
    """
    
    def __init__(self, config: Optional[FrontendOptimizerConfig] = None):
        """
        初始化前端优化器
        
        Args:
            config: 优化器配置，如果为None则使用默认配置
        """
        self.config = config or FrontendOptimizerConfig()
        
        # 初始化组件
        self.cache = ResponseCache(self.config.response_cache)
        self.paginator = DataPaginator(self.config.pagination)
        self.filter = DataFilter(self.config.data_filter)
        self.compressor = ResponseCompressor(self.config.response_cache)
        
        # 统计
        self._total_requests = 0
        self._cached_responses = 0
        self._compressed_responses = 0
        
        logger.info("FrontendOptimizer initialized")
    
    def cache_response(
        self,
        endpoint: str,
        data: Any,
        params: Optional[Dict] = None,
        ttl: Optional[int] = None
    ) -> str:
        """
        缓存响应
        
        Args:
            endpoint: API端点
            data: 响应数据
            params: 请求参数
            ttl: 缓存时间（秒）
            
        Returns:
            ETag
        """
        return self.cache.set(endpoint, data, params)
    
    def get_cached_response(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        client_etag: Optional[str] = None
    ) -> Optional[Tuple[Any, bool]]:
        """
        获取缓存的响应
        
        Args:
            endpoint: API端点
            params: 请求参数
            client_etag: 客户端ETag
            
        Returns:
            (数据, 是否未修改) 或 None
        """
        self._total_requests += 1
        
        # 验证ETag
        if client_etag and self.config.enable_etags:
            if self.cache.validate_etag(endpoint, params, client_etag):
                self._cached_responses += 1
                return None, True  # 未修改
        
        # 获取缓存数据
        cached = self.cache.get(endpoint, params)
        if cached:
            data, etag = cached
            self._cached_responses += 1
            return data, False
        
        return None, False
    
    def paginate(
        self,
        data: List[Any],
        page: int = 1,
        page_size: Optional[int] = None,
        use_cursor: bool = False
    ) -> PaginatedResponse:
        """
        分页数据
        
        Args:
            data: 完整数据列表
            page: 页码
            page_size: 每页大小
            use_cursor: 是否使用游标
            
        Returns:
            分页响应
        """
        return self.paginator.paginate(data, page, page_size, use_cursor)
    
    def paginate_with_cursor(
        self,
        data: List[Any],
        cursor: Optional[str] = None,
        page_size: Optional[int] = None
    ) -> PaginatedResponse:
        """
        使用游标分页
        
        Args:
            data: 完整数据列表
            cursor: 游标
            page_size: 每页大小
            
        Returns:
            分页响应
        """
        return self.paginator.paginate_with_cursor(data, cursor, page_size)
    
    def filter_fields(
        self,
        data: Any,
        fields: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None
    ) -> Any:
        """
        过滤数据字段
        
        Args:
            data: 原始数据
            fields: 要包含的字段
            exclude: 要排除的字段
            
        Returns:
            过滤后的数据
        """
        return self.filter.filter_fields(data, fields, exclude)
    
    def compress_response(self, data: Any) -> Tuple[bytes, CompressionType]:
        """
        压缩响应
        
        Args:
            data: 响应数据
            
        Returns:
            (压缩后的数据, 压缩类型)
        """
        if not self.config.enable_response_compression:
            if isinstance(data, bytes):
                return data, CompressionType.NONE
            elif isinstance(data, str):
                return data.encode('utf-8'), CompressionType.NONE
            else:
                return json.dumps(data, default=str).encode('utf-8'), CompressionType.NONE
        
        compressed, compression_type = self.compressor.compress(data)
        
        if compression_type != CompressionType.NONE:
            self._compressed_responses += 1
        
        return compressed, compression_type
    
    def invalidate_cache(self, endpoint_pattern: Optional[str] = None) -> int:
        """
        使缓存失效
        
        Args:
            endpoint_pattern: 端点匹配模式
            
        Returns:
            失效的缓存数量
        """
        return self.cache.invalidate(endpoint_pattern)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取优化器统计
        
        Returns:
            统计信息字典
        """
        return {
            'total_requests': self._total_requests,
            'cached_responses': self._cached_responses,
            'compressed_responses': self._compressed_responses,
            'cache_stats': self.cache.get_stats(),
            'cache_hit_rate': self._cached_responses / max(self._total_requests, 1)
        }


# 全局优化器实例（单例模式）
_global_optimizer: Optional[FrontendOptimizer] = None


def get_global_optimizer(config: Optional[FrontendOptimizerConfig] = None) -> FrontendOptimizer:
    """
    获取全局前端优化器实例
    
    Args:
        config: 优化器配置，仅在第一次调用时使用
        
    Returns:
        全局优化器实例
    """
    global _global_optimizer
    
    if _global_optimizer is None:
        _global_optimizer = FrontendOptimizer(config)
    
    return _global_optimizer


def clear_global_optimizer():
    """清除全局优化器实例"""
    global _global_optimizer
    _global_optimizer = None

"""
查询执行器组件

负责执行不同类型的查询请求，包括实时、历史、聚合和跨存储查询。
"""

import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


def _safe_logger_log(level: int, message: str) -> None:
    """在单测环境中安全输出日志，防止 handler.level 被 mock 后抛出 TypeError。"""
    seen_handlers = set()
    visited_loggers = set()
    current_logger = logger
    depth = 0

    while current_logger and id(current_logger) not in visited_loggers and depth < 10:
        visited_loggers.add(id(current_logger))
        depth += 1

        handlers_attr = getattr(current_logger, "handlers", None)
        if isinstance(handlers_attr, (list, tuple, set)):
            handlers = list(handlers_attr)
        elif isinstance(handlers_attr, logging.Handler):
            handlers = [handlers_attr]
        else:
            handlers = []

        for handler in handlers:
            if id(handler) in seen_handlers:
                continue
            seen_handlers.add(id(handler))
            level_value = getattr(handler, "level", logging.NOTSET)
            if not isinstance(level_value, int):
                try:
                    handler.setLevel(logging.NOTSET)  # type: ignore[attr-defined]
                except Exception:
                    try:
                        handler.level = logging.NOTSET  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            handler.__dict__["level"] = logging.NOTSET  # type: ignore[attr-defined]
                        except Exception:
                            pass
            if not isinstance(getattr(handler, "level", None), int):
                try:
                    object.__setattr__(handler, "level", logging.NOTSET)  # type: ignore[attr-defined]
                except Exception:
                    pass

        if not getattr(current_logger, "propagate", True):
            break

        parent_logger = getattr(current_logger, "parent", None)
        if parent_logger is None or parent_logger is current_logger:
            break
        if not isinstance(parent_logger, logging.Logger):
            break
        current_logger = parent_logger

    try:
        if level == logging.INFO and hasattr(logger, "info"):
            logger.info(message)
        elif level == logging.ERROR and hasattr(logger, "error"):
            logger.error(message)
        elif level == logging.WARNING and hasattr(logger, "warning"):
            logger.warning(message)
        elif level == logging.DEBUG and hasattr(logger, "debug"):
            logger.debug(message)
        else:
            logger.log(level, message)
    except TypeError:
        logging.getLogger(logger.name).log(level, message)


class QueryType(Enum):
    REALTIME = "realtime"
    HISTORICAL = "historical"
    AGGREGATED = "aggregated"
    CROSS_STORAGE = "cross_storage"
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"


class StorageType(Enum):
    INFLUXDB = "influxdb"
    PARQUET = "parquet"
    REDIS = "redis"
    HYBRID = "hybrid"


class QueryRequest:
    """查询请求对象，接受灵活参数以兼容历史用法"""

    __slots__ = ("query_id", "query_type", "storage_type", "params", "options")

    def __init__(
        self,
        query_id: str,
        query_type: Any,
        storage_type: Any = None,
        params: Optional[Dict[str, Any]] = None,
        options: Optional[Dict[str, Any]] = None,
        **extra: Any,
    ) -> None:
        self.query_id = query_id
        self.query_type = query_type
        self.storage_type = storage_type
        self.params = params or {}
        if not isinstance(self.params, dict):
            raise ValueError("params 必须是字典")
        self.options = options or {}
        if not isinstance(self.options, dict):
            raise ValueError("options 必须是字典或 None")
        # 兼容额外字段
        for key, value in extra.items():
            self.options[key] = value


class QueryResult:
    """查询结果对象"""

    __slots__ = ("query_id", "data", "metadata", "success", "error")

    def __init__(
        self,
        query_id: Optional[str] = None,
        data: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error: Optional[str] = None,
        **extra: Any,
    ) -> None:
        self.query_id = query_id or "unknown"
        self.data = data
        self.metadata: Dict[str, Any] = metadata.copy() if metadata else {}
        self.success = success
        self.error = error
        if extra:
            self.metadata.update(extra)

    def add_metadata(self, **kwargs: Any) -> None:
        self.metadata.update(kwargs)


class QueryExecutor:
    """查询执行器"""

    def __init__(self, storage_adapters: Optional[Dict[Any, Any]] = None, max_workers: int = 10):
        """
        初始化查询执行器

        Args:
            storage_adapters: 存储适配器字典
            max_workers: 最大并发工作线程数
        """
        self.storage_adapters: Dict[StorageType, Any] = {}
        if storage_adapters:
            for key, adapter in storage_adapters.items():
                try:
                    storage_type = self._normalize_storage_type(key)
                except ValueError:
                    # 无法识别的键，忽略以避免破坏执行
                    continue
                self.storage_adapters[storage_type] = adapter
        self.query_executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute_query(self, request: QueryRequest) -> QueryResult:
        """
        执行查询请求

        Args:
            request: 查询请求

        Returns:
            查询结果
        """
        query_type = request.query_type
        if not isinstance(query_type, QueryType):
            try:
                query_type = QueryType(query_type)
            except ValueError:
                raise ValueError(f"不支持的查询类型: {request.query_type}")
        request.query_type = query_type

        storage_type_input = request.storage_type
        if storage_type_input is None:
            storage_type_input = self._resolve_default_storage()
        storage_type = self._normalize_storage_type(storage_type_input)
        request.storage_type = storage_type

        if query_type == QueryType.REALTIME:
            return self._execute_realtime_query(request)
        if query_type == QueryType.HISTORICAL:
            return self._execute_historical_query(request)
        if query_type == QueryType.AGGREGATED:
            return self._execute_aggregated_query(request)
        if query_type == QueryType.CROSS_STORAGE:
            return self._execute_cross_storage_query(request)
        raise ValueError(f"不支持的查询类型: {query_type}")

    def execute_batch_queries(self, requests: List[QueryRequest]) -> List[QueryResult]:
        """
        批量执行查询请求

        Args:
            requests: 查询请求列表

        Returns:
            查询结果列表
        """
        futures = [
            self.query_executor.submit(self.execute_query, request)
            for request in requests
        ]

        results: List[QueryResult] = []
        for future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as exc:
                _safe_logger_log(logging.ERROR, f"查询执行失败: {exc}")
                error_result = QueryResult(query_id="error", data=None, success=False, error=str(exc))
                error_result.add_metadata(error=str(exc))
                results.append(error_result)
        return results

    def _execute_realtime_query(self, request: QueryRequest) -> QueryResult:
        adapter = self._get_adapter(request.storage_type)
        data = adapter.query(request.params)
        result = QueryResult(query_id=request.query_id, data=data)
        result.add_metadata(query_type="realtime", storage=request.storage_type.value)
        return result

    def _execute_historical_query(self, request: QueryRequest) -> QueryResult:
        adapter = self._get_adapter(request.storage_type)
        if not hasattr(adapter, "query_historical"):
            raise ValueError(f"存储适配器 {request.storage_type.value} 不支持历史查询")
        data = adapter.query_historical(request.params)
        result = QueryResult(query_id=request.query_id, data=data)
        result.add_metadata(query_type="historical", storage=request.storage_type.value)
        return result

    def _execute_aggregated_query(self, request: QueryRequest) -> QueryResult:
        adapter = self._get_adapter(request.storage_type)
        if not hasattr(adapter, "aggregate"):
            raise ValueError(f"存储适配器 {request.storage_type.value} 不支持聚合查询")
        data = adapter.aggregate(request.params)
        result = QueryResult(query_id=request.query_id, data=data)
        result.add_metadata(query_type="aggregated", storage=request.storage_type.value)
        return result

    def _execute_cross_storage_query(self, request: QueryRequest) -> QueryResult:
        results: List[Tuple[StorageType, Any]] = []
        for storage_type, adapter in self.storage_adapters.items():
            try:
                data = adapter.query(request.params)
                results.append((storage_type, data))
            except Exception as exc:
                _safe_logger_log(logging.WARNING, f"存储 {storage_type} 查询失败: {exc}")

        merged_data = self._merge_results(results)
        result = QueryResult(query_id=request.query_id, data=merged_data)
        result.add_metadata(query_type="cross_storage", storages=[stype.value for stype in self.storage_adapters.keys()])
        return result

    def _merge_results(self, results: List[Tuple[StorageType, Any]]) -> Any:
        if not results:
            return None
        if len(results) == 1:
            return results[0][1]
        merged = []
        for _, data in results:
            if data is not None:
                merged.append(data)
        return merged

    def shutdown(self) -> None:
        """关闭执行器"""
        self.query_executor.shutdown(wait=True)

    def _get_adapter(self, storage_type: StorageType) -> Any:
        adapter = self.storage_adapters.get(storage_type)
        if adapter is None:
            raise ValueError(f"未找到存储适配器: {storage_type.value if isinstance(storage_type, StorageType) else storage_type}")
        return adapter

    @staticmethod
    def _normalize_storage_type(storage_type: Any) -> StorageType:
        if isinstance(storage_type, StorageType):
            return storage_type
        try:
            return StorageType(storage_type)
        except ValueError as exc:
            raise ValueError(f"不支持的存储类型: {storage_type}") from exc

    def _resolve_default_storage(self) -> StorageType:
        if self.storage_adapters:
            return next(iter(self.storage_adapters.keys()))
        return StorageType.INFLUXDB


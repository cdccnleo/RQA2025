"""
查询请求验证器组件

负责验证查询请求的有效性和参数完整性。
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def _safe_log(level: int, message: str) -> None:
    """在测试环境中保持日志兼容性，避免 handler.level 被 mock 成非整型导致报错。"""
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
                    handler.setLevel(logging.NOTSET)
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

try:
    from .unified_query import QueryRequest, QueryType, StorageType
except ImportError:
    # 如果导入失败，定义基础类型
    from dataclasses import dataclass
    from enum import Enum
    
    class QueryType(Enum):
        REALTIME = "realtime"
        HISTORICAL = "historical"
        AGGREGATED = "aggregated"
        CROSS_STORAGE = "cross_storage"
    
    class StorageType(Enum):
        INFLUXDB = "influxdb"
        PARQUET = "parquet"
        REDIS = "redis"
        HYBRID = "hybrid"
    
    @dataclass
    class QueryRequest:
        query_id: str
        query_type: QueryType
        storage_type: StorageType
        params: Dict[str, Any]


class QueryValidator:
    """查询请求验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.validation_rules = self._initialize_validation_rules()
    
    def validate_request(self, request: QueryRequest) -> bool:
        """
        验证单个查询请求
        
        Args:
            request: 查询请求
            
        Returns:
            是否有效
        """
        if not request:
            _safe_log(logging.ERROR, "查询请求不能为空")
            return False
        
        # 验证查询类型
        if not self._validate_query_type(request):
            return False
        
        # 验证存储类型
        if not self._validate_storage_type(request):
            return False
        
        # 验证参数
        if not self._validate_params(request):
            return False
        
        return True
    
    def validate(self, request: QueryRequest) -> bool:
        """
        验证查询请求（简化接口）
        
        Args:
            request: 查询请求
            
        Returns:
            是否有效
        """
        return self.validate_request(request)
    
    def validate_requests(self, requests: List[QueryRequest]) -> bool:
        """
        验证批量查询请求
        
        Args:
            requests: 查询请求列表
            
        Returns:
            是否全部有效
        """
        if not requests:
            _safe_log(logging.ERROR, "查询请求列表不能为空")
            return False
        
        for request in requests:
            if not self.validate_request(request):
                return False
        
        return True
    
    def _validate_query_type(self, request: QueryRequest) -> bool:
        """验证查询类型"""
        try:
            if not isinstance(request.query_type, QueryType):
                _safe_log(logging.ERROR, f"无效的查询类型: {request.query_type}")
                return False
            return True
        except Exception as e:
            _safe_log(logging.ERROR, f"查询类型验证失败: {e}")
            return False
    
    def _validate_storage_type(self, request: QueryRequest) -> bool:
        """验证存储类型"""
        try:
            if not isinstance(request.storage_type, StorageType):
                _safe_log(logging.ERROR, f"无效的存储类型: {request.storage_type}")
                return False
            return True
        except Exception as e:
            _safe_log(logging.ERROR, f"存储类型验证失败: {e}")
            return False
    
    def _validate_params(self, request: QueryRequest) -> bool:
        """验证查询参数"""
        if not request.params:
            _safe_log(logging.WARNING, "查询参数为空")
            return True  # 空参数也是有效的
        
        if not isinstance(request.params, dict):
            _safe_log(logging.ERROR, "查询参数必须是字典类型")
            return False
        
        return True
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """初始化验证规则"""
        return {
            'required_params': ['query_id', 'query_type', 'storage_type'],
            'optional_params': ['timeout', 'limit', 'offset'],
        }


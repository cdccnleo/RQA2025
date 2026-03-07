"""
RQA2025 数据层接口定义

数据层提供统一的数据访问和管理能力，
通过标准化的接口定义确保数据服务的可扩展性和一致性。
"""

from typing import Any
from typing import TYPE_CHECKING

# 基础接口尽量惰性/容错导入，避免在仅使用 `interfaces.api` 时引发不必要的导入失败
try:
    from .IDataModel import IDataModel  # type: ignore
except Exception:  # pragma: no cover - 兼容旧环境
    class IDataModel:  # type: ignore
        ...

try:
    from .loader import IDataLoader  # type: ignore
except Exception:  # pragma: no cover
    class IDataLoader:  # type: ignore
        ...

try:
    from .data_interfaces import (  # type: ignore
        DataRequest,
        DataResponse,
        DataQualityReport,
        IDataAdapter,
        IMarketDataProvider,
        IDataQualityManager,
        IDataStorage,
        IDataCache,
        IDataTransformer,
        IDataServiceProvider,
        IDataProcessor,
    )
except Exception:  # pragma: no cover
    # 轻量占位，保证 import 不崩溃
    DataRequest = DataResponse = DataQualityReport = object  # type: ignore
    class IDataAdapter: ...
    class IMarketDataProvider: ...
    class IDataQualityManager: ...
    class IDataStorage: ...
    class IDataCache: ...
    class IDataTransformer: ...
    class IDataServiceProvider: ...
    class IDataProcessor: ...

try:
    from .standard_interfaces import (  # type: ignore
        IDataValidator,
        IDataRegistry,
    )
except Exception:  # pragma: no cover
    # 占位 Protocol，避免因标准接口导入失败而阻断其他模块使用
    class IDataValidator:  # type: ignore
        def validate(self, data: Any, data_type: str) -> Any: ...
        def get_validation_rules(self, data_type: str) -> Any: ...

    class IDataRegistry:  # type: ignore
        def register_data_source(self, name: str, source_config: Any) -> bool: ...
        def get_data_source(self, name: str) -> Any: ...
        def list_data_sources(self) -> Any: ...
        def unregister_data_source(self, name: str) -> bool: ...

__all__ = [
    # 数据结构
    'DataRequest',
    'DataResponse',
    'DataQualityReport',

    # 核心接口
    'IDataAdapter',
    'IMarketDataProvider',
    'IDataQualityManager',
    'IDataStorage',
    'IDataCache',
    'IDataTransformer',
    'IDataServiceProvider',
    'IDataProcessor',
    'IDataModel',
    'IDataLoader',

    # 标准接口
    'IDataValidator',
    'IDataRegistry',
]

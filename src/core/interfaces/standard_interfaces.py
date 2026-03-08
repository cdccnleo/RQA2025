"""
兼容适配层：src.interfaces.standard_interfaces
统一重导出 src.data.interfaces.standard_interfaces，解决并行测试下的导入竞态。
"""

from importlib import import_module as _import_module  # noqa

_std = _import_module("src.data.interfaces.standard_interfaces")

DataSourceType = getattr(_std, "DataSourceType", None)
IDataValidator = getattr(_std, "IDataValidator", None)
IDataRegistry = getattr(_std, "IDataRegistry", None)
IDataAdapter = getattr(_std, "IDataAdapter", None)
IDataCache = getattr(_std, "IDataCache", None)
DataRequest = getattr(_std, "DataRequest", None)
DataResponse = getattr(_std, "DataResponse", None)

__all__ = [
    "DataSourceType",
    "IDataValidator",
    "IDataRegistry",
    "IDataAdapter",
    "IDataCache",
    "DataRequest",
    "DataResponse",
]



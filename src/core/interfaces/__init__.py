"""
兼容适配层：将 src.interfaces.* 映射到 src.data.interfaces.*
用于并行测试环境下的稳定导入，避免子进程模块桩竞态。
"""

from importlib import import_module as _import_module  # noqa

# 直接重导出 data 层接口模块
_std = _import_module("src.data.interfaces.standard_interfaces")
_di = _import_module("src.data.interfaces.data_interfaces")

# 暴露常用符号
DataSourceType = getattr(_std, "DataSourceType", None)
IDataValidator = getattr(_std, "IDataValidator", None)
IDataRegistry = getattr(_std, "IDataRegistry", None)

DataRequest = getattr(_di, "DataRequest", None)
DataResponse = getattr(_di, "DataResponse", None)

# 兼容导出：分布式数据加载协议（用于部分模块的类型约束）
try:
    from typing import Protocol, Any, Dict, List, Optional, Callable  # type: ignore
except Exception:  # pragma: no cover
    class Protocol:  # type: ignore
        pass
    Any = Dict = List = Optional = Callable = object  # type: ignore

class IDistributedDataLoader(Protocol):  # type: ignore
    def distribute_load(self, tasks: List[Dict[str, Any]], **kwargs) -> List[Any]: ...
    def aggregate_results(self, results: List[Any], aggregate_fn: Optional[Callable] = None, **kwargs) -> Any: ...
    def load_distributed(self, start_date: str, end_date: str, frequency: str, **kwargs) -> List[Any]: ...
    def get_node_info(self) -> Dict[str, Any]: ...
    def get_cluster_status(self) -> Dict[str, Any]: ...

__all__ = [
    "DataSourceType",
    "IDataValidator",
    "IDataRegistry",
    "DataRequest",
    "DataResponse",
    "IDistributedDataLoader",
]



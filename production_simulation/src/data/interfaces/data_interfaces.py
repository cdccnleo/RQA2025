"""
RQA2025 数据层接口定义

本模块定义数据层提供的业务接口，
这些接口描述数据层提供的核心业务服务能力。

数据层职责：
1. 数据适配器接口 - 统一数据源访问
2. 市场数据提供者接口 - 实时和历史市场数据
3. 数据质量管理接口 - 数据校验和清理
4. 数据存储接口 - 数据持久化和缓存
5. 数据转换接口 - 数据格式转换和标准化
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime
from dataclasses import dataclass


# =============================================================================
# 数据请求和响应结构
# =============================================================================

@dataclass
class DataRequest:
    """数据请求结构"""
    symbol: str
    market: str = "CN"
    data_type: str = "stock"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: str = "1d"
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "symbol": self.symbol,
            "market": self.market,
            "data_type": self.data_type,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "interval": self.interval,
            "params": self.params or {}
        }


@dataclass
class DataResponse:
    """数据响应结构"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# =============================================================================
# 数据适配器接口
# =============================================================================

class IDataAdapter(Protocol):
    """数据适配器接口 - 统一数据源访问"""

    @abstractmethod
    def get_market_data(self, symbol: str, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
        """获取市场数据"""

    @abstractmethod
    def save_market_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """保存市场数据"""

    @abstractmethod
    def get_available_symbols(self, market: str) -> List[str]:
        """获取可用交易标的"""

    @abstractmethod
    def get_data_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取数据信息"""

    @abstractmethod
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """验证数据有效性"""


class IMarketDataProvider(Protocol):
    """市场数据提供者接口 - 实时和历史数据"""

    @abstractmethod
    def subscribe_market_data(self, symbols: List[str], callback: callable) -> str:
        """订阅市场数据"""

    @abstractmethod
    def unsubscribe_market_data(self, subscription_id: str) -> bool:
        """取消订阅市场数据"""

    @abstractmethod
    def get_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取实时报价"""

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str,
                            interval: str = "1d") -> Optional[List[Dict[str, Any]]]:
        """获取历史数据"""

    @abstractmethod
    def get_market_snapshot(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """获取市场快照"""


# =============================================================================
# 数据质量管理接口
# =============================================================================

@dataclass
class DataQualityReport:
    """数据质量报告"""
    total_records: int
    valid_records: int
    invalid_records: int
    missing_fields: Dict[str, int]
    outliers: Dict[str, List[Any]]
    quality_score: float
    recommendations: List[str]


class IDataQualityManager(Protocol):
    """数据质量管理器接口"""

    @abstractmethod
    def validate_data_quality(self, data: Any) -> DataQualityReport:
        """验证数据质量"""

    @abstractmethod
    def clean_data(self, data: Any) -> Any:
        """清理数据"""

    @abstractmethod
    def detect_anomalies(self, data: Any) -> List[Dict[str, Any]]:
        """检测数据异常"""

    @abstractmethod
    def repair_data(self, data: Any, issues: List[Dict[str, Any]]) -> Any:
        """修复数据问题"""


# =============================================================================
# 数据存储接口
# =============================================================================

class IDataStorage(Protocol):
    """数据存储接口"""

    @abstractmethod
    def store_data(self, key: str, data: Any) -> bool:
        """存储数据"""

    @abstractmethod
    def retrieve_data(self, key: str) -> Optional[Any]:
        """检索数据"""

    @abstractmethod
    def delete_data(self, key: str) -> bool:
        """删除数据"""

    @abstractmethod
    def list_data_keys(self, pattern: str = "*") -> List[str]:
        """列出数据键"""

    @abstractmethod
    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""


class IDataCache(Protocol):
    """数据缓存接口"""

    @abstractmethod
    def get_cached_data(self, key: str) -> Optional[Any]:
        """获取缓存数据"""

    @abstractmethod
    def set_cached_data(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存数据"""

    @abstractmethod
    def invalidate_cache(self, key: str) -> bool:
        """使缓存失效"""

    @abstractmethod
    def clear_cache(self) -> bool:
        """清空缓存"""

    @abstractmethod
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""


# =============================================================================
# 数据转换和标准化接口
# =============================================================================

class IDataTransformer(Protocol):
    """数据转换器接口"""

    @abstractmethod
    def transform_data(self, data: Any, target_format: str) -> Any:
        """转换数据格式"""

    @abstractmethod
    def normalize_data(self, data: Any, method: str = "standard") -> Any:
        """标准化数据"""

    @abstractmethod
    def denormalize_data(self, normalized_data: Any, params: Dict[str, Any]) -> Any:
        """反标准化数据"""

    @abstractmethod
    def validate_transformation(self, original_data: Any, transformed_data: Any) -> bool:
        """验证转换结果"""


# =============================================================================
# 数据服务提供者接口
# =============================================================================

class IDataServiceProvider(Protocol):
    """数据服务提供者接口 - 数据层的统一服务访问点"""

    @property
    def data_adapter(self) -> IDataAdapter:
        """数据适配器"""

    @property
    def market_data_provider(self) -> IMarketDataProvider:
        """市场数据提供者"""

    @property
    def data_quality_manager(self) -> IDataQualityManager:
        """数据质量管理器"""

    @property
    def data_storage(self) -> IDataStorage:
        """数据存储器"""

    @property
    def data_cache(self) -> IDataCache:
        """数据缓存器"""

    @property
    def data_transformer(self) -> IDataTransformer:
        """数据转换器"""

    @abstractmethod
    def get_service_status(self) -> str:
        """获取数据服务整体状态"""

    @abstractmethod
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据层统计摘要"""


class IDataProcessor(ABC):
    """数据处理器接口"""

    @abstractmethod
    def process(self, data: Any, **kwargs) -> Any:
        """处理数据"""

    @abstractmethod
    def get_processing_info(self) -> Dict[str, Any]:
        """获取处理信息"""

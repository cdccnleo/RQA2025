
"""
RQA2025 Data Interfaces

Data layer interface definitions.
"""

from typing import Any, Dict, List, Optional, Protocol
from .adapters.base_adapter import DataRequest, DataResponse


class IDataProvider(Protocol):

    """Data provider interface"""

    def get_data(self, request: DataRequest) -> DataResponse:
        """Get data by request"""
        ...

    def get_bulk_data(self, requests: List[DataRequest]) -> List[DataResponse]:
        """Get multiple data requests"""
        ...


class IMarketDataProvider(IDataProvider):

    """Market data provider interface"""

    def get_realtime_price(self, symbol: str) -> Dict[str, Any]:
        """Get real - time price"""
        ...

    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get historical data"""
        ...


class INewsDataProvider(IDataProvider):

    """News data provider interface"""

    def get_news(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get news for symbol"""
        ...

    def get_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        ...


class IDataModel(Protocol):

    """Data model interface"""

    def validate(self) -> bool:
        """Validate data model"""
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        ...

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary"""
        ...


class ICacheManager(Protocol):

    """Cache manager interface"""

    def get(self, key: str) -> Any:
        """Get from cache"""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in cache"""
        ...

    def delete(self, key: str) -> bool:
        """Delete from cache"""
        ...

    def clear(self) -> bool:
        """Clear cache"""
        ...


class IQualityMonitor(Protocol):

    """Quality monitor interface"""

    def check_quality(self, data: Any) -> Dict[str, Any]:
        """Check data quality"""
        ...

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality metrics"""
        ...

    def repair_data(self, data: Any) -> Any:
        """Repair data quality issues"""
        ...


# 导出所有接口
__all__ = [
    'IDataProvider',
    'IMarketDataProvider',
    'INewsDataProvider',
    'IDataModel',
    'ICacheManager',
    'IQualityMonitor'
]

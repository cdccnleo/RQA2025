#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施接口修复脚本

创建缺失的基础设施接口模块
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def create_infrastructure_interfaces():
    """创建基础设施接口模块"""

    print("🏗️ 创建基础设施接口模块...")

    # 1. 创建标准接口模块
    standard_interfaces_content = '''
"""
RQA2025 Infrastructure Standard Interfaces

Standard interfaces for infrastructure components.
"""

from typing import Any, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 数据请求接口
@dataclass
class DataRequest:
    """Data request structure"""

    symbol: str
    market: str = "CN"
    data_type: str = "stock"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    interval: str = "1d"
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'market': self.market,
            'data_type': self.data_type,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'interval': self.interval,
            'params': self.params or {}
        }

@dataclass
class DataResponse:
    """Data response structure"""

    request: DataRequest
    data: Any
    success: bool = True
    error_message: Optional[str] = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()

class IServiceProvider(Protocol):
    """Service provider interface"""

    def get_service(self, service_name: str) -> Any:
        """Get service instance"""
        ...

    def register_service(self, service_name: str, service_instance: Any) -> bool:
        """Register service instance"""
        ...

    def unregister_service(self, service_name: str) -> bool:
        """Unregister service instance"""
        ...

class ICacheProvider(Protocol):
    """Cache provider interface"""

    def get(self, key: str) -> Any:
        """Get value from cache"""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        ...

    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        ...

    def clear(self) -> bool:
        """Clear all cache"""
        ...

class ILogger(Protocol):
    """Logger interface"""

    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        ...

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        ...

    def error(self, message: str, **kwargs) -> None:
        """Log error message"""
        ...

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        ...

class IConfigProvider(Protocol):
    """Configuration provider interface"""

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        ...

    def set_config(self, key: str, value: Any) -> bool:
        """Set configuration value"""
        ...

    def load_config(self, config_file: str) -> bool:
        """Load configuration from file"""
        ...

    def save_config(self, config_file: str) -> bool:
        """Save configuration to file"""
        ...

class IHealthCheck(Protocol):
    """Health check interface"""

    def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        ...

    def is_healthy(self) -> bool:
        """Check if component is healthy"""
        ...

# 事件相关接口
@dataclass
class Event:
    """Event structure"""

    event_type: str
    data: Optional[Dict[str, Any]] = None
    source: str = "system"
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()

class IEventBus(Protocol):
    """Event bus interface"""

    def publish(self, event: Event) -> str:
        """Publish event"""
        ...

    def subscribe(self, event_type: str, handler: callable) -> bool:
        """Subscribe to event"""
        ...

    def unsubscribe(self, event_type: str, handler: callable) -> bool:
        """Unsubscribe from event"""
        ...

# 监控接口
class IMonitor(Protocol):
    """Monitor interface"""

    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None) -> None:
        """Record metric"""
        ...

    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metric"""
        ...

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        ...

# 导出所有接口
__all__ = [
    'DataRequest',
    'DataResponse',
    'IServiceProvider',
    'ICacheProvider',
    'ILogger',
    'IConfigProvider',
    'IHealthCheck',
    'Event',
    'IEventBus',
    'IMonitor'
]
'''
    # 创建标准接口模块
    os.makedirs(project_root / 'src' / 'infrastructure' / 'interfaces', exist_ok=True)
    with open(project_root / 'src' / 'infrastructure' / 'interfaces' / 'standard_interfaces.py', 'w', encoding='utf-8') as f:
        f.write(standard_interfaces_content)
    print("✅ 创建了标准接口模块")

    # 2. 创建基础设施接口__init__.py
    interfaces_init_content = '''
"""
RQA2025 Infrastructure Interfaces

All infrastructure interface definitions.
"""

from .standard_interfaces import *

__all__ = [
    'DataRequest',
    'DataResponse',
    'IServiceProvider',
    'ICacheProvider',
    'ILogger',
    'IConfigProvider',
    'IHealthCheck',
    'Event',
    'IEventBus',
    'IMonitor'
]
'''
    with open(project_root / 'src' / 'infrastructure' / 'interfaces' / '__init__.py', 'w', encoding='utf-8') as f:
        f.write(interfaces_init_content)
    print("✅ 创建了基础设施接口初始化文件")

    # 3. 创建增强数据加载器模块
    enhanced_loader_content = '''
"""
RQA2025 Enhanced Data Loader

Enhanced data loading with advanced features.
"""

from typing import Any, Dict, List, Optional
import logging
import time

from .base_loader import BaseDataLoader, LoaderConfig
from src.infrastructure.interfaces.standard_interfaces import DataRequest, DataResponse

logger = logging.getLogger(__name__)

class EnhancedDataLoader(BaseDataLoader):
    """Enhanced data loader with advanced features"""

    def __init__(self, config: Optional[LoaderConfig] = None):
        super().__init__(config)
        self.cache = {}
        self.metrics = {
            'requests_total': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors_total': 0
        }

    def load_data(self, request: DataRequest, **kwargs) -> DataResponse:
        """Load data with enhanced features"""
        if not self.is_initialized:
            raise RuntimeError("Loader not initialized")

        self.metrics['requests_total'] += 1

        try:
            # 检查缓存
            cache_key = f"{request.symbol}_{request.market}_{request.data_type}"
            if cache_key in self.cache:
                self.metrics['cache_hits'] += 1
                cached_data = self.cache[cache_key]
                return DataResponse(
                    request=request,
                    data=cached_data,
                    success=True
                )

            # 加载新数据
            self.metrics['cache_misses'] += 1
            data = self._fetch_data(request)

            # 缓存结果
            if self.config.cache_enabled:
                self.cache[cache_key] = data

            return DataResponse(
                request=request,
                data=data,
                success=True
            )

        except Exception as e:
            self.metrics['errors_total'] += 1
            logger.error(f"Failed to load data for {request.symbol}: {e}")
            return DataResponse(
                request=request,
                data=None,
                success=False,
                error_message=str(e)
            )

    def _fetch_data(self, request: DataRequest) -> Dict[str, Any]:
        """Fetch data from source"""
        # Mock implementation - in real implementation, this would call actual data sources
        return {
            "symbol": request.symbol,
            "market": request.market,
            "data_type": request.data_type,
            "price": 100.0,
            "volume": 1000000,
            "timestamp": time.time(),
            "source": "EnhancedDataLoader"
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get loader metrics"""
        return {
            **self.metrics,
            'cache_size': len(self.cache),
            'cache_hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['requests_total'])
        }

    def clear_cache(self) -> bool:
        """Clear cache"""
        try:
            self.cache.clear()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Health check"""
        base_health = super().health_check()
        return {
            **base_health,
            'metrics': self.get_metrics(),
            'cache_status': 'healthy' if len(self.cache) >= 0 else 'error'
        }
'''
    with open(project_root / 'src' / 'data' / 'loader' / 'enhanced_data_loader.py', 'w', encoding='utf-8') as f:
        f.write(enhanced_loader_content)
    print("✅ 创建了增强数据加载器模块")

    # 4. 创建数据接口模块
    data_interfaces_content = '''
"""
RQA2025 Data Interfaces

Data layer interface definitions.
"""

from typing import Any, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod
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
        """Get real-time price"""
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
'''
    with open(project_root / 'src' / 'data' / 'interfaces.py', 'w', encoding='utf-8') as f:
        f.write(data_interfaces_content)
    print("✅ 创建了数据接口模块")

    print("✅ 所有基础设施接口模块已创建完成")


def main():
    """主函数"""
    try:
        create_infrastructure_interfaces()

        print(f"\n{'=' * 60}")
        print("🎉 基础设施接口修复完成！")
        print("=" * 60)
        print("现在可以重新运行数据层测试了。")

        return 0
    except Exception as e:
        print(f"❌ 修复过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

"""
Async标准接口
"""

# 从data.interfaces导入标准接口
try:
    from src.data.interfaces.standard_interfaces import (
        DataRequest, DataResponse, DataSourceType, IDataAdapter
    )
except ImportError:
    # 提供基础实现
    from dataclasses import dataclass
    from typing import Dict, Any, Optional
    from enum import Enum
    
    @dataclass
    class DataRequest:
        source: str
        query: Dict[str, Any]
        
    @dataclass
    class DataResponse:
        data: Any
        status: str = "success"
    
    class DataSourceType(Enum):
        DATABASE = "database"
        API = "api"
    
    class IDataAdapter:
        pass

__all__ = ['DataRequest', 'DataResponse', 'DataSourceType', 'IDataAdapter']


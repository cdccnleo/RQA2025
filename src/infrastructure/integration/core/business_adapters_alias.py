"""
Core Business Adapters - 别名导入（向后兼容）

原文件business_adapters.py (580行)的功能已整合到unified_business_adapters.py
本文件提供向后兼容的导入路径

迁移说明：
- 原580行实现已整合到统一的UnifiedBusinessAdapter
- 新实现基于BaseAdapter基类，减少代码重复
- 保留完整的基础设施服务集成功能
- 添加了缓存、监控、错误恢复等高级特性

使用示例：
    # 新方式（推荐）
    from src.infrastructure.integration.unified_business_adapters import get_business_adapter, BusinessLayerType
    adapter = get_business_adapter(BusinessLayerType.TRADING)
    
    # 旧方式（向后兼容）
    from src.infrastructure.integration.core.business_adapters import BaseBusinessAdapter

更新时间: 2025-11-03
"""

# 从统一实现导入所有功能
from src.infrastructure.integration.unified_business_adapters import (
    BusinessLayerType,
    IBusinessAdapter,
    UnifiedBusinessAdapter as BaseBusinessAdapter,
    BusinessAdapterFactory,
    get_business_adapter
)

__all__ = [
    'BusinessLayerType',
    'IBusinessAdapter',
    'BaseBusinessAdapter',
    'BusinessAdapterFactory',
    'get_business_adapter'
]


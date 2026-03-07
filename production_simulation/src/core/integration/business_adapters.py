"""
业务适配器模块（向后兼容别名）

本文件已重构，指向统一的business_adapters实现。
实际实现在 unified_business_adapters.py 中，基于BaseAdapter基类。

使用方式：
    # 新的推荐方式（使用统一实现）
    from src.core.integration.unified_business_adapters import (
        UnifiedBusinessAdapter,
        BusinessAdapterFactory,
        get_business_adapter
    )
    
    # 向后兼容方式（仍然支持）
    from src.core.integration.business_adapters import (
        BusinessLayerType,
        IBusinessAdapter
    )

重构说明：
- 原有的多个business_adapters实现已整合
- 新实现基于BaseAdapter基类，功能更强大
- 支持缓存、性能监控、错误恢复等高级特性

更新时间: 2025-11-03
"""

# 从统一实现导入
from src.core.integration.unified_business_adapters import (
    BusinessLayerType,
    IBusinessAdapter,
    UnifiedBusinessAdapter as BaseBusinessAdapter,
    BusinessAdapterFactory as UnifiedBusinessAdapterFactory,
    get_business_adapter
)

__all__ = [
    'IBusinessAdapter',
    'BusinessLayerType',
    'BaseBusinessAdapter',
    'UnifiedBusinessAdapterFactory',
    'get_business_adapter'
]


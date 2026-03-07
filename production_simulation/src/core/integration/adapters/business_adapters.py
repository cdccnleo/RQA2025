"""
Integration Adapters - Business Adapters别名模块（向后兼容）

本文件是向后兼容的别名导入，实际实现已整合到unified_business_adapters.py

使用方式：
    # 推荐方式（直接使用统一实现）
    from src.core.integration.unified_business_adapters import get_business_adapter
    
    # 向后兼容方式（通过别名）
    from src.core.integration.adapters.business_adapters import BusinessLayerType

重构说明：
- 3个business_adapters文件已整合为1个统一实现
- 减少代码重复约600行
- 基于BaseAdapter基类，功能更强大

更新时间: 2025-11-03
"""

from src.core.integration.unified_business_adapters import (
    IBusinessAdapter,
    UnifiedBusinessAdapter as BaseBusinessAdapter,
    BusinessLayerType,
    BusinessAdapterFactory as UnifiedBusinessAdapterFactory,
    get_business_adapter
)

__all__ = [
    'IBusinessAdapter',
    'BaseBusinessAdapter',
    'BusinessLayerType',
    'UnifiedBusinessAdapterFactory',
    'get_business_adapter'
]


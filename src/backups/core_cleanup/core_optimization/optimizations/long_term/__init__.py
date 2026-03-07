"""
Long Term Optimizations - 长期优化组件

本模块将原 long_term_optimizations.py (1,014行) 拆分为5个职责单一的组件。

重构日期: 2025-10-25
重构原因: 消除超大文件，提升可维护性
"""

# 导入数据模型
from .models import (
    ServiceType,
    CloudProvider,
    AIType,
    Microservice,
    CloudResource,
    AIModel
)

# 导入业务组件
from .microservice_migration import MicroserviceMigrator
from .cloud_native_support import CloudNativeSupport
from .ai_integration import AIIntegrator
from .ecosystem_building import EcosystemBuilder

# 导入协调器
from .long_term_strategy import LongTermStrategy

# 向后兼容的别名
MicroserviceMigration = MicroserviceMigrator  # 旧名称
AIIntegration = AIIntegrator  # 旧名称
EcosystemBuilding = EcosystemBuilder  # 旧名称

# 导出所有组件
__all__ = [
    # 数据模型
    'ServiceType',
    'CloudProvider',
    'AIType',
    'Microservice',
    'CloudResource',
    'AIModel',
    
    # 业务组件
    'MicroserviceMigrator',
    'CloudNativeSupport',
    'AIIntegrator',
    'EcosystemBuilder',
    
    # 协调器
    'LongTermStrategy',
    
    # 向后兼容别名
    'MicroserviceMigration',
    'AIIntegration',
    'EcosystemBuilding',
]


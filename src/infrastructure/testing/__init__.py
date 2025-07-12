"""基础设施测试工具模块

提供系统测试和验证的核心工具，包括：
- 混沌工程测试框架
- 部署验证工具
- 监管合规测试
- 灾难恢复测试
"""

from .chaos_engine import ChaosEngine
from .chaos_orchestrator import ChaosOrchestrator
from .deployment_validator import DeploymentValidator
from .disaster_test import DisasterTester as DisasterTestRunner
from .regulatory_test import RegulatoryTestFramework as RegulatoryTestSuite

__all__ = [
    'ChaosEngine',
    'ChaosOrchestrator',
    'DeploymentValidator',
    'DisasterTestRunner',
    'RegulatoryTestSuite'
]

# 测试工具初始化函数
def initialize_testing(config: dict):
    """初始化测试工具环境

    Args:
        config: 测试配置字典，包含:
            - chaos_enabled: 是否启用混沌测试
            - disaster_recovery: 灾难恢复测试配置
            - regulatory_rules: 监管合规规则
    """
    from . import chaos_engine, disaster_test, regulatory_test

    if config.get('chaos_enabled', False):
        chaos_engine.initialize(config.get('chaos_params', {}))

    if config.get('disaster_recovery', {}):
        disaster_test.initialize(config['disaster_recovery'])

    if config.get('regulatory_rules', []):
        regulatory_test.load_rules(config['regulatory_rules'])

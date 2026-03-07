"""
API测试生成框架 - 重构后的模块化结构

原有架构（保持兼容）：
- TestTemplateManager: 模板管理 (旧)
- TestCaseBuilder: 测试用例构建基类 (旧)
- DataServiceTestGenerator: 数据服务测试生成 (旧)
- FeatureServiceTestGenerator: 特征服务测试生成 (旧)
- TradingServiceTestGenerator: 交易服务测试生成 (旧)
- MonitoringServiceTestGenerator: 监控服务测试生成 (旧)
- TestSuiteExporter: 测试套件导出 (旧)
- TestStatisticsCollector: 测试统计收集 (旧)
- APITestSuiteCoordinator: 统一协调器（Facade）(旧)

新增组件化架构（推荐使用）：
- components.TestTemplateManager: 新的模板管理器
- components.TestExporter: 新的测试导出器
- components.TestStatisticsCollector: 新的统计收集器
- builders.BaseTestBuilder: 测试构建器基类
- builders.DataServiceTestBuilder: 数据服务测试构建器
- builders.FeatureServiceTestBuilder: 特征服务测试构建器
- builders.TradingServiceTestBuilder: 交易服务测试构建器
- builders.MonitoringServiceTestBuilder: 监控服务测试构建器
"""

# 导入旧架构组件（保持向后兼容）
from .models import TestCase, TestScenario, TestSuite
from .template_manager import TestTemplateManager as OldTestTemplateManager
from .test_case_builder import TestCaseBuilder
from .generators import (
    DataServiceTestGenerator,
    FeatureServiceTestGenerator,
    TradingServiceTestGenerator,
    MonitoringServiceTestGenerator
)
from .exporter import TestSuiteExporter as OldTestSuiteExporter
from .statistics import TestStatisticsCollector as OldTestStatisticsCollector
from .coordinator import APITestSuiteCoordinator

# 导入新组件化架构（推荐使用）
from .components import (
    TestTemplateManager as NewTestTemplateManager,
    TestExporter,
    TestStatisticsCollector as NewTestStatisticsCollector
)
from .builders import (
    BaseTestBuilder,
    DataServiceTestBuilder,
    FeatureServiceTestBuilder,
    TradingServiceTestBuilder,
    MonitoringServiceTestBuilder
)

# 默认使用旧组件保持兼容，但提供新组件别名
TestTemplateManager = OldTestTemplateManager
TestSuiteExporter = OldTestSuiteExporter
TestStatisticsCollector = OldTestStatisticsCollector

__all__ = [
    # 数据模型
    'TestCase',
    'TestScenario',
    'TestSuite',
    
    # 旧架构组件（向后兼容）
    'OldTestTemplateManager',
    'TestCaseBuilder',
    'DataServiceTestGenerator',
    'FeatureServiceTestGenerator',
    'TradingServiceTestGenerator',
    'MonitoringServiceTestGenerator',
    'OldTestSuiteExporter',
    'OldTestStatisticsCollector',
    'APITestSuiteCoordinator',
    
    # 默认组件（向后兼容）
    'TestTemplateManager',
    'TestSuiteExporter',
    'TestStatisticsCollector',
    
    # 新组件化架构（推荐）
    'NewTestTemplateManager',
    'TestExporter',
    'NewTestStatisticsCollector',
    'BaseTestBuilder',
    'DataServiceTestBuilder',
    'FeatureServiceTestBuilder',
    'TradingServiceTestBuilder',
    'MonitoringServiceTestBuilder',
]


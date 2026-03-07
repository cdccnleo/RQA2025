"""
API管理模块配置对象

本模块提供API管理功能的各类配置对象，用于解决长参数列表问题，
采用参数对象模式(Parameter Object Pattern)提升代码可维护性。

配置类别：
- flow_configs: 流程图生成相关配置
- test_configs: 测试用例生成相关配置
- schema_configs: Schema生成相关配置
- endpoint_configs: API端点相关配置
"""

from .base_config import BaseConfig, ValidationResult, Priority, ExportFormat
from .schema_configs import (
    SchemaGenerationConfig,
    SchemaDefinitionConfig,
    SchemaPropertyConfig,
    ResponseSchemaConfig,
)
from .endpoint_configs import (
    EndpointConfig,
    EndpointParameterConfig,
    EndpointResponseConfig,
    OpenAPIDocConfig,
)
from .flow_configs import (
    FlowGenerationConfig,
    FlowNodeConfig,
    FlowEdgeConfig,
    FlowExportConfig,
)
from .test_configs import (
    TestSuiteConfig,
    TestCaseConfig,
    TestScenarioConfig,
    TestExportConfig,
)

__all__ = [
    # 基础配置
    'BaseConfig',
    'ValidationResult',
    'Priority',
    'ExportFormat',
    
    # 流程配置
    'FlowGenerationConfig',
    'FlowNodeConfig',
    'FlowEdgeConfig',
    'FlowExportConfig',
    
    # 测试配置
    'TestSuiteConfig',
    'TestCaseConfig',
    'TestScenarioConfig',
    'TestExportConfig',
    
    # Schema配置
    'SchemaGenerationConfig',
    'SchemaDefinitionConfig',
    'SchemaPropertyConfig',
    'ResponseSchemaConfig',
    
    # 端点配置
    'EndpointConfig',
    'EndpointParameterConfig',
    'EndpointResponseConfig',
    'OpenAPIDocConfig',
]


"""
OpenAPI构建器模块

提供OpenAPI文档生成的各类构建器组件

重构成果：
- SchemaBuilder: 替代251行_add_common_schemas方法
- EndpointBuilderCoordinator: 替代4个_add_*_endpoints方法
- DocumentationAssembler: 文档组装和导出
"""

from .schema_builder import (
    SchemaBuilder,
    CommonResponseBuilder,
    build_common_schemas,
    build_common_responses
)
from .endpoint_builder import (
    EndpointBuilderCoordinator,
    EndpointBuildStrategy,
    DataServiceEndpointBuilder,
    FeatureServiceEndpointBuilder,
    TradingServiceEndpointBuilder,
    MonitoringServiceEndpointBuilder
)
from .documentation_assembler import (
    DocumentationAssembler,
    APISchema
)

__all__ = [
    # Schema构建
    'SchemaBuilder',
    'CommonResponseBuilder',
    'build_common_schemas',
    'build_common_responses',
    
    # 端点构建
    'EndpointBuilderCoordinator',
    'EndpointBuildStrategy',
    'DataServiceEndpointBuilder',
    'FeatureServiceEndpointBuilder',
    'TradingServiceEndpointBuilder',
    'MonitoringServiceEndpointBuilder',
    
    # 文档组装
    'DocumentationAssembler',
    'APISchema',
]


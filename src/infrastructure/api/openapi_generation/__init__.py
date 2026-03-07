"""
OpenAPI文档生成框架

将RQAApiDocumentationGenerator拆分为多个专注的类：
- EndpointBuilder: 端点构建器
- SchemaBuilder: 模式构建器
- ServiceDocGenerators: 各服务文档生成器
- RQAApiDocCoordinator: 协调器（对外接口）
"""

from .endpoint_builder import EndpointBuilder
from .schema_builder import SchemaBuilder
from .service_doc_generators import (
    DataServiceDocGenerator,
    FeatureServiceDocGenerator,
    TradingServiceDocGenerator,
    MonitoringServiceDocGenerator
)
from .coordinator import RQAApiDocCoordinator

__all__ = [
    'EndpointBuilder',
    'SchemaBuilder',
    'DataServiceDocGenerator',
    'FeatureServiceDocGenerator',
    'TradingServiceDocGenerator',
    'MonitoringServiceDocGenerator',
    'RQAApiDocCoordinator'
]


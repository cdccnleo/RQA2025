"""
RQA API文档协调器

职责：协调各个组件生成完整的OpenAPI文档
向后兼容：提供与RQAApiDocumentationGenerator相同的接口
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List

from .endpoint_builder import EndpointBuilder, APIEndpoint
from .schema_builder import SchemaBuilder, APISchema
from .service_doc_generators import (
    DataServiceDocGenerator,
    FeatureServiceDocGenerator,
    TradingServiceDocGenerator,
    MonitoringServiceDocGenerator
)


class RQAApiDocCoordinator:
    """RQA API文档生成协调器"""
    
    def __init__(self):
        """初始化协调器"""
        # 初始化构建器
        self.endpoint_builder = EndpointBuilder()
        self.schema_builder = SchemaBuilder()
        
        # 初始化服务文档生成器
        self.data_gen = DataServiceDocGenerator(self.endpoint_builder, self.schema_builder)
        self.feature_gen = FeatureServiceDocGenerator(self.endpoint_builder, self.schema_builder)
        self.trading_gen = TradingServiceDocGenerator(self.endpoint_builder, self.schema_builder)
        self.monitoring_gen = MonitoringServiceDocGenerator(self.endpoint_builder, self.schema_builder)
        
        # API模式
        self.api_schema: APISchema = None
    
    def generate_documentation(self, output_dir: str = "docs/api") -> Dict[str, str]:
        """
        生成完整的API文档
        
        Args:
            output_dir: 输出目录
            
        Returns:
            生成的文件路径字典
        """
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建API模式
        self.api_schema = self._create_rqa_api_schema()
        
        # 生成OpenAPI规范
        openapi_spec = self._generate_openapi_spec()
        
        # 保存文件
        json_path = output_path / "openapi.json"
        yaml_path = output_path / "openapi.yaml"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(openapi_spec, f, indent=2, ensure_ascii=False)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(openapi_spec, f, allow_unicode=True, default_flow_style=False)
        
        return {
            "json": str(json_path),
            "yaml": str(yaml_path),
            "spec": openapi_spec
        }
    
    def _create_rqa_api_schema(self) -> APISchema:
        """创建RQA API模式"""
        # 清空之前的构建
        self.endpoint_builder.clear()
        self.schema_builder.clear()
        
        # 添加所有端点
        data_endpoints = self.data_gen.generate_endpoints()
        feature_endpoints = self.feature_gen.generate_endpoints()
        trading_endpoints = self.trading_gen.generate_endpoints()
        monitoring_endpoints = self.monitoring_gen.generate_endpoints()
        
        all_endpoints = (
            data_endpoints +
            feature_endpoints +
            trading_endpoints +
            monitoring_endpoints
        )
        
        # 添加通用模式
        self.schema_builder.add_common_data_schemas()
        self.schema_builder.add_common_response_schemas()
        
        # 创建API模式
        schema = APISchema(
            title="RQA2025 量化研究平台 API",
            version="1.0.0",
            description="量化研究与交易的统一API接口",
            servers=[
                {
                    "url": "http://localhost:5000",
                    "description": "开发环境"
                },
                {
                    "url": "https://api.rqa2025.com",
                    "description": "生产环境"
                }
            ],
            endpoints=all_endpoints,
            schemas=self.schema_builder.get_all_schemas(),
            security_schemes=self.schema_builder.add_security_schemes()
        )
        
        return schema
    
    def _generate_openapi_spec(self) -> Dict[str, Any]:
        """生成OpenAPI规范"""
        if not self.api_schema:
            self.api_schema = self._create_rqa_api_schema()
        
        spec = {
            "openapi": "3.0.3",
            "info": {
                "title": self.api_schema.title,
                "version": self.api_schema.version,
                "description": self.api_schema.description,
                "contact": {
                    "name": "RQA2025 Development Team",
                    "email": "dev@rqa2025.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": self.api_schema.servers,
            "paths": self._generate_paths(),
            "components": {
                "schemas": self.api_schema.schemas,
                "securitySchemes": self.api_schema.security_schemes
            },
            "tags": self._generate_tags(),
            "security": []
        }
        
        return spec

    def _generate_paths(self) -> Dict[str, Any]:
        """生成路径定义"""
        if not self.api_schema:
            self.api_schema = self._create_rqa_api_schema()

        paths: Dict[str, Dict[str, Any]] = {}

        for endpoint in self.api_schema.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}

            method_spec = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "operationId": endpoint.operation_id,
                "tags": endpoint.tags,
                "parameters": endpoint.parameters,
                "responses": endpoint.responses
            }

            if endpoint.request_body:
                method_spec["requestBody"] = endpoint.request_body

            if endpoint.security:
                method_spec["security"] = endpoint.security

            paths[endpoint.path][endpoint.method.lower()] = method_spec

        return paths

    def _generate_tags(self) -> List[Dict[str, str]]:
        """生成标签列表"""
        if not self.api_schema:
            self.api_schema = self._create_rqa_api_schema()

        tags_dict: Dict[str, str] = {}

        for endpoint in self.api_schema.endpoints:
            for tag in endpoint.tags:
                if tag not in tags_dict:
                    tags_dict[tag] = self._get_tag_description(tag)

        return [
            {"name": name, "description": desc}
            for name, desc in tags_dict.items()
        ]

    def _get_tag_description(self, tag: str) -> str:
        """获取标签描述"""
        descriptions = {
            "Data Service": "数据管理和查询服务",
            "Feature Service": "特征工程和计算服务",
            "Trading Service": "交易策略和回测服务",
            "Monitoring": "系统监控和健康检查"
        }
        return descriptions.get(tag, "")

    def get_statistics(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        if not self.api_schema:
            self.api_schema = self._create_rqa_api_schema()

        service_counts: Dict[str, int] = {}
        for endpoint in self.api_schema.endpoints:
            for tag in endpoint.tags:
                service_counts[tag] = service_counts.get(tag, 0) + 1

        return {
            "total_endpoints": len(self.api_schema.endpoints),
            "total_schemas": len(self.api_schema.schemas),
            "services": service_counts,
            "security_schemes": len(self.api_schema.security_schemes)
        }


class Coordinator(RQAApiDocCoordinator):
    """向后兼容别名"""
    pass

    def _generate_paths(self) -> Dict[str, Any]:
        """生成路径定义"""
        paths = {}
        
        for endpoint in self.api_schema.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}
            
            method_spec = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "operationId": endpoint.operation_id,
                "tags": endpoint.tags,
                "parameters": endpoint.parameters,
                "responses": endpoint.responses
            }
            
            if endpoint.request_body:
                method_spec["requestBody"] = endpoint.request_body
            
            if endpoint.security:
                method_spec["security"] = endpoint.security
            
            paths[endpoint.path][endpoint.method.lower()] = method_spec
        
        return paths
    
    def _generate_tags(self) -> list:
        """生成标签列表"""
        tags_dict = {}
        
        for endpoint in self.api_schema.endpoints:
            for tag in endpoint.tags:
                if tag not in tags_dict:
                    tags_dict[tag] = self._get_tag_description(tag)
        
        return [
            {"name": name, "description": desc}
            for name, desc in tags_dict.items()
        ]
    
    def _get_tag_description(self, tag: str) -> str:
        """获取标签描述"""
        descriptions = {
            "Data Service": "数据管理和查询服务",
            "Feature Service": "特征工程和计算服务",
            "Trading Service": "交易策略和回测服务",
            "Monitoring": "系统监控和健康检查"
        }
        return descriptions.get(tag, "")


# 向后兼容：提供原有接口
class RQAApiDocumentationGenerator(RQAApiDocCoordinator):
    """
    向后兼容类
    
    保持与原RQAApiDocumentationGenerator相同的接口
    """
    pass


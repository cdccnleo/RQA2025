"""
openapi_generator 模块

提供 openapi_generator 相关功能和接口。
"""

import json

import yaml

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAPI规范文档生成器
自动生成符合OpenAPI 3.0规范的API文档
"""


@dataclass
class APIEndpoint:
    """API端点定义"""
    path: str
    method: str
    summary: str
    description: str = ""
    operation_id: str = ""
    tags: List[str] = field(default_factory=list)
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class APISchema:
    """API模式定义"""
    title: str
    version: str
    description: str
    servers: List[Dict[str, str]] = field(default_factory=list)
    endpoints: List[APIEndpoint] = field(default_factory=list)
    schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security_schemes: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class OpenAPIGenerator:
    """OpenAPI生成器"""

    def __init__(self, api_schema: APISchema):
        self.schema = api_schema
        self.openapi_version = "3.0.3"

    def generate_openapi_spec(self) -> Dict[str, Any]:
        """生成OpenAPI规范"""
        spec = {
            "openapi": self.openapi_version,
            "info": {
                "title": self.schema.title,
                "version": self.schema.version,
                "description": self.schema.description,
                "contact": {
                    "name": "RQA2025 Development Team",
                    "email": "dev@rqa2025.com"
                },
                "license": {
                    "name": "MIT",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": self.schema.servers,
            "paths": self._generate_paths(),
            "components": {
                "schemas": self.schema.schemas,
                "securitySchemes": self.schema.security_schemes
            },
            "tags": self._generate_tags(),
            "security": self._generate_security_requirements()
        }

        return spec

    def _generate_paths(self) -> Dict[str, Any]:
        """生成路径定义"""
        paths = {}

        for endpoint in self.schema.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}

            method_spec = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "operationId": endpoint.operation_id or f"{endpoint.method.lower()}_{endpoint.path.replace('/', '_').strip('_')}",
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
        """生成标签定义"""
        all_tags = set()
        for endpoint in self.schema.endpoints:
            all_tags.update(endpoint.tags)

        return [{"name": tag, "description": f"{tag} related operations"} for tag in sorted(all_tags)]

    def _generate_security_requirements(self) -> List[Dict[str, Any]]:
        """生成安全要求"""
        security = []
        for scheme_name in self.schema.security_schemes.keys():
            security.append({scheme_name: []})
        return security

    def save_as_json(self, file_path: str):
        """保存为JSON格式"""
        spec = self.generate_openapi_spec()
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)

    def save_as_yaml(self, file_path: str):
        """保存为YAML格式"""
        spec = self.generate_openapi_spec()
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(spec, f, default_flow_style=False, allow_unicode=True)


class RQAApiDocumentationGenerator:
    """RQA2025 API文档生成器"""

    def __init__(self):
        self.api_schema = self._create_rqa_api_schema()

    def _create_rqa_api_schema(self) -> APISchema:
        """创建RQA2025 API模式"""
        schema = APISchema(
            title="RQA2025 Trading System API",
            version="1.0.0",
            description="RQA2025量化交易系统的完整API文档",
            servers=[
                {"url": "http://localhost:8000", "description": "Development server"},
                {"url": "https://api.rqa2025.com", "description": "Production server"}
            ],
            security_schemes={
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                },
                "apiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key"
                }
            }
        )

        # 添加数据服务API
        self._add_data_service_endpoints(schema)

        # 添加特征工程API
        self._add_feature_service_endpoints(schema)

        # 添加交易服务API
        self._add_trading_service_endpoints(schema)

        # 添加监控服务API
        self._add_monitoring_service_endpoints(schema)

        # 添加通用模式
        self._add_common_schemas(schema)

        return schema

    def _add_data_service_endpoints(self, schema: APISchema):
        """添加数据服务端点"""
        endpoints = [
            APIEndpoint(
                path="/api/v1/data/market/{symbol}",
                method="GET",
                summary="获取市场数据",
                description="获取指定交易对的市场数据",
                tags=["Data Service"],
                parameters=[
                    {
                        "name": "symbol",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "交易对符号，如 BTC/USDT"
                    },
                    {
                        "name": "interval",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "1d"]},
                        "description": "时间间隔"
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer", "minimum": 1, "maximum": 1000},
                        "description": "返回数据条数"
                    }
                ],
                responses={
                    "200": {
                        "description": "成功获取市场数据",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MarketDataResponse"}
                            }
                        }
                    },
                    "400": {"$ref": "#/components/responses/BadRequest"},
                    "500": {"$ref": "#/components/responses/InternalServerError"}
                }
            ),
            APIEndpoint(
                path="/api/v1/data/validate",
                method="POST",
                summary="验证数据质量",
                description="验证上传数据的质量和完整性",
                tags=["Data Service"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/DataValidationRequest"}
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "数据验证完成",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/DataValidationResponse"}
                            }
                        }
                    },
                    "400": {"$ref": "#/components/responses/BadRequest"},
                    "500": {"$ref": "#/components/responses/InternalServerError"}
                }
            )
        ]

        schema.endpoints.extend(endpoints)

    def _add_feature_service_endpoints(self, schema: APISchema):
        """添加特征工程端点"""
        endpoints = [
            APIEndpoint(
                path="/api/v1/features/compute",
                method="POST",
                summary="计算技术指标",
                description="为给定的市场数据计算技术指标",
                tags=["Feature Engineering"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/FeatureComputeRequest"}
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "特征计算完成",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/FeatureComputeResponse"}
                            }
                        }
                    },
                    "400": {"$ref": "#/components/responses/BadRequest"},
                    "500": {"$ref": "#/components/responses/InternalServerError"}
                }
            ),
            APIEndpoint(
                path="/api/v1/features/sentiment",
                method="POST",
                summary="情感分析",
                description="对新闻文本进行情感分析",
                tags=["Feature Engineering"],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/SentimentAnalysisRequest"}
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "情感分析完成",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/SentimentAnalysisResponse"}
                            }
                        }
                    }
                }
            )
        ]

        schema.endpoints.extend(endpoints)

    def _add_trading_service_endpoints(self, schema: APISchema):
        """添加交易服务端点"""
        endpoints = [
            APIEndpoint(
                path="/api/v1/trading/strategy/{strategy_id}/execute",
                method="POST",
                summary="执行交易策略",
                description="执行指定的交易策略",
                tags=["Trading"],
                parameters=[
                    {
                        "name": "strategy_id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                        "description": "策略ID"
                    }
                ],
                request_body={
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/StrategyExecutionRequest"}
                        }
                    }
                },
                responses={
                    "200": {
                        "description": "策略执行成功",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/StrategyExecutionResponse"}
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            ),
            APIEndpoint(
                path="/api/v1/trading/portfolio",
                method="GET",
                summary="获取投资组合",
                description="获取当前投资组合状态",
                tags=["Trading"],
                responses={
                    "200": {
                        "description": "成功获取投资组合",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PortfolioResponse"}
                            }
                        }
                    }
                },
                security=[{"bearerAuth": []}]
            )
        ]

        schema.endpoints.extend(endpoints)

    def _add_monitoring_service_endpoints(self, schema: APISchema):
        """添加监控服务端点"""
        endpoints = [
            APIEndpoint(
                path="/api/v1/monitoring/health",
                method="GET",
                summary="系统健康检查",
                description="获取系统健康状态",
                tags=["Monitoring"],
                responses={
                    "200": {
                        "description": "系统健康",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/HealthResponse"}
                            }
                        }
                    }
                }
            ),
            APIEndpoint(
                path="/api/v1/monitoring/metrics",
                method="GET",
                summary="获取系统指标",
                description="获取系统性能指标",
                tags=["Monitoring"],
                responses={
                    "200": {
                        "description": "成功获取指标",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/MetricsResponse"}
                            }
                        }
                    }
                }
            )
        ]

        schema.endpoints.extend(endpoints)

    def _add_common_schemas(self, schema: APISchema):
        """添加通用模式定义"""
        schemas = {
            # 响应基础模式
            "BaseResponse": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "request_id": {"type": "string"}
                },
                "required": ["success", "timestamp"]
            },

            # 错误响应
            "ErrorResponse": {
                "type": "object",
                "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
                "properties": {
                    "error": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "message": {"type": "string"},
                            "details": {"type": "object"}
                        }
                    }
                }
            },

            # 分页响应
            "PaginatedResponse": {
                "type": "object",
                "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
                "properties": {
                    "data": {"type": "array", "items": {"type": "object"}},
                    "pagination": {
                        "type": "object",
                        "properties": {
                            "page": {"type": "integer", "minimum": 1},
                            "page_size": {"type": "integer", "minimum": 1},
                            "total": {"type": "integer", "minimum": 0},
                            "total_pages": {"type": "integer", "minimum": 0}
                        }
                    }
                }
            },

            # 数据服务相关模式
            "MarketDataResponse": {
                "type": "object",
                "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
                "properties": {
                    "data": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp": {"type": "string", "format": "date-time"},
                                "symbol": {"type": "string"},
                                "open": {"type": "number"},
                                "high": {"type": "number"},
                                "low": {"type": "number"},
                                "close": {"type": "number"},
                                "volume": {"type": "number"}
                            }
                        }
                    }
                }
            },

            "DataValidationRequest": {
                "type": "object",
                "properties": {
                    "data": {"type": "array", "items": {"type": "object"}},
                    "rules": {"type": "object"},
                    "strict_mode": {"type": "boolean", "default": False}
                },
                "required": ["data"]
            },

            "DataValidationResponse": {
                "type": "object",
                "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
                "properties": {
                    "validation_result": {
                        "type": "object",
                        "properties": {
                            "is_valid": {"type": "boolean"},
                            "total_records": {"type": "integer"},
                            "valid_records": {"type": "integer"},
                            "invalid_records": {"type": "integer"},
                            "errors": {"type": "array", "items": {"type": "object"}}
                        }
                    }
                }
            },

            # 特征工程相关模式
            "FeatureComputeRequest": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "data": {"type": "array", "items": {"type": "object"}},
                    "indicators": {"type": "array", "items": {"type": "string"}},
                    "parameters": {"type": "object"}
                },
                "required": ["symbol", "data", "indicators"]
            },

            "FeatureComputeResponse": {
                "type": "object",
                "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
                "properties": {
                    "features": {"type": "array", "items": {"type": "object"}},
                    "metadata": {"type": "object"}
                }
            },

            "SentimentAnalysisRequest": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "language": {"type": "string", "default": "zh"},
                    "context": {"type": "object"}
                },
                "required": ["text"]
            },

            "SentimentAnalysisResponse": {
                "type": "object",
                "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
                "properties": {
                    "sentiment": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "number", "minimum": -1, "maximum": 1},
                            "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                        }
                    }
                }
            },

            # 交易服务相关模式
            "StrategyExecutionRequest": {
                "type": "object",
                "properties": {
                    "parameters": {"type": "object"},
                    "capital": {"type": "number", "minimum": 0},
                    "risk_level": {"type": "string", "enum": ["low", "medium", "high"]}
                },
                "required": ["parameters"]
            },

            "StrategyExecutionResponse": {
                "type": "object",
                "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
                "properties": {
                    "execution_id": {"type": "string"},
                    "status": {"type": "string"},
                    "orders": {"type": "array", "items": {"type": "object"}},
                    "summary": {"type": "object"}
                }
            },

            "PortfolioResponse": {
                "type": "object",
                "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
                "properties": {
                    "portfolio": {
                        "type": "object",
                        "properties": {
                            "total_value": {"type": "number"},
                            "cash": {"type": "number"},
                            "positions": {"type": "array", "items": {"type": "object"}},
                            "pnl": {"type": "number"}
                        }
                    }
                }
            },

            # 监控服务相关模式
            "HealthResponse": {
                "type": "object",
                "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
                "properties": {
                    "health": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                            "services": {"type": "array", "items": {"type": "object"}},
                            "uptime": {"type": "number"},
                            "version": {"type": "string"}
                        }
                    }
                }
            },

            "MetricsResponse": {
                "type": "object",
                "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
                "properties": {
                    "metrics": {
                        "type": "object",
                        "properties": {
                            "cpu_usage": {"type": "number"},
                            "memory_usage": {"type": "number"},
                            "disk_usage": {"type": "number"},
                            "network_io": {"type": "object"},
                            "request_count": {"type": "number"},
                            "error_count": {"type": "number"},
                            "response_time": {"type": "number"}
                        }
                    }
                }
            }
        }

        schema.schemas.update(schemas)

        # 添加通用响应定义
        schema.schemas.update({
            "responses": {
                "BadRequest": {
                    "description": "错误的请求",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    }
                },
                "Unauthorized": {
                    "description": "未授权",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    }
                },
                "InternalServerError": {
                    "description": "服务器内部错误",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                        }
                    }
                }
            }
        })

    def generate_documentation(self, output_dir: str = "docs/api"):
        """生成API文档"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generator = OpenAPIGenerator(self.api_schema)

        # 生成JSON格式
        json_file = output_path / "rqa2025_api.json"
        generator.save_as_json(str(json_file))

        # 生成YAML格式
        yaml_file = output_path / "rqa2025_api.yaml"
        generator.save_as_yaml(str(yaml_file))

        print(f"API文档已生成:")
        print(f"  JSON: {json_file}")
        print(f"  YAML: {yaml_file}")

        return str(json_file), str(yaml_file)


if __name__ == "__main__":
    # 生成RQA2025 API文档
    print("生成RQA2025 API文档...")

    generator = RQAApiDocumentationGenerator()
    json_file, yaml_file = generator.generate_documentation()

    print("API文档生成完成！")
    print(f"JSON文件: {json_file}")
    print(f"YAML文件: {yaml_file}")

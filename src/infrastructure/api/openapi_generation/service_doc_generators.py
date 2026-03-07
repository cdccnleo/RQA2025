"""
各服务文档生成器

职责：为不同服务生成OpenAPI文档片段
"""

from typing import List
from .endpoint_builder import EndpointBuilder, APIEndpoint
from .schema_builder import SchemaBuilder


class DataServiceDocGenerator:
    """数据服务文档生成器"""
    
    def __init__(self, endpoint_builder: EndpointBuilder, schema_builder: SchemaBuilder):
        self.endpoint_builder = endpoint_builder
        self.schema_builder = schema_builder
    
    def generate_endpoints(self) -> List[APIEndpoint]:
        """生成数据服务端点"""
        endpoints = []

        # 数据集相关端点
        endpoints.extend(self._create_dataset_endpoints())

        # 市场数据相关端点
        endpoints.extend(self._create_market_data_endpoints())

        return endpoints

    def _create_dataset_endpoints(self) -> List[APIEndpoint]:
        """创建数据集相关的端点"""
        endpoints = []

        # 获取数据集列表
        endpoints.append(self._create_dataset_list_endpoint())

        # 获取单个数据集
        endpoints.append(self._create_dataset_detail_endpoint())

        return endpoints

    def _create_dataset_list_endpoint(self) -> APIEndpoint:
        """创建数据集列表端点"""
        return self.endpoint_builder.create_endpoint(
            path="/api/v1/data/datasets",
            method="GET",
            summary="获取数据集列表",
            description="获取所有可用的数据集列表",
            tags=["Data Service"],
            parameters=[
                self.endpoint_builder.create_query_parameter(
                    "page", "页码", param_type="integer", default=1
                ),
                self.endpoint_builder.create_query_parameter(
                    "page_size", "每页数量", param_type="integer", default=20
                ),
                self.endpoint_builder.create_query_parameter(
                    "source", "数据源筛选"
                )
            ],
            responses={
                "200": {
                    "description": "成功返回数据集列表",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/PaginatedResponse"}
                        }
                    }
                }
            }
        )

    def _create_dataset_detail_endpoint(self) -> APIEndpoint:
        """创建数据集详情端点"""
        return self.endpoint_builder.create_endpoint(
            path="/api/v1/data/datasets/{dataset_id}",
            method="GET",
            summary="获取数据集详情",
            description="根据ID获取数据集的详细信息",
            tags=["Data Service"],
            parameters=[
                self.endpoint_builder.create_path_parameter(
                    "dataset_id", "数据集ID"
                )
            ],
            responses={
                "200": {
                    "description": "成功返回数据集",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/Dataset"}
                        }
                    }
                },
                "404": {
                    "description": "数据集不存在"
                }
            }
        )

        # 获取股票数据
        endpoints.append(self.endpoint_builder.create_endpoint(
            path="/api/v1/data/stocks/{symbol}",
            method="GET",
            summary="获取股票数据",
            description="获取指定股票的历史数据",
            tags=["Data Service"],
            parameters=[
                self.endpoint_builder.create_path_parameter("symbol", "股票代码"),
                self.endpoint_builder.create_query_parameter("start_date", "开始日期"),
                self.endpoint_builder.create_query_parameter("end_date", "结束日期"),
                self.endpoint_builder.create_query_parameter("frequency", "数据频率", default="daily")
            ],
            responses={
                "200": {
                    "description": "成功返回股票数据",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/StockData"}
                            }
                        }
                    }
                }
            }
        ))
        
        return endpoints

    def _create_market_data_endpoints(self) -> List[APIEndpoint]:
        """创建市场数据相关的端点"""
        endpoints = []

        # 获取股票数据
        endpoints.append(self._create_stock_data_endpoint())

        return endpoints

    def _create_stock_data_endpoint(self) -> APIEndpoint:
        """创建股票数据端点"""
        return self.endpoint_builder.create_endpoint(
            path="/api/v1/data/stocks/{symbol}",
            method="GET",
            summary="获取股票数据",
            description="获取指定股票的历史数据",
            tags=["Data Service"],
            parameters=[
                self.endpoint_builder.create_path_parameter("symbol", "股票代码"),
                self.endpoint_builder.create_query_parameter("start_date", "开始日期"),
                self.endpoint_builder.create_query_parameter("end_date", "结束日期"),
                self.endpoint_builder.create_query_parameter("frequency", "数据频率", default="daily")
            ],
            responses={
                "200": {
                    "description": "成功返回股票数据",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/StockData"}
                            }
                        }
                    }
                }
            }
        )


class FeatureServiceDocGenerator:
    """特征工程服务文档生成器"""
    
    def __init__(self, endpoint_builder: EndpointBuilder, schema_builder: SchemaBuilder):
        self.endpoint_builder = endpoint_builder
        self.schema_builder = schema_builder
    
    def generate_endpoints(self) -> List[APIEndpoint]:
        """生成特征工程服务端点"""
        endpoints = []
        
        # 计算技术指标
        endpoints.append(self.endpoint_builder.create_endpoint(
            path="/api/v1/features/technical-indicators",
            method="POST",
            summary="计算技术指标",
            description="根据提供的参数计算各种技术指标",
            tags=["Feature Service"],
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "symbol": {"type": "string"},
                                "indicators": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "params": {"type": "object"}
                            },
                            "required": ["symbol", "indicators"]
                        }
                    }
                }
            },
            responses={
                "200": {
                    "description": "成功返回计算结果",
                    "content": {
                        "application/json": {
                            "schema": {"$ref": "#/components/schemas/SuccessResponse"}
                        }
                    }
                }
            }
        ))
        
        # 特征选择
        endpoints.append(self.endpoint_builder.create_endpoint(
            path="/api/v1/features/selection",
            method="POST",
            summary="特征选择",
            description="执行特征选择算法",
            tags=["Feature Service"],
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "method": {"type": "string"},
                                "n_features": {"type": "integer"},
                                "dataset_id": {"type": "string"}
                            }
                        }
                    }
                }
            }
        ))
        
        return endpoints


class TradingServiceDocGenerator:
    """交易服务文档生成器"""
    
    def __init__(self, endpoint_builder: EndpointBuilder, schema_builder: SchemaBuilder):
        self.endpoint_builder = endpoint_builder
        self.schema_builder = schema_builder
    
    def generate_endpoints(self) -> List[APIEndpoint]:
        """生成交易服务端点"""
        endpoints = []
        
        # 回测策略
        endpoints.append(self.endpoint_builder.create_endpoint(
            path="/api/v1/trading/backtest",
            method="POST",
            summary="执行回测",
            description="对交易策略进行历史回测",
            tags=["Trading Service"],
            request_body={
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "strategy": {"type": "string"},
                                "symbols": {"type": "array", "items": {"type": "string"}},
                                "start_date": {"type": "string", "format": "date"},
                                "end_date": {"type": "string", "format": "date"},
                                "initial_capital": {"type": "number"},
                                "params": {"type": "object"}
                            },
                            "required": ["strategy", "symbols", "start_date", "end_date"]
                        }
                    }
                }
            },
            security=[{"BearerAuth": []}]
        ))
        
        # 获取回测结果
        endpoints.append(self.endpoint_builder.create_endpoint(
            path="/api/v1/trading/backtest/{backtest_id}",
            method="GET",
            summary="获取回测结果",
            description="查询回测任务的结果",
            tags=["Trading Service"],
            parameters=[
                self.endpoint_builder.create_path_parameter("backtest_id", "回测任务ID")
            ],
            security=[{"BearerAuth": []}]
        ))
        
        return endpoints


class MonitoringServiceDocGenerator:
    """监控服务文档生成器"""
    
    def __init__(self, endpoint_builder: EndpointBuilder, schema_builder: SchemaBuilder):
        self.endpoint_builder = endpoint_builder
        self.schema_builder = schema_builder
    
    def generate_endpoints(self) -> List[APIEndpoint]:
        """生成监控服务端点"""
        endpoints = []
        
        # 健康检查
        endpoints.append(self.endpoint_builder.create_endpoint(
            path="/api/v1/health",
            method="GET",
            summary="健康检查",
            description="检查服务健康状态",
            tags=["Monitoring"],
            responses={
                "200": {
                    "description": "服务健康",
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "status": {"type": "string", "example": "healthy"},
                                    "timestamp": {"type": "string", "format": "date-time"},
                                    "version": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        ))
        
        # 系统指标
        endpoints.append(self.endpoint_builder.create_endpoint(
            path="/api/v1/metrics",
            method="GET",
            summary="获取系统指标",
            description="获取系统运行指标",
            tags=["Monitoring"],
            security=[{"ApiKeyAuth": []}]
        ))
        
        return endpoints


"""
API端点构建器

使用策略模式将各服务的端点构建逻辑分离，
替代原RQAApiDocumentationGenerator中的4个_add_*_endpoints方法。

重构前: 4个方法，总计~230行
重构后: 4个策略类 + 1个协调器，平均~50行/类
"""

from typing import List, Dict, Any
from abc import ABC, abstractmethod


class APIEndpoint:
    """API端点定义（临时数据类）"""
    
    def __init__(self, path: str, method: str, summary: str, description: str,
                 tags: List[str], parameters: List[Dict] = None,
                 request_body: Dict = None, responses: Dict = None):
        self.path = path
        self.method = method
        self.summary = summary
        self.description = description
        self.tags = tags
        self.parameters = parameters or []
        self.request_body = request_body
        self.responses = responses or {}


class EndpointBuildStrategy(ABC):
    """端点构建策略基类"""
    
    @abstractmethod
    def build_endpoints(self) -> List[APIEndpoint]:
        """构建端点列表"""
        pass
    
    def _create_endpoint(
        self,
        path: str,
        method: str,
        summary: str,
        description: str,
        tags: List[str],
        parameters: List[Dict] = None,
        request_body: Dict = None,
        responses: Dict = None
    ) -> APIEndpoint:
        """创建端点的辅助方法"""
        return APIEndpoint(
            path=path,
            method=method,
            summary=summary,
            description=description,
            tags=tags,
            parameters=parameters,
            request_body=request_body,
            responses=responses
        )


class DataServiceEndpointBuilder(EndpointBuildStrategy):
    """
    数据服务端点构建器
    
    原方法: _add_data_service_endpoints(73行)
    新策略: DataServiceEndpointBuilder.build_endpoints(~50行)
    
    优化: 行数-32%, 职责单一
    """
    
    def build_endpoints(self) -> List[APIEndpoint]:
        """构建数据服务端点"""
        return [
            self._build_market_data_endpoint(),
            self._build_data_validation_endpoint(),
            self._build_kline_data_endpoint(),
            self._build_historical_data_endpoint(),
        ]
    
    def _build_market_data_endpoint(self) -> APIEndpoint:
        """构建市场数据获取端点"""
        return self._create_endpoint(
            path="/api/v1/data/market/{symbol}",
            method="GET",
            summary="获取市场数据",
            description="获取指定交易对的实时市场数据",
            tags=["Data Service"],
            parameters=[
                {"name": "symbol", "in": "path", "required": True,
                 "schema": {"type": "string"}, "description": "交易对符号"},
                {"name": "interval", "in": "query",
                 "schema": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "1d"]},
                 "description": "时间间隔"},
                {"name": "limit", "in": "query",
                 "schema": {"type": "integer", "minimum": 1, "maximum": 1000},
                 "description": "返回数据条数"}
            ],
            responses={
                "200": {"description": "成功", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/MarketDataResponse"}}}},
                "400": {"$ref": "#/components/responses/BadRequest"},
                "404": {"$ref": "#/components/responses/NotFound"},
            }
        )
    
    def _build_data_validation_endpoint(self) -> APIEndpoint:
        """构建数据验证端点"""
        return self._create_endpoint(
            path="/api/v1/data/validate",
            method="POST",
            summary="验证数据质量",
            description="验证上传数据的质量和完整性",
            tags=["Data Service"],
            request_body={
                "required": True,
                "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/DataValidationRequest"}}}
            },
            responses={
                "200": {"description": "验证完成", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/DataValidationResponse"}}}},
                "400": {"$ref": "#/components/responses/BadRequest"},
            }
        )
    
    def _build_kline_data_endpoint(self) -> APIEndpoint:
        """构建K线数据端点"""
        return self._create_endpoint(
            path="/api/v1/data/kline",
            method="GET",
            summary="获取K线数据",
            description="获取指定交易对的K线数据",
            tags=["Data Service"],
            parameters=[
                {"name": "symbol", "in": "query", "required": True,
                 "schema": {"type": "string"}},
                {"name": "interval", "in": "query", "required": True,
                 "schema": {"type": "string"}},
                {"name": "start_time", "in": "query",
                 "schema": {"type": "string", "format": "date-time"}},
                {"name": "end_time", "in": "query",
                 "schema": {"type": "string", "format": "date-time"}},
            ],
            responses={
                "200": {"description": "成功", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/MarketDataResponse"}}}},
            }
        )
    
    def _build_historical_data_endpoint(self) -> APIEndpoint:
        """构建历史数据端点"""
        return self._create_endpoint(
            path="/api/v1/data/history",
            method="GET",
            summary="查询历史数据",
            description="查询指定时间范围的历史数据",
            tags=["Data Service"],
            parameters=[
                {"name": "symbol", "in": "query", "required": True,
                 "schema": {"type": "string"}},
                {"name": "start_date", "in": "query", "required": True,
                 "schema": {"type": "string", "format": "date"}},
                {"name": "end_date", "in": "query", "required": True,
                 "schema": {"type": "string", "format": "date"}},
                {"name": "page", "in": "query",
                 "schema": {"type": "integer", "default": 1}},
                {"name": "page_size", "in": "query",
                 "schema": {"type": "integer", "default": 100}},
            ],
            responses={
                "200": {"description": "成功", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/PaginatedResponse"}}}},
            }
        )


class FeatureServiceEndpointBuilder(EndpointBuildStrategy):
    """
    特征工程服务端点构建器
    
    原方法: _add_feature_service_endpoints(58行)
    新策略: FeatureServiceEndpointBuilder.build_endpoints(~40行)
    """
    
    def build_endpoints(self) -> List[APIEndpoint]:
        """构建特征工程服务端点"""
        return [
            self._build_feature_compute_endpoint(),
            self._build_feature_extraction_endpoint(),
        ]
    
    def _build_feature_compute_endpoint(self) -> APIEndpoint:
        """构建特征计算端点"""
        return self._create_endpoint(
            path="/api/v1/features/compute",
            method="POST",
            summary="计算技术指标",
            description="计算各类技术指标特征",
            tags=["Feature Engineering"],
            request_body={
                "required": True,
                "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/FeatureComputeRequest"}}}
            },
            responses={
                "200": {"description": "计算成功", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/FeatureComputeResponse"}}}},
                "400": {"$ref": "#/components/responses/BadRequest"},
            }
        )
    
    def _build_feature_extraction_endpoint(self) -> APIEndpoint:
        """构建特征提取端点"""
        return self._create_endpoint(
            path="/api/v1/features/extract",
            method="POST",
            summary="提取特征",
            description="从原始数据中提取特征",
            tags=["Feature Engineering"],
            request_body={
                "required": True,
                "content": {"application/json": {"schema": {"type": "object"}}}
            },
            responses={
                "200": {"description": "提取成功"},
                "400": {"$ref": "#/components/responses/BadRequest"},
            }
        )


class TradingServiceEndpointBuilder(EndpointBuildStrategy):
    """
    交易服务端点构建器
    
    原方法: _add_trading_service_endpoints(59行)
    新策略: TradingServiceEndpointBuilder.build_endpoints(~45行)
    """
    
    def build_endpoints(self) -> List[APIEndpoint]:
        """构建交易服务端点"""
        return [
            self._build_create_order_endpoint(),
            self._build_cancel_order_endpoint(),
            self._build_query_orders_endpoint(),
            self._build_query_positions_endpoint(),
        ]
    
    def _build_create_order_endpoint(self) -> APIEndpoint:
        """构建创建订单端点"""
        return self._create_endpoint(
            path="/api/v1/trading/orders",
            method="POST",
            summary="创建订单",
            description="创建新的交易订单",
            tags=["Trading Service"],
            request_body={
                "required": True,
                "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/OrderRequest"}}}
            },
            responses={
                "201": {"description": "订单创建成功", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/OrderResponse"}}}},
                "400": {"$ref": "#/components/responses/BadRequest"},
            }
        )
    
    def _build_cancel_order_endpoint(self) -> APIEndpoint:
        """构建取消订单端点"""
        return self._create_endpoint(
            path="/api/v1/trading/orders/{order_id}",
            method="DELETE",
            summary="取消订单",
            description="取消指定的订单",
            tags=["Trading Service"],
            parameters=[
                {"name": "order_id", "in": "path", "required": True,
                 "schema": {"type": "string"}}
            ],
            responses={
                "200": {"description": "取消成功"},
                "404": {"$ref": "#/components/responses/NotFound"},
            }
        )
    
    def _build_query_orders_endpoint(self) -> APIEndpoint:
        """构建查询订单端点"""
        return self._create_endpoint(
            path="/api/v1/trading/orders",
            method="GET",
            summary="查询订单",
            description="查询订单列表",
            tags=["Trading Service"],
            parameters=[
                {"name": "symbol", "in": "query", "schema": {"type": "string"}},
                {"name": "status", "in": "query", "schema": {"type": "string"}},
                {"name": "limit", "in": "query", "schema": {"type": "integer"}},
            ],
            responses={
                "200": {"description": "查询成功", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/PaginatedResponse"}}}},
            }
        )
    
    def _build_query_positions_endpoint(self) -> APIEndpoint:
        """构建查询持仓端点"""
        return self._create_endpoint(
            path="/api/v1/trading/positions",
            method="GET",
            summary="查询持仓",
            description="查询当前持仓信息",
            tags=["Trading Service"],
            responses={
                "200": {"description": "查询成功"},
            }
        )


class MonitoringServiceEndpointBuilder(EndpointBuildStrategy):
    """
    监控服务端点构建器
    
    原方法: _add_monitoring_service_endpoints(40行)
    新策略: MonitoringServiceEndpointBuilder.build_endpoints(~30行)
    """
    
    def build_endpoints(self) -> List[APIEndpoint]:
        """构建监控服务端点"""
        return [
            self._build_health_check_endpoint(),
            self._build_metrics_endpoint(),
        ]
    
    def _build_health_check_endpoint(self) -> APIEndpoint:
        """构建健康检查端点"""
        return self._create_endpoint(
            path="/api/v1/health",
            method="GET",
            summary="健康检查",
            description="检查系统健康状态",
            tags=["Monitoring"],
            responses={
                "200": {"description": "系统健康", "content": {"application/json": {
                    "schema": {"$ref": "#/components/schemas/HealthCheckResponse"}}}},
            }
        )
    
    def _build_metrics_endpoint(self) -> APIEndpoint:
        """构建指标查询端点"""
        return self._create_endpoint(
            path="/api/v1/metrics",
            method="GET",
            summary="查询性能指标",
            description="查询系统性能指标",
            tags=["Monitoring"],
            parameters=[
                {"name": "metric_name", "in": "query", "schema": {"type": "string"}},
                {"name": "start_time", "in": "query", "schema": {"type": "string"}},
                {"name": "end_time", "in": "query", "schema": {"type": "string"}},
            ],
            responses={
                "200": {"description": "查询成功"},
            }
        )


class EndpointBuilderCoordinator:
    """
    端点构建协调器
    
    职责：
    - 协调各服务的端点构建器
    - 提供统一的端点管理接口
    - 支持动态添加新服务
    """
    
    def __init__(self):
        """初始化协调器"""
        self._builders = {
            'data_service': DataServiceEndpointBuilder(),
            'feature_service': FeatureServiceEndpointBuilder(),
            'trading_service': TradingServiceEndpointBuilder(),
            'monitoring_service': MonitoringServiceEndpointBuilder(),
        }
        self._all_endpoints: List[APIEndpoint] = []
    
    def build_all_endpoints(self) -> List[APIEndpoint]:
        """构建所有服务的端点"""
        all_endpoints = []
        
        for service_type, builder in self._builders.items():
            endpoints = builder.build_endpoints()
            all_endpoints.extend(endpoints)
        
        self._all_endpoints = all_endpoints
        return all_endpoints
    
    def get_endpoints_by_service(self, service_type: str) -> List[APIEndpoint]:
        """获取指定服务的端点"""
        builder = self._builders.get(service_type)
        if builder:
            return builder.build_endpoints()
        return []
    
    def add_service_builder(self, service_type: str, builder: EndpointBuildStrategy):
        """添加新服务的端点构建器"""
        self._builders[service_type] = builder
    
    def count_endpoints(self) -> Dict[str, int]:
        """统计端点数量"""
        stats = {'total': 0}
        
        for service_type, builder in self._builders.items():
            endpoints = builder.build_endpoints()
            stats[service_type] = len(endpoints)
            stats['total'] += len(endpoints)
        
        return stats


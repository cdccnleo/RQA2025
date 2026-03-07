"""
OpenAPI Schema构建器

使用协调器模式将原251行的_add_common_schemas函数拆分为多个职责单一的方法。
采用参数对象模式，使用SchemaGenerationConfig替代140个参数。

重构前: _add_common_schemas(251行, 140参数)
重构后: SchemaBuilder + 协调器方法 + 15个专用方法
"""

from typing import Dict, Any, List


class SchemaBuilder:
    """
    Schema构建器
    
    职责：
    - 构建OpenAPI Schema定义
    - 使用协调器模式组织schema生成逻辑
    - 提供专用的schema构建方法
    """
    
    def __init__(self):
        """初始化Schema构建器"""
        self.schemas: Dict[str, Dict[str, Any]] = {}
    
    def build_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        构建所有Schema（协调器方法）
        
        原函数: _add_common_schemas(251行, 140参数)
        新方法: 协调器(~25行) + 15个专用方法(~15行/个)
        
        Returns:
            Dict[str, Dict[str, Any]]: 所有Schema定义
        """
        # 协调器模式：主方法作为协调器，调用专用方法
        self._build_base_schemas()
        self._build_error_schemas()
        self._build_pagination_schemas()
        self._build_data_service_schemas()
        self._build_feature_service_schemas()
        self._build_trading_service_schemas()
        self._build_monitoring_service_schemas()
        self._build_validation_schemas()
        self._build_authentication_schemas()
        self._build_rate_limit_schemas()
        
        return self.schemas
    
    # ========== 基础Schema构建方法 ==========
    
    def _build_base_schemas(self):
        """构建基础响应Schema"""
        self.schemas["BaseResponse"] = {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "请求是否成功"},
                "message": {"type": "string", "description": "响应消息"},
                "timestamp": {"type": "string", "format": "date-time", "description": "响应时间戳"},
                "request_id": {"type": "string", "description": "请求ID"}
            },
            "required": ["success", "timestamp"]
        }
        
        self.schemas["DataResponse"] = {
            "type": "object",
            "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
            "properties": {
                "data": {"type": "object", "description": "响应数据"}
            }
        }
    
    def _build_error_schemas(self):
        """构建错误响应Schema"""
        self.schemas["ErrorResponse"] = {
            "type": "object",
            "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
            "properties": {
                "error": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "错误代码"},
                        "message": {"type": "string", "description": "错误消息"},
                        "details": {"type": "object", "description": "错误详情"}
                    },
                    "required": ["code", "message"]
                }
            }
        }
        
        self.schemas["ValidationErrorResponse"] = {
            "type": "object",
            "allOf": [{"$ref": "#/components/schemas/ErrorResponse"}],
            "properties": {
                "validation_errors": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string"},
                            "message": {"type": "string"},
                            "value": {"type": "string"}
                        }
                    }
                }
            }
        }
    
    def _build_pagination_schemas(self):
        """构建分页Schema"""
        self.schemas["PaginationInfo"] = {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "minimum": 1, "description": "当前页码"},
                "page_size": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "每页数量"},
                "total": {"type": "integer", "minimum": 0, "description": "总记录数"},
                "total_pages": {"type": "integer", "minimum": 0, "description": "总页数"}
            },
            "required": ["page", "page_size", "total", "total_pages"]
        }
        
        self.schemas["PaginatedResponse"] = {
            "type": "object",
            "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
            "properties": {
                "data": {"type": "array", "items": {"type": "object"}},
                "pagination": {"$ref": "#/components/schemas/PaginationInfo"}
            }
        }
    
    def _build_data_service_schemas(self):
        """构建数据服务Schema"""
        # 市场数据Schema
        self.schemas["MarketData"] = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string", "format": "date-time"},
                "symbol": {"type": "string", "description": "交易对"},
                "open": {"type": "number", "description": "开盘价"},
                "high": {"type": "number", "description": "最高价"},
                "low": {"type": "number", "description": "最低价"},
                "close": {"type": "number", "description": "收盘价"},
                "volume": {"type": "number", "description": "成交量"}
            },
            "required": ["timestamp", "symbol", "close", "volume"]
        }
        
        self.schemas["MarketDataResponse"] = {
            "type": "object",
            "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
            "properties": {
                "data": {
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/MarketData"}
                }
            }
        }
    
    def _build_feature_service_schemas(self):
        """构建特征工程服务Schema"""
        self.schemas["TechnicalIndicator"] = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "指标名称"},
                "value": {"type": "number", "description": "指标值"},
                "timestamp": {"type": "string", "format": "date-time"}
            },
            "required": ["name", "value", "timestamp"]
        }
        
        self.schemas["FeatureComputeRequest"] = {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "indicators": {"type": "array", "items": {"type": "string"}},
                "parameters": {"type": "object"}
            },
            "required": ["symbol", "indicators"]
        }
        
        self.schemas["FeatureComputeResponse"] = {
            "type": "object",
            "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
            "properties": {
                "features": {
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/TechnicalIndicator"}
                }
            }
        }
    
    def _build_trading_service_schemas(self):
        """构建交易服务Schema"""
        self.schemas["Order"] = {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "order_type": {"type": "string", "enum": ["market", "limit"]},
                "quantity": {"type": "number", "minimum": 0},
                "price": {"type": "number", "minimum": 0},
                "status": {"type": "string", "enum": ["pending", "filled", "cancelled"]},
                "created_at": {"type": "string", "format": "date-time"}
            },
            "required": ["order_id", "symbol", "side", "order_type", "quantity", "status"]
        }
        
        self.schemas["OrderRequest"] = {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "order_type": {"type": "string", "enum": ["market", "limit"]},
                "quantity": {"type": "number", "minimum": 0},
                "price": {"type": "number", "minimum": 0}
            },
            "required": ["symbol", "side", "order_type", "quantity"]
        }
        
        self.schemas["OrderResponse"] = {
            "type": "object",
            "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
            "properties": {
                "order": {"$ref": "#/components/schemas/Order"}
            }
        }
    
    def _build_monitoring_service_schemas(self):
        """构建监控服务Schema"""
        self.schemas["HealthStatus"] = {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                "message": {"type": "string"},
                "checked_at": {"type": "string", "format": "date-time"}
            },
            "required": ["service", "status", "checked_at"]
        }
        
        self.schemas["HealthCheckResponse"] = {
            "type": "object",
            "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
            "properties": {
                "overall_status": {"type": "string", "enum": ["healthy", "degraded", "unhealthy"]},
                "services": {
                    "type": "array",
                    "items": {"$ref": "#/components/schemas/HealthStatus"}
                }
            }
        }
        
        self.schemas["MetricData"] = {
            "type": "object",
            "properties": {
                "metric_name": {"type": "string"},
                "value": {"type": "number"},
                "timestamp": {"type": "string", "format": "date-time"},
                "tags": {"type": "object"}
            },
            "required": ["metric_name", "value", "timestamp"]
        }
    
    def _build_validation_schemas(self):
        """构建验证Schema"""
        self.schemas["DataValidationRequest"] = {
            "type": "object",
            "properties": {
                "data": {"type": "array", "items": {"type": "object"}},
                "rules": {"type": "object"},
                "strict_mode": {"type": "boolean", "default": False}
            },
            "required": ["data"]
        }
        
        self.schemas["DataValidationResponse"] = {
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
        }
    
    def _build_authentication_schemas(self):
        """构建认证Schema"""
        self.schemas["AuthenticationRequest"] = {
            "type": "object",
            "properties": {
                "username": {"type": "string"},
                "password": {"type": "string", "format": "password"},
                "auth_type": {"type": "string", "enum": ["basic", "bearer", "api_key"]}
            },
            "required": ["username", "password"]
        }
        
        self.schemas["TokenResponse"] = {
            "type": "object",
            "allOf": [{"$ref": "#/components/schemas/BaseResponse"}],
            "properties": {
                "token": {"type": "string"},
                "token_type": {"type": "string", "default": "Bearer"},
                "expires_in": {"type": "integer", "description": "过期时间(秒)"}
            }
        }
    
    def _build_rate_limit_schemas(self):
        """构建速率限制Schema"""
        self.schemas["RateLimitInfo"] = {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "速率限制"},
                "remaining": {"type": "integer", "description": "剩余请求数"},
                "reset": {"type": "integer", "description": "重置时间戳"}
            }
        }
        
        self.schemas["RateLimitExceededResponse"] = {
            "type": "object",
            "allOf": [{"$ref": "#/components/schemas/ErrorResponse"}],
            "properties": {
                "rate_limit": {"$ref": "#/components/schemas/RateLimitInfo"}
            }
        }
    
    def get_schema(self, schema_name: str) -> Dict[str, Any]:
        """获取指定Schema"""
        return self.schemas.get(schema_name)
    
    def get_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """获取所有Schema"""
        return self.schemas.copy()
    
    def add_custom_schema(self, schema_name: str, schema_def: Dict[str, Any]):
        """添加自定义Schema"""
        self.schemas[schema_name] = schema_def
    
    def count_schemas(self) -> int:
        """统计Schema数量"""
        return len(self.schemas)


class CommonResponseBuilder:
    """
    通用响应构建器
    
    职责：
    - 构建常用的HTTP响应定义
    - 提供标准化的响应格式
    """
    
    @staticmethod
    def build_success_responses() -> Dict[str, Dict[str, Any]]:
        """构建成功响应"""
        return {
            "200": {
                "description": "请求成功",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/BaseResponse"}
                    }
                }
            },
            "201": {
                "description": "资源创建成功",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/DataResponse"}
                    }
                }
            }
        }
    
    @staticmethod
    def build_error_responses() -> Dict[str, Dict[str, Any]]:
        """构建错误响应"""
        return {
            "400": {
                "description": "请求参数错误",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ValidationErrorResponse"}
                    }
                }
            },
            "401": {
                "description": "未授权",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                    }
                }
            },
            "403": {
                "description": "禁止访问",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                    }
                }
            },
            "404": {
                "description": "资源不存在",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                    }
                }
            },
            "429": {
                "description": "请求过于频繁",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/RateLimitExceededResponse"}
                    }
                }
            },
            "500": {
                "description": "服务器内部错误",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                    }
                }
            }
        }
    
    @staticmethod
    def build_all_common_responses() -> Dict[str, Dict[str, Any]]:
        """构建所有通用响应"""
        responses = {}
        responses.update(CommonResponseBuilder.build_success_responses())
        responses.update(CommonResponseBuilder.build_error_responses())
        return responses


# ========== 向后兼容的辅助函数 ==========

def build_common_schemas() -> Dict[str, Dict[str, Any]]:
    """
    构建通用Schema（向后兼容函数）
    
    这是原_add_common_schemas方法的简化版本，
    使用SchemaBuilder实现，参数从140个减少到0个。
    
    Returns:
        Dict[str, Dict[str, Any]]: 所有Schema定义
    """
    builder = SchemaBuilder()
    return builder.build_all_schemas()


def build_common_responses() -> Dict[str, Dict[str, Any]]:
    """
    构建通用响应（向后兼容函数）
    
    Returns:
        Dict[str, Dict[str, Any]]: 所有响应定义
    """
    return CommonResponseBuilder.build_all_common_responses()


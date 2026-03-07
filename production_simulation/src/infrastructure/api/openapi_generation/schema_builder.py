"""
API模式构建器

职责：构建OpenAPI schemas和数据模型定义
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class APISchema:
    """API模式定义"""
    title: str
    version: str
    description: str
    servers: List[Dict[str, str]] = field(default_factory=list)
    endpoints: List[Any] = field(default_factory=list)
    schemas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    security_schemes: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class SchemaBuilder:
    """模式构建器"""
    
    def __init__(self):
        """初始化模式构建器"""
        self.schemas: Dict[str, Dict[str, Any]] = {}
    
    def create_object_schema(
        self,
        name: str,
        properties: Dict[str, Dict[str, Any]],
        required: List[str] = None,
        description: str = ""
    ) -> Dict[str, Any]:
        """
        创建对象模式
        
        Args:
            name: 模式名称
            properties: 属性定义
            required: 必需字段列表
            description: 描述
            
        Returns:
            创建的模式
        """
        schema = {
            "type": "object",
            "properties": properties
        }
        
        if description:
            schema["description"] = description
        
        if required:
            schema["required"] = required
        
        self.schemas[name] = schema
        return schema
    
    def create_array_schema(
        self,
        name: str,
        item_schema: Dict[str, Any],
        description: str = ""
    ) -> Dict[str, Any]:
        """创建数组模式"""
        schema = {
            "type": "array",
            "items": item_schema
        }
        
        if description:
            schema["description"] = description
        
        self.schemas[name] = schema
        return schema
    
    def create_enum_schema(
        self,
        name: str,
        values: List[Any],
        value_type: str = "string",
        description: str = ""
    ) -> Dict[str, Any]:
        """创建枚举模式"""
        schema = {
            "type": value_type,
            "enum": values
        }
        
        if description:
            schema["description"] = description
        
        self.schemas[name] = schema
        return schema
    
    def add_common_data_schemas(self):
        """添加通用数据模式"""
        # 数据集模式
        self.create_object_schema(
            "Dataset",
            {
                "dataset_id": {"type": "string", "description": "数据集ID"},
                "name": {"type": "string", "description": "数据集名称"},
                "description": {"type": "string", "description": "数据集描述"},
                "source": {"type": "string", "description": "数据源"},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"}
            },
            required=["dataset_id", "name", "source"],
            description="数据集信息"
        )
        
        # 股票数据模式
        self.create_object_schema(
            "StockData",
            {
                "symbol": {"type": "string", "description": "股票代码"},
                "date": {"type": "string", "format": "date", "description": "日期"},
                "open": {"type": "number", "description": "开盘价"},
                "high": {"type": "number", "description": "最高价"},
                "low": {"type": "number", "description": "最低价"},
                "close": {"type": "number", "description": "收盘价"},
                "volume": {"type": "integer", "description": "成交量"}
            },
            required=["symbol", "date", "close"],
            description="股票数据"
        )
    
    def add_common_response_schemas(self):
        """添加通用响应模式"""
        # 成功响应
        self.create_object_schema(
            "SuccessResponse",
            {
                "success": {"type": "boolean", "example": True},
                "message": {"type": "string"},
                "data": {"type": "object"}
            },
            required=["success"],
            description="成功响应"
        )
        
        # 错误响应
        self.create_object_schema(
            "ErrorResponse",
            {
                "success": {"type": "boolean", "example": False},
                "error": {"type": "string"},
                "error_code": {"type": "string"},
                "details": {"type": "object"}
            },
            required=["success", "error"],
            description="错误响应"
        )
        
        # 分页响应
        self.create_object_schema(
            "PaginatedResponse",
            {
                "success": {"type": "boolean"},
                "data": {"type": "array", "items": {"type": "object"}},
                "pagination": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer"},
                        "page_size": {"type": "integer"},
                        "total": {"type": "integer"},
                        "total_pages": {"type": "integer"}
                    }
                }
            },
            description="分页响应"
        )
    
    def add_security_schemes(self) -> Dict[str, Dict[str, Any]]:
        """添加安全方案"""
        return {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT Token认证"
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API Key认证"
            }
        }
    
    def get_all_schemas(self) -> Dict[str, Dict[str, Any]]:
        """获取所有模式"""
        return self.schemas.copy()
    
    def clear(self):
        """清空模式"""
        self.schemas.clear()


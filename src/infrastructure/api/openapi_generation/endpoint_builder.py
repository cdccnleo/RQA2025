"""
API端点构建器

职责：构建各种API端点的OpenAPI规范
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


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


class EndpointBuilder:
    """端点构建器"""
    
    def __init__(self):
        """初始化端点构建器"""
        self.endpoints: List[APIEndpoint] = []
    
    def create_endpoint(
        self,
        path: str,
        method: str,
        summary: str,
        description: str = "",
        tags: List[str] = None,
        parameters: List[Dict[str, Any]] = None,
        request_body: Dict[str, Any] = None,
        responses: Dict[str, Dict[str, Any]] = None,
        security: List[Dict[str, Any]] = None
    ) -> APIEndpoint:
        """
        创建API端点
        
        Args:
            path: 端点路径
            method: HTTP方法
            summary: 摘要
            description: 描述
            tags: 标签列表
            parameters: 参数列表
            request_body: 请求体
            responses: 响应定义
            security: 安全要求
            
        Returns:
            APIEndpoint: 端点对象
        """
        endpoint = APIEndpoint(
            path=path,
            method=method,
            summary=summary,
            description=description or summary,
            operation_id=self._generate_operation_id(method, path),
            tags=tags or [],
            parameters=parameters or [],
            request_body=request_body,
            responses=responses or self._default_responses(),
            security=security or []
        )
        
        self.endpoints.append(endpoint)
        return endpoint
    
    def _generate_operation_id(self, method: str, path: str) -> str:
        """生成操作ID"""
        clean_path = path.replace('/', '_').replace('{', '').replace('}', '').strip('_')
        return f"{method.lower()}_{clean_path}"
    
    def _default_responses(self) -> Dict[str, Dict[str, Any]]:
        """默认响应"""
        return {
            "200": {
                "description": "成功",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "data": {"type": "object"}
                            }
                        }
                    }
                }
            },
            "400": {
                "description": "请求错误",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "error": {"type": "string"}
                            }
                        }
                    }
                }
            },
            "500": {
                "description": "服务器错误"
            }
        }
    
    def create_query_parameter(
        self,
        name: str,
        description: str,
        required: bool = False,
        param_type: str = "string",
        default: Any = None
    ) -> Dict[str, Any]:
        """创建查询参数"""
        param = {
            "name": name,
            "in": "query",
            "description": description,
            "required": required,
            "schema": {"type": param_type}
        }
        
        if default is not None:
            param["schema"]["default"] = default
        
        return param
    
    def create_path_parameter(
        self,
        name: str,
        description: str,
        param_type: str = "string"
    ) -> Dict[str, Any]:
        """创建路径参数"""
        return {
            "name": name,
            "in": "path",
            "description": description,
            "required": True,
            "schema": {"type": param_type}
        }
    
    def get_all_endpoints(self) -> List[APIEndpoint]:
        """获取所有端点"""
        return self.endpoints.copy()
    
    def clear(self):
        """清空端点列表"""
        self.endpoints.clear()


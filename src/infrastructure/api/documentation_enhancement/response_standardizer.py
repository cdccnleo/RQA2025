"""
响应标准化器

负责标准化API响应格式，添加通用响应和错误代码。

重构前: APIDocumentationEnhancer中的响应处理逻辑 (~180行)
重构后: ResponseStandardizer独立组件 (~140行)
"""

from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class APIResponseDocumentation:
    """API响应文档"""
    status_code: int
    description: str
    content_type: str = "application/json"
    schema: Dict[str, Any] = field(default_factory=dict)
    examples: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)


class ResponseStandardizer:
    """
    响应标准化器
    
    职责：
    - 创建通用响应模板
    - 创建错误代码定义
    - 标准化端点响应
    """
    
    def __init__(self):
        """初始化响应标准化器"""
        self.common_responses = self._create_common_responses()
        self.error_codes = self._create_error_codes()
    
    def _create_common_responses(self) -> Dict[str, APIResponseDocumentation]:
        """
        创建通用响应模板
        
        原方法: _create_common_responses (132行)
        新方法: 协调器方法 (~40行) + 专用方法
        """
        responses = {}
        
        # 成功响应
        responses.update(self._create_success_responses())
        
        # 客户端错误响应
        responses.update(self._create_client_error_responses())
        
        # 服务器错误响应
        responses.update(self._create_server_error_responses())
        
        return responses
    
    def _create_success_responses(self) -> Dict[str, APIResponseDocumentation]:
        """创建成功响应"""
        return {
            "success": APIResponseDocumentation(
                status_code=200,
                description="请求成功",
                schema={"type": "object", "properties": {
                    "success": {"type": "boolean"},
                    "message": {"type": "string"},
                    "data": {"type": "object"}
                }}
            ),
            "created": APIResponseDocumentation(
                status_code=201,
                description="资源创建成功",
                schema={"type": "object"}
            ),
            "no_content": APIResponseDocumentation(
                status_code=204,
                description="请求成功，无返回内容"
            )
        }
    
    def _create_client_error_responses(self) -> Dict[str, APIResponseDocumentation]:
        """创建客户端错误响应"""
        return {
            "bad_request": APIResponseDocumentation(
                status_code=400,
                description="请求参数错误",
                schema={"$ref": "#/components/schemas/ErrorResponse"}
            ),
            "unauthorized": APIResponseDocumentation(
                status_code=401,
                description="未授权，请先登录",
                schema={"$ref": "#/components/schemas/ErrorResponse"}
            ),
            "forbidden": APIResponseDocumentation(
                status_code=403,
                description="禁止访问，权限不足",
                schema={"$ref": "#/components/schemas/ErrorResponse"}
            ),
            "not_found": APIResponseDocumentation(
                status_code=404,
                description="资源不存在",
                schema={"$ref": "#/components/schemas/ErrorResponse"}
            ),
            "rate_limit": APIResponseDocumentation(
                status_code=429,
                description="请求过于频繁",
                schema={"$ref": "#/components/schemas/RateLimitExceededResponse"}
            )
        }
    
    def _create_server_error_responses(self) -> Dict[str, APIResponseDocumentation]:
        """创建服务器错误响应"""
        return {
            "internal_error": APIResponseDocumentation(
                status_code=500,
                description="服务器内部错误",
                schema={"$ref": "#/components/schemas/ErrorResponse"}
            ),
            "service_unavailable": APIResponseDocumentation(
                status_code=503,
                description="服务暂时不可用",
                schema={"$ref": "#/components/schemas/ErrorResponse"}
            )
        }
    
    def _create_error_codes(self) -> List[Dict[str, Any]]:
        """
        创建错误代码定义
        
        原方法: _create_error_codes (58行)
        新方法: 优化的专用方法 (~50行)
        """
        return [
            # 通用错误
            {"code": "E1000", "message": "未知错误", "category": "general"},
            {"code": "E1001", "message": "参数错误", "category": "validation"},
            {"code": "E1002", "message": "请求格式错误", "category": "validation"},
            
            # 认证错误
            {"code": "E2000", "message": "认证失败", "category": "authentication"},
            {"code": "E2001", "message": "Token无效", "category": "authentication"},
            {"code": "E2002", "message": "Token过期", "category": "authentication"},
            
            # 权限错误
            {"code": "E3000", "message": "权限不足", "category": "authorization"},
            {"code": "E3001", "message": "资源访问被拒绝", "category": "authorization"},
            
            # 资源错误
            {"code": "E4000", "message": "资源不存在", "category": "resource"},
            {"code": "E4001", "message": "资源已存在", "category": "resource"},
            {"code": "E4002", "message": "资源冲突", "category": "resource"},
            
            # 业务错误
            {"code": "E5000", "message": "余额不足", "category": "business"},
            {"code": "E5001", "message": "订单状态错误", "category": "business"},
            {"code": "E5002", "message": "风控拒绝", "category": "business"},
            
            # 系统错误
            {"code": "E9000", "message": "系统错误", "category": "system"},
            {"code": "E9001", "message": "服务不可用", "category": "system"},
            {"code": "E9002", "message": "请求超时", "category": "system"},
        ]
    
    def standardize_responses(self, endpoint):
        """
        标准化端点响应
        
        原方法: _standardize_responses (32行)
        新方法: 优化方法 (~25行)
        """
        # 确保有成功响应
        if not any(r.status_code == 200 for r in endpoint.responses):
            endpoint.responses.insert(0, self.common_responses['success'])
        
        # 添加通用错误响应
        error_codes = [400, 401, 403, 404, 500]
        existing_codes = [r.status_code for r in endpoint.responses]
        
        for code in error_codes:
            if code not in existing_codes:
                response_key = {
                    400: 'bad_request',
                    401: 'unauthorized',
                    403: 'forbidden',
                    404: 'not_found',
                    500: 'internal_error'
                }.get(code)
                
                if response_key in self.common_responses:
                    endpoint.responses.append(self.common_responses[response_key])
    
    def add_error_codes_to_endpoint(self, endpoint):
        """
        为端点添加错误代码
        
        原方法: _add_error_codes (25行)
        新方法: 优化方法 (~20行)
        """
        # 根据端点类型添加相关错误代码
        endpoint.error_codes = []
        
        # 所有端点添加通用错误
        endpoint.error_codes.extend([
            e for e in self.error_codes if e['category'] == 'general'
        ])
        
        # 根据认证需求添加认证错误
        if endpoint.authentication:
            endpoint.error_codes.extend([
                e for e in self.error_codes if e['category'] in ['authentication', 'authorization']
            ])
        
        # 根据方法添加特定错误
        if endpoint.method in ['POST', 'PUT', 'PATCH']:
            endpoint.error_codes.extend([
                e for e in self.error_codes if e['category'] == 'validation'
            ])


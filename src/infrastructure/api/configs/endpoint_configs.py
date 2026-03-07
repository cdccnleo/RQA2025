"""
API端点相关配置

提供API端点定义所需的各类配置对象
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from .base_config import BaseConfig, ValidationResult

# 提前导入以解决循环依赖
from .schema_configs import SchemaDefinitionConfig


@dataclass
class EndpointParameterConfig(BaseConfig):
    """端点参数配置"""
    
    name: str
    in_location: str  # path, query, header, cookie
    parameter_type: str  # string, integer, number, boolean, array, object
    description: Optional[str] = None
    
    # 约束
    required: bool = False
    deprecated: bool = False
    
    # 类型特定配置
    schema: Optional[Dict[str, Any]] = None
    example: Optional[Any] = None
    default: Optional[Any] = None
    
    # 验证规则
    pattern: Optional[str] = None
    enum: Optional[List[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    
    def _validate_impl(self, result: ValidationResult):
        """验证参数配置"""
        if not self.name:
            result.add_error("参数名称不能为空")
        
        valid_locations = ['path', 'query', 'header', 'cookie']
        if self.in_location not in valid_locations:
            result.add_error(f"参数位置必须是 {valid_locations} 之一")
        
        valid_types = ['string', 'integer', 'number', 'boolean', 'array', 'object']
        if self.parameter_type not in valid_types:
            result.add_error(f"参数类型必须是 {valid_types} 之一")
        
        # 路径参数必须是required
        if self.in_location == 'path' and not self.required:
            result.add_error("路径参数必须是必需的")


@dataclass
class EndpointResponseConfig(BaseConfig):
    """端点响应配置"""
    
    status_code: int
    description: str
    
    # 响应内容
    content_type: str = "application/json"
    schema: Optional[Dict[str, Any]] = None
    example: Optional[Any] = None
    
    # 响应头
    headers: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # 响应示例
    examples: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.example is not None and not self.examples:
            self.examples = {"default": self.example}
        super().__post_init__()
    
    def _validate_impl(self, result: ValidationResult):
        """验证响应配置"""
        if not (100 <= self.status_code < 600):
            result.add_error("HTTP状态码必须在100-599之间")
        
        if not self.description:
            result.add_error("响应描述不能为空")
        
        valid_content_types = [
            'application/json',
            'application/xml',
            'text/plain',
            'text/html',
            'application/octet-stream'
        ]
        if self.content_type not in valid_content_types:
            result.add_warning(f"不常见的Content-Type: {self.content_type}")


@dataclass
class EndpointSecurityConfig(BaseConfig):
    """端点安全配置"""

    scheme_name: str
    scheme_type: str  # http, apiKey, oauth2, openIdConnect
    scheme: str
    scopes: Optional[List[str]] = None
    description: Optional[str] = None

    def _validate_impl(self, result: ValidationResult):
        """验证安全配置"""
        if not self.scheme_name:
            result.add_error("安全方案名称不能为空")

        valid_types = ['http', 'apiKey', 'oauth2', 'openIdConnect']
        if self.scheme_type not in valid_types:
            result.add_error(f"安全方案类型必须是 {valid_types} 之一")

        if not self.scheme:
            result.add_error("安全方案标识不能为空")

        if self.scheme_type == 'oauth2' and not self.scopes:
            result.add_warning("OAuth2安全方案建议定义scopes")


@dataclass
class EndpointConfig(BaseConfig):
    """API端点配置"""
    
    path: str
    method: str  # GET, POST, PUT, DELETE, PATCH
    summary: Optional[str] = None
    operation_id: Optional[str] = None
    description: Optional[str] = None
    
    # 分类和标签
    tags: List[str] = field(default_factory=list)
    
    # 参数定义
    parameters: List[EndpointParameterConfig] = field(default_factory=list)
    
    # 请求体
    request_body: Optional[Dict[str, Any]] = None
    request_body_required: bool = False
    
    # 响应定义
    responses: List[EndpointResponseConfig] = field(default_factory=list)
    
    # 安全配置
    security: Optional[List[EndpointSecurityConfig]] = None
    
    # 其他
    deprecated: bool = False
    servers: Optional[List[Dict[str, str]]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.method = (self.method or '').upper()
        self._operation_id_provided = bool(self.operation_id)
        if not self.summary:
            self.summary = self._generate_default_summary()
        if not self.operation_id:
            self.operation_id = self._generate_operation_id()
        super().__post_init__()

    def _generate_operation_id(self) -> str:
        """根据路径和方法生成默认操作ID"""
        path_part = (self.path or '').strip('/').replace('/', '_') or 'root'
        return f"{self.method.lower()}_{path_part}".strip('_')

    def _generate_default_summary(self) -> str:
        """根据路径自动生成摘要"""
        path_fragment = (self.path or '').strip('/')
        if not path_fragment:
            path_fragment = 'root'
        readable = path_fragment.replace('-', ' ').replace('_', ' ')
        return f"{self.method.title()} {readable.title()}".strip()

    def _validate_impl(self, result: ValidationResult):
        """验证端点配置"""
        if not self.path:
            result.add_error("端点路径不能为空")
        
        if not self.path.startswith('/'):
            result.add_error("端点路径必须以/开头")
        
        valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        if self.method.upper() not in valid_methods:
            result.add_error(f"HTTP方法必须是 {valid_methods} 之一 (不支持的HTTP方法: {self.method})")
        
        if not self.summary:
            self.summary = self._generate_default_summary()
        
        if not self._operation_id_provided and self._validation_mode == "strict":
            result.add_error("操作ID不能为空")
        
        if not self.operation_id:
            self.operation_id = self._generate_operation_id()
        
        # 验证参数
        param_names = set()
        for param in self.parameters:
            param_result = param.validate()
            result.merge(param_result)
            
            param_key = f"{param.in_location}:{param.name}"
            if param_key in param_names:
                result.add_error(f"参数重复: {param.name} in {param.in_location}")
            param_names.add(param_key)
        
        # 验证响应
        if not self.responses:
            result.add_warning("端点没有定义任何响应")
        
        status_codes = set()
        for response in self.responses:
            response_result = response.validate()
            result.merge(response_result)
            
            if response.status_code in status_codes:
                result.add_error(f"响应状态码重复: {response.status_code}")
            status_codes.add(response.status_code)
        
        # 检查是否定义了成功响应
        has_success = any(200 <= r.status_code < 300 for r in self.responses)
        if not has_success:
            result.add_warning("端点没有定义成功响应(2xx)")
        
        # POST/PUT/PATCH通常需要请求体
        if self.method.upper() in ['POST', 'PUT', 'PATCH'] and not self.request_body:
            result.add_warning(f"{self.method}方法通常需要定义请求体")
    
    def add_parameter(self, param: EndpointParameterConfig):
        """添加参数"""
        self.parameters.append(param)
    
    def add_response(self, response: EndpointResponseConfig):
        """添加响应"""
        self.responses.append(response)
    
    def get_parameter(self, name: str, location: str = 'query') -> Optional[EndpointParameterConfig]:
        """获取参数"""
        for param in self.parameters:
            if param.name == name and param.in_location == location:
                return param
        return None


@dataclass
class OpenAPIDocConfig(BaseConfig):
    """OpenAPI文档配置"""
    
    # 文档信息
    title: str
    version: str
    description: Optional[str] = None
    
    # 服务器配置
    servers: List[Dict[str, str]] = field(default_factory=list)
    
    # 端点列表
    endpoints: List[EndpointConfig] = field(default_factory=list)
    
    # Schema定义
    schemas: List[SchemaDefinitionConfig] = field(default_factory=list)
    
    # 安全方案
    security_schemes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # 标签定义
    tags: List[Dict[str, str]] = field(default_factory=list)
    
    # 外部文档
    external_docs: Optional[Dict[str, str]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def _validate_impl(self, result: ValidationResult):
        """验证OpenAPI文档配置"""
        if not self.title:
            result.add_error("文档标题不能为空")
        
        if not self.version:
            result.add_error("文档版本不能为空")
        
        # 验证服务器配置
        if not self.servers:
            result.add_warning("没有定义服务器URL")
        
        # 验证端点
        if not self.endpoints:
            result.add_warning("没有定义任何API端点")
        
        endpoint_ids = set()
        for endpoint in self.endpoints:
            endpoint_result = endpoint.validate()
            result.merge(endpoint_result)
            
            endpoint_key = f"{endpoint.method}:{endpoint.path}"
            if endpoint_key in endpoint_ids:
                result.add_error(f"端点重复: {endpoint.method} {endpoint.path}")
            endpoint_ids.add(endpoint_key)
        
        # 验证Schema
        schema_names = set()
        for schema in self.schemas:
            schema_result = schema.validate()
            result.merge(schema_result)
            
            if schema.schema_name in schema_names:
                result.add_error(f"Schema名称重复: {schema.schema_name}")
            schema_names.add(schema.schema_name)
    
    def add_endpoint(self, endpoint: EndpointConfig):
        """添加端点"""
        self.endpoints.append(endpoint)
    
    def add_schema(self, schema: SchemaDefinitionConfig):
        """添加Schema"""
        self.schemas.append(schema)
    
    def add_server(self, url: str, description: str = ""):
        """添加服务器"""
        self.servers.append({
            'url': url,
            'description': description
        })
    
    def count_endpoints(self) -> Dict[str, int]:
        """统计端点数量"""
        stats = {
            'total': len(self.endpoints),
            'by_method': {},
            'by_tag': {}
        }
        
        for endpoint in self.endpoints:
            # 按方法统计
            method = endpoint.method.upper()
            stats['by_method'][method] = stats['by_method'].get(method, 0) + 1
            
            # 按标签统计
            for tag in endpoint.tags:
                stats['by_tag'][tag] = stats['by_tag'].get(tag, 0) + 1
        
        return stats


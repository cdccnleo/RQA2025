"""
Schema生成相关配置

提供OpenAPI Schema生成所需的各类配置对象
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from .base_config import BaseConfig, ValidationResult


@dataclass
class SchemaPropertyConfig(BaseConfig):
    """Schema属性配置"""
    
    name: str
    property_type: str  # string, integer, number, boolean, object, array
    description: Optional[str] = None
    
    # 类型特定配置
    format: Optional[str] = None  # date-time, email, uri等
    enum: Optional[List[Any]] = None
    pattern: Optional[str] = None
    
    # 数值约束
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    
    # 默认值和示例
    default: Optional[Any] = None
    example: Optional[Any] = None
    
    # 数组/对象特定
    items: Optional[Dict[str, Any]] = None
    properties: Optional[Dict[str, Any]] = None
    
    # 其他
    required: bool = False
    nullable: bool = False
    deprecated: bool = False
    
    def __post_init__(self):
        self.property_type = (self.property_type or "").lower()
        super().__post_init__()
    
    def _validate_impl(self, result: ValidationResult):
        """验证属性配置"""
        if not self.name:
            result.add_error("属性名称不能为空")
        
        valid_types = ['string', 'integer', 'number', 'boolean', 'object', 'array', 'null']
        if self.property_type not in valid_types:
            result.add_error(f"不支持的属性类型: {self.property_type}. 允许值: {valid_types}")
        
        # 验证数组类型
        if self.property_type == 'array' and not self.items:
            result.add_error("数组类型必须定义items")
        
        # 验证对象类型
        if self.property_type == 'object' and not self.properties:
            result.add_warning("对象类型建议定义properties")
        
        # 验证数值约束
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                result.add_error("最小值不能大于最大值")


@dataclass
class SchemaDefinitionConfig(BaseConfig):
    """Schema定义配置"""
    
    schema_name: str
    schema_type: str = "object"
    description: Optional[str] = None
    
    # 属性列表
    properties: List[SchemaPropertyConfig] = field(default_factory=list)
    required_properties: List[str] = field(default_factory=list)
    
    # 高级特性
    all_of: Optional[List[Dict[str, Any]]] = None
    one_of: Optional[List[Dict[str, Any]]] = None
    any_of: Optional[List[Dict[str, Any]]] = None
    
    # 其他
    additional_properties: bool = True
    example: Optional[Dict[str, Any]] = None
    deprecated: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.schema_type = (self.schema_type or "").lower()
        super().__post_init__()
    
    def _validate_impl(self, result: ValidationResult):
        """验证Schema定义配置"""
        if not self.schema_name:
            result.add_error("Schema名称不能为空")
        
        valid_schema_types = ['object', 'array', 'string', 'integer', 'number', 'boolean']
        if self.schema_type not in valid_schema_types:
            result.add_error("不支持的Schema类型")
        
        # 验证属性
        property_names = set()
        for prop in self.properties:
            prop_result = prop.validate()
            result.merge(prop_result)
            
            if prop.name in property_names:
                result.add_error(f"属性名称重复: {prop.name}")
            property_names.add(prop.name)
        
        # 验证required列表
        for req_prop in self.required_properties:
            if req_prop not in property_names:
                result.add_error(f"必需属性 {req_prop} 未在properties中定义")
    
    def add_property(self, prop: SchemaPropertyConfig):
        """添加属性"""
        self.properties.append(prop)
    
    def set_required(self, property_name: str):
        """设置属性为必需"""
        if property_name not in self.required_properties:
            self.required_properties.append(property_name)


@dataclass
class ResponseSchemaConfig(BaseConfig):
    """响应Schema配置"""
    
    status_code: int
    description: str
    schema: Optional[SchemaDefinitionConfig] = None
    
    # 响应头
    headers: Dict[str, Dict[str, str]] = field(default_factory=dict)
    
    # 响应示例
    examples: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def _validate_impl(self, result: ValidationResult):
        """验证响应Schema配置"""
        if not (100 <= self.status_code < 600):
            result.add_error("HTTP状态码必须在100-599之间")
        
        if not self.description:
            result.add_error("响应描述不能为空")
        
        if self.schema:
            schema_result = self.schema.validate()
            result.merge(schema_result)


@dataclass
class SchemaGenerationConfig(BaseConfig):
    """Schema生成配置"""
    
    output_format: str = "json"
    output_dir: str = "./schemas"
    strict_validation: bool = False
    custom_validators: List[Callable[['SchemaGenerationConfig', ValidationResult], None]] = field(default_factory=list)
    
    # 基础Schema集合
    base_schemas: List[SchemaDefinitionConfig] = field(default_factory=list)
    error_schemas: List[SchemaDefinitionConfig] = field(default_factory=list)
    data_schemas: List[SchemaDefinitionConfig] = field(default_factory=list)
    trading_schemas: List[SchemaDefinitionConfig] = field(default_factory=list)
    feature_schemas: List[SchemaDefinitionConfig] = field(default_factory=list)
    
    # 通用响应Schema
    common_responses: List[ResponseSchemaConfig] = field(default_factory=list)
    
    # 生成选项
    include_examples: bool = True
    include_descriptions: bool = True
    validate_schemas: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.output_format = (self.output_format or "").lower()
        super().__post_init__()
    
    def _validate_impl(self, result: ValidationResult):
        """验证Schema生成配置"""
        valid_formats = ['json', 'yaml', 'yml']
        if self.output_format not in valid_formats:
            result.add_error("不支持的输出格式")
        
        if not self.output_dir:
            result.add_error("输出目录不能为空")
        
        all_schemas = (
            self.base_schemas + 
            self.error_schemas + 
            self.data_schemas + 
            self.trading_schemas + 
            self.feature_schemas
        )
        
        if not all_schemas:
            result.add_warning("没有定义任何Schema")
        
        # 验证所有Schema
        schema_names = set()
        for schema in all_schemas:
            schema_result = schema.validate()
            result.merge(schema_result)
            
            if schema.schema_name in schema_names:
                result.add_error(f"Schema名称重复: {schema.schema_name}")
            schema_names.add(schema.schema_name)
        
        # 验证响应Schema
        for response in self.common_responses:
            response_result = response.validate()
            result.merge(response_result)
        
        # 自定义验证器
        for validator in self.custom_validators:
            if not callable(validator):
                result.add_error("自定义验证器必须是可调用对象")
                continue
            try:
                validator(self, result)
            except Exception as exc:  # pragma: no cover - 防御性
                result.add_error(f"自定义验证器执行失败: {exc}")
    
    def add_schema(self, schema: SchemaDefinitionConfig, category: str = 'base'):
        """添加Schema"""
        category_map = {
            'base': self.base_schemas,
            'error': self.error_schemas,
            'data': self.data_schemas,
            'trading': self.trading_schemas,
            'feature': self.feature_schemas,
        }
        
        target_list = category_map.get(category, self.base_schemas)
        target_list.append(schema)
    
    def get_all_schemas(self) -> List[SchemaDefinitionConfig]:
        """获取所有Schema"""
        return (
            self.base_schemas + 
            self.error_schemas + 
            self.data_schemas + 
            self.trading_schemas + 
            self.feature_schemas
        )
    
    def count_schemas(self) -> Dict[str, int]:
        """统计Schema数量"""
        return {
            'total': len(self.get_all_schemas()),
            'base': len(self.base_schemas),
            'error': len(self.error_schemas),
            'data': len(self.data_schemas),
            'trading': len(self.trading_schemas),
            'feature': len(self.feature_schemas),
        }


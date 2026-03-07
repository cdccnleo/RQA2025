"""
schema 模块

提供 schema 相关功能和接口。
"""

import re
import re

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional
"""
配置验证模式模块
提供配置验证功能
"""


class SchemaConfigValidator(ABC):

    """
schema - 配置管理

职责说明：
负责系统配置的统一管理、配置文件的读取、配置验证和配置分发

核心职责：
- 配置文件的读取和解析
- 配置参数的验证
- 配置的热重载
- 配置的分发和同步
- 环境变量管理
- 配置加密和安全

相关接口：
- IConfigComponent
- IConfigManager
- IConfigValidator
"""

    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> bool:
        """
        验证配置

        Args:
            config: 配置数据

        Returns:
            是否有效
        """

    @abstractmethod
    def get_errors(self) -> List[str]:
        """
        获取验证错误

        Returns:
            错误列表
        """


class SchemaValidator(SchemaConfigValidator):

    """基于模式的配置验证器"""

    def __init__(self, schema: Dict[str, Any]):

        self.schema = schema
        self.errors: List[str] = []

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        验证配置

        Args:
            config: 配置数据

        Returns:
            是否有效
        """
        self.errors.clear()
        return self._validate_object(config, self.schema)

    def get_errors(self) -> List[str]:
        """获取验证错误"""
        return self.errors.copy()

    def _validate_object(self, data: Any, schema: Dict[str, Any]) -> bool:
        """验证对象"""
        if not isinstance(data, dict):
            self.errors.append(f"期望对象类型，实际为 {type(data).__name__}")
            return False

        # 检查必需字段
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                self.errors.append(f"缺少必需字段: {field}")

        # 验证字段
        properties = schema.get("properties", {})
        for field, value in data.items():
            if field in properties:
                field_schema = properties[field]
                if not self._validate_value(value, field_schema, field):
                    return False

        return len(self.errors) == 0

    def _validate_value(self, value: Any, schema: Dict[str, Any], field_path: str = "") -> bool:
        """验证值"""
        schema_type = schema.get("type")

        if schema_type == "object":
            return self._validate_object(value, schema)
        elif schema_type == "array":
            return self._validate_array(value, schema, field_path)
        elif schema_type == "string":
            return self._validate_string(value, schema, field_path)
        elif schema_type == "number":
            return self._validate_number(value, schema, field_path)
        elif schema_type == "integer":
            return self._validate_integer(value, schema, field_path)
        elif schema_type == "boolean":
            return self._validate_boolean(value, schema, field_path)
        else:
            return True

    def _validate_array(self, data: Any, schema: Dict[str, Any], field_path: str) -> bool:
        """验证数组"""
        if not isinstance(data, list):
            self.errors.append(f"{field_path}: 期望数组类型，实际为 {type(data).__name__}")
            return False

        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(data):
                item_path = f"{field_path}[{i}]"
                if not self._validate_value(item, items_schema, item_path):
                    return False

        return True

    def _validate_string(self, data: Any, schema: Dict[str, Any], field_path: str) -> bool:
        """验证字符串"""
        if not isinstance(data, str):
            self.errors.append(f"{field_path}: 期望字符串类型，实际为 {type(data).__name__}")
            return False

        # 检查最小长度
        min_length = schema.get("minLength")
        if min_length and len(data) < min_length:
            self.errors.append(f"{field_path}: 字符串长度小于最小值 {min_length}")
            return False

        # 检查最大长度
        max_length = schema.get("maxLength")
        if max_length and len(data) > max_length:
            self.errors.append(f"{field_path}: 字符串长度大于最大值 {max_length}")
            return False

        # 检查模式
        pattern = schema.get("pattern")
        if pattern:
            if not re.match(pattern, data):
                self.errors.append(f"{field_path}: 字符串不匹配模式 {pattern}")
                return False

        return True

    def _validate_number(self, data: Any, schema: Dict[str, Any], field_path: str) -> bool:
        """验证数字"""
        if not isinstance(data, (int, float)):
            self.errors.append(f"{field_path}: 期望数字类型，实际为 {type(data).__name__}")
            return False

        # 检查最小值
        minimum = schema.get("minimum")
        if minimum is not None and data < minimum:
            self.errors.append(f"{field_path}: 数值小于最小值 {minimum}")
            return False

        # 检查最大值
        maximum = schema.get("maximum")
        if maximum is not None and data > maximum:
            self.errors.append(f"{field_path}: 数值大于最大值 {maximum}")
            return False

        return True

    def _validate_integer(self, data: Any, schema: Dict[str, Any], field_path: str) -> bool:
        """验证整数"""
        if not isinstance(data, int):
            self.errors.append(f"{field_path}: 期望整数类型，实际为 {type(data).__name__}")
            return False

        return self._validate_number(data, schema, field_path)

    def _validate_boolean(self, data: Any, schema: Dict[str, Any], field_path: str) -> bool:
        """验证布尔值"""
        if not isinstance(data, bool):
            self.errors.append(f"{field_path}: 期望布尔类型，实际为 {type(data).__name__}")
            return False

        return True


class ConfigSchema:

    """配置模式"""

    def __init__(self, schema: Dict[str, Any]):

        self.schema = schema

    def validate(self, config: Dict[str, Any]) -> bool:
        """验证配置"""
        validator = SchemaValidator(self.schema)
        return validator.validate(config)

    def get_errors(self) -> List[str]:
        """获取错误"""
        validator = SchemaValidator(self.schema)
        return validator.get_errors()

    def get_schema(self) -> Dict[str, Any]:
        """获取模式"""
        return self.schema.copy()


class ConfigSchemaRegistry:

    """配置模式注册表"""

    def __init__(self):

        self._schemas: Dict[str, ConfigSchema] = {}

    def register(self, name: str, schema: Dict[str, Any]) -> None:
        """注册模式"""
        self._schemas[name] = ConfigSchema(schema)

    def get(self, name: str) -> Optional[ConfigSchema]:
        """获取模式"""
        return self._schemas.get(name)

    def validate(self, name: str, config: Dict[str, Any]) -> bool:
        """验证配置"""
        schema = self.get(name)
        if schema is None:
            return False
        return schema.validate(config)

    def list_schemas(self) -> List[str]:
        """列出所有模式名称"""
        return list(self._schemas.keys())


class ConfigType(Enum):

    """配置类型枚举"""
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


class ConfigConstraint:

    """配置约束"""

    def __init__(self, constraint_type: str, value: Any):

        self.type = constraint_type
        self.value = value

    def validate(self, data: Any) -> bool:
        """验证约束"""
        if self.type == "min":
            return data >= self.value
        elif self.type == "max":
            return data <= self.value
        elif self.type == "pattern":
            return bool(re.match(self.value, str(data)))
        elif self.type == "enum":
            return data in self.value
        return True


# 预定义的配置模式
DEFAULT_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "database": {
            "type": "object",
            "properties": {
                "host": {"type": "string"},
                "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                "name": {"type": "string"},
                "user": {"type": "string"},
                "password": {"type": "string"}
            },
            "required": ["host", "port", "name"]
        },
        "logging": {
            "type": "object",
            "properties": {
                "level": {"type": "string", "pattern": "^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"},
                "file": {"type": "string"},
                "max_size": {"type": "integer", "minimum": 1},
                "backup_count": {"type": "integer", "minimum": 0}
            },
            "required": ["level"]
        },
        "cache": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "max_size": {"type": "integer", "minimum": 1},
                "ttl": {"type": "integer", "minimum": 1}
            },
            "required": ["enabled"]
        }
    },
    "required": ["database", "logging"]
}


def create_default_schema_registry() -> ConfigSchemaRegistry:
    """创建默认的模式注册表"""
    registry = ConfigSchemaRegistry()
    registry.register("default", DEFAULT_CONFIG_SCHEMA)
    return registry





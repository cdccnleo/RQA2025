"""配置验证模式管理"""

from typing import Dict, Any

class ConfigSchemaManager:
    """配置模式管理器"""

    def __init__(self, schema_path: str = None):
        """初始化模式管理器

        Args:
            schema_path: 外部模式文件路径
        """
        self._schema_path = schema_path
        self._builtin_schemas = {
            'default': self._get_default_schema(),
            'database': self._get_database_schema(),
            'cache': self._get_cache_schema()
        }

    def get_schema(self, schema_type: str = 'default') -> Dict[str, Any]:
        """获取指定类型的模式

        Args:
            schema_type: 模式类型(default/database/cache)

        Returns:
            配置模式字典

        Raises:
            SchemaError: 当模式不存在时抛出
        """
        if schema_type not in self._builtin_schemas:
            raise SchemaError(f"未知的模式类型: {schema_type}")

        return self._builtin_schemas[schema_type]

    def _get_default_schema(self) -> Dict[str, Any]:
        """获取默认模式"""
        return {
            'type': 'object',
            'properties': {
                'env': {'type': 'string', 'required': True},
                'version': {'type': 'string'}
            }
        }

    def _get_database_schema(self) -> Dict[str, Any]:
        """获取数据库配置模式"""
        return {
            'host': {'type': 'string', 'required': True},
            'port': {'type': 'number', 'min': 1024, 'max': 65535},
            'username': {'type': 'string', 'required': True},
            'password': {'type': 'string', 'required': True}
        }

    def _get_cache_schema(self) -> Dict[str, Any]:
        """获取缓存配置模式"""
        return {
            'l1_cache': {
                'max_size': {'type': 'number', 'min': 1},
                'expire_after': {'type': 'number', 'min': 0}
            },
            'l2_cache': {
                'ttl': {'type': 'number', 'min': 0},
                'max_file_size': {'type': 'string'}
            }
        }

class SchemaError(Exception):
    """模式相关错误"""
    pass

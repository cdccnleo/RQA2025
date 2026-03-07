
from typing import Dict, Any
"""配置验证模式管理"""


class SimpleConfigSchemaManager:

    """简单配置模式管理器"""

    def __init__(self, schema_path: str = None):
        """初始化模式管理器"

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
        """获取指定类型的模式"

        Args:
            schema_type: 模式类型(default / database / cache)

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


class CacheConfigSchema:
    """缓存配置模式验证器"""

    def __init__(self):
        """初始化缓存配置模式验证器"""
        self.schema = {
            'type': 'object',
            'properties': {
                'cache_type': {
                    'type': 'string',
                    'enum': ['memory', 'redis', 'file', 'hybrid'],
                    'default': 'memory'
                },
                'max_size': {
                    'type': 'integer',
                    'minimum': 1,
                    'maximum': 1000000,
                    'default': 1000
                },
                'ttl': {
                    'type': 'integer',
                    'minimum': 0,
                    'maximum': 86400,  # 24小时
                    'default': 3600
                },
                'redis_config': {
                    'type': 'object',
                    'properties': {
                        'host': {'type': 'string', 'default': 'localhost'},
                        'port': {'type': 'integer', 'minimum': 1, 'maximum': 65535, 'default': 6379},
                        'db': {'type': 'integer', 'minimum': 0, 'maximum': 15, 'default': 0},
                        'password': {'type': 'string'},
                        'max_connections': {'type': 'integer', 'minimum': 1, 'maximum': 100, 'default': 10}
                    }
                },
                'file_config': {
                    'type': 'object',
                    'properties': {
                        'cache_dir': {'type': 'string', 'default': './cache'},
                        'max_file_size': {'type': 'string', 'default': '100MB'},
                        'compression': {'type': 'boolean', 'default': True}
                    }
                },
                'eviction_policy': {
                    'type': 'string',
                    'enum': ['lru', 'lfu', 'ttl', 'random'],
                    'default': 'lru'
                },
                'monitoring': {
                    'type': 'object',
                    'properties': {
                        'enabled': {'type': 'boolean', 'default': True},
                        'stats_interval': {'type': 'integer', 'minimum': 1, 'maximum': 3600, 'default': 60}
                    }
                }
            },
            'required': ['cache_type']
        }

    def validate(self, config: Dict[str, Any]) -> bool:
        """
        验证缓存配置

        Args:
            config: 要验证的配置字典

        Returns:
            bool: 验证是否通过
        """
        # 基础验证
        if not isinstance(config, dict):
            return False

        # 检查必需字段
        if 'cache_type' not in config:
            return False

        # 检查cache_type值
        if config['cache_type'] not in ['memory', 'redis', 'file', 'hybrid']:
            return False

        # 检查数值范围
        if 'max_size' in config:
            if not isinstance(config['max_size'], int) or config['max_size'] < 1:
                return False

        if 'ttl' in config:
            if not isinstance(config['ttl'], int) or config['ttl'] < 0:
                return False

        return True

    def validate_config(self, config: Dict[str, Any]) -> tuple:
        """
        验证缓存配置并返回结果和错误信息

        Args:
            config: 要验证的配置字典

        Returns:
            tuple: (is_valid, errors) - 验证结果和错误列表
        """
        errors = []

        # 基础验证
        if not isinstance(config, dict):
            errors.append("配置必须是字典类型")
            return False, errors

        # 检查必需字段
        if 'cache_type' not in config:
            errors.append("缺少必需字段: cache_type")
            return False, errors

        # 检查cache_type值
        if config['cache_type'] not in ['memory', 'redis', 'file', 'hybrid', 'multi_level']:
            errors.append(f"无效的cache_type值: {config['cache_type']}")
            return False, errors

        # 检查数值范围
        if 'max_size' in config:
            if not isinstance(config['max_size'], int) or config['max_size'] < 1:
                errors.append("max_size必须是大于0的整数")

        if 'ttl' in config:
            if not isinstance(config['ttl'], int) or config['ttl'] < 0:
                errors.append("ttl必须是非负整数")

        return len(errors) == 0, errors

    def get_default_config(self, cache_type: str = 'memory') -> Dict[str, Any]:
        """
        获取默认配置

        Args:
            cache_type: 缓存类型

        Returns:
            Dict[str, Any]: 默认配置字典
        """
        defaults = {
            'memory': {
                'cache_type': 'memory',
                'max_size': 1000,
                'ttl': 3600,
                'eviction_policy': 'lru',
                'monitoring': {'enabled': True, 'stats_interval': 60}
            },
            'redis': {
                'cache_type': 'redis',
                'max_size': 10000,
                'ttl': 3600,
                'redis_config': {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0,
                    'max_connections': 10
                },
                'monitoring': {'enabled': True, 'stats_interval': 60}
            },
            'file': {
                'cache_type': 'file',
                'max_size': 5000,
                'ttl': 7200,
                'file_config': {
                    'cache_dir': './cache',
                    'max_file_size': '100MB',
                    'compression': True
                },
                'monitoring': {'enabled': True, 'stats_interval': 60}
            }
        }

        return defaults.get(cache_type, defaults['memory'])


class SchemaError(Exception):

    """模式相关错误"""

"""
配置验证系统使用示例

本模块展示了如何使用配置验证系统的各个组件。
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import json

from .schema import (
    ConfigType,
    ConfigSchema,
    ConfigConstraint,
    ConfigDependency,
    ConfigSchemaRegistry,
    ConfigValidator
)
from .typed_config import TypedConfigBase, config_value, get_typed_config
from .validator_factory import (
    ValidationStrategy,
    ConfigValidatorFactory
)

# 1. 定义配置模式
class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str
    port: int
    username: str
    password: str
    database: str
    max_connections: int = 10
    timeout: float = 30.0

# 2. 创建类型安全的配置类
class AppConfig(TypedConfigBase):
    """应用配置类"""
    # 服务器配置
    server_host: str = config_value("server.host", "localhost")
    server_port: int = config_value("server.port", 8080)
    debug_mode: bool = config_value("debug", False)

    # 日志配置
    log_level: LogLevel = config_value("log.level", LogLevel.INFO)
    log_file: Optional[str] = config_value("log.file", None)

    # 数据库配置
    database: DatabaseConfig = config_value("database", None)

    # 特性开关
    enable_cache: bool = config_value("features.cache", True)
    enable_metrics: bool = config_value("features.metrics", False)

    # 性能配置
    max_workers: int = config_value("performance.max_workers", 4)
    timeout: float = config_value("performance.timeout", 30.0)

def create_example_schema_registry() -> ConfigSchemaRegistry:
    """创建示例配置模式注册表"""
    registry = ConfigSchemaRegistry()

    # 注册服务器配置模式
    registry.register(ConfigSchema(
        key="server.host",
        type=ConfigType.STRING,
        description="服务器主机名",
        default="localhost",
        constraints=ConfigConstraint(
            pattern=r"^[a-zA-Z0-9\-\.]+$"
        )
    ))

    registry.register(ConfigSchema(
        key="server.port",
        type=ConfigType.INTEGER,
        description="服务器端口",
        default=8080,
        constraints=ConfigConstraint(
            min_value=1024,
            max_value=65535
        )
    ))

    # 注册日志配置模式
    registry.register(ConfigSchema(
        key="log.level",
        type=ConfigType.STRING,
        description="日志级别",
        default="info",
        constraints=ConfigConstraint(
            enum_values=["debug", "info", "warning", "error"]
        )
    ))

    registry.register(ConfigSchema(
        key="log.file",
        type=ConfigType.STRING,
        description="日志文件路径",
        required=False,
        constraints=ConfigConstraint(
            pattern=r"^[a-zA-Z0-9\-_/\\]+\.log$"
        )
    ))

    # 注册数据库配置模式
    registry.register(ConfigSchema(
        key="database.host",
        type=ConfigType.STRING,
        description="数据库主机名",
        constraints=ConfigConstraint(
            pattern=r"^[a-zA-Z0-9\-\.]+$"
        )
    ))

    registry.register(ConfigSchema(
        key="database.port",
        type=ConfigType.INTEGER,
        description="数据库端口",
        default=5432,
        constraints=ConfigConstraint(
            min_value=1024,
            max_value=65535
        )
    ))

    # 注册特性开关配置模式
    registry.register(ConfigSchema(
        key="features.cache",
        type=ConfigType.BOOLEAN,
        description="是否启用缓存",
        default=True
    ))

    registry.register(ConfigSchema(
        key="features.metrics",
        type=ConfigType.BOOLEAN,
        description="是否启用指标收集",
        default=False,
        dependencies=[
            ConfigDependency(
                key="features.cache",
                condition="dep_value == True",
                required=True
            )
        ]
    ))

    return registry

def example_usage():
    """配置验证系统使用示例"""
    # 1. 创建配置验证器
    registry = create_example_schema_registry()
    validator = ConfigValidatorFactory.create_validator(
        ValidationStrategy.ALL,
        schema_registry=registry,
        type_specs={
            "server.port": int,
            "log.level": str,
            "database.max_connections": int
        },
        range_specs={
            "server.port": {"min": 1024, "max": 65535},
            "database.max_connections": {"min": 1, "max": 100},
            "performance.timeout": {"min": 0.1, "max": 300.0}
        },
        dependency_specs={
            "features.metrics": [
                {"key": "features.cache", "required": True}
            ]
        }
    )

    # 2. 示例配置
    config = {
        "server": {
            "host": "localhost",
            "port": 8080
        },
        "log": {
            "level": "info",
            "file": "app.log"
        },
        "database": {
            "host": "db.example.com",
            "port": 5432,
            "username": "admin",
            "password": "secret",
            "database": "myapp",
            "max_connections": 20
        },
        "features": {
            "cache": True,
            "metrics": True
        },
        "performance": {
            "max_workers": 8,
            "timeout": 60.0
        }
    }

    # 3. 验证配置
    errors = validator.validate(config)
    if errors:
        print("配置验证失败:")
        for error in errors:
            print(f"- {error}")
    else:
        print("配置验证通过")

    # 4. 使用类型安全的配置访问
    class ConfigManager:
        def __init__(self, config_dict):
            self._config = config_dict

        def get(self, key):
            parts = key.split('.')
            value = self._config
            for part in parts:
                value = value[part]
            return value

    config_manager = ConfigManager(config)
    app_config = get_typed_config(AppConfig, config_manager)

    # 类型安全的配置访问示例
    print(f"\n类型安全的配置访问示例:")
    print(f"服务器地址: {app_config.server_host}:{app_config.server_port}")
    print(f"日志级别: {app_config.log_level}")
    print(f"数据库配置: {app_config.database}")
    print(f"缓存启用: {app_config.enable_cache}")
    print(f"最大工作线程: {app_config.max_workers}")

if __name__ == "__main__":
    example_usage()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理测试
Configuration Management Tests

测试配置文件的完整性和正确性，包括：
1. 配置文件格式验证
2. 配置模板渲染验证
3. 配置热更新验证
4. 配置版本控制验证
5. 环境特定配置验证
6. 配置安全验证
7. 配置迁移验证
8. 配置监控和告警验证
"""

import pytest
import os
import tempfile
import shutil
import json
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
import sys
import hashlib
import time
from datetime import datetime

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class TestConfigurationFileValidation:
    """测试配置文件格式验证"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_validator = Mock()

    def teardown_method(self):
        """测试后清理"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_yaml_configuration_parsing(self):
        """测试YAML配置文件解析"""
        # 创建测试YAML配置
        yaml_config = """
# RQA2025 应用配置
application:
  name: "RQA2025"
  version: "1.0.0"
  environment: "production"
  debug: false

database:
  host: "localhost"
  port: 5432
  name: "rqa2025"
  user: "${DB_USER}"
  password: "${DB_PASSWORD}"
  ssl_mode: "require"
  connection_pool:
    min_size: 5
    max_size: 20
    timeout: 30

redis:
  host: "${REDIS_HOST}"
  port: ${REDIS_PORT}
  db: 0
  password: "${REDIS_PASSWORD}"
  connection_timeout: 5

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    - type: "console"
      level: "INFO"
    - type: "file"
      level: "WARNING"
      filename: "/var/log/rqa2025/app.log"
      max_bytes: 10485760
      backup_count: 5

features:
  enable_caching: true
  enable_monitoring: true
  enable_metrics: true
  experimental_features: false

security:
  secret_key: "${SECRET_KEY}"
  jwt_expiration: 3600
  allowed_hosts:
    - "api.rqa2025.com"
    - "app.rqa2025.com"
"""

        config_path = Path(self.temp_dir) / "config.yaml"
        with open(config_path, 'w') as f:
            f.write(yaml_config)

        # 解析YAML配置
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            assert config is not None, "YAML配置解析失败"
            assert isinstance(config, dict), "配置应该是字典类型"

        except yaml.YAMLError as e:
            pytest.fail(f"YAML配置解析错误: {e}")

        # 验证配置结构
        required_sections = ['application', 'database', 'redis', 'logging']
        for section in required_sections:
            assert section in config, f"缺少必需的配置节: {section}"

        # 验证应用配置
        app_config = config['application']
        assert app_config['name'] == 'RQA2025', "应用名称不正确"
        assert app_config['environment'] == 'production', "环境配置不正确"

        # 验证数据库配置
        db_config = config['database']
        assert db_config['port'] == 5432, "数据库端口不正确"
        assert db_config['ssl_mode'] == 'require', "SSL模式不正确"

        # 验证环境变量占位符
        assert '${DB_USER}' in str(db_config['user']), "应该包含数据库用户环境变量"
        assert '${REDIS_HOST}' in str(config['redis']['host']), "应该包含Redis主机环境变量"

    def test_json_configuration_parsing(self):
        """测试JSON配置文件解析"""
        # 创建测试JSON配置
        json_config = {
            "application": {
                "name": "RQA2025",
                "version": "1.0.0",
                "environment": "staging"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "cors_origins": [
                    "https://app.rqa2025.com",
                    "https://admin.rqa2025.com"
                ],
                "rate_limits": {
                    "requests_per_minute": 100,
                    "burst_limit": 20
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics_port": 8001,
                "health_check_interval": 30,
                "alerts": {
                    "email_enabled": True,
                    "slack_enabled": False,
                    "thresholds": {
                        "cpu_usage": 80.0,
                        "memory_usage": 85.0,
                        "disk_usage": 90.0
                    }
                }
            }
        }

        config_path = Path(self.temp_dir) / "api_config.json"
        with open(config_path, 'w') as f:
            json.dump(json_config, f, indent=2)

        # 解析JSON配置
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)

            assert config is not None, "JSON配置解析失败"
            assert isinstance(config, dict), "配置应该是字典类型"

        except json.JSONDecodeError as e:
            pytest.fail(f"JSON配置解析错误: {e}")

        # 验证配置结构
        assert 'application' in config, "缺少应用配置"
        assert 'api' in config, "缺少API配置"
        assert 'monitoring' in config, "缺少监控配置"

        # 验证API配置
        api_config = config['api']
        assert api_config['port'] == 8000, "API端口不正确"
        assert len(api_config['cors_origins']) == 2, "CORS源数量不正确"

        # 验证监控配置
        monitoring_config = config['monitoring']
        assert monitoring_config['enabled'] is True, "监控应该启用"
        assert monitoring_config['alerts']['thresholds']['cpu_usage'] == 80.0, "CPU阈值不正确"

    def test_configuration_schema_validation(self):
        """测试配置模式验证"""
        # 定义配置模式
        config_schema = {
            "application": {
                "name": {"type": "string", "required": True},
                "version": {"type": "string", "required": True, "pattern": r"^\d+\.\d+\.\d+$"},
                "environment": {"type": "string", "required": True, "enum": ["development", "staging", "production"]},
                "debug": {"type": "boolean", "default": False}
            },
            "database": {
                "host": {"type": "string", "required": True},
                "port": {"type": "integer", "required": True, "min": 1, "max": 65535},
                "name": {"type": "string", "required": True},
                "ssl_mode": {"type": "string", "enum": ["disable", "require", "prefer"], "default": "prefer"}
            },
            "api": {
                "host": {"type": "string", "required": True},
                "port": {"type": "integer", "required": True, "min": 1, "max": 65535},
                "rate_limits": {
                    "requests_per_minute": {"type": "integer", "min": 1, "max": 10000},
                    "burst_limit": {"type": "integer", "min": 1}
                }
            }
        }

        def validate_config_against_schema(config: Dict, schema: Dict) -> List[str]:
            """根据模式验证配置"""
            errors = []

            def validate_value(key_path: str, value: Any, rules: Dict) -> List[str]:
                """验证单个值"""
                field_errors = []

                # 检查必需字段
                if rules.get("required", False) and value is None:
                    field_errors.append(f"{key_path} 是必需字段")

                if value is None:
                    return field_errors

                # 类型检查
                expected_type = rules["type"]
                if expected_type == "string" and not isinstance(value, str):
                    field_errors.append(f"{key_path} 应该是字符串类型")
                elif expected_type == "integer" and not isinstance(value, int):
                    field_errors.append(f"{key_path} 应该是整数类型")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    field_errors.append(f"{key_path} 应该是布尔类型")

                # 范围检查
                if "min" in rules and isinstance(value, (int, float)) and value < rules["min"]:
                    field_errors.append(f"{key_path} 不能小于 {rules['min']}")
                if "max" in rules and isinstance(value, (int, float)) and value > rules["max"]:
                    field_errors.append(f"{key_path} 不能大于 {rules['max']}")

                # 枚举检查
                if "enum" in rules and value not in rules["enum"]:
                    field_errors.append(f"{key_path} 必须是以下值之一: {rules['enum']}")

                # 模式检查
                if "pattern" in rules and isinstance(value, str):
                    import re
                    if not re.match(rules["pattern"], value):
                        field_errors.append(f"{key_path} 格式不符合要求")

                return field_errors

            def validate_section(section_name: str, section_config: Dict, section_schema: Dict):
                """验证配置节"""
                for field_name, rules in section_schema.items():
                    key_path = f"{section_name}.{field_name}"
                    value = section_config.get(field_name)

                    # 处理嵌套对象
                    if isinstance(rules, dict) and "type" not in rules:
                        if isinstance(value, dict):
                            validate_section(key_path, value, rules)
                        continue

                    # 处理默认值
                    if value is None and "default" in rules:
                        value = rules["default"]

                    field_errors = validate_value(key_path, value, rules)
                    errors.extend(field_errors)

            # 验证顶级配置节
            for section_name, section_schema in schema.items():
                if section_name in config:
                    section_config = config[section_name]
                    validate_section(section_name, section_config, section_schema)
                else:
                    # 检查是否有必需的顶级配置节
                    has_required_fields = any(
                        isinstance(rules, dict) and rules.get("required", False)
                        for rules in section_schema.values()
                    )
                    if has_required_fields:
                        errors.append(f"缺少必需的配置节: {section_name}")

            return errors

        # 测试有效配置
        valid_config = {
            "application": {
                "name": "RQA2025",
                "version": "1.0.0",
                "environment": "production",
                "debug": False
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "rqa2025",
                "ssl_mode": "require"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "rate_limits": {
                    "requests_per_minute": 100,
                    "burst_limit": 20
                }
            }
        }

        errors = validate_config_against_schema(valid_config, config_schema)
        assert len(errors) == 0, f"有效配置验证失败: {errors}"

        # 测试无效配置
        invalid_config = {
            "application": {
                "name": "RQA2025",
                "version": "invalid_version",
                "environment": "invalid_env"
            },
            "database": {
                "host": "localhost",
                "port": 99999,  # 无效端口
                "name": "rqa2025"
            }
        }

        errors = validate_config_against_schema(invalid_config, config_schema)
        assert len(errors) > 0, "应该检测到无效配置的错误"

        # 检查具体的错误
        error_messages = ' '.join(errors)
        assert 'invalid_version' in error_messages or 'pattern' in error_messages, "应该检测到版本格式错误"
        assert 'invalid_env' in error_messages or 'enum' in error_messages, "应该检测到无效环境"
        assert '99999' in error_messages or 'max' in error_messages, "应该检测到无效端口"


class TestConfigurationTemplateRendering:
    """测试配置模板渲染验证"""

    def setup_method(self):
        """测试前准备"""
        self.template_engine = Mock()

    def test_environment_variable_substitution(self):
        """测试环境变量替换"""
        # 创建配置模板
        template_config = {
            "database": {
                "host": "${DB_HOST}",
                "port": "${DB_PORT}",
                "user": "${DB_USER}",
                "password": "${DB_PASSWORD}"
            },
            "redis": {
                "host": "${REDIS_HOST}",
                "port": "${REDIS_PORT}",
                "password": "${REDIS_PASSWORD}"
            },
            "api": {
                "secret_key": "${API_SECRET_KEY}",
                "allowed_hosts": "${ALLOWED_HOSTS}"
            }
        }

        # 设置环境变量
        test_env_vars = {
            "DB_HOST": "prod-db.example.com",
            "DB_PORT": "5432",
            "DB_USER": "rqa2025_user",
            "DB_PASSWORD": "secure_password_123",
            "REDIS_HOST": "redis-cluster.example.com",
            "REDIS_PORT": "6379",
            "REDIS_PASSWORD": "redis_password_456",
            "API_SECRET_KEY": "api_secret_789",
            "ALLOWED_HOSTS": "api.rqa2025.com,app.rqa2025.com"
        }

        def render_template_with_env_vars(template: Dict, env_vars: Dict) -> Dict:
            """使用环境变量渲染模板"""
            import re

            def render_value(value):
                if isinstance(value, str):
                    # 替换环境变量占位符
                    def replace_var(match):
                        var_name = match.group(1)
                        return env_vars.get(var_name, match.group(0))

                    rendered = re.sub(r'\$\{([^}]+)\}', replace_var, value)
                    return rendered
                elif isinstance(value, dict):
                    return {k: render_value(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [render_value(item) for item in value]
                else:
                    return value

            return render_value(template)

        # 渲染配置模板
        rendered_config = render_template_with_env_vars(template_config, test_env_vars)

        # 验证渲染结果
        assert rendered_config['database']['host'] == 'prod-db.example.com'
        assert rendered_config['database']['port'] == '5432'
        assert rendered_config['redis']['host'] == 'redis-cluster.example.com'
        assert rendered_config['api']['secret_key'] == 'api_secret_789'

        # 验证所有占位符都被替换
        def check_no_placeholders(obj):
            """检查是否还有未替换的占位符"""
            if isinstance(obj, str):
                assert '${' not in obj, f"未替换的占位符: {obj}"
            elif isinstance(obj, dict):
                for value in obj.values():
                    check_no_placeholders(value)
            elif isinstance(obj, list):
                for item in obj.values():
                    check_no_placeholders(item)

        check_no_placeholders(rendered_config)

    def test_conditional_configuration_rendering(self):
        """测试条件配置渲染"""
        # 创建条件配置模板
        conditional_template = {
            "features": {
                "caching": {
                    "enabled": "${ENABLE_CACHING:true}",
                    "redis_host": "${REDIS_HOST}",
                    "ttl": "${CACHE_TTL:3600}"
                },
                "monitoring": {
                    "enabled": "${ENABLE_MONITORING:false}",
                    "metrics_port": "${METRICS_PORT:8001}"
                }
            },
            "environment_specific": {
                "development": {
                    "debug": True,
                    "log_level": "DEBUG"
                },
                "production": {
                    "debug": False,
                    "log_level": "WARNING"
                }
            }
        }

        def render_conditional_config(template: Dict, env_vars: Dict, environment: str) -> Dict:
            """渲染条件配置"""
            import re

            def render_value(value):
                if isinstance(value, str):
                    # 处理带默认值的环境变量
                    def replace_var(match):
                        var_expr = match.group(1)
                        if ':' in var_expr:
                            var_name, default_value = var_expr.split(':', 1)
                            return env_vars.get(var_name, default_value)
                        else:
                            return env_vars.get(var_expr, match.group(0))

                    rendered = re.sub(r'\$\{([^}]+)\}', replace_var, value)

                    # 处理布尔值转换
                    if rendered.lower() in ('true', 'false'):
                        return rendered.lower() == 'true'

                    # 处理数字转换
                    try:
                        return int(rendered)
                    except ValueError:
                        pass

                    return rendered

                elif isinstance(value, dict):
                    return {k: render_value(v) for k, v in value.items()}
                elif isinstance(value, list):
                    return [render_value(item) for item in value]
                else:
                    return value

            # 渲染配置
            rendered = render_value(template)

            # 应用环境特定配置
            if 'environment_specific' in rendered and environment in rendered['environment_specific']:
                env_config = rendered['environment_specific'][environment]
                # 在实际应用中，这里会合并环境配置
                rendered['environment'] = env_config

            return rendered

        # 测试开发环境配置
        dev_env_vars = {
            "ENABLE_CACHING": "true",
            "REDIS_HOST": "localhost",
            "ENABLE_MONITORING": "false"
        }

        dev_config = render_conditional_config(conditional_template, dev_env_vars, "development")

        assert dev_config['features']['caching']['enabled'] is True
        assert dev_config['features']['monitoring']['enabled'] is False
        assert dev_config['environment']['debug'] is True

        # 测试生产环境配置（使用默认值）
        prod_env_vars = {}

        prod_config = render_conditional_config(conditional_template, prod_env_vars, "production")

        assert prod_config['features']['caching']['enabled'] is True  # 默认值
        assert prod_config['features']['monitoring']['enabled'] is False  # 默认值
        assert prod_config['features']['caching']['ttl'] == 3600  # 默认值
        assert prod_config['environment']['debug'] is False

    def test_configuration_inheritance_and_overrides(self):
        """测试配置继承和覆盖"""
        # 定义基础配置
        base_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "pool_size": 10
            },
            "logging": {
                "level": "INFO",
                "handlers": ["console"]
            },
            "features": {
                "caching": True,
                "monitoring": True
            }
        }

        # 定义环境特定的覆盖配置
        environment_overrides = {
            "development": {
                "database": {
                    "pool_size": 5
                },
                "logging": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"]
                },
                "features": {
                    "debug_mode": True
                }
            },
            "production": {
                "database": {
                    "host": "prod-db.example.com",
                    "pool_size": 50
                },
                "logging": {
                    "handlers": ["file", "syslog"]
                },
                "features": {
                    "caching": True,
                    "monitoring": True,
                    "metrics": True
                }
            }
        }

        def merge_configs(base: Dict, overrides: Dict) -> Dict:
            """深度合并配置"""
            result = base.copy()

            def deep_merge(target, source):
                for key, value in source.items():
                    if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                        deep_merge(target[key], value)
                    else:
                        target[key] = value

            deep_merge(result, overrides)
            return result

        # 测试开发环境配置合并
        dev_config = merge_configs(base_config, environment_overrides["development"])

        assert dev_config['database']['host'] == 'localhost'  # 保留基础配置
        assert dev_config['database']['pool_size'] == 5  # 被覆盖
        assert dev_config['logging']['level'] == 'DEBUG'  # 被覆盖
        assert len(dev_config['logging']['handlers']) == 2  # 被扩展
        assert dev_config['features']['debug_mode'] is True  # 新增配置

        # 测试生产环境配置合并
        prod_config = merge_configs(base_config, environment_overrides["production"])

        assert prod_config['database']['host'] == 'prod-db.example.com'  # 被覆盖
        assert prod_config['database']['pool_size'] == 50  # 被覆盖
        assert prod_config['logging']['level'] == 'INFO'  # 保留基础配置
        assert 'syslog' in prod_config['logging']['handlers']  # 被扩展
        assert prod_config['features']['metrics'] is True  # 新增配置


class TestConfigurationHotUpdate:
    """测试配置热更新验证"""

    def setup_method(self):
        """测试前准备"""
        self.config_manager = Mock()

    def test_configuration_reload_mechanism(self):
        """测试配置重载机制"""
        # 模拟配置管理器
        class ConfigManager:
            def __init__(self):
                self.config = {}
                self.version = 0
                self.last_reload = None

            def load_config(self, config_data: Dict):
                """加载配置"""
                self.config = config_data.copy()
                self.version += 1
                self.last_reload = datetime.now()
                return self.version

            def get_config(self, key: str = None):
                """获取配置"""
                if key:
                    return self.config.get(key)
                return self.config

            def reload_config(self, new_config: Dict):
                """重载配置"""
                old_config = self.config.copy()
                new_version = self.load_config(new_config)

                # 验证配置变更
                changes = self._detect_changes(old_config, new_config)
                return new_version, changes

            def _detect_changes(self, old_config: Dict, new_config: Dict) -> List[Dict]:
                """检测配置变更"""
                changes = []

                def detect_changes_recursive(old_dict, new_dict, path=""):
                    for key in set(old_dict.keys()) | set(new_dict.keys()):
                        current_path = f"{path}.{key}" if path else key

                        if key not in old_dict:
                            changes.append({
                                'type': 'added',
                                'path': current_path,
                                'new_value': new_dict[key]
                            })
                        elif key not in new_dict:
                            changes.append({
                                'type': 'removed',
                                'path': current_path,
                                'old_value': old_dict[key]
                            })
                        elif isinstance(old_dict[key], dict) and isinstance(new_dict[key], dict):
                            detect_changes_recursive(old_dict[key], new_dict[key], current_path)
                        elif old_dict[key] != new_dict[key]:
                            changes.append({
                                'type': 'modified',
                                'path': current_path,
                                'old_value': old_dict[key],
                                'new_value': new_dict[key]
                            })

                detect_changes_recursive(old_config, new_config)
                return changes

        # 测试配置管理器
        manager = ConfigManager()

        # 初始配置
        initial_config = {
            "database": {"host": "localhost", "port": 5432},
            "redis": {"host": "localhost", "port": 6379}
        }

        version1 = manager.load_config(initial_config)
        assert version1 == 1
        assert manager.get_config('database')['host'] == 'localhost'

        # 更新配置
        updated_config = {
            "database": {"host": "prod-db.example.com", "port": 5432},
            "redis": {"host": "localhost", "port": 6379},
            "monitoring": {"enabled": True}
        }

        version2, changes = manager.reload_config(updated_config)

        # 验证版本更新
        assert version2 == 2
        assert version2 > version1

        # 验证配置更新
        assert manager.get_config('database')['host'] == 'prod-db.example.com'
        assert manager.get_config('monitoring')['enabled'] is True

        # 验证变更检测
        assert len(changes) == 2  # 一个修改，一个新增

        # 检查具体的变更
        change_types = {change['type'] for change in changes}
        assert 'modified' in change_types
        assert 'added' in change_types

        # 检查数据库主机变更
        db_host_change = next((c for c in changes if c.get('path') == 'database.host'), None)
        assert db_host_change is not None
        assert db_host_change['type'] == 'modified'
        assert db_host_change['old_value'] == 'localhost'
        assert db_host_change['new_value'] == 'prod-db.example.com'

    def test_configuration_validation_during_update(self):
        """测试更新期间的配置验证"""
        # 定义配置验证规则
        validation_rules = {
            "database.port": {"type": "integer", "min": 1, "max": 65535},
            "database.pool_size": {"type": "integer", "min": 1, "max": 100},
            "redis.port": {"type": "integer", "min": 1, "max": 65535},
            "api.port": {"type": "integer", "min": 1, "max": 65535}
        }

        def validate_config_update(current_config: Dict, new_config: Dict, rules: Dict) -> List[str]:
            """验证配置更新"""
            errors = []

            def validate_value(path: str, value: Any, rules_dict: Dict) -> List[str]:
                """验证单个配置值"""
                field_errors = []

                if path in rules_dict:
                    rule = rules_dict[path]

                    # 类型检查
                    if rule['type'] == 'integer' and not isinstance(value, int):
                        field_errors.append(f"{path} 必须是整数")
                    elif isinstance(value, int):
                        if value < rule.get('min', float('-inf')):
                            field_errors.append(f"{path} 不能小于 {rule['min']}")
                        if value > rule.get('max', float('inf')):
                            field_errors.append(f"{path} 不能大于 {rule['max']}")

                return field_errors

            def validate_recursive(config_dict: Dict, prefix: str = ""):
                """递归验证配置"""
                for key, value in config_dict.items():
                    current_path = f"{prefix}.{key}" if prefix else key

                    if isinstance(value, dict):
                        validate_recursive(value, current_path)
                    else:
                        field_errors = validate_value(current_path, value, rules)
                        errors.extend(field_errors)

            # 验证新配置
            validate_recursive(new_config)

            return errors

        # 测试有效配置更新
        valid_new_config = {
            "database": {"port": 5432, "pool_size": 20},
            "redis": {"port": 6379},
            "api": {"port": 8000}
        }

        errors = validate_config_update({}, valid_new_config, validation_rules)
        assert len(errors) == 0, f"有效配置应该没有错误: {errors}"

        # 测试无效配置更新
        invalid_new_config = {
            "database": {"port": 99999, "pool_size": 0},  # 无效端口和池大小
            "redis": {"port": "invalid"},  # 无效端口类型
            "api": {"port": 8000}
        }

        errors = validate_config_update({}, invalid_new_config, validation_rules)

        # 应该检测到多个错误
        assert len(errors) >= 3, f"应该检测到无效配置的错误: {errors}"

        # 检查具体的错误
        error_text = ' '.join(errors)
        assert '99999' in error_text or 'max' in error_text, "应该检测到无效端口"
        assert '0' in error_text or 'min' in error_text, "应该检测到无效池大小"
        assert 'invalid' in error_text or 'integer' in error_text, "应该检测到无效类型"


class TestConfigurationVersionControl:
    """测试配置版本控制验证"""

    def setup_method(self):
        """测试前准备"""
        self.version_manager = Mock()

    def test_configuration_versioning(self):
        """测试配置版本控制"""
        # 模拟配置版本管理器
        class ConfigVersionManager:
            def __init__(self):
                self.versions = {}
                self.current_version = None

            def create_version(self, config: Dict, author: str, message: str) -> str:
                """创建配置版本"""
                version_id = f"v{len(self.versions) + 1}_{int(time.time())}"
                checksum = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()

                version_info = {
                    'version_id': version_id,
                    'config': config.copy(),
                    'checksum': checksum,
                    'author': author,
                    'message': message,
                    'timestamp': datetime.now(),
                    'parent_version': self.current_version
                }

                self.versions[version_id] = version_info
                self.current_version = version_id

                return version_id

            def get_version(self, version_id: str) -> Optional[Dict]:
                """获取指定版本的配置"""
                return self.versions.get(version_id)

            def list_versions(self) -> List[Dict]:
                """列出所有版本"""
                return list(self.versions.values())

            def rollback_to_version(self, version_id: str) -> bool:
                """回滚到指定版本"""
                if version_id not in self.versions:
                    return False

                self.current_version = version_id
                return True

            def compare_versions(self, version1: str, version2: str) -> Dict:
                """比较两个版本的差异"""
                if version1 not in self.versions or version2 not in self.versions:
                    return {'error': '版本不存在'}

                config1 = self.versions[version1]['config']
                config2 = self.versions[version2]['config']

                # 简化的差异比较
                differences = {
                    'added': [],
                    'removed': [],
                    'modified': []
                }

                def compare_dicts(dict1, dict2, path=""):
                    for key in set(dict1.keys()) | set(dict2.keys()):
                        current_path = f"{path}.{key}" if path else key

                        if key not in dict1:
                            differences['added'].append(current_path)
                        elif key not in dict2:
                            differences['removed'].append(current_path)
                        elif dict1[key] != dict2[key]:
                            differences['modified'].append({
                                'path': current_path,
                                'old_value': dict1[key],
                                'new_value': dict2[key]
                            })

                compare_dicts(config1, config2)
                return differences

        # 测试版本管理器
        vm = ConfigVersionManager()

        # 创建初始版本
        config_v1 = {
            "database": {"host": "localhost", "port": 5432},
            "redis": {"host": "localhost", "port": 6379}
        }

        v1_id = vm.create_version(config_v1, "admin", "初始配置")
        assert v1_id is not None
        assert vm.current_version == v1_id

        # 创建第二个版本
        config_v2 = config_v1.copy()
        config_v2["database"]["host"] = "prod-db.example.com"
        config_v2["monitoring"] = {"enabled": True}

        v2_id = vm.create_version(config_v2, "ops", "生产环境配置更新")
        assert v2_id != v1_id
        assert vm.current_version == v2_id

        # 验证版本数量
        versions = vm.list_versions()
        assert len(versions) == 2

        # 验证版本信息
        v1_info = vm.get_version(v1_id)
        assert v1_info['author'] == 'admin'
        assert v1_info['message'] == '初始配置'

        v2_info = vm.get_version(v2_id)
        assert v2_info['author'] == 'ops'
        assert v2_info['parent_version'] == v1_id

        # 比较版本差异
        diff = vm.compare_versions(v1_id, v2_id)
        assert len(diff['added']) > 0  # 新增了monitoring
        assert len(diff['modified']) > 0  # 修改了database.host

        # 测试回滚
        assert vm.rollback_to_version(v1_id)
        assert vm.current_version == v1_id

        # 验证无效版本回滚
        assert not vm.rollback_to_version("invalid_version")

    def test_configuration_audit_trail(self):
        """测试配置审计追踪"""
        # 模拟配置审计系统
        audit_log = []

        def log_config_change(action: str, version_id: str, user: str,
                            changes: List[Dict], timestamp: datetime = None):
            """记录配置变更"""
            if timestamp is None:
                timestamp = datetime.now()

            audit_entry = {
                'action': action,
                'version_id': version_id,
                'user': user,
                'changes': changes,
                'timestamp': timestamp,
                'checksum': hashlib.md5(f"{action}{version_id}{user}{timestamp.isoformat()}".encode()).hexdigest()
            }

            audit_log.append(audit_entry)
            return audit_entry

        def get_audit_trail(version_id: str = None, user: str = None,
                          start_time: datetime = None, end_time: datetime = None) -> List[Dict]:
            """获取审计追踪"""
            filtered_log = audit_log

            if version_id:
                filtered_log = [entry for entry in filtered_log if entry['version_id'] == version_id]

            if user:
                filtered_log = [entry for entry in filtered_log if entry['user'] == user]

            if start_time:
                filtered_log = [entry for entry in filtered_log if entry['timestamp'] >= start_time]

            if end_time:
                filtered_log = [entry for entry in filtered_log if entry['timestamp'] <= end_time]

            return filtered_log

        # 记录配置变更
        changes1 = [
            {'type': 'modified', 'path': 'database.host', 'old_value': 'localhost', 'new_value': 'prod-db'}
        ]
        log_config_change('update', 'v1.1', 'admin', changes1)

        changes2 = [
            {'type': 'added', 'path': 'monitoring.enabled', 'new_value': True}
        ]
        log_config_change('update', 'v1.2', 'ops', changes2)

        changes3 = [
            {'type': 'removed', 'path': 'debug_mode'}
        ]
        log_config_change('rollback', 'v1.0', 'admin', changes3)

        # 验证审计日志
        assert len(audit_log) == 3

        # 测试审计查询
        admin_actions = get_audit_trail(user='admin')
        assert len(admin_actions) == 2

        update_actions = get_audit_trail(version_id='v1.1')
        assert len(update_actions) == 1
        assert update_actions[0]['action'] == 'update'

        # 验证审计完整性
        for entry in audit_log:
            required_fields = ['action', 'version_id', 'user', 'changes', 'timestamp', 'checksum']
            for field in required_fields:
                assert field in entry, f"审计条目缺少字段: {field}"

            # 验证校验和（简化验证）
            assert len(entry['checksum']) == 32, "校验和格式不正确"


if __name__ == "__main__":
    pytest.main([__file__])

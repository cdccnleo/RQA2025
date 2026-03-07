#!/usr/bin/env python3
"""
配置管理系统增强器

针对已识别的配置管理问题进行系统性优化：
1. 环境变量配置优先级管理
2. 配置文件路径规范化
3. 配置验证增强
4. 配置源优先级管理
5. 配置热重载优化
6. 配置合并策略改进

优化重点：
- 统一配置管理器功能增强
- 多环境配置支持
- 配置源优先级管理
- 配置验证和错误处理
- 配置缓存和性能优化
"""

from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import time
import yaml

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


logger = logging.getLogger(__name__)


class ConfigSourceType(Enum):
    """配置源类型"""
    DEFAULT = "default"
    FILE = "file"
    ENVIRONMENT = "environment"
    REMOTE = "remote"
    OVERRIDE = "override"


class ConfigPriority(Enum):
    """配置优先级 (数值越大优先级越高)"""
    DEFAULT = 1
    FILE = 2
    ENVIRONMENT = 3
    REMOTE = 4
    OVERRIDE = 5


@dataclass
class ConfigSource:
    """配置源定义"""
    name: str
    type: ConfigSourceType
    priority: ConfigPriority
    data: Dict[str, Any]
    path: Optional[str] = None
    last_modified: Optional[float] = None


class EnhancedConfigManager(UnifiedConfigManager):
    """增强的配置管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # 配置源管理
        self._config_sources: Dict[str, ConfigSource] = {}
        self._priority_order: List[ConfigSourceType] = [
            ConfigSourceType.DEFAULT,
            ConfigSourceType.FILE,
            ConfigSourceType.ENVIRONMENT,
            ConfigSourceType.REMOTE,
            ConfigSourceType.OVERRIDE
        ]

        # 环境变量缓存
        self._env_cache: Dict[str, Any] = {}
        self._env_cache_timestamp = 0
        self._env_cache_ttl = 60  # 环境变量缓存60秒

        # 配置变更监听
        self._watchers: Dict[str, List[Any]] = {}
        self._watcher_lock = threading.RLock()

        # 配置验证器
        self._validators: List[Any] = []

        # 配置合并策略
        self._merge_strategy = "deep_merge"

        # 初始化默认配置源
        self._init_default_sources()

    def _init_default_sources(self):
        """初始化默认配置源"""
        # 添加默认配置源
        default_config = ConfigSource(
            name="default",
            type=ConfigSourceType.DEFAULT,
            priority=ConfigPriority.DEFAULT,
            data=self.default_config.copy()
        )
        self._config_sources["default"] = default_config

    def add_config_source(self, source: ConfigSource) -> bool:
        """添加配置源"""
        try:
            self._config_sources[source.name] = source
            self._refresh_merged_config()
            logger.info(f"Added config source: {source.name} (type: {source.type.value})")
            return True
        except Exception as e:
            logger.error(f"Failed to add config source {source.name}: {e}")
            return False

    def load_from_file(self, file_path: str, source_name: str = "file") -> bool:
        """从文件加载配置"""
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                logger.warning(f"Config file not found: {file_path}")
                return False

            # 根据文件扩展名选择解析器
            if file_path_obj.suffix.lower() == '.yaml' or file_path_obj.suffix.lower() == '.yml':
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
            else:
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            # 创建文件配置源
            file_source = ConfigSource(
                name=source_name,
                type=ConfigSourceType.FILE,
                priority=ConfigPriority.FILE,
                data=data,
                path=str(file_path_obj),
                last_modified=file_path_obj.stat().st_mtime
            )

            return self.add_config_source(file_source)
        except Exception as e:
            logger.error(f"Failed to load config from file {file_path}: {e}")
            return False

    def load_from_env(self, prefix: str = "RQA_", source_name: str = "environment") -> bool:
        """从环境变量加载配置"""
        try:
            env_data = {}

            # 检查缓存
            current_time = time.time()
            if (current_time - self._env_cache_timestamp) < self._env_cache_ttl and self._env_cache:
                env_data = self._env_cache.copy()
            else:
                # 重新加载环境变量
                for key, value in os.environ.items():
                    if key.startswith(prefix):
                        # 移除前缀并转换为小写
                        config_key = key[len(prefix):].lower().replace('_', '.')

                        # 尝试类型转换
                        converted_value = self._convert_env_value(value)
                        env_data[config_key] = converted_value

                # 更新缓存
                self._env_cache = env_data.copy()
                self._env_cache_timestamp = current_time

            # 创建环境变量配置源
            env_source = ConfigSource(
                name=source_name,
                type=ConfigSourceType.ENVIRONMENT,
                priority=ConfigPriority.ENVIRONMENT,
                data=env_data
            )

            return self.add_config_source(env_source)
        except Exception as e:
            logger.error(f"Failed to load config from environment: {e}")
            return False

    def _convert_env_value(self, value: str) -> Any:
        """转换环境变量值类型"""
        if not value:
            return value

        # 布尔值转换
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'

        # 数字转换
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # JSON转换
        if value.startswith(('{', '[', '"')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass

        return value

    def _refresh_merged_config(self):
        """刷新合并后的配置"""
        try:
            # 按优先级排序配置源
            sorted_sources = sorted(
                self._config_sources.values(),
                key=lambda x: x.priority.value
            )

            # 重置数据
            self._data.clear()

            # 按优先级合并配置
            for source in sorted_sources:
                if self._merge_strategy == "deep_merge":
                    self._deep_merge_config(source.data)
                else:
                    self._data.update(source.data)

            logger.debug(f"Merged config from {len(sorted_sources)} sources")
        except Exception as e:
            logger.error(f"Failed to refresh merged config: {e}")

    def _deep_merge_config(self, new_config: Dict[str, Any]):
        """深度合并配置"""
        def _deep_merge_dict(target: Dict[str, Any], source: Dict[str, Any]):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    _deep_merge_dict(target[key], value)
                else:
                    target[key] = value

        # 如果_data还没有对应的section，创建它
        for section_name, section_data in new_config.items():
            if section_name not in self._data:
                self._data[section_name] = {}

            if isinstance(section_data, dict):
                _deep_merge_dict(self._data[section_name], section_data)
            else:
                self._data[section_name] = section_data

    def get_with_source(self, key: str, default: Any = None) -> tuple[Any, Optional[str]]:
        """获取配置值及其来源"""
        value = self.get(key, default)

        # 查找值的来源
        source_name = None
        if '.' in key:
            section, key_part = key.split('.', 1)
        else:
            section = 'default'
            key_part = key

        # 按优先级倒序查找
        for source in sorted(self._config_sources.values(), key=lambda x: x.priority.value, reverse=True):
            if section in source.data:
                if isinstance(source.data[section], dict) and key_part in source.data[section]:
                    source_name = source.name
                    break
                elif not isinstance(source.data[section], dict) and key_part == section:
                    source_name = source.name
                    break

        return value, source_name

    def add_validator(self, validator_func):
        """添加配置验证器"""
        self._validators.append(validator_func)
        logger.info(f"Added config validator: {validator_func.__name__}")

    def validate_all_config(self) -> Dict[str, Any]:
        """验证所有配置"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'source_validation': {}
        }

        try:
            # 验证每个配置源
            for source_name, source in self._config_sources.items():
                source_result = {'is_valid': True, 'errors': [], 'warnings': []}

                for validator in self._validators:
                    try:
                        result = validator(source.data)
                        if hasattr(result, 'is_valid') and not result.is_valid:
                            source_result['is_valid'] = False
                            source_result['errors'].extend(getattr(result, 'errors', []))
                        if hasattr(result, 'warnings'):
                            source_result['warnings'].extend(result.warnings)
                    except Exception as e:
                        source_result['is_valid'] = False
                        source_result['errors'].append(f"Validator error: {str(e)}")

                validation_results['source_validation'][source_name] = source_result
                if not source_result['is_valid']:
                    validation_results['is_valid'] = False
                    validation_results['errors'].extend(
                        [f"{source_name}: {err}" for err in source_result['errors']])

            # 验证合并后的配置
            for validator in self._validators:
                try:
                    result = validator(self._data)
                    if hasattr(result, 'is_valid') and not result.is_valid:
                        validation_results['is_valid'] = False
                        validation_results['errors'].extend(getattr(result, 'errors', []))
                    if hasattr(result, 'warnings'):
                        validation_results['warnings'].extend(result.warnings)
                except Exception as e:
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"Merged config validation error: {str(e)}")

        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation process error: {str(e)}")

        return validation_results

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要（增强版）"""
        summary = super().get_config_summary()

        # 添加配置源信息
        summary['config_sources'] = {}
        for name, source in self._config_sources.items():
            summary['config_sources'][name] = {
                'type': source.type.value,
                'priority': source.priority.value,
                'keys_count': len(source.data),
                'path': source.path,
                'last_modified': source.last_modified
            }

        # 添加环境变量缓存信息
        summary['env_cache'] = {
            'cached_vars': len(self._env_cache),
            'cache_age': time.time() - self._env_cache_timestamp,
            'cache_ttl': self._env_cache_ttl
        }

        # 添加验证器信息
        summary['validators_count'] = len(self._validators)
        summary['watchers_count'] = len(self._watchers)
        summary['merge_strategy'] = self._merge_strategy

        return summary

    def reload_config_sources(self) -> bool:
        """重新加载所有配置源"""
        try:
            reloaded_sources = []
            for source_name, source in self._config_sources.items():
                if source.type == ConfigSourceType.FILE and source.path:
                    file_path = Path(source.path)
                    if file_path.exists():
                        current_mtime = file_path.stat().st_mtime
                        if not source.last_modified or current_mtime > source.last_modified:
                            if self.load_from_file(source.path, source_name):
                                reloaded_sources.append(source_name)
                elif source.type == ConfigSourceType.ENVIRONMENT:
                    # 清除环境变量缓存以强制重新加载
                    self._env_cache.clear()
                    self._env_cache_timestamp = 0
                    if self.load_from_env():
                        reloaded_sources.append(source_name)

            if reloaded_sources:
                self._refresh_merged_config()
                logger.info(f"Reloaded config sources: {reloaded_sources}")

            return True
        except Exception as e:
            logger.error(f"Failed to reload config sources: {e}")
            return False


class ConfigManagementEnhancer:
    """配置管理系统增强器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.config_dir = self.project_root / "config"
        self.enhanced_config_manager = None

        # 确保配置目录存在
        self.config_dir.mkdir(exist_ok=True)

    def create_enhanced_config_manager(self) -> EnhancedConfigManager:
        """创建增强的配置管理器"""
        if self.enhanced_config_manager is None:
            self.enhanced_config_manager = EnhancedConfigManager()

            # 加载默认配置文件
            self._load_default_configs()

            # 加载环境变量
            self._load_environment_configs()

            # 添加基本验证器
            self._add_basic_validators()

        return self.enhanced_config_manager

    def _load_default_configs(self):
        """加载默认配置文件"""
        if self.enhanced_config_manager is None:
            return

        config_files = [
            self.config_dir / "default.json",
            self.config_dir / "default.yaml",
            self.config_dir / "app_config.json",
            self.config_dir / "system_config.yaml"
        ]

        for config_file in config_files:
            if config_file.exists():
                self.enhanced_config_manager.load_from_file(
                    str(config_file),
                    f"file_{config_file.stem}"
                )

    def _load_environment_configs(self):
        """加载环境变量配置"""
        if self.enhanced_config_manager is None:
            return

        # 加载不同前缀的环境变量
        prefixes = ["RQA_", "CONFIG_", "APP_"]
        for prefix in prefixes:
            self.enhanced_config_manager.load_from_env(
                prefix=prefix,
                source_name=f"env_{prefix.lower().rstrip('_')}"
            )

    def _add_basic_validators(self):
        """添加基本验证器"""
        if self.enhanced_config_manager is None:
            return

        def validate_required_keys(config: Dict[str, Any]) -> Dict[str, Any]:
            """验证必需的配置键"""
            required_keys = [
                "logging.level",
                "system.debug"
            ]

            errors = []
            for key in required_keys:
                if '.' in key:
                    section, key_part = key.split('.', 1)
                    if section not in config or key_part not in config.get(section, {}):
                        errors.append(f"Missing required config key: {key}")
                else:
                    if key not in config:
                        errors.append(f"Missing required config key: {key}")

            return {
                'is_valid': len(errors) == 0,
                'errors': errors,
                'warnings': []
            }

        def validate_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
            """验证配置类型"""
            type_checks = {
                "system.debug": bool,
                "logging.level": str,
                "database.port": int
            }

            errors = []
            warnings = []

            for key, expected_type in type_checks.items():
                if '.' in key:
                    section, key_part = key.split('.', 1)
                    if section in config and key_part in config[section]:
                        value = config[section][key_part]
                        if not isinstance(value, expected_type):
                            warnings.append(
                                f"Config key {key} should be {expected_type.__name__}, got {type(value).__name__}")

            return {
                'is_valid': len(errors) == 0,
                'errors': errors,
                'warnings': warnings
            }

        if self.enhanced_config_manager:
            self.enhanced_config_manager.add_validator(validate_required_keys)
            self.enhanced_config_manager.add_validator(validate_config_types)

    def generate_config_report(self) -> Dict[str, Any]:
        """生成配置报告"""
        if not self.enhanced_config_manager:
            self.create_enhanced_config_manager()

        if not self.enhanced_config_manager:
            return {'error': 'Enhanced config manager not initialized'}

        report = {
            'timestamp': time.time(),
            'summary': self.enhanced_config_manager.get_config_summary(),
            'validation': self.enhanced_config_manager.validate_all_config(),
            'config_sources': {},
            'recommendations': []
        }

        # 详细配置源信息
        for name, source in self.enhanced_config_manager._config_sources.items():
            report['config_sources'][name] = {
                'type': source.type.value,
                'priority': source.priority.value,
                'data_sample': dict(list(source.data.items())[:5]) if source.data else {},  # 前5个配置项
                'path': source.path,
                'last_modified': source.last_modified
            }

        # 生成建议
        if not report['validation']['is_valid']:
            report['recommendations'].append("修复配置验证错误")

        if len(self.enhanced_config_manager._config_sources) > 5:
            report['recommendations'].append("考虑合并相似的配置源以简化管理")

        env_cache_age = time.time() - self.enhanced_config_manager._env_cache_timestamp
        if env_cache_age > 300:  # 5分钟
            report['recommendations'].append("环境变量缓存较旧，考虑刷新")

        return report

    def optimize_existing_config_manager(self, config_manager_path: str):
        """优化现有的配置管理器"""
        config_file = Path(config_manager_path)
        if not config_file.exists():
            logger.error(f"Config manager file not found: {config_file}")
            return False

        try:
            # 创建备份
            backup_path = config_file.with_suffix(f'.backup.{int(time.time())}.py')
            backup_path.write_text(config_file.read_text(encoding='utf-8'), encoding='utf-8')
            logger.info(f"Created backup: {backup_path}")

            # 读取现有代码
            content = config_file.read_text(encoding='utf-8')

            # 添加增强功能
            enhancements = self._generate_enhancement_code()

            # 在类定义后添加增强方法
            if "class UnifiedConfigManager" in content:
                # 找到类定义的结束位置
                lines = content.split('\n')
                enhanced_lines = []
                in_class = False
                class_indent = 0

                for line in lines:
                    enhanced_lines.append(line)

                    if line.strip().startswith("class UnifiedConfigManager"):
                        in_class = True
                        class_indent = len(line) - len(line.lstrip())
                    elif in_class and line.strip() and not line.startswith(' ' * (class_indent + 1)) and not line.startswith('\t'):
                        # 类定义结束，添加增强方法
                        enhanced_lines.extend(
                            ['', '    # === 配置管理增强功能 ==='] + enhancements.split('\n'))
                        in_class = False

                if in_class and enhanced_lines:
                    # 类在文件结尾，直接添加
                    enhanced_lines.extend(['', '    # === 配置管理增强功能 ==='] + enhancements.split('\n'))

                # 写入增强后的代码
                config_file.write_text('\n'.join(enhanced_lines), encoding='utf-8')
                logger.info(f"Enhanced config manager: {config_file}")
                return True

        except Exception as e:
            logger.error(f"Failed to optimize config manager: {e}")
            return False

    def _generate_enhancement_code(self) -> str:
        """生成增强代码"""
        return '''
    def load_from_environment_variables(self, prefix: str = "RQA_") -> bool:
        """从环境变量加载配置（增强版）"""
        try:
            env_config = {}
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    config_key = key[len(prefix):].lower().replace('_', '.')
                    
                    # 智能类型转换
                    if value.lower() in ('true', 'false'):
                        converted_value = value.lower() == 'true'
                    elif value.isdigit():
                        converted_value = int(value)
                    elif '.' in value and value.replace('.', '').isdigit():
                        converted_value = float(value)
                    else:
                        converted_value = value
                        
                    env_config[config_key] = converted_value
                    
            self.update(env_config)
            return True
        except Exception as e:
            logger.error(f"Failed to load environment variables: {e}")
            return False
            
    def get_config_with_source_info(self, key: str, default: Any = None) -> Dict[str, Any]:
        """获取配置值及来源信息"""
        return {
            'value': self.get(key, default),
            'source': 'merged_config',
            'available': key in str(self._data),
            'type': type(self.get(key, default)).__name__
        }
        
    def validate_config_integrity(self) -> Dict[str, Any]:
        """验证配置完整性"""
        validation_result = {
            'is_valid': True,
            'missing_keys': [],
            'type_mismatches': [],
            'recommendations': []
        }
        
        # 检查关键配置项
        required_keys = [
            'logging.level',
            'system.debug'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                validation_result['missing_keys'].append(key)
                validation_result['is_valid'] = False
                
        if validation_result['missing_keys']:
            validation_result['recommendations'].append('添加缺失的必需配置项')
            
        return validation_result
'''


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="配置管理系统增强器")
    parser.add_argument("--project-root", default=".", help="项目根目录")
    parser.add_argument("--action", choices=["enhance",
                        "report", "test"], default="enhance", help="执行动作")
    parser.add_argument("--config-file", help="要优化的配置管理器文件路径")

    args = parser.parse_args()

    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    enhancer = ConfigManagementEnhancer(args.project_root)

    if args.action == "enhance":
        # 创建增强的配置管理器
        enhanced_manager = enhancer.create_enhanced_config_manager()
        print("✅ 创建增强配置管理器完成")

        # 如果指定了配置文件，进行优化
        if args.config_file:
            success = enhancer.optimize_existing_config_manager(args.config_file)
            if success:
                print(f"✅ 优化配置管理器完成: {args.config_file}")
            else:
                print(f"❌ 优化配置管理器失败: {args.config_file}")

    elif args.action == "report":
        # 生成配置报告
        report = enhancer.generate_config_report()
        print("\n📊 配置管理报告:")
        print(json.dumps(report, indent=2, ensure_ascii=False))

    elif args.action == "test":
        # 测试增强的配置管理器
        enhanced_manager = enhancer.create_enhanced_config_manager()

        print("🧪 测试配置管理器功能...")

        # 测试配置获取
        test_key = "system.debug"
        value, source = enhanced_manager.get_with_source(test_key, False)
        print(f"配置项 {test_key}: {value} (来源: {source})")

        # 测试配置验证
        validation = enhanced_manager.validate_all_config()
        print(f"配置验证结果: {'✅ 通过' if validation['is_valid'] else '❌ 失败'}")
        if validation['errors']:
            print(f"错误: {validation['errors']}")
        if validation['warnings']:
            print(f"警告: {validation['warnings']}")

        # 测试配置摘要
        summary = enhanced_manager.get_config_summary()
        print(f"配置摘要: {len(summary['config_sources'])} 个配置源, {summary['total_keys']} 个配置项")

        print("✅ 测试完成")


if __name__ == "__main__":
    main()

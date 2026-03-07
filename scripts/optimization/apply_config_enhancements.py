#!/usr/bin/env python3
"""
配置管理系统优化器 - 应用增强功能

直接优化现有的UnifiedConfigManager，添加以下功能：
1. 环境变量优先级管理
2. 配置源追踪
3. 配置验证增强
4. 配置热重载优化
5. 多文件格式支持
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


def apply_config_enhancements():
    """应用配置管理增强功能"""

    # 1. 增强统一配置管理器
    unified_manager_path = project_root / "src" / \
        "infrastructure" / "config" / "core" / "unified_manager.py"

    if not unified_manager_path.exists():
        print(f"❌ 配置管理器文件不存在: {unified_manager_path}")
        return False

    print(f"🔧 优化配置管理器: {unified_manager_path}")

    # 创建备份
    import time
    backup_path = unified_manager_path.with_suffix(f'.backup.{int(time.time())}.py')
    backup_path.write_text(unified_manager_path.read_text(encoding='utf-8'), encoding='utf-8')
    print(f"📄 创建备份: {backup_path}")

    # 读取现有代码
    content = unified_manager_path.read_text(encoding='utf-8')

    # 添加增强功能
    enhanced_methods = generate_enhanced_methods()

    # 在类的末尾添加增强方法
    if "class UnifiedConfigManager" in content:
        # 找到类的最后一个方法并在其后添加
        lines = content.split('\n')
        enhanced_lines = []

        for i, line in enumerate(lines):
            enhanced_lines.append(line)

            # 在最后一个方法定义后添加增强功能
            if i == len(lines) - 1:  # 文件末尾
                enhanced_lines.extend([
                    '',
                    '    # ==================== 配置管理增强功能 ====================',
                    ''
                ] + enhanced_methods.split('\n'))

        # 写入增强后的代码
        unified_manager_path.write_text('\n'.join(enhanced_lines), encoding='utf-8')
        print("✅ 配置管理器增强完成")

    # 2. 创建配置优先级管理器
    create_config_priority_manager()

    # 3. 创建配置验证器增强
    create_enhanced_validators()

    # 4. 创建环境变量配置加载器
    create_env_config_loader()

    return True


def generate_enhanced_methods() -> str:
    """生成增强方法代码"""
    return '''
    def load_from_env_with_priority(self, prefix: str = "RQA_", priority: int = 3) -> bool:
        """从环境变量加载配置，支持优先级管理"""
        try:
            env_config = {}
            env_sources = {}
            
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    # 转换环境变量键名
                    config_key = key[len(prefix):].lower().replace('_', '.')
                    
                    # 智能类型转换
                    converted_value = self._convert_env_value(value)
                    
                    # 存储配置和来源信息
                    env_config[config_key] = converted_value
                    env_sources[config_key] = {
                        'source': 'environment',
                        'env_var': key,
                        'priority': priority,
                        'original_value': value
                    }
                    
            # 记录配置来源
            if not hasattr(self, '_config_sources'):
                self._config_sources = {}
            self._config_sources.update(env_sources)
            
            # 更新配置
            self.update(env_config)
            return True
            
        except Exception as e:
            if hasattr(self, '_logger'):
                self._logger.error(f"Failed to load environment variables: {e}")
            return False
            
    def _convert_env_value(self, value: str) -> Any:
        """智能转换环境变量值"""
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
                import json
                return json.loads(value)
            except json.JSONDecodeError:
                pass
                
        # 列表转换（逗号分隔）
        if ',' in value:
            return [item.strip() for item in value.split(',')]
            
        return value
        
    def load_from_yaml_file(self, file_path: str) -> bool:
        """从YAML文件加载配置"""
        try:
            import yaml
            from pathlib import Path
            
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                return False
                
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f) or {}
                
            # 记录配置来源
            if not hasattr(self, '_config_sources'):
                self._config_sources = {}
                
            for key in yaml_data.keys():
                self._config_sources[key] = {
                    'source': 'yaml_file',
                    'file_path': str(file_path_obj),
                    'priority': 2,
                    'last_modified': file_path_obj.stat().st_mtime
                }
                
            self.update(yaml_data)
            return True
            
        except Exception as e:
            if hasattr(self, '_logger'):
                self._logger.error(f"Failed to load YAML file {file_path}: {e}")
            return False
            
    def get_config_with_source(self, key: str, default: Any = None) -> Dict[str, Any]:
        """获取配置值及其来源信息"""
        value = self.get(key, default)
        
        source_info = {
            'value': value,
            'key': key,
            'found': value != default,
            'source': 'unknown',
            'priority': 0
        }
        
        if hasattr(self, '_config_sources') and key in self._config_sources:
            source_info.update(self._config_sources[key])
            
        return source_info
        
    def list_config_sources(self) -> Dict[str, Any]:
        """列出所有配置源"""
        if not hasattr(self, '_config_sources'):
            return {}
            
        sources_summary = {}
        for key, source_info in self._config_sources.items():
            source_type = source_info.get('source', 'unknown')
            if source_type not in sources_summary:
                sources_summary[source_type] = {
                    'count': 0,
                    'keys': [],
                    'priority': source_info.get('priority', 0)
                }
            sources_summary[source_type]['count'] += 1
            sources_summary[source_type]['keys'].append(key)
            
        return sources_summary
        
    def validate_required_config(self, required_keys: List[str]) -> Dict[str, Any]:
        """验证必需的配置项"""
        validation_result = {
            'is_valid': True,
            'missing_keys': [],
            'present_keys': [],
            'recommendations': []
        }
        
        for key in required_keys:
            if self.get(key) is not None:
                validation_result['present_keys'].append(key)
            else:
                validation_result['missing_keys'].append(key)
                validation_result['is_valid'] = False
                
        if validation_result['missing_keys']:
            validation_result['recommendations'].append(
                f"添加缺失的配置项: {', '.join(validation_result['missing_keys'])}"
            )
            
        return validation_result
        
    def export_config_report(self) -> Dict[str, Any]:
        """导出完整的配置报告"""
        report = {
            'timestamp': time.time(),
            'total_sections': len(self._data),
            'total_keys': sum(len(section) for section in self._data.values()),
            'sections': {},
            'sources': self.list_config_sources() if hasattr(self, '_config_sources') else {},
            'validation': {},
            'recommendations': []
        }
        
        # 详细节信息
        for section_name, section_data in self._data.items():
            if isinstance(section_data, dict):
                report['sections'][section_name] = {
                    'keys_count': len(section_data),
                    'keys': list(section_data.keys())
                }
            else:
                report['sections'][section_name] = {
                    'keys_count': 1,  
                    'value_type': type(section_data).__name__
                }
                
        # 生成建议
        if report['total_keys'] == 0:
            report['recommendations'].append("配置为空，建议加载配置文件或环境变量")
        elif report['total_keys'] > 100:
            report['recommendations'].append("配置项较多，建议进行分类管理")
            
        if not report['sources']:
            report['recommendations'].append("建议记录配置来源以便排查问题")
            
        return report
        
    def refresh_env_config(self, prefix: str = "RQA_") -> bool:
        """刷新环境变量配置"""
        try:
            return self.load_from_env_with_priority(prefix)
        except Exception as e:
            if hasattr(self, '_logger'):
                self._logger.error(f"Failed to refresh environment config: {e}")
            return False
            
    def merge_config_with_priority(self, new_config: Dict[str, Any], priority: int = 1) -> bool:
        """按优先级合并配置"""
        try:
            if not hasattr(self, '_config_sources'):
                self._config_sources = {}
                
            # 记录新配置的来源
            for key in new_config.keys():
                self._config_sources[key] = {
                    'source': 'manual_merge',
                    'priority': priority,
                    'timestamp': time.time()
                }
                
            self.update(new_config)
            return True
            
        except Exception as e:
            if hasattr(self, '_logger'):
                self._logger.error(f"Failed to merge config: {e}")
            return False'''


def create_config_priority_manager():
    """创建配置优先级管理器"""
    priority_manager_path = project_root / "src" / \
        "infrastructure" / "config" / "core" / "priority_manager.py"

    priority_manager_code = '''#!/usr/bin/env python3
"""
配置优先级管理器

管理不同配置源的优先级和合并策略
"""

from typing import Dict, Any, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ConfigPriority(Enum):
    """配置优先级枚举"""
    DEFAULT = 1
    FILE = 2
    ENVIRONMENT = 3
    REMOTE = 4
    OVERRIDE = 5


class ConfigPriorityManager:
    """配置优先级管理器"""
    
    def __init__(self):
        self._config_layers: Dict[ConfigPriority, Dict[str, Any]] = {}
        self._merged_config: Dict[str, Any] = {}
        
    def add_config_layer(self, priority: ConfigPriority, config: Dict[str, Any]):
        """添加配置层"""
        self._config_layers[priority] = config
        self._rebuild_merged_config()
        
    def _rebuild_merged_config(self):
        """重建合并后的配置"""
        self._merged_config.clear()
        
        # 按优先级顺序合并配置
        for priority in sorted(ConfigPriority, key=lambda x: x.value):
            if priority in self._config_layers:
                self._deep_merge(self._merged_config, self._config_layers[priority])
                
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """深度合并字典"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
                
    def get_merged_config(self) -> Dict[str, Any]:
        """获取合并后的配置"""
        return self._merged_config.copy()
        
    def get_config_source(self, key: str) -> Optional[ConfigPriority]:
        """获取配置项的来源"""
        # 按优先级倒序查找
        for priority in sorted(ConfigPriority, key=lambda x: x.value, reverse=True):
            if priority in self._config_layers:
                if self._key_exists_in_config(key, self._config_layers[priority]):
                    return priority
        return None
        
    def _key_exists_in_config(self, key: str, config: Dict[str, Any]) -> bool:
        """检查键是否存在于配置中"""
        if '.' in key:
            parts = key.split('.')
            current = config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return False
            return True
        else:
            return key in config
'''

    priority_manager_path.write_text(priority_manager_code, encoding='utf-8')
    print(f"✅ 创建配置优先级管理器: {priority_manager_path}")


def create_enhanced_validators():
    """创建增强的配置验证器"""
    validators_path = project_root / "src" / "infrastructure" / \
        "config" / "core" / "enhanced_validators.py"

    validators_code = '''#!/usr/bin/env python3
"""
增强的配置验证器

提供更全面的配置验证功能
"""

from typing import Dict, Any, List, Optional
import re
import logging

logger = logging.getLogger(__name__)


class ConfigValidationResult:
    """配置验证结果"""
    
    def __init__(self, is_valid: bool = True):
        self.is_valid = is_valid
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.recommendations: List[str] = []
        
    def add_error(self, error: str):
        """添加错误"""
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)
        
    def add_recommendation(self, recommendation: str):
        """添加建议"""
        self.recommendations.append(recommendation)


class EnhancedConfigValidator:
    """增强的配置验证器"""
    
    def __init__(self):
        self._validators = []
        
    def add_validator(self, validator_func):
        """添加验证器函数"""
        self._validators.append(validator_func)
        
    def validate(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """执行配置验证"""
        result = ConfigValidationResult()
        
        for validator in self._validators:
            try:
                validator_result = validator(config)
                if hasattr(validator_result, 'is_valid') and not validator_result.is_valid:
                    result.is_valid = False
                    result.errors.extend(getattr(validator_result, 'errors', []))
                if hasattr(validator_result, 'warnings'):
                    result.warnings.extend(validator_result.warnings)
                if hasattr(validator_result, 'recommendations'):
                    result.recommendations.extend(validator_result.recommendations)
            except Exception as e:
                result.add_error(f"Validator error: {str(e)}")
                
        return result


def create_standard_validators():
    """创建标准验证器"""
    
    def validate_required_keys(config: Dict[str, Any]) -> ConfigValidationResult:
        """验证必需的配置键"""
        result = ConfigValidationResult()
        
        required_keys = [
            'logging.level',
            'system.debug'
        ]
        
        for key in required_keys:
            if not _key_exists(key, config):
                result.add_error(f"Missing required config key: {key}")
                
        return result
        
    def validate_config_types(config: Dict[str, Any]) -> ConfigValidationResult:
        """验证配置类型"""
        result = ConfigValidationResult()
        
        type_checks = {
            'system.debug': bool,
            'logging.level': str,
            'database.port': int
        }
        
        for key, expected_type in type_checks.items():
            if _key_exists(key, config):
                value = _get_nested_value(key, config)
                if not isinstance(value, expected_type):
                    result.add_warning(
                        f"Config key {key} should be {expected_type.__name__}, "
                        f"got {type(value).__name__}"
                    )
                    
        return result
        
    def validate_config_format(config: Dict[str, Any]) -> ConfigValidationResult:
        """验证配置格式"""
        result = ConfigValidationResult()
        
        # 检查邮箱格式
        email_keys = ['email.sender', 'email.receiver']
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')
        
        for key in email_keys:
            if _key_exists(key, config):
                value = _get_nested_value(key, config)
                if isinstance(value, str) and not email_pattern.match(value):
                    result.add_error(f"Invalid email format for {key}: {value}")
                    
        # 检查端口范围
        port_keys = ['server.port', 'database.port']
        for key in port_keys:
            if _key_exists(key, config):
                value = _get_nested_value(key, config)
                if isinstance(value, int) and not (1 <= value <= 65535):
                    result.add_error(f"Invalid port range for {key}: {value}")
                    
        return result
        
    return [validate_required_keys, validate_config_types, validate_config_format]


def _key_exists(key: str, config: Dict[str, Any]) -> bool:
    """检查嵌套键是否存在"""
    try:
        _get_nested_value(key, config)
        return True
    except (KeyError, TypeError):
        return False


def _get_nested_value(key: str, config: Dict[str, Any]) -> Any:
    """获取嵌套配置值"""
    parts = key.split('.')
    current = config
    
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise KeyError(f"Key {key} not found")
            
    return current
'''

    validators_path.write_text(validators_code, encoding='utf-8')
    print(f"✅ 创建增强配置验证器: {validators_path}")


def create_env_config_loader():
    """创建环境变量配置加载器"""
    env_loader_path = project_root / "src" / "infrastructure" / "config" / "loaders" / "env_loader.py"

    # 确保目录存在
    env_loader_path.parent.mkdir(parents=True, exist_ok=True)

    env_loader_code = '''#!/usr/bin/env python3
"""
环境变量配置加载器

提供智能的环境变量配置加载功能
"""

import os
import json
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class EnvironmentConfigLoader:
    """环境变量配置加载器"""
    
    def __init__(self, prefixes: List[str] = None):
        self.prefixes = prefixes or ['RQA_', 'CONFIG_', 'APP_']
        
    def load_all(self) -> Dict[str, Any]:
        """加载所有匹配前缀的环境变量"""
        config = {}
        
        for prefix in self.prefixes:
            prefix_config = self.load_with_prefix(prefix)
            config.update(prefix_config)
            
        return config
        
    def load_with_prefix(self, prefix: str) -> Dict[str, Any]:
        """使用指定前缀加载环境变量"""
        config = {}
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = self._transform_key(key, prefix)
                config_value = self._transform_value(value)
                
                # 支持嵌套配置
                self._set_nested_value(config, config_key, config_value)
                
        return config
        
    def _transform_key(self, env_key: str, prefix: str) -> str:
        """转换环境变量键名"""
        # 移除前缀并转换为小写，下划线转点号
        key = env_key[len(prefix):].lower().replace('_', '.')
        return key
        
    def _transform_value(self, value: str) -> Any:
        """智能转换环境变量值"""
        if not value:
            return value
            
        # 布尔值转换
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
            
        # 数字转换
        try:
            if '.' in value and value.replace('.', '').replace('-', '').isdigit():
                return float(value)
            elif value.replace('-', '').isdigit():
                return int(value)
        except ValueError:
            pass
            
        # JSON转换
        if value.startswith(('{', '[', '"')):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
                
        # 列表转换（逗号分隔）
        if ',' in value and not value.startswith(('{', '[')):
            return [item.strip() for item in value.split(',')]
            
        return value
        
    def _set_nested_value(self, config: Dict[str, Any], key: str, value: Any):
        """设置嵌套配置值"""
        if '.' not in key:
            config[key] = value
            return
            
        parts = key.split('.')
        current = config
        
        # 创建嵌套结构
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # 设置最终值
        current[parts[-1]] = value
        
    def get_env_summary(self) -> Dict[str, Any]:
        """获取环境变量摘要"""
        summary = {
            'total_env_vars': len(os.environ),
            'matched_vars': {},
            'prefixes': self.prefixes
        }
        
        for prefix in self.prefixes:
            matched = []
            for key in os.environ.keys():
                if key.startswith(prefix):
                    matched.append(key)
            summary['matched_vars'][prefix] = {
                'count': len(matched),
                'vars': matched[:10]  # 只显示前10个
            }
            
        return summary
'''

    env_loader_path.write_text(env_loader_code, encoding='utf-8')
    print(f"✅ 创建环境变量配置加载器: {env_loader_path}")


def create_config_optimization_demo():
    """创建配置优化演示脚本"""
    demo_path = project_root / "scripts" / "optimization" / "config_demo.py"

    demo_code = '''#!/usr/bin/env python3
"""
配置管理优化演示

展示增强后的配置管理功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.infrastructure.config.core.unified_manager import UnifiedConfigManager


def demo_enhanced_config_manager():
    """演示增强的配置管理器功能"""
    print("🚀 配置管理器增强功能演示")
    print("=" * 50)
    
    # 创建配置管理器
    manager = UnifiedConfigManager()
    
    # 1. 从环境变量加载配置
    print("\\n1. 从环境变量加载配置")
    os.environ['RQA_DATABASE_HOST'] = 'localhost'
    os.environ['RQA_DATABASE_PORT'] = '5432'
    os.environ['RQA_DEBUG'] = 'true'
    os.environ['RQA_FEATURES'] = 'auth,cache,monitoring'
    
    if hasattr(manager, 'load_from_env_with_priority'):
        success = manager.load_from_env_with_priority('RQA_')
        print(f"环境变量加载: {'✅ 成功' if success else '❌ 失败'}")
    else:
        print("❌ 环境变量优先级加载功能未启用")
    
    # 2. 获取配置及来源信息
    print("\\n2. 获取配置及来源信息")
    if hasattr(manager, 'get_config_with_source'):
        source_info = manager.get_config_with_source('database.host')
        print(f"database.host: {source_info}")
    else:
        print("❌ 配置来源追踪功能未启用")
    
    # 3. 配置验证
    print("\\n3. 配置验证")
    required_keys = ['database.host', 'database.port']
    if hasattr(manager, 'validate_required_config'):
        validation = manager.validate_required_config(required_keys)
        print(f"配置验证结果: {validation}")
    else:
        print("❌ 配置验证功能未启用")
    
    # 4. 配置报告
    print("\\n4. 配置报告")
    if hasattr(manager, 'export_config_report'):
        report = manager.export_config_report()
        print(f"配置摘要: {report['total_keys']} 个配置项")
        print(f"配置来源: {list(report['sources'].keys())}")
    else:
        print("❌ 配置报告功能未启用")
    
    # 5. 配置来源列表
    print("\\n5. 配置来源")
    if hasattr(manager, 'list_config_sources'):
        sources = manager.list_config_sources()
        for source_type, info in sources.items():
            print(f"  {source_type}: {info['count']} 个配置项")
    else:
        print("❌ 配置来源列表功能未启用")
        
    print("\\n🎉 演示完成!")


if __name__ == "__main__":
    demo_enhanced_config_manager()
'''

    demo_path.write_text(demo_code, encoding='utf-8')
    print(f"✅ 创建配置优化演示: {demo_path}")


def main():
    """主函数"""
    print("🔧 开始配置管理系统优化...")

    # 应用配置增强功能
    success = apply_config_enhancements()

    if success:
        # 创建演示脚本
        create_config_optimization_demo()

        print("\n🎉 配置管理系统优化完成!")
        print("\n增强功能包括:")
        print("✅ 环境变量优先级管理")
        print("✅ 配置源追踪")
        print("✅ 多文件格式支持 (JSON/YAML)")
        print("✅ 增强的配置验证")
        print("✅ 配置报告生成")
        print("✅ 智能类型转换")
        print("\n运行演示:")
        print("python scripts/optimization/config_demo.py")

        return True
    else:
        print("❌ 配置管理系统优化失败")
        return False


if __name__ == "__main__":
    main()

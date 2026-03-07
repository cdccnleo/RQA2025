"""
默认配置文件

定义质量检查工具的默认配置。
"""

from typing import Dict, Any

# 默认全局配置
DEFAULT_CONFIG = {
    # 全局设置
    'enabled_checkers': ['duplicate', 'interface', 'complexity'],
    'fail_on_error': True,
    'fail_on_critical': True,
    'parallel_execution': True,
    'max_workers': 4,
    'exclude_patterns': [
        '__pycache__',
        '.git',
        'node_modules',
        '*.pyc',
        '*.pyo',
        '.pytest_cache',
        'build',
        'dist',
        '*.egg-info'
    ],

    # 报告设置
    'reporters': {
        'console': {
            'enabled': True,
            'colors': True,
            'verbose': False
        },
        'json': {
            'enabled': True,
            'output_file': 'quality_report.json',
            'pretty_print': True,
            'include_metadata': True
        },
        'html': {
            'enabled': True,
            'output_file': 'quality_report.html',
            'include_charts': True,
            'theme': 'default'
        }
    },

    # 检查器特定配置
    'duplicate': {
        'min_lines': 5,
        'max_lines': 50,
        'similarity_threshold': 0.8,
        'duplicate_threshold': 3,
        'ignore_imports': True,
        'ignore_comments': True,
        'ignore_docstrings': True
    },

    'interface': {
        'check_abstract_methods': True,
        'check_method_signatures': True,
        'check_property_implementations': True,
        'allow_extra_methods': True,
        'strict_mode': False
    },

    'complexity': {
        'max_cyclomatic_complexity': 10,
        'max_lines_per_function': 50,
        'max_nesting_depth': 4,
        'max_parameters': 5,
        'min_maintainability_index': 50,
        'check_functions': True,
        'check_classes': True,
        'check_modules': True
    }
}

# 基础设施层专用配置
INFRASTRUCTURE_CONFIG = {
    **DEFAULT_CONFIG,

    # 基础设施层特定设置
    'target_directories': [
        'src/infrastructure/cache',
        'src/infrastructure/config',
        'src/infrastructure/logging',
        'src/infrastructure/monitoring'
    ],

    # 更严格的质量标准
    'complexity': {
        **DEFAULT_CONFIG['complexity'],
        'max_cyclomatic_complexity': 8,  # 更严格
        'max_lines_per_function': 40,     # 更严格
        'min_maintainability_index': 60,  # 更高要求
    },

    'duplicate': {
        **DEFAULT_CONFIG['duplicate'],
        'duplicate_threshold': 2,  # 更敏感
        'similarity_threshold': 0.75  # 更严格
    },

    # 报告设置优化
    'reporters': {
        **DEFAULT_CONFIG['reporters'],
        'html': {
            **DEFAULT_CONFIG['reporters']['html'],
            'include_charts': True
        }
    }
}


def get_config(config_type: str = 'default') -> Dict[str, Any]:
    """
    获取配置

    Args:
        config_type: 配置类型 ('default' 或 'infrastructure')

    Returns:
        Dict[str, Any]: 配置字典
    """
    if config_type == 'infrastructure':
        return INFRASTRUCTURE_CONFIG.copy()
    else:
        return DEFAULT_CONFIG.copy()


def merge_config(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        Dict[str, Any]: 合并后的配置
    """
    merged = base_config.copy()

    def deep_merge(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                deep_merge(target[key], value)
            else:
                target[key] = value
        return target

    return deep_merge(merged, override_config)

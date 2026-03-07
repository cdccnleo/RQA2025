"""
基础设施层 - 安全管理组件

security_factory 模块

安全管理相关的文件
提供安全管理相关的功能实现。
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
安全工厂类
统一管理所有安全组件的创建和配置，消除重复代码
"""

from typing import Dict, Any, Optional
from .base_security import BaseSecurityComponent

# 动态导入服务层模块，避免循环依赖
try:
    # 跨层级导入：infrastructure层组件
    from src.infrastructure.services.security import (
        EnhancedSecurityManager,
        AuthManager,
        SecurityAuditor,
        DataSanitizer
    )

except ImportError:
    # 如果导入失败，使用占位符类

    class EnhancedSecurityManager:

        def __init__(self, config=None):

            raise ImportError("EnhancedSecurityManager not available")

    class AuthManager:

        def __init__(self, **kwargs):

            raise ImportError("AuthManager not available")

    class SecurityAuditor:

        def __init__(self, **kwargs):

            raise ImportError("SecurityAuditor not available")

    class DataSanitizer:

        def __init__(self, **kwargs):

            raise ImportError("DataSanitizer not available")


class SecurityFactory:

    """安全工厂类"""

    # 支持的安全组件类型
    SUPPORTED_TYPES = {
        'base': BaseSecurityComponent,
        'enhanced': EnhancedSecurityManager,
        'auth': AuthManager,
        'auditor': SecurityAuditor,
        'sanitizer': DataSanitizer
    }

    @classmethod
    def create_security_component(cls,


                                  component_type: str,
                                  config: Optional[Dict[str, Any]] = None,
                                  **kwargs):
        """
        创建安全组件

        Args:
            component_type: 组件类型
            config: 配置参数
            **kwargs: 其他参数

        Returns:
            安全组件实例

        Raises:
            ValueError: 不支持的组件类型
        """
        if component_type not in cls.SUPPORTED_TYPES:
            raise ValueError(f"不支持的安全组件类型: {component_type}")

        component_class = cls.SUPPORTED_TYPES[component_type]

        # 合并配置
        if config:
            kwargs.update(config)

        try:
            if component_type == 'base':
                # 过滤掉BaseSecurity不支持的参数
                filtered_kwargs = {k: v for k, v in kwargs.items()
                                   if k not in ['secret_key']}
                return component_class(**filtered_kwargs)
            elif component_type == 'enhanced':
                return component_class(config=kwargs)
            elif component_type == 'auth':
                return component_class(**kwargs)
            elif component_type == 'auditor':
                return component_class(**kwargs)
            elif component_type == 'sanitizer':
                return component_class(**kwargs)
            else:
                return component_class(**kwargs)
        except Exception as e:
            raise RuntimeError(f"创建安全组件失败: {e}")

    @classmethod
    def create_default_security_stack(cls,


                                      config: Optional[Dict[str, Any]] = None):
        """
        创建默认安全组件栈

        Args:
            config: 配置参数

        Returns:
            包含所有安全组件的字典
        """
        security_stack = {}

        # 创建基础安全组件
        security_stack['base'] = cls.create_security_component('base', config)

        # 创建增强安全管理器
        security_stack['enhanced'] = cls.create_security_component('enhanced', config)

        # 创建身份验证管理器
        auth_config = config.get('auth', {}) if config else {}
        security_stack['auth'] = cls.create_security_component('auth', auth_config)

        # 创建安全审计器
        auditor_config = config.get('auditor', {}) if config else {}
        security_stack['auditor'] = cls.create_security_component('auditor', auditor_config)

        # 创建数据清理器
        sanitizer_config = config.get('sanitizer', {}) if config else {}
        security_stack['sanitizer'] = cls.create_security_component('sanitizer', sanitizer_config)

        return security_stack

    @classmethod
    def get_component_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        获取所有支持组件的详细信息

        Returns:
            组件信息字典
        """
        info = {}
        for component_type, component_class in cls.SUPPORTED_TYPES.items():
            info[component_type] = {
                'class': component_class.__name__,
                'module': component_class.__module__,
                'doc': component_class.__doc__ or 'No documentation available'
            }

        return info

    @classmethod
    def validate_security_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证安全配置

        Args:
            config: 配置字典

        Returns:
            验证结果字典
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }

        # 检查必需配置
        required_fields = ['encryption_level', 'session_timeout', 'max_login_attempts']
        for field in required_fields:
            if field not in config:
                validation_result['warnings'].append(f"缺少配置字段: {field}")

        # 验证加密级别
        if 'encryption_level' in config:
            encryption_level = config['encryption_level']
            if encryption_level not in ['low', 'medium', 'high', 'critical']:
                validation_result['errors'].append(f"无效的加密级别: {encryption_level}")
                validation_result['valid'] = False

        # 验证会话超时
        if 'session_timeout' in config:
            session_timeout = config['session_timeout']
            if session_timeout < 300:  # 5分钟
                validation_result['warnings'].append("会话超时时间过短，建议至少5分钟")
            elif session_timeout > 86400:  # 24小时
                validation_result['warnings'].append("会话超时时间过长，建议不超过24小时")

        # 验证最大登录尝试次数
        if 'max_login_attempts' in config:
            max_attempts = config['max_login_attempts']
            if max_attempts < 3:
                validation_result['warnings'].append("最大登录尝试次数过少，建议至少3次")
            elif max_attempts > 10:
                validation_result['warnings'].append("最大登录尝试次数过多，建议不超过10次")

        # 生成建议
        if not validation_result['errors']:
            validation_result['recommendations'].append("配置验证通过，建议定期审查安全设置")

        return validation_result


def create_security_manager(manager_type: str = 'enhanced',


                            config: Optional[Dict[str, Any]] = None,
                            **kwargs):
    """
    便捷函数：创建安全管理器

    Args:
        manager_type: 管理器类型
        config: 配置参数
        **kwargs: 其他参数

    Returns:
        安全管理器实例
    """
    return SecurityFactory.create_security_component(manager_type, config, **kwargs)


def get_security_factory_info() -> Dict[str, Any]:
    """
    获取安全工厂信息

    Returns:
        工厂信息字典
    """
    return {
        'supported_types': list(SecurityFactory.SUPPORTED_TYPES.keys()),
        'component_info': SecurityFactory.get_component_info(),
        'factory_class': SecurityFactory.__name__,
        'version': '1.0.0'
    }

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
部署配置管理模块
负责管理不同环境的部署配置和参数
"""

import os
import yaml
from typing import Dict, Any, Optional
from src.infrastructure.utils.logger import get_logger, configure_logging
from src.infrastructure.config.config_manager import ConfigManager

logger = get_logger(__name__)

class DeploymentManager:
    def __init__(self, config: Dict[str, Any], config_manager: Optional[ConfigManager] = None):
        """
        初始化部署管理器
        :param config: 基础配置
        :param config_manager: 可选的配置管理器实例，用于测试时注入mock对象
        """
        self.config = config
        
        # 测试钩子：允许注入mock的ConfigManager
        if config_manager is not None:
            self.config_manager = config_manager
        else:
            self.config_manager = ConfigManager(config)
            
        self.environments = {
            'dev': 'config/deploy_dev.yaml',
            'test': 'config/deploy_test.yaml',
            'prod': 'config/deploy_prod.yaml'
        }
        self.current_env = None
        self.current_config = None

    def load_environment(self, env: str) -> bool:
        """
        加载指定环境的部署配置
        :param env: 环境名称(dev/test/prod)
        :return: 是否加载成功
        """
        if env not in self.environments:
            logger.error(f"未知环境: {env}")
            return False

        config_file = self.environments[env]
        if not os.path.exists(config_file):
            logger.error(f"部署配置文件不存在: {config_file}")
            return False

        try:
            with open(config_file, 'r') as f:
                self.current_config = yaml.safe_load(f)
                self.current_env = env
                logger.info(f"成功加载 {env} 环境部署配置")
                return True
        except Exception as e:
            logger.error(f"加载部署配置失败: {str(e)}")
            return False

    def get_deployment_config(self, key: str, default: Any = None) -> Any:
        """
        获取当前环境的部署配置项
        :param key: 配置键
        :param default: 默认值
        :return: 配置值
        """
        if not self.current_config:
            logger.warning("未加载任何部署配置")
            return default

        keys = key.split('.')
        value = self.current_config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def validate_deployment(self) -> Dict[str, Any]:
        """
        验证部署配置的有效性
        :return: 验证结果和问题列表
        """
        if not self.current_config:
            return {
                'valid': False,
                'errors': ['未加载部署配置']
            }

        required_sections = [
            'database',
            'trading',
            'monitoring',
            'security'
        ]

        errors = []
        for section in required_sections:
            if section not in self.current_config:
                errors.append(f"缺少必要配置段: {section}")

        # 验证数据库连接配置
        db_config = self.current_config.get('database', {})
        if not db_config.get('host'):
            errors.append("数据库配置缺少host参数")
        if not db_config.get('port'):
            errors.append("数据库配置缺少port参数")

        # 验证交易参数
        trading_config = self.current_config.get('trading', {})
        if not trading_config.get('max_order_size'):
            errors.append("交易配置缺少max_order_size参数")
        if not trading_config.get('default_slippage'):
            errors.append("交易配置缺少default_slippage参数")

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def generate_deployment_script(self, output_path: str) -> bool:
        """
        生成部署脚本
        :param output_path: 输出路径
        :return: 是否生成成功
        """
        if not self.current_config:
            logger.error("无法生成部署脚本: 未加载部署配置")
            return False

        try:
            template = self._get_deployment_template()
            filled_template = self._fill_template(template)

            with open(output_path, 'w') as f:
                f.write(filled_template)

            logger.info(f"部署脚本已生成: {output_path}")
            return True
        except Exception as e:
            logger.error(f"生成部署脚本失败: {str(e)}")
            return False

    def _get_deployment_template(self) -> str:
        """
        获取部署模板
        :return: 模板内容
        """
        # 根据环境选择不同模板
        if self.current_env == 'prod':
            return """#!/bin/bash
# 生产环境部署脚本

# 数据库配置
export DB_HOST={db_host}
export DB_PORT={db_port}
export DB_USER={db_user}
export DB_PASS={db_pass}

# 交易参数
export MAX_ORDER_SIZE={max_order_size}
export DEFAULT_SLIPPAGE={default_slippage}

# 启动服务
./start_service.sh --env=prod
"""
        else:
            return """#!/bin/bash
# 非生产环境部署脚本

# 数据库配置
export DB_HOST={db_host}
export DB_PORT={db_port}
export DB_USER={db_user}
export DB_PASS={db_pass}

# 交易参数
export MAX_ORDER_SIZE={max_order_size}
export DEFAULT_SLIPPAGE={default_slippage}

# 启动服务
./start_service.sh --env=dev
"""

    def _fill_template(self, template: str) -> str:
        """
        填充模板参数
        :param template: 模板字符串
        :return: 填充后的字符串
        """
        if not self.current_config:
            return template
            
        db_config = self.current_config.get('database', {})
        trading_config = self.current_config.get('trading', {})

        params = {
            'db_host': db_config.get('host', 'localhost'),
            'db_port': db_config.get('port', 5432),
            'db_user': db_config.get('user', 'postgres'),
            'db_pass': db_config.get('password', ''),
            'max_order_size': trading_config.get('max_order_size', 10000),
            'default_slippage': trading_config.get('default_slippage', 0.001),
            'env': self.current_env
        }

        return template.format(**params)

    def get_environment_summary(self) -> Dict[str, Any]:
        """
        获取当前环境配置摘要
        :return: 配置摘要
        """
        if not self.current_config:
            return {}

        return {
            'environment': self.current_env,
            'database': {
                'host': self.current_config.get('database', {}).get('host'),
                'port': self.current_config.get('database', {}).get('port')
            },
            'trading': {
                'max_order_size': self.current_config.get('trading', {}).get('max_order_size'),
                'default_slippage': self.current_config.get('trading', {}).get('default_slippage')
            },
            'security': {
                'level': self.current_config.get('security', {}).get('level')
            }
        }

    def switch_environment(self, env: str) -> bool:
        """
        切换部署环境
        :param env: 目标环境
        :return: 是否切换成功
        """
        if env == self.current_env:
            return True

        if not self.load_environment(env):
            return False

        # 应用新环境的配置
        # 注意：ConfigManager没有load方法，这里只是记录日志
        logger.info(f"已切换到 {env} 环境")
        return True

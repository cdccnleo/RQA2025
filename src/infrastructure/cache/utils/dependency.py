"""
dependency 模块

提供 dependency 相关功能和接口。
"""

import logging

from typing import Dict
"""
配置依赖检查工具类
负责检查配置项之间的依赖关系
"""

logger = logging.getLogger(__name__)


class DependencyChecker:

    """配置依赖检查器"""

    @staticmethod
    def check_dependencies(new_config: Dict, full_config: Dict) -> Dict:
        """检查配置依赖关系"
        Args:
            new_config: 当前更新的配置键值对
            full_config: 完整的当前配置状态
        Returns:
            错误字典，空字典表示验证通过
        """
        errors = {}
        logger.debug(f"Starting dependency validation for {new_config}")

        # 创建临时合并配置用于验证
        temp_config = {**full_config, **new_config}

        # 检查缓存依赖关系
        if 'cache.enabled' in new_config:
            cache_errors = DependencyChecker._check_cache_dependencies(new_config, temp_config)
            errors.update(cache_errors)

        # 检查数据库依赖关系
        if 'database.enabled' in new_config:
            db_errors = DependencyChecker._check_database_dependencies(new_config, temp_config)
            errors.update(db_errors)

        # 检查交易依赖关系
        if 'trading.enabled' in new_config:
            trading_errors = DependencyChecker._check_trading_dependencies(new_config, temp_config)
            errors.update(trading_errors)

        logger.debug(f"Dependency validation result: {errors}")
        return errors

    @staticmethod
    def _check_cache_dependencies(new_config: Dict, temp_config: Dict) -> Dict:
        """检查缓存相关依赖"""
        errors = {}
        cache_enabled = False

        if 'cache.enabled' in new_config:
            cache_enabled_value = new_config['cache.enabled']
            logger.debug(f"Validating cache.enabled: {cache_enabled_value} (type: {type(cache_enabled_value)})")

            # 处理各种形式的cache.enabled值
            if isinstance(cache_enabled_value, str):
                lower_val = cache_enabled_value.lower()
                if lower_val not in ['true', 'false', '1', '0', 'yes', 'no', 'on', 'off']:
                    errors['cache.enabled'] = "Invalid string value for cache.enabled"
                    return errors
                cache_enabled = lower_val in ['true', '1', 'yes', 'on']
                logger.debug(f"Converted string to bool: {cache_enabled}")
            elif isinstance(cache_enabled_value, (int, float)):
                if cache_enabled_value not in [0, 1]:
                    errors['cache.enabled'] = "Numeric value must be 0 or 1"
                    return errors
                cache_enabled = bool(cache_enabled_value)
                logger.debug(f"Converted number to bool: {cache_enabled}")
            elif isinstance(cache_enabled_value, bool):
                cache_enabled = cache_enabled_value
            else:
                errors['cache.enabled'] = "Invalid type for cache.enabled"
                return errors

        if cache_enabled:
            # 从完整配置中检查cache.size
            cache_size = temp_config.get('cache.size')
            logger.debug(f"Checking cache.size from full config: {cache_size}")

            if cache_size is None:
                logger.error("Cache size not set when enabling cache")
                errors['cache.size'] = "Cache size must be set when cache is enabled"
            elif not isinstance(cache_size, (int, float)):
                logger.error(f"Invalid cache size type: {type(cache_size)}")
                errors['cache.size'] = "Cache size must be a number"
            elif cache_size <= 0:
                logger.error(f"Invalid cache size value: {cache_size}")
                errors['cache.size'] = "Cache size must be positive"
            else:
                logger.debug("Cache size dependency validated successfully")

        return errors

    @staticmethod
    def _check_database_dependencies(new_config: Dict, temp_config: Dict) -> Dict:
        """检查数据库相关依赖"""
        errors = {}

        if 'database.enabled' in new_config:
            db_enabled = new_config['database.enabled']

        if db_enabled:
            # 检查必要的数据库配置
            required_db_configs = ['database.host', 'database.port', 'database.name']
            for config_key in required_db_configs:
                if temp_config.get(config_key) is None:
                    errors[f'database.{config_key.split(".")[-1]}'] = f"{config_key} is required when database is enabled"

        return errors

    @staticmethod
    def _check_trading_dependencies(new_config: Dict, temp_config: Dict) -> Dict:
        """检查交易相关依赖"""
        errors = {}

        if 'trading.enabled' in new_config:
            trading_enabled = new_config['trading.enabled']

        if trading_enabled:
            # 检查必要的交易配置
            required_trading_configs = ['trading.max_order_size', 'trading.risk_limit']
            for config_key in required_trading_configs:
                if temp_config.get(config_key) is None:
                    errors[f'trading.{config_key.split(".")[-1]}'] = f"{config_key} is required when trading is enabled"

        return errors

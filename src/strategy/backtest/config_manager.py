#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
回测层统一配置管理器

实现中期优化目标：与其他模块深度集成、实现统一配置管理、完善部署流程
"""

import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:

    """回测配置"""
    # 基础配置
    initial_capital: float = 1000000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001
    benchmark: Optional[str] = None
    risk_free_rate: float = 0.03

    # 性能配置
    max_workers: Optional[int] = None
    enable_cache: bool = True
    cache_dir: str = "cache / backtest_results"
    memory_limit_gb: float = 4.0
    enable_parallel: bool = True

    # 策略配置
    strategies: List[Dict[str, Any]] = field(default_factory=list)
    risk_limits: Dict[str, float] = field(default_factory=dict)

    # 数据配置
    data_source: str = "local"
    data_path: str = "data/"
    symbols: List[str] = field(default_factory=list)
    start_date: str = "2023 - 01 - 01"
    end_date: str = "2023 - 12 - 31"

    # 监控配置
    enable_monitoring: bool = True
    monitoring_interval: int = 30
    alert_enabled: bool = True

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "logs / backtest.log"

    # 输出配置
    output_dir: str = "results/"
    save_trades: bool = True
    save_portfolio: bool = True
    save_metrics: bool = True


@dataclass
class SystemConfig:

    """系统配置"""
    # 环境配置
    environment: str = "development"  # development, staging, production
    debug_mode: bool = False

    # 数据库配置
    database_url: Optional[str] = None
    redis_url: Optional[str] = None

    # 外部服务配置
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None

    # 安全配置
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None


class ConfigManager:

    """配置管理器"""

    def __init__(self, config_dir: str = "config/"):

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self.backtest_config = BacktestConfig()
        self.system_config = SystemConfig()
        self.custom_configs = {}

        # 加载默认配置
        self._load_default_configs()

    def _load_default_configs(self):
        """加载默认配置"""
        # 检查是否存在配置文件
        backtest_config_file = self.config_dir / "backtest_config.json"
        system_config_file = self.config_dir / "system_config.json"

        if backtest_config_file.exists():
            self.load_backtest_config(backtest_config_file)

        if system_config_file.exists():
            self.load_system_config(system_config_file)

    def load_backtest_config(self, config_file: str):
        """加载回测配置"""
        try:
            with open(config_file, 'r', encoding='utf - 8') as f:
                config_data = json.load(f)

            # 更新配置
            for key, value in config_data.items():
                if hasattr(self.backtest_config, key):
                    setattr(self.backtest_config, key, value)

            logger.info(f"回测配置已加载: {config_file}")

        except Exception as e:
            logger.error(f"加载回测配置失败: {e}")

    def save_backtest_config(self, config_file: str = None):
        """保存回测配置"""
        if config_file is None:
            config_file = self.config_dir / "backtest_config.json"

        try:
            config_data = asdict(self.backtest_config)
            with open(config_file, 'w', encoding='utf - 8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"回测配置已保存: {config_file}")

        except Exception as e:
            logger.error(f"保存回测配置失败: {e}")

    def load_system_config(self, config_file: str):
        """加载系统配置"""
        try:
            with open(config_file, 'r', encoding='utf - 8') as f:
                config_data = json.load(f)

            # 更新配置
            for key, value in config_data.items():
                if hasattr(self.system_config, key):
                    setattr(self.system_config, key, value)

            logger.info(f"系统配置已加载: {config_file}")

        except Exception as e:
            logger.error(f"加载系统配置失败: {e}")

    def save_system_config(self, config_file: str = None):
        """保存系统配置"""
        if config_file is None:
            config_file = self.config_dir / "system_config.json"

        try:
            config_data = asdict(self.system_config)
            with open(config_file, 'w', encoding='utf - 8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"系统配置已保存: {config_file}")

        except Exception as e:
            logger.error(f"保存系统配置失败: {e}")

    def load_custom_config(self, name: str, config_file: str):
        """加载自定义配置"""
        try:
            with open(config_file, 'r', encoding='utf - 8') as f:
                config_data = json.load(f)

            self.custom_configs[name] = config_data
            logger.info(f"自定义配置已加载: {name} -> {config_file}")

        except Exception as e:
            logger.error(f"加载自定义配置失败: {e}")

    def save_custom_config(self, name: str, config_data: Dict[str, Any], config_file: str = None):
        """保存自定义配置"""
        if config_file is None:
            config_file = self.config_dir / f"{name}_config.json"

        try:
            self.custom_configs[name] = config_data
            with open(config_file, 'w', encoding='utf - 8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"自定义配置已保存: {name} -> {config_file}")

        except Exception as e:
            logger.error(f"保存自定义配置失败: {e}")

    def get_config(self, config_type: str = "backtest") -> Dict[str, Any]:
        """获取配置"""
        if config_type == "backtest":
            return asdict(self.backtest_config)
        elif config_type == "system":
            return asdict(self.system_config)
        elif config_type in self.custom_configs:
            return self.custom_configs[config_type]
        else:
            return {}

    def update_config(self, config_type: str, updates: Dict[str, Any]):
        """更新配置"""
        if config_type == "backtest":
            for key, value in updates.items():
                if hasattr(self.backtest_config, key):
                    setattr(self.backtest_config, key, value)
        elif config_type == "system":
            for key, value in updates.items():
                if hasattr(self.system_config, key):
                    setattr(self.system_config, key, value)
        elif config_type in self.custom_configs:
            self.custom_configs[config_type].update(updates)

        logger.info(f"配置已更新: {config_type}")

    def validate_config(self, config_type: str = "backtest") -> List[str]:
        """验证配置"""
        errors = []

        if config_type == "backtest":
            config = self.backtest_config

        if config.initial_capital <= 0:
            errors.append("初始资金必须大于0")

        if config.commission_rate < 0 or config.commission_rate > 1:
            errors.append("手续费率必须在0 - 1之间")

        if config.slippage_rate < 0 or config.slippage_rate > 1:
            errors.append("滑点率必须在0 - 1之间")

        if config.memory_limit_gb <= 0:
            errors.append("内存限制必须大于0")

        elif config_type == "system":
            config = self.system_config

        if config.environment not in ["development", "staging", "production"]:
            errors.append("环境必须是development、staging或production")

        return errors

    def export_config_template(self, config_type: str, template_file: str):
        """导出配置模板"""
        if config_type == "backtest":
            config_data = asdict(self.backtest_config)
        elif config_type == "system":
            config_data = asdict(self.system_config)
        else:
            config_data = {}

        try:
            with open(template_file, 'w', encoding='utf - 8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)

            logger.info(f"配置模板已导出: {template_file}")

        except Exception as e:
            logger.error(f"导出配置模板失败: {e}")


class DeploymentManager:

    """部署管理器"""

    def __init__(self, config_manager: ConfigManager):

        self.config_manager = config_manager
        self.deployment_configs = {}

    def create_deployment_config(self, environment: str, config_data: Dict[str, Any]):
        """创建部署配置"""
        deployment_config = {
            'environment': environment,
            'timestamp': datetime.now().isoformat(),
            'config': config_data,
            'status': 'created'
        }

        self.deployment_configs[environment] = deployment_config
        logger.info(f"部署配置已创建: {environment}")

    def validate_deployment(self, environment: str) -> List[str]:
        """验证部署配置"""
        errors = []

        if environment not in self.deployment_configs:
            errors.append(f"环境 {environment} 的部署配置不存在")
            return errors

        config = self.deployment_configs[environment]['config']

        # 验证必要配置
        required_fields = ['initial_capital', 'data_source', 'symbols']
        for field in required_fields:
            if field not in config or config[field] is None:
                errors.append(f"缺少必要配置: {field}")

        # 验证数据源
        if config.get('data_source') == 'api' and not config.get('api_endpoint'):
            errors.append("API数据源需要配置api_endpoint")

        return errors

    def deploy_config(self, environment: str) -> bool:
        """部署配置"""
        errors = self.validate_deployment(environment)
        if errors:
            logger.error(f"部署验证失败: {errors}")
            return False

        try:
            # 更新系统配置
            deployment_config = self.deployment_configs[environment]
            self.config_manager.update_config("system", deployment_config['config'])

            # 保存配置
            self.config_manager.save_system_config()
            self.config_manager.save_backtest_config()

            # 更新状态
            deployment_config['status'] = 'deployed'
            deployment_config['deployed_at'] = datetime.now().isoformat()

            logger.info(f"配置已部署到环境: {environment}")
            return True

        except Exception as e:
            logger.error(f"部署配置失败: {e}")
            return False

    def rollback_deployment(self, environment: str) -> bool:
        """回滚部署"""
        try:
            # 恢复到默认配置
            self.config_manager._load_default_configs()

            # 保存配置
            self.config_manager.save_system_config()
            self.config_manager.save_backtest_config()

            if environment in self.deployment_configs:
                self.deployment_configs[environment]['status'] = 'rolled_back'

            logger.info(f"配置已回滚: {environment}")
            return True

        except Exception as e:
            logger.error(f"回滚配置失败: {e}")
            return False


# 全局配置管理器实例
config_manager = ConfigManager()
deployment_manager = DeploymentManager(config_manager)


def load_config(config_file: str, config_type: str = "backtest"):
    """加载配置"""
    if config_type == "backtest":
        config_manager.load_backtest_config(config_file)
    elif config_type == "system":
        config_manager.load_system_config(config_file)
    else:
        config_manager.load_custom_config(config_type, config_file)


def save_config(config_file: str = None, config_type: str = "backtest"):
    """保存配置"""
    if config_type == "backtest":
        config_manager.save_backtest_config(config_file)
    elif config_type == "system":
        config_manager.save_system_config(config_file)


def get_config(config_type: str = "backtest") -> Dict[str, Any]:
    """获取配置"""
    return config_manager.get_config(config_type)


def update_config(config_type: str, updates: Dict[str, Any]):
    """更新配置"""
    config_manager.update_config(config_type, updates)


def validate_config(config_type: str = "backtest") -> List[str]:
    """验证配置"""
    return config_manager.validate_config(config_type)


def create_deployment(environment: str, config_data: Dict[str, Any]):
    """创建部署配置"""
    deployment_manager.create_deployment_config(environment, config_data)


def deploy_to_environment(environment: str) -> bool:
    """部署到环境"""
    return deployment_manager.deploy_config(environment)


def rollback_deployment(environment: str) -> bool:
    """回滚部署"""
    return deployment_manager.rollback_deployment(environment)

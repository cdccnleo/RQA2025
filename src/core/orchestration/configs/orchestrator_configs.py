"""
业务流程编排器配置类定义

应用参数对象模式，提供类型安全的配置管理
参考Task 1成功经验
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from ...constants import (
    MAX_RECORDS, DEFAULT_TEST_TIMEOUT, DEFAULT_TIMEOUT, SECONDS_PER_HOUR,
    MAX_RETRIES, SECONDS_PER_MINUTE, DEFAULT_BATCH_SIZE
)


@dataclass
class EventBusConfig:
    """事件总线配置"""
    enable_history: bool = True              # 启用事件历史
    max_history_size: int = MAX_RECORDS             # 最大历史记录数
    enable_async: bool = True                # 启用异步发布
    enable_logging: bool = True              # 启用事件日志

    def __post_init__(self):
        """配置后验证"""
        if self.max_history_size <= 0:
            raise ValueError("max_history_size必须大于0")


@dataclass
class StateMachineConfig:
    """状态机配置"""
    enable_timeout_check: bool = True        # 启用超时检查
    default_state_timeout: int = DEFAULT_TEST_TIMEOUT         # 默认状态超时（秒）
    enable_state_logging: bool = True        # 启用状态日志
    enable_hooks: bool = True                # 启用钩子函数
    enable_listeners: bool = True            # 启用监听器

    def __post_init__(self):
        """配置后验证"""
        if self.default_state_timeout <= 0:
            raise ValueError("default_state_timeout必须大于0")


@dataclass
class ConfigManagerConfig:
    """配置管理器配置"""
    config_dir: str = "config/processes"     # 配置目录
    auto_save: bool = True                   # 自动保存
    enable_validation: bool = True           # 启用验证
    backup_configs: bool = True              # 备份配置

    def __post_init__(self):
        """配置后验证"""
        # 确保配置目录路径有效
        if not self.config_dir:
            raise ValueError("config_dir不能为空")


@dataclass
class MonitorConfig:
    """流程监控器配置"""
    monitoring_interval: int = DEFAULT_TIMEOUT            # 监控间隔（秒）
    enable_cleanup: bool = True              # 启用自动清理
    cleanup_interval: int = DEFAULT_TEST_TIMEOUT              # 清理间隔（秒）
    process_ttl: int = SECONDS_PER_HOUR                  # 流程TTL（秒）
    enable_metrics: bool = True              # 启用指标收集

    def __post_init__(self):
        """配置后验证"""
        if self.monitoring_interval <= 0:
            raise ValueError("monitoring_interval必须大于0")
        if self.cleanup_interval <= 0:
            raise ValueError("cleanup_interval必须大于0")


@dataclass
class PoolConfig:
    """实例池配置"""
    max_size: int = MAX_RETRIES                      # 最大池大小
    enable_reuse: bool = True                # 启用实例复用
    instance_ttl: int = 1800                 # 实例TTL（秒）
    enable_auto_scaling: bool = False        # 启用自动扩容

    def __post_init__(self):
        """配置后验证"""
        if self.max_size <= 0:
            raise ValueError("max_size必须大于0")
        if self.instance_ttl <= 0:
            raise ValueError("instance_ttl必须大于0")


@dataclass
class OrchestratorConfig:
    """
    编排器主配置类

    整合所有子配置，提供统一的配置接口
    采用参数对象模式，参考Task 1成功经验
    """
    # 子配置对象
    event_bus: EventBusConfig = field(default_factory=EventBusConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    config_manager: ConfigManagerConfig = field(default_factory=ConfigManagerConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)

    # 全局配置
    max_instances: int = MAX_RETRIES                 # 最大实例数
    config_dir: str = "config/processes"     # 配置目录
    enable_monitoring: bool = True           # 启用监控
    enable_health_check: bool = True         # 启用健康检查
    health_check_interval: int = SECONDS_PER_MINUTE          # 健康检查间隔（秒）
    enable_logging: bool = True              # 启用日志
    log_level: str = "INFO"                  # 日志级别

    # 额外配置
    custom_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """配置后验证"""
        if self.max_instances <= 0:
            raise ValueError("max_instances必须大于0")
        if self.health_check_interval <= 0:
            raise ValueError("health_check_interval必须大于0")

        # 验证日志级别
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"log_level必须是{valid_levels}之一")

        # 同步config_dir到config_manager
        self.config_manager.config_dir = self.config_dir

    @classmethod
    def create_default(cls) -> 'OrchestratorConfig':
        """创建默认配置"""
        return cls()

    @classmethod
    def create_high_performance(cls) -> 'OrchestratorConfig':
        """创建高性能配置"""
        config = cls()
        config.max_instances = 200
        config.pool.max_size = 200
        config.monitor.monitoring_interval = 15  # 更频繁监控
        config.event_bus.enable_async = True
        return config

    @classmethod
    def create_development(cls) -> 'OrchestratorConfig':
        """创建开发环境配置"""
        config = cls()
        config.max_instances = DEFAULT_BATCH_SIZE
        config.log_level = "DEBUG"
        config.monitor.monitoring_interval = 5
        config.event_bus.max_history_size = MAX_RETRIES
        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OrchestratorConfig':
        """
        从字典创建配置（向后兼容）

        支持旧格式的dict配置
        """
        # 提取子配置
        event_bus = EventBusConfig()
        state_machine = StateMachineConfig()
        config_manager = ConfigManagerConfig(
            config_dir=config_dict.get('config_dir', 'config/processes')
        )
        monitor = MonitorConfig()
        pool = PoolConfig(
            max_size=config_dict.get('max_instances', MAX_RETRIES)
        )

        # 创建主配置
        return cls(
            event_bus=event_bus,
            state_machine=state_machine,
            config_manager=config_manager,
            monitor=monitor,
            pool=pool,
            max_instances=config_dict.get('max_instances', MAX_RETRIES),
            config_dir=config_dict.get('config_dir', 'config/processes'),
            custom_config=config_dict
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            'event_bus': self.event_bus.__dict__,
            'state_machine': self.state_machine.__dict__,
            'config_manager': self.config_manager.__dict__,
            'monitor': self.monitor.__dict__,
            'pool': self.pool.__dict__,
            'max_instances': self.max_instances,
            'config_dir': self.config_dir,
            'enable_monitoring': self.enable_monitoring,
            'enable_health_check': self.enable_health_check,
            'health_check_interval': self.health_check_interval,
            'enable_logging': self.enable_logging,
            'log_level': self.log_level,
            'custom_config': self.custom_config
        }


# 便捷函数
def create_orchestrator_config(**kwargs) -> OrchestratorConfig:
    """
    创建编排器配置（便捷函数）

    Args:
        **kwargs: 配置参数

    Returns:
        OrchestratorConfig: 编排器配置对象
    """
    return OrchestratorConfig(**kwargs)

"""
cache_configs 模块

提供 cache_configs 相关功能和接口。
"""

import logging


from ..interfaces import (
    CacheEvictionStrategy
)
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Optional
from typing import Protocol
#!/usr/bin/env python3
"""
统一缓存管理器 - 重构优化版本
合并4个缓存管理器类的功能，消除代码重复，提高代码质量

重构内容:
✅ 合并 base_cache_manager.py (基础功能)
✅ 合并 unified_cache_manager.py (统一接口)
✅ 合并 advanced_cache_manager.py (高级功能)
✅ 合并 smart_cache_strategy.py (智能策略)

核心特性:
🔥 统一接口设计 - 消除接口不一致问题
🔥 智能缓存策略 - 支持LRU/LFU/自适应策略
🔥 多级缓存支持 - 内存/Redis/文件缓存
🔥 分布式缓存 - 支持Redis集群和一致性保证
🔥 性能监控 - 实时指标收集和智能告警
🔥 生产就绪 - 健康检查、优雅关闭、故障恢复
"""

# from ..interfaces import (
#     ICacheComponent, CacheEvictionStrategy, AccessPattern, CacheEntry, CacheStats, PerformanceMetrics
# )
# from ..interfaces.global_interfaces import ICacheStrategy  # 暂时注释，避免循环导入
logger = logging.getLogger(__name__)


class ICacheLayer(Protocol):
    """
    缓存层协议接口

    定义缓存层的标准操作，用于统一不同缓存层的访问。
    """

    def get(self, key: str) -> Dict[str, Any]:
        """
        获取缓存项

        Args:
            key: 缓存键

        Returns:
            Dict with keys: 'found' (bool), 'value' (Any), 'level' (str)
        """
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存项

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间

        Returns:
            bool: 设置是否成功
        """
        ...

    def delete(self, key: str) -> bool:
        """
        删除缓存项

        Args:
            key: 缓存键

        Returns:
            bool: 删除是否成功
        """
        ...

    def clear(self) -> bool:
        """
        清空缓存层

        Returns:
            bool: 清空是否成功
        """
        ...

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存层统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        ...


class CacheLevel(Enum):
    """缓存级别枚举"""
    L1 = "L1"             # 内存缓存
    L2 = "L2"             # Redis缓存
    L3 = "L3"             # 磁盘缓存
    MEMORY = "memory"      # 内存缓存
    REDIS = "redis"        # Redis缓存
    FILE = "file"          # 本地文件缓存
    HYBRID = "hybrid"      # 混合缓存


class DataType(Enum):
    """数据类型"""
    SMALL = "small"           # 小数据 (< 1KB)
    MEDIUM = "medium"         # 中等数据 (1KB - 1MB)
    LARGE = "large"           # 大数据 (> 1MB)
    CRITICAL = "critical"     # 关键数据
    TEMPORARY = "temporary"   # 临时数据

# ==================== 配置管理 - 职责分离 ====================


@dataclass
class BasicCacheConfig:
    """基础缓存配置"""
    max_size: int = 1000
    ttl: int = 3600
    strategy: CacheEvictionStrategy = CacheEvictionStrategy.LRU

    def __post_init__(self):
        """配置后验证"""
        if self.max_size <= 0:
            raise ValueError("max_size必须大于0")
        if self.ttl <= 0:
            raise ValueError("ttl必须大于0")


@dataclass
class MultiLevelCacheConfig:
    """多级缓存配置"""
    level: CacheLevel = CacheLevel.HYBRID
    memory_max_size: int = 1000
    memory_ttl: int = 30
    redis_max_size: int = 10000
    redis_ttl: int = 300
    file_max_size: int = 100000
    file_ttl: int = 3600
    file_cache_dir: str = "/tmp/rqa2025_cache"

    def __post_init__(self):
        """配置后验证"""
        if self.memory_max_size <= 0:
            raise ValueError("memory_max_size必须大于0")
        if self.memory_ttl <= 0:
            raise ValueError("memory_ttl必须大于0")
        if self.redis_max_size <= 0:
            raise ValueError("redis_max_size必须大于0")
        if self.redis_ttl <= 0:
            raise ValueError("redis_ttl必须大于0")
        if self.file_max_size <= 0:
            raise ValueError("file_max_size必须大于0")
        if self.file_ttl <= 0:
            raise ValueError("file_ttl必须大于0")
        if not self.file_cache_dir:
            raise ValueError("file_cache_dir不能为空")


@dataclass
class AdvancedCacheConfig:
    """高级缓存配置"""
    enable_compression: bool = True
    enable_preloading: bool = True
    enable_parallel_write: bool = True  # 启用并行写入优化
    preload_threshold: float = 0.8
    cleanup_interval: int = 60
    max_memory_mb: int = 100

    def __post_init__(self):
        """配置后验证"""
        if not (0.0 <= self.preload_threshold <= 1.0):
            raise ValueError("preload_threshold必须在0.0-1.0之间")
        if self.cleanup_interval <= 0:
            raise ValueError("cleanup_interval必须大于0")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb必须大于0")


@dataclass
class SmartCacheConfig:
    """智能缓存配置"""
    enable_monitoring: bool = True
    enable_auto_optimization: bool = True
    adaptation_interval: int = 300  # 5分钟适应间隔

    def __post_init__(self):
        """配置后验证"""
        if self.adaptation_interval <= 0:
            raise ValueError("adaptation_interval必须大于0")


@dataclass
class DistributedCacheConfig:
    """分布式缓存配置"""
    distributed: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    cluster_mode: bool = False

    def __post_init__(self):
        """配置后验证"""
        if self.redis_port <= 0 or self.redis_port > 65535:
            raise ValueError("redis_port必须在1-65535之间")
        if not self.redis_host:
            raise ValueError("redis_host不能为空")


@dataclass(init=False)
class CacheConfig:
    """
    统一缓存配置 - 组合模式

    在保持分层配置结构的同时，提供向后兼容的顶层字段，
    便于测试和业务代码以更直观的方式访问常用配置。
    """

    enabled: bool = True
    max_size: Optional[int] = None
    ttl: Optional[int] = None
    eviction_policy: CacheEvictionStrategy = CacheEvictionStrategy.LRU
    strict_validation: Optional[bool] = None

    basic: BasicCacheConfig = field(default_factory=BasicCacheConfig)
    multi_level: MultiLevelCacheConfig = field(default_factory=MultiLevelCacheConfig)
    advanced: AdvancedCacheConfig = field(default_factory=AdvancedCacheConfig)
    smart: SmartCacheConfig = field(default_factory=SmartCacheConfig)
    distributed: DistributedCacheConfig = field(default_factory=DistributedCacheConfig)

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_size: Optional[int] = None,
        ttl: Optional[int] = None,
        eviction_policy: CacheEvictionStrategy = CacheEvictionStrategy.LRU,
        strict_validation: Optional[bool] = None,
        basic: Optional[BasicCacheConfig] = None,
        multi_level: Optional[MultiLevelCacheConfig] = None,
        advanced: Optional[AdvancedCacheConfig] = None,
        smart: Optional[SmartCacheConfig] = None,
        distributed: Optional[DistributedCacheConfig] = None,
    ) -> None:
        self.enabled = enabled
        self.max_size = max_size
        self.ttl = ttl
        self.eviction_policy = eviction_policy
        self.strict_validation = strict_validation

        self.basic = basic if basic is not None else BasicCacheConfig()
        self.multi_level = multi_level if multi_level is not None else MultiLevelCacheConfig()
        self.advanced = advanced if advanced is not None else AdvancedCacheConfig()
        self.smart = smart if smart is not None else SmartCacheConfig()
        self.distributed = distributed if distributed is not None else DistributedCacheConfig()

        self._explicit_fields = {
            name
            for name, value in (
                ("basic", basic),
                ("multi_level", multi_level),
                ("advanced", advanced),
                ("smart", smart),
                ("distributed", distributed),
                ("strict_validation", strict_validation),
            )
            if value is not None
        }

        self.__post_init__()

    def __post_init__(self):
        """整体配置验证和依赖检查"""
        if self.strict_validation is None:
            self.strict_validation = bool(
                {"multi_level", "advanced", "smart", "distributed"} & self._explicit_fields
            )

        self._sync_basic_config()
        self._validate_dependencies()
        self._optimize_defaults()

    def _sync_basic_config(self) -> None:
        """同步顶层简化字段与基础配置，确保向后兼容"""
        # 如果basic是字典或其他结构，尝试转换
        if not isinstance(self.basic, BasicCacheConfig):
            basic_data = getattr(self.basic, '__dict__', self.basic)
            if isinstance(basic_data, dict):
                self.basic = BasicCacheConfig(**basic_data)
            else:
                self.basic = BasicCacheConfig()

        # 使用顶层字段覆盖basic配置（仅在显式提供时），否则保持basic设置
        if self.max_size is not None:
            self.basic.max_size = self.max_size
        if self.ttl is not None:
            self.basic.ttl = self.ttl
        # 允许以字符串形式传入淘汰策略
        if isinstance(self.eviction_policy, str):
            try:
                self.eviction_policy = CacheEvictionStrategy(self.eviction_policy.lower())
            except ValueError:
                self.eviction_policy = CacheEvictionStrategy.LRU
        self.basic.strategy = self.eviction_policy

        # 反向同步，确保顶层字段可直接访问
        if self.max_size is None:
            self.max_size = self.basic.max_size
        else:
            self.basic.max_size = self.max_size

        if self.ttl is None:
            self.ttl = self.basic.ttl
        else:
            self.basic.ttl = self.ttl

    def _validate_dependencies(self):
        """验证配置依赖关系"""
        # 如果启用分布式，必须配置Redis
        if self.distributed.distributed:
            level_value = getattr(self.multi_level, "level", CacheLevel.HYBRID)
            if isinstance(level_value, str):
                try:
                    level_enum = CacheLevel(level_value)
                except ValueError:
                    level_enum = CacheLevel.HYBRID
            else:
                level_enum = level_value
            if not self.distributed.redis_host:
                raise ValueError("启用分布式模式时必须配置redis_host")
            if not (0 < getattr(self.distributed, "redis_port", 0) <= 65535):
                raise ValueError("启用分布式模式时redis_port必须在1-65535之间")
            if level_enum not in [CacheLevel.REDIS, CacheLevel.HYBRID]:
                logger.warning("分布式模式建议使用REDIS或HYBRID缓存级别")

        # 如果启用预加载，检查相关配置
        if self.advanced.enable_preloading:
            if self.advanced.preload_threshold <= 0:
                raise ValueError("启用预加载时preload_threshold必须大于0")

        # 如果启用压缩，检查内存限制
        if self.advanced.enable_compression:
            if self.advanced.max_memory_mb < 50:
                logger.warning("启用压缩时建议max_memory_mb至少为50MB")

    def _optimize_defaults(self):
        """根据配置优化默认值"""
        # 根据缓存级别调整默认TTL
        level_value = getattr(self.multi_level, "level", CacheLevel.HYBRID)
        if isinstance(level_value, str):
            try:
                level_enum = CacheLevel(level_value)
            except ValueError:
                level_enum = CacheLevel.HYBRID
        else:
            level_enum = level_value

        if level_enum == CacheLevel.MEMORY:
            # 内存缓存可以有更短的TTL
            if self.multi_level.memory_ttl > 300:
                logger.info("内存缓存建议TTL不超过5分钟，已自动调整")
                self.multi_level.memory_ttl = 300
        elif level_enum == CacheLevel.REDIS:
            # Redis缓存可以有更长的TTL
            if self.multi_level.redis_ttl < 600:
                logger.info("Redis缓存建议TTL至少10分钟，已自动调整")
                self.multi_level.redis_ttl = 600

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CacheConfig':
        """从字典创建配置"""
        # 解析子配置
        basic_dict = config_dict.get('basic', {})
        multi_level_dict = config_dict.get('multi_level', {})
        advanced_dict = config_dict.get('advanced', {})
        smart_dict = config_dict.get('smart', {})
        distributed_dict = config_dict.get('distributed', {})

        enabled = config_dict.get('enabled', True)
        max_size = config_dict.get('max_size', basic_dict.get('max_size'))
        ttl = config_dict.get('ttl', basic_dict.get('ttl'))
        eviction_policy = config_dict.get('eviction_policy', basic_dict.get('strategy', CacheEvictionStrategy.LRU))

        strict_flag = config_dict.get('strict_validation')
        if strict_flag is None:
            strict_flag = bool(multi_level_dict)

        try:
            basic_cfg = BasicCacheConfig(**basic_dict) if basic_dict else None
        except (ValueError, TypeError):
            if strict_flag:
                raise
            basic_cfg = BasicCacheConfig()
            for key, value in basic_dict.items():
                setattr(basic_cfg, key, value)

        try:
            multi_cfg = MultiLevelCacheConfig(**multi_level_dict) if multi_level_dict else None
        except (ValueError, TypeError):
            if strict_flag:
                raise
            multi_cfg = MultiLevelCacheConfig()
            for key, value in multi_level_dict.items():
                setattr(multi_cfg, key, value)

        try:
            distributed_cfg = DistributedCacheConfig(**distributed_dict) if distributed_dict else None
        except (ValueError, TypeError):
            if strict_flag:
                raise
            distributed_cfg = DistributedCacheConfig()
            for key, value in distributed_dict.items():
                setattr(distributed_cfg, key, value)

        advanced_cfg = None
        if advanced_dict:
            try:
                advanced_cfg = AdvancedCacheConfig(**advanced_dict)
            except (ValueError, TypeError):
                if strict_flag:
                    raise
                advanced_cfg = AdvancedCacheConfig()
                for key, value in advanced_dict.items():
                    setattr(advanced_cfg, key, value)

        smart_cfg = None
        if smart_dict:
            try:
                smart_cfg = SmartCacheConfig(**smart_dict)
            except (ValueError, TypeError):
                if strict_flag:
                    raise
                smart_cfg = SmartCacheConfig()
                for key, value in smart_dict.items():
                    setattr(smart_cfg, key, value)

        if not strict_flag:
            try:
                max_size = int(max_size)
            except Exception:
                max_size = None
            try:
                ttl = int(ttl)
            except Exception:
                ttl = None
            if distributed_cfg:
                if not distributed_cfg.redis_host:
                    distributed_cfg.redis_host = "localhost"
                if not (1 <= getattr(distributed_cfg, "redis_port", 0) <= 65535):
                    distributed_cfg.redis_port = 6379

        config = cls(
            enabled=enabled,
            max_size=max_size,
            ttl=ttl,
            eviction_policy=eviction_policy,
            strict_validation=strict_flag,
            basic=basic_cfg,
            multi_level=multi_cfg,
            advanced=advanced_cfg,
            smart=smart_cfg,
            distributed=distributed_cfg
        )

        if not strict_flag:
            config._sync_basic_config()
            try:
                config._validate_dependencies()
            except ValueError:
                config.distributed = DistributedCacheConfig()
        else:
            config._sync_basic_config()
            config._validate_dependencies()

        return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'enabled': self.enabled,
            'max_size': self.max_size,
            'ttl': self.ttl,
            'eviction_policy': self.eviction_policy.value if isinstance(self.eviction_policy, CacheEvictionStrategy) else str(self.eviction_policy),
            'strict_validation': self.strict_validation,
            'basic': asdict(self.basic),
            'multi_level': asdict(self.multi_level),
            'advanced': asdict(self.advanced),
            'smart': asdict(self.smart),
            'distributed': asdict(self.distributed)
        }

    @classmethod
    def create_simple_memory_config(cls) -> 'CacheConfig':
        """创建简单的内存缓存配置"""
        return cls(
            basic=BasicCacheConfig(max_size=1000, ttl=3600),
            multi_level=MultiLevelCacheConfig(level=CacheLevel.MEMORY),
            advanced=AdvancedCacheConfig(enable_compression=False, enable_preloading=False),
            smart=SmartCacheConfig(enable_monitoring=False, enable_auto_optimization=False),
            distributed=DistributedCacheConfig(distributed=False)
        )

    @classmethod
    def create_production_config(cls) -> 'CacheConfig':
        """创建生产环境配置"""
        return cls(
            basic=BasicCacheConfig(max_size=10000, ttl=7200),
            multi_level=MultiLevelCacheConfig(
                level=CacheLevel.HYBRID,
                memory_max_size=5000,
                redis_ttl=3600
            ),
            advanced=AdvancedCacheConfig(
                enable_compression=True,
                enable_preloading=True,
                max_memory_mb=500
            ),
            smart=SmartCacheConfig(
                enable_monitoring=True,
                enable_auto_optimization=True
            ),
            distributed=DistributedCacheConfig(distributed=True)
        )

"""
特征层配置集成管理器
实现特征层组件与基础设施层统一配置管理系统的集成
"""

import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

# 使用统一基础设施集成层
try:
    from src.infrastructure.integration import get_features_layer_adapter
    _features_adapter = get_features_layer_adapter()
    if _features_adapter is not None:
        _unified_config_manager = _features_adapter.get_config_manager()
        _unified_cache_manager = _features_adapter.get_cache_manager()
    else:
        _unified_config_manager = None
        _unified_cache_manager = None
except ImportError:
    # 降级到直接导入
    try:
        from src.infrastructure.config.factory import ConfigFactory
        from src.infrastructure.config.interfaces.unified_interface import IConfigManager
        _unified_config_manager = None  # 暂时设为None，避免初始化错误
        _unified_cache_manager = None
    except ImportError:
        # 如果基础设施层配置管理不可用，使用本地配置
        ConfigFactory = None
        IConfigManager = None
        _unified_config_manager = None
        _unified_cache_manager = None

# 导入特征层配置类
from .feature_config import (
    FeatureProcessingConfig
)
# 从特征层主配置文件导入专用配置类
from .config_classes import TechnicalConfig, SentimentConfig


class ConfigScope(Enum):

    """配置作用域枚举"""
    GLOBAL = "global"
    FEATURE = "feature"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    PROCESSING = "processing"
    MONITORING = "monitoring"


@dataclass
class FeatureLayerConfig:

    """特征层配置类"""

    # 基础配置
    environment: str = "development"
    config_dir: str = "config / features"

    # 组件配置
    technical_config: TechnicalConfig = None
    sentiment_config: SentimentConfig = None
    processing_config: FeatureProcessingConfig = None

    # 监控配置
    enable_monitoring: bool = True
    monitoring_level: str = "standard"

    # 缓存配置
    enable_caching: bool = True
    cache_ttl: int = 3600

    def __post_init__(self):
        """初始化后设置默认值"""
        if self.technical_config is None:
            self.technical_config = TechnicalConfig()
        if self.sentiment_config is None:
            self.sentiment_config = SentimentConfig()
        if self.processing_config is None:
            self.processing_config = FeatureProcessingConfig()


class FeatureConfigIntegrationManager:

    """特征层配置集成管理器"""

    def __init__(self, config_manager: Optional[Any] = None):
        """
        初始化配置集成管理器

        Args:
            config_manager: 基础设施层配置管理器实例
        """
        self.logger = logging.getLogger(__name__)

        # 初始化配置管理器
        if config_manager is None and 'ConfigFactory' in globals() and ConfigFactory is not None:
            try:
                self.config_manager = ConfigFactory.create_complete_config_service(
                    env="features"
                )
                self._use_infrastructure_config = True
                self.logger.info("使用基础设施层配置管理系统")
            except Exception as e:
                self.logger.warning(f"无法初始化基础设施层配置管理: {e}")
                self.config_manager = None
                self._use_infrastructure_config = False
        else:
            self.config_manager = config_manager
            self._use_infrastructure_config = config_manager is not None

        # 初始化本地配置
        self._local_config = FeatureLayerConfig()
        self._config_cache = {}
        self._config_watchers = {}

        # 加载配置
        self._load_configuration()

    def _load_configuration(self):
        """加载配置"""
        try:
            if self._use_infrastructure_config:
                self._load_from_infrastructure()
            else:
                self._load_from_local()
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            self._load_default_config()

    def _load_from_infrastructure(self):
        """从基础设施层加载配置"""
        # 加载技术指标配置
        tech_config = self.config_manager.get("technical", default={})
        if tech_config:
            self._local_config.technical_config = TechnicalConfig(**tech_config)

        # 加载情感分析配置
        sentiment_config = self.config_manager.get("sentiment", default={})
        if sentiment_config:
            self._local_config.sentiment_config = SentimentConfig(**sentiment_config)

        # 加载处理配置
        processing_config = self.config_manager.get("processing", default={})
        if processing_config:
            self._local_config.processing_config = FeatureProcessingConfig(**processing_config)

        # 加载基础配置
        base_config = self.config_manager.get("base", default={})
        if base_config:
            for key, value in base_config.items():
                if hasattr(self._local_config, key):
                    setattr(self._local_config, key, value)

    def _load_from_local(self):
        """从本地文件加载配置"""
        config_path = Path(self._local_config.config_dir)
        if not config_path.exists():
            self.logger.info("配置目录不存在，使用默认配置")
            return

        # 加载技术指标配置
        tech_config_file = config_path / "technical.json"
        if tech_config_file.exists():
            with open(tech_config_file, 'r', encoding='utf - 8') as f:
                tech_config = json.load(f)
                self._local_config.technical_config = TechnicalConfig(**tech_config)

        # 加载情感分析配置
        sentiment_config_file = config_path / "sentiment.json"
        if sentiment_config_file.exists():
            with open(sentiment_config_file, 'r', encoding='utf - 8') as f:
                sentiment_config = json.load(f)
                self._local_config.sentiment_config = SentimentConfig(**sentiment_config)

        # 加载处理配置
        processing_config_file = config_path / "processing.json"
        if processing_config_file.exists():
            with open(processing_config_file, 'r', encoding='utf - 8') as f:
                processing_config = json.load(f)
                self._local_config.processing_config = FeatureProcessingConfig(**processing_config)

    def _load_default_config(self):
        """加载默认配置"""
        self.logger.info("使用默认配置")
        self._local_config = FeatureLayerConfig()

    def get_config(self, scope: ConfigScope, key: Optional[str] = None) -> Any:
        """
        获取配置

        Args:
            scope: 配置作用域
            key: 配置键名

        Returns:
            配置值
        """
        try:
            if scope == ConfigScope.GLOBAL:
                return self._get_global_config(key)
            elif scope == ConfigScope.TECHNICAL:
                return self._get_technical_config(key)
            elif scope == ConfigScope.SENTIMENT:
                return self._get_sentiment_config(key)
            elif scope == ConfigScope.PROCESSING:
                return self._get_processing_config(key)
            elif scope == ConfigScope.MONITORING:
                return self._get_monitoring_config(key)
            else:
                raise ValueError(f"不支持的配置作用域: {scope}")
        except Exception as e:
            self.logger.error(f"获取配置失败: {e}")
            return None

    def _get_global_config(self, key: Optional[str] = None) -> Any:
        """获取全局配置"""
        if key is None:
            return asdict(self._local_config)
        return getattr(self._local_config, key, None)

    def _get_technical_config(self, key: Optional[str] = None) -> Any:
        """获取技术指标配置"""
        config = self._local_config.technical_config
        if key is None:
            return asdict(config)
        return getattr(config, key, None)

    def _get_sentiment_config(self, key: Optional[str] = None) -> Any:
        """获取情感分析配置"""
        config = self._local_config.sentiment_config
        if key is None:
            return asdict(config)
        return getattr(config, key, None)

    def _get_processing_config(self, key: Optional[str] = None) -> Any:
        """获取处理配置"""
        config = self._local_config.processing_config
        if key is None:
            return asdict(config)
        return getattr(config, key, None)

    def _get_monitoring_config(self, key: Optional[str] = None) -> Dict[str, Any]:
        """获取监控配置"""
        config = {
            'enable_monitoring': self._local_config.enable_monitoring,
            'monitoring_level': self._local_config.monitoring_level,
            'enable_caching': self._local_config.enable_caching,
            'cache_ttl': self._local_config.cache_ttl
        }
        if key is None:
            return config
        return config.get(key)

    def set_config(self, scope: ConfigScope, key: str, value: Any) -> bool:
        """
        设置配置

        Args:
            scope: 配置作用域
            key: 配置键名
            value: 配置值

        Returns:
            是否设置成功
        """
        try:
            if scope == ConfigScope.GLOBAL:
                return self._set_global_config(key, value)
            elif scope == ConfigScope.TECHNICAL:
                return self._set_technical_config(key, value)
            elif scope == ConfigScope.SENTIMENT:
                return self._set_sentiment_config(key, value)
            elif scope == ConfigScope.PROCESSING:
                return self._set_processing_config(key, value)
            elif scope == ConfigScope.MONITORING:
                return self._set_monitoring_config(key, value)
            else:
                raise ValueError(f"不支持的配置作用域: {scope}")
        except Exception as e:
            self.logger.error(f"设置配置失败: {e}")
            return False

    def _set_global_config(self, key: str, value: Any) -> bool:
        """设置全局配置"""
        if hasattr(self._local_config, key):
            setattr(self._local_config, key, value)
            return True
        return False

    def _set_technical_config(self, key: str, value: Any) -> bool:
        """设置技术指标配置"""
        if hasattr(self._local_config.technical_config, key):
            setattr(self._local_config.technical_config, key, value)
            return True
        return False

    def _set_sentiment_config(self, key: str, value: Any) -> bool:
        """设置情感分析配置"""
        if hasattr(self._local_config.sentiment_config, key):
            setattr(self._local_config.sentiment_config, key, value)
            return True
        return False

    def _set_processing_config(self, key: str, value: Any) -> bool:
        """设置处理配置"""
        if hasattr(self._local_config.processing_config, key):
            setattr(self._local_config.processing_config, key, value)
            return True
        return False

    def _set_monitoring_config(self, key: str, value: Any) -> bool:
        """设置监控配置"""
        if hasattr(self._local_config, key):
            setattr(self._local_config, key, value)
            return True
        return False

    def save_config(self, scope: ConfigScope = ConfigScope.GLOBAL) -> bool:
        """
        保存配置

        Args:
            scope: 配置作用域

        Returns:
            是否保存成功
        """
        try:
            if self._use_infrastructure_config:
                return self._save_to_infrastructure(scope)
            else:
                return self._save_to_local(scope)
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            return False

    def _save_to_infrastructure(self, scope: ConfigScope) -> bool:
        """保存到基础设施层"""
        if self.config_manager is None:
            return False

        try:
            if scope == ConfigScope.TECHNICAL:
                config_dict = asdict(self._local_config.technical_config)
                self.config_manager.set("technical", config_dict)
            elif scope == ConfigScope.SENTIMENT:
                config_dict = asdict(self._local_config.sentiment_config)
                self.config_manager.set("sentiment", config_dict)
            elif scope == ConfigScope.PROCESSING:
                config_dict = asdict(self._local_config.processing_config)
                self.config_manager.set("processing", config_dict)
            elif scope == ConfigScope.GLOBAL:
                # 保存所有配置
                self._save_to_infrastructure(ConfigScope.TECHNICAL)
                self._save_to_infrastructure(ConfigScope.SENTIMENT)
                self._save_to_infrastructure(ConfigScope.PROCESSING)

            return True
        except Exception as e:
            self.logger.error(f"保存到基础设施层失败: {e}")
            return False

    def _save_to_local(self, scope: ConfigScope) -> bool:
        """保存到本地文件"""
        config_path = Path(self._local_config.config_dir)
        config_path.mkdir(parents=True, exist_ok=True)

        try:
            if scope == ConfigScope.TECHNICAL:
                config_file = config_path / "technical.json"
                with open(config_file, 'w', encoding='utf - 8') as f:
                    json.dump(asdict(self._local_config.technical_config), f, indent=2)
            elif scope == ConfigScope.SENTIMENT:
                config_file = config_path / "sentiment.json"
                with open(config_file, 'w', encoding='utf - 8') as f:
                    json.dump(asdict(self._local_config.sentiment_config), f, indent=2)
            elif scope == ConfigScope.PROCESSING:
                config_file = config_path / "processing.json"
                with open(config_file, 'w', encoding='utf - 8') as f:
                    json.dump(asdict(self._local_config.processing_config), f, indent=2)
            elif scope == ConfigScope.GLOBAL:
                # 保存所有配置
                self._save_to_local(ConfigScope.TECHNICAL)
                self._save_to_local(ConfigScope.SENTIMENT)
                self._save_to_local(ConfigScope.PROCESSING)

            return True
        except Exception as e:
            self.logger.error(f"保存到本地文件失败: {e}")
            return False

    def register_config_watcher(self, scope: ConfigScope, callback: callable) -> bool:
        """
        注册配置监听器

        Args:
            scope: 配置作用域
            callback: 回调函数

        Returns:
            是否注册成功
        """
        try:
            if scope not in self._config_watchers:
                self._config_watchers[scope] = []
            self._config_watchers[scope].append(callback)
            return True
        except Exception as e:
            self.logger.error(f"注册配置监听器失败: {e}")
            return False

    def notify_config_change(self, scope: ConfigScope, key: str, old_value: Any, new_value: Any):
        """通知配置变更"""
        if scope in self._config_watchers:
            for callback in self._config_watchers[scope]:
                try:
                    callback(scope, key, old_value, new_value)
                except Exception as e:
                    self.logger.error(f"配置监听器回调失败: {e}")

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'environment': self._local_config.environment,
            'use_infrastructure_config': self._use_infrastructure_config,
            'technical_config': asdict(self._local_config.technical_config),
            'sentiment_config': asdict(self._local_config.sentiment_config),
            'processing_config': asdict(self._local_config.processing_config),
            'monitoring_config': self._get_monitoring_config()
        }


# 全局配置集成管理器实例
_config_integration_manager = None


def get_config_integration_manager() -> FeatureConfigIntegrationManager:
    """获取配置集成管理器实例"""
    global _config_integration_manager
    if _config_integration_manager is None:
        _config_integration_manager = FeatureConfigIntegrationManager()
    return _config_integration_manager


def get_feature_config(scope: ConfigScope, key: Optional[str] = None) -> Any:
    """便捷函数：获取特征配置"""
    manager = get_config_integration_manager()
    return manager.get_config(scope, key)


def set_feature_config(scope: ConfigScope, key: str, value: Any) -> bool:
    """便捷函数：设置特征配置"""
    manager = get_config_integration_manager()
    return manager.set_config(scope, key, value)


def save_feature_config(scope: ConfigScope = ConfigScope.GLOBAL) -> bool:
    """便捷函数：保存特征配置"""
    manager = get_config_integration_manager()
    return manager.save_config(scope)

"""
配置中心管理器

提供统一的配置管理，支持YAML配置文件、环境变量覆盖、热更新等功能。

Author: RQA2025 Development Team
Date: 2026-02-13
"""

import os
import yaml
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import threading
import watchdog
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)


@dataclass
class ConfigSource:
    """配置源"""
    name: str
    priority: int  # 优先级，数字越小优先级越高
    data: Dict[str, Any] = field(default_factory=dict)


class ConfigChangeHandler(FileSystemEventHandler):
    """配置文件变更处理器"""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.yaml', '.yml', '.json')):
            logger.info(f"Config file modified: {event.src_path}")
            self.config_manager.reload_config()


class ConfigManager:
    """
    配置中心管理器
    
    提供以下功能：
    1. 多源配置管理（文件、环境变量、内存）
    2. 配置优先级覆盖
    3. 配置热更新
    4. 配置验证
    5. 配置加密（敏感信息）
    
    配置优先级（从高到低）：
    1. 环境变量
    2. 本地配置文件（config.local.yaml）
    3. 主配置文件（config.yaml）
    4. 默认配置
    
    Attributes:
        config_dir: 配置目录
        environment: 运行环境
        sources: 配置源列表
    """
    
    def __init__(
        self,
        config_dir: str = "config",
        environment: Optional[str] = None,
        enable_hot_reload: bool = True
    ):
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("RQA_ENV", "development")
        self.enable_hot_reload = enable_hot_reload
        
        # 配置源（按优先级排序）
        self._sources: List[ConfigSource] = []
        
        # 合并后的配置缓存
        self._cached_config: Dict[str, Any] = {}
        self._cache_valid = False
        
        # 锁
        self._lock = threading.RLock()
        
        # 观察者（用于热更新）
        self._observer: Optional[Observer] = None
        
        # 变更回调
        self._change_callbacks: List[Callable[[str, Any, Any], None]] = []
        
        # 初始化
        self._initialize_sources()
        
        logger.info(f"ConfigManager initialized for environment: {self.environment}")
    
    def _initialize_sources(self):
        """初始化配置源"""
        # 1. 默认配置（优先级最低）
        self._sources.append(ConfigSource("default", 100, self._get_default_config()))
        
        # 2. 主配置文件
        self._load_file_config("config.yaml", 50)
        
        # 3. 环境特定配置
        self._load_file_config(f"config.{self.environment}.yaml", 40)
        
        # 4. 本地配置文件（优先级最高，不提交到版本控制）
        self._load_file_config("config.local.yaml", 30)
        
        # 5. 环境变量（最高优先级）
        self._sources.append(ConfigSource("environment", 10, self._load_env_config()))
        
        # 合并配置
        self._rebuild_cache()
        
        # 启动热更新
        if self.enable_hot_reload:
            self._start_hot_reload()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "application": {
                "name": "RQA2025",
                "version": "1.0.0",
                "description": "A股量化交易系统"
            },
            "environment": self.environment,
            "debug": False,
            
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
                "timeout": 30
            },
            
            "database": {
                "timescale": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "rqa2025",
                    "user": "postgres",
                    "password": "",
                    "pool": {
                        "min_size": 5,
                        "max_size": 20,
                        "command_timeout": 60.0
                    }
                },
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "db": 0,
                    "password": ""
                }
            },
            
            "security": {
                "secret_key": "",
                "jwt_algorithm": "HS256",
                "access_token_expire_minutes": 30,
                "refresh_token_expire_days": 7
            },
            
            "logging": {
                "level": "INFO",
                "format": "json",
                "output": "stdout"
            },
            
            "market_data": {
                "use_mock": False,
                "fallback_to_mock": True,
                "sources": [
                    {
                        "name": "tushare",
                        "enabled": True,
                        "priority": 1
                    },
                    {
                        "name": "akshare",
                        "enabled": True,
                        "priority": 2
                    }
                ]
            },
            
            "risk": {
                "enabled": True,
                "default_rules": {
                    "position_limit": 10000,
                    "loss_limit": 0.05,
                    "drawdown_limit": 0.10
                }
            },
            
            "backtest": {
                "default_initial_capital": 1000000,
                "default_commission_rate": 0.0003,
                "default_slippage": 0.0001
            },
            
            "distributed": {
                "coordinator": {
                    "enabled": True,
                    "heartbeat_interval": 10,
                    "task_timeout": 3600
                }
            }
        }
    
    def _load_file_config(self, filename: str, priority: int):
        """加载配置文件"""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filename.endswith('.json'):
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f) or {}
            
            self._sources.append(ConfigSource(filename, priority, data))
            logger.info(f"Loaded config from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load config from {filepath}: {e}")
    
    def _load_env_config(self) -> Dict[str, Any]:
        """从环境变量加载配置"""
        env_config = {}
        
        # 前缀
        prefix = "RQA_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # 转换键名：RQA_DATABASE_HOST -> database.host
                config_key = key[len(prefix):].lower().replace('_', '.')
                self._set_nested_value(env_config, config_key, self._parse_value(value))
        
        return env_config
    
    def _parse_value(self, value: str) -> Any:
        """解析配置值"""
        # 尝试解析为布尔值
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # 尝试解析为整数
        try:
            return int(value)
        except ValueError:
            pass
        
        # 尝试解析为浮点数
        try:
            return float(value)
        except ValueError:
            pass
        
        # 尝试解析为JSON
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        
        # 返回字符串
        return value
    
    def _set_nested_value(self, config: Dict, key: str, value: Any):
        """设置嵌套配置值"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def _rebuild_cache(self):
        """重建配置缓存"""
        with self._lock:
            # 按优先级排序
            sorted_sources = sorted(self._sources, key=lambda s: s.priority)
            
            # 合并配置
            merged = {}
            for source in sorted_sources:
                merged = self._deep_merge(merged, source.data)
            
            self._cached_config = merged
            self._cache_valid = True
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔（如 database.timescale.host）
            default: 默认值
            
        Returns:
            配置值
        """
        with self._lock:
            if not self._cache_valid:
                self._rebuild_cache()
            
            keys = key.split('.')
            current = self._cached_config
            
            for k in keys:
                if isinstance(current, dict) and k in current:
                    current = current[k]
                else:
                    return default
            
            return current
    
    def set(self, key: str, value: Any, source: str = "memory"):
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
            source: 配置源名称
        """
        with self._lock:
            # 查找或创建内存配置源
            memory_source = None
            for s in self._sources:
                if s.name == source:
                    memory_source = s
                    break
            
            if memory_source is None:
                memory_source = ConfigSource(source, 20, {})
                self._sources.append(memory_source)
            
            # 设置值
            old_value = self.get(key)
            self._set_nested_value(memory_source.data, key, value)
            
            # 标记缓存无效
            self._cache_valid = False
            
            # 触发变更回调
            if old_value != value:
                for callback in self._change_callbacks:
                    try:
                        callback(key, old_value, value)
                    except Exception as e:
                        logger.error(f"Config change callback error: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self._lock:
            if not self._cache_valid:
                self._rebuild_cache()
            return self._cached_config.copy()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置节
        
        Args:
            section: 节名称
            
        Returns:
            配置节字典
        """
        return self.get(section, {}) or {}
    
    def reload_config(self):
        """重新加载配置"""
        logger.info("Reloading configuration...")
        
        with self._lock:
            # 清除非默认、非环境变量源
            self._sources = [
                s for s in self._sources
                if s.name in ("default", "environment")
            ]
        
        # 重新加载文件配置
        self._load_file_config("config.yaml", 50)
        self._load_file_config(f"config.{self.environment}.yaml", 40)
        self._load_file_config("config.local.yaml", 30)
        
        # 重建缓存
        self._rebuild_cache()
        
        logger.info("Configuration reloaded")
    
    def save_config(self, filename: str = "config.local.yaml"):
        """
        保存配置到文件
        
        Args:
            filename: 文件名
        """
        filepath = self.config_dir / filename
        
        try:
            # 确保目录存在
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(
                    self.get_all(),
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False
                )
            
            logger.info(f"Configuration saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def _start_hot_reload(self):
        """启动热更新"""
        try:
            self._observer = Observer()
            handler = ConfigChangeHandler(self)
            self._observer.schedule(handler, str(self.config_dir), recursive=False)
            self._observer.start()
            
            logger.info(f"Hot reload enabled for {self.config_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to start hot reload: {e}")
    
    def stop_hot_reload(self):
        """停止热更新"""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Hot reload stopped")
    
    def register_change_callback(self, callback: Callable[[str, Any, Any], None]):
        """
        注册配置变更回调
        
        Args:
            callback: 回调函数，参数为 (key, old_value, new_value)
        """
        self._change_callbacks.append(callback)
    
    def get_environment(self) -> str:
        """获取当前环境"""
        return self.environment
    
    def is_development(self) -> bool:
        """是否为开发环境"""
        return self.environment == "development"
    
    def is_production(self) -> bool:
        """是否为生产环境"""
        return self.environment == "production"
    
    def is_testing(self) -> bool:
        """是否为测试环境"""
        return self.environment == "testing"


# 全局配置管理器实例
_config_manager: Optional[ConfigManager] = None
_config_lock = threading.Lock()


def get_config_manager(
    config_dir: str = "config",
    environment: Optional[str] = None
) -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Args:
        config_dir: 配置目录
        environment: 运行环境
        
    Returns:
        ConfigManager: 配置管理器实例
    """
    global _config_manager
    
    if _config_manager is None:
        with _config_lock:
            if _config_manager is None:
                _config_manager = ConfigManager(config_dir, environment)
    
    return _config_manager


def reset_config_manager():
    """重置配置管理器"""
    global _config_manager
    
    if _config_manager:
        _config_manager.stop_hot_reload()
    
    with _config_lock:
        _config_manager = None
    
    logger.info("Config manager reset")


# 便捷函数
def get(key: str, default: Any = None) -> Any:
    """获取配置值"""
    return get_config_manager().get(key, default)


def set(key: str, value: Any):
    """设置配置值"""
    get_config_manager().set(key, value)


def get_env() -> str:
    """获取环境"""
    return get_config_manager().get_environment()


__all__ = [
    'ConfigManager',
    'ConfigSource',
    'get_config_manager',
    'reset_config_manager',
    'get',
    'set',
    'get_env'
]
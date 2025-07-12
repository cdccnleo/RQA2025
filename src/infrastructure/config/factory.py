from typing import Optional
from .services.config_loader_service import ConfigLoaderService
from .strategies.json_loader import JSONLoader
from .strategies.env_loader import EnvLoader
from .strategies.hybrid_loader import HybridLoader
from src.infrastructure.cache.thread_safe_cache import ThreadSafeTTLCache
from .services.version_service import VersionService
from .services.event_service import ConfigEventBus
from .validation.schema import ConfigValidator
from .validation.typed_config import TypedConfigBase

class ConfigFactory:
    """配置服务工厂类"""

    @staticmethod
    def create_file_config_service(
        config_dir: str = "config",
        enable_cache: bool = True,
        enable_watcher: bool = True,
        enable_version: bool = True,
        schema_validator: Optional[ConfigValidator] = None,
        typed_validator: Optional[TypedConfigBase] = None
    ) -> ConfigLoaderService:
        """
        创建基于文件的配置服务
        
        Args:
            config_dir: 配置文件目录
            enable_cache: 是否启用缓存
            enable_watcher: 是否启用文件监听
            enable_version: 是否启用版本管理
            schema_validator: 可选的JSON Schema验证器
            typed_validator: 可选的类型安全验证器
            
        Returns:
            配置服务实例
        """
        loader = JSONLoader(config_dir)

        # 初始化缓存服务
        cache = ThreadSafeTTLCache() if enable_cache else None

        # 初始化其他服务
        watcher = ConfigEventBus() if enable_watcher else None
        version_manager = VersionService() if enable_version else None

        # 创建配置服务实例
        service = ConfigLoaderService(
            loader=loader,
            cache=cache,
            watcher=watcher,
            version_manager=version_manager
        )
        
        # 添加验证器
        if schema_validator:
            service.add_validator(schema_validator)
        if typed_validator:
            service.add_validator(typed_validator)
            
        return service

    @staticmethod
    def create_loader(loader_type: str, **kwargs):
        """创建配置加载器
        
        Args:
            loader_type: 加载器类型 (json/yaml/env/hybrid)
            **kwargs: 加载器参数
            
        Returns:
            配置加载器实例
        """
        if loader_type == "json":
            return JSONLoader(**kwargs)
        elif loader_type == "yaml":
            from .strategies.yaml_loader import YAMLLoader
            return YAMLLoader(**kwargs)
        elif loader_type == "env":
            return EnvLoader(**kwargs)
        elif loader_type == "hybrid":
            return HybridLoader(**kwargs)
        else:
            raise ValueError(f"不支持的加载器类型: {loader_type}")

    @staticmethod
    def create_env_config_service(
        prefix: str = "APP_",
        enable_cache: bool = True
    ) -> ConfigLoaderService:
        """创建基于环境变量的配置服务"""
        loader = EnvLoader(prefix)

        cache = ThreadSafeTTLCache() if enable_cache else None

        return ConfigLoaderService(loader, cache)

    @staticmethod
    def create_hybrid_config_service(
        config_dir: str = "config",
        env_prefix: str = "APP_",
        enable_cache: bool = True,
        enable_watcher: bool = True,
        enable_version: bool = True,
        schema_validator: Optional[ConfigValidator] = None,
        typed_validator: Optional[TypedConfigBase] = None,
        env_priority: bool = False
    ) -> ConfigLoaderService:
        """
        创建混合配置服务(文件+环境变量)
        
        Args:
            config_dir: 配置文件目录
            env_prefix: 环境变量前缀
            enable_cache: 是否启用缓存
            enable_watcher: 是否启用文件监听
            enable_version: 是否启用版本管理
            schema_validator: 可选的JSON Schema验证器
            typed_validator: 可选的类型安全验证器
            env_priority: 环境变量是否优先于文件配置
            
        Returns:
            配置服务实例
        """
        # 初始化加载器
        file_loader = JSONLoader(config_dir)
        env_loader = EnvLoader(env_prefix)
        loader = HybridLoader(
            primary_loader=env_loader if env_priority else file_loader,
            secondary_loader=file_loader if env_priority else env_loader
        )

        # 初始化服务组件
        cache = ThreadSafeTTLCache() if enable_cache else None
        watcher = ConfigEventBus() if enable_watcher else None
        version_manager = VersionService() if enable_version else None

        # 创建配置服务实例
        service = ConfigLoaderService(
            loader=loader,
            cache=cache,
            watcher=watcher,
            version_manager=version_manager
        )
        
        # 添加验证器
        if schema_validator:
            service.add_validator(schema_validator)
        if typed_validator:
            service.add_validator(typed_validator)
            
        return service

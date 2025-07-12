"""配置加载服务 (v3.0)

职责：
1. 管理配置加载流程
2. 处理配置验证
3. 协调多源配置合并
"""
import os
import time
from typing import Dict, List, Optional, Tuple, Any
from src.infrastructure.config.interfaces.config_loader import IConfigLoader
from src.infrastructure.config.interfaces.config_validator import IConfigValidator
from ..exceptions import ConfigLoadError

class ConfigLoaderService:
    def __init__(
        self,
        loader: IConfigLoader,
        validator: IConfigValidator,
        sources: Optional[List[str]] = None
    ):
        """初始化加载服务

        Args:
            loader: 配置加载器
            validator: 配置验证器
            sources: 配置源列表
        """
        self._loader = loader
        self._validator = validator
        self._sources = sources or []

    def load(self, env: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """加载并验证配置

        Returns:
            Tuple[配置数据, 元数据]
        """
        # 调用内部方法获取配置和加载器元数据
        config, loader_meta = self._load_and_merge(env)

        # 创建服务元数据
        service_meta = {
            'env': env,
            'sources': self._sources,
            'timestamp': time.time()
        }

        # 合并加载器返回的元数据
        if loader_meta:
            service_meta.update(loader_meta)

        return config, service_meta

    def _load_and_merge(self, env: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """加载并合并多源配置，返回配置和加载器元数据"""
        if not self._sources:
            raise ConfigLoadError("没有配置源", context={'env': env})

        # 如果是单文件加载
        if len(self._sources) == 1:
            source = self._sources[0]
            try:
                # 加载器返回 (config, loader_meta)
                config, loader_meta = self._loader.load(source.format(env=env))
                is_valid, validation_errors = self._validator.validate(config)
                if not is_valid:
                    raise ConfigLoadError(
                        f"{source}验证失败",
                        context={'errors': validation_errors}
                    )
                return config, loader_meta
            except Exception as e:
                raise ConfigLoadError(
                    f"{source}加载失败",
                    context={'error': str(e)}
                )

        # 多源配置合并逻辑
        merged = {}
        combined_meta = {}
        errors = []
        for source in self._sources:
            try:
                # 每个加载器返回 (config, loader_meta)
                config, loader_meta = self._loader.load(source.format(env=env))
                is_valid, validation_errors = self._validator.validate(config)
                if not is_valid:
                    errors.append(f"{source}验证失败: {validation_errors}")
                    continue
                merged.update(config)
                # 合并元数据
                if loader_meta:
                    combined_meta.update(loader_meta)
            except Exception as e:
                errors.append(f"{source}加载失败: {str(e)}")

        if not merged and errors:
            raise ConfigLoadError(
                f"无法加载{env}配置",
                context={'errors': errors}
            )
        return merged, combined_meta

    def batch_load(self, envs: List[str]) -> Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]]:
        """批量加载配置

        Returns:
            环境到(配置,元数据)的映射
        """
        return {env: self.load(env) for env in envs}

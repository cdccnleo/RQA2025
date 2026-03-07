#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略服务层性能配置
Strategy Service Layer Performance Configuration

基于业务流程驱动架构，提供性能优化的配置管理和最佳实践。
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import os
import json
import logging

from .performance_optimizer import OptimizationConfig

logger = logging.getLogger(__name__)


@dataclass
class PerformanceProfile:

    """性能配置档"""
    name: str
    description: str
    config: OptimizationConfig
    recommended_use_case: str
    performance_targets: Dict[str, Any] = field(default_factory=dict)


class PerformanceConfigManager:

    """
    性能配置管理器
    Performance Configuration Manager

    提供不同场景下的性能配置管理和自动调优。
    """

    def __init__(self, config_file: str = "config / performance_profiles.json"):

        self.config_file = config_file
        self.profiles: Dict[str, PerformanceProfile] = {}
        self.current_profile: Optional[str] = None

        # 初始化内置配置档
        self._init_builtin_profiles()

        # 加载自定义配置
        self._load_custom_profiles()

        logger.info(f"PerformanceConfigManager initialized with {len(self.profiles)} profiles")

    def _init_builtin_profiles(self):
        """初始化内置性能配置档"""

        # 高性能配置 - 用于低延迟高并发场景
        high_performance_config = OptimizationConfig(
            max_concurrent_strategies=50,
            cache_ttl_seconds=600,
            memory_limit_mb=1000,
            cpu_limit_percent=90,
            batch_size=500,
            enable_async_processing=True,
            enable_memory_optimization=True,
            enable_cache_optimization=True
        )

        self.profiles["high_performance"] = PerformanceProfile(
            name="high_performance",
            description="高性能配置 - 适用于低延迟高并发场景",
            config=high_performance_config,
            recommended_use_case="生产环境高频交易",
            performance_targets={
                "response_time_ms": "< 10",
                "throughput_ops_per_sec": "> 5000",
                "memory_usage_mb": "< 800",
                "cpu_usage_percent": "< 85"
            }
        )

        # 平衡配置 - 默认配置，平衡性能和资源使用
        balanced_config = OptimizationConfig(
            max_concurrent_strategies=20,
            cache_ttl_seconds=300,
            memory_limit_mb=500,
            cpu_limit_percent=70,
            batch_size=100,
            enable_async_processing=True,
            enable_memory_optimization=True,
            enable_cache_optimization=True
        )

        self.profiles["balanced"] = PerformanceProfile(
            name="balanced",
            description="平衡配置 - 适用于一般量化策略",
            config=balanced_config,
            recommended_use_case="开发和测试环境",
            performance_targets={
                "response_time_ms": "< 50",
                "throughput_ops_per_sec": "> 2000",
                "memory_usage_mb": "< 400",
                "cpu_usage_percent": "< 60"
            }
        )

        # 资源节省配置 - 用于资源受限环境
        resource_efficient_config = OptimizationConfig(
            max_concurrent_strategies=5,
            cache_ttl_seconds=180,
            memory_limit_mb=200,
            cpu_limit_percent=50,
            batch_size=20,
            enable_async_processing=False,
            enable_memory_optimization=True,
            enable_cache_optimization=True
        )

        self.profiles["resource_efficient"] = PerformanceProfile(
            name="resource_efficient",
            description="资源节省配置 - 适用于资源受限环境",
            config=resource_efficient_config,
            recommended_use_case="轻量级应用和开发环境",
            performance_targets={
                "response_time_ms": "< 200",
                "throughput_ops_per_sec": "> 500",
                "memory_usage_mb": "< 150",
                "cpu_usage_percent": "< 40"
            }
        )

        # 批量处理配置 - 适用于大批量数据处理
        batch_processing_config = OptimizationConfig(
            max_concurrent_strategies=10,
            cache_ttl_seconds=1800,
            memory_limit_mb=800,
            cpu_limit_percent=80,
            batch_size=1000,
            enable_async_processing=True,
            enable_memory_optimization=True,
            enable_cache_optimization=True
        )

        self.profiles["batch_processing"] = PerformanceProfile(
            name="batch_processing",
            description="批量处理配置 - 适用于大批量数据处理",
            config=batch_processing_config,
            recommended_use_case="批量回测和数据分析",
            performance_targets={
                "response_time_ms": "< 100",
                "throughput_ops_per_sec": "> 1000",
                "memory_usage_mb": "< 600",
                "cpu_usage_percent": "< 75"
            }
        )

    def _load_custom_profiles(self):
        """加载自定义配置档"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf - 8') as f:
                    custom_profiles = json.load(f)

                for profile_name, profile_data in custom_profiles.items():
                    config_data = profile_data.get("config", {})
                    config = OptimizationConfig(**config_data)

                    profile = PerformanceProfile(
                        name=profile_name,
                        description=profile_data.get("description", ""),
                        config=config,
                        recommended_use_case=profile_data.get("recommended_use_case", ""),
                        performance_targets=profile_data.get("performance_targets", {})
                    )

                    self.profiles[profile_name] = profile

                logger.info(f"Loaded {len(custom_profiles)} custom profiles")

        except Exception as e:
            logger.warning(f"Failed to load custom profiles: {e}")

    def save_custom_profiles(self):
        """保存自定义配置档"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)

            custom_profiles = {}
            for name, profile in self.profiles.items():
                if name not in ["high_performance", "balanced", "resource_efficient", "batch_processing"]:
                    custom_profiles[name] = {
                        "description": profile.description,
                        "config": {
                            "max_concurrent_strategies": profile.config.max_concurrent_strategies,
                            "cache_ttl_seconds": profile.config.cache_ttl_seconds,
                            "memory_limit_mb": profile.config.memory_limit_mb,
                            "cpu_limit_percent": profile.config.cpu_limit_percent,
                            "batch_size": profile.config.batch_size,
                            "enable_async_processing": profile.config.enable_async_processing,
                            "enable_memory_optimization": profile.config.enable_memory_optimization,
                            "enable_cache_optimization": profile.config.enable_cache_optimization
                        },
                        "recommended_use_case": profile.recommended_use_case,
                        "performance_targets": profile.performance_targets
                    }

            with open(self.config_file, 'w', encoding='utf - 8') as f:
                json.dump(custom_profiles, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(custom_profiles)} custom profiles")

        except Exception as e:
            logger.error(f"Failed to save custom profiles: {e}")

    def get_profile(self, profile_name: str) -> Optional[PerformanceProfile]:
        """获取配置档"""
        return self.profiles.get(profile_name)

    def list_profiles(self) -> List[str]:
        """列出所有配置档"""
        return list(self.profiles.keys())

    def set_current_profile(self, profile_name: str) -> bool:
        """设置当前配置档"""
        if profile_name in self.profiles:
            self.current_profile = profile_name
            logger.info(f"Current performance profile set to: {profile_name}")
            return True
        else:
            logger.warning(f"Performance profile not found: {profile_name}")
            return False

    def get_current_profile(self) -> Optional[PerformanceProfile]:
        """获取当前配置档"""
        if self.current_profile:
            return self.profiles.get(self.current_profile)
        return None

    def get_recommended_profile(self, use_case: str) -> Optional[str]:
        """根据使用场景推荐配置档"""
        recommendations = {
            "production": "high_performance",
            "development": "balanced",
            "testing": "balanced",
            "lightweight": "resource_efficient",
            "batch_processing": "batch_processing",
            "high_frequency": "high_performance",
            "low_latency": "high_performance"
        }

        return recommendations.get(use_case.lower())

    def create_custom_profile(self, name: str, description: str,


                              config: OptimizationConfig,
                              recommended_use_case: str = "",
                              performance_targets: Dict[str, Any] = None) -> bool:
        """创建自定义配置档"""
        if name in self.profiles:
            logger.warning(f"Profile {name} already exists")
            return False

        profile = PerformanceProfile(
            name=name,
            description=description,
            config=config,
            recommended_use_case=recommended_use_case,
            performance_targets=performance_targets or {}
        )

        self.profiles[name] = profile
        self.save_custom_profiles()

        logger.info(f"Created custom profile: {name}")
        return True

    def update_profile(self, name: str, **kwargs) -> bool:
        """更新配置档"""
        if name not in self.profiles:
            logger.warning(f"Profile {name} not found")
            return False

        profile = self.profiles[name]

        # 更新配置
        if "description" in kwargs:
            profile.description = kwargs["description"]
        if "recommended_use_case" in kwargs:
            profile.recommended_use_case = kwargs["recommended_use_case"]
        if "performance_targets" in kwargs:
            profile.performance_targets = kwargs["performance_targets"]

        # 更新OptimizationConfig
        config_updates = {}
        for key in ["max_concurrent_strategies", "cache_ttl_seconds", "memory_limit_mb",
                    "cpu_limit_percent", "batch_size", "enable_async_processing",
                    "enable_memory_optimization", "enable_cache_optimization"]:
            if key in kwargs:
                config_updates[key] = kwargs[key]

        if config_updates:
            for key, value in config_updates.items():
                setattr(profile.config, key, value)

        self.save_custom_profiles()
        logger.info(f"Updated profile: {name}")
        return True

    def delete_profile(self, name: str) -> bool:
        """删除配置档"""
        if name not in self.profiles:
            logger.warning(f"Profile {name} not found")
            return False

        # 不允许删除内置配置档
        builtin_profiles = ["high_performance", "balanced",
                            "resource_efficient", "batch_processing"]
        if name in builtin_profiles:
            logger.warning(f"Cannot delete built - in profile: {name}")
            return False

        del self.profiles[name]
        self.save_custom_profiles()

        if self.current_profile == name:
            self.current_profile = None

        logger.info(f"Deleted profile: {name}")
        return True

    def get_profile_comparison(self) -> Dict[str, Any]:
        """获取配置档对比"""
        comparison = {}

        for name, profile in self.profiles.items():
            comparison[name] = {
                "description": profile.description,
                "max_concurrent_strategies": profile.config.max_concurrent_strategies,
                "cache_ttl_seconds": profile.config.cache_ttl_seconds,
                "memory_limit_mb": profile.config.memory_limit_mb,
                "cpu_limit_percent": profile.config.cpu_limit_percent,
                "batch_size": profile.config.batch_size,
                "recommended_use_case": profile.recommended_use_case,
                "performance_targets": profile.performance_targets
            }

        return comparison

    def auto_select_profile(self, system_info: Dict[str, Any]) -> str:
        """
        基于系统信息自动选择配置档

        Args:
            system_info: 系统信息，包含CPU核心数、内存大小等

        Returns:
            推荐的配置档名称
        """
        cpu_cores = system_info.get("cpu_cores", 4)
        memory_gb = system_info.get("memory_gb", 8)
        use_case = system_info.get("use_case", "general")

        # 基于CPU核心数和内存大小选择配置
        if cpu_cores >= 16 and memory_gb >= 32:
            return "high_performance"
        elif cpu_cores >= 8 and memory_gb >= 16:
            return "balanced"
        elif cpu_cores <= 2 or memory_gb <= 4:
            return "resource_efficient"
        else:
            return "balanced"


# 全局配置管理器实例
_config_manager = None


def get_performance_config_manager() -> PerformanceConfigManager:
    """获取全局性能配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = PerformanceConfigManager()
    return _config_manager


def get_optimization_config_for_use_case(use_case: str) -> OptimizationConfig:
    """
    根据使用场景获取优化配置

    Args:
        use_case: 使用场景

    Returns:
        优化配置
    """
    manager = get_performance_config_manager()

    # 尝试获取推荐配置档
    profile_name = manager.get_recommended_profile(use_case)
    if profile_name:
        profile = manager.get_profile(profile_name)
    if profile:
        return profile.config

    # 返回平衡配置作为默认值
    balanced_profile = manager.get_profile("balanced")
    if balanced_profile:
        return balanced_profile.config

    # 返回默认配置
    return OptimizationConfig()

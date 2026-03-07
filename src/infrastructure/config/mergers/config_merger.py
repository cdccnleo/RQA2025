
import copy

import time
from enum import Enum
from typing import Dict, Any, List, Optional, Callable
import logging
"""
基础设施层 - 配置管理组件

config_merger 模块

高级配置合并器，支持多种合并策略和冲突解决
"""

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """合并策略枚举"""
    OVERWRITE = "overwrite"  # 覆盖模式
    MERGE = "merge"  # 合并模式
    DEEP_MERGE = "deep_merge"  # 深度合并模式
    PRESERVE = "preserve"  # 保留模式
    CUSTOM = "custom"  # 自定义模式


class ConflictResolution(Enum):
    """冲突解决策略枚举"""
    SOURCE_WINS = "source_wins"  # 源配置优先
    TARGET_WINS = "target_wins"  # 目标配置优先
    MERGE_VALUES = "merge_values"  # 合并值
    THROW_ERROR = "throw_error"  # 抛出错误
    CUSTOM_RESOLVER = "custom_resolver"  # 自定义解决器


class ConfigMerger:
    """高级配置合并器"""

    def __init__(self, strategy: MergeStrategy = MergeStrategy.DEEP_MERGE,
                 conflict_resolution: ConflictResolution = ConflictResolution.SOURCE_WINS):
        """
        初始化配置合并器

        Args:
            strategy: 合并策略
            conflict_resolution: 冲突解决策略
        """
        self.strategy = strategy
        self.conflict_resolution = conflict_resolution
        self.custom_merge_func: Optional[Callable] = None
        self.custom_conflict_resolver: Optional[Callable] = None

        # 合并统计
        self.stats = {
            'total_merges': 0,
            'successful_merges': 0,
            'conflict_count': 0,
            'merge_time': 0.0
        }

    def merge(self, *configs: Dict[str, Any], key_path: str = "") -> Dict[str, Any]:
        """合并一个或多个配置字典"""

        if not configs:
            return {}

        configs_list = list(configs)

        # 兼容旧签名：merge(target, source, key_path)
        if configs_list and isinstance(configs_list[-1], str) and not isinstance(configs_list[-1], dict):
            if not key_path:
                key_path = configs_list[-1]
            configs_list = configs_list[:-1]

        if len(configs_list) == 1:
            return copy.deepcopy(configs_list[0])

        result = copy.deepcopy(configs_list[0])
        for next_config in configs_list[1:]:
            result = self._merge_single(result, next_config, key_path)

        return result

    def _merge_single(self, target: Dict[str, Any], source: Dict[str, Any], key_path: str = "") -> Dict[str, Any]:
        start_time = time.time()

        try:
            self.stats['total_merges'] += 1

            if self.strategy == MergeStrategy.OVERWRITE:
                result = self._merge_overwrite(target, source)
            elif self.strategy == MergeStrategy.MERGE:
                result = self._merge_shallow(target, source)
            elif self.strategy == MergeStrategy.DEEP_MERGE:
                result = self._merge_deep(target, source, key_path)
            elif self.strategy == MergeStrategy.PRESERVE:
                result = self._merge_preserve(target, source)
            elif self.strategy == MergeStrategy.CUSTOM:
                result = self._merge_custom(target, source)
            else:
                raise ValueError(f"Unsupported merge strategy: {self.strategy}")

            self.stats['successful_merges'] += 1
            self.stats['merge_time'] = time.time() - start_time

            return result

        except Exception as e:
            logger.error(f"Config merge failed at path '{key_path}': {e}")
            raise

    def _merge_overwrite(self, target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        """覆盖合并：完全替换目标配置"""
        return copy.deepcopy(source)

    def _merge_shallow(self, target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        """浅层合并：只合并第一级键"""
        result = copy.deepcopy(target)
        result.update(source)
        return result

    def _merge_deep(self, target: Dict[str, Any], source: Dict[str, Any],
                    key_path: str = "") -> Dict[str, Any]:
        """深度合并：递归合并嵌套字典"""
        result = copy.deepcopy(target)

        for key, source_value in source.items():
            current_path = f"{key_path}.{key}" if key_path else key

            if key in result:
                target_value = result[key]

                # 检查是否都是字典，进行递归合并
                if isinstance(target_value, dict) and isinstance(source_value, dict):
                    result[key] = self._merge_deep(target_value, source_value, current_path)
                # 检查是否都是列表，进行列表合并
                elif isinstance(target_value, list) and isinstance(source_value, list):
                    result[key] = self._merge_lists(target_value, source_value, current_path)
                else:
                    # 处理冲突
                    result[key] = self._resolve_conflict(
                        key, target_value, source_value, current_path)
            else:
                result[key] = copy.deepcopy(source_value)

        return result

    def _merge_preserve(self, target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        """保留合并：只添加不存在的键"""
        result = copy.deepcopy(target)

        for key, value in source.items():
            if key not in result:
                result[key] = copy.deepcopy(value)

        return result

    def _merge_custom(self, target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        """自定义合并"""
        if self.custom_merge_func:
            return self.custom_merge_func(target, source)
        else:
            raise ValueError("Custom merge function not set")

    def _merge_lists(self, target_list: List[Any], source_list: List[Any],
                     key_path: str) -> List[Any]:
        """合并列表"""
        # 默认策略：追加源列表到目标列表
        result = copy.deepcopy(target_list)

        for item in source_list:
            if item not in result:  # 避免重复
                result.append(copy.deepcopy(item))

        return result

    def _resolve_conflict(self, key: str, target_value: Any, source_value: Any,
                          key_path: str) -> Any:
        """解决配置冲突"""
        self.stats['conflict_count'] += 1

        if self.conflict_resolution == ConflictResolution.SOURCE_WINS:
            return copy.deepcopy(source_value)
        elif self.conflict_resolution == ConflictResolution.TARGET_WINS:
            return copy.deepcopy(target_value)
        elif self.conflict_resolution == ConflictResolution.MERGE_VALUES:
            return self._merge_values(target_value, source_value, key_path)
        elif self.conflict_resolution == ConflictResolution.THROW_ERROR:
            raise ValueError(f"Configuration conflict at '{key_path}': "
                             f"target={target_value}, source={source_value}")
        elif self.conflict_resolution == ConflictResolution.CUSTOM_RESOLVER:
            if self.custom_conflict_resolver:
                return self.custom_conflict_resolver(key, target_value, source_value, key_path)
            else:
                raise ValueError("Custom conflict resolver not set")
        else:
            # 默认使用源配置优先
            return copy.deepcopy(source_value)

    def _merge_values(self, target_value: Any, source_value: Any, key_path: str) -> Any:
        """合并值"""
        # 如果都是字典，递归合并
        if isinstance(target_value, dict) and isinstance(source_value, dict):
            return self._merge_deep(target_value, source_value, key_path)

        # 如果都是列表，合并列表
        if isinstance(target_value, list) and isinstance(source_value, list):
            return self._merge_lists(target_value, source_value, key_path)

        # 如果都是字符串，连接字符串
        if isinstance(target_value, str) and isinstance(source_value, str):
            return f"{target_value};{source_value}"

        # 其他情况，返回源值
        return copy.deepcopy(source_value)

    def set_custom_merge_function(self, func: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]):
        """设置自定义合并函数"""
        self.custom_merge_func = func

    def set_custom_conflict_resolver(self, func: Callable[[str, Any, Any, str], Any]):
        """设置自定义冲突解决器"""
        self.custom_conflict_resolver = func

    def get_merge_stats(self) -> Dict[str, Any]:
        """获取合并统计信息"""
        return self.stats.copy()

    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_merges': 0,
            'successful_merges': 0,
            'conflict_count': 0,
            'merge_time': 0.0
        }


class HierarchicalConfigMerger(ConfigMerger):
    """层次化配置合并器"""

    def __init__(self, priority_order: Optional[List[str]] = None, **kwargs):
        """
        初始化层次化配置合并器

        Args:
            priority_order: 配置源优先级顺序
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.priority_order = priority_order or ['default', 'environment', 'user', 'application']

    def merge_hierarchical(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        按层次合并多个配置源

        Args:
            configs: 各配置源的配置字典

        Returns:
            合并后的配置
        """
        result = {}

        # 按优先级顺序合并
        for source_name in self.priority_order:
            if source_name in configs:
                source_config = configs[source_name]
                result = self.merge(result, source_config, f"hierarchy.{source_name}")

        return result


class EnvironmentAwareConfigMerger(ConfigMerger):
    """环境感知配置合并器"""

    def __init__(self, environment: str = 'development', **kwargs):
        """
        初始化环境感知配置合并器

        Args:
            environment: 当前环境
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.environment = environment

    def merge_with_environment(self, base_config: Dict[str, Any],
                               env_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并基础配置和环境特定配置

        Args:
            base_config: 基础配置
            env_configs: 环境配置字典

        Returns:
            合并后的配置
        """
        result = copy.deepcopy(base_config)

        # 合并环境特定的配置
        if self.environment in env_configs:
            env_config = env_configs[self.environment]
            result = self.merge(result, env_config, f"env.{self.environment}")

        return result


class ProfileBasedConfigMerger(ConfigMerger):
    """基于配置文件的配置合并器"""

    def __init__(self, active_profiles: Optional[List[str]] = None, **kwargs):
        """
        初始化基于配置文件的合并器

        Args:
            active_profiles: 激活的配置文件
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.active_profiles = active_profiles or ['default']

    def merge_with_profiles(self, base_config: Dict[str, Any],
                            profile_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        合并基础配置和配置文件

        Args:
            base_config: 基础配置
            profile_configs: 配置文件字典

        Returns:
            合并后的配置
        """
        result = copy.deepcopy(base_config)

        # 合并激活的配置文件
        for profile in self.active_profiles:
            if profile in profile_configs:
                profile_config = profile_configs[profile]
                result = self.merge(result, profile_config, f"profile.{profile}")

        return result

# 便捷函数


def merge_configs(target: Dict[str, Any], source: Dict[str, Any],
                  strategy: MergeStrategy = MergeStrategy.DEEP_MERGE) -> Dict[str, Any]:
    """
    便捷的配置合并函数

    Args:
        target: 目标配置
        source: 源配置
        strategy: 合并策略

    Returns:
        合并后的配置
    """
    merger = ConfigMerger(strategy=strategy)
    return merger.merge(target, source)


def merge_hierarchical_configs(configs: Dict[str, Dict[str, Any]],
                               priority_order: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    便捷的层次化配置合并函数

    Args:
        configs: 配置源字典
        priority_order: 优先级顺序

    Returns:
        合并后的配置
    """
    merger = HierarchicalConfigMerger(priority_order=priority_order)
    return merger.merge_hierarchical(configs)


def merge_environment_configs(base_config: Dict[str, Any],
                              env_configs: Dict[str, Dict[str, Any]],
                              environment: str) -> Dict[str, Any]:
    """
    便捷的环境配置合并函数

    Args:
        base_config: 基础配置
        env_configs: 环境配置字典
        environment: 当前环境

    Returns:
        合并后的配置
    """
    merger = EnvironmentAwareConfigMerger(environment=environment)
    return merger.merge_with_environment(base_config, env_configs)





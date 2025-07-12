from typing import Dict, List, Tuple
from src.infrastructure.config.strategies.base_loader import IConfigLoader
from src.infrastructure.config.error import ConfigLoadError

class HybridLoader(IConfigLoader):
    """混合配置加载器，支持按优先级合并多个来源的配置

    示例用法:
        loaders = [
            (JSONLoader(), 100),  # 高优先级
            (EnvLoader(), 50),    # 中优先级
            (YAMLLoader(), 10)    # 低优先级
        ]
        loader = HybridLoader(loaders)
        config = loader.load('app_config')
    """

    def __init__(self, loaders: List[Tuple[IConfigLoader, int]]):
        """初始化混合加载器

        Args:
            loaders: 加载器列表，每个元素为(loader, priority)元组
                    priority越高优先级越高
        """
        self._loaders = sorted(loaders, key=lambda x: -x[1])  # 按优先级降序排序

    def load(self, source: str) -> Dict:
        """加载配置

        Args:
            source: 配置源标识符

        Returns:
            合并后的配置字典

        Raises:
            ConfigLoadError: 当所有加载器都失败时抛出
        """
        configs = []
        for loader, _ in self._loaders:
            try:
                config = loader.load(source)
                if config:  # 只添加非空配置
                    configs.append(config)
            except ConfigLoadError:
                continue

        if not configs:
            raise ConfigLoadError(f"所有加载器都无法加载配置: {source}")

        return self._merge_with_priority(configs)

    def _merge_with_priority(self, configs: List[Dict]) -> Dict:
        """按优先级合并配置

        高优先级配置会覆盖低优先级配置
        对于嵌套字典会递归合并

        Args:
            configs: 待合并的配置列表，按优先级排序

        Returns:
            合并后的配置字典
        """
        merged = {}
        for config in configs:
            merged = self._deep_merge(merged, config)
        return merged

    def _deep_merge(self, base: Dict, new: Dict) -> Dict:
        """深度合并两个字典

        Args:
            base: 基础字典
            new: 要合并的新字典

        Returns:
            合并后的字典
        """
        result = base.copy()
        for k, v in new.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = self._deep_merge(result[k], v)
            else:
                result[k] = v
        return result

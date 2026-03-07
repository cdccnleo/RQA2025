"""增强版统一配置管理器兼容层。

提供 `UnifiedConfigManager` 的轻量包装，以兼容历史测试用例
从 `unified_manager_enhanced` 模块导入管理器的场景。
当前实现直接复用 `config_manager_complete` 中的实现，
同时保留扩展 hook，便于后续增强功能接入。
"""

from __future__ import annotations

from typing import Any, Dict

from .config_manager_complete import UnifiedConfigManager as _UnifiedConfigManager


class UnifiedConfigManager(_UnifiedConfigManager):
    """面向历史用例的增强别名实现。

    继承自完整版 `UnifiedConfigManager`，添加少量兼容性辅助方法，
    使依赖 `unified_manager_enhanced` 的测试能够顺利运行。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._enhanced_initialized = True

    # 兼容性：部分测试会显式调用该方法
    def _initialize_enhanced_features(self) -> None:  # pragma: no cover - 简单hook
        self._enhanced_initialized = True

    # 兼容性：提供详细的状态信息
    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.setdefault('initialized', getattr(self, '_initialized', False))
        status.setdefault('sections_count', len(getattr(self, '_data', {})))
        status.setdefault('total_keys', sum(
            len(values) for values in getattr(self, '_data', {}).values()
            if isinstance(values, dict)
        ))
        return status

    # 兼容性：暴露配置来源信息（如基础实现缺省则回退为内存）
    def get_config_with_source_info(self, key: str, default_section: str = 'default') -> Dict[str, Any]:
        info = super().get_config_with_source_info(key, default_section)
        info.setdefault('source', 'memory')
        info.setdefault('available', info.get('value') is not None)
        info.setdefault('type', type(info.get('value')).__name__ if info.get('value') is not None else 'NoneType')
        return info



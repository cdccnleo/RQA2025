"""
统一监控适配器兼容模块

为了兼容旧版测试和业务代码，保留 ``src.infrastructure.resource.unified_monitor_adapter``
 的导入路径，内部直接委托给 ``monitoring.unified_monitor_adapter``。
"""

from __future__ import annotations

try:
    from .monitoring.unified_monitor_adapter import UnifiedMonitor  # type: ignore
except Exception as exc:  # pragma: no cover - 仅用于降级
    class UnifiedMonitor:  # type: ignore[override]
        """回退UnifiedMonitor，在真实实现不可用时提供清晰错误。"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "UnifiedMonitor 实现不可用，监控适配器无法初始化"
            ) from exc


__all__ = ["UnifiedMonitor"]


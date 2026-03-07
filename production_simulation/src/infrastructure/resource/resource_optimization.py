"""
资源优化模块（别名模块）
提供向后兼容的导入路径

实际实现在 ``resource/core/resource_optimization.py`` 或
``resource/core/resource_optimization_engine.py`` 中。
"""

from __future__ import annotations

try:
    from .core.resource_optimization import ResourceOptimizer as _ResourceOptimizer
except Exception:  # pragma: no cover - 当高层封装缺失时触发
    _ResourceOptimizer = None  # type: ignore[assignment]

try:
    from .core.resource_optimization_engine import ResourceOptimizationEngine
except Exception:  # pragma: no cover - 当底层引擎缺失时触发
    ResourceOptimizationEngine = None  # type: ignore[assignment]


if _ResourceOptimizer is not None:
    ResourceOptimizer = _ResourceOptimizer
else:  # pragma: no cover - 降级路径
    class ResourceOptimizer:  # type: ignore[override]
        """回退 ResourceOptimizer，在真实实现缺失时提示使用者。"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ResourceOptimizer 实现不可用，请检查依赖或初始化顺序"
            )


if ResourceOptimizationEngine is not None:
    ResourceOptimization = ResourceOptimizationEngine
else:  # pragma: no cover - 降级路径
    class ResourceOptimization:  # type: ignore[override]
        """回退 ResourceOptimization 引擎。"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ResourceOptimizationEngine 实现不可用，请检查依赖或初始化顺序"
            )


__all__ = ["ResourceOptimizer", "ResourceOptimization", "ResourceOptimizationEngine"]


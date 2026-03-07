"""
资源管理兼容模块

历史代码通过 ``src.infrastructure.resource.resource_manager`` 导入
资源管理器。核心实现已经迁移到 ``core.resource_manager`` 中，
此文件作为轻量外观提供向后兼容的导入路径。
"""

from __future__ import annotations

try:
    from .core.resource_manager import CoreResourceManager as ResourceManager  # type: ignore
except Exception as exc:  # pragma: no cover - 回退分支主要用于调试
    # 提供一个安全的占位实现，避免导入失败导致调用端崩溃
    class ResourceManager:  # type: ignore[override]
        """回退ResourceManager，在核心实现无法导入时提供明确信息。"""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "CoreResourceManager 无法导入，资源管理功能不可用"
            ) from exc


__all__ = ["ResourceManager"]


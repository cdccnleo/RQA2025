"""
数据版本管理模块

提供轻量级的数据版本管理能力，支持简单的版本创建、查询和清理，
并保持与历史测试用例的兼容性。
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from ..core.version import Version


def _ensure_version(value: Union[str, Version]) -> Version:
    """内部工具：把输入统一转换为 Version 对象。"""
    return value if isinstance(value, Version) else Version(value)


def _clone_data(data: Any) -> Any:
    """确保返回数据的副本，避免外部修改内部状态。"""
    try:
        return deepcopy(data)
    except Exception:
        # deepcopy 失败时尽量保持原值（例如某些模拟对象无法复制）
        return data


@dataclass
class VersionInfo:
    """
    数据版本信息。

    该数据类同时满足历史测试对字段的要求，并提供常用的辅助方法。
    """

    version: Version
    data: Any
    timestamp: datetime
    checksum: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": str(self.version),
            "timestamp": self.timestamp.isoformat(),
            "checksum": self.checksum,
            "metadata": deepcopy(self.metadata),
            "tags": list(self.tags),
        }


class DataVersionManager:
    """
    数据版本管理器。

    - `save_version`/`create_version`：保存数据并生成新的语义化版本。
    - `get_version`：按名称和版本号获取已保存的数据。
    - `list_versions`：列出全部或指定数据集的版本。
    - `cleanup_old_versions`：按照时间阈值清理旧版本。

    默认情况下版本号以 `patch` 维度自增（1.0.0 -> 1.0.1）。
    """

    def __init__(self, base_path: Union[str, Path] = "./data_versions"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._store: Dict[str, List[VersionInfo]] = defaultdict(list)

    # ------------------------------------------------------------------
    # 版本创建与保存
    # ------------------------------------------------------------------
    def save_version(
        self,
        data_id: str,
        data: Any,
        *,
        version: Union[str, Version, None] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        checksum: Optional[str] = None,
    ) -> Version:
        """
        保存数据版本。

        Args:
            data_id: 数据集标识
            data: 待保存的数据对象
            version: 指定版本号，缺省则自动递增补丁号
            metadata: 元数据
            tags: 标签集合
            checksum: 预计算的校验和，未提供则自动计算

        Returns:
            Version: 保存后的版本对象
        """
        metadata = metadata or {}
        tags_list = list(tags or [])

        if version is None:
            version = self._next_patch_version(data_id)
        else:
            version = _ensure_version(version)

        checksum = checksum or self._calculate_checksum(data)

        info = VersionInfo(
            version=Version(str(version)),
            data=_clone_data(data),
            timestamp=datetime.now(),
            checksum=checksum,
            metadata=deepcopy(metadata),
            tags=tags_list,
        )

        self._store[data_id].append(info)
        return info.version

    def create_version(
        self,
        data_id: str,
        data: Any,
        version: Union[str, Version, None] = None,
        **kwargs: Any,
    ) -> Version:
        """
        `save_version` 的语义化别名，保持旧接口兼容。

        额外的关键字参数会传递给 `save_version`（如 metadata/tags）。
        """
        return self.save_version(data_id, data, version=version, **kwargs)

    # ------------------------------------------------------------------
    # 数据读取
    # ------------------------------------------------------------------
    def get_version(
        self,
        data_id: str,
        version: Union[str, Version, None] = None,
    ) -> Optional[Any]:
        """获取指定数据版本，未指定版本时返回最新版本。"""
        versions = self._store.get(data_id, [])
        if not versions:
            return None

        if version is None:
            return _clone_data(versions[-1].data)

        version = _ensure_version(version)
        for info in reversed(versions):
            if info.version == version:
                return _clone_data(info.data)

        return None

    def get_version_info(
        self,
        data_id: str,
        version: Union[str, Version],
    ) -> Optional[VersionInfo]:
        """返回内部存储的版本信息对象。"""
        version = _ensure_version(version)
        for info in self._store.get(data_id, []):
            if info.version == version:
                return info

        return None

    # ------------------------------------------------------------------
    # 枚举与统计
    # ------------------------------------------------------------------
    def list_versions(self, data_id: Optional[str] = None) -> List[str]:
        """
        列出版本信息。

        Args:
            data_id: 指定数据集名称，为空则返回所有数据集的最新版本列表。

        Returns:
            List[str]: 版本字符串列表。
        """
        if data_id is not None:
            return [str(info.version) for info in self._store.get(data_id, [])]

        result: List[str] = []
        for infos in self._store.values():
            result.extend(str(info.version) for info in infos)
        return result

    def get_version_history(self, data_id: str) -> List[VersionInfo]:
        """返回指定数据集的版本历史（按时间顺序）。"""
        return list(self._store.get(data_id, []))

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def cleanup_old_versions(self, days: int = 30) -> int:
        """
        清理早于指定天数的版本。

        Args:
            days: 保留的天数阈值

        Returns:
            int: 被清理的版本数量
        """
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0

        for data_id, infos in list(self._store.items()):
            retained: List[VersionInfo] = []
            for info in infos:
                if info.timestamp >= cutoff:
                    retained.append(info)
                else:
                    removed += 1
            if retained:
                self._store[data_id] = retained
            else:
                self._store.pop(data_id, None)

        return removed

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def _next_patch_version(self, data_id: str) -> Version:
        """根据已有版本自动计算下一个补丁版本。"""
        versions = self._store.get(data_id)
        if not versions:
            return Version("1.0.0")
        latest = Version(str(versions[-1].version))
        latest.increment_patch()
        return latest

    @staticmethod
    def _calculate_checksum(data: Any) -> str:
        """使用 sha256 生成简单的内容校验和。"""
        try:
            if isinstance(data, (dict, list)):
                payload = json.dumps(data, sort_keys=True, default=str)
            else:
                payload = json.dumps(data, default=str)
        except TypeError:
            payload = str(data)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

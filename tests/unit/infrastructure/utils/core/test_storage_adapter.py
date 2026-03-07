#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StorageAdapter 基类测试。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import pytest

from src.infrastructure.utils.core import storage


class _DummyStorage(storage.StorageAdapter):
    def save(self, key: str, data: Any, **kwargs) -> bool:
        return True

    def load(self, key: str, **kwargs) -> Any:
        return None

    def delete(self, key: str, **kwargs) -> bool:
        return True

    def exists(self, key: str, **kwargs) -> bool:
        return False

    def list_keys(self, prefix: str = "", **kwargs) -> List[str]:
        return []


def test_storage_adapter_creates_directory(tmp_path: Path) -> None:
    base = tmp_path / "data"
    adapter = _DummyStorage(str(base))
    assert adapter.base_path == base
    assert base.exists()

    stats = adapter.get_stats()
    assert stats["adapter_type"] == "_DummyStorage"
    assert stats["base_path"] == str(base)


def test_storage_adapter_permission_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "restricted"

    def _raise_permission(self: Path, parents: bool = True, exist_ok: bool = True) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "mkdir", _raise_permission)
    with pytest.raises(PermissionError, match="创建存储目录失败"):
        _DummyStorage(str(base))


def test_storage_adapter_os_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "broken"

    def _raise_os_error(self: Path, parents: bool = True, exist_ok: bool = True) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", _raise_os_error)
    with pytest.raises(OSError, match="disk full"):
        _DummyStorage(str(base))


def test_storage_adapter_runtime_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    base = tmp_path / "unexpected"

    def _raise_other(self: Path, parents: bool = True, exist_ok: bool = True) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(Path, "mkdir", _raise_other)
    with pytest.raises(RuntimeError, match="存储初始化失败"):
        _DummyStorage(str(base))

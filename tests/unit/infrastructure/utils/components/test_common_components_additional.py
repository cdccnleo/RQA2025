#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对 common_components 的补充测试，用于覆盖关键分支与工厂方法。
"""

from __future__ import annotations

from typing import Dict

import pytest

from src.infrastructure.utils.components.common_components import (
    CommonComponent,
    CommonComponentConstants,
    CommonComponentFactory,
)


def _ensure_common_concrete() -> None:
    """解除抽象限制，提供默认 get_status 实现。"""
    if "get_status" not in CommonComponent.__dict__ or getattr(
        CommonComponent, "__abstractmethods__", frozenset()
    ):

        def _get_status(self) -> Dict[str, object]:
            return {
                "status": self.status,
                "metadata": getattr(self, "status_manager", None),
                "specific": self._get_component_specific_status(),
            }

        setattr(CommonComponent, "get_status", _get_status)
        setattr(CommonComponent, "__abstractmethods__", frozenset())


def _instantiate_common(common_id: int, component_type: str = "Common") -> CommonComponent:
    """确保 CommonComponent 具备 get_status 实现后再实例化。"""

    _ensure_common_concrete()
    return CommonComponent(common_id, component_type)


_ensure_common_concrete()


def test_common_component_initial_metadata_and_status() -> None:
    component = _instantiate_common(10, component_type="Service")

    info = component.get_info()
    assert info["common_id"] == 10
    assert info["component_type"] == "Service"
    assert info["version"] == CommonComponentConstants.COMPONENT_VERSION

    # 初始化时 metadata 已写入
    assert component.status_manager._metadata["common_id"] == 10
    assert "creation_time" in component.status_manager._metadata

    status = component._get_component_specific_status()
    assert status["health"] == "good"
    assert status["active"] is True


class _FlakyStatusManager:
    """首次访问 component_name 抛错，用于命中异常分支。"""

    def __init__(self) -> None:
        self._calls = 0
        self.component_type = "Common"

    @property
    def component_name(self) -> str:
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("boom")
        return "Fallback"

    def add_metadata(self, key: str, value: object) -> None:
        """占位方法，兼容构造流程。"""


def test_common_component_process_error_path() -> None:
    component = _instantiate_common(16)
    component.status_manager = _FlakyStatusManager()

    result = component.process({"payload": 1})
    assert result["status"] == "error"
    assert result["error_type"] == "RuntimeError"
    assert result["component_name"] == "Fallback"


def test_common_component_factory_supports_digit_and_prefix() -> None:
    factory = CommonComponentFactory()

    comp_from_digit = factory.create_component("10")
    assert isinstance(comp_from_digit, CommonComponent)
    assert comp_from_digit.get_common_id() == 10

    comp_from_prefix = factory.create_component("common_16")
    assert comp_from_prefix.get_common_id() == 16


def test_common_component_factory_invalid_id_raises() -> None:
    factory = CommonComponentFactory()
    with pytest.raises(ValueError):
        factory.create_component("common_999")


def test_common_component_factory_fallback_to_super_returns_none() -> None:
    factory = CommonComponentFactory()
    assert factory.create_component("unknown_type") is None


def test_common_component_factory_utilities() -> None:
    available = CommonComponentFactory.get_available_commons()
    assert available == sorted(CommonComponentFactory.SUPPORTED_COMMON_IDS)

    all_instances = CommonComponentFactory.create_all_commons()
    assert set(all_instances.keys()) == set(CommonComponentFactory.SUPPORTED_COMMON_IDS)
    assert all(isinstance(comp, CommonComponent) for comp in all_instances.values())

    info: Dict[str, object] = CommonComponentFactory.get_factory_info()
    assert info["factory_name"] == "CommonComponentFactory"
    assert info["total_commons"] == len(CommonComponentFactory.SUPPORTED_COMMON_IDS)
    assert info["supported_ids"] == available


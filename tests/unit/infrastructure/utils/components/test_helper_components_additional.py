#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
helper_components 额外单测，覆盖 HelperComponent 及其工厂的关键分支。
"""

from __future__ import annotations

import pytest

from src.infrastructure.utils.components import helper_components


def test_helper_component_basic_info_and_status():
    component = helper_components.HelperComponent(helper_id=2, component_type="Helper")

    info = component.get_info()
    assert info["helper_id"] == 2
    assert info["component_name"] == "Helper_Component_2"
    # 修复后的描述应包含具体类型
    assert info["description"] == "统一Helper组件实现"

    status = component.get_status()
    assert status["status"] == helper_components.HelperComponentConstants.STATUS_ACTIVE
    assert status["helper_id"] == 2


def test_helper_component_factory_digits_and_helper_prefix():
    factory = helper_components.HelperComponentFactory()

    comp_by_digit = helper_components.HelperComponentFactory.create_component(2)
    assert isinstance(comp_by_digit, helper_components.HelperComponent)
    assert comp_by_digit.get_helper_id() == 2

    comp_by_name = factory.create_component("helper_8")
    assert comp_by_name.get_helper_id() == 8


def test_helper_component_factory_unsupported_id():
    factory = helper_components.HelperComponentFactory()
    with pytest.raises(ValueError):
        factory.create_component("helper_999")


def test_helper_component_factory_helpers_collection():
    helpers = helper_components.HelperComponentFactory.create_all_helpers()
    ids = helper_components.HelperComponentFactory.get_available_helpers()
    assert set(helpers.keys()) == set(ids)
    assert all(isinstance(comp, helper_components.HelperComponent) for comp in helpers.values())


def test_helper_component_factory_info_fields():
    info = helper_components.HelperComponentFactory.get_factory_info()
    assert info["factory_name"] == "HelperComponentFactory"
    assert info["total_helpers"] == len(helper_components.HelperComponentConstants.SUPPORTED_HELPER_IDS)
    assert "created_at" in info



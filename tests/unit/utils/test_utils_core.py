#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils Core Module测试

测试工具核心模块的功能
"""

import pytest
from unittest.mock import Mock


class TestUtilsCore:
    """测试UtilsCore类"""

    def test_utils_core_init(self):
        """测试UtilsCore初始化"""
        from src.utils.core import UtilsCore

        core = UtilsCore()
        assert core.version == "1.0.0"

    def test_get_version(self):
        """测试get_version方法"""
        from src.utils.core import UtilsCore

        core = UtilsCore()
        assert core.get_version() == "1.0.0"

    def test_validate_config_valid(self):
        """测试validate_config方法 - 有效配置"""
        from src.utils.core import UtilsCore

        core = UtilsCore()
        valid_config = {"key": "value"}
        assert core.validate_config(valid_config) is True

    def test_validate_config_invalid_type(self):
        """测试validate_config方法 - 无效类型"""
        from src.utils.core import UtilsCore

        core = UtilsCore()
        invalid_config = "not a dict"
        assert core.validate_config(invalid_config) is False

    def test_validate_config_empty_dict(self):
        """测试validate_config方法 - 空字典"""
        from src.utils.core import UtilsCore

        core = UtilsCore()
        empty_config = {}
        assert core.validate_config(empty_config) is False

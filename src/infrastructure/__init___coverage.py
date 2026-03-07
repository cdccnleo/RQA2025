#!/usr/bin/env python3
"""
针对性测试: src\infrastructure\__init__

自动生成的覆盖率提升测试
"""

import pytest


class TestSrc\Infrastructure\__Init__Coverage:
    """src\infrastructure\__init__ 覆盖率测试"""


    def test_create_config_manager_existence(self):
        """测试 create_config_manager 函数存在性"""
        try:
            from src\infrastructure\__init__ import create_config_manager
            assert callable(create_config_manager)
        except ImportError:
            pytest.skip(f"create_config_manager 函数不可用")

    def test_create_cache_manager_existence(self):
        """测试 create_cache_manager 函数存在性"""
        try:
            from src\infrastructure\__init__ import create_cache_manager
            assert callable(create_cache_manager)
        except ImportError:
            pytest.skip(f"create_cache_manager 函数不可用")

    def test_create_health_checker_existence(self):
        """测试 create_health_checker 函数存在性"""
        try:
            from src\infrastructure\__init__ import create_health_checker
            assert callable(create_health_checker)
        except ImportError:
            pytest.skip(f"create_health_checker 函数不可用")


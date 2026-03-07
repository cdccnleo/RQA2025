#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层versioning/__init__.py模块测试

测试目标：提升versioning/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.versioning模块
"""

import pytest


class TestVersioningInit:
    """测试versioning模块初始化"""
    
    def test_version_import(self):
        """测试Version导入"""
        from src.infrastructure.versioning import Version
        
        assert Version is not None
    
    def test_version_comparator_import(self):
        """测试VersionComparator导入"""
        from src.infrastructure.versioning import VersionComparator
        
        assert VersionComparator is not None
    
    def test_version_proxy_import(self):
        """测试VersionProxy导入"""
        from src.infrastructure.versioning import VersionProxy
        
        assert VersionProxy is not None
    
    def test_version_manager_import(self):
        """测试VersionManager导入"""
        from src.infrastructure.versioning import VersionManager
        
        assert VersionManager is not None
    
    def test_version_policy_import(self):
        """测试VersionPolicy导入"""
        from src.infrastructure.versioning import VersionPolicy
        
        assert VersionPolicy is not None
    
    def test_data_version_manager_import(self):
        """测试DataVersionManager导入"""
        from src.infrastructure.versioning import DataVersionManager
        
        assert DataVersionManager is not None
    
    def test_version_info_import(self):
        """测试VersionInfo导入"""
        from src.infrastructure.versioning import VersionInfo
        
        assert VersionInfo is not None
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.versioning import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        expected_exports = [
            "Version",
            "VersionComparator",
            "VersionProxy",
            "VersionManager",
            "VersionPolicy",
            "DataVersionManager",
            "VersionInfo"
        ]
        for export in expected_exports:
            assert export in __all__, f"{export} should be in __all__"
    
    def test_module_version(self):
        """测试模块版本"""
        from src.infrastructure.versioning import __version__
        
        assert isinstance(__version__, str)
        assert __version__ == "2.0.0"


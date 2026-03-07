#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层distributed/__init__.py模块测试

测试目标：提升distributed/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.distributed模块
"""

import pytest


class TestDistributedInit:
    """测试distributed模块初始化"""
    
    def test_distributed_lock_manager_import(self):
        """测试DistributedLockManager导入"""
        from src.infrastructure.distributed import DistributedLockManager
        
        assert DistributedLockManager is not None
    
    def test_config_center_manager_import(self):
        """测试ConfigCenterManager导入"""
        from src.infrastructure.distributed import ConfigCenterManager
        
        assert ConfigCenterManager is not None
    
    def test_distributed_monitoring_manager_import(self):
        """测试DistributedMonitoringManager导入"""
        from src.infrastructure.distributed import DistributedMonitoringManager
        
        assert DistributedMonitoringManager is not None
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.distributed import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "DistributedLockManager" in __all__
        assert "ConfigCenterManager" in __all__
        assert "DistributedMonitoringManager" in __all__


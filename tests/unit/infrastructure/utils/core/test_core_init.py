#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层utils/core/__init__.py模块测试

测试目标：提升utils/core/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.utils.core模块
"""

import pytest


class TestUtilsCoreInit:
    """测试utils/core模块初始化"""
    
    def test_base_component_import(self):
        """测试BaseComponent导入"""
        from src.infrastructure.utils.core import BaseComponent
        
        assert BaseComponent is not None
    
    def test_ibase_component_import(self):
        """测试IBaseComponent接口导入"""
        from src.infrastructure.utils.core import IBaseComponent
        
        assert IBaseComponent is not None
    
    def test_base_component_constants_import(self):
        """测试BaseComponentConstants导入"""
        from src.infrastructure.utils.core import BaseComponentConstants
        
        assert BaseComponentConstants is not None
    
    def test_base_component_factory_import(self):
        """测试BaseComponentFactory导入"""
        from src.infrastructure.utils.core import BaseComponentFactory
        
        assert BaseComponentFactory is not None
    
    def test_idatabase_adapter_import(self):
        """测试IDatabaseAdapter接口导入"""
        from src.infrastructure.utils.core import IDatabaseAdapter
        
        assert IDatabaseAdapter is not None
    
    def test_iresource_component_import(self):
        """测试IResourceComponent接口导入"""
        from src.infrastructure.utils.core import IResourceComponent
        
        assert IResourceComponent is not None
    
    def test_connection_status_import(self):
        """测试ConnectionStatus导入"""
        from src.infrastructure.utils.core import ConnectionStatus
        
        assert ConnectionStatus is not None
    
    def test_query_result_import(self):
        """测试QueryResult导入"""
        from src.infrastructure.utils.core import QueryResult
        
        assert QueryResult is not None
    
    def test_write_result_import(self):
        """测试WriteResult导入"""
        from src.infrastructure.utils.core import WriteResult
        
        assert WriteResult is not None
    
    def test_health_check_result_import(self):
        """测试HealthCheckResult导入"""
        from src.infrastructure.utils.core import HealthCheckResult
        
        assert HealthCheckResult is not None
    
    def test_infrastructure_error_import(self):
        """测试InfrastructureError导入"""
        from src.infrastructure.utils.core import InfrastructureError
        
        assert InfrastructureError is not None
    
    def test_configuration_error_import(self):
        """测试ConfigurationError导入"""
        from src.infrastructure.utils.core import ConfigurationError
        
        assert ConfigurationError is not None
    
    def test_data_processing_error_import(self):
        """测试DataProcessingError导入"""
        from src.infrastructure.utils.core import DataProcessingError
        
        assert DataProcessingError is not None
    
    def test_unified_error_handler_import(self):
        """测试UnifiedErrorHandler导入"""
        from src.infrastructure.utils.core import UnifiedErrorHandler
        
        assert UnifiedErrorHandler is not None
    
    def test_get_error_handler_import(self):
        """测试get_error_handler函数导入"""
        from src.infrastructure.utils.core import get_error_handler
        
        assert callable(get_error_handler)
    
    def test_storage_adapter_import(self):
        """测试StorageAdapter导入"""
        from src.infrastructure.utils.core import StorageAdapter
        
        assert StorageAdapter is not None
    
    def test_infrastructure_status_manager_import(self):
        """测试InfrastructureStatusManager导入"""
        from src.infrastructure.utils.core import InfrastructureStatusManager
        
        assert InfrastructureStatusManager is not None
    
    def test_base_component_with_status_import(self):
        """测试BaseComponentWithStatus导入"""
        from src.infrastructure.utils.core import BaseComponentWithStatus
        
        assert BaseComponentWithStatus is not None
    
    def test_infrastructure_duplicate_resolver_import(self):
        """测试InfrastructureDuplicateResolver导入"""
        from src.infrastructure.utils.core import InfrastructureDuplicateResolver
        
        assert InfrastructureDuplicateResolver is not None
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.utils.core import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "BaseComponent" in __all__
        assert "IDatabaseAdapter" in __all__
        assert "InfrastructureError" in __all__


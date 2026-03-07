#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层error/__init__.py模块测试

测试目标：提升error/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.error模块
"""

import pytest


class TestErrorInit:
    """测试error模块初始化"""
    
    def test_error_handler_import(self):
        """测试ErrorHandler导入"""
        from src.infrastructure.error import ErrorHandler
        
        assert ErrorHandler is not None
    
    def test_data_loader_error_import(self):
        """测试DataLoaderError导入"""
        from src.infrastructure.error import DataLoaderError
        
        assert DataLoaderError is not None
        assert issubclass(DataLoaderError, Exception)
    
    def test_data_validation_error_import(self):
        """测试DataValidationError导入"""
        from src.infrastructure.error import DataValidationError
        
        assert DataValidationError is not None
        assert issubclass(DataValidationError, Exception)
    
    def test_data_processing_error_import(self):
        """测试DataProcessingError导入"""
        from src.infrastructure.error import DataProcessingError
        
        assert DataProcessingError is not None
        assert issubclass(DataProcessingError, Exception)
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.error import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "ErrorHandler" in __all__
        assert "DataLoaderError" in __all__
        assert "DataValidationError" in __all__
        assert "DataProcessingError" in __all__
    
    def test_exception_usage(self):
        """测试异常使用"""
        from src.infrastructure.error import DataLoaderError
        
        # 测试异常可以正常抛出
        with pytest.raises(DataLoaderError):
            raise DataLoaderError("Test error")


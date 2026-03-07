#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
QualityComponents 单元测试
覆盖QualityComponent和QualityComponentFactory的核心功能。
"""

import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from datetime import datetime
from unittest.mock import patch
from typing import Dict, Any

from src.data.quality.quality_components import (
    IQualityComponent,
    QualityComponent,
    QualityComponentFactory,
)


class TestQualityComponent:
    """QualityComponent 测试"""

    def test_get_info_and_status(self):
        """测试获取组件信息和状态"""
        component = QualityComponent(quality_id=1)
        
        info = component.get_info()
        assert info["quality_id"] == 1
        assert info["component_name"] == "Quality_Component_1"
        assert info["component_type"] == "Quality"
        assert info["version"] == "2.0.0"
        assert "creation_time" in info
        
        status = component.get_status()
        assert status["quality_id"] == 1
        assert status["status"] == "active"
        assert status["health"] == "good"

    def test_process_success_and_error(self):
        """测试数据处理成功和错误场景"""
        component = QualityComponent(quality_id=6)
        test_data = {"field": "value", "number": 42}
        
        # 正常处理
        result = component.process(test_data)
        assert result["quality_id"] == 6
        assert result["status"] == "success"
        assert result["input_data"] == test_data
        assert "processed_at" in result
        
        # 模拟异常处理（通过 __getattribute__ 拦截）
        class FailingComponent(QualityComponent):
            def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
                raise ValueError("Processing failed")
        
        failing = FailingComponent(quality_id=11)
        error_result = failing.process(test_data)
        assert error_result["status"] == "error"
        assert "error" in error_result
        assert error_result["error_type"] == "ValueError"

    def test_initialize_and_validate(self):
        """测试初始化和验证功能"""
        component = QualityComponent(quality_id=16)
        
        # 初始化
        config = {"param": "value"}
        assert component.initialize(config) is True
        assert hasattr(component, "config")
        assert hasattr(component, "initialized_at")
        
        # 验证
        assert component.validate() is True
        assert component.validate({"data": "test"}) is True

    def test_get_quality_id(self):
        """测试获取quality ID"""
        component = QualityComponent(quality_id=21)
        assert component.get_quality_id() == 21


class TestQualityComponentFactory:
    """QualityComponentFactory 测试"""

    def test_factory_with_custom_ids(self):
        """测试工厂使用自定义ID列表"""
        # 默认支持的ID应该包括 [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66]
        supported = QualityComponentFactory.SUPPORTED_QUALITY_IDS
        assert isinstance(supported, list)
        assert len(supported) > 0

    def test_create_component_with_valid_id(self):
        """测试使用有效ID创建组件"""
        component = QualityComponentFactory.create_component(1)
        assert isinstance(component, QualityComponent)
        assert component.quality_id == 1
        assert component.component_type == "Quality"

    def test_create_component_with_invalid_id(self):
        """测试使用无效ID创建组件应抛出异常"""
        with pytest.raises(ValueError, match="不支持的quality ID"):
            QualityComponentFactory.create_component(999)

    def test_get_available_qualitys(self):
        """测试获取所有可用的quality ID"""
        ids = QualityComponentFactory.get_available_qualitys()
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(id_val, int) for id_val in ids)

    def test_create_all_qualitys(self):
        """测试创建所有quality组件"""
        all_components = QualityComponentFactory.create_all_qualitys()
        assert isinstance(all_components, dict)
        assert len(all_components) == len(QualityComponentFactory.SUPPORTED_QUALITY_IDS)
        
        for quality_id, component in all_components.items():
            assert isinstance(component, QualityComponent)
            assert component.quality_id == quality_id

    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = QualityComponentFactory.get_factory_info()
        assert info["factory_name"] == "QualityComponentFactory"
        assert info["version"] == "2.0.0"
        assert info["total_qualities"] == len(QualityComponentFactory.SUPPORTED_QUALITY_IDS)
        assert "supported_ids" in info
        assert "created_at" in info


class TestQualityComponentLegacyFunctions:
    """向后兼容函数的测试"""

    def test_legacy_create_functions(self):
        """测试向后兼容的创建函数"""
        from src.data.quality.quality_components import (
            create_quality_quality_component_1,
            create_quality_quality_component_6,
            create_quality_quality_component_11,
        )
        
        comp1 = create_quality_quality_component_1()
        assert isinstance(comp1, QualityComponent)
        assert comp1.quality_id == 1
        
        comp6 = create_quality_quality_component_6()
        assert comp6.quality_id == 6
        
        comp11 = create_quality_quality_component_11()
        assert comp11.quality_id == 11


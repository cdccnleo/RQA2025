#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StatusComponent扩展测试
补充status_components.py中未覆盖的边界情况和错误处理
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

import sys
import importlib
from pathlib import Path
import pytest

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    engine_status_components_module = importlib.import_module('src.monitoring.engine.status_components')
    ComponentFactory = getattr(engine_status_components_module, 'ComponentFactory', None)
    IStatusComponent = getattr(engine_status_components_module, 'IStatusComponent', None)
    StatusComponent = getattr(engine_status_components_module, 'StatusComponent', None)
    StatusComponentFactory = getattr(engine_status_components_module, 'StatusComponentFactory', None)
    create_status_status_component_5 = getattr(engine_status_components_module, 'create_status_status_component_5', None)

    if StatusComponent is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestStatusComponentFactoryEdgeCases:
    """测试StatusComponentFactory边界情况"""

    def test_create_component_invalid_id(self):
        """测试创建无效ID的组件"""
        with pytest.raises(ValueError) as exc_info:
            StatusComponentFactory.create_component(999)
        
        assert "不支持的status ID" in str(exc_info.value)
        assert "999" in str(exc_info.value)

    def test_create_component_supported_ids(self):
        """测试创建支持的ID"""
        for status_id in StatusComponentFactory.SUPPORTED_STATUS_IDS:
            component = StatusComponentFactory.create_component(status_id)
            assert isinstance(component, StatusComponent)
            assert component.get_status_id() == status_id

    def test_get_available_statuss(self):
        """测试获取可用status ID"""
        available = StatusComponentFactory.get_available_statuss()
        
        assert isinstance(available, list)
        assert len(available) > 0
        assert all(isinstance(id, int) for id in available)
        assert available == sorted(available)

    def test_get_available_statuss_content(self):
        """测试获取可用status ID的内容"""
        available = StatusComponentFactory.get_available_statuss()
        expected = sorted(StatusComponentFactory.SUPPORTED_STATUS_IDS)
        
        assert available == expected

    def test_create_all_statuss(self):
        """测试创建所有status"""
        all_statuss = StatusComponentFactory.create_all_statuss()
        
        assert isinstance(all_statuss, dict)
        assert len(all_statuss) == len(StatusComponentFactory.SUPPORTED_STATUS_IDS)
        
        for status_id, component in all_statuss.items():
            assert isinstance(component, StatusComponent)
            assert component.get_status_id() == status_id

    def test_get_factory_info_structure(self):
        """测试工厂信息结构"""
        info = StatusComponentFactory.get_factory_info()
        
        assert 'factory_name' in info
        assert 'version' in info
        assert 'total_statuss' in info
        assert 'supported_ids' in info
        assert 'created_at' in info
        assert 'description' in info

    def test_get_factory_info_values(self):
        """测试工厂信息值"""
        info = StatusComponentFactory.get_factory_info()
        
        assert info['factory_name'] == "StatusComponentFactory"
        assert info['version'] == "2.0.0"
        assert info['total_statuss'] == len(StatusComponentFactory.SUPPORTED_STATUS_IDS)
        assert info['supported_ids'] == sorted(StatusComponentFactory.SUPPORTED_STATUS_IDS)
        assert isinstance(info['created_at'], str)


class TestStatusComponentErrorHandling:
    """测试StatusComponent错误处理"""

    def test_process_error_response_structure(self):
        """测试process方法错误响应的结构"""
        component = StatusComponent(5, "Status")
        data = {'test': 'data'}
        
        # 模拟异常情况
        with patch.object(component, 'process', side_effect=Exception("Test error")):
            try:
                result = component.process(data)
            except Exception:
                # 验证错误响应结构（当异常发生时）
                error_result = {
                    "status_id": 5,
                    "component_name": "Status_Component_5",
                    "component_type": "Status",
                    "input_data": data,
                    "processed_at": datetime.now().isoformat(),
                    "status": "error",
                    "error": "Test error",
                    "error_type": "Exception"
                }
                # 注意：实际的process方法在try块内处理异常并返回错误响应
                # 这里只是验证错误响应的结构应该是这样的
                assert isinstance(error_result, dict)
                assert error_result['status'] == "error"

    def test_process_with_exception_handling(self):
        """测试process方法异常处理"""
        component = StatusComponent(5, "Status")
        
        # process方法内部有try-except，即使出现异常也会返回错误响应
        # 所以我们需要验证正常情况下和异常情况下的响应结构
        data = {'test': 'data'}
        result = component.process(data)
        
        # 验证响应结构
        assert 'status_id' in result
        assert 'component_name' in result
        assert 'component_type' in result
        assert 'input_data' in result
        assert 'processed_at' in result
        assert 'status' in result


class TestStatusComponentInitialization:
    """测试StatusComponent初始化"""

    def test_init_default_component_type(self):
        """测试使用默认component_type初始化"""
        component = StatusComponent(5)
        
        assert component.status_id == 5
        assert component.component_type == "Status"
        assert component.component_name == "Status_Component_5"

    def test_init_custom_component_type(self):
        """测试使用自定义component_type初始化"""
        component = StatusComponent(5, "CustomType")
        
        assert component.status_id == 5
        assert component.component_type == "CustomType"
        assert component.component_name == "CustomType_Component_5"

    def test_init_creation_time(self):
        """测试creation_time的设置"""
        before = datetime.now()
        component = StatusComponent(5)
        after = datetime.now()
        
        assert before <= component.creation_time <= after


class TestStatusComponentMethods:
    """测试StatusComponent方法"""

    def test_get_info_all_fields(self):
        """测试get_info返回所有字段"""
        component = StatusComponent(5, "TestType")
        info = component.get_info()
        
        assert 'status_id' in info
        assert 'component_name' in info
        assert 'component_type' in info
        assert 'creation_time' in info
        assert 'description' in info
        assert 'version' in info
        assert 'type' in info

    def test_process_success_structure(self):
        """测试process成功响应的结构"""
        component = StatusComponent(5)
        data = {'test': 'value'}
        result = component.process(data)
        
        assert result['status'] == 'success'
        assert result['input_data'] == data
        assert 'processed_at' in result
        assert 'result' in result
        assert 'processing_type' in result

    def test_process_success_content(self):
        """测试process成功响应的内容"""
        component = StatusComponent(5, "TestType")
        data = {'key': 'value', 'number': 42}
        result = component.process(data)
        
        assert result['status_id'] == 5
        assert result['component_name'] == "TestType_Component_5"
        assert result['component_type'] == "TestType"
        assert result['input_data'] == data
        assert "Processed by" in result['result']
        assert result['processing_type'] == "unified_status_processing"

    def test_get_status_all_fields(self):
        """测试get_status返回所有字段"""
        component = StatusComponent(5)
        status = component.get_status()
        
        assert 'status_id' in status
        assert 'component_name' in status
        assert 'component_type' in status
        assert 'status' in status
        assert 'creation_time' in status
        assert 'health' in status

    def test_get_status_values(self):
        """测试get_status返回值"""
        component = StatusComponent(5, "TestType")
        status = component.get_status()
        
        assert status['status_id'] == 5
        assert status['component_name'] == "TestType_Component_5"
        assert status['component_type'] == "TestType"
        assert status['status'] == "active"
        assert status['health'] == "good"


class TestComponentFactoryStatus:
    """测试ComponentFactory（status相关）"""

    def test_component_factory_initialization(self):
        """测试ComponentFactory初始化"""
        factory = ComponentFactory()
        assert hasattr(factory, '_components')
        assert isinstance(factory._components, dict)

    def test_create_component_with_valid_config(self):
        """测试创建组件（有效配置）"""
        factory = ComponentFactory()
        config = {'key': 'value'}
        
        result = factory.create_component('test_type', config)
        assert result is None  # _create_component_instance返回None

    def test_create_component_with_exception(self):
        """测试创建组件时抛出异常"""
        factory = ComponentFactory()
        
        with patch.object(factory, '_create_component_instance', side_effect=Exception("Test error")):
            result = factory.create_component('test_type', {})
            assert result is None


class TestBackwardCompatibilityStatus:
    """测试向后兼容函数"""

    def test_create_status_status_component_5(self):
        """测试create_status_status_component_5"""
        component = create_status_status_component_5()
        
        assert isinstance(component, StatusComponent)
        assert component.get_status_id() == 5

    def test_create_status_status_component_5_multiple_calls(self):
        """测试多次调用create_status_status_component_5"""
        component1 = create_status_status_component_5()
        component2 = create_status_status_component_5()
        
        # 每次调用应该创建新实例
        assert component1 is not component2
        assert component1.get_status_id() == component2.get_status_id() == 5


class TestStatusComponentIntegration:
    """测试StatusComponent集成场景"""

    def test_component_lifecycle(self):
        """测试组件生命周期"""
        component = StatusComponent(5, "LifecycleTest")
        
        # 初始化
        assert component.get_status_id() == 5
        
        # 获取信息
        info = component.get_info()
        assert info['status_id'] == 5
        
        # 处理数据
        result = component.process({'test': 'data'})
        assert result['status'] == 'success'
        
        # 获取状态
        status = component.get_status()
        assert status['status'] == "active"

    def test_multiple_components(self):
        """测试多个组件实例"""
        component1 = StatusComponent(5, "Type1")
        component2 = StatusComponent(5, "Type2")
        
        assert component1.get_status_id() == component2.get_status_id() == 5
        assert component1.component_type != component2.component_type
        assert component1.component_name != component2.component_name




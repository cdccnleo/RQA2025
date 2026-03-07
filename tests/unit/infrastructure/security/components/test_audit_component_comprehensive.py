#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计组件综合测试
测试AuditComponent、ComponentFactory和AuditComponentFactory的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, patch

from src.infrastructure.security.components.audit_component import (
    ComponentFactory,
    IAuditComponent,
    AuditComponent,
    AuditComponentFactory
)


class TestComponentFactory:
    """测试通用组件工厂"""

    def test_initialization(self):
        """测试初始化"""
        factory = ComponentFactory()

        assert hasattr(factory, '_components')
        assert isinstance(factory._components, dict)
        assert len(factory._components) == 0

    def test_create_component_abstract_method(self):
        """测试创建组件的抽象方法"""
        factory = ComponentFactory()

        # 由于_create_component_instance返回None，create_component应该返回None
        result = factory.create_component("test_type", {"config": "value"})

        assert result is None

    def test_create_component_with_exception(self):
        """测试创建组件时发生异常"""
        factory = ComponentFactory()

        # 模拟异常情况
        with patch.object(factory, '_create_component_instance', side_effect=Exception("Test error")):
            result = factory.create_component("test_type", {"config": "value"})

            assert result is None

    def test_create_component_instance_default(self):
        """测试创建组件实例的默认实现"""
        factory = ComponentFactory()

        result = factory._create_component_instance("test_type", {"config": "value"})

        assert result is None


class TestIAuditComponent:
    """测试审计组件接口"""

    def test_interface_is_abstract(self):
        """测试接口是抽象的"""
        # IAuditComponent是抽象类，不能直接实例化
        with pytest.raises(TypeError):
            IAuditComponent()


class TestAuditComponent:
    """测试审计组件实现"""

    def test_initialization_minimal(self):
        """测试最小化初始化"""
        component = AuditComponent(audit_id=4)  # 使用支持的ID

        assert component.audit_id == 4
        assert component.component_type == "Audit"
        assert component.component_name == "Audit_Component_4"
        assert component.creation_time is not None
        assert isinstance(component.creation_time, datetime)

    def test_initialization_complete(self):
        """测试完整初始化"""
        component = AuditComponent(
            audit_id=10,  # 使用支持的ID
            component_type="CustomAudit"
        )

        assert component.audit_id == 10
        assert component.component_type == "CustomAudit"
        assert component.component_name == "CustomAudit_Component_10"

    def test_get_audit_id(self):
        """测试获取审计ID"""
        component = AuditComponent(audit_id=16)  # 使用支持的ID

        audit_id = component.get_audit_id()

        assert audit_id == 16

    def test_get_info(self):
        """测试获取组件信息"""
        component = AuditComponent(audit_id=22, component_type="TestAudit")  # 使用支持的ID

        info = component.get_info()

        assert isinstance(info, dict)
        assert info["audit_id"] == 22
        assert info["component_type"] == "TestAudit"
        assert info["component_name"] == "TestAudit_Component_22"
        assert "creation_time" in info
        assert "description" in info
        assert "version" in info

    def test_process_data(self):
        """测试处理数据"""
        component = AuditComponent(audit_id=28)  # 使用支持的ID

        input_data = {"action": "test", "user": "testuser", "resource": "/api/test"}
        result = component.process(input_data)

        assert isinstance(result, dict)
        assert result["audit_id"] == 28
        assert result["input_data"] == input_data
        assert result["processed_at"] is not None
        assert result["status"] == "success"
        assert "result" in result
        assert "processing_type" in result

    def test_process_data_multiple(self):
        """测试处理多个数据"""
        component = AuditComponent(audit_id=34)  # 使用支持的ID

        # 处理多个数据
        for i in range(3):
            input_data = {"action": f"action_{i}", "count": i}
            result = component.process(input_data)
            assert result["status"] == "success"
            assert result["input_data"]["action"] == f"action_{i}"

    def test_get_status(self):
        """测试获取状态"""
        component = AuditComponent(audit_id=40, component_type="TestComponent")  # 使用支持的ID

        status = component.get_status()

        assert isinstance(status, dict)
        assert status["audit_id"] == 40
        assert status["component_type"] == "TestComponent"
        assert status["status"] == "active"
        assert status["component_name"] == "TestComponent_Component_40"
        assert "creation_time" in status
        assert "health" in status

    def test_get_status_after_processing(self):
        """测试处理数据后获取状态"""
        component = AuditComponent(audit_id=46)  # 使用支持的ID

        # 处理一些数据
        component.process({"action": "test"})
        component.process({"action": "test2"})

        status = component.get_status()

        # 状态应该仍然是active（实现是固定的）
        assert status["status"] == "active"
        assert status["audit_id"] == 46

    def test_component_lifecycle(self):
        """测试组件生命周期"""
        component = AuditComponent(audit_id=52)  # 使用支持的ID

        # 获取初始信息
        info = component.get_info()
        assert info["audit_id"] == 52

        # 处理数据
        result = component.process({"action": "start"})
        assert result["status"] == "success"

        # 获取状态
        status = component.get_status()
        assert status["status"] == "active"

        # 多次处理
        for i in range(5):
            result = component.process({"action": f"process_{i}"})
            assert result["status"] == "success"


class TestAuditComponentFactory:
    """测试审计组件工厂"""

    def test_create_component(self):
        """测试创建组件"""
        component = AuditComponentFactory.create_component(4)  # 使用支持的ID

        assert isinstance(component, AuditComponent)
        assert component.audit_id == 4
        assert component.component_type == "Audit"

    def test_get_available_audits(self):
        """测试获取可用审计"""
        audits = AuditComponentFactory.get_available_audits()

        assert isinstance(audits, list)
        # 默认应该返回一些审计ID
        assert len(audits) > 0
        assert all(isinstance(audit_id, int) for audit_id in audits)

    def test_create_all_audits(self):
        """测试创建所有审计组件"""
        all_audits = AuditComponentFactory.create_all_audits()

        assert isinstance(all_audits, dict)
        assert len(all_audits) > 0

        # 检查每个值都是AuditComponent实例
        for audit_id, component in all_audits.items():
            assert isinstance(audit_id, int)
            assert isinstance(component, AuditComponent)
            assert component.audit_id == audit_id

    def test_create_all_audits_completeness(self):
        """测试创建所有审计的完整性"""
        available_audits = AuditComponentFactory.get_available_audits()
        all_audits = AuditComponentFactory.create_all_audits()

        # 创建的审计数量应该等于可用审计数量
        assert len(all_audits) == len(available_audits)

        # 所有的可用审计ID都应该在创建的结果中
        assert set(all_audits.keys()) == set(available_audits)

    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = AuditComponentFactory.get_factory_info()

        assert isinstance(info, dict)
        assert "factory_name" in info
        assert "version" in info
        assert "total_audits" in info
        assert "supported_ids" in info
        assert "created_at" in info
        assert "description" in info

    def test_factory_info_content(self):
        """测试工厂信息内容"""
        info = AuditComponentFactory.get_factory_info()

        assert info["factory_name"] == "AuditComponentFactory"
        assert info["version"] == "2.0.0"
        assert isinstance(info["supported_ids"], list)
        assert isinstance(info["total_audits"], int)
        assert info["total_audits"] == 10  # SUPPORTED_AUDIT_IDS的数量
        assert info["description"] == "统一审计组件工厂"


class TestAuditComponentIntegration:
    """测试审计组件集成功能"""

    def test_component_factory_workflow(self):
        """测试组件工厂工作流"""
        # 获取可用审计
        available_audits = AuditComponentFactory.get_available_audits()
        assert len(available_audits) > 0

        # 创建所有审计
        all_components = AuditComponentFactory.create_all_audits()
        assert len(all_components) == len(available_audits)

        # 验证每个组件都能正常工作
        for audit_id, component in all_components.items():
            # 测试基本功能
            info = component.get_info()
            assert info["audit_id"] == audit_id

            # 测试处理数据
            test_data = {"action": f"test_action_{audit_id}", "user": "test_user"}
            result = component.process(test_data)
            assert result["status"] == "success"

            # 测试状态获取
            status = component.get_status()
            assert status["audit_id"] == audit_id
            assert status["process_count"] >= 1

    def test_component_lifecycle_management(self):
        """测试组件生命周期管理"""
        # 创建组件
        component = AuditComponentFactory.create_component(4)  # 使用支持的ID

        # 初始状态
        initial_status = component.get_status()
        assert initial_status["status"] == "active"
        assert initial_status["process_count"] == 0

        # 处理一系列操作
        operations = [
            {"action": "login", "user": "user1", "success": True},
            {"action": "access", "resource": "/api/data", "method": "GET"},
            {"action": "modify", "resource": "/api/user", "changes": ["email"]},
            {"action": "logout", "user": "user1", "session_duration": 3600}
        ]

        for op in operations:
            component.process(op)

        # 最终状态
        final_status = component.get_status()
        assert final_status["status"] == "processed"
        assert final_status["process_count"] == len(operations)

        # 验证元数据
        assert component.metadata.get("process_count") == len(operations)

    def test_multiple_components_isolation(self):
        """测试多个组件的隔离性"""
        component1 = AuditComponentFactory.create_component(4)
        component2 = AuditComponentFactory.create_component(10)

        # 组件应该相互隔离
        assert component1.audit_id != component2.audit_id
        assert component1.metadata is not component2.metadata

        # 处理不同数据
        component1.process({"action": "component1_action"})
        component2.process({"action": "component2_action"})
        component2.process({"action": "component2_action2"})

        # 验证状态独立性
        status1 = component1.get_status()
        status2 = component2.get_status()

        assert status1["audit_id"] == 4
        assert status2["audit_id"] == 10


class TestErrorHandling:
    """测试错误处理"""

    def test_component_creation_edge_cases(self):
        """测试组件创建边界情况"""
        # 测试有效ID
        component = AuditComponent(audit_id=0)
        assert component.audit_id == 0

        component = AuditComponent(audit_id=-1)
        assert component.audit_id == -1

        # 测试大ID
        large_id = 999999
        component = AuditComponent(audit_id=large_id)
        assert component.audit_id == large_id

    def test_process_invalid_data(self):
        """测试处理无效数据"""
        component = AuditComponent(audit_id=16)  # 使用支持的ID

        # 处理None数据
        result = component.process(None)
        assert result["status"] == "success"  # 应该能处理None

        # 处理空字典
        result = component.process({})
        assert result["status"] == "success"

        # 处理复杂数据
        complex_data = {
            "nested": {"deep": {"value": 123}},
            "list": [1, 2, 3],
            "datetime": datetime.now()
        }
        result = component.process(complex_data)
        assert result["status"] == "success"
        assert result["input_data"] == complex_data

    def test_factory_edge_cases(self):
        """测试工厂边界情况"""
        # 创建所有审计多次
        audits1 = AuditComponentFactory.create_all_audits()
        audits2 = AuditComponentFactory.create_all_audits()

        # 结果应该是一致的
        assert len(audits1) == len(audits2)
        assert set(audits1.keys()) == set(audits2.keys())

        # 但是对象应该是不同的实例
        for audit_id in audits1.keys():
            assert audits1[audit_id] is not audits2[audit_id]

    def test_info_and_status_consistency(self):
        """测试信息和状态的一致性"""
        component = AuditComponent(audit_id=999)

        info = component.get_info()
        status = component.get_status()

        # 基本字段应该一致
        assert info["audit_id"] == status["audit_id"]
        assert info["component_type"] == status["component_type"]
        assert info["status"] == status["status"]

        # 处理数据后仍然一致
        component.process({"action": "test"})
        info_after = component.get_info()
        status_after = component.get_status()

        assert info_after["audit_id"] == status_after["audit_id"]
        # 状态总是"active"
        assert status_after["status"] == "active"


class TestPerformance:
    """测试性能"""

    def test_component_creation_performance(self):
        """测试组件创建性能"""
        import time

        start_time = time.time()

        # 创建所有支持的组件多次
        components = {}
        available_ids = AuditComponentFactory.get_available_audits()
        for i, audit_id in enumerate(available_ids):
            component = AuditComponentFactory.create_component(audit_id)
            components[i] = component

        end_time = time.time()

        # 100个组件创建应该在1秒内完成
        duration = end_time - start_time
        assert duration < 1.0

        # 验证所有支持的组件都创建成功
        assert len(components) == len(available_ids)

    def test_data_processing_performance(self):
        """测试数据处理性能"""
        component = AuditComponent(audit_id=22)  # 使用支持的ID

        import time

        # 处理1000个数据项
        test_data = {"action": "performance_test", "counter": 0}

        start_time = time.time()
        for i in range(1000):
            test_data["counter"] = i
            result = component.process(test_data)
            assert result["status"] == "success"

        end_time = time.time()

        # 1000个数据处理应该在2秒内完成
        duration = end_time - start_time
        assert duration < 2.0

        # 验证处理计数
        status = component.get_status()
        assert status["process_count"] == 1000

    def test_factory_operations_performance(self):
        """测试工厂操作性能"""
        import time

        start_time = time.time()

        # 执行多次工厂操作
        for _ in range(50):
            available = AuditComponentFactory.get_available_audits()
            assert len(available) > 0

            info = AuditComponentFactory.get_factory_info()
            assert isinstance(info, dict)

        end_time = time.time()

        # 50次工厂操作应该在1秒内完成
        duration = end_time - start_time
        assert duration < 1.0

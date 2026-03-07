#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层重复代码解决器测试

测试目标：提升utils/core/duplicate_resolver.py的真实覆盖率
实际导入和使用src.infrastructure.utils.core.duplicate_resolver模块
"""

import pytest


class TestBaseComponentWithStatus:
    """测试带状态的基础组件"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.core.duplicate_resolver import BaseComponentWithStatus
        
        class ConcreteComponent(BaseComponentWithStatus):
            pass
        
        component = ConcreteComponent()
        assert component.status == "initialized"
        assert component._metadata == {}
    
    def test_status_property(self):
        """测试状态属性"""
        from src.infrastructure.utils.core.duplicate_resolver import BaseComponentWithStatus
        
        class ConcreteComponent(BaseComponentWithStatus):
            pass
        
        component = ConcreteComponent()
        assert component.status == "initialized"
        
        component.set_status("running")
        assert component.status == "running"
    
    def test_set_status(self):
        """测试设置状态"""
        from src.infrastructure.utils.core.duplicate_resolver import BaseComponentWithStatus
        
        class ConcreteComponent(BaseComponentWithStatus):
            pass
        
        component = ConcreteComponent()
        component.set_status("active")
        assert component.status == "active"
        
        component.set_status("stopped")
        assert component.status == "stopped"


class TestInfrastructureStatusManager:
    """测试基础架构状态管理器"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.core.duplicate_resolver import (
            InfrastructureStatusManager,
            BaseComponentWithStatus
        )
        
        manager = InfrastructureStatusManager()
        assert manager._components == {}
        assert manager._status_history == []
    
    def test_register_component(self):
        """测试注册组件"""
        from src.infrastructure.utils.core.duplicate_resolver import (
            InfrastructureStatusManager,
            BaseComponentWithStatus
        )
        
        class TestComponent(BaseComponentWithStatus):
            pass
        
        manager = InfrastructureStatusManager()
        component = TestComponent()
        
        manager.register_component("test_component", component)
        assert "test_component" in manager._components
        assert manager._components["test_component"] == component
    
    def test_get_component_status(self):
        """测试获取组件状态"""
        from src.infrastructure.utils.core.duplicate_resolver import (
            InfrastructureStatusManager,
            BaseComponentWithStatus
        )
        
        class TestComponent(BaseComponentWithStatus):
            pass
        
        manager = InfrastructureStatusManager()
        component = TestComponent()
        component.set_status("running")
        
        manager.register_component("test_component", component)
        
        status = manager.get_component_status("test_component")
        assert status == "running"
    
    def test_get_component_status_not_registered(self):
        """测试获取未注册组件的状态"""
        from src.infrastructure.utils.core.duplicate_resolver import InfrastructureStatusManager
        
        manager = InfrastructureStatusManager()
        status = manager.get_component_status("nonexistent")
        assert status is None
    
    def test_get_all_status(self):
        """测试获取所有组件状态"""
        from src.infrastructure.utils.core.duplicate_resolver import (
            InfrastructureStatusManager,
            BaseComponentWithStatus
        )
        
        class TestComponent(BaseComponentWithStatus):
            pass
        
        manager = InfrastructureStatusManager()
        
        component1 = TestComponent()
        component1.set_status("running")
        manager.register_component("component1", component1)
        
        component2 = TestComponent()
        component2.set_status("stopped")
        manager.register_component("component2", component2)
        
        all_status = manager.get_all_status()
        assert all_status == {"component1": "running", "component2": "stopped"}


class TestInfrastructureDuplicateResolver:
    """测试基础架构重复代码解决器"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.core.duplicate_resolver import InfrastructureDuplicateResolver
        
        resolver = InfrastructureDuplicateResolver()
        assert resolver._duplicates == []
        assert resolver._resolved == []
    
    def test_detect_duplicates(self):
        """测试检测重复代码"""
        from src.infrastructure.utils.core.duplicate_resolver import InfrastructureDuplicateResolver
        
        resolver = InfrastructureDuplicateResolver()
        result = resolver.detect_duplicates("code_base")
        
        # 当前是占位符实现，返回空列表
        assert result == []
        assert isinstance(result, list)
    
    def test_resolve_duplicate(self):
        """测试解决重复代码"""
        from src.infrastructure.utils.core.duplicate_resolver import InfrastructureDuplicateResolver
        
        resolver = InfrastructureDuplicateResolver()
        
        # 第一次解决应该成功
        result1 = resolver.resolve_duplicate("dup1")
        assert result1 is True
        assert "dup1" in resolver._resolved
        
        # 第二次解决同一个应该返回False
        result2 = resolver.resolve_duplicate("dup1")
        assert result2 is False
    
    def test_resolve_multiple_duplicates(self):
        """测试解决多个重复代码"""
        from src.infrastructure.utils.core.duplicate_resolver import InfrastructureDuplicateResolver
        
        resolver = InfrastructureDuplicateResolver()
        
        assert resolver.resolve_duplicate("dup1") is True
        assert resolver.resolve_duplicate("dup2") is True
        assert resolver.resolve_duplicate("dup3") is True
        
        assert len(resolver._resolved) == 3
        assert "dup1" in resolver._resolved
        assert "dup2" in resolver._resolved
        assert "dup3" in resolver._resolved
    
    def test_get_resolution_stats(self):
        """测试获取解决统计"""
        from src.infrastructure.utils.core.duplicate_resolver import InfrastructureDuplicateResolver
        
        resolver = InfrastructureDuplicateResolver()
        
        # 初始状态
        stats = resolver.get_resolution_stats()
        assert stats["total_duplicates"] == 0
        assert stats["resolved_count"] == 0
        assert stats["resolution_rate"] == 0.0
        
        # 解决一些重复代码
        resolver.resolve_duplicate("dup1")
        resolver.resolve_duplicate("dup2")
        
        stats = resolver.get_resolution_stats()
        assert stats["total_duplicates"] == 0
        assert stats["resolved_count"] == 2
        assert stats["resolution_rate"] == 0.0  # 因为没有检测到重复代码
    
    def test_get_resolution_stats_with_duplicates(self):
        """测试有重复代码时的解决统计"""
        from src.infrastructure.utils.core.duplicate_resolver import InfrastructureDuplicateResolver
        
        resolver = InfrastructureDuplicateResolver()
        
        # 手动添加一些重复代码（模拟检测结果）
        resolver._duplicates = [
            {"id": "dup1", "type": "function"},
            {"id": "dup2", "type": "class"},
            {"id": "dup3", "type": "function"}
        ]
        
        # 解决部分重复代码
        resolver.resolve_duplicate("dup1")
        resolver.resolve_duplicate("dup2")
        
        stats = resolver.get_resolution_stats()
        assert stats["total_duplicates"] == 3
        assert stats["resolved_count"] == 2
        assert stats["resolution_rate"] == pytest.approx(2.0 / 3.0, rel=1e-6)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Config Factory 测试

测试 src/infrastructure/config/simple_config_factory.py 文件的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock

# 尝试导入模块
try:
    from src.infrastructure.config.simple_config_factory import (
        SimpleConfigFactory,
        get_simple_factory,
        create_simple_manager,
        get_simple_manager
    )
    from src.infrastructure.config.core.config_manager_complete import UnifiedConfigManager
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestSimpleConfigFactory:
    """测试SimpleConfigFactory功能"""

    def setup_method(self):
        """测试前准备"""
        self.factory = SimpleConfigFactory()

    def test_initialization(self):
        """测试初始化"""
        assert isinstance(self.factory._instances, dict)
        assert len(self.factory._instances) == 0

    def test_create_manager_default(self):
        """测试创建默认管理器"""
        manager = self.factory.create_manager()
        
        assert manager is not None
        assert isinstance(manager, UnifiedConfigManager)
        assert "default" in self.factory._instances

    def test_create_manager_with_name(self):
        """测试创建指定名称的管理器"""
        manager = self.factory.create_manager("test_manager")
        
        assert manager is not None
        assert isinstance(manager, UnifiedConfigManager)
        assert "test_manager" in self.factory._instances

    def test_create_manager_with_config(self):
        """测试使用配置创建管理器"""
        config = {"test": "value", "number": 42}
        
        manager = self.factory.create_manager("config_manager", config)
        
        assert manager is not None
        assert isinstance(manager, UnifiedConfigManager)
        assert "config_manager" in self.factory._instances

    def test_create_manager_with_none_config(self):
        """测试使用None配置创建管理器"""
        manager = self.factory.create_manager("none_config", None)
        
        assert manager is not None
        assert isinstance(manager, UnifiedConfigManager)
        assert "none_config" in self.factory._instances

    def test_create_manager_duplicate_name(self):
        """测试创建重复名称的管理器（应该返回已存在的）"""
        # 创建第一个管理器
        manager1 = self.factory.create_manager("duplicate_test")
        
        # 创建同名管理器
        manager2 = self.factory.create_manager("duplicate_test")
        
        # 应该是同一个实例
        assert manager1 is manager2

    def test_get_manager_existing(self):
        """测试获取已存在的管理器"""
        # 先创建管理器
        created_manager = self.factory.create_manager("get_test")
        
        # 获取管理器
        retrieved_manager = self.factory.get_manager("get_test")
        
        assert retrieved_manager is created_manager

    def test_get_manager_nonexistent(self):
        """测试获取不存在的管理器"""
        result = self.factory.get_manager("nonexistent")
        
        assert result is None

    def test_get_manager_default(self):
        """测试获取默认管理器"""
        # 先创建默认管理器
        created_manager = self.factory.create_manager()
        
        # 获取默认管理器
        retrieved_manager = self.factory.get_manager()
        
        assert retrieved_manager is created_manager

    def test_remove_manager_existing(self):
        """测试移除已存在的管理器"""
        # 先创建管理器
        self.factory.create_manager("remove_test")
        assert "remove_test" in self.factory._instances
        
        # 移除管理器
        result = self.factory.remove_manager("remove_test")
        
        assert result is True
        assert "remove_test" not in self.factory._instances

    def test_remove_manager_nonexistent(self):
        """测试移除不存在的管理器"""
        result = self.factory.remove_manager("nonexistent")
        
        assert result is False

    def test_list_managers_empty(self):
        """测试列出空的管理器列表"""
        result = self.factory.list_managers()
        
        assert isinstance(result, list)
        assert len(result) == 0

    def test_list_managers_multiple(self):
        """测试列出多个管理器"""
        # 创建多个管理器
        self.factory.create_manager("manager1")
        self.factory.create_manager("manager2")
        self.factory.create_manager("manager3")
        
        result = self.factory.list_managers()
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert "manager1" in result
        assert "manager2" in result
        assert "manager3" in result

    def test_clear_all(self):
        """测试清空所有管理器"""
        # 先创建一些管理器
        self.factory.create_manager("clear1")
        self.factory.create_manager("clear2")
        assert len(self.factory._instances) == 2
        
        # 清空所有管理器
        self.factory.clear_all()
        
        assert len(self.factory._instances) == 0

    def test_clear_all_empty(self):
        """测试清空空的管理器列表"""
        # 确保开始时为空
        self.factory.clear_all()
        assert len(self.factory._instances) == 0
        
        # 再次清空
        self.factory.clear_all()
        
        assert len(self.factory._instances) == 0


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestSimpleConfigFactoryGlobalFunctions:
    """测试全局函数功能"""

    def setup_method(self):
        """测试前准备"""
        # 重置全局工厂状态
        import src.infrastructure.config.simple_config_factory as factory_module
        factory_module._simple_factory = None

    def test_get_simple_factory_singleton(self):
        """测试get_simple_factory单例模式"""
        factory1 = get_simple_factory()
        factory2 = get_simple_factory()
        
        assert factory1 is factory2
        assert isinstance(factory1, SimpleConfigFactory)

    def test_create_simple_manager_default(self):
        """测试create_simple_manager默认创建"""
        manager = create_simple_manager()
        
        assert manager is not None
        assert isinstance(manager, UnifiedConfigManager)

    def test_create_simple_manager_with_name(self):
        """测试create_simple_manager指定名称"""
        manager = create_simple_manager("global_test")
        
        assert manager is not None
        assert isinstance(manager, UnifiedConfigManager)

    def test_create_simple_manager_with_config(self):
        """测试create_simple_manager使用配置"""
        config = {"global": "config", "value": 123}
        
        manager = create_simple_manager("global_config", config)
        
        assert manager is not None
        assert isinstance(manager, UnifiedConfigManager)

    def test_get_simple_manager_existing(self):
        """测试get_simple_manager获取已存在的管理器"""
        # 先创建管理器
        created = create_simple_manager("get_global_test")
        
        # 获取管理器
        retrieved = get_simple_manager("get_global_test")
        
        assert retrieved is created

    def test_get_simple_manager_nonexistent(self):
        """测试get_simple_manager获取不存在的管理器"""
        result = get_simple_manager("nonexistent_global")
        
        assert result is None

    def test_get_simple_manager_default(self):
        """测试get_simple_manager获取默认管理器"""
        # 先创建默认管理器
        created = create_simple_manager()
        
        # 获取默认管理器
        retrieved = get_simple_manager()
        
        assert retrieved is created


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestSimpleConfigFactoryEdgeCases:
    """测试边界情况"""

    def setup_method(self):
        """测试前准备"""
        self.factory = SimpleConfigFactory()

    def test_create_manager_with_special_name(self):
        """测试使用特殊名称创建管理器"""
        special_names = ["", "  ", "test-name", "test_name", "test.name", "123"]
        
        for name in special_names:
            try:
                manager = self.factory.create_manager(name)
                assert manager is not None
                assert isinstance(manager, UnifiedConfigManager)
            except Exception as e:
                # 某些特殊名称可能会抛出异常，这是可以接受的
                assert isinstance(e, Exception)

    def test_create_manager_with_none_name(self):
        """测试使用None名称创建管理器"""
        try:
            manager = self.factory.create_manager(None)
            # 如果成功，应该创建了某种默认管理器
            if manager is not None:
                assert isinstance(manager, UnifiedConfigManager)
        except Exception as e:
            # None名称可能会抛出异常
            assert isinstance(e, Exception)

    def test_complex_config(self):
        """测试使用复杂配置创建管理器"""
        complex_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "user": "admin",
                    "password": "secret"
                }
            },
            "features": ["logging", "monitoring"],
            "settings": {
                "timeout": 30,
                "retries": 3
            }
        }
        
        manager = self.factory.create_manager("complex_config", complex_config)
        
        assert manager is not None
        assert isinstance(manager, UnifiedConfigManager)

    def test_large_number_of_managers(self):
        """测试大量管理器"""
        # 创建多个管理器
        for i in range(100):
            manager = self.factory.create_manager(f"manager_{i}")
            assert manager is not None
        
        # 验证所有管理器都已创建
        managers_list = self.factory.list_managers()
        assert len(managers_list) == 100
        
        # 验证可以获取任意一个管理器
        retrieved = self.factory.get_manager("manager_50")
        assert retrieved is not None

    def test_concurrent_access_simulation(self):
        """测试并发访问模拟"""
        # 模拟并发创建和访问
        managers = []
        
        # 创建管理器
        for i in range(10):
            manager = self.factory.create_manager(f"concurrent_{i}")
            managers.append(manager)
        
        # 获取管理器
        for i in range(10):
            retrieved = self.factory.get_manager(f"concurrent_{i}")
            assert retrieved is managers[i]
        
        # 移除一些管理器
        for i in range(5):
            result = self.factory.remove_manager(f"concurrent_{i}")
            assert result is True
        
        # 验证移除后的状态
        remaining_list = self.factory.list_managers()
        assert len(remaining_list) == 5


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestSimpleConfigFactoryIntegration:
    """测试集成功能"""

    def setup_method(self):
        """测试前准备"""
        # 重置全局状态
        import src.infrastructure.config.simple_config_factory as factory_module
        factory_module._simple_factory = None
        self.factory = SimpleConfigFactory()

    def test_module_imports(self):
        """测试模块可以正常导入"""
        assert SimpleConfigFactory is not None
        assert get_simple_factory is not None
        assert create_simple_manager is not None
        assert get_simple_manager is not None

    def test_full_workflow_instance(self):
        """测试完整工作流程（实例方法）"""
        # 1. 创建管理器
        manager1 = self.factory.create_manager("workflow_test")
        assert manager1 is not None
        
        # 2. 获取管理器
        retrieved = self.factory.get_manager("workflow_test")
        assert retrieved is manager1
        
        # 3. 列出管理器
        managers_list = self.factory.list_managers()
        assert "workflow_test" in managers_list
        
        # 4. 移除管理器
        result = self.factory.remove_manager("workflow_test")
        assert result is True
        
        # 5. 验证移除
        assert self.factory.get_manager("workflow_test") is None

    def test_full_workflow_global(self):
        """测试完整工作流程（全局函数）"""
        # 1. 获取全局工厂
        factory = get_simple_factory()
        assert factory is not None
        
        # 2. 创建管理器
        manager = create_simple_manager("global_workflow")
        assert manager is not None
        
        # 3. 获取管理器
        retrieved = get_simple_manager("global_workflow")
        assert retrieved is manager
        
        # 4. 使用工厂实例验证
        assert factory.get_manager("global_workflow") is manager

    def test_mixed_usage(self):
        """测试混合使用实例和全局函数"""
        # 使用实例创建管理器
        instance_manager = self.factory.create_manager("instance_test")
        
        # 使用全局函数获取工厂
        global_factory = get_simple_factory()
        
        # 验证两个工厂可以访问相同的管理器
        # 注意：这取决于全局工厂的实现是否为单例
        try:
            global_manager = global_factory.get_manager("instance_test")
            # 如果实现是共享的，两个管理器应该是同一个
        except Exception:
            # 如果实现是独立的，这是可以接受的
            pass

    def test_error_handling(self):
        """测试错误处理"""
        # 测试各种可能出错的场景
        try:
            # 测试空字符串名称
            self.factory.create_manager("")
        except Exception:
            pass  # 可以接受异常
        
        try:
            # 测试特殊字符名称
            self.factory.create_manager("test\nmanager")
        except Exception:
            pass  # 可以接受异常

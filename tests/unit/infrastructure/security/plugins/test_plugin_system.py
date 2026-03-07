#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 插件系统测试

全面测试插件系统的所有功能，包括：
- 插件信息管理
- 插件加载和卸载
- 能力注册和调用
- 插件发现
- 依赖验证
- 错误处理
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import importlib.util
import sys
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
from typing import Dict, Any

from src.infrastructure.security.plugins.plugin_system import (
    PluginInfo, SecurityPlugin, PluginManager, get_plugin_manager,
    load_security_plugin, unload_security_plugin, call_plugin_capability
)


class TestPluginInfo:
    """PluginInfo测试"""

    def test_plugin_info_creation(self):
        """测试插件信息创建"""
        info = PluginInfo(
            name="test_plugin",
            version="1.0.0",
            description="测试插件",
            author="Test Author",
            dependencies=["dep1", "dep2"],
            capabilities=["auth", "audit"]
        )

        assert info.name == "test_plugin"
        assert info.version == "1.0.0"
        assert info.description == "测试插件"
        assert info.author == "Test Author"
        assert info.dependencies == ["dep1", "dep2"]
        assert info.capabilities == ["auth", "audit"]
        assert info.config_schema == {}

    def test_plugin_info_defaults(self):
        """测试插件信息默认值"""
        info = PluginInfo(
            name="minimal_plugin",
            version="1.0.0",
            description="最小插件",
            author="Anonymous"
        )

        assert info.dependencies == []
        assert info.capabilities == []
        assert info.config_schema == {}


class MockSecurityPlugin(SecurityPlugin):
    """模拟安全插件"""

    def __init__(self, plugin_name="mock_plugin"):
        self.name = plugin_name
        self.initialized = False
        self.shutdown_called = False

    @property
    def plugin_info(self) -> PluginInfo:
        return PluginInfo(
            name=self.name,
            version="1.0.0",
            description=f"模拟{self.name}插件",
            author="Test Author",
            capabilities=["mock_capability"]
        )

    def initialize(self, config: Dict[str, Any]) -> bool:
        self.initialized = True
        self.config = config
        return True

    def shutdown(self) -> None:
        self.shutdown_called = True

    def mock_capability(self):
        return f"mock result from {self.name}"


class TestSecurityPlugin:
    """SecurityPlugin测试"""

    def test_abstract_methods(self):
        """测试抽象方法"""
        # SecurityPlugin是抽象类，不能直接实例化
        with pytest.raises(TypeError):
            SecurityPlugin()

    def test_mock_plugin_implementation(self):
        """测试模拟插件实现"""
        plugin = MockSecurityPlugin("test_mock")

        # 测试插件信息
        info = plugin.plugin_info
        assert info.name == "test_mock"
        assert info.version == "1.0.0"

        # 测试初始化
        config = {"key": "value"}
        assert plugin.initialize(config) == True
        assert plugin.initialized == True
        assert plugin.config == config

        # 测试关闭
        plugin.shutdown()
        assert plugin.shutdown_called == True

    def test_get_capability(self):
        """测试获取能力"""
        plugin = MockSecurityPlugin()

        # 存在的功能
        capability = plugin.get_capability("mock_capability")
        assert capability is not None
        assert callable(capability)

        # 不存在的功能
        non_existent = plugin.get_capability("non_existent")
        assert non_existent is None


class TestPluginManager:
    """PluginManager测试"""

    @pytest.fixture
    def plugin_manager(self, tmp_path):
        """插件管理器fixture"""
        return PluginManager([tmp_path])

    def test_initialization(self, plugin_manager):
        """测试初始化"""
        assert isinstance(plugin_manager.plugin_dirs, list)
        assert len(plugin_manager.plugin_dirs) >= 1
        assert isinstance(plugin_manager.loaded_plugins, dict)
        assert isinstance(plugin_manager.plugin_configs, dict)
        assert isinstance(plugin_manager.capability_registry, dict)

    def test_load_plugin_from_file(self, plugin_manager, tmp_path):
        """测试从文件加载插件"""
        # 使用已有的CustomAuthPlugin进行测试
        try:
            # 加载现有的插件
            success = plugin_manager.load_plugin("custom_auth_plugin")
            assert success == True

            # 验证插件已加载
            assert "custom_auth_plugin" in plugin_manager.loaded_plugins
            plugin = plugin_manager.get_plugin("custom_auth_plugin")
            assert plugin is not None
            assert plugin.plugin_info.name == "custom_auth"

            # 测试能力调用
            results = plugin_manager.call_capability("pre_auth_check", "test_user")
            assert len(results) == 1
            assert "allowed" in results[0]["result"]

        finally:
            # 清理
            if "custom_auth_plugin" in plugin_manager.loaded_plugins:
                plugin_manager.unload_plugin("custom_auth_plugin")

    def test_load_plugin_file_not_found(self, plugin_manager):
        """测试插件文件不存在的情况"""
        success = plugin_manager.load_plugin("nonexistent_plugin")
        assert success == False

    def test_unload_plugin(self, plugin_manager):
        """测试卸载插件"""
        # 先加载一个插件
        plugin_manager.loaded_plugins["test_plugin"] = MockSecurityPlugin("test_plugin")
        plugin_manager.plugin_configs["test_plugin"] = {}

        # 卸载插件
        success = plugin_manager.unload_plugin("test_plugin")
        assert success == True

        # 验证插件已卸载
        assert "test_plugin" not in plugin_manager.loaded_plugins

    def test_unload_nonexistent_plugin(self, plugin_manager):
        """测试卸载不存在的插件"""
        success = plugin_manager.unload_plugin("nonexistent")
        assert success == True  # 不存在的插件算作已卸载

    def test_get_plugin(self, plugin_manager):
        """测试获取插件"""
        plugin = MockSecurityPlugin("get_test")
        plugin_manager.loaded_plugins["get_test"] = plugin

        retrieved = plugin_manager.get_plugin("get_test")
        assert retrieved == plugin

        nonexistent = plugin_manager.get_plugin("nonexistent")
        assert nonexistent is None

    def test_list_plugins(self, plugin_manager):
        """测试列出插件"""
        plugin = MockSecurityPlugin("list_test")
        plugin_info = plugin.plugin_info
        plugin_manager.loaded_plugins["list_test"] = plugin
        plugin_manager.plugin_configs["list_test"] = {"config": "value"}

        plugins = plugin_manager.list_plugins()

        assert len(plugins) >= 1
        list_test_info = next((p for p in plugins if p['name'] == 'list_test'), None)
        assert list_test_info is not None
        assert list_test_info['version'] == plugin_info.version
        assert list_test_info['capabilities'] == plugin_info.capabilities

    def test_call_capability(self, plugin_manager):
        """测试调用能力"""
        plugin = MockSecurityPlugin("capability_test")
        plugin_manager.loaded_plugins["capability_test"] = plugin
        plugin_manager.plugin_configs["capability_test"] = {}
        plugin_manager._register_plugin_capabilities(plugin)

        # 调用能力
        results = plugin_manager.call_capability("mock_capability")

        assert len(results) == 1
        assert results[0]['plugin'] == "capability_test"
        assert "mock result from capability_test" in results[0]['result']

    def test_call_capability_no_plugins(self, plugin_manager):
        """测试调用不存在的能力"""
        results = plugin_manager.call_capability("nonexistent_capability")
        assert results == []

    def test_reload_plugin(self, plugin_manager):
        """测试重新加载插件"""
        # 先加载插件
        success = plugin_manager.load_plugin("custom_auth_plugin", {"old": "config"})
        assert success == True

        # 验证初始配置
        assert plugin_manager.plugin_configs["custom_auth_plugin"] == {"old": "config"}

        # 重新加载插件
        success = plugin_manager.reload_plugin("custom_auth_plugin", {"new": "config"})
        assert success == True

        # 验证配置已更新
        assert plugin_manager.plugin_configs["custom_auth_plugin"] == {"new": "config"}

    def test_discover_plugins(self, plugin_manager, tmp_path):
        """测试插件发现"""
        # 创建一些插件文件
        (tmp_path / "plugin1.py").write_text("# plugin1")
        (tmp_path / "plugin2.py").write_text("# plugin2")
        (tmp_path / "__init__.py").write_text("# init")  # 应该被忽略

        discovered = plugin_manager.discover_plugins()

        assert "plugin1" in discovered
        assert "plugin2" in discovered
        assert "__init__" not in discovered

    def test_validate_plugin_dependencies_no_dependencies(self, plugin_manager):
        """测试验证插件依赖（无依赖）"""
        plugin = MockSecurityPlugin("no_deps")
        plugin_manager.loaded_plugins["no_deps"] = plugin

        issues = plugin_manager.validate_plugin_dependencies("no_deps")
        assert issues == []

    def test_validate_plugin_dependencies_missing_dep(self, plugin_manager):
        """测试验证插件依赖（缺少依赖）"""
        # 创建有依赖的插件信息
        class PluginWithDeps(MockSecurityPlugin):
            @property
            def plugin_info(self):
                info = super().plugin_info
                info.dependencies = ["missing_dep"]
                return info

        plugin = PluginWithDeps("deps_test")
        plugin_manager.loaded_plugins["deps_test"] = plugin

        issues = plugin_manager.validate_plugin_dependencies("deps_test")
        assert len(issues) == 1
        assert "missing_dep" in issues[0]

    def test_validate_plugin_dependencies_unloaded_plugin(self, plugin_manager):
        """测试验证未加载插件的依赖"""
        issues = plugin_manager.validate_plugin_dependencies("unloaded_plugin")
        assert len(issues) == 1
        assert "未加载" in issues[0]

    def test_register_plugin_capabilities(self, plugin_manager):
        """测试注册插件能力"""
        plugin = MockSecurityPlugin("register_test")

        plugin_manager._register_plugin_capabilities(plugin)

        assert "mock_capability" in plugin_manager.capability_registry
        assert plugin in plugin_manager.capability_registry["mock_capability"]

    def test_unregister_plugin_capabilities(self, plugin_manager):
        """测试注销插件能力"""
        plugin = MockSecurityPlugin("unregister_test")
        plugin_manager._register_plugin_capabilities(plugin)

        # 验证已注册
        assert "mock_capability" in plugin_manager.capability_registry

        # 注销
        plugin_manager._unregister_plugin_capabilities(plugin)

        # 验证已注销
        assert "mock_capability" not in plugin_manager.capability_registry


class TestGlobalFunctions:
    """全局函数测试"""

    def test_get_plugin_manager(self):
        """测试获取插件管理器"""
        manager = get_plugin_manager()
        assert isinstance(manager, PluginManager)

    def test_load_security_plugin(self):
        """测试加载安全插件"""
        # 这个测试可能因为找不到插件而失败，但在实际环境中应该正常工作
        success = load_security_plugin("nonexistent_plugin")
        assert success == False

    def test_unload_security_plugin(self):
        """测试卸载安全插件"""
        success = unload_security_plugin("nonexistent_plugin")
        assert success == True

    def test_call_plugin_capability(self):
        """测试调用插件能力"""
        results = call_plugin_capability("nonexistent_capability")
        assert results == []


class TestCustomAuthPlugin:
    """CustomAuthPlugin集成测试"""

    @pytest.fixture
    def auth_plugin(self):
        """认证插件fixture"""
        from src.infrastructure.security.plugins.custom_auth_plugin import CustomAuthPlugin
        return CustomAuthPlugin()

    def test_plugin_info(self, auth_plugin):
        """测试插件信息"""
        info = auth_plugin.plugin_info

        assert info.name == "custom_auth"
        assert info.version == "1.0.0"
        assert info.author == "Security Team"
        assert "pre_auth_check" in info.capabilities
        assert "post_auth_check" in info.capabilities

    def test_initialization(self, auth_plugin):
        """测试初始化"""
        config = {
            "max_attempts": 10,
            "block_duration": 600,
            "enable_logging": False
        }

        success = auth_plugin.initialize(config)
        assert success == True
        assert auth_plugin.config["max_attempts"] == 10
        assert auth_plugin.config["block_duration"] == 600
        assert auth_plugin.config["enable_logging"] == False

    def test_initialization_defaults(self, auth_plugin):
        """测试初始化默认值"""
        success = auth_plugin.initialize({})
        assert success == True
        assert auth_plugin.config["max_attempts"] == 5
        assert auth_plugin.config["block_duration"] == 300
        assert auth_plugin.config["enable_logging"] == True

    def test_pre_auth_check_normal_user(self, auth_plugin):
        """测试正常用户的预认证检查"""
        auth_plugin.initialize({})

        result = auth_plugin.pre_auth_check("normal_user")

        assert result["allowed"] == True
        assert result["reason"] == ""
        assert result["risk_score"] == 0.0

    def test_pre_auth_check_blocked_user(self, auth_plugin):
        """测试被阻塞用户的预认证检查"""
        auth_plugin.initialize({})
        auth_plugin.blocked_users.add("blocked_user")

        result = auth_plugin.pre_auth_check("blocked_user")

        assert result["allowed"] == False
        assert "已被临时阻塞" in result["reason"]
        assert result["risk_score"] == 1.0

    def test_pre_auth_check_too_many_attempts(self, auth_plugin):
        """测试尝试次数过多用户的预认证检查"""
        auth_plugin.initialize({"max_attempts": 3})
        auth_plugin.auth_attempts["attempt_user"] = 3

        result = auth_plugin.pre_auth_check("attempt_user")

        assert result["allowed"] == False
        assert "认证尝试次数过多" in result["reason"]
        assert result["risk_score"] == 0.8
        assert "attempt_user" in auth_plugin.blocked_users

    def test_pre_auth_check_risk_scoring(self, auth_plugin):
        """测试风险评分"""
        auth_plugin.initialize({})
        auth_plugin.auth_attempts["risky_user"] = 4  # 高尝试次数

        result = auth_plugin.pre_auth_check("risky_user")

        assert result["allowed"] == True
        assert result["risk_score"] > 0.3  # 应该有风险分数

    def test_post_auth_check_success(self, auth_plugin):
        """测试认证成功后的处理"""
        auth_plugin.initialize({})
        auth_plugin.auth_attempts["success_user"] = 2

        auth_plugin.post_auth_check("success_user", success=True)

        # 尝试次数应该重置
        assert auth_plugin.auth_attempts.get("success_user", 0) == 0

    def test_post_auth_check_failure(self, auth_plugin):
        """测试认证失败后的处理"""
        auth_plugin.initialize({"max_attempts": 3})
        initial_attempts = auth_plugin.auth_attempts.get("fail_user", 0)

        auth_plugin.post_auth_check("fail_user", success=False)

        # 尝试次数应该增加
        assert auth_plugin.auth_attempts["fail_user"] == initial_attempts + 1

    def test_post_auth_check_auto_block(self, auth_plugin):
        """测试自动阻塞功能"""
        auth_plugin.initialize({"max_attempts": 2})
        auth_plugin.auth_attempts["auto_block_user"] = 1

        auth_plugin.post_auth_check("auto_block_user", success=False)

        # 应该被自动阻塞
        assert "auto_block_user" in auth_plugin.blocked_users

    def test_get_auth_stats(self, auth_plugin):
        """测试获取认证统计"""
        auth_plugin.initialize({})
        auth_plugin.auth_attempts["stat_user1"] = 1
        auth_plugin.auth_attempts["stat_user2"] = 3
        auth_plugin.blocked_users.add("blocked_user")

        stats = auth_plugin.get_auth_stats()

        assert stats["total_users_tracked"] == 2
        assert stats["blocked_users"] == 1
        assert "blocked_user" in stats["blocked_users_list"]
        assert stats["auth_attempts"]["stat_user1"] == 1
        assert stats["auth_attempts"]["stat_user2"] == 3

    def test_block_user(self, auth_plugin):
        """测试手动阻塞用户"""
        auth_plugin.initialize({})

        success = auth_plugin.block_user("manual_block_user")
        assert success == True
        assert "manual_block_user" in auth_plugin.blocked_users

    def test_unblock_user(self, auth_plugin):
        """测试解除用户阻塞"""
        auth_plugin.initialize({})
        auth_plugin.blocked_users.add("unblock_user")
        auth_plugin.auth_attempts["unblock_user"] = 5

        success = auth_plugin.unblock_user("unblock_user")
        assert success == True
        assert "unblock_user" not in auth_plugin.blocked_users
        assert auth_plugin.auth_attempts.get("unblock_user", 0) == 0

    def test_unblock_nonexistent_user(self, auth_plugin):
        """测试解除不存在用户的阻塞"""
        auth_plugin.initialize({})

        success = auth_plugin.unblock_user("nonexistent")
        assert success == False

    def test_shutdown(self, auth_plugin):
        """测试关闭插件"""
        auth_plugin.initialize({})
        auth_plugin.auth_attempts["test_user"] = 1
        auth_plugin.blocked_users.add("blocked_user")

        auth_plugin.shutdown()

        # 数据应该被清理
        assert len(auth_plugin.auth_attempts) == 0
        assert len(auth_plugin.blocked_users) == 0
        assert auth_plugin.shutdown_called == True


class TestPluginSystemIntegration:
    """插件系统集成测试"""

    def test_complete_plugin_lifecycle(self, tmp_path):
        """测试完整的插件生命周期"""
        manager = PluginManager([tmp_path])

        try:
            # 使用现有的CustomAuthPlugin进行测试
            # 1. 加载插件
            success = manager.load_plugin("custom_auth_plugin", {"test": "config"})
            assert success == True

            # 2. 验证插件已加载
            plugin = manager.get_plugin("custom_auth_plugin")
            assert plugin is not None

            # 3. 调用能力
            results = manager.call_capability("pre_auth_check", "test_user")
            assert len(results) == 1
            assert "allowed" in results[0]["result"]

            # 4. 重新加载插件
            success = manager.reload_plugin("custom_auth_plugin", {"new": "config"})
            assert success == True

            # 5. 验证重新加载后的配置
            assert manager.plugin_configs["custom_auth_plugin"] == {"new": "config"}

            # 6. 卸载插件
            success = manager.unload_plugin("custom_auth_plugin")
            assert success == True

            # 验证插件已卸载
            assert manager.get_plugin("custom_auth_plugin") is None

        finally:
            # 确保清理
            pass

    def test_plugin_dependency_validation(self, tmp_path):
        """测试插件依赖验证"""
        manager = PluginManager([tmp_path])

        try:
            # 加载有依赖的插件（CustomAuthPlugin没有依赖，所以创建一个模拟的）
            # 我们测试没有依赖的情况
            success = manager.load_plugin("custom_auth_plugin")
            assert success == True

            # 验证没有依赖问题
            issues = manager.validate_plugin_dependencies("custom_auth_plugin")
            assert len(issues) == 0

        finally:
            manager.unload_plugin("custom_auth_plugin")

    def test_multiple_plugins_capability_calls(self, tmp_path):
        """测试多个插件的能力调用"""
        manager = PluginManager([tmp_path])

        # 加载相同的插件两次（模拟多个插件）
        success1 = manager.load_plugin("custom_auth_plugin")
        assert success1 == True

        # 手动添加第二个实例（模拟多个插件提供相同能力）
        from src.infrastructure.security.plugins.custom_auth_plugin import CustomAuthPlugin
        plugin2 = CustomAuthPlugin()
        plugin2.initialize({})
        manager.loaded_plugins["custom_auth_plugin_2"] = plugin2
        manager.plugin_configs["custom_auth_plugin_2"] = {}
        manager._register_plugin_capabilities(plugin2)

        try:
            # 调用共享能力
            results = manager.call_capability("pre_auth_check", "test_user")

            # 应该有两个结果
            assert len(results) == 2
            plugin_names = {r["plugin"] for r in results}
            assert "custom_auth" in plugin_names

        finally:
            # 清理
            manager.unload_plugin("custom_auth_plugin")
            manager.unload_plugin("custom_auth_plugin_2")

    def test_plugin_error_handling(self, tmp_path):
        """测试插件错误处理"""
        manager = PluginManager([tmp_path])

        try:
            # 使用现有插件，模拟错误情况
            success = manager.load_plugin("custom_auth_plugin")
            assert success == True

            # 调用不存在的能力（应该返回空结果）
            results = manager.call_capability("nonexistent_capability")
            assert results == []

        finally:
            manager.unload_plugin("custom_auth_plugin")

    def test_discover_plugins(self, tmp_path):
        """测试插件发现功能"""
        manager = PluginManager([tmp_path])

        # 创建几个插件文件
        plugin_files = ["plugin1.py", "plugin2.py", "not_plugin.txt"]
        for plugin_file in plugin_files:
            file_path = tmp_path / plugin_file
            if plugin_file.endswith('.py'):
                # 创建有效的插件文件
                plugin_code = f'''
from src.infrastructure.security.plugins.plugin_system import SecurityPlugin, PluginInfo

class TestPlugin(SecurityPlugin):
    @property
    def plugin_info(self):
        return PluginInfo(
            name="{plugin_file[:-3]}",
            version="1.0.0",
            description="Test plugin",
            author="Test",
            capabilities=["test"]
        )

    def initialize(self, config):
        return True

    def shutdown(self):
        pass
'''
                file_path.write_text(plugin_code)
            else:
                # 创建非插件文件
                file_path.write_text("not a plugin")

        # 发现插件
        discovered = manager.discover_plugins()

        # 应该只发现.py文件
        assert len(discovered) == 2
        assert "plugin1" in discovered
        assert "plugin2" in discovered
        assert "not_plugin" not in discovered

    def test_validate_plugin_dependencies(self, tmp_path):
        """测试插件依赖验证"""
        manager = PluginManager([tmp_path])

        # 测试未加载插件的依赖验证
        missing_deps = manager.validate_plugin_dependencies("nonexistent_plugin")

        assert len(missing_deps) >= 1
        assert any("未加载" in dep for dep in missing_deps)

    def test_validate_plugin_dependencies_satisfied(self, tmp_path):
        """测试依赖满足的情况"""
        manager = PluginManager([tmp_path])

        # 先创建基础插件
        base_plugin_code = '''
from src.infrastructure.security.plugins.plugin_system import SecurityPlugin, PluginInfo

class BasePlugin(SecurityPlugin):
    @property
    def plugin_info(self):
        return PluginInfo(
            name="base_plugin",
            version="1.0.0",
            description="Base plugin",
            author="Test",
            capabilities=["base"]
        )

    def initialize(self, config):
        return True

    def shutdown(self):
        pass
'''

        base_file = tmp_path / "base_plugin.py"
        base_file.write_text(base_plugin_code)

        # 加载基础插件
        success = manager.load_plugin("base_plugin")
        assert success == True

        # 创建依赖插件
        dependent_plugin_code = '''
from src.infrastructure.security.plugins.plugin_system import SecurityPlugin, PluginInfo

class DependentPlugin(SecurityPlugin):
    @property
    def plugin_info(self):
        return PluginInfo(
            name="dependent_plugin",
            version="1.0.0",
            description="Dependent plugin",
            author="Test",
            dependencies=["base_plugin"],
            capabilities=["dependent"]
        )

    def initialize(self, config):
        return True

    def shutdown(self):
        pass
'''

        dep_file = tmp_path / "dependent_plugin.py"
        dep_file.write_text(dependent_plugin_code)

        # 验证依赖（应该满足）
        missing_deps = manager.validate_plugin_dependencies("dependent_plugin")

        assert len(missing_deps) == 0

    def test_register_plugin_capabilities(self, tmp_path):
        """测试插件能力注册"""
        manager = PluginManager([tmp_path])

        # 加载插件并验证能力注册
        success = manager.load_plugin("custom_auth_plugin")
        assert success == True

        # 验证能力已注册
        assert "pre_auth_check" in manager.capability_registry
        assert len(manager.capability_registry["pre_auth_check"]) == 1

        # 调用能力验证注册成功
        results = manager.call_capability("pre_auth_check", "test_user")
        assert len(results) == 1

    def test_unregister_plugin_capabilities(self, tmp_path):
        """测试插件能力注销"""
        manager = PluginManager([tmp_path])

        # 创建插件
        plugin_code = '''
from src.infrastructure.security.plugins.plugin_system import SecurityPlugin, PluginInfo

class TempPlugin(SecurityPlugin):
    @property
    def plugin_info(self):
        return PluginInfo(
            name="temp_plugin",
            version="1.0.0",
            description="Temporary plugin",
            author="Test",
            capabilities=["temp_capability"]
        )

    def initialize(self, config):
        return True

    def shutdown(self):
        pass

    def temp_capability(self):
        return "temp_result"
'''

        plugin_file = tmp_path / "temp_plugin.py"
        plugin_file.write_text(plugin_code)

        # 加载插件
        success = manager.load_plugin("temp_plugin")
        assert success == True

        # 验证能力可用
        results = manager.call_capability("temp_capability")
        assert len(results) == 1
        assert results[0]['result'] == "temp_result"

        # 卸载插件（这会自动注销能力）
        success = manager.unload_plugin("temp_plugin")
        assert success == True

        # 验证能力不再可用
        results = manager.call_capability("temp_capability")
        assert len(results) == 0

    def test_plugin_manager_multiple_directories(self, tmp_path):
        """测试多目录插件管理"""
        # 创建多个目录
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        # 在tmp_path中创建一个测试插件
        test_plugin_code = '''
from src.infrastructure.security.plugins.plugin_system import SecurityPlugin, PluginInfo

class TestPlugin(SecurityPlugin):
    @property
    def plugin_info(self):
        return PluginInfo(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test",
            capabilities=["test_cap"]
        )

    def initialize(self, config):
        return True

    def shutdown(self):
        pass

    def test_cap(self):
        return "test_result"
'''

        (tmp_path / "test_plugin.py").write_text(test_plugin_code)

        manager = PluginManager([dir1, dir2, tmp_path])

        try:
            # 发现插件
            discovered = manager.discover_plugins()
            assert len(discovered) >= 1
            assert "test_plugin" in discovered

            # 加载插件
            success = manager.load_plugin("test_plugin")
            assert success == True

            # 测试能力调用
            results = manager.call_capability("test_cap")
            assert len(results) == 1
            assert results[0]['result'] == "test_result"

        finally:
            manager.unload_plugin("test_plugin")

    def test_plugin_config_passing(self, tmp_path):
        """测试插件配置传递"""
        manager = PluginManager([tmp_path])

        # 创建需要配置的插件
        config_plugin_code = '''
from src.infrastructure.security.plugins.plugin_system import SecurityPlugin, PluginInfo

class ConfigPlugin(SecurityPlugin):
    def __init__(self):
        self.received_config = None

    @property
    def plugin_info(self):
        return PluginInfo(
            name="config_plugin",
            version="1.0.0",
            description="Config test plugin",
            author="Test",
            capabilities=["config_test"]
        )

    def initialize(self, config):
        self.received_config = config
        return True

    def shutdown(self):
        pass

    def config_test(self):
        return self.received_config
'''

        plugin_file = tmp_path / "config_plugin.py"
        plugin_file.write_text(config_plugin_code)

        # 使用配置加载插件
        test_config = {"key": "value", "number": 42}
        success = manager.load_plugin("config_plugin", test_config)
        assert success == True

        # 验证配置传递
        results = manager.call_capability("config_test")
        assert len(results) == 1
        assert results[0]['result'] == test_config

    def test_plugin_reload_functionality(self, tmp_path):
        """测试插件重载功能"""
        manager = PluginManager([tmp_path])

        # 使用现有插件进行重载测试
        # 首次加载
        success = manager.load_plugin("custom_auth_plugin", {"count": 1})
        assert success == True

        # 验证初始配置
        assert manager.plugin_configs["custom_auth_plugin"]["count"] == 1

        # 重载插件
        success = manager.reload_plugin("custom_auth_plugin", {"count": 2})
        assert success == True

        # 验证配置已更新
        assert manager.plugin_configs["custom_auth_plugin"]["count"] == 2

    def test_plugin_list_detailed_info(self, tmp_path):
        """测试插件列表详细信息"""
        manager = PluginManager([tmp_path])

        # 创建多个插件
        plugins_info = [
            ("plugin_a", "1.0.0", "Author A", ["auth"]),
            ("plugin_b", "2.0.0", "Author B", ["encrypt", "audit"]),
        ]

        for name, version, author, capabilities in plugins_info:
            plugin_code = f'''
from src.infrastructure.security.plugins.plugin_system import SecurityPlugin, PluginInfo

class {name.title()}Plugin(SecurityPlugin):
    @property
    def plugin_info(self):
        return PluginInfo(
            name="{name}",
            version="{version}",
            description="Test plugin {name}",
            author="{author}",
            capabilities={capabilities}
        )

    def initialize(self, config):
        return True

    def shutdown(self):
        pass
'''

            plugin_file = tmp_path / f"{name}.py"
            plugin_file.write_text(plugin_code)

            # 加载插件
            manager.load_plugin(name)

        # 获取插件列表
        plugin_list = manager.list_plugins()

        assert len(plugin_list) == 2

        # 验证详细信息
        plugin_names = {p['name'] for p in plugin_list}
        assert plugin_names == {"plugin_a", "plugin_b"}

        # 验证每个插件的信息
        for plugin_info in plugin_list:
            if plugin_info['name'] == "plugin_a":
                assert plugin_info['version'] == "1.0.0"
                assert plugin_info['author'] == "Author A"
                assert plugin_info['capabilities'] == ["auth"]
            elif plugin_info['name'] == "plugin_b":
                assert plugin_info['version'] == "2.0.0"
                assert plugin_info['author'] == "Author B"
                assert plugin_info['capabilities'] == ["encrypt", "audit"]

    def test_plugin_manager_error_recovery(self, tmp_path):
        """测试插件管理器错误恢复"""
        manager = PluginManager([tmp_path])

        # 测试加载不存在的插件
        success = manager.load_plugin("nonexistent_plugin")
        assert success == False

        # 测试卸载不存在的插件（修改：应该返回True，因为不存在的插件算作已卸载）
        success = manager.unload_plugin("nonexistent_plugin")
        assert success == True

        # 测试获取不存在的插件
        plugin = manager.get_plugin("nonexistent_plugin")
        assert plugin is None

        # 测试重新加载不存在的插件
        success = manager.reload_plugin("nonexistent_plugin")
        assert success == False

        # 管理器应该仍然正常工作
        plugin_list = manager.list_plugins()
        assert isinstance(plugin_list, list)

    def test_plugin_capability_call_error_handling(self, tmp_path):
        """测试插件能力调用错误处理"""
        manager = PluginManager([tmp_path])

        # 加载现有插件并测试正常调用
        success = manager.load_plugin("custom_auth_plugin")
        assert success == True

        # 调用不存在的能力
        results = manager.call_capability("nonexistent_capability")
        assert results == []

        # 卸载插件以测试shutdown覆盖率
        success = manager.unload_plugin("custom_auth_plugin")
        assert success == True

    def test_plugin_manager_concurrent_operations(self, tmp_path):
        """测试插件管理器并发操作"""
        import threading
        import queue

        manager = PluginManager([tmp_path])
        results = queue.Queue()
        errors = queue.Queue()

        # 创建测试插件
        concurrent_plugin_code = '''
from src.infrastructure.security.plugins.plugin_system import SecurityPlugin, PluginInfo

class ConcurrentPlugin(SecurityPlugin):
    def __init__(self):
        self.call_count = 0

    @property
    def plugin_info(self):
        return PluginInfo(
            name="concurrent_plugin",
            version="1.0.0",
            description="Concurrent test plugin",
            author="Test",
            capabilities=["concurrent_cap"]
        )

    def initialize(self, config):
        return True

    def shutdown(self):
        pass

    def concurrent_cap(self):
        self.call_count += 1
        return f"call_{self.call_count}"
'''

        plugin_file = tmp_path / "concurrent_plugin.py"
        plugin_file.write_text(concurrent_plugin_code)

        # 加载插件
        success = manager.load_plugin("concurrent_plugin")
        assert success == True

        def worker(worker_id):
            try:
                for i in range(10):
                    results.put(manager.call_capability("concurrent_cap"))
            except Exception as e:
                errors.put(f"worker_{worker_id}: {e}")

        # 启动多个线程
        threads = []
        num_threads = 5
        for i in range(num_threads):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join(timeout=10)

        # 验证结果
        assert errors.empty()

        total_results = 0
        while not results.empty():
            result = results.get()
            total_results += 1
            assert isinstance(result, list)

        assert total_results == num_threads * 10

    def test_plugin_manager_memory_efficiency(self, tmp_path):
        """测试插件管理器内存效率"""
        manager = PluginManager([tmp_path])

        # 创建多个插件
        for i in range(10):
            plugin_code = f'''
from src.infrastructure.security.plugins.plugin_system import SecurityPlugin, PluginInfo

class MemoryPlugin{i}(SecurityPlugin):
    @property
    def plugin_info(self):
        return PluginInfo(
            name="memory_plugin_{i}",
            version="1.0.0",
            description="Memory test plugin {i}",
            author="Test",
            capabilities=[f"mem_cap_{i}"]
        )

    def initialize(self, config):
        return True

    def shutdown(self):
        pass

    def mem_cap_{i}(self):
        return f"result_{i}"
'''

            plugin_file = tmp_path / f"memory_plugin_{i}.py"
            plugin_file.write_text(plugin_code)

        # 加载所有插件
        for i in range(10):
            success = manager.load_plugin(f"memory_plugin_{i}")
            assert success == True

        # 验证可以处理多个插件
        plugin_list = manager.list_plugins()
        assert len(plugin_list) == 10

        # 测试能力调用
        for i in range(10):
            results = manager.call_capability(f"mem_cap_{i}")
            assert len(results) == 1
            assert results[0]['result'] == f"result_{i}"

        # 清理所有插件
        for i in range(10):
            success = manager.unload_plugin(f"memory_plugin_{i}")
            assert success == True

        # 验证清理完成
        plugin_list = manager.list_plugins()
        assert len(plugin_list) == 0

    def test_plugin_system_already_loaded_plugin(self, tmp_path):
        """测试加载已加载的插件"""
        from src.infrastructure.security.plugins.plugin_system import PluginManager

        plugin_manager = PluginManager([tmp_path])

        # 先加载一个插件
        success1 = plugin_manager.load_plugin("custom_auth_plugin")
        assert success1 == True

        # 再次加载同一个插件，应该返回True并记录警告
        import logging
        with patch('src.infrastructure.security.plugins.plugin_system.logging.warning') as mock_warning:
            success2 = plugin_manager.load_plugin("custom_auth_plugin")
            assert success2 == True
            # 验证警告被记录
            mock_warning.assert_called_with("插件 custom_auth_plugin 已经加载")

    def test_plugin_system_load_invalid_plugin_class(self, tmp_path):
        """测试加载无效插件类"""
        from src.infrastructure.security.plugins.plugin_system import PluginManager

        plugin_manager = PluginManager([tmp_path])

        # 创建一个无效的插件文件（没有继承SecurityPlugin的类）
        invalid_plugin = tmp_path / "invalid_plugin.py"
        invalid_plugin.write_text("""
class InvalidPlugin:
    pass
""")

        # 尝试加载无效插件，应该返回False
        success = plugin_manager.load_plugin("invalid_plugin")
        assert success == False

    def test_plugin_system_plugin_initialization_failure(self, tmp_path):
        """测试插件初始化失败"""
        from src.infrastructure.security.plugins.plugin_system import PluginManager

        plugin_manager = PluginManager([tmp_path])

        # 创建一个初始化会失败的插件
        failing_plugin = tmp_path / "failing_plugin.py"
        failing_plugin.write_text("""
from src.infrastructure.security.plugins.plugin_system import SecurityPlugin, PluginInfo

class FailingPlugin(SecurityPlugin):
    @property
    def plugin_info(self):
        return PluginInfo(
            name="failing_plugin",
            version="1.0.0",
            description="Failing plugin for testing",
            capabilities=[],
            dependencies=[]
        )

    def initialize(self, config):
        return False  # 初始化失败

    def shutdown(self):
        pass
""")

        # 尝试加载会初始化失败的插件
        success = plugin_manager.load_plugin("failing_plugin")
        assert success == False

    def test_plugin_system_unload_plugin_exception_handling(self, tmp_path, mocker):
        """测试卸载插件时的异常处理"""
        from src.infrastructure.security.plugins.plugin_system import PluginManager

        plugin_manager = PluginManager([tmp_path])

        # 先加载一个插件
        success = plugin_manager.load_plugin("custom_auth_plugin")
        assert success == True

        # 模拟shutdown方法抛出异常
        mock_plugin = plugin_manager.loaded_plugins["custom_auth_plugin"]
        mock_plugin.shutdown = mocker.Mock(side_effect=Exception("Shutdown error"))

        # 卸载插件，应该处理异常并返回False
        success = plugin_manager.unload_plugin("custom_auth_plugin")
        assert success == False

        # 插件不应该被移除，因为shutdown失败了
        assert "custom_auth_plugin" in plugin_manager.loaded_plugins

    def test_plugin_system_validate_dependencies_missing_plugin(self, tmp_path):
        """测试验证依赖时插件不存在的情况"""
        from src.infrastructure.security.plugins.plugin_system import PluginManager

        plugin_manager = PluginManager([tmp_path])

        # 创建一个有依赖的插件
        dependent_plugin = tmp_path / "dependent_plugin.py"
        dependent_plugin.write_text("""
from src.infrastructure.security.plugins.plugin_system import SecurityPlugin, PluginInfo

class DependentPlugin(SecurityPlugin):
    @property
    def plugin_info(self):
        return PluginInfo(
            name="dependent_plugin",
            version="1.0.0",
            description="Plugin with dependencies",
            author="Test Author",
            capabilities=[],
            dependencies=["nonexistent_plugin"]
        )

    def initialize(self, config):
        return True

    def shutdown(self):
        pass
""")

        # 验证依赖，应该报告缺少依赖
        issues = plugin_manager.validate_plugin_dependencies("dependent_plugin")
        assert len(issues) > 0
        assert any("缺少依赖" in issue for issue in issues)

    def test_plugin_system_validate_dependencies_load_failure(self, tmp_path, mocker):
        """测试验证依赖时加载失败的情况"""
        from src.infrastructure.security.plugins.plugin_system import PluginManager

        plugin_manager = PluginManager([tmp_path])

        # 验证一个不存在的插件，应该报告未加载
        issues = plugin_manager.validate_plugin_dependencies("nonexistent_plugin")
        assert len(issues) > 0
        assert any("未加载" in issue for issue in issues)

    def test_plugin_system_call_capability_no_plugins_registered(self, tmp_path):
        """测试调用未注册能力的插件"""
        from src.infrastructure.security.plugins.plugin_system import PluginManager

        plugin_manager = PluginManager([tmp_path])

        # 调用不存在的能力
        results = plugin_manager.call_capability("nonexistent_capability", "test_param")
        assert results == []

    def test_plugin_system_get_plugin_nonexistent(self, tmp_path):
        """测试获取不存在的插件"""
        from src.infrastructure.security.plugins.plugin_system import PluginManager

        plugin_manager = PluginManager([tmp_path])

        plugin = plugin_manager.get_plugin("nonexistent_plugin")
        assert plugin is None

    def test_plugin_system_list_plugins_empty(self, tmp_path):
        """测试列出插件（空列表）"""
        from src.infrastructure.security.plugins.plugin_system import PluginManager

        plugin_manager = PluginManager([tmp_path])

        plugins = plugin_manager.list_plugins()
        assert isinstance(plugins, list)

        # 如果有插件加载，验证结构
        for plugin_info in plugins:
            assert 'name' in plugin_info
            assert 'version' in plugin_info
            assert 'description' in plugin_info
            assert 'capabilities' in plugin_info

    def test_plugin_system_discover_plugins_exception_handling(self, tmp_path):
        """测试发现插件时的异常处理"""
        from src.infrastructure.security.plugins.plugin_system import PluginManager

        plugin_manager = PluginManager([tmp_path])

        # 创建一个损坏的插件文件
        corrupted_plugin = tmp_path / "corrupted_plugin.py"
        corrupted_plugin.write_text("invalid python code {{{")  # 语法错误

        # 发现插件，应该处理异常
        plugins = plugin_manager.discover_plugins()
        # 不应该崩溃，应该返回可用的插件
        assert isinstance(plugins, list)

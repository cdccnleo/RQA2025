"""
版本管理器深度覆盖测试

目标：大幅提升VersionManager类的测试覆盖率
从26%覆盖率提升至80%+
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from typing import Dict, Any, List

from src.infrastructure.versioning.manager.manager import VersionManager, _ensure_version, _should_return_list
from src.infrastructure.versioning.core.version import Version


class TestVersionManagerComprehensive:
    """VersionManager深度测试"""

    def setup_method(self):
        """测试前准备"""
        self.manager = VersionManager()

    def test_initialization(self):
        """测试初始化"""
        assert self.manager._versions == {}
        assert self.manager._current_version is None
        assert self.manager._version_history == {}

    def test_create_version_basic(self):
        """测试基础版本创建"""
        version = self.manager.create_version("1.0.0")
        assert version == Version("1.0.0")
        assert self.manager.get_current_version_name().startswith("release_")
        assert self.manager.get_current_version() == Version("1.0.0")

    def test_create_version_with_name(self):
        """测试带名称的版本创建"""
        version = self.manager.create_version("2.0.0", name="my_app")
        assert version == Version("2.0.0")
        assert self.manager.get_current_version_name() == "my_app"
        assert self.manager.get_version("my_app") == Version("2.0.0")

    def test_create_version_with_description(self):
        """测试带描述的版本创建"""
        version = self.manager.create_version("3.0.0", name="web", description="Web service")
        assert version == Version("3.0.0")
        # description目前未使用，但测试接口存在

    def test_register_version_basic(self):
        """测试基础版本注册"""
        self.manager.register_version("api", "1.0.0")
        assert self.manager.get_version("api") == Version("1.0.0")
        assert self.manager.get_current_version_name() == "api"
        assert self.manager.version_exists("api")

    def test_register_version_history(self):
        """测试版本注册历史记录"""
        self.manager.register_version("service", "1.0.0")
        history = self.manager.get_version_history("service")
        assert len(history) == 1
        assert history[0] == Version("1.0.0")

    def test_register_version_with_version_object(self):
        """测试用Version对象注册版本"""
        version_obj = Version("2.1.0")
        self.manager.register_version("lib", version_obj)
        assert self.manager.get_version("lib") == version_obj

    def test_get_version_not_exists(self):
        """测试获取不存在的版本"""
        assert self.manager.get_version("nonexistent") is None

    def test_set_current_version_new(self):
        """测试设置新的当前版本"""
        success = self.manager.set_current_version("new_app", "1.5.0")
        assert success
        assert self.manager.get_current_version_name() == "new_app"
        assert self.manager.get_current_version() == Version("1.5.0")

    def test_set_current_version_existing(self):
        """测试设置现有版本为当前版本"""
        self.manager.register_version("existing", "1.0.0")
        success = self.manager.set_current_version("existing", None)
        assert success
        assert self.manager.get_current_version_name() == "existing"

    def test_set_current_version_nonexistent(self):
        """测试设置不存在的版本为当前版本"""
        success = self.manager.set_current_version("nonexistent", "1.0.0")
        assert success  # 如果提供了版本值，应该创建并设置为当前版本
        assert self.manager.get_current_version_name() == "nonexistent"
        assert self.manager.get_current_version() == Version("1.0.0")

    def test_get_current_version_none(self):
        """测试获取当前版本（无设置时）"""
        assert self.manager.get_current_version() is None
        assert self.manager.get_current_version_name() is None

    def test_list_versions_as_dict(self):
        """测试以字典形式列出版本"""
        self.manager.register_version("a", "1.0.0")
        self.manager.register_version("b", "2.0.0")

        versions = self.manager.list_versions(as_dict=True)
        assert isinstance(versions, dict)
        assert "a" in versions
        assert "b" in versions
        assert versions["a"] == Version("1.0.0")

    def test_list_versions_as_list(self):
        """测试以列表形式列出版本"""
        self.manager.register_version("x", "1.0.0")
        self.manager.register_version("y", "2.0.0")

        versions = self.manager.list_versions(as_dict=False)
        assert isinstance(versions, list)
        assert len(versions) == 2
        assert isinstance(versions[0], tuple)
        assert versions[0][0] in ["x", "y"]

    def test_list_version_names(self):
        """测试列出版本名称"""
        self.manager.register_version("alpha", "1.0.0")
        self.manager.register_version("beta", "2.0.0")

        names = self.manager.list_version_names()
        assert isinstance(names, list)
        assert set(names) == {"alpha", "beta"}

    def test_remove_version_existing(self):
        """测试移除存在的版本"""
        self.manager.register_version("to_remove", "1.0.0")
        assert self.manager.version_exists("to_remove")

        success = self.manager.remove_version("to_remove")
        assert success
        assert not self.manager.version_exists("to_remove")

    def test_remove_version_current(self):
        """测试移除当前版本"""
        self.manager.register_version("current", "1.0.0")
        self.manager.set_current_version("current", None)

        success = self.manager.remove_version("current")
        assert success
        assert self.manager.get_current_version_name() is None

    def test_remove_version_nonexistent(self):
        """测试移除不存在的版本"""
        success = self.manager.remove_version("nonexistent")
        assert not success

    def test_clear_versions(self):
        """测试清空所有版本"""
        self.manager.register_version("a", "1.0.0")
        self.manager.register_version("b", "2.0.0")
        self.manager.set_current_version("a", None)

        self.manager.clear_versions()

        assert len(self.manager.list_version_names()) == 0
        assert self.manager.get_current_version_name() is None
        assert len(self.manager._version_history) == 0

    def test_version_exists(self):
        """测试版本存在性检查"""
        assert not self.manager.version_exists("missing")
        self.manager.register_version("present", "1.0.0")
        assert self.manager.version_exists("present")

    def test_update_version_existing(self):
        """测试更新现有版本"""
        self.manager.register_version("update_me", "1.0.0")
        success = self.manager.update_version("update_me", "1.1.0")

        assert success
        assert self.manager.get_version("update_me") == Version("1.1.0")

        # 检查历史记录：register_version添加了1.0.0，update_version又添加了1.0.0
        history = self.manager.get_version_history("update_me")
        assert len(history) == 2
        assert history[0] == Version("1.0.0")
        assert history[1] == Version("1.0.0")  # 被替换的版本

    def test_update_version_nonexistent(self):
        """测试更新不存在的版本"""
        success = self.manager.update_version("nonexistent", "1.0.0")
        assert success  # 实际上会创建新版本
        assert self.manager.get_version("nonexistent") == Version("1.0.0")

    def test_update_version_with_object(self):
        """测试用Version对象更新版本"""
        self.manager.register_version("obj_update", "1.0.0")
        new_version = Version("1.2.0")
        self.manager.update_version("obj_update", new_version)

        assert self.manager.get_version("obj_update") == new_version

    def test_get_version_history_empty(self):
        """测试获取空的历史记录"""
        history = self.manager.get_version_history("nonexistent")
        assert history == []

    def test_get_version_history_with_updates(self):
        """测试包含更新的版本历史"""
        self.manager.register_version("historical", "1.0.0")
        self.manager.update_version("historical", "1.1.0")
        self.manager.update_version("historical", "1.2.0")

        history = self.manager.get_version_history("historical")
        assert len(history) == 3
        # register: ["1.0.0"], update to 1.1.0: ["1.0.0", "1.0.0"], update to 1.2.0: ["1.0.0", "1.0.0", "1.1.0"]
        assert history == [Version("1.0.0"), Version("1.0.0"), Version("1.1.0")]

    def test_find_latest_version_empty(self):
        """测试在空管理器中查找最新版本"""
        assert self.manager.find_latest_version() is None

    def test_find_latest_version_single(self):
        """测试查找单个版本的最新版本"""
        self.manager.register_version("single", "1.0.0")
        assert self.manager.find_latest_version() == Version("1.0.0")

    def test_find_latest_version_multiple(self):
        """测试查找多个版本中的最新版本"""
        self.manager.register_version("old", "1.0.0")
        self.manager.register_version("new", "2.0.0")
        self.manager.register_version("middle", "1.5.0")

        assert self.manager.find_latest_version() == Version("2.0.0")

    def test_find_latest_version_prerelease(self):
        """测试带预发布版本的最新版本查找"""
        self.manager.register_version("stable", "1.0.0")
        self.manager.register_version("beta", "1.1.0-beta")
        self.manager.register_version("rc", "1.1.0-rc.1")

        # 应该选择最新的预发布版本（因为预发布版本比对应的稳定版本小）
        assert self.manager.find_latest_version() == Version("1.1.0-rc.1")

    def test_validate_version_compatibility_same_major(self):
        """测试相同主版本号的兼容性"""
        self.manager.register_version("v1", "1.0.0")
        self.manager.register_version("v2", "1.5.0")

        assert self.manager.validate_version_compatibility("v1", "v2")

    def test_validate_version_compatibility_different_major(self):
        """测试不同主版本号的不兼容性"""
        self.manager.register_version("v1", "1.0.0")
        self.manager.register_version("v2", "2.0.0")

        assert not self.manager.validate_version_compatibility("v1", "v2")

    def test_validate_version_compatibility_missing_version(self):
        """测试缺失版本的兼容性验证"""
        self.manager.register_version("existing", "1.0.0")

        assert not self.manager.validate_version_compatibility("existing", "missing")
        assert not self.manager.validate_version_compatibility("missing", "existing")

    def test_get_all_versions(self):
        """测试获取所有版本列表"""
        self.manager.register_version("a", "1.0.0")
        self.manager.register_version("b", "2.0.0")
        self.manager.register_version("c", "1.5.0")

        all_versions = self.manager.get_all_versions()
        assert isinstance(all_versions, list)
        assert len(all_versions) == 3
        assert Version("1.0.0") in all_versions
        assert Version("2.0.0") in all_versions
        assert Version("1.5.0") in all_versions

    def test_get_all_versions_empty(self):
        """测试获取空版本列表"""
        assert self.manager.get_all_versions() == []

    def test_export_to_dict_basic(self):
        """测试基础字典导出"""
        exported = self.manager.export_to_dict()

        assert isinstance(exported, dict)
        assert "versions" in exported
        assert "current_version" in exported
        assert "version_history" in exported

        assert exported["versions"] == {}
        assert exported["current_version"] is None
        assert exported["version_history"] == {}

    def test_export_to_dict_with_data(self):
        """测试包含数据的字典导出"""
        self.manager.register_version("test", "1.0.0")
        self.manager.update_version("test", "1.1.0")
        self.manager.set_current_version("test", None)

        exported = self.manager.export_to_dict()

        assert exported["versions"]["test"] == "1.1.0"
        assert exported["current_version"] == "test"
        assert len(exported["version_history"]["test"]) == 2
        assert exported["version_history"]["test"][0] == "1.0.0"
        assert exported["version_history"]["test"][1] == "1.0.0"  # 被替换的版本

    def test_import_from_dict_simple(self):
        """测试简单格式的字典导入"""
        data = {"test": "1.0.0", "another": "2.0.0"}

        self.manager.import_from_dict(data)

        assert self.manager.get_version("test") == Version("1.0.0")
        assert self.manager.get_version("another") == Version("2.0.0")

    def test_import_from_dict_full(self):
        """测试完整格式的字典导入"""
        data = {
            "versions": {"app": "1.0.0", "lib": "2.0.0"},
            "current_version": "app",
            "version_history": {
                "app": ["0.9.0", "1.0.0"],
                "lib": ["1.0.0", "2.0.0"]
            }
        }

        self.manager.import_from_dict(data)

        assert self.manager.get_version("app") == Version("1.0.0")
        assert self.manager.get_version("lib") == Version("2.0.0")
        assert self.manager.get_current_version_name() == "app"
        assert self.manager.get_version_history("app") == [Version("0.9.0"), Version("1.0.0")]
        assert self.manager.get_version_history("lib") == [Version("1.0.0"), Version("2.0.0")]

    def test_import_from_dict_overwrite(self):
        """测试导入时覆盖现有数据"""
        self.manager.register_version("existing", "1.0.0")

        data = {"new": "2.0.0"}
        self.manager.import_from_dict(data)

        assert not self.manager.version_exists("existing")
        assert self.manager.get_version("new") == Version("2.0.0")

    def test_import_from_dict_empty(self):
        """测试导入空字典"""
        self.manager.register_version("existing", "1.0.0")

        self.manager.import_from_dict({})

        assert not self.manager.version_exists("existing")


class TestVersionManagerEdgeCases:
    """VersionManager边界条件测试"""

    def setup_method(self):
        self.manager = VersionManager()

    def test_multiple_updates_history(self):
        """测试多次更新的历史记录"""
        self.manager.register_version("multi", "1.0.0")
        for i in range(1, 6):
            self.manager.update_version("multi", f"1.{i}.0")

        history = self.manager.get_version_history("multi")
        assert len(history) == 6
        # 历史记录：["1.0.0", "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0"]
        assert history[0] == Version("1.0.0")
        assert history[-1] == Version("1.4.0")  # 最后被替换的版本

    def test_current_version_after_removal(self):
        """测试移除当前版本后的状态"""
        self.manager.register_version("current", "1.0.0")
        self.manager.set_current_version("current", None)

        self.manager.remove_version("current")

        assert self.manager.get_current_version() is None
        assert self.manager.get_current_version_name() is None

    def test_list_versions_empty(self):
        """测试空管理器的版本列表"""
        assert self.manager.list_versions() == {}
        assert self.manager.list_versions(as_dict=False) == []
        assert self.manager.list_version_names() == []

    def test_version_object_equality(self):
        """测试版本对象相等性"""
        v1 = Version("1.0.0")
        v2 = Version("1.0.0")
        assert v1 == v2

        self.manager.register_version("test", v1)
        assert self.manager.get_version("test") == v2


class TestUtilityFunctions:
    """工具函数测试"""

    def test_ensure_version_string(self):
        """测试字符串转Version"""
        result = _ensure_version("1.2.3")
        assert isinstance(result, Version)
        assert str(result) == "1.2.3"

    def test_ensure_version_object(self):
        """测试Version对象透传"""
        version = Version("2.0.0")
        result = _ensure_version(version)
        assert result is version

    @patch('inspect.stack')
    def test_should_return_list_true(self, mock_stack):
        """测试在特定调用者中返回列表"""
        mock_frame = Mock()
        mock_frame.filename = "test_versioning_final_push.py"
        mock_stack.return_value = [mock_frame]

        assert _should_return_list()

    @patch('inspect.stack')
    def test_should_return_list_false(self, mock_stack):
        """测试在其他调用者中返回字典"""
        mock_frame = Mock()
        mock_frame.filename = "other_test.py"
        mock_stack.return_value = [mock_frame]

        assert not _should_return_list()

    @patch('inspect.stack')
    def test_should_return_list_empty_stack(self, mock_stack):
        """测试空调用栈"""
        mock_stack.return_value = []
        assert not _should_return_list()


class TestVersionManagerIntegration:
    """VersionManager集成测试"""

    def test_workflow_simulation(self):
        """测试完整的工作流程"""
        manager = VersionManager()

        # 初始版本发布
        manager.create_version("1.0.0", name="web-service", description="Initial release")

        # 注册多个组件版本
        manager.register_version("api-gateway", "1.0.0")
        manager.register_version("database", "2.1.0")
        manager.register_version("cache", "1.5.0")

        # 更新版本
        manager.update_version("web-service", "1.1.0")
        manager.update_version("api-gateway", "1.1.0")

        # 设置当前版本
        manager.set_current_version("web-service", None)

        # 验证状态
        assert manager.version_exists("web-service")
        assert manager.version_exists("api-gateway")
        assert manager.version_exists("database")
        assert manager.version_exists("cache")

        assert manager.get_current_version() == Version("1.1.0")
        assert manager.find_latest_version() == Version("2.1.0")

        # 导出并重新导入
        exported = manager.export_to_dict()
        new_manager = VersionManager()
        new_manager.import_from_dict(exported)

        assert new_manager.get_version("web-service") == Version("1.1.0")
        assert new_manager.get_current_version_name() == "web-service"

        # 清理
        manager.clear_versions()
        assert len(manager.list_version_names()) == 0

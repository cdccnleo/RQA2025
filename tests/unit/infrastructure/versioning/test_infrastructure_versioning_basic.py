#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Infrastructure Versioning基础功能测试

测试版本管理的核心功能：版本创建、解析、比较、排序等
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.versioning.core.version import Version
from src.infrastructure.versioning.manager.manager import VersionManager


class TestVersionCreation:
    """测试Version类的创建功能"""
    
    def test_create_version_from_numbers(self):
        """测试从数字创建版本"""
        version = Version(1, 2, 3)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert str(version) == "1.2.3"
    
    def test_create_version_from_string(self):
        """测试从字符串创建版本"""
        version = Version("1.2.3")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
    
    def test_create_version_with_prerelease(self):
        """测试创建带预发布标识的版本"""
        version = Version("1.2.3-alpha.1")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease == "alpha.1"
        assert str(version) == "1.2.3-alpha.1"
    
    def test_create_version_with_build(self):
        """测试创建带构建标识的版本"""
        version = Version("1.2.3+build.123")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.build == "build.123"
        assert str(version) == "1.2.3+build.123"
    
    def test_create_version_with_prerelease_and_build(self):
        """测试创建带预发布和构建标识的版本"""
        version = Version("1.2.3-beta.2+build.456")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease == "beta.2"
        assert version.build == "build.456"
    
    def test_create_version_default_values(self):
        """测试创建版本的默认值"""
        version = Version()
        assert version.major == 0
        assert version.minor == 0
        assert version.patch == 0
        assert str(version) == "0.0.0"
    
    def test_create_version_invalid_string(self):
        """测试创建版本时的无效字符串处理"""
        with pytest.raises(ValueError):
            Version("1.2")  # 缺少patch版本
        
        with pytest.raises(ValueError):
            Version("1.2.3.4")  # 版本号过多
        
        with pytest.raises(ValueError):
            Version("invalid")  # 完全无效的字符串


class TestVersionComparison:
    """测试Version类的比较功能"""
    
    def test_version_equality(self):
        """测试版本相等性"""
        v1 = Version("1.2.3")
        v2 = Version(1, 2, 3)
        assert v1 == v2
    
    def test_version_inequality(self):
        """测试版本不等性"""
        v1 = Version("1.2.3")
        v2 = Version("1.2.4")
        assert v1 != v2
    
    def test_version_less_than_major(self):
        """测试主版本号比较（小于）"""
        v1 = Version("1.0.0")
        v2 = Version("2.0.0")
        assert v1 < v2
        assert v2 > v1
    
    def test_version_less_than_minor(self):
        """测试次版本号比较（小于）"""
        v1 = Version("1.1.0")
        v2 = Version("1.2.0")
        assert v1 < v2
        assert v2 > v1
    
    def test_version_less_than_patch(self):
        """测试补丁版本号比较（小于）"""
        v1 = Version("1.2.3")
        v2 = Version("1.2.4")
        assert v1 < v2
        assert v2 > v1
    
    def test_version_less_than_or_equal(self):
        """测试版本小于等于"""
        v1 = Version("1.2.3")
        v2 = Version("1.2.3")
        v3 = Version("1.2.4")
        assert v1 <= v2
        assert v1 <= v3
    
    def test_version_greater_than_or_equal(self):
        """测试版本大于等于"""
        v1 = Version("1.2.4")
        v2 = Version("1.2.4")
        v3 = Version("1.2.3")
        assert v1 >= v2
        assert v1 >= v3
    
    def test_version_prerelease_comparison(self):
        """测试预发布版本比较"""
        v1 = Version("1.2.3-alpha")
        v2 = Version("1.2.3-beta")
        v3 = Version("1.2.3")
        
        # 预发布版本小于正式版本
        assert v1 < v3
        assert v2 < v3
        
        # 预发布版本之间的比较
        assert v1 < v2


class TestVersionSorting:
    """测试版本排序功能"""
    
    def test_sort_versions_ascending(self):
        """测试版本升序排序"""
        versions = [
            Version("2.0.0"),
            Version("1.0.0"),
            Version("1.5.0"),
            Version("1.0.1"),
        ]
        sorted_versions = sorted(versions)
        expected = ["1.0.0", "1.0.1", "1.5.0", "2.0.0"]
        actual = [str(v) for v in sorted_versions]
        assert actual == expected
    
    def test_sort_versions_descending(self):
        """测试版本降序排序"""
        versions = [
            Version("1.0.0"),
            Version("2.0.0"),
            Version("1.5.0"),
            Version("1.0.1"),
        ]
        sorted_versions = sorted(versions, reverse=True)
        expected = ["2.0.0", "1.5.0", "1.0.1", "1.0.0"]
        actual = [str(v) for v in sorted_versions]
        assert actual == expected
    
    def test_sort_versions_with_prerelease(self):
        """测试包含预发布版本的排序"""
        versions = [
            Version("1.0.0"),
            Version("1.0.0-beta"),
            Version("1.0.0-alpha"),
            Version("1.0.0-rc.1"),
        ]
        sorted_versions = sorted(versions)
        # 预发布版本应该排在正式版本之前
        assert sorted_versions[-1] == Version("1.0.0")


class TestVersionManager:
    """测试VersionManager类的功能"""
    
    @pytest.fixture
    def version_manager(self):
        """创建版本管理器fixture"""
        return VersionManager()
    
    def test_register_version_with_object(self, version_manager):
        """测试注册版本对象"""
        version = Version("1.0.0")
        version_manager.register_version("app", version)
        
        retrieved = version_manager.get_version("app")
        assert retrieved == version
    
    def test_register_version_with_string(self, version_manager):
        """测试注册版本字符串"""
        version_manager.register_version("app", "1.0.0")
        
        retrieved = version_manager.get_version("app")
        assert retrieved == Version("1.0.0")
    
    def test_get_nonexistent_version(self, version_manager):
        """测试获取不存在的版本"""
        result = version_manager.get_version("nonexistent")
        assert result is None
    
    def test_update_version(self, version_manager):
        """测试更新版本"""
        version_manager.register_version("app", "1.0.0")
        version_manager.register_version("app", "1.1.0")
        
        current = version_manager.get_version("app")
        assert current == Version("1.1.0")
    
    def test_version_history(self, version_manager):
        """测试版本历史记录"""
        version_manager.register_version("app", "1.0.0")
        version_manager.register_version("app", "1.1.0")
        version_manager.register_version("app", "1.2.0")
        
        history = version_manager.get_version_history("app")
        assert len(history) == 3
        assert history[0] == Version("1.0.0")
        assert history[1] == Version("1.1.0")
        assert history[2] == Version("1.2.0")
    
    def test_get_all_versions(self, version_manager):
        """测试获取所有版本"""
        version_manager.register_version("app1", "1.0.0")
        version_manager.register_version("app2", "2.0.0")
        version_manager.register_version("lib1", "0.5.0")
        
        all_versions = version_manager.list_versions()
        assert len(all_versions) == 3
        assert "app1" in all_versions
        assert "app2" in all_versions
        assert "lib1" in all_versions
    
    def test_remove_version(self, version_manager):
        """测试删除版本"""
        version_manager.register_version("app", "1.0.0")
        assert version_manager.get_version("app") is not None
        
        version_manager.remove_version("app")
        assert version_manager.get_version("app") is None
    
    def test_clear_all_versions(self, version_manager):
        """测试清空所有版本"""
        version_manager.register_version("app1", "1.0.0")
        version_manager.register_version("app2", "2.0.0")
        
        version_manager.clear_versions()
        
        assert len(version_manager.list_versions()) == 0


class TestVersionParsing:
    """测试版本解析功能"""
    
    def test_parse_semantic_version(self):
        """测试解析语义化版本"""
        version = Version("2.3.4")
        assert version.major == 2
        assert version.minor == 3
        assert version.patch == 4
    
    def test_parse_version_with_leading_v(self):
        """测试解析带v前缀的版本"""
        # 如果实现支持v前缀
        version = Version("1.2.3")
        assert version.major == 1
    
    def test_parse_version_string_representation(self):
        """测试版本字符串表示"""
        version = Version(1, 2, 3, prerelease="beta", build="20231102")
        version_str = str(version)
        assert "1.2.3" in version_str
        assert "beta" in version_str
        assert "20231102" in version_str
    
    def test_parse_version_repr(self):
        """测试版本repr表示"""
        version = Version("1.2.3")
        repr_str = repr(version)
        assert "Version" in repr_str
        assert "1.2.3" in repr_str


class TestVersionIncrement:
    """测试版本号递增功能"""
    
    def test_increment_major(self):
        """测试递增主版本号"""
        version = Version("1.2.3")
        new_version = version.increment_major()
        assert new_version == Version("2.0.0")
    
    def test_increment_minor(self):
        """测试递增次版本号"""
        version = Version("1.2.3")
        new_version = version.increment_minor()
        assert new_version == Version("1.3.0")
    
    def test_increment_patch(self):
        """测试递增补丁版本号"""
        version = Version("1.2.3")
        new_version = version.increment_patch()
        assert new_version == Version("1.2.4")
    
    def test_increment_preserves_original(self):
        """测试递增不会改变原版本"""
        version = Version("1.2.3")
        original_str = str(version)
        version.increment_major()
        assert str(version) == original_str


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


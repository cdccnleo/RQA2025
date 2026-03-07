"""
测试版本管理核心

覆盖 version.py 中的所有类和功能
"""

import pytest
from src.infrastructure.versioning.core.version import Version


class TestVersion:
    """Version 类测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        version = Version()

        assert version.major == 0
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease is None
        assert version.build is None

    def test_initialization_with_integers(self):
        """测试用整数初始化"""
        version = Version(major=1, minor=2, patch=3)

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
        assert version.build is None

    def test_initialization_with_string_simple(self):
        """测试用简单字符串初始化"""
        version = Version("1.2.3")

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
        assert version.build is None

    def test_initialization_with_string_prerelease(self):
        """测试用带预发布的字符串初始化"""
        version = Version("1.2.3-alpha.1")

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease == "alpha.1"
        assert version.build is None

    def test_initialization_with_string_build(self):
        """测试用带构建信息的字符串初始化"""
        version = Version("1.2.3+build.123")

        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
        assert version.build == "build.123"

    def test_initialization_with_string_full(self):
        """测试用完整字符串初始化"""
        version = Version("2.1.0-beta.2+build.456")

        assert version.major == 2
        assert version.minor == 1
        assert version.patch == 0
        assert version.prerelease == "beta.2"
        assert version.build == "build.456"

    def test_initialization_with_prerelease_and_build_params(self):
        """测试用参数指定预发布和构建信息"""
        version = Version(major=1, minor=0, patch=0, prerelease="rc.1", build="build.789")

        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease == "rc.1"
        assert version.build == "build.789"

    def test_str_representation_simple(self):
        """测试简单版本的字符串表示"""
        version = Version(1, 2, 3)

        assert str(version) == "1.2.3"

    def test_str_representation_with_prerelease(self):
        """测试带预发布的字符串表示"""
        version = Version(1, 2, 3, prerelease="alpha.1")

        assert str(version) == "1.2.3-alpha.1"

    def test_str_representation_with_build(self):
        """测试带构建信息的字符串表示"""
        version = Version(1, 2, 3, build="build.123")

        assert str(version) == "1.2.3+build.123"

    def test_str_representation_full(self):
        """测试完整版本的字符串表示"""
        version = Version(2, 1, 0, prerelease="beta.2", build="build.456")

        assert str(version) == "2.1.0-beta.2+build.456"

    def test_repr_representation(self):
        """测试repr表示"""
        version = Version(1, 2, 3)

        repr_str = repr(version)
        assert "Version" in repr_str
        assert "1.2.3" in repr_str

    def test_equality_same_version(self):
        """测试相同版本的相等性"""
        version1 = Version(1, 2, 3)
        version2 = Version(1, 2, 3)

        assert version1 == version2
        assert version2 == version1

    def test_equality_different_versions(self):
        """测试不同版本的不相等性"""
        version1 = Version(1, 2, 3)
        version2 = Version(1, 2, 4)

        assert version1 != version2
        assert version2 != version1

    def test_equality_with_string(self):
        """测试与字符串的相等性"""
        version = Version(1, 2, 3)

        assert version == "1.2.3"
        assert "1.2.3" == version

    def test_equality_with_different_string(self):
        """测试与不同字符串的不相等性"""
        version = Version(1, 2, 3)

        assert version != "1.2.4"
        assert "1.2.4" != version

    def test_equality_with_prerelease(self):
        """测试带预发布的相等性"""
        version1 = Version(1, 2, 3, prerelease="alpha.1")
        version2 = Version(1, 2, 3, prerelease="alpha.1")

        assert version1 == version2

        version3 = Version(1, 2, 3, prerelease="alpha.2")
        assert version1 != version3

    def test_less_than_comparison(self):
        """测试小于比较"""
        version1 = Version(1, 2, 3)
        version2 = Version(1, 2, 4)
        version3 = Version(1, 3, 0)

        assert version1 < version2
        assert version2 < version3
        assert not version2 < version1

    def test_less_than_or_equal_comparison(self):
        """测试小于等于比较"""
        version1 = Version(1, 2, 3)
        version2 = Version(1, 2, 3)
        version3 = Version(1, 2, 4)

        assert version1 <= version2
        assert version1 <= version3
        assert not version3 <= version1

    def test_greater_than_comparison(self):
        """测试大于比较"""
        version1 = Version(1, 2, 4)
        version2 = Version(1, 2, 3)
        version3 = Version(1, 1, 5)

        assert version1 > version2
        assert version2 > version3
        assert not version2 > version1

    def test_greater_than_or_equal_comparison(self):
        """测试大于等于比较"""
        version1 = Version(1, 2, 3)
        version2 = Version(1, 2, 3)
        version3 = Version(1, 2, 2)

        assert version1 >= version2
        assert version1 >= version3
        assert not version3 >= version1

    def test_comparison_with_strings(self):
        """测试与字符串的比较"""
        version = Version(1, 2, 3)

        assert version < "1.2.4"
        assert version > "1.2.2"
        assert version == "1.2.3"
        assert version <= "1.2.3"
        assert version >= "1.2.3"

    def test_prerelease_version_comparison(self):
        """测试预发布版本比较"""
        stable = Version(1, 0, 0)
        prerelease = Version(1, 0, 0, prerelease="alpha.1")

        # 预发布版本小于稳定版本
        assert prerelease < stable
        assert stable > prerelease

        # 相同主版本的预发布版本比较
        prerelease1 = Version(1, 0, 0, prerelease="alpha.1")
        prerelease2 = Version(1, 0, 0, prerelease="alpha.2")

        assert prerelease1 < prerelease2
        assert prerelease2 > prerelease1

    def test_invalid_version_string(self):
        """测试无效版本字符串"""
        with pytest.raises(ValueError):
            Version("invalid")

        # "1.2" 现在是有效的，会自动填充patch版本为0
        version = Version("1.2")
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 0

        with pytest.raises(ValueError):
            Version("a.b.c")


class TestVersionIntegration:
    """Version 集成测试"""

    def test_version_sorting(self):
        """测试版本排序"""
        versions = [
            Version(1, 0, 0),
            Version(1, 0, 1),
            Version(1, 1, 0),
            Version(2, 0, 0),
            Version(1, 0, 0, prerelease="alpha.1"),
            Version(1, 0, 0, prerelease="beta.1"),
        ]

        sorted_versions = sorted(versions)

        # 预发布版本排在稳定版本之前
        assert str(sorted_versions[0]) == "1.0.0-alpha.1"
        assert str(sorted_versions[1]) == "1.0.0-beta.1"
        assert str(sorted_versions[2]) == "1.0.0"
        assert str(sorted_versions[3]) == "1.0.1"
        assert str(sorted_versions[4]) == "1.1.0"
        assert str(sorted_versions[5]) == "2.0.0"

    def test_semantic_versioning_workflow(self):
        """测试语义版本化工作流"""
        # 初始版本
        current_version = Version(1, 0, 0)

        # 补丁版本更新
        patch_version = Version(current_version.major, current_version.minor, current_version.patch + 1)
        assert str(patch_version) == "1.0.1"

        # 次版本更新
        minor_version = Version(current_version.major, current_version.minor + 1, 0)
        assert str(minor_version) == "1.1.0"

        # 主版本更新
        major_version = Version(current_version.major + 1, 0, 0)
        assert str(major_version) == "2.0.0"

        # 验证版本顺序
        assert current_version < patch_version
        assert patch_version < minor_version
        assert minor_version < major_version

    def test_version_parsing_edge_cases(self):
        """测试版本解析边界情况"""
        # 只有主版本
        version = Version("5")
        assert version.major == 5
        assert version.minor == 0
        assert version.patch == 0

        # 主版本和次版本
        version = Version("2.1")
        assert version.major == 2
        assert version.minor == 1
        assert version.patch == 0

        # 带多个预发布标识符
        version = Version("1.0.0-alpha.1.beta.2")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.prerelease == "alpha.1.beta.2"

        # 构建信息包含多个部分
        version = Version("1.0.0+build.123.sha.abcdef")
        assert version.major == 1
        assert version.minor == 0
        assert version.patch == 0
        assert version.build == "build.123.sha.abcdef"

    def test_version_equality_edge_cases(self):
        """测试版本相等性边界情况"""
        # 相同版本不同表示方式
        version1 = Version(1, 2, 3)
        version2 = Version("1.2.3")

        assert version1 == version2

        # 带预发布和不带预发布
        stable = Version(1, 0, 0)
        prerelease = Version(1, 0, 0, prerelease="alpha.1")

        assert stable != prerelease

        # 带构建信息和不带构建信息（构建信息在比较时被忽略）
        with_build = Version(1, 0, 0, build="build.123")
        without_build = Version(1, 0, 0)

        assert with_build == without_build

    def test_version_string_roundtrip(self):
        """测试版本字符串往返转换"""
        test_cases = [
            "1.0.0",
            "2.1.3",
            "1.0.0-alpha.1",
            "2.0.0-beta.2",
            "1.0.0+build.123",
            "2.1.0-rc.1+build.456.sha.abc123"
        ]

        for version_str in test_cases:
            version = Version(version_str)
            assert str(version) == version_str

    def test_version_comparison_with_mixed_types(self):
        """测试版本与混合类型的比较"""
        version = Version(1, 2, 3)

        # 与字符串比较
        assert version > "1.2.2"
        assert version < "1.2.4"
        assert version == "1.2.3"

        # 与整数比较（应该抛出异常或特殊处理）
        with pytest.raises(TypeError):
            version < 5

    def test_prerelease_ordering(self):
        """测试预发布版本排序"""
        versions = [
            Version("1.0.0-alpha.1"),
            Version("1.0.0-alpha.2"),
            Version("1.0.0-beta.1"),
            Version("1.0.0-beta.2"),
            Version("1.0.0-rc.1"),
            Version("1.0.0-rc.2"),
            Version("1.0.0")
        ]

        sorted_versions = sorted(versions)

        expected_order = [
            "1.0.0-alpha.1",
            "1.0.0-alpha.2",
            "1.0.0-beta.1",
            "1.0.0-beta.2",
            "1.0.0-rc.1",
            "1.0.0-rc.2",
            "1.0.0"
        ]

        actual_order = [str(v) for v in sorted_versions]
        assert actual_order == expected_order

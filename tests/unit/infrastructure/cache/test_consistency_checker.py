#!/usr/bin/env python3
"""
基础设施层 - 一致性检查工具测试

测试consistency_checker.py中的接口一致性检查逻辑
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from abc import ABC, abstractmethod
from src.infrastructure.cache.interfaces.consistency_checker import InterfaceConsistencyChecker


# 测试用的接口和实现类
class ITestInterface(ABC):
    """测试用接口"""

    @abstractmethod
    def method_one(self) -> str:
        """方法一"""
        pass

    @abstractmethod
    def method_two(self, param: int) -> bool:
        """方法二"""
        pass

    @property
    @abstractmethod
    def property_one(self) -> str:
        """属性一"""
        pass


class ValidImplementation(ITestInterface):
    """有效的接口实现"""

    def method_one(self) -> str:
        return "test"

    def method_two(self, param: int) -> bool:
        return param > 0

    @property
    def property_one(self) -> str:
        return "property_value"


class InvalidImplementation:
    """无效的接口实现（未继承接口）"""
    pass


class PartialImplementation(ITestInterface):
    """部分实现的接口（缺少方法和属性）"""

    def method_one(self) -> str:
        return "test"

    # 故意不实现method_two和property_one


class TestInterfaceConsistencyChecker:
    """测试接口一致性检查器"""

    def test_check_interface_implementation_valid(self):
        """测试检查有效的接口实现"""
        result = InterfaceConsistencyChecker.check_interface_implementation(
            ITestInterface, ValidImplementation
        )

        assert result['interface'] == 'ITestInterface'
        assert result['implementation'] == 'ValidImplementation'
        assert result['is_consistent'] is True
        assert len(result['issues']) == 0
        assert result['method_coverage'] == 1.0

    def test_check_interface_implementation_not_inherited(self):
        """测试检查未继承接口的实现"""
        result = InterfaceConsistencyChecker.check_interface_implementation(
            ITestInterface, InvalidImplementation
        )

        assert result['interface'] == 'ITestInterface'
        assert result['implementation'] == 'InvalidImplementation'
        assert result['is_consistent'] is False
        assert len(result['issues']) == 1
        assert '未继承' in result['issues'][0]
        assert result['method_coverage'] == 0.0

    def test_check_interface_implementation_partial(self):
        """测试检查部分实现的接口"""
        result = InterfaceConsistencyChecker.check_interface_implementation(
            ITestInterface, PartialImplementation
        )

        assert result['interface'] == 'ITestInterface'
        assert result['implementation'] == 'PartialImplementation'
        assert result['is_consistent'] is False
        assert len(result['issues']) > 0  # 应该有缺少方法的错误
        assert result['method_coverage'] < 1.0

    def test_check_interface_implementation_same_class(self):
        """测试检查接口本身（抽象类）"""
        result = InterfaceConsistencyChecker.check_interface_implementation(
            ITestInterface, ITestInterface
        )

        assert result['interface'] == 'ITestInterface'
        assert result['implementation'] == 'ITestInterface'
        assert result['is_consistent'] is True  # 接口继承自己是有效的
        assert result['method_coverage'] == 1.0

    def test_get_abstract_methods(self):
        """测试获取抽象方法"""
        methods = InterfaceConsistencyChecker._get_abstract_methods(ITestInterface)

        expected_methods = ['method_one', 'method_two', 'property_one']
        assert set(methods) == set(expected_methods)

    def test_get_abstract_methods_no_abstract(self):
        """测试获取非抽象类的抽象方法"""
        methods = InterfaceConsistencyChecker._get_abstract_methods(ValidImplementation)
        assert methods == []

    def test_get_abstract_methods_empty_interface(self):
        """测试获取空接口的抽象方法"""
        class EmptyInterface(ABC):
            pass

        methods = InterfaceConsistencyChecker._get_abstract_methods(EmptyInterface)
        assert methods == []

    def test_check_naming_convention_interface(self):
        """测试检查接口命名规范"""
        result = InterfaceConsistencyChecker.check_naming_convention(ITestInterface)

        assert result['class_name'] == 'ITestInterface'
        assert result['compliance_score'] == 1.0
        assert len(result['issues']) == 0

    def test_check_naming_convention_valid_implementation(self):
        """测试检查有效实现的命名规范"""
        result = InterfaceConsistencyChecker.check_naming_convention(ValidImplementation)

        assert result['class_name'] == 'ValidImplementation'
        assert result['compliance_score'] == 0.8  # 由于方法名不符合规范
        assert len(result['issues']) > 0

    def test_check_naming_convention_standard_methods(self):
        """测试检查标准方法命名"""

        class StandardImplementation:
            """使用标准方法名的实现类"""

            def get_cache_size(self):
                return 100

            def set_cache_value(self, key, value):
                pass

            def delete_cache_item(self, key):
                pass

            def has_cache_key(self, key):
                return True

            def clear_all_cache(self):
                pass

            def get_cache_stats(self):
                return {}

            def initialize_component(self):
                pass

            def get_component_status(self):
                return "ok"

            def shutdown_component(self):
                pass

            def health_check(self):
                return True

            @property
            def component_name(self):
                return "test"

            @property
            def component_type(self):
                return "cache"

        result = InterfaceConsistencyChecker.check_naming_convention(StandardImplementation)

        assert result['class_name'] == 'StandardImplementation'
        assert result['compliance_score'] == 1.0  # 所有方法都符合规范
        assert len(result['issues']) == 0

    def test_check_naming_convention_non_standard_methods(self):
        """测试检查非标准方法命名"""

        class NonStandardImplementation:
            """使用非标准方法名的实现类"""

            def do_something(self):
                pass

            def process_data(self):
                pass

            def calculate_value(self):
                pass

        result = InterfaceConsistencyChecker.check_naming_convention(NonStandardImplementation)

        assert result['class_name'] == 'NonStandardImplementation'
        assert result['compliance_score'] == 0.8
        assert len(result['issues']) > 0
        assert any('不符合规范' in issue for issue in result['issues'])

    def test_check_naming_convention_empty_class(self):
        """测试检查空类的命名规范"""

        class EmptyClass:
            pass

        result = InterfaceConsistencyChecker.check_naming_convention(EmptyClass)

        assert result['class_name'] == 'EmptyClass'
        assert result['compliance_score'] == 1.0  # 没有方法，所以符合
        assert len(result['issues']) == 0

    def test_integration_check_complete_interface(self):
        """集成测试：检查完整的接口实现"""
        result = InterfaceConsistencyChecker.check_interface_implementation(
            ITestInterface, ValidImplementation
        )

        # 验证所有必要的字段都存在
        required_fields = ['interface', 'implementation', 'is_consistent', 'issues', 'method_coverage']
        for field in required_fields:
            assert field in result

        # 验证结果正确性
        assert result['is_consistent'] is True
        assert result['method_coverage'] == 1.0
        assert len(result['issues']) == 0

    def test_integration_check_with_naming(self):
        """集成测试：结合命名规范检查"""
        # 先检查接口实现
        impl_result = InterfaceConsistencyChecker.check_interface_implementation(
            ITestInterface, ValidImplementation
        )

        # 再检查命名规范
        naming_result = InterfaceConsistencyChecker.check_naming_convention(ValidImplementation)

        # 验证两个检查都能正常工作
        assert impl_result['is_consistent'] is True
        assert naming_result['class_name'] == 'ValidImplementation'
        assert 'compliance_score' in naming_result

    def test_edge_case_no_abstract_methods(self):
        """边缘情况测试：接口没有抽象方法"""

        class EmptyInterface(ABC):
            pass

        class EmptyImplementation(EmptyInterface):
            pass

        result = InterfaceConsistencyChecker.check_interface_implementation(
            EmptyInterface, EmptyImplementation
        )

        assert result['is_consistent'] is True
        assert result['method_coverage'] == 1.0  # 避免除零错误
        assert len(result['issues']) == 0

    def test_edge_case_private_methods_ignored(self):
        """边缘情况测试：私有方法被忽略"""

        class InterfaceWithPrivate(ABC):
            @abstractmethod
            def public_method(self):
                pass

            @abstractmethod
            def _private_method(self):
                pass

        methods = InterfaceConsistencyChecker._get_abstract_methods(InterfaceWithPrivate)
        assert 'public_method' in methods
        assert '_private_method' not in methods  # 私有方法被过滤掉

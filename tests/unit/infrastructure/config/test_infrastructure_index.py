"""
测试infrastructure_index.py模块
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest

try:
    from src.infrastructure.config.tools.infrastructure_index import (
        get_interface_statistics,
        get_interfaces_by_category,
        get_interfaces_by_priority,
        get_interfaces_by_dependency,
        get_interface_status,
        get_interface_test_status,
        get_interface_documentation_status,
        INTERFACE_CATEGORIES,
        INTERFACE_PRIORITY,
        INTERFACE_DEPENDENCIES,
        INTERFACE_IMPLEMENTATION_STATUS,
        INTERFACE_TEST_STATUS,
        INTERFACE_DOCUMENTATION_STATUS
    )
except ImportError:
    pytest.skip("infrastructure_index模块导入失败", allow_module_level=True)


class TestInfrastructureIndex:
    """测试infrastructure_index模块"""

    def test_get_interface_statistics(self):
        """测试获取接口统计信息"""
        stats = get_interface_statistics()
        
        # 验证返回的统计信息结构
        assert isinstance(stats, dict)
        assert 'total_interfaces' in stats
        assert 'implemented_interfaces' in stats
        assert 'tested_interfaces' in stats
        assert 'documented_interfaces' in stats
        assert 'implementation_rate' in stats
        assert 'test_coverage_rate' in stats
        assert 'documentation_rate' in stats
        
        # 验证数据类型
        assert isinstance(stats['total_interfaces'], int)
        assert isinstance(stats['implemented_interfaces'], int)
        assert isinstance(stats['tested_interfaces'], int)
        assert isinstance(stats['documented_interfaces'], int)
        assert isinstance(stats['implementation_rate'], float)
        assert isinstance(stats['test_coverage_rate'], float)
        assert isinstance(stats['documentation_rate'], float)

    def test_get_interfaces_by_category(self):
        """测试按类别获取接口"""
        # 测试已知类别
        config_interfaces = get_interfaces_by_category('configuration')
        assert isinstance(config_interfaces, list)
        
        monitoring_interfaces = get_interfaces_by_category('monitoring')
        assert isinstance(monitoring_interfaces, list)
        
        cache_interfaces = get_interfaces_by_category('cache')
        assert isinstance(cache_interfaces, list)
        
        # 测试不存在的类别
        unknown_interfaces = get_interfaces_by_category('unknown_category')
        assert unknown_interfaces == []

    def test_get_interfaces_by_priority(self):
        """测试按优先级获取接口"""
        # 测试已知优先级
        high_priority = get_interfaces_by_priority('high')
        assert isinstance(high_priority, list)
        
        medium_priority = get_interfaces_by_priority('medium')
        assert isinstance(medium_priority, list)
        
        low_priority = get_interfaces_by_priority('low')
        assert isinstance(low_priority, list)
        
        # 测试不存在的优先级
        unknown_priority = get_interfaces_by_priority('unknown_priority')
        assert unknown_priority == []

    def test_get_interfaces_by_dependency(self):
        """测试按依赖关系获取接口"""
        # 测试已知接口的依赖
        dependencies = get_interfaces_by_dependency('IConfigurationManager')
        assert isinstance(dependencies, list)
        
        # 测试不存在的接口
        unknown_dependencies = get_interfaces_by_dependency('UnknownInterface')
        assert unknown_dependencies == []

    def test_get_interface_status(self):
        """测试获取接口状态"""
        # 测试已知接口
        status = get_interface_status('IConfigurationManager')
        assert isinstance(status, dict)
        assert 'status' in status
        assert 'category' in status
        
        # 测试不存在的接口
        unknown_status = get_interface_status('UnknownInterface')
        assert unknown_status['status'] == 'unknown'
        assert unknown_status['category'] == 'unknown'

    def test_get_interface_test_status(self):
        """测试获取接口测试状态"""
        # 测试已知接口
        test_status = get_interface_test_status('IConfigurationManager')
        assert isinstance(test_status, dict)
        assert 'status' in test_status
        assert 'category' in test_status
        
        # 测试不存在的接口
        unknown_test_status = get_interface_test_status('UnknownInterface')
        assert unknown_test_status['status'] == 'unknown'
        assert unknown_test_status['category'] == 'unknown'

    def test_get_interface_documentation_status(self):
        """测试获取接口文档状态"""
        # 测试已知接口
        doc_status = get_interface_documentation_status('IConfigurationManager')
        assert isinstance(doc_status, dict)
        assert 'status' in doc_status
        assert 'category' in doc_status
        
        # 测试不存在的接口
        unknown_doc_status = get_interface_documentation_status('UnknownInterface')
        assert unknown_doc_status['status'] == 'unknown'
        assert unknown_doc_status['category'] == 'unknown'

    def test_interface_categories_structure(self):
        """测试接口分类结构"""
        assert isinstance(INTERFACE_CATEGORIES, dict)
        
        for category, interfaces in INTERFACE_CATEGORIES.items():
            assert isinstance(category, str)
            assert isinstance(interfaces, list)
            
            for interface in interfaces:
                assert isinstance(interface, str)

    def test_interface_priority_structure(self):
        """测试接口优先级结构"""
        assert isinstance(INTERFACE_PRIORITY, dict)
        
        for priority, interfaces in INTERFACE_PRIORITY.items():
            assert isinstance(priority, str)
            assert isinstance(interfaces, list)
            
            for interface in interfaces:
                assert isinstance(interface, str)

    def test_interface_dependencies_structure(self):
        """测试接口依赖关系结构"""
        assert isinstance(INTERFACE_DEPENDENCIES, dict)
        
        for interface, dependencies in INTERFACE_DEPENDENCIES.items():
            assert isinstance(interface, str)
            assert isinstance(dependencies, list)
            
            for dependency in dependencies:
                assert isinstance(dependency, str)

    def test_interface_implementation_status_structure(self):
        """测试接口实现状态结构"""
        assert isinstance(INTERFACE_IMPLEMENTATION_STATUS, dict)
        
        for category, statuses in INTERFACE_IMPLEMENTATION_STATUS.items():
            assert isinstance(category, str)
            assert isinstance(statuses, dict)
            assert 'implemented' in statuses
            assert 'partially_implemented' in statuses
            assert 'not_implemented' in statuses
            
            for status_type, interfaces in statuses.items():
                assert isinstance(interfaces, list)
                for interface in interfaces:
                    assert isinstance(interface, str)

    def test_interface_test_status_structure(self):
        """测试接口测试状态结构"""
        assert isinstance(INTERFACE_TEST_STATUS, dict)
        
        for category, statuses in INTERFACE_TEST_STATUS.items():
            assert isinstance(category, str)
            assert isinstance(statuses, dict)
            assert 'tested' in statuses
            assert 'partially_tested' in statuses
            assert 'not_tested' in statuses

    def test_interface_documentation_status_structure(self):
        """测试接口文档状态结构"""
        assert isinstance(INTERFACE_DOCUMENTATION_STATUS, dict)
        
        for category, statuses in INTERFACE_DOCUMENTATION_STATUS.items():
            assert isinstance(category, str)
            assert isinstance(statuses, dict)
            assert 'documented' in statuses
            assert 'partially_documented' in statuses
            assert 'not_documented' in statuses

    def test_statistics_calculation(self):
        """测试统计信息计算的正确性"""
        stats = get_interface_statistics()
        
        # 验证比例值在0-1之间
        assert 0 <= stats['implementation_rate'] <= 1
        assert 0 <= stats['test_coverage_rate'] <= 1
        assert 0 <= stats['documentation_rate'] <= 1
        
        # 验证总数不为负数
        assert stats['total_interfaces'] >= 0
        assert stats['implemented_interfaces'] >= 0
        assert stats['tested_interfaces'] >= 0
        assert stats['documented_interfaces'] >= 0

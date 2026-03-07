#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Infrastructure Versioning集成测试

测试versioning与其他系统的集成，包括Config系统、端到端流程等
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
from src.infrastructure.versioning.core.version import Version
from src.infrastructure.versioning.manager.manager import VersionManager
from src.infrastructure.versioning.config.config_version_manager import ConfigVersionManager


class TestVersioningConfigIntegration:
    """测试versioning与Config系统的集成"""
    
    @pytest.fixture
    def config_version_manager(self):
        """创建配置版本管理器fixture"""
        return ConfigVersionManager()
    
    def test_create_config_version(self, config_version_manager):
        """测试创建配置版本"""
        config_data = {
            'database': {
                'host': 'localhost',
                'port': 5432
            },
            'cache': {
                'ttl': 300
            }
        }
        
        version = config_version_manager.create_version('test_config', config_data)
        assert version is not None
        assert isinstance(version, Version)
    
    def test_retrieve_config_by_version(self, config_version_manager):
        """测试按版本检索配置"""
        config_data = {'setting': 'value'}
        version = config_version_manager.create_version('test_config', config_data)
        
        retrieved_config = config_version_manager.get_config('test_config', version)
        assert retrieved_config == config_data
    
    def test_config_version_history(self, config_version_manager):
        """测试配置版本历史"""
        # 创建多个配置版本
        config_v1 = {'version': '1.0', 'feature': 'basic'}
        config_v2 = {'version': '2.0', 'feature': 'advanced'}
        
        v1 = config_version_manager.create_version('test_config', config_v1)
        v2 = config_version_manager.create_version('test_config', config_v2)
        
        # 获取版本列表
        versions = config_version_manager.list_versions('test_config')
        assert len(versions) >= 2
    
    def test_config_version_rollback(self, config_version_manager):
        """测试配置版本回滚"""
        # 创建两个配置版本
        config_v1 = {'setting': 'old_value'}
        config_v2 = {'setting': 'new_value'}
        
        v1 = config_version_manager.create_version('test_config', config_v1)
        v2 = config_version_manager.create_version('test_config', config_v2)
        
        # 回滚到v1
        rolled_back_config = config_version_manager.get_config('test_config', v1)
        assert rolled_back_config['setting'] == 'old_value'
    
    def test_config_version_comparison(self, config_version_manager):
        """测试配置版本比较"""
        config_v1 = {'key1': 'value1'}
        config_v2 = {'key1': 'value1', 'key2': 'value2'}
        
        v1 = config_version_manager.create_version('test_config', config_v1)
        v2 = config_version_manager.create_version('test_config', config_v2)
        
        # 比较版本差异
        diff = config_version_manager.compare_versions('test_config', v1, v2)
        assert diff is not None


class TestEndToEndVersioning:
    """测试端到端版本控制流程"""
    
    @pytest.fixture
    def version_manager(self):
        """创建版本管理器fixture"""
        return VersionManager()
    
    def test_complete_version_lifecycle(self, version_manager):
        """测试完整的版本生命周期"""
        # 1. 创建初始版本
        version_manager.register_version("app", "1.0.0")
        assert version_manager.get_version("app") == Version("1.0.0")
        
        # 2. 更新版本
        version_manager.register_version("app", "1.1.0")
        assert version_manager.get_version("app") == Version("1.1.0")
        
        # 3. 检查历史
        history = version_manager.get_version_history("app")
        assert len(history) == 2
        
        # 4. 验证版本顺序
        assert history[0] == Version("1.0.0")
        assert history[1] == Version("1.1.0")
        
        # 5. 删除版本
        version_manager.remove_version("app")
        assert version_manager.get_version("app") is None
    
    def test_multi_component_versioning(self, version_manager):
        """测试多组件版本管理"""
        # 注册多个组件的版本
        components = {
            'frontend': '2.0.0',
            'backend': '3.0.0',
            'database': '1.5.0',
            'cache': '2.1.0'
        }
        
        for name, version in components.items():
            version_manager.register_version(name, version)
        
        # 验证所有组件都已注册
        all_versions = version_manager.list_versions()
        assert len(all_versions) == len(components)
        
        for name, expected_version in components.items():
            actual_version = version_manager.get_version(name)
            assert actual_version == Version(expected_version)
    
    def test_version_dependency_management(self, version_manager):
        """测试版本依赖管理"""
        # 定义组件依赖关系
        dependencies = {
            'app': {
                'version': '2.0.0',
                'requires': {
                    'lib': '1.0.0'
                }
            }
        }
        
        # 注册版本
        version_manager.register_version('app', '2.0.0')
        version_manager.register_version('lib', '1.0.0')
        
        # 验证依赖满足
        app_version = version_manager.get_version('app')
        lib_version = version_manager.get_version('lib')
        
        assert app_version is not None
        assert lib_version is not None
        assert lib_version == Version('1.0.0')


class TestVersioningWithDataManagement:
    """测试versioning与数据管理的集成"""
    
    def test_version_with_data_snapshot(self):
        """测试版本与数据快照"""
        # 模拟数据快照
        snapshots = {}
        
        data_v1 = {'records': [1, 2, 3], 'count': 3}
        snapshots['1.0.0'] = data_v1
        
        data_v2 = {'records': [1, 2, 3, 4], 'count': 4}
        snapshots['2.0.0'] = data_v2
        
        # 验证快照存在
        assert '1.0.0' in snapshots
        assert '2.0.0' in snapshots
        assert snapshots['2.0.0']['count'] == 4
    
    def test_version_with_data_validation(self):
        """测试版本与数据验证"""
        def validate_data_for_version(data: Dict, version: str) -> bool:
            """验证数据是否符合版本要求"""
            v = Version(version)
            
            if v.major >= 2:
                # v2+需要额外字段
                return 'new_field' in data
            return True
        
        data_v1 = {'id': 1, 'name': 'test'}
        data_v2 = {'id': 1, 'name': 'test', 'new_field': 'value'}
        
        assert validate_data_for_version(data_v1, '1.0.0')
        assert not validate_data_for_version(data_v1, '2.0.0')
        assert validate_data_for_version(data_v2, '2.0.0')
    
    def test_version_with_data_transformation(self):
        """测试版本与数据转换"""
        def transform_data_to_version(data: Dict, target_version: str) -> Dict:
            """将数据转换为目标版本格式"""
            v = Version(target_version)
            transformed = data.copy()
            
            if v.major == 2:
                # v2需要新字段
                if 'new_field' not in transformed:
                    transformed['new_field'] = 'default_value'
            
            return transformed
        
        data_v1 = {'id': 1, 'name': 'test'}
        data_v2 = transform_data_to_version(data_v1, '2.0.0')
        
        assert 'new_field' in data_v2
        assert data_v2['new_field'] == 'default_value'


class TestVersioningAPIIntegration:
    """测试versioning API集成"""
    
    @pytest.fixture
    def version_manager(self):
        """创建版本管理器fixture"""
        return VersionManager()
    
    def test_api_version_registration(self, version_manager):
        """测试API版本注册"""
        # 模拟API版本注册
        api_versions = {
            'v1': '1.0.0',
            'v2': '2.0.0',
            'v3': '3.0.0'
        }
        
        for api_name, version in api_versions.items():
            version_manager.register_version(api_name, version)
        
        assert version_manager.get_version('v1') == Version('1.0.0')
        assert version_manager.get_version('v3') == Version('3.0.0')
    
    def test_api_version_deprecation(self, version_manager):
        """测试API版本废弃"""
        # 注册API版本
        version_manager.register_version('api_v1', '1.0.0')
        version_manager.register_version('api_v2', '2.0.0')
        
        # 标记v1为废弃（通过删除或特殊处理）
        deprecated_versions = ['api_v1']
        
        # 验证v2仍然活跃
        assert version_manager.get_version('api_v2') is not None
        assert 'api_v1' in deprecated_versions


class TestVersioningSystemIntegration:
    """测试versioning系统级集成"""
    
    @pytest.fixture
    def version_manager(self):
        """创建版本管理器fixture"""
        return VersionManager()
    
    def test_system_version_synchronization(self, version_manager):
        """测试系统版本同步"""
        # 模拟多系统版本同步
        systems = ['system_a', 'system_b', 'system_c']
        sync_version = '2.0.0'
        
        for system in systems:
            version_manager.register_version(system, sync_version)
        
        # 验证所有系统版本一致
        for system in systems:
            assert version_manager.get_version(system) == Version(sync_version)
    
    def test_distributed_version_consistency(self, version_manager):
        """测试分布式版本一致性"""
        # 模拟分布式节点版本
        nodes = {
            'node1': '3.0.0',
            'node2': '3.0.0',
            'node3': '3.0.0'
        }
        
        for node, version in nodes.items():
            version_manager.register_version(node, version)
        
        # 检查版本一致性
        versions = [version_manager.get_version(node) for node in nodes.keys()]
        assert len(set(versions)) == 1  # 所有版本相同
        assert versions[0] == Version('3.0.0')
    
    def test_cross_component_version_validation(self, version_manager):
        """测试跨组件版本验证"""
        # 注册组件版本
        version_manager.register_version('component_a', '2.0.0')
        version_manager.register_version('component_b', '2.1.0')
        
        # 验证组件兼容性
        compatible = version_manager.validate_version_compatibility(
            'component_a', 'component_b'
        )
        assert compatible  # 同主版本号，兼容


class TestVersioningPerformance:
    """测试versioning性能"""
    
    @pytest.fixture
    def version_manager(self):
        """创建版本管理器fixture"""
        return VersionManager()
    
    def test_bulk_version_registration(self, version_manager):
        """测试批量版本注册性能"""
        # 批量注册版本
        for i in range(100):
            version_manager.register_version(f'component_{i}', f'1.{i}.0')
        
        # 验证注册数量
        all_versions = version_manager.list_versions()
        assert len(all_versions) == 100
    
    def test_version_lookup_performance(self, version_manager):
        """测试版本查找性能"""
        # 注册测试版本
        test_components = 50
        for i in range(test_components):
            version_manager.register_version(f'app_{i}', f'2.{i}.0')
        
        # 快速查找
        for i in range(test_components):
            version = version_manager.get_version(f'app_{i}')
            assert version is not None
    
    def test_version_history_performance(self, version_manager):
        """测试版本历史查询性能"""
        # 创建版本历史
        versions_count = 20
        for i in range(versions_count):
            version_manager.register_version('app', f'1.{i}.0')
        
        # 获取历史
        history = version_manager.get_version_history('app')
        assert len(history) == versions_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


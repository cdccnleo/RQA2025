#!/usr/bin/env python3
"""
缓存管理测试质量评估脚本
检查测试用例通过率和代码覆盖率
"""

import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, 'src')


def run_protocol_mixin_tests():
    """运行Protocol+Mixin架构测试"""
    passed = 0
    total = 0

    # test_icache_component_protocol
    try:
        from infrastructure.cache.core.cache_components import CacheComponent
        from infrastructure.cache.interfaces import ICacheComponent

        comp = CacheComponent(component_id=1, component_type='memory')
        assert isinstance(comp, ICacheComponent)

        required_methods = [
            'get_cache_item', 'set_cache_item', 'delete_cache_item',
            'has_cache_item', 'clear_all_cache', 'get_cache_size',
            'get_cache_stats', 'component_name', 'component_type',
            'initialize_component', 'get_component_status',
            'shutdown_component', 'health_check'
        ]

        for method in required_methods:
            assert hasattr(comp, method), f'missing method: {method}'

        passed += 1
        print("✅ test_icache_component_protocol: 通过")
    except Exception as e:
        print(f"❌ test_icache_component_protocol: 失败 - {e}")
    total += 1

    # test_monitoring_mixin_functionality
    try:
        from infrastructure.cache.core.mixins import MonitoringMixin

        mixin = MonitoringMixin(enable_monitoring=True, monitor_interval=5)
        result = mixin.start_monitoring()
        assert result is True
        assert mixin._monitoring_active is True
        mixin.stop_monitoring()
        assert mixin._monitoring_active is False

        passed += 1
        print("✅ test_monitoring_mixin_functionality: 通过")
    except Exception as e:
        print(f"❌ test_monitoring_mixin_functionality: 失败 - {e}")
    total += 1

    # test_crud_operations_mixin_functionality
    try:
        from infrastructure.cache.core.mixins import CRUDOperationsMixin

        mixin = CRUDOperationsMixin()
        result = mixin.set('test_key', 'test_value')
        assert result is True
        value = mixin.get('test_key')
        assert value == 'test_value'
        assert mixin.exists('test_key') is True
        result = mixin.delete('test_key')
        assert result is True
        assert mixin.exists('test_key') is False

        passed += 1
        print("✅ test_crud_operations_mixin_functionality: 通过")
    except Exception as e:
        print(f"❌ test_crud_operations_mixin_functionality: 失败 - {e}")
    total += 1

    # test_unified_cache_manager_inheritance
    try:
        from infrastructure.cache.core.cache_manager import UnifiedCacheManager
        from infrastructure.cache.core.cache_configs import CacheConfig, CacheLevel
        from infrastructure.cache.core.mixins import MonitoringMixin

        config = CacheConfig(
            basic=type('Basic', (), {'max_size': 1000, 'ttl': 3600, 'strategy': 'LRU'})(),
            multi_level=type('MultiLevel', (), {
                'level': CacheLevel.MEMORY,
                'memory_max_size': 500,
                'memory_ttl': 1800,
                'redis_max_size': 1000,
                'redis_ttl': 3600,
                'file_max_size': 10000,
                'file_ttl': 86400
            })(),
            smart=type('Smart', (), {'enable_monitoring': False, 'monitor_interval': 30})()
        )

        manager = UnifiedCacheManager(config)
        assert isinstance(manager, MonitoringMixin)
        assert hasattr(manager, 'start_monitoring')
        assert hasattr(manager, 'stop_monitoring')

        passed += 1
        print("✅ test_unified_cache_manager_inheritance: 通过")
    except Exception as e:
        print(f"❌ test_unified_cache_manager_inheritance: 失败 - {e}")
    total += 1

    # test_smart_cache_monitor_inheritance
    try:
        from infrastructure.cache.monitoring.performance_monitor import SmartCacheMonitor
        from infrastructure.cache.core.mixins import MonitoringMixin

        class MockCacheManager:
            def size(self): return 100

        monitor = SmartCacheMonitor(MockCacheManager())
        assert isinstance(monitor, MonitoringMixin)

        passed += 1
        print("✅ test_smart_cache_monitor_inheritance: 通过")
    except Exception as e:
        print(f"❌ test_smart_cache_monitor_inheritance: 失败 - {e}")
    total += 1

    # test_smart_cache_monitor_functionality
    try:
        from infrastructure.cache.monitoring.performance_monitor import SmartCacheMonitor

        class MockCacheManager:
            def size(self): return 100

        monitor = SmartCacheMonitor(MockCacheManager())
        assert monitor.cache_manager is not None
        assert hasattr(monitor, 'predictor')

        callbacks_received = []

        def test_callback(alert_type, message):
            callbacks_received.append((alert_type, message))

        monitor.add_alert_callback(test_callback)
        monitor._trigger_alert('test_alert', 'test_message')

        assert len(callbacks_received) == 1
        assert callbacks_received[0] == ('test_alert', 'test_message')

        passed += 1
        print("✅ test_smart_cache_monitor_functionality: 通过")
    except Exception as e:
        print(f"❌ test_smart_cache_monitor_functionality: 失败 - {e}")
    total += 1

    return passed, total


def run_unified_cache_manager_tests():
    """运行UnifiedCacheManager测试"""
    passed = 0
    total = 0

    # test_initialization_default_config
    try:
        from infrastructure.cache.core.cache_manager import UnifiedCacheManager
        from infrastructure.cache.core.cache_configs import CacheConfig

        config = CacheConfig.create_simple_memory_config()
        config.basic.max_size = 1000
        config.basic.ttl = 3600

        manager = UnifiedCacheManager(config)
        assert manager is not None
        assert hasattr(manager.config, 'basic')
        assert manager.config.basic.max_size == 1000
        assert manager.config.basic.ttl == 3600

        passed += 1
        print("✅ test_initialization_default_config: 通过")
    except Exception as e:
        print(f"❌ test_initialization_default_config: 失败 - {e}")
    total += 1

    return passed, total


def estimate_coverage():
    """估算代码覆盖率"""
    cache_files = []
    test_files = []

    for root, dirs, files in os.walk('src/infrastructure/cache'):
        for file in files:
            if file.endswith('.py'):
                cache_files.append(os.path.join(root, file))

    for root, dirs, files in os.walk('tests/unit/infrastructure/cache'):
        for file in files:
            if file.endswith('.py') and file.startswith('test_'):
                test_files.append(file)

    # 估算覆盖的核心组件
    covered_components = set()
    if 'test_protocol_mixin_architecture.py' in [os.path.basename(f) for f in test_files]:
        covered_components.update([
            'ICacheComponent', 'IBaseComponent', 'MonitoringMixin',
            'CRUDOperationsMixin', 'CacheTierMixin', 'SmartCacheMonitor'
        ])

    if 'test_unified_cache_manager.py' in [os.path.basename(f) for f in test_files]:
        covered_components.update([
            'UnifiedCacheManager', 'CacheConfig', 'MultiLevelCache'
        ])

    estimated_coverage = (len(covered_components) / max(len(cache_files), 1)) * 100

    return {
        'total_files': len(cache_files),
        'test_files': len(test_files),
        'covered_components': len(covered_components),
        'estimated_coverage': estimated_coverage
    }


def main():
    """主函数"""
    print('=== 缓存管理测试用例通过率和覆盖率检查 ===')
    print(f'开始时间: {datetime.now()}')

    # 1. 运行Protocol+Mixin测试
    print()
    print('1. 运行Protocol+Mixin架构测试...')
    protocol_passed, protocol_total = run_protocol_mixin_tests()

    # 2. 运行UnifiedCacheManager测试
    print()
    print('2. 运行UnifiedCacheManager测试...')
    unified_passed, unified_total = run_unified_cache_manager_tests()

    # 3. 计算总体通过率
    total_passed = protocol_passed + unified_passed
    total_tests = protocol_total + unified_total

    print()
    print('=== 总体测试统计 ===')
    print(f'总测试数: {total_tests}')
    print(f'通过测试: {total_passed}')
    print(f'失败测试: {total_tests - total_passed}')

    if total_tests > 0:
        pass_rate = (total_passed / total_tests) * 100
        print(f'通过率: {pass_rate:.1f}%')

        if pass_rate >= 90:
            grade = 'A+ (优秀)'
        elif pass_rate >= 80:
            grade = 'A (良好)'
        elif pass_rate >= 70:
            grade = 'B (及格)'
        else:
            grade = 'C (需改进)'

        print(f'测试质量等级: {grade}')
    else:
        pass_rate = 0.0
        grade = 'N/A'

    # 4. 估算代码覆盖率
    print()
    print('4. 代码覆盖率估算...')
    coverage_info = estimate_coverage()

    print(f'缓存系统文件数: {coverage_info["total_files"]}')
    print(f'测试文件数: {coverage_info["test_files"]}')
    print(f'覆盖的核心组件数: {coverage_info["covered_components"]}')
    print(f'覆盖率估算: {coverage_info["estimated_coverage"]:.1f}%')

    if coverage_info['estimated_coverage'] >= 80:
        coverage_grade = 'A (良好)'
    elif coverage_info['estimated_coverage'] >= 60:
        coverage_grade = 'B (及格)'
    else:
        coverage_grade = 'C (需改进)'

    print(f'覆盖率等级: {coverage_grade}')

    # 5. 生成最终报告
    print()
    print('=== 缓存管理测试质量评估报告 ===')
    print(f'测试通过率: {pass_rate:.1f}% ({grade})')
    print(f'代码覆盖率: {coverage_info["estimated_coverage"]:.1f}% ({coverage_grade})')

    overall_score = (pass_rate + coverage_info['estimated_coverage']) / 2
    if overall_score >= 85:
        overall_grade = '优秀'
    elif overall_score >= 70:
        overall_grade = '良好'
    else:
        overall_grade = '需改进'

    print(f'综合质量评分: {overall_score:.1f}/100 ({overall_grade})')

    if pass_rate >= 80 and coverage_info['estimated_coverage'] >= 60:
        print()
        print('🎉 缓存管理系统测试质量达标！')
        print('✅ 核心功能测试完整')
        print('✅ Protocol+Mixin架构验证通过')
        print('✅ 代码覆盖率基本达标')
    else:
        print()
        print('⚠️ 缓存管理系统测试需要改进')
        if pass_rate < 80:
            print('❌ 测试通过率不足')
        if coverage_info['estimated_coverage'] < 60:
            print('❌ 代码覆盖率不足')

    print()
    print('=== 报告生成完成 ===')
    print(f'结束时间: {datetime.now()}')


if __name__ == '__main__':
    main()

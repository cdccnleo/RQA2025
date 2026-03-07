"""
基础设施层核心组件优化验证脚本

快速验证优化后的代码功能完整性

作者: RQA2025团队
创建时间: 2025-10-23
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_parameter_objects():
    """测试参数对象模块"""
    print("📦 测试参数对象模块...")
    
    try:
        from src.infrastructure.core.parameter_objects import (
            HealthCheckParams,
            ConfigValidationParams,
            ResourceUsageParams
        )
        
        # 测试健康检查参数
        params1 = HealthCheckParams(service_name="test")
        assert params1.timeout == 30
        assert params1.retry_count == 3
        print("  ✅ HealthCheckParams 创建成功")
        
        # 测试配置验证参数
        params2 = ConfigValidationParams(
            value=100,
            expected_type=int,
            min_value=0,
            max_value=200
        )
        assert params2.validate() is True
        print("  ✅ ConfigValidationParams 验证成功")
        
        # 测试资源使用参数
        params3 = ResourceUsageParams(
            resource_type="memory",
            current_usage=850,
            total_capacity=1000
        )
        assert params3.usage_percentage == 85.0
        assert params3.is_warning_level is True
        print("  ✅ ResourceUsageParams 计算属性正常")
        
        print("  ✅ 参数对象模块测试通过\n")
        return True
        
    except Exception as e:
        print(f"  ❌ 参数对象模块测试失败: {e}\n")
        return False


def test_semantic_constants():
    """测试语义化常量"""
    print("🔢 测试语义化常量...")
    
    try:
        from src.infrastructure.core.constants import (
            CacheConstants,
            MonitoringConstants,
            NetworkConstants
        )
        
        # 测试缓存常量
        assert CacheConstants.ONE_KB == 1024
        assert CacheConstants.ONE_MB == 1048576
        assert CacheConstants.DEFAULT_CACHE_SIZE == CacheConstants.ONE_KB
        assert CacheConstants.DEFAULT_TTL == CacheConstants.ONE_HOUR
        print("  ✅ CacheConstants 语义化成功")
        
        # 测试监控常量
        assert hasattr(MonitoringConstants, 'CPU_USAGE_THRESHOLD_PERCENT')
        assert MonitoringConstants.TEN_THOUSAND == 10000
        assert MonitoringConstants.MAX_METRICS_QUEUE_SIZE == MonitoringConstants.TEN_THOUSAND
        print("  ✅ MonitoringConstants 语义化成功")
        
        # 测试网络常量
        assert NetworkConstants.EIGHT_KB == 8192
        assert NetworkConstants.DEFAULT_BUFFER_SIZE == NetworkConstants.EIGHT_KB
        print("  ✅ NetworkConstants 语义化成功")
        
        print("  ✅ 语义化常量测试通过\n")
        return True
        
    except Exception as e:
        print(f"  ❌ 语义化常量测试失败: {e}\n")
        return False


def test_mock_services():
    """测试Mock服务基类"""
    print("🎭 测试Mock服务基类...")
    
    try:
        from src.infrastructure.core.mock_services import (
            BaseMockService,
            SimpleMockDict,
            SimpleMockLogger,
            SimpleMockMonitor
        )
        
        # 测试SimpleMockDict
        mock_dict = SimpleMockDict(service_name="test_cache")
        mock_dict.set("key1", "value1")
        assert mock_dict.get("key1") == "value1"
        assert mock_dict.exists("key1") is True
        assert mock_dict.call_count == 3  # set, get, exists
        print("  ✅ SimpleMockDict 工作正常")
        
        # 测试SimpleMockLogger
        mock_logger = SimpleMockLogger()
        mock_logger.info("test message")
        mock_logger.error("error message")
        logs = mock_logger.get_logs()
        assert len(logs) == 2
        print("  ✅ SimpleMockLogger 工作正常")
        
        # 测试SimpleMockMonitor
        mock_monitor = SimpleMockMonitor()
        mock_monitor.record_metric("cpu", 75.5)
        mock_monitor.increment_counter("requests", 10)
        assert mock_monitor.get_metric_values("cpu")[0] == 75.5
        assert mock_monitor.get_counter_value("requests") == 10
        print("  ✅ SimpleMockMonitor 工作正常")
        
        # 测试健康检查
        assert mock_dict.is_healthy() is True
        health = mock_dict.check_health()
        assert health['service'] == "test_cache"
        print("  ✅ Mock健康检查功能正常")
        
        # 测试失败模式
        mock_dict.set_failure_mode(True, ValueError("Test error"))
        try:
            mock_dict.get("key1")
            assert False, "应该抛出异常"
        except ValueError as e:
            assert "Test error" in str(e)
            print("  ✅ Mock失败模式功能正常")
        
        print("  ✅ Mock服务基类测试通过\n")
        return True
        
    except Exception as e:
        print(f"  ❌ Mock服务基类测试失败: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """测试向后兼容性"""
    print("🔄 测试向后兼容性...")
    
    try:
        # 测试原有异常类仍然工作
        from src.infrastructure.core.exceptions import (
            InfrastructureException,
            ConfigurationError,
            CacheError
        )
        
        try:
            raise ConfigurationError("test error", "test_key")
        except InfrastructureException as e:
            assert "test error" in str(e)
            print("  ✅ 异常类向后兼容")
        
        # 测试原有常量仍然可用
        from src.infrastructure.core.constants import (
            DEFAULT_TIMEOUT,
            DEFAULT_CACHE_SIZE,
            DEFAULT_POOL_SIZE
        )
        
        assert DEFAULT_CACHE_SIZE == 1024
        assert DEFAULT_POOL_SIZE == 10
        print("  ✅ 常量快捷访问向后兼容")
        
        print("  ✅ 向后兼容性测试通过\n")
        return True
        
    except Exception as e:
        print(f"  ❌ 向后兼容性测试失败: {e}\n")
        return False


def main():
    """主测试函数"""
    print("=" * 70)
    print("基础设施层核心组件优化验证测试")
    print("=" * 70)
    print()
    
    results = []
    
    # 执行各项测试
    results.append(("参数对象模块", test_parameter_objects()))
    results.append(("语义化常量", test_semantic_constants()))
    results.append(("Mock服务基类", test_mock_services()))
    results.append(("向后兼容性", test_backward_compatibility()))
    
    # 汇总结果
    print("=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")
    
    print()
    print(f"总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有优化验证测试通过！")
        print("✅ 代码质量改进成功")
        print("✅ 功能完整性保持")
        print("✅ 向后兼容性保证")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 个测试失败")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)


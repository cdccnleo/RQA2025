#!/usr/bin/env python3
"""
带内存泄漏保护的测试运行脚本
"""

import sys
import os
import gc
import psutil
import subprocess


def setup_memory_protection():
    """设置内存保护"""
    # 设置垃圾回收阈值
    gc.set_threshold(700, 10, 10)

    # 强制初始垃圾回收
    gc.collect()

    print(f"🔧 内存保护已启用")
    print(f"   当前内存: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")


def cleanup_between_tests():
    """测试间清理"""
    # 强制垃圾回收
    gc.collect()

    # 清理全局缓存
    cleanup_global_caches()

    print(f"🧹 测试间清理完成 - 内存: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")


def cleanup_global_caches():
    """清理全局缓存"""
    try:
        # 清理日志管理器缓存
        import sys
        if 'src.engine.logging.unified_logger' in sys.modules:
            module = sys.modules['src.engine.logging.unified_logger']
            if hasattr(module, '_engine_loggers'):
                module._engine_loggers.clear()

        # 清理增强日志管理器
        if 'src.infrastructure.logging.enhanced_log_manager' in sys.modules:
            module = sys.modules['src.infrastructure.logging.enhanced_log_manager']
            if hasattr(module, 'cleanup_global_log_manager'):
                module.cleanup_global_log_manager()

        # 清理配置管理器缓存
        if 'src.infrastructure.config.unified_manager' in sys.modules:
            module = sys.modules['src.infrastructure.config.unified_manager']
            # 清理配置缓存
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if hasattr(attr, 'clear') and callable(attr.clear):
                    try:
                        attr.clear()
                    except:
                        pass

    except Exception as e:
        print(f"⚠️  清理全局缓存时出错: {e}")


def run_test_with_protection(test_path: str, max_failures: int = 10) -> bool:
    """在内存保护下运行单个测试"""
    print(f"\n🚀 运行测试: {test_path}")

    # 测试前清理
    cleanup_between_tests()

    # 运行测试
    cmd = [
        sys.executable, "-m", "pytest", test_path,
        "--tb=short",
        "--maxfail", str(max_failures),
        "--disable-warnings"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=300, encoding='utf-8', errors='ignore')

        # 测试后清理
        cleanup_between_tests()

        if result.returncode == 0:
            print(f"✅ 测试通过: {test_path}")
            return True
        else:
            print(f"❌ 测试失败: {test_path}")
            print(f"错误输出: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ 测试超时: {test_path}")
        return False
    except Exception as e:
        print(f"💥 测试执行错误: {test_path} - {e}")
        return False


def run_infrastructure_tests():
    """运行基础设施层测试"""
    print("🏗️  开始基础设施层测试 (带内存保护)")

    # 设置内存保护
    setup_memory_protection()

    # 测试模块列表
    test_modules = [
        "tests/unit/infrastructure/config/",
        "tests/unit/infrastructure/logging/",
        "tests/unit/infrastructure/database/",
        "tests/unit/infrastructure/monitoring/",
        "tests/unit/infrastructure/error/",
        "tests/unit/infrastructure/resource/",
        "tests/unit/infrastructure/security/",
        "tests/unit/infrastructure/storage/",
        "tests/unit/infrastructure/scheduler/",
        "tests/unit/infrastructure/notification/"
    ]

    results = []

    for test_module in test_modules:
        if os.path.exists(test_module):
            success = run_test_with_protection(test_module)
            results.append((test_module, success))
        else:
            print(f"⚠️  测试模块不存在: {test_module}")
            results.append((test_module, False))

    # 输出结果统计
    print("\n📊 测试结果统计:")
    passed = sum(1 for _, success in results if success)
    total = len(results)

    for module, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {module}: {status}")

    print(f"\n🎯 总体结果: {passed}/{total} 模块通过")

    return passed == total


def main():
    """主函数"""
    if len(sys.argv) > 1:
        # 运行指定的测试
        test_path = sys.argv[1]
        success = run_test_with_protection(test_path)
        sys.exit(0 if success else 1)
    else:
        # 运行所有基础设施测试
        success = run_infrastructure_tests()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

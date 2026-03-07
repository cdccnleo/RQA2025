#!/usr/bin/env python3
"""
简化的基础设施层测试运行脚本
"""

import os
import sys
import subprocess


def run_single_test(test_path):
    """运行单个测试文件"""
    print(f"\n运行测试: {test_path}")
    print("-" * 50)

    if not os.path.exists(test_path):
        print(f"❌ 测试文件不存在: {test_path}")
        return False

    try:
        # 使用简单的命令运行测试
        cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
        result = subprocess.run(cmd, timeout=60)

        if result.returncode == 0:
            print(f"✅ 测试通过: {test_path}")
            return True
        else:
            print(f"❌ 测试失败: {test_path}")
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ 测试超时: {test_path}")
        return False
    except Exception as e:
        print(f"💥 运行错误: {test_path} - {e}")
        return False


def main():
    """主函数"""
    print("基础设施层测试验证")
    print("=" * 50)

    # 测试文件列表
    test_files = [
        "tests/unit/infrastructure/database/test_influxdb_error_handler.py",
        "tests/unit/infrastructure/m_logging/test_log_manager.py",
        "tests/unit/infrastructure/monitoring/test_application_monitor.py",
        "tests/unit/infrastructure/health/test_health_checker.py",
        "tests/unit/infrastructure/m_logging/test_log_sampler.py",
        "tests/unit/infrastructure/m_logging/test_log_aggregator.py",
        "tests/unit/infrastructure/m_logging/test_resource_manager.py",
        "tests/unit/infrastructure/m_logging/test_log_compressor.py",
        "tests/unit/infrastructure/m_logging/test_security_filter.py",
        "tests/unit/infrastructure/m_logging/test_quant_filter.py",
        "tests/unit/infrastructure/monitoring/test_backtest_monitor.py",
        "tests/unit/infrastructure/web/test_app_factory.py",
        "tests/unit/infrastructure/error/test_error_handler.py",
        "tests/unit/infrastructure/m_logging/test_log_metrics.py",
        "tests/unit/infrastructure/config/test_config_manager.py",
        "tests/unit/infrastructure/database/test_database_manager.py"
    ]

    passed = 0
    failed = 0
    total = 0

    for test_file in test_files:
        total += 1
        if run_single_test(test_file):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 50)
    print("测试结果总结")
    print("=" * 50)
    print(f"总测试数: {total}")
    print(f"通过: {passed}")
    print(f"失败: {failed}")

    if total > 0:
        success_rate = (passed / total) * 100
        print(f"成功率: {success_rate:.1f}%")

    if passed == total:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print(f"\n⚠️ 有 {failed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())

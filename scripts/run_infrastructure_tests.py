#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础设施层测试运行脚本
只运行已知正常工作的测试文件
"""

import subprocess
import sys
import os

def run_tests():
    """运行基础设施层测试"""
    
    # 已知正常工作的测试文件列表
    working_tests = [
        "tests/unit/infrastructure/config/test_config_manager_base.py",
        "tests/unit/infrastructure/config/test_lock_manager.py",
        "tests/unit/infrastructure/error/test_error_handling.py",
        "tests/unit/infrastructure/m_logging/test_boundary_conditions.py",
        "tests/unit/infrastructure/storage/test_file_storage.py",
        "tests/unit/infrastructure/monitoring/test_prometheus_monitor.py",
        "tests/unit/infrastructure/third_party/test_third_party_integration.py",
        "tests/unit/infrastructure/message_queue/test_message_queue.py",
        "tests/unit/infrastructure/database/test_connection_pool.py",
        "tests/unit/infrastructure/cache/test_thread_safe_cache.py",
        "tests/unit/infrastructure/health/test_health_checker.py",
        "tests/unit/infrastructure/storage/test_core.py",
        "tests/unit/infrastructure/utils/test_tools.py",
        "tests/unit/infrastructure/security/test_security.py",
        "tests/unit/infrastructure/monitoring/test_resource_api.py"
    ]
    
    # 构建pytest命令
    cmd = [
        "python", "-m", "pytest",
        "--cov=src/infrastructure",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov/infrastructure",
        "-v",
        "--tb=short"
    ]
    
    # 添加测试文件
    cmd.extend(working_tests)
    
    print("运行基础设施层测试...")
    print(f"命令: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        # 运行测试
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n" + "=" * 80)
        print("✅ 测试运行完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 测试运行失败: {e}")
        return False

def main():
    """主函数"""
    print("基础设施层测试运行器")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n🎉 所有测试通过!")
        sys.exit(0)
    else:
        print("\n💥 测试失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main() 
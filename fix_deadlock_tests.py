#!/usr/bin/env python3
"""
测试死锁检测和修复脚本
"""

import subprocess
import sys
import time
import signal
import os

def run_test_with_timeout(test_path, timeout=30):
    """运行测试并设置超时"""
    print(f"🔍 运行测试: {test_path}")
    print(f"⏰ 超时设置: {timeout}秒")
    
    try:
        # 运行测试命令
        cmd = [
            "python", "-m", "pytest",
            test_path,
            "-v", "-s",
            "--tb=short"
        ]
        
        # 设置超时
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        if result.returncode == 0:
            print("✅ 测试通过")
            return True
        else:
            print("❌ 测试失败")
            print("错误输出:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ 测试超时 ({timeout}秒)")
        return False
    except Exception as e:
        print(f"❌ 执行错误: {e}")
        return False

def check_specific_deadlock_test():
    """检查特定的死锁测试"""
    test_path = "tests/unit/infrastructure/error/test_error_handling_comprehensive.py::TestErrorHandlingComprehensive::test_trading_error_handler"
    
    print("🎯 检查死锁测试用例...")
    
    # 运行测试
    success = run_test_with_timeout(test_path, timeout=20)
    
    if success:
        print("✅ 死锁问题已解决")
    else:
        print("❌ 死锁问题仍然存在")
        print("💡 建议:")
        print("1. 检查测试用例中的多线程使用")
        print("2. 确保所有锁都有超时机制")
        print("3. 考虑使用mock替代真实的多线程测试")

def check_all_infrastructure_tests():
    """检查所有基础设施测试"""
    print("🔍 检查所有基础设施测试...")
    
    test_dir = "tests/unit/infrastructure/"
    success = run_test_with_timeout(test_dir, timeout=60)
    
    if success:
        print("✅ 所有基础设施测试通过")
    else:
        print("❌ 部分测试失败或超时")

if __name__ == "__main__":
    print("🚀 开始死锁检测和修复...")
    
    # 检查特定死锁测试
    check_specific_deadlock_test()
    
    print("\n" + "="*50)
    
    # 检查所有基础设施测试
    check_all_infrastructure_tests()
    
    print("\n✅ 死锁检测完成") 
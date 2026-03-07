#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存模块测试运行脚本

运行所有缓存模块的增强测试以验证覆盖率提升效果
"""

import subprocess
import sys
import os

def run_test_file(test_file):
    """运行单个测试文件"""
    print(f"Running {test_file}...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            f"tests/unit/infrastructure/cache/{test_file}", 
            "-v", "--tb=short"
        ], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        return False

def main():
    """主函数"""
    print("开始运行缓存模块增强测试...")
    
    # 测试文件列表
    test_files = [
        "test_cache_manager_coverage_enhancement.py",
        "test_multi_level_cache_enhancement.py",
        "test_cache_comprehensive_coverage.py"
    ]
    
    passed_tests = 0
    total_tests = len(test_files)
    
    # 运行每个测试文件
    for test_file in test_files:
        if run_test_file(test_file):
            passed_tests += 1
            print(f"✅ {test_file} 通过")
        else:
            print(f"❌ {test_file} 失败")
    
    # 输出总结
    print("\n" + "="*50)
    print(f"测试总结: {passed_tests}/{total_tests} 个测试文件通过")
    
    if passed_tests == total_tests:
        print("🎉 所有增强测试都通过了!")
        return 0
    else:
        print("⚠️  部分测试失败，请检查上述输出")
        return 1

if __name__ == "__main__":
    sys.exit(main())
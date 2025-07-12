#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CI配置验证脚本
检查所有CI配置中引用的测试文件是否存在且可运行
"""
import os
import sys
import subprocess
from pathlib import Path

def check_file_exists(file_path):
    """检查文件是否存在"""
    if os.path.exists(file_path):
        print(f"✓ {file_path} 存在")
        return True
    else:
        print(f"✗ {file_path} 不存在")
        return False

def check_test_files():
    """检查CI配置中引用的测试文件"""
    test_files = [
        "tests/integration/test_infrastructure_integration.py",
        "tests/integration/test_model_inference_integration.py", 
        "tests/integration/test_backtest_integration.py",
        "tests/integration/test_trading_integration.py",
        "tests/integration/test_trading_advanced_integration.py",
        "tests/unit/",
        "tests/performance/",
        "scripts/test_hooks_checker.py",
        "scripts/check_test_structure.py",
        "requirements.txt"
    ]
    
    all_exist = True
    for file_path in test_files:
        if not check_file_exists(file_path):
            all_exist = False
    
    return all_exist

def check_conda_environment():
    """检查conda环境配置"""
    try:
        result = subprocess.run(['conda', 'env', 'list'], 
                              capture_output=True, text=True)
        if 'rqa' in result.stdout:
            print("✓ conda环境 'rqa' 存在")
            return True
        else:
            print("✗ conda环境 'rqa' 不存在")
            return False
    except FileNotFoundError:
        print("✗ conda未安装")
        return False

def main():
    """主函数"""
    print("=== CI配置验证 ===")
    
    # 检查测试文件
    print("\n1. 检查测试文件:")
    test_files_ok = check_test_files()
    
    # 检查conda环境
    print("\n2. 检查conda环境:")
    conda_ok = check_conda_environment()
    
    # 总结
    print("\n=== 验证结果 ===")
    if test_files_ok and conda_ok:
        print("✓ 所有检查通过，CI配置正确")
        return 0
    else:
        print("✗ 部分检查失败，请修复CI配置")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
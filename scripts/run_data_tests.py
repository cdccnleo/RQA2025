#!/usr/bin/env python3
"""
数据层核心功能测试运行脚本
"""
import sys
import os
import subprocess
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_tests(test_type="all", parallel=True, coverage=True):
    """
    运行数据层测试
    
    Args:
        test_type: 测试类型 ("unit", "integration", "all")
        parallel: 是否并行运行
        coverage: 是否生成覆盖率报告
    """
    print("=" * 60)
    print("开始运行数据层核心功能测试")
    print("=" * 60)
    
    # 设置测试参数
    pytest_args = ["-v", "--tb=short"]
    
    if parallel:
        pytest_args.extend(["-n", "auto"])
    
    if coverage:
        pytest_args.extend([
            "--cov=src/data",
            "--cov=src/infrastructure",
            "--cov-report=html:htmlcov/data_coverage",
            "--cov-report=term-missing"
        ])
    
    # 根据测试类型选择测试文件
    if test_type == "unit":
        test_paths = [
            "tests/unit/data/test_data_manager.py",
            "tests/unit/data/test_base_loader.py", 
            "tests/unit/data/test_validator.py"
        ]
    elif test_type == "integration":
        test_paths = [
            "tests/integration/test_data_infrastructure_integration.py"
        ]
    else:  # all
        test_paths = [
            "tests/unit/data/",
            "tests/integration/test_data_infrastructure_integration.py"
        ]
    
    # 运行测试
    start_time = time.time()
    
    for test_path in test_paths:
        print(f"\n运行测试: {test_path}")
        print("-" * 40)
        
        cmd = ["python", "-m", "pytest"] + pytest_args + [test_path]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 输出测试结果
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("错误输出:")
                print(result.stderr)
            
            if result.returncode == 0:
                print(f"✅ {test_path} 测试通过")
            else:
                print(f"❌ {test_path} 测试失败")
                return False
                
        except Exception as e:
            print(f"❌ 运行测试时出错: {e}")
            return False
    
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "=" * 60)
    print(f"测试完成! 总耗时: {duration:.2f}秒")
    print("=" * 60)
    
    return True

def run_performance_tests():
    """运行性能测试"""
    print("\n" + "=" * 60)
    print("运行性能测试")
    print("=" * 60)
    
    pytest_args = [
        "-v",
        "--tb=short",
        "-m", "performance",
        "tests/unit/data/test_data_manager.py",
        "tests/unit/data/test_base_loader.py"
    ]
    
    cmd = ["python", "-m", "pytest"] + pytest_args
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ 性能测试通过")
        else:
            print("❌ 性能测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 运行性能测试时出错: {e}")
        return False
    
    return True

def run_error_handling_tests():
    """运行错误处理测试"""
    print("\n" + "=" * 60)
    print("运行错误处理测试")
    print("=" * 60)
    
    pytest_args = [
        "-v",
        "--tb=short",
        "-m", "error_handling",
        "tests/unit/data/test_data_manager.py",
        "tests/unit/data/test_base_loader.py",
        "tests/unit/data/test_validator.py"
    ]
    
    cmd = ["python", "-m", "pytest"] + pytest_args
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ 错误处理测试通过")
        else:
            print("❌ 错误处理测试失败")
            return False
            
    except Exception as e:
        print(f"❌ 运行错误处理测试时出错: {e}")
        return False
    
    return True

def generate_test_report():
    """生成测试报告"""
    print("\n" + "=" * 60)
    print("生成测试报告")
    print("=" * 60)
    
    # 创建报告目录
    report_dir = Path("test_reports")
    report_dir.mkdir(exist_ok=True)
    
    # 生成HTML报告
    cmd = [
        "python", "-m", "pytest",
        "--html=test_reports/data_test_report.html",
        "--self-contained-html",
        "tests/unit/data/",
        "tests/integration/test_data_infrastructure_integration.py"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 测试报告生成成功")
            print(f"报告位置: {report_dir.absolute()}")
        else:
            print("❌ 测试报告生成失败")
            if result.stderr:
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 生成测试报告时出错: {e}")
        return False
    
    return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="数据层核心功能测试运行器")
    parser.add_argument(
        "--test-type", 
        choices=["unit", "integration", "all"], 
        default="all",
        help="测试类型"
    )
    parser.add_argument(
        "--no-parallel", 
        action="store_true",
        help="禁用并行测试"
    )
    parser.add_argument(
        "--no-coverage", 
        action="store_true",
        help="禁用覆盖率报告"
    )
    parser.add_argument(
        "--performance-only", 
        action="store_true",
        help="仅运行性能测试"
    )
    parser.add_argument(
        "--error-handling-only", 
        action="store_true",
        help="仅运行错误处理测试"
    )
    parser.add_argument(
        "--generate-report", 
        action="store_true",
        help="生成测试报告"
    )
    
    args = parser.parse_args()
    
    # 检查是否在正确的环境中
    try:
        import pandas as pd
        import pytest
        print("✅ 环境检查通过")
    except ImportError as e:
        print(f"❌ 环境检查失败: {e}")
        print("请确保在conda rqa环境中运行")
        return False
    
    # 根据参数运行相应的测试
    if args.performance_only:
        success = run_performance_tests()
    elif args.error_handling_only:
        success = run_error_handling_tests()
    else:
        success = run_tests(
            test_type=args.test_type,
            parallel=not args.no_parallel,
            coverage=not args.no_coverage
        )
    
    # 生成报告
    if args.generate_report and success:
        generate_test_report()
    
    if success:
        print("\n🎉 所有测试通过!")
        return True
    else:
        print("\n💥 测试失败!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
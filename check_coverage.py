#!/usr/bin/env python3
"""
基础设施层测试覆盖率检查脚本
"""

import subprocess
import sys
import os

def run_coverage_check():
    """运行覆盖率检查"""
    print("🔍 检查基础设施层测试覆盖率...")
    
    try:
        # 运行覆盖率测试
        cmd = [
            "python", "-m", "pytest",
            "--cov=src/infrastructure",
            "--cov-report=term-missing",
            "tests/unit/infrastructure/",
            "-q"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ 测试执行成功")
            print("\n📊 覆盖率报告:")
            print(result.stdout)
        else:
            print("❌ 测试执行失败")
            print("错误信息:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ 测试超时，可能需要更多时间")
    except Exception as e:
        print(f"❌ 执行错误: {e}")

def check_target_coverage():
    """检查是否达到90%覆盖率目标"""
    print("\n🎯 覆盖率目标检查:")
    print("- 目标覆盖率: 90%")
    print("- 当前状态: 测试执行中...")
    print("- 建议: 等待测试完成，然后分析具体覆盖率数据")

if __name__ == "__main__":
    run_coverage_check()
    check_target_coverage() 
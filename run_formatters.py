#!/usr/bin/env python3
"""
Phase 3: 代码格式化和质量检查脚本
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"正在执行: {description}")
    print(f"命令: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=300
        )
        
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        print(f"返回码: {result.returncode}")
        return result.returncode == 0
        
    except Exception as e:
        print(f"错误: {e}")
        return False


def main():
    """主函数"""
    print("="*60)
    print("RQA2025 Phase 3: 代码格式化和质量检查")
    print("="*60)
    
    # 1. 运行Black格式化
    print("\n步骤 1: 运行Black代码格式化...")
    black_cmd = [
        sys.executable, "-m", "black",
        "src",
        "--line-length", "100",
        "--target-version", "py39",
        "--extend-exclude", "backups|production_simulation|docs|reports"
    ]
    run_command(black_cmd, "Black代码格式化")
    
    # 2. 运行isort排序导入
    print("\n步骤 2: 运行isort导入排序...")
    isort_cmd = [
        sys.executable, "-m", "isort",
        "src",
        "--profile", "black",
        "--line-length", "100",
        "--skip", "backups",
        "--skip", "production_simulation",
        "--skip", "docs",
        "--skip", "reports"
    ]
    run_command(isort_cmd, "isort导入排序")
    
    # 3. 运行Flake8检查
    print("\n步骤 3: 运行Flake8代码检查...")
    flake8_cmd = [
        sys.executable, "-m", "flake8",
        "src",
        "--max-line-length=100",
        "--extend-ignore=E203,W503",
        "--exclude=backups,production_simulation,docs,reports,__pycache__,.git",
        "--count",
        "--statistics",
        "--output-file=flake8_phase3_report.txt"
    ]
    run_command(flake8_cmd, "Flake8代码检查")
    
    # 4. 显示Flake8统计
    print("\n步骤 4: 显示Flake8统计信息...")
    flake8_stats_cmd = [
        sys.executable, "-m", "flake8",
        "src",
        "--max-line-length=100",
        "--extend-ignore=E203,W503",
        "--exclude=backups,production_simulation,docs,reports,__pycache__,.git",
        "--count",
        "--statistics"
    ]
    run_command(flake8_stats_cmd, "Flake8统计信息")
    
    print("\n" + "="*60)
    print("Phase 3 格式化完成!")
    print("="*60)
    print("\n查看详细报告: flake8_phase3_report.txt")


if __name__ == "__main__":
    main()

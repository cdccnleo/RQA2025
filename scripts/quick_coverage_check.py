#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速覆盖率检查脚本
只运行测试并输出简要统计，不生成详细报告
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
TEST_LOGS_DIR = PROJECT_ROOT / "test_logs"
TEST_LOGS_DIR.mkdir(exist_ok=True)

QUICK_CHECK_LOG = TEST_LOGS_DIR / "quick_coverage_check_latest.log"


def quick_check():
    """快速检查测试通过率和总覆盖率"""
    print("执行快速覆盖率检查...")
    
    cmd = [
        "conda", "run", "-n", "rqa",
        "pytest", "tests/unit/features",
        "-n", "auto",
        "--cov=src.features",
        "--cov-report=term-missing",
        "--tb=line",
        "-q"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        output = result.stdout + result.stderr
        
        # 提取关键信息
        lines = output.split('\n')
        test_stats = []
        total_coverage = None
        
        for line in lines:
            if ("passed" in line or "failed" in line) and "warnings" in line:
                test_stats.append(line)
            if "TOTAL" in line and "%" in line:
                total_coverage = line
        
        # 写入日志
        with open(QUICK_CHECK_LOG, 'w', encoding='utf-8') as f:
            f.write(f"快速覆盖率检查 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            f.write(output)
        
        # 打印摘要
        print("\n" + "="*80)
        print("快速检查结果:")
        print("="*80)
        for stat in test_stats:
            print(f"  {stat}")
        if total_coverage:
            print(f"\n  {total_coverage}")
        print("="*80)
        print(f"\n详细日志已保存到: {QUICK_CHECK_LOG}\n")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"执行失败: {e}")
        return False


if __name__ == "__main__":
    success = quick_check()
    sys.exit(0 if success else 1)

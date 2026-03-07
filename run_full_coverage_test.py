#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行完整覆盖率测试并生成详细报告
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

def run_coverage_test():
    """运行覆盖率测试"""
    print("="*80)
    print("🚀 运行完整覆盖率测试")
    print("="*80)
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    reports_dir = Path("reports/coverage_full")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_dir = reports_dir / f"html_{timestamp}"
    json_file = reports_dir / f"coverage_{timestamp}.json"
    
    # 构建pytest命令
    cmd = [
        "pytest",
        "-v",
        "--tb=short",
        "--cov=src",
        f"--cov-report=html:{html_dir}",
        f"--cov-report=json:{json_file}",
        "--cov-report=term-missing",
        "-n", "auto",  # 并行执行
        "--maxfail=50",  # 最多50个失败后停止
        "tests/"
    ]
    
    print("🚀 执行命令:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30分钟超时
        )
        
        output = result.stdout + result.stderr
        
        # 解析结果
        print("\n" + "="*80)
        print("📊 测试结果")
        print("="*80 + "\n")
        
        # 提取测试统计
        for line in output.split('\n'):
            if any(keyword in line.lower() for keyword in ['passed', 'failed', 'error', 'skipped', 'coverage']):
                print(line)
        
        # 读取覆盖率JSON
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                cov_data = json.load(f)
                total_cov = cov_data.get('totals', {}).get('percent_covered', 0)
                
                print(f"\n📊 总体覆盖率: {total_cov:.2f}%")
                if total_cov >= 80:
                    print("✅ 覆盖率目标达成！(≥80%)")
                else:
                    print(f"❌ 覆盖率未达标，差距: {80 - total_cov:.2f}%")
        
        print(f"\n📁 详细报告: {html_dir}/index.html")
        print(f"📄 JSON报告: {json_file}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ 测试超时 (>30分钟)")
        return False
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False

if __name__ == "__main__":
    success = run_coverage_test()
    sys.exit(0 if success else 1)


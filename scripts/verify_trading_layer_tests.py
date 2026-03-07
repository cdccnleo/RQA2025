#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证Trading层测试通过率
质量优先原则：必须先确保100%测试通过率
"""

import subprocess
import sys
import json
from datetime import datetime
from pathlib import Path

def run_trading_tests():
    """运行Trading层测试并收集结果"""
    print("=" * 60)
    print("  Trading层质量验证 - 测试通过率检查")
    print("=" * 60)
    print()
    
    # 运行测试
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/unit/trading/",
        "-n", "auto",
        "-q",
        "--tb=line",
        "--maxfail=20",
        "-v"
    ]
    
    print("运行测试命令:", " ".join(cmd))
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    
    # 输出结果
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # 分析结果
    output_lines = result.stdout.split('\n')
    
    stats = {
        'passed': 0,
        'failed': 0,
        'error': 0,
        'skipped': 0,
        'total': 0,
        'timestamp': datetime.now().isoformat()
    }
    
    for line in output_lines:
        if 'passed' in line.lower() and 'failed' in line.lower():
            # 解析结果行，例如: "100 passed, 5 failed, 2 error, 10 skipped in 10.5s"
            parts = line.split()
            for i, part in enumerate(parts):
                if part == 'passed' and i > 0:
                    try:
                        stats['passed'] = int(parts[i-1])
                    except:
                        pass
                elif part == 'failed' and i > 0:
                    try:
                        stats['failed'] = int(parts[i-1])
                    except:
                        pass
                elif part == 'error' and i > 0:
                    try:
                        stats['error'] = int(parts[i-1])
                    except:
                        pass
                elif part == 'skipped' and i > 0:
                    try:
                        stats['skipped'] = int(parts[i-1])
                    except:
                        pass
    
    stats['total'] = stats['passed'] + stats['failed'] + stats['error'] + stats['skipped']
    
    # 计算通过率
    if stats['total'] > 0:
        pass_rate = (stats['passed'] / stats['total']) * 100
        stats['pass_rate'] = pass_rate
    else:
        stats['pass_rate'] = 0
    
    # 保存结果
    output_dir = Path("test_logs")
    output_dir.mkdir(exist_ok=True)
    
    result_file = output_dir / "trading_layer_test_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 打印统计
    print()
    print("=" * 60)
    print("  测试统计结果")
    print("=" * 60)
    print(f"总测试数: {stats['total']}")
    print(f"通过: {stats['passed']}")
    print(f"失败: {stats['failed']}")
    print(f"错误: {stats['error']}")
    print(f"跳过: {stats['skipped']}")
    print(f"通过率: {stats['pass_rate']:.2f}%")
    print()
    
    # 质量评估
    if stats['failed'] == 0 and stats['error'] == 0:
        print("✅ 测试通过率: 100% (质量优先原则达成)")
        print("✅ 状态: 可以继续推进覆盖率验证")
    else:
        print(f"❌ 测试通过率: {stats['pass_rate']:.2f}% (未达到100%)")
        print(f"❌ 失败测试: {stats['failed']}个")
        print(f"❌ 错误测试: {stats['error']}个")
        print("⚠️  必须先修复失败的测试才能继续")
    
    print()
    print(f"结果已保存到: {result_file}")
    print()
    
    return stats, result.returncode

if __name__ == "__main__":
    stats, exit_code = run_trading_tests()
    sys.exit(exit_code)


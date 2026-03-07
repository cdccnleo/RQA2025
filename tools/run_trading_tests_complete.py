#!/usr/bin/env python3
"""
运行Trading层所有测试并生成完整统计报告
"""
import subprocess
import re
import json
from pathlib import Path

def run_trading_tests():
    """运行Trading层测试并收集结果"""
    print("=" * 60)
    print("  Trading层测试执行 - 质量优先原则")
    print("=" * 60)
    print()
    
    # 运行pytest获取详细结果
    result = subprocess.run(
        [
            "pytest",
            "tests/unit/trading/",
            "-v",
            "--tb=no",
            "-q"
        ],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    # 打印输出
    output = result.stdout + result.stderr
    
    # 解析输出获取统计信息
    # 查找类似 "123 passed, 5 failed, 2 skipped in 45.67s" 的模式
    pattern = r'(\d+)\s+passed.*?(\d+)\s+failed.*?(\d+)\s+skipped|(\d+)\s+passed.*?(\d+)\s+failed|(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+error'
    matches = re.findall(pattern, output, re.IGNORECASE)
    
    total = 0
    passed = 0
    failed = 0
    errors = 0
    skipped = 0
    
    # 解析结果
    lines = output.split('\n')
    for line in lines:
        # 查找统计行
        if 'passed' in line.lower() or 'failed' in line.lower() or 'error' in line.lower():
            # 提取数字
            nums = re.findall(r'\d+', line)
            if 'passed' in line.lower():
                try:
                    idx = line.lower().find('passed')
                    if idx > 0:
                        passed = int(re.findall(r'\d+', line[:idx])[-1])
                        total = max(total, passed)
                except:
                    pass
            if 'failed' in line.lower():
                try:
                    idx = line.lower().find('failed')
                    if idx > 0:
                        failed = int(re.findall(r'\d+', line[:idx])[-1])
                        total = max(total, failed)
                except:
                    pass
            if 'error' in line.lower() and 'ERROR' in line:
                try:
                    errors = len([l for l in lines if 'ERROR' in l])
                except:
                    pass
            if 'skipped' in line.lower():
                try:
                    idx = line.lower().find('skipped')
                    if idx > 0:
                        skipped = int(re.findall(r'\d+', line[:idx])[-1])
                except:
                    pass
    
    # 从输出行中提取失败的测试
    failed_tests = []
    error_tests = []
    for line in lines:
        if 'FAILED' in line or 'FAIL' in line:
            failed_tests.append(line.strip())
        elif 'ERROR' in line and 'collecting' not in line.lower():
            error_tests.append(line.strip())
    
    total = passed + failed + errors if total == 0 else total
        
    print("\n" + "=" * 60)
    print("  📊 Trading层测试结果统计")
    print("=" * 60)
    print(f"  总测试数: {total}")
    print(f"  ✅ 通过: {passed}")
    print(f"  ❌ 失败: {failed}")
    print(f"  ⚠️  错误: {errors}")
    print(f"  ⏭️  跳过: {skipped}")
    if total > 0:
        pass_rate = (passed / total) * 100
        print(f"  通过率: {pass_rate:.2f}%")
        print()
        if pass_rate == 100:
            print("  ✅ 100% 通过率达成！")
        else:
            print(f"  ⚠️  需要修复 {failed + errors} 个失败的测试")
    
    # 列出失败的测试（前20个）
    if failed_tests or error_tests:
        print("\n" + "=" * 60)
        print("  ❌ 失败的测试列表（前20个）")
        print("=" * 60)
        for test in (failed_tests + error_tests)[:20]:
            print(f"  - {test}")
    
    # 打印输出最后部分
    print("\n" + "=" * 60)
    print("  📋 测试输出（最后50行）")
    print("=" * 60)
    for line in lines[-50:]:
        if line.strip():
            print(line)
    
    return {
        'total': total,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'skipped': skipped,
        'pass_rate': (passed / total * 100) if total > 0 else 0
    }

if __name__ == "__main__":
    run_trading_tests()


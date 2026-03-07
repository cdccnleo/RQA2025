#!/usr/bin/env python3
"""
获取Trading层测试完整统计
"""
import subprocess
import re
import sys

def get_test_summary():
    """运行测试并提取统计信息"""
    print("=" * 60)
    print("  Trading层测试执行 - 质量优先原则")
    print("=" * 60)
    print()
    
    # 运行pytest获取统计
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
    
    # 解析输出
    output = result.stdout + result.stderr
    lines = output.split('\n')
    
    # 查找统计行
    summary_line = None
    for line in reversed(lines):
        if 'passed' in line.lower() and ('failed' in line.lower() or 'error' in line.lower()):
            summary_line = line
            break
    
    print("测试输出（最后50行）:")
    print("-" * 60)
    for line in lines[-50:]:
        if line.strip():
            print(line)
    
    print()
    print("=" * 60)
    if summary_line:
        print("  📊 测试统计")
        print("=" * 60)
        print(f"  {summary_line}")
        
        # 提取数字
        passed_match = re.search(r'(\d+)\s+passed', summary_line, re.IGNORECASE)
        failed_match = re.search(r'(\d+)\s+failed', summary_line, re.IGNORECASE)
        error_match = re.search(r'(\d+)\s+error', summary_line, re.IGNORECASE)
        skipped_match = re.search(r'(\d+)\s+skipped', summary_line, re.IGNORECASE)
        
        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        errors = int(error_match.group(1)) if error_match else 0
        skipped = int(skipped_match.group(1)) if skipped_match else 0
        total = passed + failed + errors + skipped
        
        print()
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
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'pass_rate': (passed / total * 100) if total > 0 else 0
        }
    else:
        print("  ⚠️  无法解析测试统计")
        return None

if __name__ == "__main__":
    summary = get_test_summary()
    sys.exit(0 if summary and summary['pass_rate'] == 100 else 1)


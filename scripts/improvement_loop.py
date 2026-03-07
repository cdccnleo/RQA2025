#!/usr/bin/env python3
"""
持续改进循环脚本
"""

import subprocess
import time
import json


def run_improvement_loop():
    """运行持续改进循环"""
    print("🔄 开始持续改进循环...")

    cycle_results = {}

    # 1. 运行代码质量检查
    print("  📋 步骤1: 代码质量检查")
    try:
        result = subprocess.run(['python', 'scripts/code_quality_check.py'],
                                capture_output=True, text=True, timeout=300)
        cycle_results['quality_check'] = {
            'success': result.returncode == 0,
            'output': result.stdout,
            'errors': result.stderr
        }
    except Exception as e:
        cycle_results['quality_check'] = {'success': False, 'error': str(e)}

    # 2. 运行自动化修复
    print("  🔧 步骤2: 自动化修复")
    try:
        result = subprocess.run(['python', 'scripts/automated_fixes.py'],
                                capture_output=True, text=True, timeout=300)
        cycle_results['automated_fixes'] = {
            'success': result.returncode == 0,
            'output': result.stdout,
            'errors': result.stderr
        }
    except Exception as e:
        cycle_results['automated_fixes'] = {'success': False, 'error': str(e)}

    # 3. 性能监控
    print("  📊 步骤3: 性能监控")
    try:
        result = subprocess.run(['python', 'scripts/performance_monitor.py'],
                                capture_output=True, text=True, timeout=60)
        cycle_results['performance_monitor'] = {
            'success': result.returncode == 0,
            'output': result.stdout,
            'errors': result.stderr
        }
    except Exception as e:
        cycle_results['performance_monitor'] = {'success': False, 'error': str(e)}

    # 保存循环结果
    cycle_data = {
        'timestamp': time.time(),
        'cycle_results': cycle_results,
        'summary': {
            'total_steps': len(cycle_results),
            'successful_steps': sum(1 for r in cycle_results.values() if r.get('success', False)),
            'failed_steps': sum(1 for r in cycle_results.values() if not r.get('success', False))
        }
    }

    with open('improvement_cycle_results.json', 'w', encoding='utf-8') as f:
        json.dump(cycle_data, f, indent=2, ensure_ascii=False)

    print("✅ 持续改进循环完成")
    print(
        f"   成功步骤: {cycle_data['summary']['successful_steps']}/{cycle_data['summary']['total_steps']}")

    return cycle_data


if __name__ == "__main__":
    run_improvement_loop()

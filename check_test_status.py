#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查当前测试状态脚本
"""

import subprocess
import sys
import os

def check_module_status(module_name):
    """检查单个模块的测试状态"""
    try:
        cmd = [sys.executable, '-m', 'pytest', f'tests/unit/{module_name}/', '-x', '-q', '--tb=no']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            output = result.stdout + result.stderr
            passed = output.count('PASSED')
            failed = output.count('FAILED')
            total = passed + failed
            success_rate = (passed / total * 100) if total > 0 else 0
            return {'status': 'success', 'passed': passed, 'failed': failed, 'rate': success_rate}
        else:
            return {'status': 'failed', 'error': result.returncode}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def main():
    print('🔍 检查当前测试状态...')

    modules = ['ml', 'infrastructure', 'features', 'strategy', 'trading', 'data', 'core']
    results = {}

    for module in modules:
        if os.path.exists(f'tests/unit/{module}'):
            print(f'检查 {module} 模块...')
            results[module] = check_module_status(module)
        else:
            results[module] = {'status': 'not_found'}

    print('\n📊 当前测试状态汇总:')
    issues = []

    for module, result in results.items():
        if result['status'] == 'success':
            rate = result['rate']
            status = '✅' if rate >= 95 else '⚠️' if rate >= 80 else '❌'
            print(f'{status} {module}: {rate:.1f}% ({result["passed"]}/{result["passed"]+result["failed"]})')
            if rate < 95:
                issues.append(module)
        elif result['status'] == 'not_found':
            print(f'⚠️ {module}: 测试目录不存在')
        else:
            print(f'❌ {module}: 测试失败')
            issues.append(module)

    print(f'\n🎯 需要改进的模块: {issues}')
    print(f'📈 总体状态: {"✅ 优秀" if not issues else "⚠️ 需改进"}')

    return issues

if __name__ == "__main__":
    issues = main()
    sys.exit(0 if not issues else 1)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断未达标模块的测试问题

检查:
1. 测试收集情况
2. 测试执行情况
3. 失败原因分析
"""

import subprocess
import json
import re
from pathlib import Path
from datetime import datetime

# 未达标的11个模块（<60%覆盖率）
FAILING_MODULES = [
    ('health', 18),
    ('distributed', 26),
    ('versioning', 25),
    ('monitoring', 26),
    ('security', 27),
    ('utils', 29),
    ('config', 30),
    ('logging', 31),
    ('cache', 37),
    ('api', 42),
    ('resource', 52),
]


def diagnose_module(module: str) -> dict:
    """诊断单个模块的问题"""
    
    print(f"\n{'='*60}")
    print(f"诊断模块: {module}")
    print(f"{'='*60}")
    
    test_path = f"tests/unit/infrastructure/{module}"
    
    result = {
        'module': module,
        'can_collect': False,
        'test_count': 0,
        'errors': [],
        'diagnosis': ''
    }
    
    # 1. 测试收集
    cmd_collect = [
        'python', '-m', 'pytest',
        test_path,
        '--collect-only',
        '-q'
    ]
    
    try:
        proc = subprocess.run(
            cmd_collect,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path.cwd()
        )
        
        # 解析收集数量
        match = re.search(r'(\d+) test', proc.stdout)
        if match:
            test_count = int(match.group(1))
            result['can_collect'] = True
            result['test_count'] = test_count
            print(f"✅ 可以收集测试: {test_count}个")
        else:
            print(f"❌ 无法收集测试")
            result['diagnosis'] = 'collection_failed'
            # 查找错误
            errors = re.findall(r'(ERROR|NameError|ImportError|TypeError).*', proc.stdout + proc.stderr)
            result['errors'] = errors[:5]  # 最多5个错误
            
    except Exception as e:
        print(f"❌ 收集异常: {e}")
        result['diagnosis'] = f'exception: {e}'
    
    # 2. 如果能收集，尝试运行少量测试
    if result['can_collect'] and result['test_count'] > 0:
        cmd_run = [
            'python', '-m', 'pytest',
            test_path,
            '-q',
            '--maxfail=3',
            '--tb=no',
            '-x'  # 第一个失败就停止
        ]
        
        try:
            proc_run = subprocess.run(
                cmd_run,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=Path.cwd()
            )
            
            # 解析执行结果
            passed_match = re.search(r'(\d+) passed', proc_run.stdout)
            failed_match = re.search(r'(\d+) failed', proc_run.stdout)
            error_match = re.search(r'(\d+) error', proc_run.stdout)
            
            result['passed'] = int(passed_match.group(1)) if passed_match else 0
            result['failed'] = int(failed_match.group(1)) if failed_match else 0
            result['errored'] = int(error_match.group(1)) if error_match else 0
            
            print(f"   通过: {result['passed']}, 失败: {result['failed']}, 错误: {result['errored']}")
            
            # 提取第一个失败的错误信息
            if result['failed'] > 0 or result['errored'] > 0:
                error_lines = re.findall(r'(FAILED|ERROR).*', proc_run.stdout)
                if error_lines:
                    result['first_error'] = error_lines[0]
                    print(f"   第一个错误: {error_lines[0][:100]}")
                    
        except subprocess.TimeoutExpired:
            print(f"⏰ 执行超时")
            result['diagnosis'] = 'execution_timeout'
        except Exception as e:
            print(f"❌ 执行异常: {e}")
    
    return result


def main():
    """主函数"""
    
    print("="*70)
    print("未达标模块诊断分析")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"诊断模块数: {len(FAILING_MODULES)}")
    print()
    
    results = []
    
    for module, current_cov in FAILING_MODULES:
        result = diagnose_module(module)
        result['current_coverage'] = current_cov
        results.append(result)
    
    # 保存结果
    output_file = Path('test_logs/module_diagnosis_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'total_modules': len(results),
            'modules': results
        }, f, indent=2, ensure_ascii=False)
    
    # 生成摘要
    print(f"\n{'='*70}")
    print("诊断摘要")
    print(f"{'='*70}")
    
    can_collect = sum(1 for r in results if r['can_collect'])
    has_passed = sum(1 for r in results if r.get('passed', 0) > 0)
    
    print(f"可以收集测试的模块: {can_collect}/{len(results)}")
    print(f"有测试通过的模块: {has_passed}/{len(results)}")
    print()
    
    print(f"诊断结果已保存至: {output_file}")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0


if __name__ == '__main__':
    exit(main())


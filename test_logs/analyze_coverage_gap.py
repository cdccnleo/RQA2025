#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""深入分析覆盖率差异的诊断脚本"""

import subprocess
import json
import sys
from pathlib import Path

def run_coverage_comparison():
    """对比单独测试和完整测试的差异"""
    
    results = {}
    
    # 测试模块列表
    modules = ['api', 'logging', 'cache', 'config', 'resource', 'utils']
    
    print("="*80)
    print("覆盖率差异深度分析")
    print("="*80)
    print()
    
    for module in modules:
        print(f"\n{'='*80}")
        print(f"分析模块: {module}")
        print(f"{'='*80}")
        
        # 1. 单独模块测试（详细）
        print(f"\n[1] 单独模块测试（包含文件列表）")
        cmd = [
            'python', '-m', 'pytest',
            f'tests/unit/infrastructure/{module}',
            f'--cov=src/infrastructure/{module}',
            '--cov-report=term-missing',
            '-q', '--tb=no', '--maxfail=1'
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=60
            )
            
            # 提取覆盖率信息
            output = result.stdout + result.stderr
            
            # 查找TOTAL行
            for line in output.split('\n'):
                if 'TOTAL' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            stmts = int(parts[-3])
                            miss = int(parts[-2])
                            cov = parts[-1].replace('%', '')
                            print(f"   语句数: {stmts}")
                            print(f"   未覆盖: {miss}")
                            print(f"   覆盖率: {cov}%")
                            
                            results[module] = {
                                'isolated': {
                                    'statements': stmts,
                                    'missed': miss,
                                    'coverage': cov
                                }
                            }
                        except:
                            pass
                    break
            
            # 显示0%覆盖的文件
            print(f"\n   0%覆盖的文件:")
            zero_files = []
            for line in output.split('\n'):
                if 'src\\infrastructure\\' + module in line and '0%' in line:
                    # 提取文件名
                    parts = line.split()
                    if parts:
                        filepath = parts[0]
                        zero_files.append(filepath)
                        print(f"      - {filepath}")
            
            if not zero_files:
                print(f"      (无0%文件)")
            
            results[module]['zero_files_isolated'] = zero_files
            
        except Exception as e:
            print(f"   错误: {e}")
        
        # 2. 检查测试文件数量和内容
        print(f"\n[2] 测试文件分析")
        test_dir = Path(f'tests/unit/infrastructure/{module}')
        if test_dir.exists():
            test_files = list(test_dir.glob('test_*.py'))
            print(f"   测试文件数: {len(test_files)}")
            
            # 统计测试用例数
            total_tests = 0
            for test_file in test_files[:5]:  # 显示前5个
                try:
                    with open(test_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        test_count = content.count('def test_')
                        print(f"   - {test_file.name}: {test_count}个测试")
                        total_tests += test_count
                except:
                    pass
            
            print(f"   总测试用例（估算）: ~{total_tests}+")
        else:
            print(f"   测试目录不存在")
        
        # 3. 检查源文件数量
        print(f"\n[3] 源文件分析")
        src_dir = Path(f'src/infrastructure/{module}')
        if src_dir.exists():
            py_files = list(src_dir.rglob('*.py'))
            py_files = [f for f in py_files if '__pycache__' not in str(f)]
            print(f"   源文件总数: {len(py_files)}")
            
            # 统计代码行数
            total_lines = 0
            for py_file in py_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = len([l for l in f.readlines() if l.strip() and not l.strip().startswith('#')])
                        total_lines += lines
                except:
                    pass
            
            print(f"   总代码行数（估算）: ~{total_lines}")
        
        print(f"\n[4] 诊断结论")
        if module in results:
            iso_data = results[module].get('isolated', {})
            stmts = iso_data.get('statements', 0)
            zero_count = len(results[module].get('zero_files_isolated', []))
            
            if zero_count > 0:
                print(f"   ⚠️ 仍有{zero_count}个文件0%覆盖")
                print(f"   💡 建议: 为这些文件添加针对性测试")
            
            if stmts > 0:
                test_count = total_tests if 'total_tests' in locals() else 0
                if test_count > 0:
                    ratio = stmts / test_count if test_count > 0 else 0
                    print(f"   测试效率: {ratio:.1f}语句/测试")
                    if ratio < 5:
                        print(f"   ⚠️ 测试效率较低，可能是基础测试过多")
    
    print("\n" + "="*80)
    print("总体分析")
    print("="*80)
    
    # 保存结果
    with open('test_logs/coverage_gap_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n详细结果已保存至: test_logs/coverage_gap_analysis.json")
    
    return results

if __name__ == '__main__':
    run_coverage_comparison()


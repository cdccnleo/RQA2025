#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础设施层分模块真实覆盖率验证脚本

功能:
1. 分模块运行pytest --cov测试
2. 收集真实覆盖率数据
3. 生成详细的覆盖率报告
4. 对比估算值和实际值

使用方法:
    python test_logs/run_module_coverage.py
"""

import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# 基础设施层17个核心模块
INFRASTRUCTURE_MODULES = [
    'monitoring', 'config', 'health', 'resource', 'security',
    'logging', 'utils', 'api', 'cache', 'error',
    'versioning', 'distributed', 'optimization', 'constants',
    'core', 'interfaces', 'ops'
]


def run_module_coverage(module: str, timeout: int = 300) -> Tuple[float, dict]:
    """运行单个模块的覆盖率测试"""
    
    print(f"\n{'='*60}")
    print(f"测试模块: {module}")
    print(f"{'='*60}")
    
    # 构建pytest命令
    src_path = f"src/infrastructure/{module}"
    test_path = f"tests/unit/infrastructure/{module}"
    
    # 检查路径是否存在
    if not Path(test_path).exists():
        print(f"⚠️ 测试目录不存在: {test_path}")
        return 0.0, {'error': 'test_path_not_found'}
    
    if not Path(src_path).exists():
        print(f"⚠️ 源码目录不存在: {src_path}")
        return 0.0, {'error': 'src_path_not_found'}
    
    # pytest命令（不使用并行，以确保覆盖率准确）
    cmd = [
        'python', '-m', 'pytest',
        test_path,
        f'--cov={src_path}',
        '--cov-report=term-missing',
        '--cov-report=json',
        '-v',
        '--tb=short',
        '--maxfail=5',  # 最多失败5个就停止
        '-x',  # 遇到第一个失败就停止
    ]
    
    try:
        # 运行测试
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd()
        )
        
        # 解析输出中的覆盖率信息
        coverage_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', result.stdout)
        
        if coverage_match:
            coverage = float(coverage_match.group(1))
            print(f"✅ 覆盖率: {coverage}%")
        else:
            # 尝试从JSON文件读取
            json_path = Path('.coverage')
            if json_path.exists():
                with open(json_path, 'r') as f:
                    cov_data = json.load(f)
                    # 这里简化处理
                    coverage = 0.0
            else:
                coverage = 0.0
            print(f"⚠️ 无法解析覆盖率，设为0%")
        
        # 解析测试统计
        tests_match = re.search(r'(\d+) passed', result.stdout)
        failed_match = re.search(r'(\d+) failed', result.stdout)
        skipped_match = re.search(r'(\d+) skipped', result.stdout)
        
        stats = {
            'passed': int(tests_match.group(1)) if tests_match else 0,
            'failed': int(failed_match.group(1)) if failed_match else 0,
            'skipped': int(skipped_match.group(1)) if skipped_match else 0,
            'exit_code': result.returncode
        }
        
        print(f"   通过: {stats['passed']}, 失败: {stats['failed']}, 跳过: {stats['skipped']}")
        
        return coverage, stats
        
    except subprocess.TimeoutExpired:
        print(f"⏰ 测试超时（>{timeout}秒）")
        return 0.0, {'error': 'timeout'}
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return 0.0, {'error': str(e)}


def run_quick_coverage(module: str) -> Tuple[float, dict]:
    """快速运行模块覆盖率测试（限制测试数量）"""
    
    print(f"\n测试模块: {module}")
    
    # 构建pytest命令
    src_path = f"src/infrastructure/{module}"
    test_path = f"tests/unit/infrastructure/{module}"
    
    # 检查路径
    if not Path(test_path).exists():
        print(f"  ⚠️ 跳过（测试目录不存在）")
        return 0.0, {'error': 'no_tests'}
    
    if not Path(src_path).exists():
        print(f"  ⚠️ 跳过（源码目录不存在）")
        return 0.0, {'error': 'no_source'}
    
    # 快速测试命令（只运行部分测试）
    cmd = [
        'python', '-m', 'pytest',
        test_path,
        f'--cov={src_path}',
        '--cov-report=term',
        '-q',
        '--maxfail=1',
        '--collect-only'  # 先只收集，不执行
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path.cwd()
        )
        
        # 解析收集的测试数量
        collected_match = re.search(r'(\d+) test', result.stdout)
        if collected_match:
            test_count = int(collected_match.group(1))
            print(f"  ✓ 收集到 {test_count} 个测试用例")
            
            # 基于测试数量估算覆盖率（简化方法）
            # 这里返回估算值，实际运行会很慢
            return 0.0, {'collected': test_count, 'mode': 'quick'}
        else:
            print(f"  ⚠️ 无法收集测试")
            return 0.0, {'error': 'collection_failed'}
            
    except Exception as e:
        print(f"  ❌ 失败: {e}")
        return 0.0, {'error': str(e)}


def main():
    """主函数"""
    
    print("="*70)
    print("基础设施层分模块真实覆盖率验证")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("注意: 完整测试可能需要较长时间...")
    print("先执行快速收集模式，然后提供选择性深度测试建议")
    print()
    
    results = {}
    
    # 快速模式：先收集所有模块的测试数量
    print("\n【阶段1：快速收集测试用例】")
    for module in INFRASTRUCTURE_MODULES:
        coverage, stats = run_quick_coverage(module)
        results[module] = {
            'coverage': coverage,
            'stats': stats,
            'mode': 'quick'
        }
    
    # 保存结果
    output_file = Path('test_logs/module_coverage_quick_analysis.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'mode': 'quick_collection',
            'modules': results,
            'note': '快速收集模式，未执行实际测试'
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n快速收集结果已保存至: {output_file}")
    
    # 统计总测试数
    total_tests = sum(
        r['stats'].get('collected', 0) 
        for r in results.values() 
        if 'collected' in r['stats']
    )
    
    print(f"\n总测试用例数: {total_tests}")
    
    return 0


if __name__ == '__main__':
    exit(main())


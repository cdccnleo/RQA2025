#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统计基础设施层测试用例数量
"""

import os
import ast
from pathlib import Path
from collections import defaultdict

def count_tests_in_file(filepath):
    """统计单个文件中的测试用例数量"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        test_count = 0
        
        for node in ast.walk(tree):
            # 统计测试函数
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    test_count += 1
        
        return test_count
    except Exception as e:
        print(f"⚠️  无法解析 {filepath}: {e}")
        return 0

def count_all_infrastructure_tests():
    """统计所有基础设施层测试"""
    test_base = Path("tests/unit/infrastructure")
    
    if not test_base.exists():
        print(f"❌ 测试目录不存在: {test_base}")
        return
    
    module_stats = defaultdict(lambda: {'files': 0, 'tests': 0, 'file_list': []})
    total_files = 0
    total_tests = 0
    
    # 遍历所有测试文件
    for test_file in test_base.rglob("test_*.py"):
        # 获取模块名（第一级子目录）
        rel_path = test_file.relative_to(test_base)
        parts = rel_path.parts
        
        if len(parts) > 0:
            module = parts[0]
        else:
            module = "root"
        
        # 统计测试数量
        test_count = count_tests_in_file(test_file)
        
        if test_count > 0:
            module_stats[module]['files'] += 1
            module_stats[module]['tests'] += test_count
            module_stats[module]['file_list'].append({
                'file': str(rel_path),
                'tests': test_count
            })
            
            total_files += 1
            total_tests += test_count
    
    # 打印结果
    print("="*100)
    print("🧪 基础设施层测试用例统计")
    print("="*100)
    print()
    
    # 按模块排序
    sorted_modules = sorted(module_stats.items(), key=lambda x: x[1]['tests'], reverse=True)
    
    print(f"{'模块':<20} {'测试文件数':<12} {'测试用例数':<12} {'平均每文件':<12}")
    print("-"*100)
    
    for module, stats in sorted_modules:
        avg = stats['tests'] / stats['files'] if stats['files'] > 0 else 0
        print(f"{module:<20} {stats['files']:<12} {stats['tests']:<12} {avg:<12.1f}")
    
    print("-"*100)
    print(f"{'总计':<20} {total_files:<12} {total_tests:<12} {total_tests/total_files if total_files > 0 else 0:<12.1f}")
    print()
    
    # 详细文件列表（可选）
    print("="*100)
    print("📋 详细文件列表（测试用例最多的前20个文件）")
    print("="*100)
    print()
    
    all_files = []
    for module, stats in module_stats.items():
        for file_info in stats['file_list']:
            all_files.append({
                'module': module,
                'file': file_info['file'],
                'tests': file_info['tests']
            })
    
    # 按测试数量排序
    all_files.sort(key=lambda x: x['tests'], reverse=True)
    
    print(f"{'排名':<6} {'模块':<15} {'测试用例数':<12} {'文件'}")
    print("-"*100)
    
    for idx, file_info in enumerate(all_files[:20], 1):
        print(f"{idx:<6} {file_info['module']:<15} {file_info['tests']:<12} {file_info['file']}")
    
    print()
    print("="*100)
    print(f"✅ 统计完成！基础设施层共有 {total_tests} 个测试用例")
    print("="*100)
    
    return {
        'total_files': total_files,
        'total_tests': total_tests,
        'modules': dict(module_stats)
    }

if __name__ == "__main__":
    count_all_infrastructure_tests()


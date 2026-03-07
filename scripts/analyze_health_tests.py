#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析Health模块测试质量问题
为什么4,145个测试只覆盖17.7%的代码？
"""

import os
import ast
from pathlib import Path
from collections import defaultdict, Counter

def analyze_test_file(filepath):
    """分析单个测试文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        info = {
            'file': str(filepath),
            'test_functions': [],
            'test_classes': [],
            'imports': [],
            'has_mock': False,
            'has_pytest_mark': False,
            'total_lines': len(content.split('\n')),
        }
        
        # 分析导入
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    info['imports'].append(alias.name)
                    if 'mock' in alias.name.lower() or 'unittest.mock' in alias.name:
                        info['has_mock'] = True
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                info['imports'].append(module)
                if 'mock' in module.lower():
                    info['has_mock'] = True
        
        # 分析测试函数和类
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('test_'):
                    # 检查是否有pytest装饰器
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name) and 'mark' in decorator.id:
                            info['has_pytest_mark'] = True
                        elif isinstance(decorator, ast.Attribute):
                            if 'mark' in getattr(decorator, 'attr', ''):
                                info['has_pytest_mark'] = True
                    
                    # 计算函数行数
                    func_lines = 1
                    if hasattr(node, 'end_lineno'):
                        func_lines = node.end_lineno - node.lineno + 1
                    
                    info['test_functions'].append({
                        'name': node.name,
                        'lines': func_lines,
                        'has_docstring': ast.get_docstring(node) is not None
                    })
            
            elif isinstance(node, ast.ClassDef):
                if node.name.startswith('Test'):
                    info['test_classes'].append({
                        'name': node.name,
                        'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')])
                    })
        
        return info
    
    except Exception as e:
        return {
            'file': str(filepath),
            'error': str(e)
        }

def analyze_health_module():
    """分析Health模块所有测试"""
    test_dir = Path("tests/unit/infrastructure/health")
    
    if not test_dir.exists():
        print(f"❌ 测试目录不存在: {test_dir}")
        return
    
    all_files = []
    total_tests = 0
    total_lines = 0
    
    # 收集所有测试文件
    for test_file in test_dir.rglob("test_*.py"):
        info = analyze_test_file(test_file)
        if 'error' not in info:
            all_files.append(info)
            total_tests += len(info['test_functions'])
            total_lines += info['total_lines']
    
    print("="*100)
    print("🔍 Health模块测试质量分析报告")
    print("="*100)
    print()
    
    print(f"📊 基础统计")
    print(f"  测试文件数：{len(all_files)}")
    print(f"  测试函数数：{total_tests}")
    print(f"  总代码行数：{total_lines:,}")
    print(f"  平均每文件测试数：{total_tests/len(all_files) if all_files else 0:.1f}")
    print(f"  平均每文件行数：{total_lines/len(all_files) if all_files else 0:.1f}")
    print()
    
    # 分析测试大小分布
    test_sizes = []
    for f in all_files:
        for test_func in f['test_functions']:
            test_sizes.append(test_func['lines'])
    
    if test_sizes:
        print(f"📏 测试函数大小分布")
        print(f"  最小：{min(test_sizes)} 行")
        print(f"  最大：{max(test_sizes)} 行")
        print(f"  平均：{sum(test_sizes)/len(test_sizes):.1f} 行")
        print(f"  中位数：{sorted(test_sizes)[len(test_sizes)//2]} 行")
        
        # 统计小测试（可能是空测试或skip）
        tiny_tests = [s for s in test_sizes if s <= 3]
        print(f"  ⚠️ 极小测试（≤3行）：{len(tiny_tests)} 个（{len(tiny_tests)/len(test_sizes)*100:.1f}%）")
        print()
    
    # 找出测试最多的文件
    files_by_test_count = sorted(all_files, key=lambda x: len(x['test_functions']), reverse=True)
    
    print(f"📋 测试数量最多的Top 10文件")
    print(f"{'排名':<6} {'测试数':<10} {'文件行数':<10} {'文件名'}")
    print("-"*100)
    for idx, f in enumerate(files_by_test_count[:10], 1):
        filename = Path(f['file']).name
        print(f"{idx:<6} {len(f['test_functions']):<10} {f['total_lines']:<10} {filename}")
    print()
    
    # 分析import使用
    has_mock_count = sum(1 for f in all_files if f['has_mock'])
    has_pytest_mark_count = sum(1 for f in all_files if f['has_pytest_mark'])
    
    print(f"🔧 测试技术使用")
    print(f"  使用Mock的文件：{has_mock_count} ({has_mock_count/len(all_files)*100:.1f}%)")
    print(f"  使用pytest.mark的文件：{has_pytest_mark_count} ({has_pytest_mark_count/len(all_files)*100:.1f}%)")
    print()
    
    # 统计最常见的导入
    all_imports = []
    for f in all_files:
        all_imports.extend(f['imports'])
    
    import_counter = Counter(all_imports)
    print(f"📦 最常用的导入（Top 10）")
    for imp, count in import_counter.most_common(10):
        print(f"  {imp:<40} {count:>4} 次")
    print()
    
    # 识别可能的问题
    print("⚠️ 潜在问题识别")
    
    # 1. 查找可能重复的测试
    test_name_counter = Counter()
    for f in all_files:
        for test_func in f['test_functions']:
            test_name_counter[test_func['name']] += 1
    
    duplicate_names = {name: count for name, count in test_name_counter.items() if count > 5}
    if duplicate_names:
        print(f"  1. 高度重复的测试名称（可能表示重复测试）：")
        for name, count in sorted(duplicate_names.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"     '{name}' 出现 {count} 次")
    
    # 2. 查找极小的测试
    tiny_test_files = []
    for f in all_files:
        tiny_count = sum(1 for t in f['test_functions'] if t['lines'] <= 3)
        if tiny_count > 10:
            tiny_test_files.append((Path(f['file']).name, tiny_count, len(f['test_functions'])))
    
    if tiny_test_files:
        print(f"  2. 包含大量极小测试的文件（可能是skip或空测试）：")
        for filename, tiny, total in sorted(tiny_test_files, key=lambda x: x[1], reverse=True)[:10]:
            print(f"     {filename}: {tiny}/{total} 个极小测试（{tiny/total*100:.1f}%）")
    
    # 3. 查找超大文件
    large_files = [(Path(f['file']).name, f['total_lines'], len(f['test_functions'])) 
                   for f in all_files if f['total_lines'] > 1000]
    if large_files:
        print(f"  3. 超大测试文件（>1000行）：")
        for filename, lines, tests in sorted(large_files, key=lambda x: x[1], reverse=True):
            print(f"     {filename}: {lines} 行, {tests} 个测试")
    
    print()
    print("="*100)
    print("💡 初步结论")
    print("="*100)
    
    avg_test_size = sum(test_sizes)/len(test_sizes) if test_sizes else 0
    tiny_ratio = len(tiny_tests)/len(test_sizes)*100 if test_sizes else 0
    
    if tiny_ratio > 30:
        print(f"⚠️ 严重问题：{tiny_ratio:.1f}% 的测试极小（≤3行），可能大量测试被skip或为空")
    
    if avg_test_size < 5:
        print(f"⚠️ 测试平均大小仅{avg_test_size:.1f}行，可能测试不够充分")
    
    if has_mock_count / len(all_files) < 0.3:
        print(f"⚠️ 仅{has_mock_count/len(all_files)*100:.1f}%的文件使用Mock，可能很多测试依赖真实对象导致覆盖率低")
    
    print()
    print("建议：")
    print("1. 检查skip的测试，确定是否应该启用")
    print("2. 审查极小测试，补充测试内容")
    print("3. 增加Mock使用，提高测试独立性")
    print("4. 检查重复测试名称，合并或优化")
    print("5. 重构超大测试文件，提高可维护性")

if __name__ == "__main__":
    analyze_health_module()


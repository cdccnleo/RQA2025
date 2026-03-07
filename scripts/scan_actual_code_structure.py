#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扫描实际代码结构 - 快速识别存在的类和函数
"""

import os
import ast
from pathlib import Path
from collections import defaultdict

def scan_python_file(filepath):
    """扫描Python文件，提取类和函数"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                # 只记录顶级函数
                if isinstance(node, ast.FunctionDef) and not isinstance(getattr(node, 'parent', None), ast.ClassDef):
                    functions.append(node.name)
        
        return {
            'file': str(filepath),
            'classes': classes,
            'functions': functions,
            'lines': len(content.split('\n'))
        }
    except Exception as e:
        return {
            'file': str(filepath),
            'error': str(e)
        }

def scan_module(module_path, max_files=30):
    """扫描模块，限制文件数"""
    results = []
    
    if not os.path.exists(module_path):
        return results
    
    py_files = list(Path(module_path).rglob("*.py"))[:max_files]
    
    for py_file in py_files:
        if '__pycache__' not in str(py_file):
            result = scan_python_file(py_file)
            if 'error' not in result and (result['classes'] or result['functions']):
                results.append(result)
    
    return results

def print_module_structure(module_name, max_files=30):
    """打印模块结构"""
    module_path = f"src/infrastructure/{module_name}"
    
    print(f"\n{'='*80}")
    print(f"📁 {module_name} 模块代码结构（前{max_files}个文件）")
    print('='*80)
    
    results = scan_module(module_path, max_files)
    
    if not results:
        print(f"❌ 未找到模块或无有效Python文件")
        return
    
    print(f"\n找到 {len(results)} 个有效文件\n")
    
    for idx, result in enumerate(results[:max_files], 1):
        filename = Path(result['file']).name
        rel_path = Path(result['file']).relative_to(f"src/infrastructure/{module_name}")
        
        print(f"{idx}. {rel_path} ({result['lines']}行)")
        
        if result['classes']:
            print(f"   类: {', '.join(result['classes'][:5])}")
            if len(result['classes']) > 5:
                print(f"        ... 还有 {len(result['classes'])-5} 个类")
        
        if result['functions']:
            funcs = [f for f in result['functions'] if not f.startswith('_')]
            if funcs:
                print(f"   函数: {', '.join(funcs[:3])}")
                if len(funcs) > 3:
                    print(f"          ... 还有 {len(funcs)-3} 个函数")
        print()

def main():
    """主函数 - 扫描重点模块"""
    modules = ['health', 'logging']
    
    for module in modules:
        print_module_structure(module, max_files=20)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础设施层测试覆盖率分析脚本
分析现有测试用例并识别需要补充的模块
"""
import os
import sys
import importlib
import inspect
from pathlib import Path
from collections import defaultdict

def analyze_infrastructure_modules():
    """分析基础设施层模块"""
    infrastructure_path = Path("src/infrastructure")
    modules = {}
    
    # 遍历基础设施层目录
    for item in infrastructure_path.rglob("*.py"):
        if item.name.startswith("__"):
            continue
            
        # 计算相对路径
        rel_path = item.relative_to(infrastructure_path)
        module_name = str(rel_path).replace("\\", ".").replace("/", ".").replace(".py", "")
        
        # 读取文件内容
        try:
            with open(item, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 统计代码行数
            lines = content.split('\n')
            code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            
            modules[module_name] = {
                'path': str(item),
                'lines': len(lines),
                'code_lines': len(code_lines),
                'classes': [],
                'functions': []
            }
            
            # 尝试解析类和函数
            try:
                spec = importlib.util.spec_from_file_location(module_name, item)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 获取类
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and obj.__module__ == module.__name__:
                            modules[module_name]['classes'].append(name)
                        elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                            modules[module_name]['functions'].append(name)
                            
            except Exception as e:
                print(f"无法解析模块 {module_name}: {e}")
                
        except Exception as e:
            print(f"无法读取文件 {item}: {e}")
    
    return modules

def analyze_test_coverage():
    """分析测试覆盖率"""
    test_path = Path("tests/unit/infrastructure")
    test_files = {}
    
    if test_path.exists():
        for item in test_path.rglob("*.py"):
            if item.name.startswith("__"):
                continue
                
            rel_path = item.relative_to(test_path)
            test_name = str(rel_path).replace("\\", ".").replace("/", ".").replace(".py", "")
            
            try:
                with open(item, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                lines = content.split('\n')
                test_methods = [line for line in lines if line.strip().startswith('def test_')]
                
                test_files[test_name] = {
                    'path': str(item),
                    'lines': len(lines),
                    'test_methods': len(test_methods),
                    'methods': [line.split('def ')[1].split('(')[0] for line in test_methods]
                }
                
            except Exception as e:
                print(f"无法读取测试文件 {item}: {e}")
    
    return test_files

def identify_missing_tests(modules, test_files):
    """识别缺失的测试"""
    missing_tests = []
    covered_modules = set()
    
    # 分析已测试的模块
    for test_name, test_info in test_files.items():
        # 尝试匹配测试文件与源模块
        for module_name in modules.keys():
            if module_name.replace('.', '_') in test_name or test_name.endswith(module_name.split('.')[-1]):
                covered_modules.add(module_name)
                break
    
    # 识别未测试的模块
    for module_name, module_info in modules.items():
        if module_name not in covered_modules:
            missing_tests.append({
                'module': module_name,
                'path': module_info['path'],
                'lines': module_info['lines'],
                'classes': module_info['classes'],
                'functions': module_info['functions'],
                'priority': 'high' if module_info['lines'] > 100 else 'medium'
            })
    
    return missing_tests

def generate_coverage_report():
    """生成覆盖率报告"""
    print("=== 基础设施层测试覆盖率分析 ===\n")
    
    # 分析模块
    print("1. 分析基础设施层模块...")
    modules = analyze_infrastructure_modules()
    print(f"发现 {len(modules)} 个模块")
    
    # 分析测试
    print("\n2. 分析现有测试用例...")
    test_files = analyze_test_coverage()
    print(f"发现 {len(test_files)} 个测试文件")
    
    # 识别缺失的测试
    print("\n3. 识别缺失的测试...")
    missing_tests = identify_missing_tests(modules, test_files)
    print(f"发现 {len(missing_tests)} 个模块缺少测试")
    
    # 统计信息
    total_lines = sum(module['lines'] for module in modules.values())
    total_test_lines = sum(test['lines'] for test in test_files.values())
    total_test_methods = sum(test['test_methods'] for test in test_files.values())
    
    print(f"\n=== 统计信息 ===")
    print(f"总模块数: {len(modules)}")
    print(f"总代码行数: {total_lines}")
    print(f"总测试文件数: {len(test_files)}")
    print(f"总测试方法数: {total_test_methods}")
    print(f"总测试代码行数: {total_test_lines}")
    print(f"缺少测试的模块数: {len(missing_tests)}")
    
    # 按优先级排序缺失的测试
    high_priority = [test for test in missing_tests if test['priority'] == 'high']
    medium_priority = [test for test in missing_tests if test['priority'] == 'medium']
    
    print(f"\n=== 高优先级缺失测试 ({len(high_priority)} 个) ===")
    for test in high_priority[:10]:  # 显示前10个
        print(f"- {test['module']} ({test['lines']} 行)")
        if test['classes']:
            print(f"  类: {', '.join(test['classes'])}")
        if test['functions']:
            print(f"  函数: {', '.join(test['functions'][:5])}...")
    
    print(f"\n=== 中优先级缺失测试 ({len(medium_priority)} 个) ===")
    for test in medium_priority[:10]:  # 显示前10个
        print(f"- {test['module']} ({test['lines']} 行)")
    
    # 生成测试建议
    print(f"\n=== 测试建议 ===")
    print("1. 优先为高优先级模块创建测试")
    print("2. 重点关注核心功能模块")
    print("3. 确保测试覆盖率达到90%")
    
    return {
        'modules': modules,
        'test_files': test_files,
        'missing_tests': missing_tests,
        'high_priority': high_priority,
        'medium_priority': medium_priority
    }

if __name__ == "__main__":
    import importlib.util
    report = generate_coverage_report()
    
    # 保存报告到文件
    with open("docs/infrastructure_coverage_analysis.md", "w", encoding="utf-8") as f:
        f.write("# 基础设施层测试覆盖率分析报告\n\n")
        f.write(f"- 总模块数: {len(report['modules'])}\n")
        f.write(f"- 总测试文件数: {len(report['test_files'])}\n")
        f.write(f"- 缺少测试的模块数: {len(report['missing_tests'])}\n")
        f.write(f"- 高优先级缺失测试: {len(report['high_priority'])}\n")
        f.write(f"- 中优先级缺失测试: {len(report['medium_priority'])}\n")
        
        f.write("\n## 高优先级缺失测试\n\n")
        for test in report['high_priority']:
            f.write(f"- {test['module']} ({test['lines']} 行)\n")
            
        f.write("\n## 中优先级缺失测试\n\n")
        for test in report['medium_priority']:
            f.write(f"- {test['module']} ({test['lines']} 行)\n")
    
    print(f"\n报告已保存到 docs/infrastructure_coverage_analysis.md") 
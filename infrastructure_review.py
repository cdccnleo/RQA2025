"""
基础设施层代码审查分析脚本

本脚本用于对基础设施层代码进行全面的组织结构、代码重叠和冗余分析。
"""

import os
import re
from collections import defaultdict
import json


def analyze_duplicate_files():
    """分析重复文件名问题"""
    print("🔍 分析重复文件名...")

    infra_dir = 'src/infrastructure'
    duplicate_files = defaultdict(list)

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                duplicate_files[file].append(os.path.join(root, file))

    duplicate_count = 0
    for filename, paths in duplicate_files.items():
        if len(paths) > 1:
            duplicate_count += 1

    print(f"发现 {duplicate_count} 个重复文件名")
    return duplicate_files


def analyze_module_structure():
    """分析模块目录结构"""
    print("🏗️ 分析模块目录结构...")

    infra_dir = 'src/infrastructure'
    modules = ['cache', 'config', 'error', 'health', 'logging', 'resource', 'utils']

    structure_report = {}

    for module in modules:
        module_path = os.path.join(infra_dir, module)
        if not os.path.exists(module_path):
            continue

        # 统计文件数量
        total_files = 0
        subdirs = []

        for root, dirs, files in os.walk(module_path):
            py_files = [f for f in files if f.endswith('.py')]
            total_files += len(py_files)

            if root == module_path:
                subdirs = [d for d in dirs if not d.startswith('__')]

        # 检查根目录文件过多的问题
        root_files = []
        for item in os.listdir(module_path):
            if item.endswith('.py') and not item.startswith('__'):
                root_files.append(item)

        structure_report[module] = {
            'total_files': total_files,
            'subdirs': subdirs,
            'root_files_count': len(root_files),
            'has_too_many_root_files': len(root_files) > 10
        }

    return structure_report


def analyze_code_overlap():
    """分析代码重叠和冗余"""
    print("🔄 分析代码重叠和冗余...")

    infra_dir = 'src/infrastructure'

    # 1. 分析类名重复
    class_names = defaultdict(list)
    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 查找类定义
                    classes = re.findall(r'class\s+(\w+)', content)
                    for cls in classes:
                        class_names[cls].append(file_path.replace('src/infrastructure/', ''))
                except:
                    pass

    duplicate_classes = {name: paths for name, paths in class_names.items() if len(paths) > 1}

    # 2. 分析导入语句重复
    import_patterns = defaultdict(list)
    common_imports = [
        'from typing import', 'import typing', 'from abc import', 'import abc',
        'from dataclasses import', 'import dataclasses', 'import asyncio',
        'import threading', 'import time', 'import os', 'import sys'
    ]

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    for line_num, line in enumerate(lines[:20]):  # 只检查前20行
                        line = line.strip()
                        for pattern in common_imports:
                            if line.startswith(pattern):
                                import_patterns[pattern].append(
                                    file_path.replace('src/infrastructure/', ''))
                                break
                except:
                    pass

    # 3. 功能相似文件模式分析
    patterns_to_check = [
        ('cache', '缓存相关'),
        ('config', '配置相关'),
        ('monitor', '监控相关'),
        ('handler', '处理器相关'),
        ('service', '服务相关'),
        ('manager', '管理器相关'),
        ('factory', '工厂相关'),
        ('validator', '验证器相关'),
        ('logger', '日志相关'),
        ('error', '错误相关'),
        ('health', '健康检查相关'),
        ('resource', '资源管理相关'),
        ('utils', '工具相关')
    ]

    functional_patterns = {}
    for pattern, desc in patterns_to_check:
        matching_files = []
        for root, dirs, files in os.walk(infra_dir):
            for file in files:
                if pattern in file.lower() and file.endswith('.py'):
                    matching_files.append(os.path.join(
                        root, file).replace('src/infrastructure/', ''))

        functional_patterns[pattern] = {
            'description': desc,
            'count': len(matching_files),
            'files': matching_files[:5]  # 只保留前5个示例
        }

    return {
        'duplicate_classes': duplicate_classes,
        'import_patterns': dict(import_patterns),
        'functional_patterns': functional_patterns
    }


def analyze_interfaces():
    """分析接口规范性"""
    print("🔧 分析接口规范性...")

    infra_dir = 'src/infrastructure'

    # 1. 检查接口文件
    interface_files = []
    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file == 'interfaces.py':
                interface_files.append(os.path.join(root, file))

    # 2. 分析接口命名规范
    interface_classes = []
    non_standard_interfaces = []

    for interface_file in interface_files:
        try:
            with open(interface_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找接口类定义
            classes = re.findall(r'class\s+(\w+)', content)
            for cls in classes:
                interface_classes.append(cls)
                if not cls.startswith('I'):
                    non_standard_interfaces.append(
                        (cls, interface_file.replace('src/infrastructure/', '')))
        except Exception as e:
            print(f'读取文件 {interface_file} 时出错: {e}')

    # 3. 检查接口文档
    documented_interfaces = 0
    total_interfaces = 0

    for interface_file in interface_files:
        try:
            with open(interface_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 统计有文档的接口
            interface_blocks = re.findall(r'class\s+I\w+.*?(?=class|\Z)', content, re.DOTALL)
            for block in interface_blocks:
                total_interfaces += 1
                if '"""' in block:
                    documented_interfaces += 1
        except:
            pass

    # 4. 检查接口继承方式
    abc_usage = 0
    protocol_usage = 0

    for interface_file in interface_files:
        try:
            with open(interface_file, 'r', encoding='utf-8') as f:
                content = f.read()

            if 'from abc import' in content or 'import abc' in content:
                abc_usage += 1
            if 'from typing import Protocol' in content or 'Protocol' in content:
                protocol_usage += 1
        except:
            pass

    return {
        'interface_files_count': len(interface_files),
        'total_interfaces': len(interface_classes),
        'i_prefix_interfaces': len([c for c in interface_classes if c.startswith('I')]),
        'non_standard_interfaces': non_standard_interfaces,
        'documented_interfaces': documented_interfaces,
        'total_interface_classes': total_interfaces,
        'abc_usage': abc_usage,
        'protocol_usage': protocol_usage,
        'interface_files': [f.replace('src/infrastructure/', '') for f in interface_files]
    }


def analyze_dependencies():
    """分析模块间依赖关系"""
    print("🔗 分析模块间依赖关系...")

    infra_dir = 'src/infrastructure'
    modules = ['cache', 'config', 'error', 'health', 'logging', 'resource', 'utils']

    # 简化的循环依赖检查
    potential_cycles = []

    # 检查已知的循环依赖模式
    cycle_patterns = [
        ('resource', 'logging'),
        ('logging', 'error'),
        ('error', 'logging'),
        ('cache', 'config'),
        ('health', 'config'),
        ('utils', 'cache'),
        ('utils', 'config'),
        ('utils', 'logging')
    ]

    for src, dst in cycle_patterns:
        src_path = os.path.join(infra_dir, src)
        dst_path = os.path.join(infra_dir, dst)

        if os.path.exists(src_path) and os.path.exists(dst_path):
            # 简单的导入检查
            has_import = False
            for root, dirs, files in os.walk(src_path):
                for file in files:
                    if file.endswith('.py'):
                        try:
                            with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                content = f.read()
                                if f'from src.infrastructure.{dst}' in content or f'import src.infrastructure.{dst}' in content:
                                    has_import = True
                                    break
                        except:
                            pass
                if has_import:
                    break

            if has_import:
                # 检查反向依赖
                has_reverse = False
                for root, dirs, files in os.walk(dst_path):
                    for file in files:
                        if file.endswith('.py'):
                            try:
                                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    if f'from src.infrastructure.{src}' in content or f'import src.infrastructure.{src}' in content:
                                        has_reverse = True
                                        break
                            except:
                                pass
                    if has_reverse:
                        break

                if has_reverse:
                    potential_cycles.append(f'{src} ↔ {dst}')

    return {
        'potential_cycles': potential_cycles,
        'cycle_count': len(potential_cycles)
    }


def generate_final_report():
    """生成综合分析报告"""
    print("📊 生成基础设施层代码审查综合报告")
    print("=" * 60)

    # 执行各项分析
    duplicate_files = analyze_duplicate_files()
    structure_report = analyze_module_structure()
    overlap_report = analyze_code_overlap()
    interface_report = analyze_interfaces()
    dependency_report = analyze_dependencies()

    # 计算总体统计
    total_files = 0
    total_lines = 0
    infra_dir = 'src/infrastructure'

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                total_files += 1
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        total_lines += len(f.readlines())
                except:
                    pass

    # 输出总体统计
    print("\n📈 总体统计:")
    print(f"  总文件数: {total_files}")
    print(f"  总行数: {total_lines:,}")
    print(f"  平均每文件行数: {total_lines//total_files if total_files > 0 else 0}")

    # 输出重复文件名统计
    duplicate_count = sum(1 for paths in duplicate_files.values() if len(paths) > 1)
    print(f"\n📋 重复文件名统计:")
    print(f"  重复文件名总数: {duplicate_count}")

    # 输出模块结构分析
    print("\n🏗️ 模块结构分析:")
    problematic_modules = [m for m, data in structure_report.items()
                           if data['has_too_many_root_files']]
    print(f"  结构良好模块: {len(structure_report) - len(problematic_modules)}")
    print(f"  需优化模块: {len(problematic_modules)}")
    if problematic_modules:
        print(f"  问题模块: {', '.join(problematic_modules)}")

    # 输出代码重叠分析
    print("\n🔄 代码重叠分析:")
    print(f"  重复类名总数: {len(overlap_report['duplicate_classes'])}")
    print(f"  功能模式文件数: {sum(p['count'] for p in overlap_report['functional_patterns'].values())}")

    # 输出接口规范分析
    print("\n🔧 接口规范分析:")
    print(f"  接口文件数: {interface_report['interface_files_count']}")
    print(f"  总接口数: {interface_report['total_interfaces']}")
    print(f"  I前缀接口: {interface_report['i_prefix_interfaces']}")
    print(f"  非标准接口: {len(interface_report['non_standard_interfaces'])}")

    doc_rate = 0
    if interface_report['total_interface_classes'] > 0:
        doc_rate = (interface_report['documented_interfaces'] /
                    interface_report['total_interface_classes']) * 100
    print(f"  文档覆盖率: {doc_rate:.1f}%")

    # 输出依赖关系分析
    print("\n🔗 依赖关系分析:")
    print(f"  潜在循环依赖: {dependency_report['cycle_count']}")

    # 代码质量评估
    print("\n🏆 代码质量评估:")
    quality_score = 0
    max_score = 100

    # 接口文档评分 (20分)
    doc_score = min(20, doc_rate * 0.2)
    quality_score += doc_score
    print(f"  接口文档: {doc_score:.1f}/20")

    # 结构组织评分 (30分)
    structure_score = max(0, 30 - len(problematic_modules) * 5)
    quality_score += structure_score
    print(f"  代码结构: {structure_score:.1f}/30")

    # 依赖关系评分 (25分)
    dependency_score = max(0, 25 - dependency_report['cycle_count'] * 5)
    quality_score += dependency_score
    print(f"  依赖关系: {dependency_score:.1f}/25")

    # 命名规范评分 (25分)
    naming_score = min(25, interface_report['i_prefix_interfaces'] /
                       max(1, interface_report['total_interfaces']) * 25)
    quality_score += naming_score
    print(f"  命名规范: {naming_score:.1f}/25")

    print(f"\n📊 总体评分: {quality_score:.1f}/{max_score}")

    # 优化建议
    print("\n💡 优化建议:")
    suggestions = [
        "1. 完善接口文档 - 当前文档覆盖率不足",
        "2. 重构模块结构 - 多个模块根目录文件过多",
        "3. 消除重复代码 - 大量重复类名需要整合",
        "4. 优化依赖关系 - 解决潜在的循环依赖",
        "5. 统一命名规范 - 确保所有接口使用I前缀",
        "6. 改善代码组织 - 将utils模块进一步细分",
        "7. 减少文件重复 - 清理重复文件名",
        "8. 优化导入结构 - 统一导入语句规范"
    ]

    for suggestion in suggestions:
        print(f"  {suggestion}")

    # 保存详细报告到文件
    report_data = {
        'summary': {
            'total_files': total_files,
            'total_lines': total_lines,
            'duplicate_files_count': duplicate_count,
            'quality_score': quality_score
        },
        'structure_analysis': structure_report,
        'overlap_analysis': overlap_report,
        'interface_analysis': interface_report,
        'dependency_analysis': dependency_report,
        'suggestions': suggestions
    }

    with open('infrastructure_review_report.json', 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    print("\n📄 详细报告已保存到: infrastructure_review_report.json")
    return report_data


if __name__ == "__main__":
    generate_final_report()

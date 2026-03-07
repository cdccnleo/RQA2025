#!/usr/bin/env python3
"""
基础设施层配置管理代码组织分析脚本

基于代码审查清单进行全面分析
"""

import os


def analyze_code_organization():
    """分析代码组织、重叠和冗余情况"""

    print('=== 🔍 基础设施层配置管理代码组织审查 ===')
    print('基于 docs/code_review_checklist.md 标准进行审查')
    print('审查时间: 2025年9月23日')
    print('审查对象: src/infrastructure/config/')
    print('重点: 代码组织、重叠和冗余情况')
    print()

    config_dir = 'src/infrastructure/config'
    if not os.path.exists(config_dir):
        print(f'❌ 配置目录不存在: {config_dir}')
        return

    # 1. 检查目录结构和文件组织
    print('📁 1. 代码组织结构分析:')
    dirs = []
    files = []

    for root, dirnames, filenames in os.walk(config_dir):
        for dirname in dirnames:
            if not dirname.startswith('__'):
                dirs.append(dirname)
        for filename in filenames:
            if filename.endswith('.py') and not filename.startswith('__'):
                files.append(os.path.join(root, filename))

    print(f'   📂 目录数量: {len(dirs)} 个')
    print(f'   📄 文件数量: {len([f for f in files if "__pycache__" not in f])} 个')

    print('   🗂️ 目录结构:')
    for d in sorted(dirs):
        print(f'     - {d}/')

    print()

    # 2. 检查重复类定义
    print('🔄 2. 重复类定义检查:')
    classes = {}
    for file_path in files:
        if '__pycache__' in file_path:
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    if line.strip().startswith('class '):
                        class_name = line.split('class ')[1].split('(')[0].split(':')[0].strip()
                        if class_name not in classes:
                            classes[class_name] = []
                        classes[class_name].append(file_path)
        except Exception as e:
            print(f'   ⚠️ 读取文件失败: {file_path} - {e}')

    duplicates = {k: v for k, v in classes.items() if len(v) > 1}
    print(f'   ⚠️ 发现重复类定义: {len(duplicates)} 个')

    if duplicates:
        print('   📋 重复类详情:')
        for class_name, file_list in sorted(list(duplicates.items())[:8]):
            print(f'     - {class_name}: {len(file_list)} 个位置')
            for f in file_list[:2]:
                print(f'       └─ {os.path.relpath(f, config_dir)}')

    print()

    # 3. 检查重复方法
    print('⚙️ 3. 重复方法检查:')
    methods = {}
    for file_path in files:
        if '__pycache__' in file_path:
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                method_patterns = [
                    ('validate_config', 'def validate_config('),
                    ('load_config', 'def load_config('),
                    ('save_config', 'def save_config('),
                    ('get_config', 'def get_config('),
                    ('set_config', 'def set_config(')
                ]
                for method_name, pattern in method_patterns:
                    if pattern in content:
                        if method_name not in methods:
                            methods[method_name] = []
                        methods[method_name].append(file_path)
        except Exception as e:
            print(f'   ⚠️ 读取文件失败: {file_path} - {e}')

    print(f'   🔍 发现潜在重复方法模式: {len(methods)} 种')
    for method_name, file_list in methods.items():
        if len(file_list) > 2:
            print(f'     - {method_name}: {len(file_list)} 个文件')
            for f in file_list[:3]:
                print(f'       └─ {os.path.relpath(f, config_dir)}')

    print()

    # 4. 检查导入重复
    print('📦 4. 导入语句重复检查:')
    imports = {}
    for file_path in files:
        if '__pycache__' in file_path:
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                import_lines = [line.strip() for line in content.split('\n')
                                if line.strip().startswith(('import ', 'from '))]
                for imp in import_lines:
                    if imp not in imports:
                        imports[imp] = []
                    imports[imp].append(file_path)
        except Exception as e:
            print(f'   ⚠️ 读取文件失败: {file_path} - {e}')

    # 找出重复导入
    repeated_imports = {k: v for k, v in imports.items() if len(v) > 3}
    print(f'   🔄 高频重复导入: {len(repeated_imports)} 种')

    for imp, files_list in sorted(list(repeated_imports.items())[:5]):
        print(f'     - "{imp}": {len(files_list)} 个文件')

    print()

    # 5. 检查文件大小分布
    print('📏 5. 文件大小分布分析:')
    file_sizes = {}
    for file_path in files:
        if '__pycache__' in file_path:
            continue
        try:
            size = os.path.getsize(file_path)
            file_sizes[file_path] = size
        except:
            pass

    if file_sizes:
        sizes = list(file_sizes.values())
        avg_size = sum(sizes) / len(sizes)
        max_size = max(sizes)
        max_file = max(file_sizes.keys(), key=lambda k: file_sizes[k])

        print(f'   📊 平均文件大小: {avg_size:.0f} 字节 ({avg_size/1024:.1f} KB)')
        print(
            f'   📈 最大文件: {os.path.relpath(max_file, config_dir)} ({max_size:,} 字节, {max_size/1024:.1f} KB)')

        # 检查大文件
        large_files = [f for f, s in file_sizes.items() if s > 15000]  # >15KB
        if large_files:
            print(f'   ⚠️ 大文件数量 (>15KB): {len(large_files)} 个')
            for f in large_files[:3]:
                size_kb = file_sizes[f] / 1024
                print(f'     - {os.path.relpath(f, config_dir)}: {size_kb:.1f} KB')

    print()

    # 6. 代码复杂度初步分析
    print('🧹 6. 代码质量初步分析:')
    total_lines = 0
    total_classes = 0
    total_functions = 0

    for file_path in files:
        if '__pycache__' in file_path:
            continue
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = [line for line in content.split(
                    '\n') if line.strip() and not line.strip().startswith('#')]
                total_lines += len(lines)
                total_classes += content.count('class ')
                total_functions += content.count('def ')
        except:
            pass

    print(f'   📝 总代码行数: {total_lines:,} 行')
    print(f'   🏗️ 总类定义数: {total_classes} 个')
    print(f'   ⚙️ 总函数定义数: {total_functions} 个')

    if total_lines > 0 and total_classes > 0:
        print(f'   📈 平均每类方法数: {total_functions/total_classes:.1f} 个')
        print(f'   📉 代码密度: {total_classes/total_lines*100:.2f}% 类/行')

    print()
    print('🎯 审查结论:')

    # 问题严重程度评估
    issues = {
        'critical': [],
        'major': [],
        'minor': []
    }

    # Critical: 必须立即修复
    if len(duplicates) > 10:
        issues['critical'].append(f'重复类定义过多 ({len(duplicates)}个)')
    if len([m for m, f in methods.items() if len(f) > 3]) > 0:
        issues['critical'].append('重复方法模式严重')

    # Major: 建议尽快修复
    if len(repeated_imports) > 5:
        issues['major'].append(f'导入语句重复严重 ({len(repeated_imports)}种)')
    if len(large_files) > 2:
        issues['major'].append(f'大文件过多 ({len(large_files)}个)')

    # Minor: 可选改进
    issues['minor'].append('代码组织结构需要持续优化')

    print('   🔴 Critical (必须修复):')
    for issue in issues['critical']:
        print(f'     - {issue}')

    print('   🟡 Major (建议修复):')
    for issue in issues['major']:
        print(f'     - {issue}')

    print('   🟢 Minor (可选改进):')
    for issue in issues['minor']:
        print(f'     - {issue}')

    print()
    print('📋 建议行动计划:')
    print('   1. 清理重复类定义 (P0)')
    print('   2. 重构重复方法实现 (P0)')
    print('   3. 优化导入语句 (P1)')
    print('   4. 拆分大文件 (P1)')
    print('   5. 建立代码规范 (P2)')

    # 生成详细报告
    report_file = 'INFRASTRUCTURE_CONFIG_CODE_REVIEW_ORGANIZATION.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('# 基础设施层配置管理代码组织审查报告\n\n')
        f.write('## 📊 审查概览\n\n')
        f.write(f'- **审查时间**: 2025年9月23日\n')
        f.write(f'- **审查对象**: src/infrastructure/config/\n')
        f.write(f'- **审查标准**: docs/code_review_checklist.md\n')
        f.write(
            f'- **发现问题**: {len(duplicates)}个重复类，{len([m for m, f in methods.items() if len(f) > 2])}个重复方法模式\n\n')

        f.write('## 🔍 详细分析结果\n\n')

        # 重复类分析
        f.write('### 重复类定义分析\n\n')
        f.write(f'发现 {len(duplicates)} 个重复类定义：\n\n')
        for class_name, file_list in sorted(list(duplicates.items())[:10]):
            f.write(f'- **{class_name}** ({len(file_list)} 个位置):\n')
            for file_path in file_list:
                f.write(f'  - `{os.path.relpath(file_path, config_dir)}`\n')
            f.write('\n')

        # 重复方法分析
        f.write('### 重复方法分析\n\n')
        f.write(f'发现 {len([m for m, f in methods.items() if len(f) > 2])} 个重复方法模式：\n\n')
        for method_name, file_list in methods.items():
            if len(file_list) > 2:
                f.write(f'- **{method_name}** ({len(file_list)} 个文件):\n')
                for file_path in file_list:
                    f.write(f'  - `{os.path.relpath(file_path, config_dir)}`\n')
                f.write('\n')

        # 导入重复分析
        f.write('### 导入重复分析\n\n')
        f.write(f'发现 {len(repeated_imports)} 种高频重复导入：\n\n')
        for imp, files_list in sorted(list(repeated_imports.items())[:10]):
            f.write(f'- **{imp}** ({len(files_list)} 个文件)\n')

        f.write('\n## 🎯 改进建议\n\n')
        f.write('### P0 - 紧急任务\n')
        f.write('1. 清理重复类定义\n')
        f.write('2. 重构重复方法实现\n\n')

        f.write('### P1 - 重要任务\n')
        f.write('1. 优化导入语句重复\n')
        f.write('2. 拆分过大的文件\n\n')

        f.write('### P2 - 优化任务\n')
        f.write('1. 建立代码组织规范\n')
        f.write('2. 完善文档和注释\n')

    print(f'\\n📄 详细报告已生成: {report_file}')


if __name__ == '__main__':
    analyze_code_organization()

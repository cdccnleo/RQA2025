"""
基础设施层代码审查复核脚本

对基础设施层代码组织、重叠和冗余情况进行全面复核
"""

import os
from pathlib import Path
from collections import defaultdict


def analyze_duplicate_files():
    """分析重复文件名"""
    print('🔍 重复文件名分析:')
    print('=' * 30)

    infra_dir = Path('src/infrastructure')
    duplicate_files = defaultdict(list)

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                rel_path = str(Path(root).relative_to(infra_dir))
                duplicate_files[file].append(rel_path)

    duplicate_count = 0
    critical_duplicates = []

    for filename, locations in duplicate_files.items():
        if len(locations) > 1:
            duplicate_count += 1
            if duplicate_count <= 15:  # 显示前15个
                print(f'  {filename}: {len(locations)} 个位置')
                for loc in locations[:3]:
                    print(f'    - {loc}')
                if len(locations) > 3:
                    print(f'    ... 还有 {len(locations) - 3} 个位置')

            # 检查关键重复文件
            if filename in ['__init__.py', 'interfaces.py', 'base.py', 'core.py']:
                critical_duplicates.append((filename, len(locations)))

    print(f'\n总计发现 {duplicate_count} 个重复文件名')

    if critical_duplicates:
        print('\n⚠️ 关键重复文件:')
        for filename, count in critical_duplicates:
            print(f'  - {filename}: {count} 个位置')

    return duplicate_count


def analyze_module_structure():
    """分析模块结构"""
    print('\n🏗️ 模块结构分析:')
    print('=' * 30)

    infra_dir = Path('src/infrastructure')
    modules = {}

    for item in os.listdir(infra_dir):
        if os.path.isdir(infra_dir / item) and not item.startswith('.'):
            module_path = infra_dir / item
            total_files = len([f for f in module_path.rglob('*.py') if f.is_file()])
            subdirs = [d for d in module_path.iterdir() if d.is_dir()
                       and not d.name.startswith('.')]
            root_files = len([f for f in module_path.glob('*.py') if f.is_file()])

            modules[item] = {
                'total_files': total_files,
                'subdirs': len(subdirs),
                'root_files': root_files,
                'subdirs_list': [d.name for d in subdirs]
            }

            print(f'  {item}: {total_files} 个文件 ({len(subdirs)} 个子目录, {root_files} 个根目录文件)')

    return modules


def analyze_code_organization_issues(modules, duplicate_count):
    """分析代码组织问题"""
    print('\n🔍 代码组织问题识别:')
    print('=' * 30)

    issues = []

    for module, info in modules.items():
        if info['root_files'] > 10:
            issues.append(f'{module}模块根目录文件过多 ({info["root_files"]} 个)')

        # 检查是否有过多的子目录
        if info['subdirs'] > 15:
            issues.append(f'{module}模块子目录过多 ({info["subdirs"]} 个)')

    if duplicate_count > 20:
        issues.append(f'重复文件名过多 ({duplicate_count} 个)')

    if issues:
        for issue in issues:
            print(f'  ⚠️ {issue}')
    else:
        print('  ✅ 未发现明显代码组织问题')

    return issues


def analyze_code_redundancy():
    """分析代码冗余情况"""
    print('\n🔄 代码冗余分析:')
    print('=' * 30)

    infra_dir = Path('src/infrastructure')

    # 分析重复的类名
    class_names = defaultdict(list)

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 查找类定义
                    import re
                    classes = re.findall(r'class\s+(\w+)', content)
                    rel_path = str(file_path.relative_to(infra_dir))

                    for cls in classes:
                        class_names[cls].append(rel_path)

                except Exception as e:
                    continue

    # 分析重复类名
    duplicate_classes = {name: locs for name, locs in class_names.items() if len(locs) > 1}
    duplicate_class_count = len(duplicate_classes)

    print(f'发现 {duplicate_class_count} 个重复类名')

    if duplicate_class_count > 0:
        print('\n重复类名示例 (前10个):')
        for i, (cls_name, locations) in enumerate(list(duplicate_classes.items())[:10]):
            print(f'  {cls_name}: {len(locations)} 个位置')
            for loc in locations[:2]:
                print(f'    - {loc}')
            if len(locations) > 2:
                print(f'    ... 还有 {len(locations) - 2} 个位置')

    return duplicate_class_count


def analyze_import_structure():
    """分析导入结构"""
    print('\n📦 导入结构分析:')
    print('=' * 30)

    infra_dir = Path('src/infrastructure')
    import_issues = []

    wildcard_imports = 0
    long_imports = 0

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    for line_num, line in enumerate(lines[:50]):  # 只检查前50行
                        line = line.strip()
                        if 'from ' in line and ' import *' in line:
                            wildcard_imports += 1
                        elif line.startswith('from ') and len(line) > 100:
                            long_imports += 1

                except Exception as e:
                    continue

    print(f'通配符导入: {wildcard_imports} 个')
    print(f'过长导入语句: {long_imports} 个')

    if wildcard_imports > 0:
        import_issues.append(f'仍存在 {wildcard_imports} 个通配符导入')

    if long_imports > 0:
        import_issues.append(f'仍存在 {long_imports} 个过长导入语句')

    if import_issues:
        print('⚠️ 导入问题:')
        for issue in import_issues:
            print(f'  - {issue}')
    else:
        print('✅ 导入结构良好')

    return import_issues


def generate_final_report(modules, duplicate_count, duplicate_class_count, issues, import_issues):
    """生成最终报告"""
    print('\n📊 基础设施层代码审查复核报告')
    print('=' * 50)

    total_files = sum(info['total_files'] for info in modules.values())

    print(f'总模块数: {len(modules)}')
    print(f'总文件数: {total_files}')
    print(f'重复文件名数: {duplicate_count}')
    print(f'重复类名数: {duplicate_class_count}')
    print(f'平均每模块文件数: {total_files // len(modules) if modules else 0}')

    # 质量评估
    quality_score = 100

    # 重复文件名扣分
    if duplicate_count > 10:
        quality_score -= min(10, duplicate_count - 10)

    # 重复类名扣分
    if duplicate_class_count > 50:
        quality_score -= min(15, (duplicate_class_count - 50) // 5)

    # 代码组织问题扣分
    quality_score -= len(issues) * 2

    # 导入问题扣分
    quality_score -= len(import_issues) * 5

    quality_score = max(0, quality_score)

    print(f'\\n🏆 代码质量评分: {quality_score}/100')

    if quality_score >= 90:
        print('评价: 优秀 - 代码组织良好，冗余可控')
    elif quality_score >= 80:
        print('评价: 良好 - 代码组织基本合理，需要小幅优化')
    elif quality_score >= 70:
        print('评价: 一般 - 存在较多代码组织问题')
    else:
        print('评价: 需要改进 - 代码组织和冗余问题严重')

    # 优化建议
    print('\\n💡 优化建议:')
    suggestions = []

    if duplicate_count > 20:
        suggestions.append('减少重复文件名，考虑重构目录结构')

    if duplicate_class_count > 100:
        suggestions.append('整合重复类名，实现代码复用')

    if issues:
        suggestions.extend(issues)

    if import_issues:
        suggestions.extend(import_issues)

    if not suggestions:
        suggestions.append('代码组织和冗余情况良好，继续保持')

    for i, suggestion in enumerate(suggestions, 1):
        print(f'{i}. {suggestion}')


def main():
    """主函数"""
    print('🚀 基础设施层代码审查复核')
    print('=' * 40)

    # 执行各项分析
    duplicate_count = analyze_duplicate_files()
    modules = analyze_module_structure()
    issues = analyze_code_organization_issues(modules, duplicate_count)
    duplicate_class_count = analyze_code_redundancy()
    import_issues = analyze_import_structure()

    # 生成最终报告
    generate_final_report(modules, duplicate_count, duplicate_class_count, issues, import_issues)

    print('\\n✅ 代码审查复核完成')


if __name__ == "__main__":
    main()

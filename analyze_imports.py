"""
分析基础设施层导入语句规范
"""

import os
from pathlib import Path


def analyze_imports():
    """分析导入语句规范"""
    infra_dir = Path('src/infrastructure')

    print('🔍 分析基础设施层导入语句规范...')
    print('=' * 40)

    # 统计导入语句
    import_stats = {
        'from_import': 0,  # from module import name
        'direct_import': 0,  # import module
        'relative_import': 0,  # from .module import name
        'wildcard_import': 0,  # from module import *
        'long_imports': 0,  # 过长的导入语句
    }

    # 分析导入问题
    issues = []

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()

                    for line_num, line in enumerate(lines[:50]):  # 只检查前50行
                        line = line.strip()

                        if line.startswith('import '):
                            import_stats['direct_import'] += 1

                            # 检查导入语句长度
                            if len(line) > 100:
                                import_stats['long_imports'] += 1
                                rel_path = str(file_path.relative_to(infra_dir))
                                issues.append(f'过长导入: {rel_path}:{line_num+1}')

                        elif line.startswith('from '):
                            import_stats['from_import'] += 1

                            # 检查相对导入
                            if ' from .' in line or line.startswith('from .'):
                                import_stats['relative_import'] += 1

                            # 检查通配符导入
                            if ' import *' in line:
                                import_stats['wildcard_import'] += 1
                                rel_path = str(file_path.relative_to(infra_dir))
                                issues.append(f'通配符导入: {rel_path}:{line_num+1}')

                            # 检查导入语句长度
                            if len(line) > 100:
                                import_stats['long_imports'] += 1
                                rel_path = str(file_path.relative_to(infra_dir))
                                issues.append(f'过长导入: {rel_path}:{line_num+1}')

                except Exception as e:
                    rel_path = str(file_path.relative_to(infra_dir))
                    issues.append(f'读取文件错误 {rel_path}: {e}')

    print('📊 导入语句统计:')
    print(f'  from...import 语句: {import_stats["from_import"]}')
    print(f'  import 语句: {import_stats["direct_import"]}')
    print(f'  相对导入: {import_stats["relative_import"]}')
    print(f'  通配符导入: {import_stats["wildcard_import"]}')
    print(f'  过长导入: {import_stats["long_imports"]}')

    print('\n⚠️  发现的问题:')
    if issues:
        for issue in issues[:10]:  # 只显示前10个问题
            print(f'  - {issue}')
        if len(issues) > 10:
            remaining = len(issues) - 10
            print(f'  ... 还有 {remaining} 个问题')
    else:
        print('  ✅ 未发现明显问题')

    total_imports = import_stats['from_import'] + import_stats['direct_import']
    print(f'\n总导入语句数: {total_imports}')

    return import_stats, issues


def suggest_optimizations(import_stats, issues):
    """提供优化建议"""
    print('\n💡 导入优化建议:')
    print('=' * 20)

    suggestions = []

    # 分析导入类型分布
    total = import_stats['from_import'] + import_stats['direct_import']
    if total > 0:
        from_ratio = import_stats['from_import'] / total * 100
        print(f'  from...import比例: {from_ratio:.1f}%')
        if from_ratio < 60:
            suggestions.append('考虑使用更多的from...import语句来减少import语句的数量')
        elif from_ratio > 90:
            suggestions.append('from...import使用过多，考虑适当使用直接import')

    # 检查问题
    if import_stats['wildcard_import'] > 0:
        suggestions.append(f'消除 {import_stats["wildcard_import"]} 个通配符导入，提高代码可维护性')

    if import_stats['long_imports'] > 0:
        suggestions.append(f'简化 {import_stats["long_imports"]} 个过长的导入语句')

    if import_stats['relative_import'] > 0:
        suggestions.append(f'检查 {import_stats["relative_import"]} 个相对导入的必要性')

    # 一般性建议
    suggestions.extend([
        '统一导入语句的排序：标准库 -> 第三方库 -> 本地模块',
        '将相关导入分组，用空行分隔',
        '避免循环导入，通过重构或延迟导入解决',
        '使用__all__列表控制模块的公共接口'
    ])

    for i, suggestion in enumerate(suggestions, 1):
        print(f'{i}. {suggestion}')

    return suggestions


def main():
    """主函数"""
    print("🚀 开始分析导入语句规范")
    print("=" * 30)

    try:
        # 分析导入
        import_stats, issues = analyze_imports()

        # 提供优化建议
        suggestions = suggest_optimizations(import_stats, issues)

        print("\n✅ 导入语句分析完成！")
        print(f"发现 {len(issues)} 个潜在问题")

    except Exception as e:
        print(f"\n❌ 分析过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()

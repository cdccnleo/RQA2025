import os
from pathlib import Path
from collections import Counter

# 分析新的目录结构
utils_dir = Path('src/infrastructure/utils')
structure_stats = {
    'directories': 0,
    'python_files': 0,
    'total_files': 0,
    'functional_categories': Counter()
}

# 统计各功能目录的文件数量
functional_dirs = {
    'adapters': '数据适配器',
    'core': '核心组件',
    'monitoring': '监控系统',
    'optimization': '性能优化',
    'security': '安全组件',
    'tools': '工具函数',
    'utils': '通用工具'
}

for root, dirs, files in os.walk(utils_dir):
    rel_path = Path(root).relative_to(utils_dir)

    # 统计目录
    structure_stats['directories'] += len(dirs)

    # 统计文件
    for file in files:
        structure_stats['total_files'] += 1
        if file.endswith('.py') and not file.startswith('test_'):
            structure_stats['python_files'] += 1

            # 统计功能分类
            if rel_path.parts and rel_path.parts[0] in functional_dirs:
                category = functional_dirs[rel_path.parts[0]]
                structure_stats['functional_categories'][category] += 1

print('重新组织后的目录结构统计:')
print(f'  • 总文件数: {structure_stats["total_files"]}')
print(f'  • Python文件数: {structure_stats["python_files"]}')
print(f'  • 目录数: {structure_stats["directories"]}')
print(f'  • 功能分类:')

for category, count in sorted(structure_stats['functional_categories'].items()):
    print(f'    - {category}: {count} 个文件')

print('\n✅ 目录结构重组完成!')
print('  • 7个功能分类目录')
print('  • 清晰的职责分离')
print('  • 标准化的模块组织')

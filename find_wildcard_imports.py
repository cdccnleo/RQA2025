"""
查找和分析通配符导入
"""

import os
import re
from pathlib import Path


def find_wildcard_imports():
    """查找所有通配符导入"""
    infra_dir = Path('src/infrastructure')

    print('🔍 查找所有通配符导入...')
    print('=' * 30)

    wildcard_imports = []

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
                            rel_path = str(file_path.relative_to(infra_dir))
                            wildcard_imports.append({
                                'file': rel_path,
                                'line': line_num + 1,
                                'import': line
                            })

                except Exception as e:
                    print(f'读取文件 {file_path} 时出错: {e}')

    print(f'找到 {len(wildcard_imports)} 个通配符导入:')
    print()

    for i, item in enumerate(wildcard_imports, 1):
        print(f'{i:2d}. {item["file"]}:{item["line"]}')
        print(f'    {item["import"]}')
        print()

    if wildcard_imports:
        print('🎯 优化策略:')
        print('1. 将通配符导入替换为显式导入')
        print('2. 只导入实际使用的类和函数')
        print('3. 保持代码的可读性和维护性')

        # 保存结果到文件
        with open('wildcard_imports_report.txt', 'w', encoding='utf-8') as f:
            f.write('通配符导入报告\n')
            f.write('=' * 30 + '\n\n')
            f.write(f'找到 {len(wildcard_imports)} 个通配符导入:\n\n')

            for i, item in enumerate(wildcard_imports, 1):
                f.write(f'{i:2d}. {item["file"]}:{item["line"]}\n')
                f.write(f'    {item["import"]}\n\n')

            f.write('优化策略:\n')
            f.write('1. 将通配符导入替换为显式导入\n')
            f.write('2. 只导入实际使用的类和函数\n')
            f.write('3. 保持代码的可读性和维护性\n')

        print(f'\n📄 详细报告已保存到: wildcard_imports_report.txt')
    else:
        print('✅ 未发现通配符导入')

    return wildcard_imports


def analyze_wildcard_usage(wildcard_imports):
    """分析通配符导入的使用情况"""
    if not wildcard_imports:
        return

    print('\n🔍 分析通配符导入的使用情况...')

    for item in wildcard_imports:
        file_path = Path('src/infrastructure') / item['file']

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取导入的模块名
            import_match = re.match(r'from\s+([^\s]+)\s+import\s+\*', item['import'])
            if import_match:
                module_name = import_match.group(1)

                # 查找文件中对该模块的使用
                # 这是一个简化的分析，实际使用需要更复杂的AST分析
                print(f'\n📂 分析 {item["file"]} 中的 {module_name} 使用情况:')

                # 查找可能的类使用
                class_pattern = r'\b([A-Z]\w+)\s*\('
                classes = re.findall(class_pattern, content)

                if classes:
                    unique_classes = list(set(classes))
                    print(f'  可能的类使用: {unique_classes[:5]}...' if len(
                        unique_classes) > 5 else f'  可能的类使用: {unique_classes}')
                else:
                    print('  未发现明显的类使用')

        except Exception as e:
            print(f'分析文件 {file_path} 时出错: {e}')


if __name__ == "__main__":
    wildcard_imports = find_wildcard_imports()
    analyze_wildcard_usage(wildcard_imports)

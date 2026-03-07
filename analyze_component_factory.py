"""
ComponentFactory重复类名分析工具
"""

import os
from pathlib import Path


def analyze_component_factory():
    infra_dir = Path('src/infrastructure')

    print('ComponentFactory重复类名详细分析')
    print('=' * 50)

    component_factory_locations = []

    for root, dirs, files in os.walk(infra_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        lines = content.split('\n')

                    for line_num, line in enumerate(lines):
                        if 'class ComponentFactory' in line:
                            rel_path = str(file_path.relative_to(infra_dir))
                            component_factory_locations.append({
                                'file': rel_path,
                                'line': line_num + 1,
                                'class_line': line.strip()
                            })

                            # 提取类定义的几行内容
                            start_line = max(0, line_num - 2)
                            end_line = min(len(lines), line_num + 10)
                            context = lines[start_line:end_line]
                            component_factory_locations[-1]['context'] = context

                except Exception as e:
                    continue

    print(f'找到 {len(component_factory_locations)} 个ComponentFactory类定义')

    for i, loc in enumerate(component_factory_locations, 1):
        print(f'\n{i}. {loc["file"]}:{loc["line"]}')
        print(f'   {loc["class_line"]}')

        # 显示上下文
        context = loc.get('context', [])
        for j, ctx_line in enumerate(context):
            marker = '>>>' if j == 2 else '   '  # 标记类定义行
            print(f'{marker} {ctx_line.rstrip()}')


if __name__ == "__main__":
    analyze_component_factory()

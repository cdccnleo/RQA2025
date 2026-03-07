#!/usr/bin/env python3
"""
最终修复脚本
专门修复math_utils.py中的函数定义错误
"""

import re


def fix_math_utils():
    with open('src/infrastructure/utils/math_utils.py', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 修复函数定义错误模式
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 检查是否是函数定义后跟错误参数行
        if re.match(r'def \w+\([^)]*,?\):?$', line.strip()) and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line.startswith(')') and next_line.endswith(','):
                # 这是一个错误的参数行
                # 找到完整的参数列表
                param_lines = []
                j = i + 1
                while j < len(lines) and lines[j].strip().endswith(','):
                    param_lines.append(lines[j].strip()[:-1])  # 去掉逗号
                    j += 1

                if j < len(lines) and lines[j].strip().startswith('"""'):
                    # 找到了docstring，修复函数定义
                    func_name = line.split('(')[0].replace('def ', '')
                    params = ', '.join(param_lines[:-1]) + param_lines[-1] if param_lines else ''

                    fixed_lines.append(f'def {func_name}({params}):')
                    i = j  # 跳到docstring
                    continue

        fixed_lines.append(line)
        i += 1

    # 写回文件
    with open('src/infrastructure/utils/math_utils.py', 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))

    print('✅ 已修复 math_utils.py 的函数定义错误')


if __name__ == "__main__":
    fix_math_utils()

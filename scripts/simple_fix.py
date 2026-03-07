#!/usr/bin/env python3
"""
简单修复脚本
"""

import re


def simple_fix():
    with open('src/infrastructure/utils/math_utils.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复函数定义中的多行参数问题
    content = re.sub(r'def (\w+)\(([^)]*),?\):\s*\n\s*\)([^,]*),\s*\n\s*([^)]*)\):\s*\n\s*"""([^"]*)"""',
                     r'def \1(\2, \3, \4):\n    """\5"""', content)

    # 写回文件
    with open('src/infrastructure/utils/math_utils.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print('✅ 简单修复完成')


if __name__ == "__main__":
    simple_fix()

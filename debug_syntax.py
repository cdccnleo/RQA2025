#!/usr/bin/env python3
"""
调试语法错误的脚本
"""

import ast


def debug_syntax(file_path):
    """调试语法错误"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 逐行检查语法
    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        if not stripped:
            continue

        # 尝试编译到这一行
        try:
            code = ''.join(lines[:i])
            ast.parse(code)
        except SyntaxError as e:
            print(f"第{i}行语法错误: {e}")
            print(f"第{i}行内容: {repr(line)}")
            break


if __name__ == "__main__":
    debug_syntax('tests/unit/infrastructure/base/test_additional_infrastructure.py')

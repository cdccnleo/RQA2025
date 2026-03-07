#!/usr/bin/env python3
"""
批量修复standard_interfaces.py中的语法错误
"""

import re


def fix_standard_interfaces():
    """批量修复standard_interfaces.py中的语法错误"""
    file_path = "src/infrastructure/interfaces/standard_interfaces.py"

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 修复字典初始化错误
    content = re.sub(
        r'return \{\}\s*\n(\s+)(\w+):',
        r'return {\n\1\2:',
        content,
        flags=re.MULTILINE
    )

    # 修复多行字典错误
    content = re.sub(
        r'(\w+):\s*([^,\n]+),\s*\n(\s+)(\w+):\s*([^,\n]+),\s*\n(\s+)(\w+):\s*([^,\n]+)',
        r'\1: \2,\n\3\4: \5,\n\6\7: \8',
        content,
        flags=re.MULTILINE
    )

    # 修复缺少的右括号
    content = re.sub(
        r'(\w+):\s*([^,\n]+)\s*\n\s*$',
        r'\1: \2\n}',
        content,
        flags=re.MULTILINE
    )

    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print('✅ 已批量修复 standard_interfaces.py 的语法错误')


def validate_fix():
    """验证修复结果"""
    try:
        with open('src/infrastructure/interfaces/standard_interfaces.py', 'r', encoding='utf-8') as f:
            compile(f.read(), 'standard_interfaces.py', 'exec')
        print('✅ standard_interfaces.py 语法正确')
        return True
    except SyntaxError as e:
        print(f'❌ 语法错误: {e}')
        return False


if __name__ == "__main__":
    fix_standard_interfaces()
    validate_fix()

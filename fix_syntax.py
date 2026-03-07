#!/usr/bin/env python3
"""修复测试文件语法错误的脚本"""

import re

def fix_syntax_errors():
    """修复utils测试文件的所有语法错误"""

    # Read the file
    with open('tests/unit/infrastructure/utils/test_utils_enhancement.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern 1: Fix lines ending with ") where they should end with "
    # Find patterns like: "key": "value")
    content = re.sub(r'(\"\w+\":\s*\"[^\"]*\")\)', r'\1', content)

    # Pattern 2: Fix dictionary endings where } is followed by ) incorrectly
    # Find patterns like: }\s*\}\s*\)
    content = re.sub(r'(\}\s*)\}\s*\)', r'\1)', content)

    # Write back
    with open('tests/unit/infrastructure/utils/test_utils_enhancement.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print("语法错误修复完成")

if __name__ == "__main__":
    fix_syntax_errors()

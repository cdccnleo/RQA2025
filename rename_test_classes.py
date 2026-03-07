#!/usr/bin/env python3
"""
将测试辅助类重命名为Mock开头，避免pytest警告
"""
import os
import re
from pathlib import Path

def rename_test_classes(content):
    """重命名测试辅助类"""

    # 查找所有以Test开头但有__init__的类
    pattern = r'class\s+(Test\w+)\s*\([^)]*\):\s*\n(?:\s*""".*?""")?\s*\n\s*def\s+__init__'

    # 首先收集所有需要重命名的类
    class_mappings = {}
    for match in re.finditer(pattern, content, flags=re.MULTILINE | re.DOTALL):
        old_name = match.group(1)
        new_name = re.sub(r'^Test', 'Mock', old_name)
        class_mappings[old_name] = new_name

    # 替换类定义
    def replace_class(match):
        class_name = match.group(1)
        new_name = re.sub(r'^Test', 'Mock', class_name)
        return match.group(0).replace(class_name, new_name)

    content = re.sub(pattern, replace_class, content, flags=re.MULTILINE | re.DOTALL)

    # 替换类名在代码中的其他引用
    for old_name, new_name in class_mappings.items():
        # 替换类名引用，但避免替换字符串中的类名
        content = re.sub(rf'\b{re.escape(old_name)}\b(?![\"\'])', new_name, content)

    return content

def main():
    test_dir = "tests/unit/infrastructure"
    processed_count = 0

    for py_file in Path(test_dir).rglob('*.py'):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            content = rename_test_classes(content)

            if content != original_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                processed_count += 1
                print(f"已处理: {py_file}")

        except Exception as e:
            print(f"错误处理 {py_file}: {e}")

    print(f"总共处理了 {processed_count} 个文件")

if __name__ == "__main__":
    main()

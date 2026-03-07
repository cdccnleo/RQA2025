#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

def fix_excessive_relative_imports():
    """
    修复过度嵌套的相对导入，将其转换为绝对导入
    """
    fixed_count = 0

    for root, dirs, files in os.walk('src'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    original_content = content

                    # 将 ......infrastructure. 替换为 src.infrastructure.
                    content = re.sub(r'from \.\.\.\.\.\.+infrastructure\.', 'from src.infrastructure.', content)
                    content = re.sub(r'import \.\.\.\.\.\.+infrastructure\.', 'from src.infrastructure.', content)

                    # 将 ......core. 替换为 src.core.
                    content = re.sub(r'from \.\.\.\.\.\.+core\.', 'from src.core.', content)
                    content = re.sub(r'import \.\.\.\.\.\.+core\.', 'from src.core.', content)

                    # 将 ......utils. 替换为 src.infrastructure.utils.
                    content = re.sub(r'from \.\.\.\.\.\.+utils\.', 'from src.infrastructure.utils.', content)
                    content = re.sub(r'import \.\.\.\.\.\.+utils\.', 'from src.infrastructure.utils.', content)

                    # 处理其他可能的过度嵌套导入（通用规则）
                    # 将 4个或更多点的导入替换为 src. 前缀
                    def replace_excessive_dots(match):
                        dots_and_module = match.group(0)
                        module_part = match.group(1)
                        return f'from src.{module_part}'

                    content = re.sub(r'from \.\.\.\.\.\.+([^\'\"\s]+)', replace_excessive_dots, content)

                    def replace_excessive_import_dots(match):
                        module_part = match.group(1)
                        return f'from src.{module_part} import *'

                    content = re.sub(r'import \.\.\.\.\.\.+([^\'\"\s]+)', replace_excessive_import_dots, content)

                    if content != original_content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        fixed_count += 1

                except Exception as e:
                    print(f'Error processing {file_path}: {e}')

    return fixed_count

if __name__ == '__main__':
    count = fix_excessive_relative_imports()
    print(f'修复了 {count} 个文件的过度嵌套相对导入')

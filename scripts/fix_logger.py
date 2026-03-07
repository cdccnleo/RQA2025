#!/usr/bin/env python3
"""
修复logger.py中的语法错误
"""

import re


def fix_logger_file():
    with open('src/infrastructure/utils/helpers/logger.py', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 修复RotatingFileHandler调用
    content = re.sub(
        r'logging\.handlers\.RotatingFileHandler\(\)\s*\n(\s+)([^,]+),\s*\n(\s+)([^,]+),\s*\n(\s+)([^)]+)\)',
        r'logging.handlers.RotatingFileHandler(\n\1\2,\n\2\4,\n\2\6)',
        content
    )

    # 修复Formatter调用
    content = re.sub(
        r'logging\.Formatter\(\)\s*\n(\s+)(\'[^\']+\')',
        r'logging.Formatter(\n\1\2)',
        content
    )

    # 写回文件
    with open('src/infrastructure/utils/helpers/logger.py', 'w', encoding='utf-8') as f:
        f.write(content)

    print('✅ 已修复 logger.py 的语法错误')


if __name__ == "__main__":
    fix_logger_file()

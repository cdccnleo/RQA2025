#!/usr/bin/env python3
"""
基础语法修复脚本
修复最基本的语法问题
"""

import os
import sys
import ast
import glob
import logging

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fix_syntax_errors():
    """修复语法错误"""
    test_layers = [
        'tests/unit/infrastructure',
        'tests/unit/features',
        'tests/unit/ml',
        'tests/unit/trading',
        'tests/unit/risk',
        'tests/unit/core'
    ]

    fixed_count = 0

    for layer_path in test_layers:
        if not os.path.exists(layer_path):
            continue

        test_files = glob.glob(f'{layer_path}/**/*comprehensive.py', recursive=True)

        for test_file in test_files:
            if fix_single_file(test_file):
                fixed_count += 1

    return fixed_count


def fix_single_file(file_path: str) -> bool:
    """修复单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 修复字符串转义问题
        content = fix_string_escapes(content)

        # 验证语法
        try:
            ast.parse(content)
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False

        if syntax_ok and content != original_content:
            # 备份并写入
            backup_path = f"{file_path}.basic_backup"
            if not os.path.exists(backup_path):
                with open(backup_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True

    except Exception as e:
        logger.error(f"修复文件 {file_path} 时出错: {e}")

    return False


def fix_string_escapes(content: str) -> str:
    """修复字符串转义问题"""
    # 修复常见的转义问题
    fixes = [
        ('print("\\n===', 'print("\\n=='),
        ('===\\n")', '===")'),
        ('print(f"\\n===', 'print(f"\\n=='),
        ('f"\\n===', 'f"\\n=='),
        ('\\\\n===', '\\n==='),
        ('===\\\\n")', '===")'),
    ]

    for old, new in fixes:
        content = content.replace(old, new)

    return content


if __name__ == "__main__":
    print("🔧 开始基础语法修复...")
    fixed = fix_syntax_errors()
    print(f"✅ 修复完成! 修复了 {fixed} 个文件")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查Python文件语法错误的脚本
"""

import os
import ast
import sys
from pathlib import Path


def check_python_file(file_path):
    """检查单个Python文件的语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {e}"


def find_syntax_errors(src_dir):
    """查找src目录下所有有语法错误的Python文件"""
    syntax_errors = []

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                is_valid, error_msg = check_python_file(file_path)
                if not is_valid:
                    syntax_errors.append((file_path, error_msg))

    return syntax_errors


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent.parent
    src_dir = project_root / "src"

    print(f"🔍 检查 {src_dir} 目录下的Python文件语法...")

    syntax_errors = find_syntax_errors(str(src_dir))

    if syntax_errors:
        print(f"\n❌ 发现 {len(syntax_errors)} 个语法错误的文件:")
        for file_path, error_msg in syntax_errors:
            print(f"   {file_path}")
            print(f"      {error_msg}")

        print(f"\n📝 建议将这些文件添加到 .coveragerc 的 omit 列表中:")
        for file_path, _ in syntax_errors:
            relative_path = os.path.relpath(file_path, str(project_root))
            print(f"   {relative_path}")
    else:
        print("✅ 所有Python文件语法正确！")

    return len(syntax_errors)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

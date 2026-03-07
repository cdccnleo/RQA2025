#!/usr/bin/env python3
"""
检查基础设施层测试文件的语法错误
"""

import glob
import ast


def check_syntax(file_path):
    """检查单个文件的语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 尝试解析AST
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Other error: {e}"


def main():
    """主函数"""
    print("检查基础设施层测试文件的语法错误...")

    # 获取所有基础设施层测试文件
    test_files = []
    patterns = [
        'tests/unit/infrastructure/**/*.py',
        'tests/unit/infrastructure/*.py'
    ]

    for pattern in patterns:
        test_files.extend(glob.glob(pattern, recursive=True))

    print(f"找到 {len(test_files)} 个测试文件")

    syntax_errors = []

    for file_path in test_files:
        is_valid, error_msg = check_syntax(file_path)
        if not is_valid:
            syntax_errors.append((file_path, error_msg))
            print(f"❌ {file_path}: {error_msg}")
        else:
            print(f"✅ {file_path}: OK")

    print(f"\n总结:")
    print(f"总文件数: {len(test_files)}")
    print(f"语法错误文件数: {len(syntax_errors)}")
    print(f"语法正确文件数: {len(test_files) - len(syntax_errors)}")

    if syntax_errors:
        print(f"\n需要修复的文件:")
        for file_path, error in syntax_errors:
            print(f"  - {file_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
批量修复基础设施层测试文件中的导入路径问题
将 'from src.infrastructure' 改为 'from infrastructure'
"""

import re
import glob


def fix_imports_in_file(file_path):
    """修复单个文件中的导入路径"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修复导入路径
        original_content = content
        content = re.sub(r'from src\.infrastructure', 'from infrastructure', content)
        content = re.sub(r'import src\.infrastructure', 'import infrastructure', content)

        # 如果内容有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ 已修复: {file_path}")
            return True
        else:
            print(f"- 无需修复: {file_path}")
            return False
    except Exception as e:
        print(f"✗ 修复失败: {file_path} - {e}")
        return False


def main():
    """主函数"""
    # 查找所有基础设施层测试文件
    test_patterns = [
        "tests/unit/infrastructure/**/*.py",
        "tests/infrastructure/**/*.py"
    ]

    all_files = []
    for pattern in test_patterns:
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)

    print(f"找到 {len(all_files)} 个测试文件")

    # 修复每个文件
    fixed_count = 0
    for file_path in all_files:
        if fix_imports_in_file(file_path):
            fixed_count += 1

    print(f"\n修复完成！共修复了 {fixed_count} 个文件")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
批量修复Python文件编码声明
"""

from pathlib import Path


def fix_encoding_declaration(file_path):
    """为Python文件添加编码声明"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否已有编码声明
        if '# -*- coding: utf-8 -*-' in content:
            return False

        # 检查文件是否有shebang行
        lines = content.split('\n')
        if lines and lines[0].startswith('#!/'):
            # 在shebang后添加编码声明
            lines.insert(1, '# -*- coding: utf-8 -*-')
        else:
            # 在文件开头添加编码声明
            lines.insert(0, '# -*- coding: utf-8 -*-')

        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        return True

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False


def main():
    """主函数"""
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "tests" / "unit"

    if not test_dir.exists():
        print(f"测试目录不存在: {test_dir}")
        return

    fixed_count = 0
    total_count = 0

    # 递归处理所有Python文件
    for py_file in test_dir.rglob("*.py"):
        total_count += 1
        if fix_encoding_declaration(py_file):
            fixed_count += 1
            print(f"已修复: {py_file}")

    print(f"\n修复完成:")
    print(f"总文件数: {total_count}")
    print(f"修复文件数: {fixed_count}")


if __name__ == "__main__":
    main()

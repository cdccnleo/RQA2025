#!/usr/bin/env python3
"""
交易层测试缩进修复脚本

修复交易层测试文件中的缩进错误
"""

from pathlib import Path


def fix_indentation_issues(file_path: Path) -> bool:
    """修复单个文件的缩进问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        fixed_lines = []
        modified = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            # 查找函数定义后没有正确缩进的文档字符串
            if (stripped.startswith('"""') and stripped.endswith('"""') and
                    i > 0 and lines[i-1].strip().startswith('def ')):
                # 检查前一行是否是函数定义
                prev_line = lines[i-1].strip()
                if prev_line.startswith('def ') and not prev_line.endswith(':'):
                    # 函数定义没有冒号，添加冒号
                    lines[i-1] = prev_line + ':'
                    modified = True
                # 检查缩进是否正确
                if not line.startswith('    '):
                    lines[i] = '    ' + line
                    modified = True
            elif (stripped.startswith('"""') and stripped.endswith('"""') and
                  i > 0 and 'def ' in lines[i-1]):
                # 处理更复杂的情况
                if not line.startswith('    ') and not line.startswith('\t'):
                    lines[i] = '    ' + line
                    modified = True

        if modified:
            new_content = '\n'.join(lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True

        return False

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False


def fix_all_trading_tests():
    """修复所有交易层测试文件"""
    test_dir = Path("tests/unit/trading")
    fixed_count = 0

    if not test_dir.exists():
        print(f"测试目录不存在: {test_dir}")
        return 0

    for file_path in test_dir.rglob("test_*.py"):
        if fix_indentation_issues(file_path):
            print(f"已修复: {file_path}")
            fixed_count += 1

    print(f"\n修复完成: {fixed_count} 个文件已修复")
    return fixed_count


if __name__ == "__main__":
    fix_all_trading_tests()

#!/usr/bin/env python3
"""
缓存测试导入语法修复脚本

修复缓存测试文件中的导入语法错误
"""

import re
from pathlib import Path


def fix_import_syntax_issues(file_path: Path) -> bool:
    """修复单个文件的导入语法问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        lines = content.split('\n')
        fixed_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 处理except ImportError没有对应try的问题
            if stripped.startswith('except ImportError:'):
                # 查找前面的导入语句并添加try
                if i > 0 and (lines[i-1].strip().startswith('from ') or lines[i-1].strip().startswith('import ')):
                    # 在前一行添加try
                    lines.insert(i-1, 'try:')
                    i += 1  # 跳过新添加的行
                elif i > 1 and (lines[i-2].strip().startswith('from ') or lines[i-2].strip().startswith('import ')):
                    # 在前两行添加try
                    lines.insert(i-2, 'try:')
                    i += 1
                elif i > 2 and (lines[i-3].strip().startswith('from ') or lines[i-3].strip().startswith('import ')):
                    # 在前三行添加try
                    lines.insert(i-3, 'try:')
                    i += 1

            # 处理导入语句后的缩进问题
            elif (stripped.startswith('from ') or stripped.startswith('import ')) and i + 1 < len(lines):
                next_line = lines[i+1].strip()
                if next_line.startswith('except ImportError:') and not line.startswith('try:'):
                    # 添加try语句
                    lines.insert(i, 'try:')
                    i += 1

            # 处理pytest.main缩进问题
            elif stripped.startswith('pytest.main(['):
                # 确保pytest.main正确缩进
                if not line.startswith('    ') and not line.startswith('\t'):
                    line = '    ' + stripped

            # 处理if __name__ == "__main__"缩进问题
            elif stripped.startswith('if __name__ == "__main__":'):
                if line.startswith(' ') or line.startswith('\t'):
                    # 移除缩进
                    line = stripped

            # 处理函数定义问题
            elif stripped.startswith('def ') and not stripped.endswith(':'):
                line = stripped + ':'

            fixed_lines.append(line)
            i += 1

        # 重新组合内容
        new_content = '\n'.join(fixed_lines)

        # 修复一些常见的语法错误模式
        new_content = re.sub(r'^\s*except ImportError:\s*$',
                             r'    except ImportError:\n        pass', new_content, flags=re.MULTILINE)

        # 修复多行导入的缩进
        lines = new_content.split('\n')
        in_try_block = False
        for i, line in enumerate(lines):
            if line.strip().startswith('try:'):
                in_try_block = True
            elif line.strip().startswith('except ImportError:'):
                in_try_block = False
            elif in_try_block and (line.strip().startswith('from ') or line.strip().startswith('import ')):
                # 确保try块内的导入语句正确缩进
                if not line.startswith('    ') and not line.startswith('\t'):
                    lines[i] = '    ' + line.strip()

        new_content = '\n'.join(lines)

        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True

        return False

    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False


def fix_all_cache_tests():
    """修复所有缓存测试文件"""
    cache_test_dir = Path("tests/unit/infrastructure/cache")
    fixed_count = 0

    if not cache_test_dir.exists():
        print(f"缓存测试目录不存在: {cache_test_dir}")
        return 0

    for file_path in cache_test_dir.rglob("test_*.py"):
        if fix_import_syntax_issues(file_path):
            print(f"已修复: {file_path}")
            fixed_count += 1

    print(f"\n修复完成: {fixed_count} 个文件已修复")
    return fixed_count


if __name__ == "__main__":
    fix_all_cache_tests()

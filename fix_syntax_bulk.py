#!/usr/bin/env python3
"""
批量修复Python语法错误脚本
主要修复常见的语法错误模式
"""

import os
import re
import ast
from pathlib import Path

def fix_incomplete_try_except(content):
    """修复不完整的try-except语句"""
    # 模式1: try: \nexcept ImportError as e:
    pattern1 = re.compile(r'(\s+)try:\s*\n(\s+)except\s+ImportError\s+as\s+e\s*:', re.MULTILINE)
    content = pattern1.sub(r'\1try:\n\1    # 导入检查\n\1    import sys\n\2except ImportError as e:', content)

    # 模式2: try: \n        except ImportError as e:
    pattern2 = re.compile(r'(\s+)try:\s*\n(\s+)except\s+ImportError\s+as\s+e\s*:', re.MULTILINE)
    content = pattern2.sub(r'\1try:\n\1    # 导入检查\n\1    import os\n\2except ImportError as e:', content)

    return content

def fix_indentation_errors(content):
    """修复缩进错误"""
    lines = content.split('\n')
    fixed_lines = []
    indent_stack = []

    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if not stripped:
            fixed_lines.append(line)
            continue

        # 计算当前行的缩进
        indent = len(line) - len(line.lstrip())

        # 检查是否是函数定义或类定义
        if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except ', 'finally:', 'else:')):
            # 这些语句开始新的缩进块
            pass
        elif indent > 0 and indent_stack and indent < indent_stack[-1]:
            # 缩进减少，可能是错误
            # 检查上一行是否正确结束了块
            if i > 0:
                prev_line = lines[i-1].rstrip()
                if prev_line and not prev_line.endswith(':') and not prev_line.startswith(' ' * indent):
                    # 可能需要修复缩进
                    if stripped.startswith(('def ', 'class ')):
                        # 函数或类定义应该在模块级别
                        line = stripped

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def fix_docstring_in_imports(content):
    """修复导入语句中的文档字符串错误"""
    # 模式: from xxx import (\n"""\n
    pattern = re.compile(r'(from\s+[\w.]+\s+import\s*\(\s*)\n\s*"""', re.MULTILINE)
    content = pattern.sub(r'\1', content)

    # 移除多余的文档字符串
    lines = content.split('\n')
    in_import_block = False
    result_lines = []

    for line in lines:
        if line.strip().startswith('from ') and 'import (' in line:
            in_import_block = True
            result_lines.append(line)
        elif in_import_block and line.strip().startswith('"""'):
            # 跳过文档字符串
            continue
        elif in_import_block and line.strip().endswith(')'):
            in_import_block = False
            result_lines.append(line)
        elif in_import_block and line.strip().startswith('from '):
            # 新的导入语句
            result_lines.append(line)
        else:
            result_lines.append(line)

    return '\n'.join(result_lines)

def process_file(file_path):
    """处理单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 应用各种修复
        content = fix_incomplete_try_except(content)
        content = fix_indentation_errors(content)
        content = fix_docstring_in_imports(content)

        # 检查语法是否正确
        try:
            ast.parse(content)
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ 修复: {file_path}")
                return True
            else:
                print(f"ℹ️ 无需修复: {file_path}")
                return False
        except SyntaxError as e:
            print(f"❌ 语法错误仍存在: {file_path} - {e}")
            return False

    except Exception as e:
        print(f"❌ 处理错误: {file_path} - {e}")
        return False

def main():
    """主函数"""
    src_dir = Path('src')
    total_files = 0
    fixed_files = 0

    print("🔧 开始批量修复Python语法错误...")
    print("=" * 50)

    for py_file in src_dir.rglob('*.py'):
        total_files += 1
        if process_file(py_file):
            fixed_files += 1

    print("=" * 50)
    print(f"📊 处理完成: {total_files} 个文件")
    print(f"🔧 修复文件: {fixed_files} 个")
    print(f"修复率: {(fixed_files/total_files*100):.1f}%")
if __name__ == '__main__':
    main()

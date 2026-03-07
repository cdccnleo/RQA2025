#!/usr/bin/env python3
"""
智能语法错误修复脚本
专门修复infrastructure层的复杂语法错误
"""

import re
from pathlib import Path
from typing import List, Tuple


class SmartSyntaxFixer:
    """智能语法错误修复器"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def fix_all_errors(self) -> List[Tuple[str, int, str]]:
        """修复所有语法错误"""
        fixed_files = []

        for py_file in self.root_dir.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                original_content = content

                # 应用各种修复
                content = self._fix_import_syntax(content)
                content = self._fix_string_literals(content)
                content = self._fix_parentheses_matching(content)
                content = self._fix_indentation_issues(content)
                content = self._fix_colon_issues(content)
                content = self._fix_decorator_syntax(content)

                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixed_files.append((str(py_file), 0, "Fixed advanced syntax issues"))

            except Exception as e:
                fixed_files.append((str(py_file), 0, f"Error: {str(e)}"))

        return fixed_files

    def _fix_import_syntax(self, content: str) -> str:
        """修复导入语法错误"""
        # 修复 from module import.submodule 形式的错误
        content = re.sub(r'from\s+(\w+(?:\.\w+)*)\s+import\.(\w+)',
                         r'from \1.\2', content)

        # 修复 from module import, 形式的错误
        content = re.sub(r'from\s+(\w+(?:\.\w+)*)\s+import\s*,',
                         r'from \1 import ', content)

        return content

    def _fix_string_literals(self, content: str) -> str:
        """修复字符串字面量错误"""
        lines = content.split('\n')
        fixed_lines = []

        i = 0
        while i < len(lines):
            line = lines[i]

            # 检查未闭合的字符串
            if self._has_unclosed_string(line):
                # 尝试找到下一行的结束引号
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # 如果下一行有引号，合并这两行
                    if '"' in next_line or "'" in next_line:
                        # 简单合并，移除换行符
                        line = line.rstrip() + ' ' + next_line.lstrip()
                        i += 1  # 跳过下一行
                    else:
                        # 添加结束引号
                        if line.count('"') % 2 == 1:
                            line += '"'
                        elif line.count("'") % 2 == 1:
                            line += "'"

            fixed_lines.append(line)
            i += 1

        return '\n'.join(fixed_lines)

    def _has_unclosed_string(self, line: str) -> bool:
        """检查行是否有未闭合的字符串"""
        # 简单的检查：引号数量是否为奇数
        double_quotes = line.count('"')
        single_quotes = line.count("'")

        # 排除转义的引号
        escaped_double = line.count('\\"')
        escaped_single = line.count("\\'")

        return (double_quotes - escaped_double) % 2 == 1 or (single_quotes - escaped_single) % 2 == 1

    def _fix_parentheses_matching(self, content: str) -> str:
        """修复括号匹配错误"""
        # 修复方括号不匹配
        content = self._fix_bracket_matching(content, '[', ']')

        # 修复圆括号不匹配
        content = self._fix_bracket_matching(content, '(', ')')

        # 修复花括号不匹配
        content = self._fix_bracket_matching(content, '{', '}')

        return content

    def _fix_bracket_matching(self, content: str, open_bracket: str, close_bracket: str) -> str:
        """修复特定类型括号的匹配"""
        lines = content.split('\n')
        fixed_lines = []

        for line in lines:
            open_count = line.count(open_bracket)
            close_count = line.count(close_bracket)

            if open_count > close_count:
                # 添加缺失的闭合括号
                line += close_bracket * (open_count - close_count)
            elif close_count > open_count:
                # 移除多余的闭合括号
                for _ in range(close_count - open_count):
                    line = line.rsplit(close_bracket, 1)[0]

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_indentation_issues(self, content: str) -> str:
        """修复缩进问题"""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # 如果是空行或注释，保持原样
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue

            # 检查是否需要缩进
            if (stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except ', 'finally:', 'with ', 'else:', 'elif ')) or
                    stripped.endswith(':')):
                # 这是一个需要后续缩进的块
                fixed_lines.append(line)

                # 检查下一行是否需要缩进
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    next_stripped = next_line.strip()

                    if (next_stripped and
                        not next_stripped.startswith('#') and
                        not next_stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except ', 'finally:', 'with ')) and
                            not next_line.startswith('    ')):  # 没有4个空格缩进
                        # 添加适当的缩进
                        current_indent = len(line) - len(line.lstrip())
                        new_indent = current_indent + 4
                        lines[i + 1] = ' ' * new_indent + next_stripped
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_colon_issues(self, content: str) -> str:
        """修复冒号相关问题"""
        # 修复缺少冒号的语句
        patterns = [
            (r'^\s*(def\s+\w+\s*\([^)]*\))\s*$', r'\1:'),
            (r'^\s*(class\s+\w+(?:\([^)]*\)))\s*$', r'\1:'),
            (r'^\s*(if\s+[^:]+)\s*$', r'\1:'),
            (r'^\s*(for\s+[^:]+)\s*$', r'\1:'),
            (r'^\s*(while\s+[^:]+)\s*$', r'\1:'),
            (r'^\s*(try)\s*$', r'\1:'),
            (r'^\s*(except)\s*$', r'\1:'),
            (r'^\s*(finally)\s*$', r'\1:'),
            (r'^\s*(with\s+[^:]+)\s*$', r'\1:'),
            (r'^\s*(else)\s*$', r'\1:'),
            (r'^\s*(elif\s+[^:]+)\s*$', r'\1:')
        ]

        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

        return content

    def _fix_decorator_syntax(self, content: str) -> str:
        """修复装饰器语法"""
        # 修复 @dataclass 后面直接跟其他语句的问题
        content = re.sub(r'(@\w+.*?)(\n\s*)([^\s@])', r'\1\2\n\3', content)

        return content


def main():
    """主函数"""
    print("🔧 智能语法错误修复工具")
    print("=" * 50)

    fixer = SmartSyntaxFixer('src/infrastructure')

    print("📝 正在修复infrastructure层高级语法错误...")
    fixed_files = fixer.fix_all_errors()

    print(f"\n✅ 修复完成! 共处理 {len(fixed_files)} 个文件")

    if fixed_files:
        print("\n修复的文件列表:")
        for file_path, line_num, message in fixed_files:
            status = "✅" if "Error" not in message else "❌"
            print(f"  {status} {Path(file_path).name}: {message}")

    # 重新检查语法错误
    print("\n🔍 重新检查语法错误...")
    import subprocess
    import sys

    try:
        result = subprocess.run([
            sys.executable, 'scripts/testing/syntax_checker.py'
        ], capture_output=True, text=True, cwd='.', encoding='utf-8', errors='replace')

        print("语法检查结果:")
        if result.stdout:
            print(result.stdout)

        if result.stderr:
            print("错误信息:")
            print(result.stderr)

    except Exception as e:
        print(f"❌ 重新检查失败: {e}")


if __name__ == "__main__":
    main()

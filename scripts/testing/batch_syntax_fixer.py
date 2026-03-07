#!/usr/bin/env python3
"""
批量语法错误修复脚本
自动修复infrastructure层的常见语法错误
"""

import re
from pathlib import Path
from typing import List, Tuple


class BatchSyntaxFixer:
    """批量语法错误修复器"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def fix_indented_block_errors(self) -> List[Tuple[str, int, str]]:
        """修复缩进块错误"""
        fixed_files = []

        for py_file in self.root_dir.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                original_content = content

                # 修复1: class定义后的docstring缩进问题
                content = self._fix_class_docstring_indentation(content)

                # 修复2: 方法定义后的空行缩进问题
                content = self._fix_method_indentation(content)

                # 修复3: 字符串引号不匹配问题
                content = self._fix_string_quotes(content)

                # 修复4: 导入语句语法错误
                content = self._fix_import_statements(content)

                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixed_files.append((str(py_file), 0, "Fixed indentation and syntax issues"))

            except Exception as e:
                fixed_files.append((str(py_file), 0, f"Error: {str(e)}"))

        return fixed_files

    def _fix_class_docstring_indentation(self, content: str) -> str:
        """修复class定义后的docstring缩进问题"""
        # 匹配 class ClassName: 后面直接跟 """ 的模式
        pattern = r'(class\s+\w+:\s*$)\s*("""[^"]*""")'
        replacement = r'\1\n    \2'

        return re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)

    def _fix_method_indentation(self, content: str) -> str:
        """修复方法定义后的缩进问题"""
        # 匹配 def method_name(self): 后面直接跟代码块的模式
        lines = content.split('\n')
        fixed_lines = []
        i = 0

        while i < len(lines):
            line = lines[i]
            fixed_lines.append(line)

            # 检查是否是方法或函数定义
            if re.match(r'^\s*def\s+\w+\s*\([^)]*\)\s*:\s*$', line):
                # 查找下一行，如果不是空行或注释，就添加缩进
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1

                if j < len(lines) and lines[j].strip() and not lines[j].strip().startswith('#'):
                    # 添加适当的缩进
                    indent = len(line) - len(line.lstrip()) + 4
                    lines[j] = ' ' * indent + lines[j].lstrip()

            i += 1

        return '\n'.join(fixed_lines)

    def _fix_string_quotes(self, content: str) -> str:
        """修复字符串引号不匹配问题"""
        # 修复单引号内的双引号问题
        content = re.sub(r"'([^']*)\"([^']*)'([^']*)\"", r"'\1'\2'\3'", content)

        # 修复未闭合的字符串
        lines = content.split('\n')
        in_string = False
        quote_char = None

        for i, line in enumerate(lines):
            if not in_string:
                # 检查行开始是否有未闭合的字符串
                if line.count('"') % 2 == 1 or line.count("'") % 2 == 1:
                    in_string = True
                    quote_char = '"' if line.count('"') % 2 == 1 else "'"
            else:
                # 在字符串中，查找结束引号
                if quote_char in line:
                    in_string = False
                    quote_char = None

        return content

    def _fix_import_statements(self, content: str) -> str:
        """修复导入语句语法错误"""
        # 移除重复的导入
        lines = content.split('\n')
        imports = set()
        fixed_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                if stripped not in imports:
                    imports.add(stripped)
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)


def main():
    """主函数"""
    print("🔧 批量语法错误修复工具")
    print("=" * 50)

    fixer = BatchSyntaxFixer('src/infrastructure')

    print("📝 正在修复infrastructure层语法错误...")
    fixed_files = fixer.fix_indented_block_errors()

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
        ], capture_output=True, text=True, cwd='.')

        print("语法检查结果:")
        print(result.stdout)

        if result.stderr:
            print("错误信息:")
            print(result.stderr)

    except Exception as e:
        print(f"❌ 重新检查失败: {e}")


if __name__ == "__main__":
    main()

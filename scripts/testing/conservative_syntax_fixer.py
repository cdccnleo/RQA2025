#!/usr/bin/env python3
"""
保守语法错误修复脚本
专门修复字符串字面量和最基本的语法问题
"""

from pathlib import Path
from typing import List, Tuple


class ConservativeSyntaxFixer:
    """保守语法错误修复器"""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def fix_basic_errors(self) -> List[Tuple[str, int, str]]:
        """修复基本语法错误"""
        fixed_files = []

        for py_file in self.root_dir.rglob('*.py'):
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                original_content = content

                # 应用保守的修复
                content = self._fix_string_literals_only(content)

                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixed_files.append((str(py_file), 0, "Fixed string literals"))

            except Exception as e:
                fixed_files.append((str(py_file), 0, f"Error: {str(e)}"))

        return fixed_files

    def _fix_string_literals_only(self, content: str) -> str:
        """只修复字符串字面量问题"""
        lines = content.split('\n')
        fixed_lines = []
        in_multiline_string = False
        string_delimiter = None

        for i, line in enumerate(lines):
            # 检查是否在多行字符串中
            if in_multiline_string:
                # 查找结束引号
                if string_delimiter in line:
                    in_multiline_string = False
                    string_delimiter = None
                fixed_lines.append(line)
                continue

            # 检查单行字符串问题
            stripped = line.strip()

            # 跳过注释和空行
            if not stripped or stripped.startswith('#'):
                fixed_lines.append(line)
                continue

            # 检查是否有未闭合的字符串
            double_quotes = line.count('"')
            single_quotes = line.count("'")

            # 排除转义的引号
            escaped_double = line.count('\\"')
            escaped_single = line.count("\\'")

            unclosed_double = (double_quotes - escaped_double) % 2 == 1
            unclosed_single = (single_quotes - escaped_single) % 2 == 1

            if unclosed_double or unclosed_single:
                # 尝试修复
                if unclosed_double and '"' in line:
                    # 检查是否是多行字符串开始
                    if line.rstrip().endswith('"""') or line.rstrip().endswith("'''"):
                        # 这可能是多行字符串的开始
                        if '"""' in line:
                            string_delimiter = '"""'
                        else:
                            string_delimiter = "'''"
                        in_multiline_string = True
                    else:
                        # 单行字符串，添加结束引号
                        line += '"'
                elif unclosed_single and "'" in line:
                    # 单行字符串，添加结束引号
                    line += "'"
                # 如果引号不匹配，可能是其他问题，保持原样

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)


def main():
    """主函数"""
    print("🔧 保守语法错误修复工具")
    print("=" * 50)

    fixer = ConservativeSyntaxFixer('src/infrastructure')

    print("📝 正在修复infrastructure层字符串字面量错误...")
    fixed_files = fixer.fix_basic_errors()

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

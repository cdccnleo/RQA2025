#!/usr/bin/env python3
"""
基础设施层语法错误修复工具

检查并修复基础设施层Python文件中的语法错误
"""

import os
import re
import ast
import sys
from pathlib import Path
from typing import List, Tuple


class SyntaxFixer:
    """语法错误修复器"""

    def __init__(self, infrastructure_path: str):
        self.infrastructure_path = Path(infrastructure_path)
        self.fixed_files = []
        self.errors_found = []

    def find_python_files(self) -> List[Path]:
        """查找所有Python文件"""
        python_files = []
        for root, dirs, files in os.walk(self.infrastructure_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        return python_files

    def check_syntax(self, file_path: Path) -> Tuple[bool, str]:
        """检查Python文件语法"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 尝试解析AST
            ast.parse(content, filename=str(file_path))
            return True, ""

        except SyntaxError as e:
            return False, f"SyntaxError: {e}"
        except IndentationError as e:
            return False, f"IndentationError: {e}"
        except Exception as e:
            return False, f"Error: {e}"

    def fix_indentation_errors(self, file_path: Path) -> bool:
        """修复缩进错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            fixed_lines = []
            in_dict_literal = False
            dict_start_indent = 0

            for i, line in enumerate(lines):
                stripped = line.strip()
                indent = len(line) - len(line.lstrip())

                # 检测字典字面量开始
                if '=' in stripped and '{' in stripped and not in_dict_literal:
                    # 查找return语句后的字典开始
                    if stripped.startswith('return {') or ('return' in stripped and '{' in stripped):
                        in_dict_literal = True
                        dict_start_indent = indent
                        # 修复字典字面量的开始
                        if 'return {}' in stripped:
                            # 找到下一行的缩进
                            next_indent = dict_start_indent + 4
                            fixed_lines.append(line)
                            continue

                # 处理字典字面量内部
                if in_dict_literal and stripped.startswith('"') and ':' in stripped:
                    # 字典键值对应该增加缩进
                    expected_indent = dict_start_indent + 4
                    if indent != expected_indent:
                        line = ' ' * expected_indent + line.lstrip()

                # 检测字典字面量结束
                if in_dict_literal and stripped.startswith('}'):
                    in_dict_literal = False

                fixed_lines.append(line)

            fixed_content = '\n'.join(fixed_lines)

            # 写入修复后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            return True

        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
            return False

    def fix_syntax_errors(self, file_path: Path) -> bool:
        """修复语法错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 修复常见的字典字面量问题
            # 模式1: return {} 后面直接跟字典内容
            pattern1 = r'return \{\}\s*\n\s*(".*?":\s*.*?,?\s*\n\s*)*\}'
            def replacement1(m): return self._fix_return_dict(m.group(0))

            content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE | re.DOTALL)

            # 写入修复后的内容
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return True

        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
            return False

    def _fix_return_dict(self, match: str) -> str:
        """修复return语句后的字典"""
        lines = match.strip().split('\n')
        if len(lines) < 2:
            return match

        # 找到return语句
        return_line = lines[0]
        indent = len(return_line) - len(return_line.lstrip())

        # 修复字典内容缩进
        fixed_lines = [return_line]
        for line in lines[1:]:
            if line.strip():
                # 确保字典内容正确缩进
                if line.strip().startswith('"') and ':' in line:
                    fixed_lines.append(' ' * (indent + 4) + line.strip())
                else:
                    fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def process_files(self):
        """处理所有Python文件"""
        python_files = self.find_python_files()
        print(f"Found {len(python_files)} Python files in infrastructure layer")

        for file_path in python_files:
            print(f"Checking {file_path.name}...")

            # 检查语法
            is_valid, error = self.check_syntax(file_path)

            if not is_valid:
                print(f"  ❌ Syntax error in {file_path.name}: {error}")
                self.errors_found.append((file_path, error))

                # 尝试修复
                if "IndentationError" in error:
                    if self.fix_indentation_errors(file_path):
                        print(f"  ✅ Fixed indentation in {file_path.name}")
                        self.fixed_files.append(file_path)
                    else:
                        print(f"  ❌ Failed to fix {file_path.name}")
                else:
                    if self.fix_syntax_errors(file_path):
                        print(f"  ✅ Fixed syntax in {file_path.name}")
                        self.fixed_files.append(file_path)
                    else:
                        print(f"  ❌ Failed to fix {file_path.name}")
            else:
                print(f"  ✅ {file_path.name} is syntactically correct")

    def report(self):
        """生成报告"""
        print(f"\n{'='*50}")
        print("INFRASTRUCTURE SYNTAX FIX REPORT")
        print(f"{'='*50}")
        print(f"Files processed: {len(self.find_python_files())}")
        print(f"Files with syntax errors: {len(self.errors_found)}")
        print(f"Files successfully fixed: {len(self.fixed_files)}")

        if self.errors_found:
            print(f"\nFiles with remaining errors ({len(self.errors_found)}):")
            for file_path, error in self.errors_found:
                print(f"  - {file_path.name}: {error}")

        if self.fixed_files:
            print(f"\nSuccessfully fixed files ({len(self.fixed_files)}):")
            for file_path in self.fixed_files:
                print(f"  - {file_path.name}")


def main():
    """主函数"""
    infrastructure_path = "src/infrastructure"

    if not os.path.exists(infrastructure_path):
        print(f"Error: {infrastructure_path} not found")
        sys.exit(1)

    fixer = SyntaxFixer(infrastructure_path)
    fixer.process_files()
    fixer.report()


if __name__ == "__main__":
    main()

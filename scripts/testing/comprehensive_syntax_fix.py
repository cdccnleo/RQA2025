#!/usr/bin/env python3
"""
综合语法错误修复脚本

专门修复SyntaxError和IndentationError
"""

import os
import re
import ast
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveSyntaxFixer:
    """综合语法修复器"""

    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.fixed_files = []
        self.errors = []

    def find_all_test_files(self) -> list:
        """查找所有测试文件"""
        test_files = []
        if self.test_dir.exists():
            for file_path in self.test_dir.rglob("test_*.py"):
                test_files.append(file_path)
        return test_files

    def check_syntax(self, file_path: Path) -> tuple:
        """检查文件语法"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            ast.parse(content)
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, f"其他错误: {str(e)}"

    def fix_syntax_errors(self, file_path: Path) -> bool:
        """修复语法错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 修复常见的语法错误模式
            lines = content.split('\n')
            fixed_lines = []

            for i, line in enumerate(lines):
                # 修复不完整的导入语句
                if line.strip().startswith('from ') and not line.strip().endswith('import'):
                    # 查找下一行是否包含import
                    if i + 1 < len(lines) and 'import' in lines[i + 1]:
                        line = line.rstrip() + ' ' + lines[i + 1].strip()
                        lines[i + 1] = ''  # 标记下一行为已处理

                # 修复不完整的函数定义
                elif line.strip().startswith('def ') and not line.strip().endswith(':'):
                    line = line.rstrip() + ':'

                # 修复不完整的类定义
                elif line.strip().startswith('class ') and not line.strip().endswith(':'):
                    line = line.rstrip() + ':'

                # 修复不完整的if语句
                elif re.match(r'^\s*if\s+.*[^:]$', line):
                    line = line.rstrip() + ':'

                # 修复不完整的for语句
                elif re.match(r'^\s*for\s+.*[^:]$', line):
                    line = line.rstrip() + ':'

                # 修复不完整的while语句
                elif re.match(r'^\s*while\s+.*[^:]$', line):
                    line = line.rstrip() + ':'

                # 修复不完整的try语句
                elif re.match(r'^\s*try\s*[^:]$', line):
                    line = line.rstrip() + ':'

                # 修复字符串引号不匹配
                line = self.fix_string_quotes(line)

                # 修复括号不匹配
                line = self.fix_parentheses(line)

                if lines[i] != '':  # 只添加未被标记为删除的行
                    fixed_lines.append(line)

            # 重新组合内容
            new_content = '\n'.join(fixed_lines)

            # 修复缩进问题
            new_content = self.fix_indentation(new_content)

            # 移除多余的空行
            new_content = re.sub(r'\n\s*\n\s*\n', '\n\n', new_content)

            if new_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                self.fixed_files.append(str(file_path))
                logger.info(f"已修复语法错误: {file_path}")
                return True

            return False

        except Exception as e:
            logger.error(f"处理文件 {file_path} 时出错: {e}")
            self.errors.append((str(file_path), str(e)))
            return False

    def fix_string_quotes(self, line: str) -> str:
        """修复字符串引号问题"""
        # 修复混合引号
        line = re.sub(r"'''", '"""', line)  # 统一使用三双引号
        line = re.sub(r"'''", '"""', line)

        # 修复未闭合的字符串
        single_quotes = line.count("'") - line.count("\\'")
        double_quotes = line.count('"') - line.count('\\"')

        if single_quotes % 2 != 0:
            # 有未闭合的单引号，尝试修复
            if "'''" in line or '"""' in line:
                pass  # 三引号字符串，可能是多行
            else:
                line = line.rstrip() + "'"

        if double_quotes % 2 != 0:
            # 有未闭合的双引号，尝试修复
            if "'''" in line or '"""' in line:
                pass  # 三引号字符串，可能是多行
            else:
                line = line.rstrip() + '"'

        return line

    def fix_parentheses(self, line: str) -> str:
        """修复括号匹配问题"""
        # 计算各种括号的数量
        round_open = line.count('(') - line.count('\\(')
        round_close = line.count(')') - line.count('\\)')
        square_open = line.count('[') - line.count('\\[')
        square_close = line.count(']') - line.count('\\]')
        curly_open = line.count('{') - line.count('\\{')
        curly_close = line.count('}') - line.count('\\}')

        # 修复圆括号
        if round_open > round_close:
            line = line.rstrip() + ')' * (round_open - round_close)
        elif round_close > round_open:
            # 多余的闭括号，移除末尾的
            line = line.rstrip()
            while line.endswith(')') and round_close > round_open:
                line = line[:-1]
                round_close -= 1

        # 修复方括号
        if square_open > square_close:
            line = line.rstrip() + ']' * (square_open - square_close)
        elif square_close > square_open:
            line = line.rstrip()
            while line.endswith(']') and square_close > square_open:
                line = line[:-1]
                square_close -= 1

        # 修复花括号
        if curly_open > curly_close:
            line = line.rstrip() + '}' * (curly_open - curly_close)
        elif curly_close > curly_open:
            line = line.rstrip()
            while line.endswith('}') and curly_close > curly_open:
                line = line[:-1]
                curly_close -= 1

        return line

    def fix_indentation(self, content: str) -> str:
        """修复缩进问题"""
        lines = content.split('\n')
        fixed_lines = []
        indent_stack = []

        for line in lines:
            stripped = line.strip()

            # 跳过空行
            if not stripped:
                fixed_lines.append('')
                continue

            # 计算应该的缩进级别
            current_indent = len(line) - len(line.lstrip())

            # 处理缩进关键字
            if any(stripped.startswith(keyword) for keyword in ['def ', 'class ', 'if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except ', 'finally:', 'with ']):
                # 这些关键字通常需要增加缩进级别
                if indent_stack:
                    expected_indent = indent_stack[-1] + 4
                else:
                    expected_indent = 0
                indent_stack.append(expected_indent)
            elif stripped.startswith(('except ', 'finally:', 'else:', 'elif ')):
                # 这些是同一级别的
                if indent_stack:
                    expected_indent = indent_stack[-1]
                else:
                    expected_indent = 0
            else:
                # 普通语句
                if indent_stack:
                    expected_indent = indent_stack[-1]
                else:
                    expected_indent = 0

            # 处理去缩进关键字
            if stripped.startswith(('return', 'break', 'continue', 'pass', 'raise')):
                if indent_stack:
                    indent_stack.pop()

            # 修复缩进
            if current_indent != expected_indent:
                line = ' ' * expected_indent + stripped

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def fix_all_files(self) -> tuple:
        """修复所有文件"""
        test_files = self.find_all_test_files()
        logger.info(f"找到 {len(test_files)} 个测试文件")

        fixed_count = 0
        syntax_errors_before = 0
        syntax_errors_after = 0

        for file_path in test_files:
            # 检查修复前的语法
            syntax_ok_before, _ = self.check_syntax(file_path)
            if not syntax_ok_before:
                syntax_errors_before += 1

            # 尝试修复
            if self.fix_syntax_errors(file_path):
                fixed_count += 1

            # 检查修复后的语法
            syntax_ok_after, _ = self.check_syntax(file_path)
            if not syntax_ok_after:
                syntax_errors_after += 1

        logger.info(f"修复完成: {fixed_count} 个文件已修复")
        logger.info(f"修复前语法错误: {syntax_errors_before} 个")
        logger.info(f"修复后语法错误: {syntax_errors_after} 个")

        return fixed_count, len(self.errors)


def main():
    """主函数"""
    test_dir = "tests/unit/infrastructure"

    if not os.path.exists(test_dir):
        logger.error(f"测试目录不存在: {test_dir}")
        return

    fixer = ComprehensiveSyntaxFixer(test_dir)
    fixed_count, error_count = fixer.fix_all_files()

    print("\n综合语法修复总结:")
    print(f"- 处理的文件数: {len(fixer.find_all_test_files())}")
    print(f"- 修复的文件数: {fixed_count}")
    print(f"- 出错的文件数: {error_count}")

    if fixer.fixed_files:
        print("\n修复的文件列表 (前10个):")
        for file in fixer.fixed_files[:10]:
            print(f"  - {file}")
        if len(fixer.fixed_files) > 10:
            print(f"  ... 还有 {len(fixer.fixed_files) - 10} 个文件")

    if fixer.errors:
        print("\n出错的文件列表:")
        for file, error in fixer.errors[:5]:
            print(f"  - {file}: {error}")
        if len(fixer.errors) > 5:
            print(f"  ... 还有 {len(fixer.errors) - 5} 个错误")


if __name__ == "__main__":
    main()

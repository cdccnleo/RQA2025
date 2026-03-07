#!/usr/bin/env python3
"""
代码质量语法错误修复工具

专门修复阻止代码分析的语法错误，包括：
- 缩进错误修复
- 字符串字面量错误修复
- 导入错误修复
- 基本语法问题修复
"""

from src.utils.logger import get_logger
import os
import sys
import re
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


logger = get_logger(__name__)


class SyntaxErrorFixer:
    """语法错误修复器"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)

    def find_syntax_errors(self, max_files: int = 100) -> List[Dict[str, Any]]:
        """查找语法错误"""
        syntax_errors = []

        # 只检查src目录
        src_path = self.project_path / "src"
        if not src_path.exists():
            return syntax_errors

        count = 0
        for py_file in src_path.rglob('*.py'):
            if count >= max_files:
                break

            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                # 尝试编译检查语法
                compile(content, str(py_file), 'exec')

            except SyntaxError as e:
                error_info = {
                    'file_path': str(py_file.relative_to(self.project_path)),
                    'error_type': 'syntax_error',
                    'error_message': str(e),
                    'line_number': e.lineno,
                    'offset': e.offset,
                    'text': e.text
                }
                syntax_errors.append(error_info)
                count += 1

            except Exception as e:
                # 其他类型的错误
                error_info = {
                    'file_path': str(py_file.relative_to(self.project_path)),
                    'error_type': 'other_error',
                    'error_message': str(e)
                }
                syntax_errors.append(error_info)
                count += 1

        return syntax_errors

    def fix_syntax_error(self, file_path: Path, error_info: Dict[str, Any]) -> bool:
        """修复单个语法错误"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            lines = content.splitlines()
            fixed_content = content
            changes_made = []

            error_type = error_info.get('error_type')
            line_number = error_info.get('line_number', 0)
            error_message = error_info.get('error_message', '')

            # 根据错误类型进行修复
            if 'unexpected indent' in error_message:
                fixed_content, change = self._fix_indentation_error(lines, line_number)
                if change:
                    changes_made.append(change)

            elif 'unindent does not match' in error_message:
                fixed_content, change = self._fix_unindent_error(lines, line_number)
                if change:
                    changes_made.append(change)

            elif 'EOL while scanning string literal' in error_message:
                fixed_content, change = self._fix_string_literal_error(lines, line_number)
                if change:
                    changes_made.append(change)

            elif 'EOF while scanning' in error_message:
                fixed_content, change = self._fix_eof_string_error(lines, line_number)
                if change:
                    changes_made.append(change)

            elif 'expected an indented block' in error_message:
                fixed_content, change = self._fix_missing_indentation(lines, line_number)
                if change:
                    changes_made.append(change)

            elif 'positional argument follows keyword argument' in error_message:
                fixed_content, change = self._fix_argument_order(lines, line_number)
                if change:
                    changes_made.append(change)

            elif 'invalid syntax' in error_message:
                fixed_content, change = self._fix_invalid_syntax(lines, line_number, error_message)
                if change:
                    changes_made.append(change)

            # 验证修复是否成功
            if fixed_content != content:
                try:
                    compile(fixed_content, str(file_path), 'exec')

                    # 保存修复后的文件
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)

                    logger.info(f"语法错误修复成功: {file_path} - {', '.join(changes_made)}")
                    return True

                except SyntaxError as e:
                    logger.warning(f"修复后仍有语法错误: {file_path} - {e}")
                    return False

            return False

        except Exception as e:
            logger.error(f"修复语法错误失败: {file_path} - {e}")
            return False

    def _fix_indentation_error(self, lines: List[str], line_number: int) -> Tuple[str, str]:
        """修复缩进错误"""
        if line_number <= 0 or line_number > len(lines):
            return '\n'.join(lines), ""

        line_idx = line_number - 1
        current_line = lines[line_idx]

        # 检查前一行来确定正确的缩进级别
        prev_indent = 0
        for i in range(line_idx - 1, -1, -1):
            prev_line = lines[i].strip()
            if prev_line and not prev_line.startswith('#'):
                prev_indent = len(lines[i]) - len(lines[i].lstrip())
                break

        # 检查当前行的缩进
        current_indent = len(current_line) - len(current_line.lstrip())

        # 如果当前行缩进太多，减少缩进
        if current_indent > prev_indent + 4:
            # 减少缩进到合适级别
            new_indent = prev_indent + 4
            new_line = ' ' * new_indent + current_line.lstrip()
            lines[line_idx] = new_line
            return '\n'.join(lines), f"修复缩进错误: 行{line_number}缩进从{current_indent}调整为{new_indent}"

        return '\n'.join(lines), ""

    def _fix_unindent_error(self, lines: List[str], line_number: int) -> Tuple[str, str]:
        """修复取消缩进错误"""
        if line_number <= 0 or line_number > len(lines):
            return '\n'.join(lines), ""

        line_idx = line_number - 1
        current_line = lines[line_idx]

        # 检查前几行的缩进模式
        indents = []
        for i in range(max(0, line_idx - 10), line_idx):
            if lines[i].strip() and not lines[i].strip().startswith('#'):
                indent = len(lines[i]) - len(lines[i].lstrip())
                indents.append(indent)

        if indents:
            # 使用最常见的缩进级别
            target_indent = max(set(indents), key=indents.count)
            current_indent = len(current_line) - len(current_line.lstrip())

            if current_indent != target_indent:
                new_line = ' ' * target_indent + current_line.lstrip()
                lines[line_idx] = new_line
                return '\n'.join(lines), f"修复取消缩进错误: 行{line_number}缩进调整为{target_indent}"

        return '\n'.join(lines), ""

    def _fix_string_literal_error(self, lines: List[str], line_number: int) -> Tuple[str, str]:
        """修复字符串字面量错误"""
        if line_number <= 0 or line_number > len(lines):
            return '\n'.join(lines), ""

        line_idx = line_number - 1
        current_line = lines[line_idx]

        # 检查未闭合的字符串
        if current_line.count('"') % 2 != 0 or current_line.count("'") % 2 != 0:
            # 尝试添加缺失的引号
            if current_line.count('"') % 2 != 0:
                lines[line_idx] = current_line + '"'
                return '\n'.join(lines), f"修复字符串字面量错误: 行{line_number}添加缺失的双引号"
            elif current_line.count("'") % 2 != 0:
                lines[line_idx] = current_line + "'"
                return '\n'.join(lines), f"修复字符串字面量错误: 行{line_number}添加缺失的单引号"

        return '\n'.join(lines), ""

    def _fix_eof_string_error(self, lines: List[str], line_number: int) -> Tuple[str, str]:
        """修复EOF字符串错误"""
        if line_number <= 0 or line_number > len(lines):
            return '\n'.join(lines), ""

        line_idx = line_number - 1

        # 检查三引号字符串是否完整
        content = '\n'.join(lines)
        triple_quotes = ['"""', "'''"]

        for quote in triple_quotes:
            quote_count = content.count(quote)
            if quote_count % 2 != 0:
                # 添加缺失的三引号
                lines.append(quote)
                return '\n'.join(lines), f"修复EOF字符串错误: 添加缺失的三引号{quote}"

        return '\n'.join(lines), ""

    def _fix_missing_indentation(self, lines: List[str], line_number: int) -> Tuple[str, str]:
        """修复缺少缩进错误"""
        if line_number <= 0 or line_number > len(lines):
            return '\n'.join(lines), ""

        line_idx = line_number - 1
        current_line = lines[line_idx]

        # 检查前一行的缩进
        prev_indent = 0
        for i in range(line_idx - 1, -1, -1):
            if lines[i].strip():
                prev_indent = len(lines[i]) - len(lines[i].lstrip())
                break

        # 添加适当的缩进
        if current_line.strip():  # 非空行
            expected_indent = prev_indent + 4
            if not current_line.startswith(' ') or len(current_line) - len(current_line.lstrip()) < expected_indent:
                new_line = ' ' * expected_indent + current_line.lstrip()
                lines[line_idx] = new_line
                return '\n'.join(lines), f"修复缺少缩进错误: 行{line_number}添加{expected_indent}个空格缩进"

        return '\n'.join(lines), ""

    def _fix_argument_order(self, lines: List[str], line_number: int) -> Tuple[str, str]:
        """修复参数顺序错误"""
        if line_number <= 0 or line_number > len(lines):
            return '\n'.join(lines), ""

        line_idx = line_number - 1
        current_line = lines[line_idx]

        # 查找函数调用中的关键字参数和位置参数
        # 这是一个简单的修复，实际可能需要更复杂的逻辑
        if '=' in current_line and '(' in current_line:
            # 简单地将位置参数移到关键字参数之前
            # 这是一个临时修复，更复杂的逻辑需要AST分析
            pass

        return '\n'.join(lines), ""

    def _fix_invalid_syntax(self, lines: List[str], line_number: int, error_message: str) -> Tuple[str, str]:
        """修复无效语法错误"""
        if line_number <= 0 or line_number > len(lines):
            return '\n'.join(lines), ""

        line_idx = line_number - 1
        current_line = lines[line_idx]

        # 处理一些常见的无效语法错误
        if 'unexpected character after line continuation character' in error_message:
            # 移除行尾的反斜杠后的非法字符
            if current_line.rstrip().endswith('\\'):
                lines[line_idx] = current_line.rstrip()[:-1]
                return '\n'.join(lines), f"修复无效语法: 行{line_number}移除非法行继续字符"

        return '\n'.join(lines), ""

    def fix_all_syntax_errors(self, max_files: int = 50) -> Dict[str, Any]:
        """修复所有语法错误"""
        logger.info("开始语法错误修复...")

        # 查找语法错误
        syntax_errors = self.find_syntax_errors(max_files * 2)  # 多查找一些以防有重复

        # 限制处理数量
        errors_to_fix = syntax_errors[:max_files]

        logger.info(f"发现 {len(syntax_errors)} 个语法错误，准备修复前 {len(errors_to_fix)} 个")

        # 修复错误
        fixed_count = 0
        failed_count = 0

        for error_info in errors_to_fix:
            file_path = self.project_path / error_info['file_path']

            if self.fix_syntax_error(file_path, error_info):
                fixed_count += 1
            else:
                failed_count += 1

            if (fixed_count + failed_count) % 10 == 0:
                logger.info(
                    f"已处理 {fixed_count + failed_count} 个文件，成功 {fixed_count} 个，失败 {failed_count} 个")

        # 生成报告
        report = {
            'total_errors_found': len(syntax_errors),
            'errors_processed': len(errors_to_fix),
            'fixed_count': fixed_count,
            'failed_count': failed_count,
            'fix_rate': fixed_count / len(errors_to_fix) if errors_to_fix else 0,
            'error_types': self._categorize_errors(syntax_errors),
            'summary': {
                'phase': 'Phase 4C Week 2',
                'objective': '语法错误专项修复',
                'achievements': [
                    f'发现 {len(syntax_errors)} 个语法错误',
                    f'成功修复 {fixed_count} 个语法错误',
                    f'修复成功率 {fixed_count / len(errors_to_fix) * 100:.1f}%',
                    '为后续代码质量分析扫清障碍'
                ],
                'next_steps': [
                    '重新运行代码质量评估工具',
                    '继续进行类型提示和文档补全',
                    '扩大修复范围到更多文件',
                    '建立语法检查的自动化流程'
                ]
            }
        }

        logger.info("语法错误修复完成")
        return report

    def _categorize_errors(self, errors: List[Dict[str, Any]]) -> Dict[str, int]:
        """分类错误类型"""
        categories = {}

        for error in errors:
            error_type = error.get('error_type', 'unknown')
            categories[error_type] = categories.get(error_type, 0) + 1

        return categories

    def save_report(self, report: Dict[str, Any], output_file: str):
        """保存报告"""
        import json

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"语法修复报告已保存到: {output_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='代码质量语法错误修复工具')
    parser.add_argument('--max-files', type=int, default=50, help='最大处理文件数量 (默认: 50)')

    args = parser.parse_args()

    fixer = SyntaxErrorFixer(str(project_root))

    # 执行修复
    report = fixer.fix_all_syntax_errors(max_files=args.max_files)

    # 保存报告
    output_file = "code_quality_syntax_fix_report.json"
    fixer.save_report(report, output_file)

    # 打印摘要
    print("🔧 语法错误修复报告")
    print("=" * 60)
    print(f"🔍 发现错误: {report['total_errors_found']} 个")
    print(f"📁 处理文件: {report['errors_processed']} 个")
    print(f"✅ 修复成功: {report['fixed_count']} 个")
    print(f"❌ 修复失败: {report['failed_count']} 个")
    print(f"📊 修复率: {report['fix_rate']*100:.1f}%")

    print(f"\n📋 错误类型分布:")
    for error_type, count in report['error_types'].items():
        print(f"  • {error_type}: {count} 个")

    print("
🎯 修复成果: " for achievement in report['summary']['achievements']:
        print(f"  • {achievement}")

    print("
🚀 后续计划: " for next_step in report['summary']['next_steps']:
        print(f"  • {next_step}")

    print(f"\n📄 详细报告已保存: {output_file}")


if __name__ == "__main__":
    main()

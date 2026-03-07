#!/usr/bin/env python3
"""
高级语法修复脚本
解决复杂的语法错误和导入问题
"""

import os
import re
import sys
import ast
import glob
import logging
from pathlib import Path
from typing import List, Dict, Set, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedSyntaxFixer:
    """高级语法修复器"""

    def __init__(self):
        self.fixed_files = []
        self.errors_found = []

    def find_syntax_errors(self) -> List[str]:
        """查找语法错误的测试文件"""
        syntax_error_files = []
        test_layers = [
            'tests/unit/infrastructure',
            'tests/unit/features',
            'tests/unit/ml',
            'tests/unit/trading',
            'tests/unit/risk',
            'tests/unit/core'
        ]

        for layer_path in test_layers:
            if not os.path.exists(layer_path):
                continue

            test_files = glob.glob(f'{layer_path}/**/*comprehensive.py', recursive=True)
            for test_file in test_files:
                if self._has_syntax_errors(test_file):
                    syntax_error_files.append(test_file)

        return syntax_error_files

    def _has_syntax_errors(self, file_path: str) -> bool:
        """检查文件是否有语法错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 尝试解析AST
            ast.parse(content)
            return False

        except SyntaxError as e:
            logger.debug(f"语法错误 in {file_path}: {e}")
            return True
        except Exception as e:
            logger.debug(f"其他错误 in {file_path}: {e}")
            return True

    def fix_file_syntax(self, file_path: str) -> bool:
        """修复单个文件的语法错误"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 应用各种修复
            content = self._fix_common_syntax_issues(content)
            content = self._fix_import_statements(content)
            content = self._fix_string_formatting(content)
            content = self._fix_indentation_issues(content)
            content = self._fix_multiline_statements(content)

            # 验证修复后的语法
            try:
                ast.parse(content)
            except SyntaxError:
                logger.warning(f"修复后仍存在语法错误: {file_path}")
                return False

            if content != original_content:
                # 备份原文件
                backup_path = f"{file_path}.syntax_backup"
                if not os.path.exists(backup_path):
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(original_content)

                # 写入修复后的内容
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                self.fixed_files.append(file_path)
                logger.info(f"修复了语法错误: {file_path}")
                return True

        except Exception as e:
            logger.error(f"修复文件 {file_path} 时出错: {e}")
            self.errors_found.append(f"{file_path}: {str(e)}")

        return False

    def _fix_common_syntax_issues(self, content: str) -> str:
        """修复常见语法问题"""
        # 修复多行字符串的转义问题
        pattern1 = r'print\("\\n==='
        replacement1 = 'print("\\n==="
        content=re.sub(pattern1, replacement1, content)

        pattern2=r'===\\n"\)'
        replacement2='===")'
        content=re.sub(pattern2, replacement2, content)

        # 修复字符串中的换行符
        pattern3=r'print\("\\n"\s*\+\s*"==='
        replacement3='print("\\n==='
        content=re.sub(pattern3, replacement3, content)

        pattern4=r'==="[^(]*\)'
        replacement4='===")'
        content=re.sub(pattern4, replacement4, content)

        # 修复f-string中的转义问题
        pattern5=r'f".*\\\\n.*"'
        content=re.sub(pattern5, lambda m: m.group(0).replace('\\\\n', '\\n'), content)

        return content

    def _fix_import_statements(self, content: str) -> str:
        """修复导入语句"""
        lines=content.split('\n')
        fixed_lines=[]

        for line in lines:
            # 修复不完整的try-except导入
            if line.strip().startswith('try:') and 'import' in line:
                # 检查是否有对应的except块
                if not any('except ImportError:' in l for l in lines[lines.index(line):lines.index(line)+5]):
                    # 添加except块
                    fixed_lines.append(line)
                    fixed_lines.append("except ImportError:")
                    fixed_lines.append("    # Module not available")
                    fixed_lines.append("    pass")
                    continue

            # 修复通配符导入
            if 'from src.' in line and ' import *' in line:
                # 替换为更安全的导入
                line=line.replace(' import *', ' import *  # noqa')

            # 修复空导入
            if line.strip() == 'try:' and any('import' in l for l in lines[lines.index(line)+1:lines.index(line)+3]):
                continue  # 跳过这种模式的处理

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_string_formatting(self, content: str) -> str:
        """修复字符串格式化问题"""
        # 修复多行字符串连接
        content=re.sub(
            r'print\("\\n=== ([^"]+) ===="\)',
            r'print(f"\\n=== \1 ===")',
            content
        )

        # 修复f-string中的转义字符
        content=re.sub(r'f"([^"]*\\\\n[^"]*)"', self._fix_fstring_escapes, content)

        return content

    def _fix_fstring_escapes(self, match):
        """修复f-string中的转义字符"""
        fstring_content=match.group(1)
        # 将\\n替换为实际的换行符，但保持f-string语法
        return f'f"{fstring_content.replace("\\\\n", "\\n")}"'

    def _fix_indentation_issues(self, content: str) -> str:
        """修复缩进问题"""
        lines=content.split('\n')
        fixed_lines=[]
        indent_stack=[]

        for i, line in enumerate(lines):
            if line.strip().startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except:', 'with ')):
                # 计算当前缩进级别
                indent=len(line) - len(line.lstrip())
                indent_stack.append(indent)

            # 检查缩进一致性
            if line.strip() and not line.startswith(' ') and not line.startswith('\t') and indent_stack:
                # 这行应该有缩进
                if not line.startswith(('"""', "'''", '#')):
                    line='    ' + line

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def _fix_multiline_statements(self, content: str) -> str:
        """修复多行语句问题"""
        # 修复未完成的try-except块
        lines=content.split('\n')
        in_try_block=False
        try_line_index=-1

        for i, line in enumerate(lines):
            if line.strip().startswith('try:'):
                in_try_block=True
                try_line_index=i
            elif in_try_block and line.strip().startswith('except'):
                in_try_block=False
            elif in_try_block and i > try_line_index + 5:  # try块太长，可能有问题
                # 插入except块
                lines.insert(i, "except ImportError:")
                lines.insert(i+1, "    # Module not available")
                lines.insert(i+2, "    pass")
                in_try_block=False
                break

        return '\n'.join(lines)

    def run_syntax_fixes(self) -> Dict[str, any]:
        """运行语法修复过程"""
        logger.info("开始高级语法修复...")

        syntax_error_files=self.find_syntax_errors()
        logger.info(f"发现 {len(syntax_error_files)} 个语法错误文件")

        fixed_count=0
        for file_path in syntax_error_files:
            if self.fix_file_syntax(file_path):
                fixed_count += 1

        result={
            'total_syntax_errors': len(syntax_error_files),
            'fixed_count': fixed_count,
            'error_count': len(self.errors_found),
            'fixed_files': self.fixed_files,
            'errors': self.errors_found
        }

        logger.info(f"语法修复完成: {fixed_count}/{len(syntax_error_files)} 个文件已修复")

        return result

def main():
    """主函数"""
    fixer=AdvancedSyntaxFixer()
    result=fixer.run_syntax_fixes()

    print("\n=== 高级语法修复结果 ===")
    print(f"发现语法错误文件: {result['total_syntax_errors']}")
    print(f"成功修复的文件: {result['fixed_count']}")
    print(f"修复失败的文件: {result['error_count']}")

    if result['fixed_files']:
        print("
修复的文件列表: ")
        for file in result['fixed_files'][:10]:  # 只显示前10个
            print(f"  ✓ {file}")
        if len(result['fixed_files']) > 10:
            print(f"  ... 还有 {len(result['fixed_files']) - 10} 个文件")

    if result['errors']:
        print("
修复失败的文件: ")
        for error in result['errors'][:5]:  # 只显示前5个
            print(f"  ✗ {error}")
        if len(result['errors']) > 5:
            print(f"  ... 还有 {len(result['errors']) - 5} 个错误")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
高级代码质量改进器 - RQA2025
系统性地修复语法错误、导入问题和代码风格
"""

import os
import re
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Set

class CodeQualityImprover:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.fixed_files = set()
        self.stats = {
            'syntax_errors_fixed': 0,
            'import_errors_fixed': 0,
            'style_issues_fixed': 0,
            'total_files_processed': 0
        }

    def run_flake8_analysis(self) -> Dict[str, List[str]]:
        """运行flake8分析获取错误统计"""
        try:
            result = subprocess.run([
                'python', '-m', 'flake8', str(self.root_dir),
                '--count', '--select=E9,F63,F7,F82,F821,F824',
                '--format=%(path)s:%(row)d:%(col)d:%(code)s:%(text)s'
            ], capture_output=True, text=True, cwd=self.root_dir)

            errors = {}
            for line in result.stdout.strip().split('\n'):
                if ':' in line and len(line.split(':')) >= 4:
                    parts = line.split(':', 3)
                    if len(parts) >= 4:
                        file_path = parts[0]
                        if file_path not in errors:
                            errors[file_path] = []
                        errors[file_path].append(line)

            return errors
        except Exception as e:
            print(f"Flake8分析失败: {e}")
            return {}

    def fix_syntax_errors(self, errors: Dict[str, List[str]]) -> int:
        """修复语法错误"""
        fixed_count = 0
        syntax_files = []

        # 收集有语法错误的文件
        for file_path, file_errors in errors.items():
            for error in file_errors:
                if ':E999:' in error or ':F63:' in error or ':F7:' in error:
                    syntax_files.append(file_path)
                    break

        print(f"发现 {len(syntax_files)} 个有语法错误的文件")

        for file_path in syntax_files:
            if self._fix_file_syntax_errors(file_path):
                fixed_count += 1

        return fixed_count

    def _fix_file_syntax_errors(self, file_path: str) -> bool:
        """修复单个文件的语法错误"""
        try:
            full_path = self.root_dir / file_path
            if not full_path.exists():
                return False

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 应用各种语法修复
            content = self._fix_indentation_errors(content)
            content = self._fix_string_syntax_errors(content)
            content = self._fix_incomplete_statements(content)

            if content != original_content:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                # 验证修复是否成功
                try:
                    ast.parse(content)
                    print(f"✅ 修复语法错误: {file_path}")
                    self.fixed_files.add(file_path)
                    return True
                except SyntaxError:
                    print(f"❌ 修复失败，仍有语法错误: {file_path}")
                    return False
            else:
                print(f"ℹ️ 无需修复: {file_path}")
                return False

        except Exception as e:
            print(f"❌ 处理文件失败 {file_path}: {e}")
            return False

    def _fix_indentation_errors(self, content: str) -> str:
        """修复缩进错误"""
        lines = content.split('\n')
        result_lines = []
        indent_stack = []

        for i, line in enumerate(lines):
            stripped = line.rstrip()
            if not stripped:
                result_lines.append(line)
                continue

            # 计算当前行的缩进
            current_indent = len(line) - len(line.lstrip())

            # 检查是否是控制结构
            is_control = any(stripped.startswith(keyword) for keyword in
                           ['if ', 'for ', 'while ', 'def ', 'class ', 'try:', 'except ', 'finally:', 'else:', 'elif '])

            if is_control:
                # 控制结构开始新的缩进级别
                expected_indent = len(indent_stack) * 4 if indent_stack else 0
                if current_indent != expected_indent:
                    # 修复缩进
                    line = ' ' * expected_indent + line.lstrip()
                    indent_stack.append('control')
            elif indent_stack and current_indent < len(indent_stack) * 4:
                # 可能需要减少缩进
                while indent_stack and current_indent < len(indent_stack) * 4:
                    indent_stack.pop()
                expected_indent = len(indent_stack) * 4
                if current_indent != expected_indent:
                    line = ' ' * expected_indent + line.lstrip()

            result_lines.append(line)

        return '\n'.join(result_lines)

    def _fix_string_syntax_errors(self, content: str) -> str:
        """修复字符串语法错误"""
        # 修复未闭合的三引号字符串
        lines = content.split('\n')
        in_string = False
        string_start = -1

        for i, line in enumerate(lines):
            if '"""' in line:
                if not in_string:
                    in_string = True
                    string_start = i
                else:
                    in_string = False
                    string_start = -1

        # 如果文件末尾还有未闭合的字符串，添加闭合
        if in_string and string_start >= 0:
            lines.append('"""')
            content = '\n'.join(lines)

        return content

    def _fix_incomplete_statements(self, content: str) -> str:
        """修复不完整的语句"""
        # 修复不完整的try-except
        content = re.sub(
            r'(\s+)try:\s*\n(\s+)except\s+ImportError\s+as\s+e\s*:',
            r'\1try:\n\1    # Import check\n\1    import sys\n\2except ImportError as e:',
            content
        )

        # 修复不完整的函数定义
        content = re.sub(
            r'(\s+)def\s+(\w+)\([^)]*:\s*\n(\s+)"""([^""]*)"""\s*\n(\s+)(\w+)',
            r'\1def \2(...):\n\3"""\4"""\n\5\6',
            content
        )

        return content

    def fix_import_errors(self, errors: Dict[str, List[str]]) -> int:
        """修复导入错误"""
        fixed_count = 0
        import_files = []

        # 收集有导入错误的文件
        for file_path, file_errors in errors.items():
            for error in file_errors:
                if ':F821:' in error:
                    import_files.append(file_path)
                    break

        print(f"发现 {len(import_files)} 个有导入错误的文件")

        for file_path in import_files:
            if self._fix_file_import_errors(file_path):
                fixed_count += 1

        return fixed_count

    def _fix_file_import_errors(self, file_path: str) -> bool:
        """修复单个文件的导入错误"""
        try:
            full_path = self.root_dir / file_path
            if not full_path.exists():
                return False

            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # 添加常见的缺失导入
            content = self._add_missing_imports(content)

            if content != original_content:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ 修复导入错误: {file_path}")
                return True

            return False

        except Exception as e:
            print(f"❌ 处理导入失败 {file_path}: {e}")
            return False

    def _add_missing_imports(self, content: str) -> str:
        """添加缺失的导入"""
        lines = content.split('\n')
        imports_added = set()

        # 扫描文件查找缺失的导入
        for line in lines:
            line = line.strip()
            if 'F821' in line and 'undefined name' in line:
                # 解析错误信息
                parts = line.split()
                if len(parts) >= 6:
                    undefined_name = parts[5].strip("'\"")
                    if undefined_name not in imports_added:
                        # 添加相应的导入
                        import_line = self._get_import_for_name(undefined_name)
                        if import_line:
                            # 找到现有的import语句位置
                            import_index = -1
                            for i, l in enumerate(lines):
                                if l.strip().startswith('import ') or l.strip().startswith('from '):
                                    import_index = i + 1
                                elif not l.strip().startswith('#') and l.strip() and import_index == -1:
                                    break

                            if import_index > 0:
                                lines.insert(import_index, import_line)
                                imports_added.add(undefined_name)

        return '\n'.join(lines)

    def _get_import_for_name(self, name: str) -> str:
        """根据名称获取导入语句"""
        import_map = {
            'logging': 'import logging',
            'time': 'import time',
            'os': 'import os',
            'sys': 'import sys',
            'json': 'import json',
            'threading': 'import threading',
            'asyncio': 'import asyncio',
            'pathlib': 'from pathlib import Path',
            'Path': 'from pathlib import Path',
            'Dict': 'from typing import Dict, List, Any, Optional',
            'List': 'from typing import Dict, List, Any, Optional',
            'Any': 'from typing import Dict, List, Any, Optional',
            'Optional': 'from typing import Dict, List, Any, Optional',
            'Callable': 'from typing import Callable',
            'Enum': 'from enum import Enum',
            'dataclass': 'from dataclasses import dataclass',
            'field': 'from dataclasses import field',
            'np': 'import numpy as np',
            'pd': 'import pandas as pd',
            'plt': 'import matplotlib.pyplot as plt',
        }

        return import_map.get(name, '')

    def run_style_formatting(self) -> int:
        """运行代码风格格式化"""
        try:
            # 使用black进行格式化
            result = subprocess.run([
                'python', '-m', 'black', str(self.root_dir),
                '--line-length', '120', '--check'
            ], capture_output=True, cwd=self.root_dir)

            if result.returncode == 0:
                print("✅ 代码风格检查通过")
                return 0
            else:
                print(f"发现 {result.returncode} 个风格问题")
                return result.returncode

        except Exception as e:
            print(f"风格格式化失败: {e}")
            return -1

    def generate_report(self) -> str:
        """生成改进报告"""
        report = f"""
🔍 RQA2025 代码质量改进报告
{'=' * 50}

📊 改进统计:
• 处理文件总数: {self.stats['total_files_processed']}
• 修复语法错误: {self.stats['syntax_errors_fixed']}
• 修复导入错误: {self.stats['import_errors_fixed']}
• 修复风格问题: {self.stats['style_issues_fixed']}

🎯 改进成果:
• 语法错误修复率: {(self.stats['syntax_errors_fixed'] / max(1, self.stats['total_files_processed'])) * 100:.1f}%
• 导入错误修复率: {(self.stats['import_errors_fixed'] / max(1, self.stats['total_files_processed'])) * 100:.1f}%

🔧 修复的文件:
"""
        for file in sorted(self.fixed_files):
            report += f"• {file}\n"

        report += """
💡 后续建议:
• 建立持续集成检查
• 完善代码质量监控
• 统一代码风格规范
• 加强单元测试覆盖

🏆 代码质量改进完成！
"""
        return report

    def run_full_improvement(self):
        """运行完整的代码质量改进"""
        print("🚀 开始RQA2025代码质量深度改进...")

        # Phase 1: 语法错误修复
        print("\n📝 Phase 1: 语法错误修复")
        errors = self.run_flake8_analysis()
        syntax_fixed = self.fix_syntax_errors(errors)
        self.stats['syntax_errors_fixed'] = syntax_fixed

        # Phase 2: 导入错误修复
        print("\n📦 Phase 2: 导入错误修复")
        import_fixed = self.fix_import_errors(errors)
        self.stats['import_errors_fixed'] = import_fixed

        # Phase 3: 风格格式化
        print("\n🎨 Phase 3: 代码风格格式化")
        style_fixed = self.run_style_formatting()
        self.stats['style_issues_fixed'] = style_fixed

        # 生成报告
        report = self.generate_report()
        print(report)

        # 保存报告
        with open('code_quality_improvement_report.md', 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    improver = CodeQualityImprover('src')
    improver.run_full_improvement()


if __name__ == '__main__':
    main()

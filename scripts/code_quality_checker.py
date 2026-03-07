#!/usr/bin/env python3
"""
RQA2025 代码质量检查工具
Code Quality Checker for RQA2025

自动化检查代码质量，包括：
- 魔法数字检测
- 长方法检测
- 异常处理检查
- 类型提示检查
- 文档完整性检查
"""

import os
import re
import ast
import json
from typing import Dict, List, Any
from pathlib import Path
import argparse


class CodeQualityChecker:
    """代码质量检查器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            'magic_numbers': [],
            'long_methods': [],
            'missing_docstrings': [],
            'missing_type_hints': [],
            'exception_handling': [],
            'summary': {}
        }

    def check_file(self, file_path: Path) -> Dict[str, Any]:
        """检查单个文件"""
        file_results = {
            'magic_numbers': [],
            'long_methods': [],
            'missing_docstrings': [],
            'missing_type_hints': [],
            'exception_handling': []
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')

            # 检查魔法数字
            file_results['magic_numbers'] = self._check_magic_numbers(content, str(file_path))

            # 检查长方法
            file_results['long_methods'] = self._check_long_methods(content, str(file_path))

            # 检查文档字符串
            file_results['missing_docstrings'] = self._check_docstrings(content, str(file_path))

            # 检查异常处理
            file_results['exception_handling'] = self._check_exception_handling(
                content, str(file_path))

        except Exception as e:
            print(f"Error checking file {file_path}: {e}")

        return file_results

    def _check_magic_numbers(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """检查魔法数字"""
        magic_numbers = []
        lines = content.split('\n')

        # 常见的魔法数字模式
        patterns = [
            r'\b\d{2,}\b',  # 两位及以上的数字
            r'\b0\.\d{2,}\b',  # 小数点后两位及以上的小数
        ]

        exclude_patterns = [
            r'import\s+.*\d+',  # 导入语句
            r'__version__\s*=\s*.*\d+',  # 版本号
            r'#.*\d+',  # 注释中的数字
            r'""".*\d+.*"""',  # 文档字符串中的数字
            r"'''.*\d+.*'''",  # 文档字符串中的数字
            r'=\s*\d+\s*#.*常量',  # 常量定义
            r'=\s*\d+\s*#.*Constant',  # 常量定义
        ]

        for line_num, line in enumerate(lines, 1):
            # 跳过排除模式
            if any(re.search(pattern, line) for pattern in exclude_patterns):
                continue

            # 检查魔法数字
            for pattern in patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    number = match.group()
                    # 排除常见的非魔法数字
                    if number in ['0', '1', '10', '100', '1000', '0.0', '1.0']:
                        continue

                    magic_numbers.append({
                        'file': file_path,
                        'line': line_num,
                        'number': number,
                        'context': line.strip()
                    })

        return magic_numbers

    def _check_long_methods(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """检查长方法"""
        long_methods = []
        lines = content.split('\n')

        # 简单的方法长度检查（基于缩进）
        current_method = None
        method_start = 0
        indent_level = 0

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()

            # 检测方法定义
            if re.match(r'def\s+\w+', stripped):
                if current_method and (line_num - method_start) > 50:  # 长方法阈值
                    long_methods.append({
                        'file': file_path,
                        'method': current_method,
                        'start_line': method_start,
                        'end_line': line_num - 1,
                        'length': line_num - method_start
                    })

                current_method = stripped.split('(')[0].replace('def ', '')
                method_start = line_num
                indent_level = len(line) - len(line.lstrip())

            # 检测缩进变化（方法结束）
            elif current_method and line and not line[indent_level:].strip():
                # 空行，暂时忽略
                pass

        # 检查最后一个方法
        if current_method and (line_num - method_start) > 50:
            long_methods.append({
                'file': file_path,
                'method': current_method,
                'start_line': method_start,
                'end_line': line_num,
                'length': line_num - method_start
            })

        return long_methods

    def _check_docstrings(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """检查缺失的文档字符串"""
        missing_docs = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # 检查是否有文档字符串
                    if not ast.get_docstring(node):
                        missing_docs.append({
                            'file': file_path,
                            'type': 'class' if isinstance(node, ast.ClassDef) else 'function',
                            'name': node.name,
                            'line': node.lineno
                        })
        except SyntaxError:
            # 如果文件无法解析，跳过
            pass

        return missing_docs

    def _check_exception_handling(self, content: str, file_path: str) -> List[Dict[str, Any]]:
        """检查异常处理"""
        exception_issues = []
        lines = content.split('\n')

        # 检查空的except块
        for line_num, line in enumerate(lines, 1):
            if re.search(r'except.*:\s*$', line.strip()):
                # 检查下一行是否为空或只有pass
                if line_num < len(lines):
                    next_line = lines[line_num].strip()
                    if not next_line or next_line == 'pass':
                        exception_issues.append({
                            'file': file_path,
                            'line': line_num,
                            'issue': 'empty_except_block',
                            'context': line.strip()
                        })

        return exception_issues

    def run_check(self, include_patterns: List[str] = None,
                  exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """运行完整检查"""
        if include_patterns is None:
            include_patterns = ['*.py']
        if exclude_patterns is None:
            exclude_patterns = ['__pycache__', '.git', 'venv', 'node_modules']

        # 收集所有Python文件
        python_files = []
        for pattern in include_patterns:
            if pattern == '*.py':
                for root, dirs, files in os.walk(self.project_root):
                    # 排除目录
                    dirs[:] = [d for d in dirs if not any(excl in d for excl in exclude_patterns)]

                    for file in files:
                        if file.endswith('.py'):
                            python_files.append(Path(root) / file)

        # 检查每个文件
        total_files = len(python_files)
        processed_files = 0

        for file_path in python_files:
            if processed_files % 50 == 0:
                print(f"Processing {processed_files}/{total_files} files...")

            file_results = self.check_file(file_path)

            # 合并结果
            for category in ['magic_numbers', 'long_methods', 'missing_docstrings', 'exception_handling']:
                self.results[category].extend(file_results[category])

            processed_files += 1

        # 生成摘要
        self.results['summary'] = {
            'total_files_checked': total_files,
            'magic_numbers_count': len(self.results['magic_numbers']),
            'long_methods_count': len(self.results['long_methods']),
            'missing_docstrings_count': len(self.results['missing_docstrings']),
            'exception_issues_count': len(self.results['exception_handling'])
        }

        return self.results

    def save_report(self, output_file: str):
        """保存检查报告"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"Report saved to {output_file}")
        print("\nSummary:")
        summary = self.results['summary']
        print(f"- Files checked: {summary['total_files_checked']}")
        print(f"- Magic numbers found: {summary['magic_numbers_count']}")
        print(f"- Long methods found: {summary['long_methods_count']}")
        print(f"- Missing docstrings: {summary['missing_docstrings_count']}")
        print(f"- Exception handling issues: {summary['exception_issues_count']}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025 Code Quality Checker')
    parser.add_argument('--project-root', default='.',
                        help='Project root directory')
    parser.add_argument('--output', default='code_quality_report.json',
                        help='Output report file')
    parser.add_argument('--include', nargs='+', default=['*.py'],
                        help='File patterns to include')
    parser.add_argument('--exclude', nargs='+', default=['__pycache__', '.git', 'venv', 'node_modules'],
                        help='Directories to exclude')

    args = parser.parse_args()

    checker = CodeQualityChecker(args.project_root)
    results = checker.run_check(args.include, args.exclude)
    checker.save_report(args.output)


if __name__ == '__main__':
    main()

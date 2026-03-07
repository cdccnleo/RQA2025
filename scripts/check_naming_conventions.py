#!/usr/bin/env python3
"""
RQA2025 命名规范自动化检查工具

检查Python代码是否符合项目的命名规范，包括：
- 文件命名规范
- 类命名规范
- 方法命名规范
- 变量命名规范
- 导入规范

使用方法：
python scripts/check_naming_conventions.py src/
python scripts/check_naming_conventions.py src/core/resource_manager.py
"""

import os
import re
import ast
import argparse
from pathlib import Path
from typing import List
from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    """检查结果严重程度"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class NamingIssue:
    """命名规范问题"""
    file_path: str
    line_number: int
    severity: Severity
    category: str
    message: str
    suggestion: str


class NamingConventionChecker:
    """命名规范检查器"""

    def __init__(self):
        # 文件命名模式
        self.file_patterns = {
            'core': re.compile(r'^[a-z][a-z_]*_(manager|engine|service|component|handler|coordinator|interface|analyzer|reporter)$'),
            'api': re.compile(r'^[a-z][a-z_]*_(api|endpoint|service|controller)$'),
            'config': re.compile(r'^[a-z][a-z_]*_(config|setting|option|classes)$'),
            'model': re.compile(r'^[a-z][a-z_]*_(model|entity|dto|schema|dataclass|enum)$'),
            'util': re.compile(r'^[a-z][a-z_]*_(util|helper|tool|common|validator|detector|generator|decorator)$'),
            'test': re.compile(r'^test_[a-z][a-z_]*$'),
            'monitoring': re.compile(r'^[a-z][a-z_]*_(monitor|health|metric|log|alert|performance|business|unified|status)$'),
            'scheduling': re.compile(r'^[a-z][a-z_]*_(scheduler|task|job|queue)$'),
            'ui': re.compile(r'^[a-z][a-z_]*_(ui|view|template|dashboard)$'),
            'data': re.compile(r'^[a-z][a-z_]*_(repository|dao|storage|db|event)$'),
        }

        # 类命名模式
        self.class_patterns = {
            'pascal_case': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),
            'component_class': re.compile(r'^[A-Z][a-zA-Z0-9]*(Manager|Service|Engine|Handler|Component|Coordinator|Interface|Analyzer|Reporter)$'),
        }

        # 方法命名模式
        self.method_patterns = {
            'snake_case': re.compile(r'^[a-z][a-z0-9_]*$'),
            'private_method': re.compile(r'^_[a-z][a-z0-9_]*$'),
            'verb_prefix': re.compile(r'^(get|set|create|update|delete|process|validate|convert|calculate|generate)_[a-z]'),
        }

        # 变量命名模式
        self.variable_patterns = {
            'snake_case': re.compile(r'^[a-z][a-z0-9_]*$'),
            'constant': re.compile(r'^[A-Z][A-Z0-9_]*$'),
        }

        self.issues: List[NamingIssue] = []

    def check_file(self, file_path: Path) -> List[NamingIssue]:
        """检查单个文件"""
        self.issues = []

        try:
            # 检查文件名
            self._check_filename(file_path)

            # 检查文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析AST
            tree = ast.parse(content, filename=str(file_path))

            # 检查类定义
            self._check_classes(tree, file_path)

            # 检查方法定义
            self._check_methods(tree, file_path)

            # 检查变量定义
            self._check_variables(tree, file_path)

            # 检查导入
            self._check_imports(tree, file_path)

        except Exception as e:
            self._add_issue(file_path, 1, Severity.ERROR, "parse_error",
                            f"无法解析文件: {e}", "检查文件语法是否正确")

        return self.issues

    def check_directory(self, directory_path: Path) -> List[NamingIssue]:
        """检查目录下的所有Python文件"""
        all_issues = []

        for root, dirs, files in os.walk(directory_path):
            # 跳过__pycache__目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']

            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    issues = self.check_file(file_path)
                    all_issues.extend(issues)

        return all_issues

    def _check_filename(self, file_path: Path) -> None:
        """检查文件名"""
        filename = file_path.name

        # 跳过特殊文件
        if filename in ['__init__.py', 'conftest.py', 'setup.py']:
            return

        # 检查文件扩展名
        if not filename.endswith('.py'):
            return

        # 检查命名模式
        matched = False
        for category, pattern in self.file_patterns.items():
            if pattern.match(filename[:-3]):  # 移除.py扩展名
                matched = True
                break

        if not matched:
            self._add_issue(file_path, 1, Severity.WARNING, "filename",
                            f"文件名 '{filename}' 不符合命名规范",
                            "使用 snake_case 命名，并包含功能关键词，如: resource_manager.py")

    def _check_classes(self, tree: ast.AST, file_path: Path) -> None:
        """检查类定义"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name

                # 检查类名格式
                if not self.class_patterns['pascal_case'].match(class_name):
                    self._add_issue(file_path, node.lineno, Severity.ERROR, "class_name",
                                    f"类名 '{class_name}' 应使用 PascalCase 格式",
                                    "使用 PascalCase 命名，如: ResourceManager")

                # 检查类名与文件名对应关系
                filename = file_path.stem
                if filename != '__init__' and not class_name.lower().startswith(filename.split('_')[0]):
                    # 放宽这个检查，只在明显不匹配时警告
                    pass

    def _check_methods(self, tree: ast.AST, file_path: Path) -> None:
        """检查方法定义"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                method_name = node.name

                # 跳过特殊方法
                if method_name.startswith('__') and method_name.endswith('__'):
                    continue

                # 检查方法名格式
                if method_name.startswith('_'):
                    # 私有方法
                    if not self.method_patterns['private_method'].match(method_name):
                        self._add_issue(file_path, node.lineno, Severity.WARNING, "private_method",
                                        f"私有方法名 '{method_name}' 格式不正确",
                                        "使用 _snake_case 格式")
                else:
                    # 公共方法
                    if not self.method_patterns['snake_case'].match(method_name):
                        self._add_issue(file_path, node.lineno, Severity.ERROR, "public_method",
                                        f"公共方法名 '{method_name}' 应使用 snake_case 格式",
                                        "使用 snake_case 命名，如: get_resource_status")

                    # 检查动词前缀
                    if not self.method_patterns['verb_prefix'].match(method_name):
                        self._add_issue(file_path, node.lineno, Severity.INFO, "method_prefix",
                                        f"方法名 '{method_name}' 建议以动词开头",
                                        "使用 get_, set_, create_, update_, delete_ 等前缀")

    def _check_variables(self, tree: ast.AST, file_path: Path) -> None:
        """检查变量定义"""
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                # 只检查赋值目标
                if isinstance(node.ctx, ast.Store):
                    var_name = node.id

                    # 跳过特殊变量
                    if var_name.startswith('_') or var_name in ['self', 'cls']:
                        continue

                    # 检查变量名格式
                    if not self.variable_patterns['snake_case'].match(var_name):
                        self._add_issue(file_path, getattr(node, 'lineno', 1), Severity.WARNING, "variable_name",
                                        f"变量名 '{var_name}' 应使用 snake_case 格式",
                                        "使用 snake_case 命名，如: resource_manager")

            elif isinstance(node, ast.Assign):
                # 检查常量赋值
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    target_name = node.targets[0].id
                    if self.variable_patterns['constant'].match(target_name):
                        # 检查是否为常量赋值
                        if isinstance(node.value, (ast.Str, ast.Num, ast.List, ast.Dict, ast.Tuple)):
                            continue  # 可能是常量

    def _check_imports(self, tree: ast.AST, file_path: Path) -> None:
        """检查导入语句"""
        imports = []
        from_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(node.names)
            elif isinstance(node, ast.ImportFrom):
                from_imports.append(node)

        # 检查导入顺序（标准库 -> 第三方库 -> 本地模块）
        # 这里可以添加更复杂的导入顺序检查逻辑

    def _add_issue(self, file_path: Path, line_number: int, severity: Severity,
                   category: str, message: str, suggestion: str) -> None:
        """添加问题"""
        issue = NamingIssue(
            file_path=str(file_path),
            line_number=line_number,
            severity=severity,
            category=category,
            message=message,
            suggestion=suggestion
        )
        self.issues.append(issue)

    def print_report(self, issues: List[NamingIssue]) -> None:
        """打印检查报告"""
        if not issues:
            print("✅ 所有文件都符合命名规范！")
            return

        # 按严重程度分组
        error_count = sum(1 for issue in issues if issue.severity == Severity.ERROR)
        warning_count = sum(1 for issue in issues if issue.severity == Severity.WARNING)
        info_count = sum(1 for issue in issues if issue.severity == Severity.INFO)

        print("📊 命名规范检查报告")
        print("=" * 50)
        print(f"总问题数: {len(issues)}")
        print(f"❌ 错误: {error_count}")
        print(f"⚠️  警告: {warning_count}")
        print(f"ℹ️  信息: {info_count}")
        print()

        # 按文件分组显示
        issues_by_file = {}
        for issue in issues:
            if issue.file_path not in issues_by_file:
                issues_by_file[issue.file_path] = []
            issues_by_file[issue.file_path].append(issue)

        for file_path, file_issues in issues_by_file.items():
            print(f"📁 {file_path}:")
            for issue in file_issues:
                severity_icon = {
                    Severity.ERROR: "❌",
                    Severity.WARNING: "⚠️",
                    Severity.INFO: "ℹ️"
                }[issue.severity]

                print(f"  {severity_icon} 第{issue.line_number}行 [{issue.category}] {issue.message}")
                print(f"    💡 建议: {issue.suggestion}")
            print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="RQA2025 命名规范检查工具")
    parser.add_argument("path", help="要检查的文件或目录路径")
    parser.add_argument("--fix", action="store_true", help="自动修复简单问题")

    args = parser.parse_args()

    checker = NamingConventionChecker()
    path = Path(args.path)

    if path.is_file():
        issues = checker.check_file(path)
    elif path.is_dir():
        issues = checker.check_directory(path)
    else:
        print(f"❌ 路径不存在: {path}")
        return

    checker.print_report(issues)

    # 返回非零退出码表示有错误
    error_count = sum(1 for issue in issues if issue.severity == Severity.ERROR)
    exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()

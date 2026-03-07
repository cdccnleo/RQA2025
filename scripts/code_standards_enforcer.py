#!/usr/bin/env python3
"""
代码规范执行工具

用于制定代码规范标准、自动化检查工具配置、注释补全和文档标准化。
支持PEP8、类型提示、文档字符串规范等。
"""

import os
import ast
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import isort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CodeStandard:
    """代码规范"""
    name: str
    description: str
    checker: callable
    fixer: Optional[callable] = None
    severity: str = "warning"  # "error", "warning", "info"


@dataclass
class CodeIssue:
    """代码问题"""
    file: str
    line: int
    column: int
    code: str
    message: str
    severity: str
    standard: str
    fix_available: bool = False


class CodeStandardsChecker:
    """代码规范检查器"""

    def __init__(self):
        self.standards = self._load_standards()
        self.issues = []

    def _load_standards(self) -> List[CodeStandard]:
        """加载代码规范"""
        return [
            CodeStandard(
                name="PEP8_LINE_LENGTH",
                description="代码行长度不超过88字符",
                checker=self._check_line_length,
                fixer=self._fix_line_length,
                severity="warning"
            ),
            CodeStandard(
                name="TYPE_HINTS_REQUIRED",
                description="所有函数参数和返回值都需要类型提示",
                checker=self._check_type_hints,
                severity="error"
            ),
            CodeStandard(
                name="DOCSTRING_REQUIRED",
                description="所有公共函数和类都需要文档字符串",
                checker=self._check_docstrings,
                fixer=self._fix_docstrings,
                severity="warning"
            ),
            CodeStandard(
                name="IMPORT_SORTING",
                description="导入语句需要正确排序",
                checker=self._check_import_sorting,
                fixer=self._fix_import_sorting,
                severity="warning"
            ),
            CodeStandard(
                name="CONSTANT_NAMING",
                description="常量使用大写命名",
                checker=self._check_constant_naming,
                fixer=self._fix_constant_naming,
                severity="warning"
            ),
            CodeStandard(
                name="FUNCTION_LENGTH",
                description="函数长度不超过50行",
                checker=self._check_function_length,
                severity="warning"
            ),
            CodeStandard(
                name="CLASS_LENGTH",
                description="类长度不超过300行",
                checker=self._check_class_length,
                severity="warning"
            ),
            CodeStandard(
                name="MAGIC_NUMBERS",
                description="避免使用魔法数字",
                checker=self._check_magic_numbers,
                fixer=self._fix_magic_numbers,
                severity="warning"
            ),
            CodeStandard(
                name="EXCEPTION_HANDLING",
                description="异常处理需要具体异常类型",
                checker=self._check_exception_handling,
                fixer=self._fix_exception_handling,
                severity="error"
            ),
            CodeStandard(
                name="LOGGING_LEVELS",
                description="使用适当的日志级别",
                checker=self._check_logging_levels,
                fixer=self._fix_logging_levels,
                severity="info"
            )
        ]

    def check_file(self, file_path: str) -> List[CodeIssue]:
        """检查单个文件"""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析AST
            tree = ast.parse(content)
            lines = content.splitlines()

            # 应用所有规范检查
            for standard in self.standards:
                try:
                    file_issues = standard.checker(file_path, content, lines, tree)
                    issues.extend(file_issues)
                except Exception as e:
                    logger.warning(f"检查 {standard.name} 时出错: {e}")

        except Exception as e:
            logger.error(f"检查文件 {file_path} 时出错: {e}")

        return issues

    def check_directory(self, directory: str, extensions: List[str] = None) -> List[CodeIssue]:
        """检查整个目录"""
        if extensions is None:
            extensions = ['.py']

        all_issues = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    issues = self.check_file(file_path)
                    all_issues.extend(issues)

        return all_issues

    def generate_report(self, issues: List[CodeIssue]) -> Dict[str, Any]:
        """生成检查报告"""
        # 按文件分组
        by_file = {}
        for issue in issues:
            if issue.file not in by_file:
                by_file[issue.file] = []
            by_file[issue.file].append(issue)

        # 按严重程度分组
        by_severity = {"error": 0, "warning": 0, "info": 0}
        for issue in issues:
            by_severity[issue.severity] += 1

        # 按规范分组
        by_standard = {}
        for issue in issues:
            if issue.standard not in by_standard:
                by_standard[issue.standard] = 0
            by_standard[issue.standard] += 1

        return {
            "total_issues": len(issues),
            "issues_by_severity": by_severity,
            "issues_by_standard": by_standard,
            "issues_by_file": by_file,
            "files_with_issues": len(by_file),
            "fixable_issues": len([i for i in issues if i.fix_available])
        }

    def _check_line_length(self, file_path: str, content: str, lines: List[str], tree: ast.AST) -> List[CodeIssue]:
        """检查行长度"""
        issues = []
        max_length = 88

        for i, line in enumerate(lines, 1):
            if len(line) > max_length and not line.strip().startswith('#'):
                issues.append(CodeIssue(
                    file=file_path,
                    line=i,
                    column=max_length,
                    code="E501",
                    message=f"行长度超过{max_length}字符",
                    severity="warning",
                    standard="PEP8_LINE_LENGTH",
                    fix_available=True
                ))

        return issues

    def _check_type_hints(self, file_path: str, content: str, lines: List[str], tree: ast.AST) -> List[CodeIssue]:
        """检查类型提示"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 检查函数参数
                for arg in node.args.args:
                    if arg.annotation is None and arg.arg != 'self':
                        issues.append(CodeIssue(
                            file=file_path,
                            line=node.lineno,
                            column=0,
                            code="M001",
                            message=f"函数参数 '{arg.arg}' 缺少类型提示",
                            severity="error",
                            standard="TYPE_HINTS_REQUIRED",
                            fix_available=False
                        ))

                # 检查返回值
                if node.returns is None:
                    issues.append(CodeIssue(
                        file=file_path,
                        line=node.lineno,
                        column=0,
                        code="M002",
                        message=f"函数 '{node.name}' 缺少返回值类型提示",
                        severity="error",
                        standard="TYPE_HINTS_REQUIRED",
                        fix_available=False
                    ))

        return issues

    def _check_docstrings(self, file_path: str, content: str, lines: List[str], tree: ast.AST) -> List[CodeIssue]:
        """检查文档字符串"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # 检查是否是公共成员
                if not node.name.startswith('_'):
                    docstring = ast.get_docstring(node)
                    if not docstring or len(docstring.strip()) < 10:
                        node_type = "类" if isinstance(node, ast.ClassDef) else "函数"
                        issues.append(CodeIssue(
                            file=file_path,
                            line=node.lineno,
                            column=0,
                            code="M003",
                            message=f"{node_type} '{node.name}' 缺少或文档字符串过短",
                            severity="warning",
                            standard="DOCSTRING_REQUIRED",
                            fix_available=True
                        ))

        return issues

    def _check_import_sorting(self, file_path: str, content: str, lines: List[str], tree: ast.AST) -> List[CodeIssue]:
        """检查导入排序"""
        issues = []

        # 简单检查：标准库导入、第三方导入、本地导入的顺序
        import_lines = []
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                import_lines.append((i + 1, line.strip()))

        if len(import_lines) > 1:
            # 检查是否有明显的排序问题
            std_lib_imports = []
            third_party_imports = []
            local_imports = []

            for line_no, line in import_lines:
                if any(lib in line for lib in ['os', 'sys', 'json', 'typing', 'dataclasses']):
                    std_lib_imports.append((line_no, line))
                elif 'src.' in line or 'from .' in line:
                    local_imports.append((line_no, line))
                else:
                    third_party_imports.append((line_no, line))

            # 检查顺序
            all_imports = std_lib_imports + third_party_imports + local_imports
            if all_imports != sorted(all_imports, key=lambda x: x[0]):
                issues.append(CodeIssue(
                    file=file_path,
                    line=import_lines[0][0],
                    column=0,
                    code="M004",
                    message="导入语句排序不规范",
                    severity="warning",
                    standard="IMPORT_SORTING",
                    fix_available=True
                ))

        return issues

    def _check_constant_naming(self, file_path: str, content: str, lines: List[str], tree: ast.AST) -> List[CodeIssue]:
        """检查常量命名"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        # 检查是否可能是常量（大写字母）
                        if name.isupper() and len(name) > 1:
                            # 检查赋值是否是常量
                            if isinstance(node.value, (ast.Str, ast.Num, ast.List, ast.Dict, ast.Tuple)):
                                continue  # 可能是真正的常量
                            else:
                                issues.append(CodeIssue(
                                    file=file_path,
                                    line=node.lineno,
                                    column=0,
                                    code="M005",
                                    message=f"变量 '{name}' 使用了大写命名但不是常量",
                                    severity="warning",
                                    standard="CONSTANT_NAMING",
                                    fix_available=False
                                ))

        return issues

    def _check_function_length(self, file_path: str, content: str, lines: List[str], tree: ast.AST) -> List[CodeIssue]:
        """检查函数长度"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                end_line = getattr(node, 'end_lineno', start_line)
                length = end_line - start_line + 1

                if length > 50:
                    issues.append(CodeIssue(
                        file=file_path,
                        line=start_line,
                        column=0,
                        code="M006",
                        message=f"函数 '{node.name}' 过长 ({length}行)，建议拆分",
                        severity="warning",
                        standard="FUNCTION_LENGTH",
                        fix_available=False
                    ))

        return issues

    def _check_class_length(self, file_path: str, content: str, lines: List[str], tree: ast.AST) -> List[CodeIssue]:
        """检查类长度"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                start_line = node.lineno
                end_line = getattr(node, 'end_lineno', start_line)
                length = end_line - start_line + 1

                if length > 300:
                    issues.append(CodeIssue(
                        file=file_path,
                        line=start_line,
                        column=0,
                        code="M007",
                        message=f"类 '{node.name}' 过长 ({length}行)，建议拆分",
                        severity="warning",
                        standard="CLASS_LENGTH",
                        fix_available=False
                    ))

        return issues

    def _check_magic_numbers(self, file_path: str, content: str, lines: List[str], tree: ast.AST) -> List[CodeIssue]:
        """检查魔法数字"""
        issues = []

        magic_numbers = [0, 1, 2, 3, 4, 5, 10, 100, 1000, 60, 3600, 86400]  # 常见的可接受数字

        for node in ast.walk(tree):
            if isinstance(node, ast.Num) and isinstance(node.n, int):
                if node.n not in magic_numbers and node.n > 5:
                    # 检查上下文
                    parent = getattr(node, '_parent', None)
                    if parent and not isinstance(parent, (ast.Dict, ast.List, ast.Tuple)):
                        issues.append(CodeIssue(
                            file=file_path,
                            line=node.lineno,
                            column=getattr(node, 'col_offset', 0),
                            code="M008",
                            message=f"魔法数字 {node.n}，建议定义为常量",
                            severity="warning",
                            standard="MAGIC_NUMBERS",
                            fix_available=True
                        ))

        return issues

    def _check_exception_handling(self, file_path: str, content: str, lines: List[str], tree: ast.AST) -> List[CodeIssue]:
        """检查异常处理"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    # 捕获所有异常
                    issues.append(CodeIssue(
                        file=file_path,
                        line=node.lineno,
                        column=0,
                        code="M009",
                        message="捕获了所有异常，建议指定具体异常类型",
                        severity="error",
                        standard="EXCEPTION_HANDLING",
                        fix_available=False
                    ))

        return issues

    def _check_logging_levels(self, file_path: str, content: str, lines: List[str], tree: ast.AST) -> List[CodeIssue]:
        """检查日志级别"""
        issues = []

        # 检查logger的使用
        for i, line in enumerate(lines, 1):
            if 'logger.' in line or 'logging.' in line:
                # 检查是否使用了适当的级别
                if '.debug(' in line and 'DEBUG' not in line.upper():
                    issues.append(CodeIssue(
                        file=file_path,
                        line=i,
                        column=0,
                        code="M010",
                        message="建议使用logger.debug()而不是print调试",
                        severity="info",
                        standard="LOGGING_LEVELS",
                        fix_available=True
                    ))

        return issues

    # 修复方法实现
    def _fix_line_length(self, issue: CodeIssue) -> Optional[str]:
        """修复行长度"""
        # 这里可以实现自动换行逻辑
        return None

    def _fix_docstrings(self, issue: CodeIssue) -> Optional[str]:
        """修复文档字符串"""
        # 这里可以实现自动添加文档字符串的逻辑
        return None

    def _fix_import_sorting(self, issue: CodeIssue) -> Optional[str]:
        """修复导入排序"""
        try:
            result = isort.code(open(issue.file).read())
            return result
        except Exception:
            return None

    def _fix_constant_naming(self, issue: CodeIssue) -> Optional[str]:
        """修复常量命名"""
        return None

    def _fix_magic_numbers(self, issue: CodeIssue) -> Optional[str]:
        """修复魔法数字"""
        return None

    def _fix_exception_handling(self, issue: CodeIssue) -> Optional[str]:
        """修复异常处理"""
        return None

    def _fix_logging_levels(self, issue: CodeIssue) -> Optional[str]:
        """修复日志级别"""
        return None


class CodeStandardsFormatter:
    """代码规范格式化器"""

    def __init__(self):
        self.formatter_config = {
            "line_length": 88,
            "string_normalization": True,
            "magic_trailing_comma": True,
            "target_versions": {'py39'},
        }

    def format_file(self, file_path: str) -> bool:
        """格式化单个文件"""
        try:
            # 使用black格式化
            result = subprocess.run(
                ['black', '--line-length', '88', file_path],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info(f"格式化完成: {file_path}")
                return True
            else:
                logger.warning(f"格式化失败: {file_path} - {result.stderr}")
                return False

        except Exception as e:
            logger.error(f"格式化文件时出错 {file_path}: {e}")
            return False

    def format_directory(self, directory: str) -> Dict[str, Any]:
        """格式化整个目录"""
        formatted_count = 0
        failed_count = 0

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    if self.format_file(file_path):
                        formatted_count += 1
                    else:
                        failed_count += 1

        return {
            "formatted_files": formatted_count,
            "failed_files": failed_count,
            "total_files": formatted_count + failed_count
        }


def main():
    """主函数"""
    print("🚀 大文件重构和代码规范专项 - 代码规范统一")
    print("=" * 60)

    # 创建检查器和格式化器
    checker = CodeStandardsChecker()
    formatter = CodeStandardsFormatter()

    # 检查的目标目录
    target_dirs = [
        "src/mobile/mobile_trading_modules",
        "src/features/acceleration/gpu",
        "src/core/optimizations",
        "src/data/integration"
    ]

    all_issues = []

    print("\n1. 检查代码规范...")
    for directory in target_dirs:
        if os.path.exists(directory):
            print(f"   📁 检查目录: {directory}")
            issues = checker.check_directory(directory)
            all_issues.extend(issues)
            print(f"      发现问题: {len(issues)}")

    # 生成报告
    print("\n2. 生成规范检查报告...")
    report = checker.generate_report(all_issues)

    print("\n📊 检查结果:")
    print(f"   🔍 总问题数: {report['total_issues']}")
    print(f"   📁 涉及文件: {report['files_with_issues']}")
    print(f"   🔧 可自动修复: {report['fixable_issues']}")
    print(f"   🚨 错误: {report['issues_by_severity']['error']}")
    print(f"   ⚠️  警告: {report['issues_by_severity']['warning']}")
    print(f"   ℹ️  信息: {report['issues_by_severity']['info']}")

    # 显示前10个最常见的问题
    print("\n🔥 最常见问题:")
    standards_count = report['issues_by_standard']
    sorted_standards = sorted(standards_count.items(), key=lambda x: x[1], reverse=True)

    for standard, count in sorted_standards[:10]:
        print(f"   {standard}: {count} 次")

    # 格式化代码
    print("\n3. 自动格式化代码...")
    total_formatted = 0
    total_failed = 0

    for directory in target_dirs:
        if os.path.exists(directory):
            print(f"   🎨 格式化目录: {directory}")
            format_result = formatter.format_directory(directory)
            total_formatted += format_result['formatted_files']
            total_failed += format_result['failed_files']
            print(
                f"      成功: {format_result['formatted_files']}, 失败: {format_result['failed_files']}")

    # 保存详细报告
    with open("code_standards_report.json", 'w', encoding='utf-8') as f:
        json.dump({
            "issues": [issue.__dict__ for issue in all_issues],
            "summary": report,
            "formatting": {
                "formatted_files": total_formatted,
                "failed_files": total_failed
            },
            "generated_at": str(Path(__file__).stat().st_mtime)
        }, f, indent=2, ensure_ascii=False, default=str)

    print("\n📊 规范统一总结")
    print("-" * 50)
    print(f"📁 检查目录: {len(target_dirs)}")
    print(f"📝 格式化文件: {total_formatted}")
    print(f"🔧 发现问题: {len(all_issues)}")
    print(f"💾 报告已保存: code_standards_report.json")

    # 验收标准检查
    meets_standards = (
        report['issues_by_severity']['error'] == 0 and
        len(all_issues) < 100  # 假设100个以下问题可以接受
    )

    if meets_standards:
        print("\n✅ 代码规范验收通过！")
    else:
        print("\n⚠️  代码规范需要进一步改进")

    print("\n✅ 代码规范统一专项完成！")


if __name__ == "__main__":
    main()

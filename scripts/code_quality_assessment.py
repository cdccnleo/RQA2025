#!/usr/bin/env python3
"""
代码质量评估工具

用于评估当前代码质量状态，生成详细的质量报告。
包括代码规范检查、类型提示覆盖率、文档覆盖率等指标。
"""

from src.utils.logger import get_logger
import sys
import ast
import re
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


logger = get_logger(__name__)


@dataclass
class QualityMetrics:
    """质量指标"""
    total_files: int = 0
    total_lines: int = 0
    python_files: int = 0
    code_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    functions_count: int = 0
    classes_count: int = 0
    type_hints_coverage: float = 0.0
    docstring_coverage: float = 0.0
    complexity_average: float = 0.0
    imports_count: int = 0
    violations_count: int = 0


@dataclass
class FileQualityReport:
    """文件质量报告"""
    file_path: str
    lines_count: int
    functions: List[Dict[str, Any]] = field(default_factory=list)
    classes: List[Dict[str, Any]] = field(default_factory=list)
    type_hints_score: float = 0.0
    docstring_score: float = 0.0
    complexity_score: float = 0.0
    violations: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


class CodeQualityAnalyzer:
    """代码质量分析器"""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path)
        self.exclude_patterns = [
            '__pycache__',
            '.git',
            'venv',
            'env',
            'node_modules',
            '.pytest_cache',
            'htmlcov',
            'build',
            'dist',
            '*.egg-info',
            'test_*',
            '*_test.py',
            'backup_*',
            'logging_backup*',
            'file_organization_backup*',
            'config_backup*',
            'interface_inheritance_backup*',
            'deployment*',
            'examples*',
            'tools*',
            'reports*',
            'temp*',
            'cache*',
            'data_cache*'
        ]

    def should_analyze_file(self, file_path: Path) -> bool:
        """判断是否应该分析文件"""
        # 检查是否是Python文件
        if not file_path.suffix == '.py':
            return False

        # 检查是否在排除目录中
        for pattern in self.exclude_patterns:
            if pattern in str(file_path):
                return False

        # 检查文件名
        if any(pattern in file_path.name for pattern in ['test_', '_test.py']):
            return False

        return True

    def analyze_file(self, file_path: Path) -> FileQualityReport:
        """分析单个文件"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            lines = content.splitlines()

            report = FileQualityReport(
                file_path=str(file_path.relative_to(self.project_path)),
                lines_count=len(lines)
            )

            # 分析函数和类
            self._analyze_functions_and_classes(tree, report)

            # 计算类型提示覆盖率
            report.type_hints_score = self._calculate_type_hints_coverage(tree)

            # 计算文档覆盖率
            report.docstring_score = self._calculate_docstring_coverage(tree)

            # 计算复杂度
            report.complexity_score = self._calculate_complexity(tree)

            # 检查代码规范问题
            report.violations = self._check_code_style(content, lines)

            return report

        except Exception as e:
            logger.warning(f"分析文件失败 {file_path}: {e}")
            return FileQualityReport(
                file_path=str(file_path.relative_to(self.project_path)),
                lines_count=0,
                issues=[f"分析失败: {str(e)}"]
            )

    def _analyze_functions_and_classes(self, tree: ast.AST, report: FileQualityReport):
        """分析函数和类"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'line': node.lineno,
                    'args_count': len(node.args.args),
                    'has_docstring': self._has_docstring(node),
                    'has_type_hints': self._has_type_hints(node),
                    'complexity': self._calculate_function_complexity(node)
                }
                report.functions.append(func_info)

            elif isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'line': node.lineno,
                    'methods_count': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                    'has_docstring': self._has_docstring(node)
                }
                report.classes.append(class_info)

    def _has_docstring(self, node: ast.AST) -> bool:
        """检查是否有文档字符串"""
        if not node.body:
            return False

        first_stmt = node.body[0]
        return isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Str)

    def _has_type_hints(self, func_node: ast.FunctionDef) -> bool:
        """检查函数是否有类型提示"""
        # 检查返回类型注解
        if func_node.returns:
            return True

        # 检查参数类型注解
        for arg in func_node.args.args:
            if arg.annotation:
                return True

        return False

    def _calculate_type_hints_coverage(self, tree: ast.AST) -> float:
        """计算类型提示覆盖率"""
        total_functions = 0
        typed_functions = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if self._has_type_hints(node):
                    typed_functions += 1

        return typed_functions / total_functions if total_functions > 0 else 0

    def _calculate_docstring_coverage(self, tree: ast.AST) -> float:
        """计算文档覆盖率"""
        total_functions = 0
        documented_functions = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                total_functions += 1
                if self._has_docstring(node):
                    documented_functions += 1

        return documented_functions / total_functions if total_functions > 0 else 0

    def _calculate_complexity(self, tree: ast.AST) -> float:
        """计算代码复杂度"""
        complexities = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_function_complexity(node)
                complexities.append(complexity)

        return sum(complexities) / len(complexities) if complexities else 0

    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """计算函数复杂度"""
        complexity = 1  # 基础复杂度

        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.With,
                                 ast.Try, ast.ExceptHandler, ast.Assert)):
                complexity += 1
            elif isinstance(node, ast.BoolOp) and len(node.values) > 1:
                complexity += len(node.values) - 1

        return complexity

    def _check_code_style(self, content: str, lines: List[str]) -> List[str]:
        """检查代码风格问题"""
        violations = []

        # 检查行长度
        for i, line in enumerate(lines, 1):
            if len(line) > 88:  # PEP8建议88字符
                violations.append(f"行长度过长 ({len(line)} > 88): 第{i}行")

        # 检查导入排序
        imports = []
        in_imports = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                imports.append(stripped)
                in_imports = True
            elif in_imports and stripped and not stripped.startswith('#'):
                break

        # 检查魔法数字
        magic_numbers = []
        for i, line in enumerate(lines, 1):
            # 查找数字常量（排除合理使用）
            numbers = re.findall(r'\b\d+\b', line)
            for num in numbers:
                # 排除合理数字：0, 1, 小于10的循环变量, 常见端口等
                if num not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                               '80', '443', '8080', '8000', '5432', '6379', '100', '200', '300', '400', '500']:
                    if f'magic_number_{num}' not in line:  # 排除已处理的情况
                        magic_numbers.append(f"魔法数字 {num} 在第{i}行")

        violations.extend(magic_numbers[:10])  # 限制报告数量

        # 检查命名规范
        for node in ast.walk(ast.parse(content)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                name = node.name
                if not re.match(r'^[a-z_][a-z0-9_]*$', name):
                    violations.append(f"命名不符合规范: {name}")

        return violations

    def analyze_project(self) -> Dict[str, Any]:
        """分析整个项目"""
        logger.info("开始代码质量分析...")

        all_files = []
        quality_metrics = QualityMetrics()

        # 收集所有Python文件
        for py_file in self.project_path.rglob('*.py'):
            if self.should_analyze_file(py_file):
                all_files.append(py_file)

        quality_metrics.total_files = len(all_files)
        logger.info(f"发现 {len(all_files)} 个Python文件待分析")

        # 分析每个文件
        file_reports = []
        for file_path in all_files:
            report = self.analyze_file(file_path)
            file_reports.append(report)

            # 累计统计信息
            quality_metrics.python_files += 1
            quality_metrics.total_lines += report.lines_count
            quality_metrics.functions_count += len(report.functions)
            quality_metrics.classes_count += len(report.classes)
            quality_metrics.violations_count += len(report.violations)

        # 计算综合指标
        if file_reports:
            quality_metrics.type_hints_coverage = sum(
                r.type_hints_score for r in file_reports) / len(file_reports)
            quality_metrics.docstring_coverage = sum(
                r.docstring_score for r in file_reports) / len(file_reports)
            quality_metrics.complexity_average = sum(
                r.complexity_score for r in file_reports) / len(file_reports)

        # 生成报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'project_path': str(self.project_path),
            'quality_metrics': {
                'total_files': quality_metrics.total_files,
                'python_files': quality_metrics.python_files,
                'total_lines': quality_metrics.total_lines,
                'functions_count': quality_metrics.functions_count,
                'classes_count': quality_metrics.classes_count,
                'type_hints_coverage': round(quality_metrics.type_hints_coverage * 100, 2),
                'docstring_coverage': round(quality_metrics.docstring_coverage * 100, 2),
                'complexity_average': round(quality_metrics.complexity_average, 2),
                'violations_count': quality_metrics.violations_count
            },
            'file_reports': [self._file_report_to_dict(r) for r in file_reports],
            'summary': self._generate_summary(quality_metrics, file_reports)
        }

        logger.info("代码质量分析完成")
        return report

    def _file_report_to_dict(self, report: FileQualityReport) -> Dict[str, Any]:
        """转换文件报告为字典"""
        return {
            'file_path': report.file_path,
            'lines_count': report.lines_count,
            'functions_count': len(report.functions),
            'classes_count': len(report.classes),
            'type_hints_score': round(report.type_hints_score * 100, 2),
            'docstring_score': round(report.docstring_score * 100, 2),
            'complexity_score': round(report.complexity_score, 2),
            'violations_count': len(report.violations),
            'issues': report.issues
        }

    def _generate_summary(self, metrics: QualityMetrics, reports: List[FileQualityReport]) -> Dict[str, Any]:
        """生成总结报告"""
        # 计算质量评分
        quality_score = self._calculate_quality_score(metrics)

        # 找出问题最多的文件
        worst_files = sorted(reports, key=lambda r: len(r.violations), reverse=True)[:10]

        # 找出复杂度最高的函数
        complex_functions = []
        for report in reports:
            for func in report.functions:
                if func['complexity'] > 10:
                    complex_functions.append({
                        'file': report.file_path,
                        'function': func['name'],
                        'complexity': func['complexity']
                    })

        complex_functions = sorted(
            complex_functions, key=lambda x: x['complexity'], reverse=True)[:10]

        return {
            'overall_quality_score': round(quality_score, 2),
            'grade': self._get_quality_grade(quality_score),
            'worst_files': worst_files[:5],
            'complex_functions': complex_functions[:5],
            'recommendations': self._generate_recommendations(metrics, quality_score)
        }

    def _calculate_quality_score(self, metrics: QualityMetrics) -> float:
        """计算总体质量评分"""
        score = 0

        # 类型提示覆盖率 (25%)
        score += metrics.type_hints_coverage * 25

        # 文档覆盖率 (25%)
        score += metrics.docstring_coverage * 25

        # 复杂度控制 (20%)
        complexity_penalty = max(0, (metrics.complexity_average - 5) / 5)  # 超过5的复杂度开始扣分
        score += (1 - min(complexity_penalty, 1)) * 20

        # 代码规范 (20%)
        violations_per_file = metrics.violations_count / max(metrics.python_files, 1)
        norm_penalty = min(violations_per_file / 10, 1)  # 每文件10个违规开始扣分
        score += (1 - norm_penalty) * 20

        # 代码规模合理性 (10%)
        lines_per_file = metrics.total_lines / max(metrics.python_files, 1)
        size_penalty = max(0, (lines_per_file - 300) / 200)  # 超过300行开始扣分
        score += (1 - min(size_penalty, 1)) * 10

        return max(0, min(100, score))

    def _get_quality_grade(self, score: float) -> str:
        """获取质量等级"""
        if score >= 90:
            return "A (优秀)"
        elif score >= 80:
            return "B (良好)"
        elif score >= 70:
            return "C (一般)"
        elif score >= 60:
            return "D (需改进)"
        else:
            return "F (严重问题)"

    def _generate_recommendations(self, metrics: QualityMetrics, score: float) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if metrics.type_hints_coverage < 0.7:
            recommendations.append("🔧 提升类型提示覆盖率，目标>70%")

        if metrics.docstring_coverage < 0.8:
            recommendations.append("📝 完善文档字符串覆盖，目标>80%")

        if metrics.complexity_average > 8:
            recommendations.append("🔄 重构高复杂度函数，平均复杂度控制在8以内")

        if metrics.violations_count > 100:
            recommendations.append("🎯 修复代码规范问题，减少违规数量")

        if score < 70:
            recommendations.append("🚀 启动全面代码质量改进专项行动")

        return recommendations

    def save_report(self, report: Dict[str, Any], output_file: str):
        """保存报告"""
        import json

        # 转换报告为可序列化的格式
        serializable_report = self._make_serializable(report)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)

    def _make_serializable(self, obj: Any) -> Any:
        """将对象转换为可序列化的格式"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # 处理数据类对象
            return self._make_serializable(vars(obj))
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # 转换为字符串
            return str(obj)

        logger.info(f"质量报告已保存到: {output_file}")


def main():
    """主函数"""
    # 只分析src目录
    src_path = project_root / "src"
    if not src_path.exists():
        print(f"❌ src目录不存在: {src_path}")
        return

    analyzer = CodeQualityAnalyzer(str(src_path))

    # 分析项目
    report = analyzer.analyze_project()

    # 保存报告
    output_file = "code_quality_assessment_report.json"
    analyzer.save_report(report, output_file)

    # 打印摘要
    metrics = report['quality_metrics']
    summary = report['summary']

    print("🎯 代码质量评估报告")
    print("=" * 60)
    print(f"📊 总体评分: {summary['overall_quality_score']}/100 ({summary['grade']})")
    print(f"📁 分析文件: {metrics['python_files']} 个")
    print(f"📝 总行数: {metrics['total_lines']} 行")
    print(f"🔧 函数数量: {metrics['functions_count']} 个")
    print(f"🏗️  类数量: {metrics['classes_count']} 个")
    print(f"🏷️  类型提示覆盖: {metrics['type_hints_coverage']}%")
    print(f"📖 文档覆盖: {metrics['docstring_coverage']}%")
    print(f"🔀 平均复杂度: {metrics['complexity_average']}")
    print(f"⚠️  规范问题: {metrics['violations_count']} 个")
    print()

    print("💡 改进建议:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")
    print()

    print(f"📄 详细报告已保存: {output_file}")


if __name__ == "__main__":
    main()

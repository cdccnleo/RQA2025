#!/usr/bin/env python3
"""
AI智能化代码分析器 - RQA2025项目专用
用于对代码进行智能化分析和审查

作者: AI Assistant
版本: 2.1.0
更新时间: 2025-09-27
"""

import os
import ast
import re
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path

# 设置编码，确保中文正确显示
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


class AnalysisSeverity(Enum):
    """分析严重程度"""
    CRITICAL = "critical"  # 严重问题
    HIGH = "high"         # 高优先级
    MEDIUM = "medium"     # 中等优先级
    LOW = "low"          # 低优先级
    INFO = "info"        # 信息提示


class AnalysisCategory(Enum):
    """分析类别"""
    ARCHITECTURE = "architecture"     # 架构设计
    PERFORMANCE = "performance"       # 性能优化
    SECURITY = "security"            # 安全问题
    MAINTAINABILITY = "maintainability"  # 可维护性
    RELIABILITY = "reliability"       # 可靠性
    COMPLIANCE = "compliance"        # 合规性
    BEST_PRACTICES = "best_practices"  # 最佳实践


@dataclass
class AnalysisIssue:
    """分析问题"""
    file_path: str
    line_number: int
    category: AnalysisCategory
    severity: AnalysisSeverity
    title: str
    description: str
    suggestion: str
    code_snippet: str = ""
    confidence: float = 1.0  # 置信度 0-1


@dataclass
class FileAnalysis:
    """文件分析结果"""
    file_path: str
    language: str
    lines_of_code: int
    complexity_score: float
    maintainability_index: float
    issues: List[AnalysisIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ModuleAnalysis:
    """模块分析结果"""
    module_path: str
    files: List[FileAnalysis] = field(default_factory=list)
    total_lines: int = 0
    total_complexity: float = 0
    average_maintainability: float = 0
    issues_by_category: Dict[str, int] = field(default_factory=dict)
    issues_by_severity: Dict[str, int] = field(default_factory=dict)


class AIIntelligentCodeAnalyzer:
    """AI智能化代码分析器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analysis_results: Dict[str, ModuleAnalysis] = {}

        # 量化交易系统专用规则
        self.quantitative_trading_rules = {
            'async_patterns': [
                r'await\s+\w+\([^)]*\)',  # 异步调用
                r'asyncio\.\w+',          # asyncio使用
                r'async\s+def',           # 异步函数定义
            ],
            'performance_patterns': [
                r'for\s+\w+\s+in\s+range',  # 循环性能
                r'\.append\([^)]*\)',       # 列表操作
                r'dict\([^)]*\)',           # 字典创建
            ],
            'security_patterns': [
                r'exec\([^)]*\)',          # 代码执行
                r'eval\([^)]*\)',          # 表达式求值
                r'pickle\.\w+',            # pickle序列化
            ],
            'architecture_patterns': [
                r'class\s+\w+.*:',         # 类定义
                r'def\s+\w+.*:',           # 函数定义
                r'import\s+\w+',           # 导入语句
            ]
        }

        # 健康检查系统专用规则
        self.health_check_rules = {
            'health_check_methods': [
                r'def\s+(check_|health_|monitor_|validate_)',
                r'async\s+def\s+(check_|health_|monitor_|validate_)',
            ],
            'error_handling': [
                r'try:', r'except\s+\w+:',
                r'finally:', r'raise\s+\w+',
            ],
            'logging_patterns': [
                r'logger\.\w+\([^)]*\)',
                r'logging\.\w+\([^)]*\)',
            ],
            'metrics_patterns': [
                r'metrics?\.\w+\([^)]*\)',
                r'counter\.\w+\([^)]*\)',
                r'gauge\.\w+\([^)]*\)',
            ]
        }

    def analyze_module(self, module_path: str) -> ModuleAnalysis:
        """分析指定模块"""
        print(f"🔍 开始分析模块: {module_path}")

        module_analysis = ModuleAnalysis(module_path=module_path)
        module_full_path = self.project_root / module_path

        if not module_full_path.exists():
            print(f"❌ 模块路径不存在: {module_full_path}")
            return module_analysis

        # 检查是文件还是目录
        if module_full_path.is_file() and module_full_path.suffix == '.py':
            # 单个文件
            file_analysis = self._analyze_file(module_full_path)
            module_analysis.files.append(file_analysis)

            # 汇总模块统计
            module_analysis.total_lines += file_analysis.lines_of_code
            module_analysis.total_complexity += file_analysis.complexity_score
            module_analysis.average_maintainability = (
                module_analysis.average_maintainability + file_analysis.maintainability_index
            ) / len(module_analysis.files) if module_analysis.files else 0

            # 汇总问题
            for issue in file_analysis.issues:
                severity_key = issue.severity.value
                category_key = issue.category.value
                module_analysis.issues_by_severity[severity_key] = module_analysis.issues_by_severity.get(
                    severity_key, 0) + 1
                module_analysis.issues_by_category[category_key] = module_analysis.issues_by_category.get(
                    category_key, 0) + 1
        else:
            # 递归分析所有Python文件
            for file_path in module_full_path.rglob('*.py'):
                if file_path.is_file() and not file_path.name.startswith('__'):
                    file_analysis = self._analyze_file(file_path)
                    module_analysis.files.append(file_analysis)

                    # 汇总模块统计
                    module_analysis.total_lines += file_analysis.lines_of_code
                    module_analysis.total_complexity += file_analysis.complexity_score

                    # 统计问题
                    for issue in file_analysis.issues:
                        category_key = issue.category.value
                        severity_key = issue.severity.value
                        module_analysis.issues_by_category[category_key] = \
                            module_analysis.issues_by_category.get(category_key, 0) + 1
                        module_analysis.issues_by_severity[severity_key] = \
                            module_analysis.issues_by_severity.get(severity_key, 0) + 1

        # 计算平均可维护性指数
        if module_analysis.files:
            maintainability_scores = [
                f.maintainability_index for f in module_analysis.files if f.maintainability_index > 0]
            if maintainability_scores:
                module_analysis.average_maintainability = sum(
                    maintainability_scores) / len(maintainability_scores)

        self.analysis_results[module_path] = module_analysis
        return module_analysis

    def _analyze_file(self, file_path: Path) -> FileAnalysis:
        """分析单个文件"""
        file_analysis = FileAnalysis(
            file_path=str(file_path.relative_to(self.project_root)),
            language="python",
            lines_of_code=0,
            complexity_score=0.0,
            maintainability_index=0.0
        )

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            lines = content.split('\n')
            file_analysis.lines_of_code = len([line for line in lines if line.strip()])

            # 计算复杂度
            file_analysis.complexity_score = self._calculate_complexity(content)

            # 计算可维护性指数
            file_analysis.maintainability_index = self._calculate_maintainability_index(content)

            # 分析代码问题
            file_analysis.issues = self._analyze_code_issues(content, str(file_path))

            # 提取依赖
            file_analysis.dependencies = self._extract_dependencies(content)

            # 计算额外指标
            file_analysis.metrics = self._calculate_metrics(content)

        except Exception as e:
            print(f"❌ 分析文件失败 {file_path}: {e}")
            file_analysis.issues.append(AnalysisIssue(
                file_path=str(file_path),
                line_number=0,
                category=AnalysisCategory.RELIABILITY,
                severity=AnalysisSeverity.HIGH,
                title="文件分析失败",
                description=f"无法分析文件内容: {str(e)}",
                suggestion="检查文件编码和语法正确性"
            ))

        return file_analysis

    def _calculate_complexity(self, content: str) -> float:
        """计算代码复杂度"""
        complexity = 0

        # 基于AST的复杂度计算
        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # 函数复杂度基于参数数量和内部结构
                    complexity += 1 + len(node.args.args) * 0.5

                    # 检查内部控制流
                    for child in ast.walk(node):
                        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                            complexity += 0.5
                        elif isinstance(child, ast.With):
                            complexity += 0.2

                elif isinstance(node, ast.ClassDef):
                    complexity += 2  # 类定义增加复杂度

        except SyntaxError:
            complexity = 10  # 语法错误的文件给高复杂度评分

        return complexity

    def _calculate_maintainability_index(self, content: str) -> float:
        """计算可维护性指数 (0-100)"""
        if not content.strip():
            return 0

        lines = content.split('\n')
        loc = len([line for line in lines if line.strip()])

        # 简化的MI计算
        # MI = 171 - 5.2 * ln(Halstead Volume) - 0.23 * CC - 16.2 * ln(LOC)
        # 这里使用简化版本

        base_score = 100

        # 基于代码行数的惩罚
        if loc > 500:
            base_score -= 20
        elif loc > 200:
            base_score -= 10
        elif loc > 100:
            base_score -= 5

        # 基于复杂度惩罚
        complexity = self._calculate_complexity(content)
        if complexity > 20:
            base_score -= 15
        elif complexity > 10:
            base_score -= 8
        elif complexity > 5:
            base_score -= 3

        # 基于代码质量的奖励
        if '"""' in content or "'''" in content:  # 有文档字符串
            base_score += 5

        if len(re.findall(r'#\s*\w+', content)) > 5:  # 有注释
            base_score += 3

        return max(0, min(100, base_score))

    def _analyze_code_issues(self, content: str, file_path: str) -> List[AnalysisIssue]:
        """分析代码问题"""
        issues = []
        lines = content.split('\n')

        # 量化交易系统专用检查
        issues.extend(self._check_quantitative_trading_patterns(content, file_path))

        # 健康检查系统专用检查
        issues.extend(self._check_health_check_patterns(content, file_path))

        # 通用代码质量检查
        issues.extend(self._check_general_code_quality(content, lines, file_path))

        # 架构一致性检查
        issues.extend(self._check_architecture_consistency(content, file_path))

        return issues

    def _check_quantitative_trading_patterns(self, content: str, file_path: str) -> List[AnalysisIssue]:
        """检查量化交易模式"""
        issues = []

        # 检查异步模式使用
        async_count = sum(1 for pattern in self.quantitative_trading_rules['async_patterns']
                          for match in re.finditer(pattern, content))

        if 'health' in file_path.lower() and async_count == 0:
            issues.append(AnalysisIssue(
                file_path=file_path,
                line_number=1,
                category=AnalysisCategory.PERFORMANCE,
                severity=AnalysisSeverity.MEDIUM,
                title="缺少异步处理",
                description="健康检查系统应该使用异步处理以提高并发性能",
                suggestion="考虑使用async/await模式处理健康检查"
            ))

        # 检查性能模式
        performance_issues = 0
        for pattern in self.quantitative_trading_rules['performance_patterns']:
            performance_issues += len(re.findall(pattern, content))

        if performance_issues > 10:
            issues.append(AnalysisIssue(
                file_path=file_path,
                line_number=1,
                category=AnalysisCategory.PERFORMANCE,
                severity=AnalysisSeverity.LOW,
                title="潜在性能优化点",
                description=f"检测到{performance_issues}个潜在的性能优化点",
                suggestion="考虑使用numpy向量化操作或缓存机制"
            ))

        return issues

    def _check_health_check_patterns(self, content: str, file_path: str) -> List[AnalysisIssue]:
        """检查健康检查模式"""
        issues = []

        # 检查健康检查方法命名
        health_methods = 0
        for pattern in self.health_check_rules['health_check_methods']:
            health_methods += len(re.findall(pattern, content))

        if health_methods == 0 and 'health' in file_path.lower():
            issues.append(AnalysisIssue(
                file_path=file_path,
                line_number=1,
                category=AnalysisCategory.ARCHITECTURE,
                severity=AnalysisSeverity.HIGH,
                title="缺少健康检查方法",
                description="健康检查模块应该包含标准化的健康检查方法",
                suggestion="添加check_、health_、monitor_或validate_前缀的方法"
            ))

        # 检查错误处理
        error_handling_count = sum(1 for pattern in self.health_check_rules['error_handling']
                                   for match in re.finditer(pattern, content))

        if error_handling_count == 0:
            issues.append(AnalysisIssue(
                file_path=file_path,
                line_number=1,
                category=AnalysisCategory.RELIABILITY,
                severity=AnalysisSeverity.MEDIUM,
                title="缺少错误处理",
                description="代码缺少必要的错误处理机制",
                suggestion="添加try-except块处理异常情况"
            ))

        # 检查日志记录
        logging_count = sum(1 for pattern in self.health_check_rules['logging_patterns']
                            for match in re.finditer(pattern, content))

        if logging_count == 0 and 'health' in file_path.lower():
            issues.append(AnalysisIssue(
                file_path=file_path,
                line_number=1,
                category=AnalysisCategory.BEST_PRACTICES,
                severity=AnalysisSeverity.LOW,
                title="缺少日志记录",
                description="健康检查系统应该有适当的日志记录",
                suggestion="添加关键操作的日志记录"
            ))

        return issues

    def _check_general_code_quality(self, content: str, lines: List[str], file_path: str) -> List[AnalysisIssue]:
        """检查通用代码质量"""
        issues = []

        # 检查过长的行
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(AnalysisIssue(
                    file_path=file_path,
                    line_number=i,
                    category=AnalysisCategory.BEST_PRACTICES,
                    severity=AnalysisSeverity.LOW,
                    title="过长代码行",
                    description=f"代码行长度超过120字符: {len(line)}",
                    suggestion="将长行拆分为多行以提高可读性",
                    code_snippet=line.strip()
                ))

        # 检查过长的方法
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_lines = len(content.split('\n')[node.lineno-1:node.end_lineno])
                    if method_lines > 50:
                        issues.append(AnalysisIssue(
                            file_path=file_path,
                            line_number=node.lineno,
                            category=AnalysisCategory.MAINTAINABILITY,
                            severity=AnalysisSeverity.MEDIUM,
                            title="方法过长",
                            description=f"方法'{node.name}'过长: {method_lines}行",
                            suggestion="考虑将方法拆分为更小的函数"
                        ))
        except SyntaxError:
            pass

        # 检查硬编码的魔法数字
        magic_numbers = re.findall(r'\b\d{2,}\b', content)
        if len(magic_numbers) > 5:
            issues.append(AnalysisIssue(
                file_path=file_path,
                line_number=1,
                category=AnalysisCategory.MAINTAINABILITY,
                severity=AnalysisSeverity.LOW,
                title="过多魔法数字",
                description=f"检测到{len(magic_numbers)}个可能的魔法数字",
                suggestion="将魔法数字定义为常量"
            ))

        return issues

    def _check_architecture_consistency(self, content: str, file_path: str) -> List[AnalysisIssue]:
        """检查架构一致性"""
        issues = []

        # 检查是否遵循了统一基础设施集成层的模式
        if 'infrastructure' in file_path and 'health' in file_path:
            # 检查是否使用了统一的接口
            if 'UnifiedInterface' not in content and 'IHealthChecker' not in content:
                issues.append(AnalysisIssue(
                    file_path=file_path,
                    line_number=1,
                    category=AnalysisCategory.ARCHITECTURE,
                    severity=AnalysisSeverity.MEDIUM,
                    title="架构接口不一致",
                    description="基础设施层组件应该使用统一的接口",
                    suggestion="继承或实现UnifiedInterface或相应的接口类"
                ))

            # 检查是否使用了适配器模式
            if 'Adapter' not in content and 'adapter' not in file_path:
                issues.append(AnalysisIssue(
                    file_path=file_path,
                    line_number=1,
                    category=AnalysisCategory.ARCHITECTURE,
                    severity=AnalysisSeverity.LOW,
                    title="缺少适配器模式",
                    description="基础设施层应该使用适配器模式实现集成",
                    suggestion="考虑实现相应的适配器类"
                ))

        return issues

    def _extract_dependencies(self, content: str) -> List[str]:
        """提取文件依赖"""
        dependencies = []

        # 提取import语句
        import_pattern = r'^(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))'
        matches = re.findall(import_pattern, content, re.MULTILINE)

        for match in matches:
            module = match[0] or match[1]
            if module and not module.startswith(('typing', 'dataclasses', 'abc')):
                dependencies.append(module.split('.')[0])

        return list(set(dependencies))

    def _calculate_metrics(self, content: str) -> Dict[str, Any]:
        """计算额外指标"""
        metrics = {}

        try:
            tree = ast.parse(content)

            # 计算类和函数数量
            classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
            functions = sum(1 for node in ast.walk(tree) if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef)))

            metrics['classes'] = classes
            metrics['functions'] = functions
            metrics['total_symbols'] = classes + functions

            # 计算注释率
            lines = content.split('\n')
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            docstring_lines = len(re.findall(
                r'"""[\s\S]*?"""', content)) + len(re.findall(r"'''[\s\S]*?'''", content))
            total_lines = len([line for line in lines if line.strip()])

            if total_lines > 0:
                metrics['comment_ratio'] = (comment_lines + docstring_lines) / total_lines
            else:
                metrics['comment_ratio'] = 0

            # 计算导入数量
            imports = len(re.findall(r'^(?:from\s+|\bimport\s+)', content, re.MULTILINE))
            metrics['imports'] = imports

        except SyntaxError:
            metrics['parse_error'] = True

        return metrics

    def generate_report(self, module_analysis: ModuleAnalysis) -> str:
        """生成分析报告"""
        report = []
        report.append("=" * 80)
        report.append(f"🎯 AI智能化代码分析报告 - {module_analysis.module_path}")
        report.append("=" * 80)
        report.append("")

        # 总体统计
        report.append("📊 总体统计")
        report.append(f"   📁 文件数量: {len(module_analysis.files)}")
        report.append(f"   📝 总代码行数: {module_analysis.total_lines}")
        report.append(f"   🔄 总复杂度: {module_analysis.total_complexity:.2f}")
        report.append(f"   📈 平均可维护性: {module_analysis.average_maintainability:.2f}")
        report.append("")

        # 问题统计
        if module_analysis.issues_by_severity or module_analysis.issues_by_category:
            report.append("🚨 问题统计")

            # 按严重程度统计
            severity_order = [AnalysisSeverity.CRITICAL, AnalysisSeverity.HIGH,
                              AnalysisSeverity.MEDIUM, AnalysisSeverity.LOW, AnalysisSeverity.INFO]
            for severity in severity_order:
                count = module_analysis.issues_by_severity.get(severity.value, 0)
                if count > 0:
                    report.append(f"   {severity.value.upper()}: {count}")

            report.append("")

            # 按类别统计
            for category, count in module_analysis.issues_by_category.items():
                report.append(f"   {category}: {count}")

            report.append("")

        # 文件详情
        report.append("📋 文件详情")
        for file in sorted(module_analysis.files, key=lambda x: x.lines_of_code, reverse=True):
            report.append(f"   📄 {file.file_path}")
            report.append(f"      代码行数: {file.lines_of_code}")
            report.append(f"      🔄 复杂度: {file.complexity_score:.2f}")
            report.append(f"      📈 可维护性: {file.maintainability_index:.2f}")
            if file.issues:
                report.append(f"      问题数量: {len(file.issues)}")
                # 显示前3个问题
                for issue in file.issues[:3]:
                    report.append(f"         - [{issue.severity.value.upper()}] {issue.title}")
            report.append("")

        # 详细问题列表
        all_issues = []
        for file in module_analysis.files:
            all_issues.extend(file.issues)

        if all_issues:
            report.append("🔍 详细问题列表")
            # 按严重程度排序
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3, 'info': 4}
            sorted_issues = sorted(all_issues, key=lambda x: (
                severity_order.get(x.severity.value, 5), x.file_path))

            for issue in sorted_issues:
                report.append(
                    f"   📍 [{issue.severity.value.upper()}] {issue.file_path}:{issue.line_number}")
                report.append(f"      💡 {issue.title}")
                report.append(f"      📝 {issue.description}")
                report.append(f"      💡 {issue.suggestion}")
                if issue.code_snippet:
                    report.append(f"      ```python\n      {issue.code_snippet}\n      ```")
                report.append("")

        # 架构建议
        report.append("🏗️ 架构建议")
        if module_analysis.average_maintainability < 70:
            report.append("   ⚠️ 可维护性指数较低，建议进行重构")
        if module_analysis.total_complexity > 100:
            report.append("   ⚠️ 系统复杂度较高，建议模块化拆分")
        if not any('async' in str(file.metrics.get('functions', 0)) for file in module_analysis.files):
            report.append("   💡 建议增加异步处理支持")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def save_report(self, module_analysis: ModuleAnalysis, output_file: str):
        """保存报告到文件"""
        report = self.generate_report(module_analysis)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✅ 分析报告已保存到: {output_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='AI智能化代码分析器 - RQA2025项目专用')
    parser.add_argument('module_path', help='要分析的模块路径')
    parser.add_argument('--quiet', '-q', action='store_true', help='安静模式，不输出到控制台')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--no-save', action='store_true', help='不保存报告到文件')

    args = parser.parse_args()

    module_path = args.module_path

    # 创建分析器
    analyzer = AIIntelligentCodeAnalyzer(".")

    # 分析模块
    analysis_result = analyzer.analyze_module(module_path)

    # 生成报告
    report = analyzer.generate_report(analysis_result)

    # 输出报告（除非是quiet模式）
    if not args.quiet:
        print(report)

    # 保存报告
    if not args.no_save:
        if args.output:
            output_file = args.output
        else:
            output_file = f"analysis_report_{module_path.replace('/', '_').replace(os.sep, '_')}.md"
        analyzer.save_report(analysis_result, output_file)


if __name__ == "__main__":
    main()

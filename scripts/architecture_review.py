#!/usr/bin/env python3
"""
架构审查工具

提供全面的架构审查功能，包括：
- 代码结构审查
- 依赖关系分析
- 设计模式检查
- 架构一致性验证
- 性能和安全审查
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import networkx as nx
from dataclasses import dataclass


@dataclass
class ReviewIssue:
    """审查问题"""
    category: str
    severity: str  # critical, high, medium, low
    title: str
    description: str
    file_path: str
    line_number: Optional[int]
    suggestion: str
    rule_id: str


@dataclass
class ReviewResult:
    """审查结果"""
    total_files: int
    issues: List[ReviewIssue]
    summary: Dict[str, int]
    recommendations: List[str]


class ArchitectureReviewer:
    """架构审查器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.config_dir = self.project_root / "config"

        # 审查规则
        self.rules = self._load_review_rules()

    def _load_review_rules(self) -> Dict[str, Any]:
        """加载审查规则"""
        return {
            "structure_rules": {
                "max_file_size": 500,  # 最大文件行数
                "max_class_size": 200,  # 最大类行数
                "max_function_size": 50,  # 最大函数行数
                "max_module_depth": 3,  # 最大模块深度
            },
            "dependency_rules": {
                "allow_circular_deps": False,
                "max_imports_per_file": 15,
                "forbidden_imports": ["import *"],
            },
            "design_rules": {
                "require_interfaces": True,
                "max_inheritance_depth": 3,
                "require_docstrings": True,
                "naming_conventions": {
                    "classes": "PascalCase",
                    "functions": "snake_case",
                    "variables": "snake_case",
                    "constants": "UPPER_CASE"
                }
            },
            "architecture_rules": {
                "layer_dependencies": {
                    "infrastructure": ["core"],
                    "data": ["core", "infrastructure"],
                    "features": ["core", "infrastructure", "data"],
                    "ml": ["core", "infrastructure", "features"],
                    "gateway": ["core", "infrastructure"],
                    "backtest": ["core", "infrastructure", "features", "ml"],
                    "trading": ["core", "infrastructure", "features", "ml", "backtest"],
                    "risk": ["core", "infrastructure", "features", "ml"],
                    "engine": ["core", "infrastructure", "features", "ml", "trading", "risk"]
                }
            }
        }

    def perform_comprehensive_review(self) -> ReviewResult:
        """执行全面架构审查"""
        print("🔍 开始架构审查...")

        all_issues = []
        total_files = 0

        # 1. 结构审查
        structure_issues = self._review_code_structure()
        all_issues.extend(structure_issues)

        # 2. 依赖关系审查
        dependency_issues = self._review_dependencies()
        all_issues.extend(dependency_issues)

        # 3. 设计模式审查
        design_issues = self._review_design_patterns()
        all_issues.extend(design_issues)

        # 4. 架构一致性审查
        architecture_issues = self._review_architecture_consistency()
        all_issues.extend(architecture_issues)

        # 5. 性能和安全审查
        performance_issues = self._review_performance_security()
        all_issues.extend(performance_issues)

        # 统计文件数
        for py_file in self.src_dir.rglob("*.py"):
            if not py_file.name.startswith("_"):
                total_files += 1

        # 生成摘要和建议
        summary = self._generate_summary(all_issues)
        recommendations = self._generate_recommendations(all_issues)

        result = ReviewResult(
            total_files=total_files,
            issues=all_issues,
            summary=summary,
            recommendations=recommendations
        )

        print(f"📊 审查完成，发现 {len(all_issues)} 个问题")
        return result

    def _review_code_structure(self) -> List[ReviewIssue]:
        """审查代码结构"""
        issues = []

        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.splitlines()

                # 检查文件大小
                if len(lines) > self.rules["structure_rules"]["max_file_size"]:
                    issues.append(ReviewIssue(
                        category="结构问题",
                        severity="medium",
                        title="文件过大",
                        description=f"文件行数({len(lines)})超过限制({self.rules['structure_rules']['max_file_size']})",
                        file_path=str(py_file.relative_to(self.project_root)),
                        line_number=None,
                        suggestion="考虑将文件拆分为多个模块",
                        rule_id="STRUCTURE_001"
                    ))

                # 检查导入数量
                if len(lines) > 0:
                    import_count = sum(
                        1 for line in lines if line.strip().startswith(('import ', 'from ')))
                    if import_count > self.rules["dependency_rules"]["max_imports_per_file"]:
                        issues.append(ReviewIssue(
                            category="依赖问题",
                            severity="low",
                            title="过多导入",
                            description=f"文件导入数量({import_count})过多",
                            file_path=str(py_file.relative_to(self.project_root)),
                            line_number=None,
                            suggestion="考虑使用__all__或重新组织导入",
                            rule_id="DEPENDENCY_001"
                        ))

            except Exception as e:
                issues.append(ReviewIssue(
                    category="结构问题",
                    severity="low",
                    title="文件读取错误",
                    description=f"无法读取文件: {e}",
                    file_path=str(py_file.relative_to(self.project_root)),
                    line_number=None,
                    suggestion="检查文件编码和权限",
                    rule_id="STRUCTURE_002"
                ))

        return issues

    def _review_dependencies(self) -> List[ReviewIssue]:
        """审查依赖关系"""
        issues = []

        # 构建依赖图
        dependency_graph = self._build_dependency_graph()

        # 检查循环依赖
        try:
            cycles = list(nx.simple_cycles(dependency_graph))
            for cycle in cycles:
                issues.append(ReviewIssue(
                    category="依赖问题",
                    severity="high",
                    title="循环依赖",
                    description=f"检测到循环依赖: {' -> '.join(cycle)}",
                    file_path="多个文件",
                    line_number=None,
                    suggestion="重构代码以消除循环依赖",
                    rule_id="DEPENDENCY_002"
                ))
        except:
            pass

        # 检查架构层依赖违规
        layer_deps = self._analyze_layer_dependencies()
        for violation in layer_deps:
            issues.append(ReviewIssue(
                category="架构问题",
                severity="high",
                title="架构层依赖违规",
                description=violation["description"],
                file_path=violation["file"],
                line_number=None,
                suggestion=violation["suggestion"],
                rule_id="ARCHITECTURE_001"
            ))

        return issues

    def _review_design_patterns(self) -> List[ReviewIssue]:
        """审查设计模式"""
        issues = []

        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)

                for node in ast.walk(tree):
                    # 检查类定义
                    if isinstance(node, ast.ClassDef):
                        # 检查继承深度
                        if len(node.bases) > self.rules["design_rules"]["max_inheritance_depth"]:
                            issues.append(ReviewIssue(
                                category="设计问题",
                                severity="medium",
                                title="过度继承",
                                description=f"类 {node.name} 继承深度过深",
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=node.lineno,
                                suggestion="考虑使用组合而不是继承",
                                rule_id="DESIGN_001"
                            ))

                        # 检查文档字符串
                        if not ast.get_docstring(node) and self.rules["design_rules"]["require_docstrings"]:
                            issues.append(ReviewIssue(
                                category="设计问题",
                                severity="low",
                                title="缺少类文档",
                                description=f"类 {node.name} 缺少文档字符串",
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=node.lineno,
                                suggestion="为类添加文档字符串",
                                rule_id="DESIGN_002"
                            ))

                    # 检查函数定义
                    elif isinstance(node, ast.FunctionDef):
                        # 检查函数大小
                        if len(node.body) > self.rules["structure_rules"]["max_function_size"]:
                            issues.append(ReviewIssue(
                                category="结构问题",
                                severity="medium",
                                title="函数过长",
                                description=f"函数 {node.name} 过长",
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=node.lineno,
                                suggestion="将函数拆分为多个子函数",
                                rule_id="STRUCTURE_003"
                            ))

                        # 检查文档字符串
                        if not ast.get_docstring(node) and self.rules["design_rules"]["require_docstrings"]:
                            issues.append(ReviewIssue(
                                category="设计问题",
                                severity="low",
                                title="缺少函数文档",
                                description=f"函数 {node.name} 缺少文档字符串",
                                file_path=str(py_file.relative_to(self.project_root)),
                                line_number=node.lineno,
                                suggestion="为函数添加文档字符串",
                                rule_id="DESIGN_003"
                            ))

            except:
                continue

        return issues

    def _review_architecture_consistency(self) -> List[ReviewIssue]:
        """审查架构一致性"""
        issues = []

        # 检查架构文档与代码的一致性
        doc_consistency = self._check_doc_consistency()
        issues.extend(doc_consistency)

        # 检查事件类型定义一致性
        event_consistency = self._check_event_consistency()
        issues.extend(event_consistency)

        return issues

    def _review_performance_security(self) -> List[ReviewIssue]:
        """审查性能和安全"""
        issues = []

        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查SQL注入风险
                if "execute(" in content and ("%" in content or "+" in content):
                    issues.append(ReviewIssue(
                        category="安全问题",
                        severity="high",
                        title="SQL注入风险",
                        description="检测到可能的SQL注入风险",
                        file_path=str(py_file.relative_to(self.project_root)),
                        line_number=None,
                        suggestion="使用参数化查询或ORM",
                        rule_id="SECURITY_001"
                    ))

                # 检查硬编码密钥
                if "password" in content.lower() and ("123" in content or "admin" in content):
                    issues.append(ReviewIssue(
                        category="安全问题",
                        severity="high",
                        title="硬编码凭据",
                        description="检测到硬编码的凭据",
                        file_path=str(py_file.relative_to(self.project_root)),
                        line_number=None,
                        suggestion="使用环境变量或配置文件",
                        rule_id="SECURITY_002"
                    ))

                # 检查性能问题
                if "for " in content and "range(" in content and "10000" in content:
                    issues.append(ReviewIssue(
                        category="性能问题",
                        severity="medium",
                        title="潜在性能问题",
                        description="检测到大循环可能影响性能",
                        file_path=str(py_file.relative_to(self.project_root)),
                        line_number=None,
                        suggestion="考虑使用向量化操作或分页",
                        rule_id="PERFORMANCE_001"
                    ))

            except:
                continue

        return issues

    def _build_dependency_graph(self) -> nx.DiGraph:
        """构建依赖图"""
        graph = nx.DiGraph()

        for py_file in self.src_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            module_name = str(py_file.relative_to(self.src_dir)
                              ).replace("/", ".").replace(".py", "")
            graph.add_node(module_name)

        return graph

    def _analyze_layer_dependencies(self) -> List[Dict[str, str]]:
        """分析层级依赖"""
        violations = []

        # 这里应该实现具体的层级依赖分析逻辑
        # 由于需要具体的代码分析，这里提供一个示例框架

        return violations

    def _check_doc_consistency(self) -> List[ReviewIssue]:
        """检查文档一致性"""
        issues = []

        # 检查架构文档是否存在
        arch_doc = self.project_root / "docs" / "architecture" / "BUSINESS_PROCESS_DRIVEN_ARCHITECTURE.md"
        if not arch_doc.exists():
            issues.append(ReviewIssue(
                category="文档问题",
                severity="high",
                title="缺少架构文档",
                description="项目缺少主要的架构设计文档",
                file_path="docs/architecture/",
                line_number=None,
                suggestion="创建架构设计文档",
                rule_id="DOC_001"
            ))

        return issues

    def _check_event_consistency(self) -> List[ReviewIssue]:
        """检查事件一致性"""
        issues = []

        # 这里应该检查事件类型定义的一致性

        return issues

    def _generate_summary(self, issues: List[ReviewIssue]) -> Dict[str, int]:
        """生成问题摘要"""
        summary = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }

        for issue in issues:
            if issue.severity in summary:
                summary[issue.severity] += 1

        return summary

    def _generate_recommendations(self, issues: List[ReviewIssue]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if len(issues) == 0:
            recommendations.append("✅ 架构审查通过，代码质量良好")
            return recommendations

        # 按严重程度排序
        critical_issues = [i for i in issues if i.severity == "critical"]
        high_issues = [i for i in issues if i.severity == "high"]

        if critical_issues:
            recommendations.append(f"🚨 紧急修复 {len(critical_issues)} 个严重问题")
        if high_issues:
            recommendations.append(f"⚠️ 优先处理 {len(high_issues)} 个重要问题")

        # 具体建议
        categories = set(i.category for i in issues)
        for category in categories:
            cat_issues = [i for i in issues if i.category == category]
            recommendations.append(f"📋 {category}: 发现 {len(cat_issues)} 个问题需要处理")

        return recommendations

    def generate_review_report(self, result: ReviewResult) -> str:
        """生成审查报告"""
        report = f"""# 架构审查报告

## 📊 审查概览

**审查时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**审查文件**: {result.total_files} 个
**发现问题**: {len(result.issues)} 个

### 问题分布

| 严重程度 | 数量 | 占比 |
|---------|------|------|
| 🚨 严重 | {result.summary['critical']} | {result.summary['critical']/max(len(result.issues),1)*100:.1f}% |
| ⚠️ 重要 | {result.summary['high']} | {result.summary['high']/max(len(result.issues),1)*100:.1f}% |
| 📋 中等 | {result.summary['medium']} | {result.summary['medium']/max(len(result.issues),1)*100:.1f}% |
| ℹ️ 轻微 | {result.summary['low']} | {result.summary['low']/max(len(result.issues),1)*100:.1f}% |

## 🔧 改进建议

"""

        for rec in result.recommendations:
            report += f"- {rec}\n"

        report += f"""

## 📋 详细问题列表

"""

        # 按严重程度和类别分组
        severity_order = {"critical": "🚨 严重", "high": "⚠️ 重要", "medium": "📋 中等", "low": "ℹ️ 轻微"}
        categories = {}

        for issue in result.issues:
            key = f"{issue.category}"
            if key not in categories:
                categories[key] = []
            categories[key].append(issue)

        for category, cat_issues in categories.items():
            report += f"### {category}\n\n"

            # 按严重程度排序
            for severity in ["critical", "high", "medium", "low"]:
                sev_issues = [i for i in cat_issues if i.severity == severity]
                if not sev_issues:
                    continue

                report += f"#### {severity_order[severity]}\n\n"

                for issue in sev_issues:
                    report += f"**{issue.title}** ({issue.rule_id})\n"
                    report += f"- **文件**: `{issue.file_path}`\n"
                    if issue.line_number:
                        report += f"- **行号**: {issue.line_number}\n"
                    report += f"- **描述**: {issue.description}\n"
                    report += f"- **建议**: {issue.suggestion}\n\n"

        report += f"""## 📈 质量指标

### 代码质量评分
- **结构完整性**: {self._calculate_structure_score(result):.1f}/10
- **依赖合理性**: {self._calculate_dependency_score(result):.1f}/10
- **设计规范性**: {self._calculate_design_score(result):.1f}/10
- **架构一致性**: {self._calculate_architecture_score(result):.1f}/10
- **性能安全性**: {self._calculate_security_score(result):.1f}/10

### 综合评分
**总分**: {self._calculate_overall_score(result):.1f}/10

---

**报告生成**: 自动化审查工具
**审查标准**: 基于项目架构规范
**建议处理**: 按严重程度优先处理问题
"""

        return report

    def _calculate_structure_score(self, result: ReviewResult) -> float:
        """计算结构评分"""
        structure_issues = [i for i in result.issues if i.category == "结构问题"]
        return max(0, 10 - len(structure_issues) * 2)

    def _calculate_dependency_score(self, result: ReviewResult) -> float:
        """计算依赖评分"""
        dependency_issues = [i for i in result.issues if i.category == "依赖问题"]
        return max(0, 10 - len(dependency_issues) * 3)

    def _calculate_design_score(self, result: ReviewResult) -> float:
        """计算设计评分"""
        design_issues = [i for i in result.issues if i.category == "设计问题"]
        return max(0, 10 - len(design_issues) * 1.5)

    def _calculate_architecture_score(self, result: ReviewResult) -> float:
        """计算架构评分"""
        arch_issues = [i for i in result.issues if i.category in ["架构问题", "文档问题"]]
        return max(0, 10 - len(arch_issues) * 2)

    def _calculate_security_score(self, result: ReviewResult) -> float:
        """计算安全评分"""
        security_issues = [i for i in result.issues if i.category in ["安全问题", "性能问题"]]
        return max(0, 10 - len(security_issues) * 4)

    def _calculate_overall_score(self, result: ReviewResult) -> float:
        """计算综合评分"""
        scores = [
            self._calculate_structure_score(result),
            self._calculate_dependency_score(result),
            self._calculate_design_score(result),
            self._calculate_architecture_score(result),
            self._calculate_security_score(result)
        ]
        return sum(scores) / len(scores)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='架构审查工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--output', help='输出报告文件')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='报告格式')

    args = parser.parse_args()

    reviewer = ArchitectureReviewer(args.project)
    result = reviewer.perform_comprehensive_review()

    if args.format == 'json':
        report_data = {
            "summary": result.summary,
            "total_files": result.total_files,
            "total_issues": len(result.issues),
            "issues": [
                {
                    "category": issue.category,
                    "severity": issue.severity,
                    "title": issue.title,
                    "description": issue.description,
                    "file_path": issue.file_path,
                    "suggestion": issue.suggestion,
                    "rule_id": issue.rule_id
                }
                for issue in result.issues
            ],
            "recommendations": result.recommendations
        }

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(report_data, ensure_ascii=False, indent=2))

    else:
        report = reviewer.generate_review_report(result)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
        else:
            print(report)


if __name__ == "__main__":
    main()

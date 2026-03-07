#!/usr/bin/env python3
"""
深度架构审计工具

对src目录进行深度审计，检查：
1. 目录内容的架构符合性
2. 文件内容的职责匹配度
3. 导入依赖的合理性
4. 命名规范的统一性
5. 接口设计的标准性
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import re


class DeepArchitectureAuditor:
    """深度架构审计器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"

        # 架构职责映射
        self.layer_responsibilities = {
            "core": {
                "keywords": ["event", "bus", "container", "orchestrator", "process", "integration"],
                "should_not_contain": ["trading", "risk", "ml", "feature", "data"]
            },
            "infrastructure": {
                "keywords": ["config", "cache", "logging", "security", "error", "resource", "health", "utils"],
                "should_not_contain": ["trading", "strategy", "model", "feature"]
            },
            "data": {
                "keywords": ["adapter", "collector", "validator", "quality", "loader", "parser", "source"],
                "should_not_contain": ["trading", "strategy", "model", "feature"]
            },
            "gateway": {
                "keywords": ["api", "gateway", "route", "auth", "rate", "limit", "middleware"],
                "should_not_contain": ["trading", "strategy", "model", "feature"]
            },
            "features": {
                "keywords": ["feature", "engineering", "distributed", "acceleration", "processor", "extract"],
                "should_not_contain": ["trading", "strategy", "model"]
            },
            "ml": {
                "keywords": ["model", "ml", "predict", "train", "inference", "ensemble", "tuning"],
                "should_not_contain": ["trading", "strategy"]
            },
            "backtest": {
                "keywords": ["strategy", "backtest", "analyzer", "evaluation", "performance", "simulation"],
                "should_not_contain": ["production", "live", "real_trading"]
            },
            "risk": {
                "keywords": ["risk", "compliance", "checker", "monitor", "limit", "threshold", "validation"],
                "should_not_contain": ["trading", "order", "execution"]
            },
            "trading": {
                "keywords": ["trading", "order", "execution", "executor", "manager", "broker", "exchange"],
                "should_not_contain": ["backtest", "simulation"]
            },
            "engine": {
                "keywords": ["monitor", "logging", "alert", "dashboard", "metric", "performance", "health"],
                "should_not_contain": ["trading", "order", "strategy", "model"]
            }
        }

        # 接口规范检查
        self.interface_patterns = {
            "abstract_base_class": re.compile(r"class\s+(\w+)\(ABC\):"),
            "abstractmethod": re.compile(r"@abstractmethod"),
            "interface_naming": re.compile(r"class\s+I\w+Component\(ABC\):"),
            "base_implementation": re.compile(r"class\s+Base\w+Component\(I\w+Component\):")
        }

    def perform_deep_audit(self) -> Dict[str, Any]:
        """执行深度架构审计"""
        print("🔍 执行深度架构审计...")

        audit_result = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "layer_audit": {},
            "content_analysis": {},
            "interface_compliance": {},
            "dependency_analysis": {},
            "issues": [],
            "recommendations": []
        }

        # 1. 层级审计
        print("📊 步骤1: 层级职责审计")
        audit_result["layer_audit"] = self._audit_layer_responsibilities()

        # 2. 内容分析
        print("📋 步骤2: 内容职责匹配分析")
        audit_result["content_analysis"] = self._analyze_content_responsibilities()

        # 3. 接口规范检查
        print("🔗 步骤3: 接口规范检查")
        audit_result["interface_compliance"] = self._check_interface_compliance()

        # 4. 依赖分析
        print("⚡ 步骤4: 依赖关系分析")
        audit_result["dependency_analysis"] = self._analyze_dependencies()

        # 5. 生成摘要和建议
        audit_result["summary"] = self._generate_summary(audit_result)
        audit_result["issues"] = self._collect_issues(audit_result)
        audit_result["recommendations"] = self._generate_recommendations(audit_result)

        print(f"✅ 深度审计完成，发现 {len(audit_result['issues'])} 个问题")
        return audit_result

    def _audit_layer_responsibilities(self) -> Dict[str, Any]:
        """审计层级职责"""
        layer_audit = {}

        for layer, responsibilities in self.layer_responsibilities.items():
            layer_path = self.src_dir / layer
            if not layer_path.exists():
                continue

            layer_audit[layer] = {
                "layer": layer,
                "exists": True,
                "file_count": 0,
                "responsibility_match": 0,
                "violation_count": 0,
                "issues": []
            }

            # 检查所有Python文件
            for py_file in layer_path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                layer_audit[layer]["file_count"] += 1

                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()

                    # 检查职责匹配
                    keyword_matches = sum(1 for keyword in responsibilities["keywords"]
                                          if keyword.lower() in content)
                    violation_matches = sum(1 for violation in responsibilities["should_not_contain"]
                                            if violation.lower() in content)

                    layer_audit[layer]["responsibility_match"] += keyword_matches
                    layer_audit[layer]["violation_count"] += violation_matches

                    # 记录严重违规
                    if violation_matches > 0:
                        layer_audit[layer]["issues"].append({
                            "file": str(py_file.relative_to(self.src_dir)),
                            "violations": [v for v in responsibilities["should_not_contain"]
                                           if v.lower() in content],
                            "severity": "high" if violation_matches > 2 else "medium"
                        })

                except Exception as e:
                    layer_audit[layer]["issues"].append({
                        "file": str(py_file.relative_to(self.src_dir)),
                        "error": str(e),
                        "severity": "low"
                    })

        return layer_audit

    def _analyze_content_responsibilities(self) -> Dict[str, Any]:
        """分析内容职责匹配度"""
        content_analysis = {}

        for layer in self.layer_responsibilities.keys():
            layer_path = self.src_dir / layer
            if not layer_path.exists():
                continue

            content_analysis[layer] = {
                "total_files": 0,
                "high_match_files": 0,
                "medium_match_files": 0,
                "low_match_files": 0,
                "no_match_files": 0,
                "detailed_analysis": []
            }

            for py_file in layer_path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                content_analysis[layer]["total_files"] += 1

                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read().lower()

                    # 计算匹配度
                    keywords = self.layer_responsibilities[layer]["keywords"]
                    match_score = sum(1 for keyword in keywords if keyword.lower() in content)
                    max_possible = len(keywords)

                    match_percentage = (match_score / max_possible) * 100 if max_possible > 0 else 0

                    # 分类文件
                    if match_percentage >= 70:
                        content_analysis[layer]["high_match_files"] += 1
                        category = "high"
                    elif match_percentage >= 40:
                        content_analysis[layer]["medium_match_files"] += 1
                        category = "medium"
                    elif match_percentage > 0:
                        content_analysis[layer]["low_match_files"] += 1
                        category = "low"
                    else:
                        content_analysis[layer]["no_match_files"] += 1
                        category = "none"

                    content_analysis[layer]["detailed_analysis"].append({
                        "file": str(py_file.relative_to(self.src_dir)),
                        "match_score": match_percentage,
                        "matched_keywords": [k for k in keywords if k.lower() in content],
                        "category": category
                    })

                except Exception as e:
                    content_analysis[layer]["detailed_analysis"].append({
                        "file": str(py_file.relative_to(self.src_dir)),
                        "error": str(e),
                        "category": "error"
                    })

        return content_analysis

    def _check_interface_compliance(self) -> Dict[str, Any]:
        """检查接口规范符合性"""
        interface_compliance = {}

        for layer in self.layer_responsibilities.keys():
            layer_path = self.src_dir / layer
            if not layer_path.exists():
                continue

            interface_compliance[layer] = {
                "interface_files": 0,
                "base_implementation_files": 0,
                "standard_interfaces": 0,
                "non_standard_interfaces": 0,
                "interface_issues": []
            }

            # 检查接口文件
            for py_file in layer_path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # 检查接口模式
                    if "interfaces.py" in py_file.name:
                        interface_compliance[layer]["interface_files"] += 1

                        # 检查标准接口定义
                        if self.interface_patterns["interface_naming"].search(content):
                            interface_compliance[layer]["standard_interfaces"] += 1
                        else:
                            interface_compliance[layer]["non_standard_interfaces"] += 1
                            interface_compliance[layer]["interface_issues"].append({
                                "file": str(py_file.relative_to(self.src_dir)),
                                "issue": "接口命名不符合标准规范"
                            })

                    # 检查基础实现文件
                    if "base.py" in py_file.name:
                        interface_compliance[layer]["base_implementation_files"] += 1

                        # 检查基础实现模式
                        if self.interface_patterns["base_implementation"].search(content):
                            pass  # 符合标准
                        else:
                            interface_compliance[layer]["interface_issues"].append({
                                "file": str(py_file.relative_to(self.src_dir)),
                                "issue": "基础实现类不符合标准模式"
                            })

                except Exception as e:
                    interface_compliance[layer]["interface_issues"].append({
                        "file": str(py_file.relative_to(self.src_dir)),
                        "issue": f"文件读取错误: {e}"
                    })

        return interface_compliance

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """分析依赖关系"""
        dependency_analysis = {}

        for layer in self.layer_responsibilities.keys():
            layer_path = self.src_dir / layer
            if not layer_path.exists():
                continue

            dependency_analysis[layer] = {
                "internal_imports": 0,
                "external_imports": 0,
                "cross_layer_imports": 0,
                "circular_dependencies": [],
                "dependency_issues": []
            }

            for py_file in layer_path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('import ') or line.startswith('from '):
                            # 分析导入语句
                            if 'src.' in line:
                                if f'src.{layer}' in line:
                                    dependency_analysis[layer]["internal_imports"] += 1
                                else:
                                    dependency_analysis[layer]["cross_layer_imports"] += 1
                                    dependency_analysis[layer]["dependency_issues"].append({
                                        "file": str(py_file.relative_to(self.src_dir)),
                                        "import": line,
                                        "issue": "跨层级导入"
                                    })
                            else:
                                dependency_analysis[layer]["external_imports"] += 1

                except Exception as e:
                    dependency_analysis[layer]["dependency_issues"].append({
                        "file": str(py_file.relative_to(self.src_dir)),
                        "issue": f"依赖分析错误: {e}"
                    })

        return dependency_analysis

    def _generate_summary(self, audit_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成审计摘要"""
        summary = {
            "total_layers_audited": len(audit_result["layer_audit"]),
            "total_files_analyzed": 0,
            "responsibility_violations": 0,
            "interface_compliance_rate": 0,
            "dependency_issues": 0,
            "quality_score": 0
        }

        # 计算文件总数
        for layer_analysis in audit_result["content_analysis"].values():
            summary["total_files_analyzed"] += layer_analysis["total_files"]

        # 计算职责违规数
        for layer_audit in audit_result["layer_audit"].values():
            summary["responsibility_violations"] += layer_audit["violation_count"]

        # 计算接口符合率
        total_interfaces = 0
        standard_interfaces = 0
        for interface_comp in audit_result["interface_compliance"].values():
            total_interfaces += interface_comp["interface_files"]
            standard_interfaces += interface_comp["standard_interfaces"]

        summary["interface_compliance_rate"] = (
            standard_interfaces / total_interfaces * 100) if total_interfaces > 0 else 100

        # 计算依赖问题数
        for dep_analysis in audit_result["dependency_analysis"].values():
            summary["dependency_issues"] += len(dep_analysis["dependency_issues"])

        # 计算综合质量分数
        base_score = 100
        violation_penalty = summary["responsibility_violations"] * 5
        dependency_penalty = summary["dependency_issues"] * 3
        interface_bonus = (summary["interface_compliance_rate"] - 80) * 0.5

        summary["quality_score"] = max(
            0, base_score - violation_penalty - dependency_penalty + interface_bonus)

        return summary

    def _collect_issues(self, audit_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """收集所有问题"""
        issues = []

        # 收集层级审计问题
        for layer, layer_audit in audit_result["layer_audit"].items():
            for issue in layer_audit["issues"]:
                issues.append({
                    "type": "layer_responsibility",
                    "layer": layer,
                    "severity": issue.get("severity", "medium"),
                    **issue
                })

        # 收集接口问题
        for layer, interface_comp in audit_result["interface_compliance"].items():
            for issue in interface_comp["interface_issues"]:
                issues.append({
                    "type": "interface_compliance",
                    "layer": layer,
                    "severity": "medium",
                    **issue
                })

        # 收集依赖问题
        for layer, dep_analysis in audit_result["dependency_analysis"].items():
            for issue in dep_analysis["dependency_issues"]:
                severity = "high" if "跨层级导入" in issue.get("issue", "") else "medium"
                issues.append({
                    "type": "dependency",
                    "layer": layer,
                    "severity": severity,
                    **issue
                })

        return issues

    def _generate_recommendations(self, audit_result: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于问题生成建议
        issues = audit_result["issues"]

        high_severity = len([i for i in issues if i["severity"] == "high"])
        medium_severity = len([i for i in issues if i["severity"] == "medium"])

        if high_severity > 0:
            recommendations.append(f"🔴 紧急修复: 处理 {high_severity} 个高严重度架构问题")

        if medium_severity > 0:
            recommendations.append(f"🟡 重点关注: 处理 {medium_severity} 个中等严重度问题")

        # 基于审计结果生成具体建议
        layer_audit = audit_result["layer_audit"]
        content_analysis = audit_result["content_analysis"]

        # 检查职责匹配度低的文件
        for layer, analysis in content_analysis.items():
            no_match_files = analysis["no_match_files"]
            if no_match_files > 0:
                recommendations.append(f"📋 检查 {layer} 层级: {no_match_files} 个文件职责匹配度低")

        # 检查跨层级导入
        for layer, dep_analysis in audit_result["dependency_analysis"].items():
            cross_layer_imports = dep_analysis["cross_layer_imports"]
            if cross_layer_imports > 0:
                recommendations.append(f"⚡ 优化 {layer} 层级: 减少 {cross_layer_imports} 个跨层级导入")

        # 基于质量分数提供总体建议
        quality_score = audit_result["summary"]["quality_score"]
        if quality_score >= 90:
            recommendations.append("✅ 架构质量优秀，继续保持")
        elif quality_score >= 70:
            recommendations.append("🟢 架构质量良好，持续改进")
        elif quality_score >= 50:
            recommendations.append("🟡 架构质量一般，需要重点改进")
        else:
            recommendations.append("🔴 架构质量需紧急改进")

        return recommendations

    def generate_audit_report(self, audit_result: Dict[str, Any]) -> str:
        """生成审计报告"""
        report = f"""# 深度架构审计报告

## 📊 审计概览

**审计时间**: {audit_result['timestamp']}
**审计范围**: src目录深度分析
**发现问题**: {len(audit_result['issues'])} 个
**质量评分**: {audit_result['summary']['quality_score']:.1f}/100

### 审计维度
- **层级职责审计**: 检查各层级文件内容是否符合架构职责
- **内容匹配分析**: 分析文件职责匹配度
- **接口规范检查**: 验证接口设计标准性
- **依赖关系分析**: 检查导入依赖的合理性

---

## 🏗️ 层级职责审计结果

"""

        # 层级审计结果
        for layer, layer_audit in audit_result['layer_audit'].items():
            report += f"### {layer.upper()} 层级\n"
            report += f"**文件数量**: {layer_audit['file_count']} 个\n"
            report += f"**职责匹配度**: {layer_audit['responsibility_match']} 个关键词匹配\n"
            report += f"**职责违规数**: {layer_audit['violation_count']} 个违规项\n\n"

            if layer_audit['issues']:
                report += "**发现问题**:\n"
                for issue in layer_audit['issues']:
                    severity_emoji = "🔴" if issue.get('severity') == 'high' else "🟡"
                    report += f"- {severity_emoji} {issue.get('file', 'N/A')}: "
                    if 'violations' in issue:
                        report += f"包含违规内容: {', '.join(issue['violations'])}\n"
                    else:
                        report += f"{issue.get('error', '未知错误')}\n"
                report += "\n"

        report += "## 📋 内容职责匹配分析\n\n"

        # 内容分析结果
        for layer, analysis in audit_result['content_analysis'].items():
            report += f"### {layer.upper()} 层级匹配统计\n"
            report += f"- **总文件数**: {analysis['total_files']} 个\n"
            report += f"- **高匹配文件**: {analysis['high_match_files']} 个 (≥70%)\n"
            report += f"- **中等匹配文件**: {analysis['medium_match_files']} 个 (40-70%)\n"
            report += f"- **低匹配文件**: {analysis['low_match_files']} 个 (>0-40%)\n"
            report += f"- **无匹配文件**: {analysis['no_match_files']} 个 (0%)\n\n"

        report += "## 🔗 接口规范检查\n\n"

        # 接口检查结果
        for layer, interface_comp in audit_result['interface_compliance'].items():
            if interface_comp['interface_files'] > 0 or interface_comp['base_implementation_files'] > 0:
                report += f"### {layer.upper()} 层级接口\n"
                report += f"- **接口文件**: {interface_comp['interface_files']} 个\n"
                report += f"- **基础实现文件**: {interface_comp['base_implementation_files']} 个\n"
                report += f"- **标准接口**: {interface_comp['standard_interfaces']} 个\n"
                report += f"- **非标准接口**: {interface_comp['non_standard_interfaces']} 个\n\n"

                if interface_comp['interface_issues']:
                    report += "**接口问题**:\n"
                    for issue in interface_comp['interface_issues']:
                        report += f"- ⚠️ {issue.get('file', 'N/A')}: {issue.get('issue', 'N/A')}\n"
                    report += "\n"

        report += "## ⚡ 依赖关系分析\n\n"

        # 依赖分析结果
        for layer, dep_analysis in audit_result['dependency_analysis'].items():
            report += f"### {layer.upper()} 层级依赖\n"
            report += f"- **内部导入**: {dep_analysis['internal_imports']} 个\n"
            report += f"- **外部导入**: {dep_analysis['external_imports']} 个\n"
            report += f"- **跨层级导入**: {dep_analysis['cross_layer_imports']} 个\n\n"

            if dep_analysis['dependency_issues']:
                report += "**依赖问题**:\n"
                for issue in dep_analysis['dependency_issues']:
                    report += f"- ⚠️ {issue.get('file', 'N/A')}: {issue.get('issue', 'N/A')}\n"
                report += "\n"

        # 详细问题列表
        if audit_result['issues']:
            report += "## 🔍 详细问题列表\n\n"

            for issue in audit_result['issues']:
                severity_emoji = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(issue['severity'], "⚪")

                report += f"### {severity_emoji} {issue['type'].replace('_', ' ').title()}\n"
                report += f"**层级**: {issue.get('layer', 'N/A')}\n"
                report += f"**文件**: `{issue.get('file', 'N/A')}`\n"
                report += f"**严重程度**: {issue['severity']}\n"

                if 'violations' in issue:
                    report += f"**违规内容**: {', '.join(issue['violations'])}\n"
                if 'import' in issue:
                    report += f"**导入语句**: `{issue['import']}`\n"
                if 'issue' in issue:
                    report += f"**问题描述**: {issue['issue']}\n"

                report += "\n"

        # 建议
        if audit_result['recommendations']:
            report += "## 💡 改进建议\n\n"
            for rec in audit_result['recommendations']:
                report += f"- {rec}\n"
            report += "\n"

        report += f"""## 📈 审计质量评分

### 综合指标
- **总审计层级**: {audit_result['summary']['total_layers_audited']} 个
- **分析文件数**: {audit_result['summary']['total_files_analyzed']} 个
- **职责违规数**: {audit_result['summary']['responsibility_violations']} 个
- **接口符合率**: {audit_result['summary']['interface_compliance_rate']:.1f}%
- **依赖问题数**: {audit_result['summary']['dependency_issues']} 个
- **综合质量分**: {audit_result['summary']['quality_score']:.1f}/100

### 评分标准
- **90-100**: 优秀架构质量
- **70-89**: 良好架构质量
- **50-69**: 一般架构质量
- **0-49**: 需要改进架构质量

---

**审计工具**: scripts/deep_architecture_audit.py
**审计标准**: 基于架构设计文档 v5.0
**建议处理**: 按严重程度从高到低修复问题
"""

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='深度架构审计工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--output', help='输出报告文件')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='报告格式')

    args = parser.parse_args()

    auditor = DeepArchitectureAuditor(args.project)
    audit_result = auditor.perform_deep_audit()

    if args.format == 'json':
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(audit_result, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(audit_result, ensure_ascii=False, indent=2))
    else:
        report = auditor.generate_audit_report(audit_result)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
        else:
            print(report)


if __name__ == "__main__":
    main()

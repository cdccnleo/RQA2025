#!/usr/bin/env python3
"""
基础设施层专项复核脚本

对src/infrastructure目录进行全面复核，检查：
1. 接口命名规范符合性
2. 目录结构合理性
3. 文档完整性
4. 跨层级导入合理性
5. 职责边界符合性
6. 代码规范统一性
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import re


class InfrastructureReviewer:
    """基础设施层复核器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.infrastructure_dir = self.project_root / "src" / "infrastructure"

        # 基础设施层应该包含的8个功能分类
        self.expected_categories = {
            "config": "配置管理",
            "cache": "缓存系统",
            "logging": "日志系统",
            "security": "安全管理",
            "error": "错误处理",
            "resource": "资源管理",
            "health": "健康检查",
            "utils": "工具组件"
        }

        # 接口命名规范检查
        self.interface_patterns = {
            "interface_naming": re.compile(r"^class\s+I[A-Z]\w+Component\("),
            "base_implementation": re.compile(r"^class\s+Base[A-Z]\w+Component\("),
            "factory_naming": re.compile(r"^class\s+I[A-Z]\w+FactoryComponent\(")
        }

        # 跨层级导入检查
        self.allowed_cross_layer_imports = {
            "src.utils.logger": "基础设施层工具",
            "src.infrastructure": "当前层级",
            "src.core": "核心服务层",
        }

    def perform_comprehensive_review(self) -> Dict[str, Any]:
        """执行全面复核"""
        print("🔍 开始基础设施层专项复核...")

        review_result = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "structure_analysis": {},
            "interface_compliance": {},
            "documentation_quality": {},
            "cross_layer_imports": {},
            "responsibility_boundaries": {},
            "issues": [],
            "recommendations": []
        }

        # 1. 目录结构分析
        print("📊 步骤1: 目录结构分析")
        review_result["structure_analysis"] = self._analyze_directory_structure()

        # 2. 接口规范检查
        print("🔗 步骤2: 接口规范检查")
        review_result["interface_compliance"] = self._check_interface_compliance()

        # 3. 文档质量评估
        print("📋 步骤3: 文档质量评估")
        review_result["documentation_quality"] = self._assess_documentation_quality()

        # 4. 跨层级导入检查
        print("⚡ 步骤4: 跨层级导入检查")
        review_result["cross_layer_imports"] = self._check_cross_layer_imports()

        # 5. 职责边界验证
        print("🎯 步骤5: 职责边界验证")
        review_result["responsibility_boundaries"] = self._validate_responsibility_boundaries()

        # 6. 生成综合报告
        review_result["summary"] = self._generate_summary(review_result)
        review_result["issues"] = self._collect_issues(review_result)
        review_result["recommendations"] = self._generate_recommendations(review_result)

        print(f"✅ 基础设施层复核完成，发现 {len(review_result['issues'])} 个问题")
        return review_result

    def _analyze_directory_structure(self) -> Dict[str, Any]:
        """分析目录结构"""
        structure_analysis = {
            "total_files": 0,
            "total_directories": 0,
            "category_composition": {},
            "file_distribution": {},
            "structure_issues": []
        }

        # 统计文件和目录
        for item in self.infrastructure_dir.rglob("*"):
            if item.is_file() and item.name.endswith('.py') and not item.name.startswith('_'):
                structure_analysis["total_files"] += 1

                # 按分类统计
                relative_path = item.relative_to(self.infrastructure_dir)
                top_category = str(relative_path).split(
                    '/')[0] if '/' in str(relative_path) else 'root'

                if top_category not in structure_analysis["category_composition"]:
                    structure_analysis["category_composition"][top_category] = 0
                structure_analysis["category_composition"][top_category] += 1

            elif item.is_dir() and not item.name.startswith('_'):
                structure_analysis["total_directories"] += 1

        # 检查预期的8个功能分类是否存在
        for expected_category, description in self.expected_categories.items():
            if expected_category not in structure_analysis["category_composition"]:
                structure_analysis["structure_issues"].append({
                    "type": "missing_category",
                    "category": expected_category,
                    "description": f"缺少预期的功能分类: {description}"
                })

        # 检查是否有意外的分类
        unexpected_categories = []
        for category in structure_analysis["category_composition"]:
            if category not in self.expected_categories and category != 'root':
                unexpected_categories.append(category)

        if unexpected_categories:
            structure_analysis["structure_issues"].append({
                "type": "unexpected_categories",
                "categories": unexpected_categories,
                "description": f"发现意外的功能分类: {', '.join(unexpected_categories)}"
            })

        return structure_analysis

    def _check_interface_compliance(self) -> Dict[str, Any]:
        """检查接口规范符合性"""
        interface_compliance = {
            "total_interfaces": 0,
            "standard_interfaces": 0,
            "non_standard_interfaces": 0,
            "base_implementations": 0,
            "factory_interfaces": 0,
            "interface_issues": []
        }

        # 检查所有Python文件中的接口定义
        for py_file in self.infrastructure_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')
                for i, line in enumerate(lines):
                    line = line.strip()

                    # 检查接口命名
                    if self.interface_patterns["interface_naming"].search(line):
                        interface_compliance["total_interfaces"] += 1
                        interface_compliance["standard_interfaces"] += 1

                    elif line.startswith("class I") and line.endswith("(ABC):"):
                        interface_compliance["total_interfaces"] += 1
                        interface_compliance["non_standard_interfaces"] += 1
                        interface_compliance["interface_issues"].append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "line": i + 1,
                            "interface": line,
                            "issue": "接口命名不符合标准格式 I{Name}Component"
                        })

                    # 检查基础实现
                    elif self.interface_patterns["base_implementation"].search(line):
                        interface_compliance["base_implementations"] += 1

                    elif line.startswith("class Base") and line.endswith("(ABC):"):
                        interface_compliance["interface_issues"].append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "line": i + 1,
                            "interface": line,
                            "issue": "基础实现类命名不符合标准格式 Base{Name}Component"
                        })

                    # 检查工厂接口
                    elif self.interface_patterns["factory_naming"].search(line):
                        interface_compliance["factory_interfaces"] += 1

            except Exception as e:
                interface_compliance["interface_issues"].append({
                    "file": str(py_file.relative_to(self.project_root)),
                    "issue": f"文件读取错误: {e}"
                })

        return interface_compliance

    def _assess_documentation_quality(self) -> Dict[str, Any]:
        """评估文档质量"""
        documentation_quality = {
            "total_files": 0,
            "documented_interfaces": 0,
            "undocumented_interfaces": 0,
            "documentation_issues": []
        }

        # 检查接口文档
        for py_file in self.infrastructure_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                documentation_quality["total_files"] += 1

                # 检查是否有模块级文档
                if not content.strip().startswith('"""') and not content.strip().startswith("'''"):
                    documentation_quality["documentation_issues"].append({
                        "file": str(py_file.relative_to(self.project_root)),
                        "issue": "缺少模块级文档字符串"
                    })

                # 检查接口文档
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'class I' in line and '(ABC):' in line:
                        # 检查是否有类文档
                        if i + 1 < len(lines) and not lines[i + 1].strip().startswith('"""'):
                            documentation_quality["undocumented_interfaces"] += 1
                            documentation_quality["documentation_issues"].append({
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": i + 1,
                                "interface": line.strip(),
                                "issue": "接口类缺少文档字符串"
                            })
                        else:
                            documentation_quality["documented_interfaces"] += 1

            except Exception as e:
                documentation_quality["documentation_issues"].append({
                    "file": str(py_file.relative_to(self.project_root)),
                    "issue": f"文档检查错误: {e}"
                })

        return documentation_quality

    def _check_cross_layer_imports(self) -> Dict[str, Any]:
        """检查跨层级导入"""
        cross_layer_imports = {
            "total_imports": 0,
            "internal_imports": 0,
            "external_imports": 0,
            "cross_layer_imports": 0,
            "import_issues": []
        }

        # 检查所有Python文件的导入
        for py_file in self.infrastructure_dir.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        cross_layer_imports["total_imports"] += 1

                        # 检查跨层级导入
                        if 'src.' in line:
                            if 'src.infrastructure' in line:
                                cross_layer_imports["internal_imports"] += 1
                            else:
                                cross_layer_imports["cross_layer_imports"] += 1

                                # 检查是否是合理的跨层级导入
                                is_reasonable = False
                                for allowed_import in self.allowed_cross_layer_imports:
                                    if allowed_import in line:
                                        is_reasonable = True
                                        break

                                if not is_reasonable:
                                    cross_layer_imports["import_issues"].append({
                                        "file": str(py_file.relative_to(self.project_root)),
                                        "import": line,
                                        "issue": "不合理的跨层级导入"
                                    })
                        else:
                            cross_layer_imports["external_imports"] += 1

            except Exception as e:
                cross_layer_imports["import_issues"].append({
                    "file": str(py_file.relative_to(self.project_root)),
                    "issue": f"导入检查错误: {e}"
                })

        return cross_layer_imports

    def _validate_responsibility_boundaries(self) -> Dict[str, Any]:
        """验证职责边界"""
        responsibility_boundaries = {
            "boundary_compliance": {},
            "responsibility_issues": []
        }

        # 定义各功能分类的职责关键词
        category_keywords = {
            "config": ["config", "configuration", "settings", "properties"],
            "cache": ["cache", "memory", "redis", "storage"],
            "logging": ["log", "logger", "logging", "trace"],
            "security": ["security", "auth", "encrypt", "permission"],
            "error": ["error", "exception", "fail", "retry"],
            "resource": ["resource", "gpu", "cpu", "memory", "quota"],
            "health": ["health", "check", "monitor", "status"],
            "utils": ["util", "helper", "tool", "common"]
        }

        # 验证各分类的职责边界
        for category, keywords in category_keywords.items():
            category_path = self.infrastructure_dir / category
            if category_path.exists():
                responsibility_boundaries["boundary_compliance"][category] = {
                    "exists": True,
                    "keyword_matches": 0,
                    "compliance_score": 0
                }

                # 检查分类下的文件是否符合职责
                for py_file in category_path.rglob("*.py"):
                    if py_file.name.startswith("_"):
                        continue

                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read().lower()

                        # 计算关键词匹配度
                        keyword_matches = sum(1 for keyword in keywords if keyword in content)
                        responsibility_boundaries["boundary_compliance"][category]["keyword_matches"] += keyword_matches

                    except Exception as e:
                        responsibility_boundaries["responsibility_issues"].append({
                            "file": str(py_file.relative_to(self.project_root)),
                            "issue": f"职责检查错误: {e}"
                        })

                # 计算符合度
                total_files = len(list(category_path.rglob("*.py")))
                if total_files > 0:
                    avg_matches = responsibility_boundaries["boundary_compliance"][category]["keyword_matches"] / total_files
                    responsibility_boundaries["boundary_compliance"][category]["compliance_score"] = min(
                        100, avg_matches * 25)

        return responsibility_boundaries

    def _generate_summary(self, review_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成复核总结"""
        summary = {
            "review_time": review_result["timestamp"],
            "total_issues": len(review_result["issues"]),
            "structure_score": 0,
            "interface_score": 0,
            "documentation_score": 0,
            "import_score": 0,
            "boundary_score": 0,
            "overall_score": 0
        }

        # 计算各项评分
        struct_analysis = review_result["structure_analysis"]
        interface_comp = review_result["interface_compliance"]
        doc_quality = review_result["documentation_quality"]
        cross_imports = review_result["cross_layer_imports"]
        boundaries = review_result["responsibility_boundaries"]

        # 结构评分
        expected_categories = len(self.expected_categories)
        found_categories = len(
            [c for c in struct_analysis["category_composition"] if c in self.expected_categories])
        summary["structure_score"] = (found_categories / expected_categories) * 100

        # 接口评分
        if interface_comp["total_interfaces"] > 0:
            standard_ratio = interface_comp["standard_interfaces"] / \
                interface_comp["total_interfaces"]
            summary["interface_score"] = standard_ratio * 100

        # 文档评分
        total_interfaces = interface_comp["total_interfaces"]
        if total_interfaces > 0:
            documented_ratio = doc_quality["documented_interfaces"] / total_interfaces
            summary["documentation_score"] = documented_ratio * 100

        # 导入评分
        total_cross_imports = cross_imports["cross_layer_imports"]
        reasonable_imports = total_cross_imports - len(cross_imports["import_issues"])
        if total_cross_imports > 0:
            reasonable_ratio = reasonable_imports / total_cross_imports
            summary["import_score"] = reasonable_ratio * 100
        else:
            summary["import_score"] = 100

        # 职责边界评分
        boundary_scores = [b["compliance_score"]
                           for b in boundaries["boundary_compliance"].values() if b["exists"]]
        if boundary_scores:
            summary["boundary_score"] = sum(boundary_scores) / len(boundary_scores)

        # 综合评分
        weights = {
            "structure_score": 0.2,
            "interface_score": 0.25,
            "documentation_score": 0.2,
            "import_score": 0.15,
            "boundary_score": 0.2
        }

        overall_score = sum(score * weights[metric] for metric, score in summary.items()
                            if metric.endswith("_score") and metric != "overall_score")
        summary["overall_score"] = overall_score

        return summary

    def _collect_issues(self, review_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """收集所有问题"""
        issues = []

        # 收集结构问题
        for issue in review_result["structure_analysis"]["structure_issues"]:
            issues.append({"type": "structure", "severity": "medium", **issue})

        # 收集接口问题
        for issue in review_result["interface_compliance"]["interface_issues"]:
            severity = "high" if "不符合标准格式" in issue["issue"] else "medium"
            issues.append({"type": "interface", "severity": severity, **issue})

        # 收集文档问题
        for issue in review_result["documentation_quality"]["documentation_issues"]:
            issues.append({"type": "documentation", "severity": "low", **issue})

        # 收集导入问题
        for issue in review_result["cross_layer_imports"]["import_issues"]:
            issues.append({"type": "import", "severity": "medium", **issue})

        # 收集职责边界问题
        for issue in review_result["responsibility_boundaries"]["responsibility_issues"]:
            issues.append({"type": "responsibility", "severity": "low", **issue})

        return issues

    def _generate_recommendations(self, review_result: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        summary = review_result["summary"]

        if summary["structure_score"] < 100:
            recommendations.append("🏗️ 完善基础设施层目录结构，确保8个功能分类都存在")

        if summary["interface_score"] < 100:
            recommendations.append("🔗 修复接口命名规范，确保所有接口符合I{Name}Component格式")

        if summary["documentation_score"] < 80:
            recommendations.append("📋 完善接口文档，为所有接口添加详细说明")

        if summary["import_score"] < 90:
            recommendations.append("⚡ 优化跨层级导入，减少不合理的依赖关系")

        if summary["boundary_score"] < 70:
            recommendations.append("🎯 优化职责边界，确保各功能分类职责明确")

        if summary["overall_score"] >= 90:
            recommendations.append("✅ 基础设施层整体质量优秀，继续保持")
        elif summary["overall_score"] >= 75:
            recommendations.append("🟢 基础设施层质量良好，需要小幅改进")
        elif summary["overall_score"] >= 60:
            recommendations.append("🟡 基础设施层质量一般，需要重点改进")
        else:
            recommendations.append("🔴 基础设施层质量需要全面改进")

        return recommendations

    def generate_review_report(self, review_result: Dict[str, Any]) -> str:
        """生成复核报告"""
        report = f"""# 基础设施层专项复核报告

## 📊 复核概览

**复核时间**: {review_result['summary']['review_time']}
**基础设施层综合评分**: {review_result['summary']['overall_score']:.1f}/100
**发现问题**: {len(review_result['issues'])} 个

### 分项评分
| 评分项目 | 分数 | 权重 |
|---------|------|------|
| 目录结构 | {review_result['summary']['structure_score']:.1f} | 20% |
| 接口规范 | {review_result['summary']['interface_score']:.1f} | 25% |
| 文档质量 | {review_result['summary']['documentation_score']:.1f} | 20% |
| 导入合理性 | {review_result['summary']['import_score']:.1f} | 15% |
| 职责边界 | {review_result['summary']['boundary_score']:.1f} | 20% |

---

## 🏗️ 目录结构分析

### 总体统计
- **总文件数**: {review_result['structure_analysis']['total_files']} 个
- **总目录数**: {review_result['structure_analysis']['total_directories']} 个
- **功能分类**: {len(review_result['structure_analysis']['category_composition'])} 个

### 功能分类分布
"""

        # 功能分类分布
        for category, count in review_result['structure_analysis']['category_composition'].items():
            description = self.expected_categories.get(category, "未知分类")
            report += f"- **{category}** ({description}): {count} 个文件\n"

        # 结构问题
        if review_result['structure_analysis']['structure_issues']:
            report += "\n### 结构问题\n"
            for issue in review_result['structure_analysis']['structure_issues']:
                report += f"- ⚠️ {issue['description']}\n"

        report += f"""

## 🔗 接口规范检查

### 接口统计
- **总接口数**: {review_result['interface_compliance']['total_interfaces']} 个
- **标准接口**: {review_result['interface_compliance']['standard_interfaces']} 个
- **非标准接口**: {review_result['interface_compliance']['non_standard_interfaces']} 个
- **基础实现**: {review_result['interface_compliance']['base_implementations']} 个
- **工厂接口**: {review_result['interface_compliance']['factory_interfaces']} 个

### 接口符合率
**标准符合率**: {review_result['summary']['interface_score']:.1f}%

"""

        # 接口问题
        if review_result['interface_compliance']['interface_issues']:
            report += "\n### 接口问题\n"
            for issue in review_result['interface_compliance']['interface_issues']:
                severity_emoji = "🔴" if issue.get('severity') == 'high' else "🟡"
                report += f"- {severity_emoji} {issue['file']}:{issue.get('line', 'N/A')} - {issue['issue']}\n"

        report += f"""

## 📋 文档质量评估

### 文档统计
- **总文件数**: {review_result['documentation_quality']['total_files']} 个
- **已文档化接口**: {review_result['documentation_quality']['documented_interfaces']} 个
- **未文档化接口**: {review_result['documentation_quality']['undocumented_interfaces']} 个

### 文档覆盖率
**文档覆盖率**: {review_result['summary']['documentation_score']:.1f}%

"""

        # 文档问题
        if review_result['documentation_quality']['documentation_issues']:
            report += "\n### 文档问题\n"
            for issue in review_result['documentation_quality']['documentation_issues']:
                report += f"- 📝 {issue['file']} - {issue['issue']}\n"

        report += f"""

## ⚡ 跨层级导入检查

### 导入统计
- **总导入数**: {review_result['cross_layer_imports']['total_imports']} 个
- **内部导入**: {review_result['cross_layer_imports']['internal_imports']} 个
- **外部导入**: {review_result['cross_layer_imports']['external_imports']} 个
- **跨层级导入**: {review_result['cross_layer_imports']['cross_layer_imports']} 个

### 导入合理性
**合理导入率**: {review_result['summary']['import_score']:.1f}%

"""

        # 导入问题
        if review_result['cross_layer_imports']['import_issues']:
            report += "\n### 导入问题\n"
            for issue in review_result['cross_layer_imports']['import_issues']:
                report += f"- ⚠️ {issue['file']} - {issue['issue']}: {issue['import']}\n"

        report += f"""

## 🎯 职责边界验证

### 分类职责符合度
"""

        # 职责边界
        for category, compliance in review_result['responsibility_boundaries']['boundary_compliance'].items():
            if compliance['exists']:
                description = self.expected_categories.get(category, "未知分类")
                score = compliance['compliance_score']
                report += f"- **{category}** ({description}): {score:.1f}% 符合度\n"

        # 职责问题
        if review_result['responsibility_boundaries']['responsibility_issues']:
            report += "\n### 职责问题\n"
            for issue in review_result['responsibility_boundaries']['responsibility_issues']:
                report += f"- 🎯 {issue['file']} - {issue['issue']}\n"

        # 详细问题列表
        if review_result['issues']:
            report += f"""

## 🔍 详细问题列表

### 按严重程度排序

"""

            # 按严重程度分组
            high_severity = [i for i in review_result['issues'] if i['severity'] == 'high']
            medium_severity = [i for i in review_result['issues'] if i['severity'] == 'medium']
            low_severity = [i for i in review_result['issues'] if i['severity'] == 'low']

            if high_severity:
                report += "### 🔴 高严重度问题\n"
                for issue in high_severity:
                    report += f"- **{issue['type'].title()}**: {issue.get('description', issue.get('issue', 'N/A'))}\n"
                    if 'file' in issue:
                        report += f"  文件: `{issue['file']}`\n"
                    if 'interface' in issue:
                        report += f"  接口: `{issue['interface']}`\n"
                    report += "\n"

            if medium_severity:
                report += "### 🟡 中等严重度问题\n"
                for issue in medium_severity:
                    report += f"- **{issue['type'].title()}**: {issue.get('description', issue.get('issue', 'N/A'))}\n"
                    if 'file' in issue:
                        report += f"  文件: `{issue['file']}`\n"
                    if 'interface' in issue:
                        report += f"  接口: `{issue['interface']}`\n"
                    report += "\n"

            if low_severity:
                report += "### 🟢 低严重度问题\n"
                for issue in low_severity:
                    report += f"- **{issue['type'].title()}**: {issue.get('description', issue.get('issue', 'N/A'))}\n"
                    if 'file' in issue:
                        report += f"  文件: `{issue['file']}`\n"
                    if 'interface' in issue:
                        report += f"  接口: `{issue['interface']}`\n"
                    report += "\n"

        # 改进建议
        if review_result['recommendations']:
            report += f"""

## 💡 改进建议

"""
            for rec in review_result['recommendations']:
                report += f"- {rec}\n"

        report += f"""

---

**复核工具**: scripts/infrastructure_review.py
**复核标准**: 基于架构设计文档 v5.0
**建议处理**: 按严重程度从高到低修复问题
"""

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='基础设施层专项复核工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--output', help='输出报告文件')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='报告格式')

    args = parser.parse_args()

    reviewer = InfrastructureReviewer(args.project)
    review_result = reviewer.perform_comprehensive_review()

    if args.format == 'json':
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(review_result, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(review_result, ensure_ascii=False, indent=2))
    else:
        report = reviewer.generate_review_report(review_result)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
        else:
            print(report)


if __name__ == "__main__":
    main()

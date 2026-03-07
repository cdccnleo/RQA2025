#!/usr/bin/env python3
"""
架构一致性检查工具

检查src目录结构是否与架构设计文档一致
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class ArchitectureConsistencyChecker:
    """架构一致性检查器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"

        # 预期的架构层次映射
        self.expected_architecture = {
            "core": {
                "description": "核心服务层 - 事件总线、依赖注入、流程编排",
                "components": ["event_bus", "container", "business_process_orchestrator"],
                "status": "✅ 存在"
            },
            "infrastructure": {
                "description": "基础设施层 - 配置管理、缓存系统、日志系统等",
                "subdirs": ["cache", "config", "logging", "security", "error", "resource", "health", "utils"],
                "status": "✅ 存在（已拆分）"
            },
            "data": {
                "description": "数据采集层 - 数据源适配、实时采集、数据验证",
                "subdirs": ["adapters", "collector", "validator", "quality_monitor"],
                "status": "✅ 存在"
            },
            "gateway": {
                "description": "API网关层 - 路由转发、认证授权、限流熔断",
                "components": ["api_gateway"],
                "status": "✅ 存在"
            },
            "features": {
                "description": "特征处理层 - 智能特征工程、分布式处理、硬件加速",
                "subdirs": ["engineering", "distributed", "acceleration"],
                "status": "✅ 存在"
            },
            "ml": {
                "description": "模型推理层 - 集成学习、模型管理、实时推理",
                "subdirs": ["integration", "models", "engine"],
                "status": "✅ 存在"
            },
            "backtest": {
                "description": "策略决策层 - 策略生成器、策略框架",
                "components": ["engine", "analyzer", "strategy_framework"],
                "status": "✅ 存在"
            },
            "risk": {
                "description": "风控合规层 - 风控API、中国市场规则、风险控制器",
                "components": ["checker", "monitor", "api"],
                "status": "✅ 存在"
            },
            "trading": {
                "description": "交易执行层 - 订单管理、执行引擎、智能路由",
                "components": ["executor", "manager", "risk"],
                "status": "✅ 存在"
            },
            "engine": {
                "description": "监控反馈层 - 系统监控、业务监控、性能监控",
                "subdirs": ["monitoring", "logging", "optimization"],
                "status": "✅ 存在"
            }
        }

        # 已知的冗余/重复目录映射
        self.redundant_directories = {
            "acceleration": {
                "should_be": "features/acceleration",
                "reason": "硬件加速组件应该在特征处理层下"
            },
            "adapters": {
                "should_be": "data/adapters",
                "reason": "数据适配器应该在数据采集层下"
            },
            "analysis": {
                "should_be": "backtest/analysis 或 engine/analysis",
                "reason": "分析功能需要确定具体归属层级"
            },
            "deployment": {
                "should_be": "infrastructure/deployment",
                "reason": "部署相关功能应该在基础设施层"
            },
            "integration": {
                "should_be": "core/integration",
                "reason": "系统集成功能应该在核心服务层"
            },
            "models": {
                "should_be": "ml/models",
                "reason": "模型管理应该在模型推理层下"
            },
            "monitoring": {
                "should_be": "engine/monitoring",
                "reason": "系统监控应该在监控反馈层"
            },
            "services": {
                "should_be": "infrastructure/services",
                "reason": "通用服务应该在基础设施层"
            },
            "tuning": {
                "should_be": "ml/tuning 或 backtest/tuning",
                "reason": "调优功能需要确定具体归属层级"
            },
            "utils": {
                "should_be": "infrastructure/utils",
                "reason": "通用工具应该在基础设施层"
            }
        }

    def check_directory_structure(self) -> Dict[str, Any]:
        """检查目录结构一致性"""
        print("🔍 检查src目录结构一致性...")

        result = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "issues": [],
            "recommendations": [],
            "details": {}
        }

        # 检查主要架构层级
        for layer, config in self.expected_architecture.items():
            layer_path = self.src_dir / layer
            if layer_path.exists():
                result["details"][layer] = self._check_layer_consistency(layer, config)
            else:
                result["issues"].append({
                    "type": "missing_layer",
                    "severity": "high",
                    "layer": layer,
                    "description": f"缺少架构层级目录: {layer}",
                    "expected": config["description"]
                })

        # 检查冗余目录
        for dir_name, info in self.redundant_directories.items():
            dir_path = self.src_dir / dir_name
            if dir_path.exists():
                result["issues"].append({
                    "type": "redundant_directory",
                    "severity": "medium",
                    "directory": dir_name,
                    "description": f"发现冗余目录: {dir_name}",
                    "expected_location": info["should_be"],
                    "reason": info["reason"]
                })

        # 检查未分类的目录
        all_expected_dirs = set(self.expected_architecture.keys()) | set(
            self.redundant_directories.keys())
        actual_dirs = set()

        for item in self.src_dir.iterdir():
            if item.is_dir() and not item.name.startswith('_') and not item.name.startswith('.'):
                actual_dirs.add(item.name)

        unexpected_dirs = actual_dirs - all_expected_dirs
        for dir_name in unexpected_dirs:
            result["issues"].append({
                "type": "unexpected_directory",
                "severity": "low",
                "directory": dir_name,
                "description": f"发现未分类目录: {dir_name}",
                "recommendation": "需要确定该目录所属的架构层级"
            })

        # 生成摘要
        result["summary"] = {
            "total_layers": len(self.expected_architecture),
            "existing_layers": len([d for d in result["details"].values() if d.get("exists", False)]),
            "missing_layers": len([i for i in result["issues"] if i["type"] == "missing_layer"]),
            "redundant_directories": len([i for i in result["issues"] if i["type"] == "redundant_directory"]),
            "unexpected_directories": len([i for i in result["issues"] if i["type"] == "unexpected_directory"]),
            "total_issues": len(result["issues"])
        }

        # 生成建议
        result["recommendations"] = self._generate_recommendations(result["issues"])

        return result

    def _check_layer_consistency(self, layer: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """检查单个层级的一致性"""
        layer_path = self.src_dir / layer
        layer_info = {
            "layer": layer,
            "description": config["description"],
            "exists": layer_path.exists(),
            "subdirs": [],
            "files": [],
            "issues": []
        }

        if not layer_path.exists():
            return layer_info

        # 检查子目录
        if "subdirs" in config:
            for subdir in config["subdirs"]:
                subdir_path = layer_path / subdir
                if subdir_path.exists():
                    layer_info["subdirs"].append({
                        "name": subdir,
                        "exists": True,
                        "status": "✅ 存在"
                    })
                else:
                    layer_info["subdirs"].append({
                        "name": subdir,
                        "exists": False,
                        "status": "❌ 缺失"
                    })
                    layer_info["issues"].append(f"缺少子目录: {subdir}")

        # 检查主要组件
        if "components" in config:
            for component in config["components"]:
                component_path = layer_path / f"{component}.py"
                if component_path.exists():
                    layer_info["files"].append({
                        "name": f"{component}.py",
                        "exists": True,
                        "status": "✅ 存在"
                    })
                else:
                    layer_info["files"].append({
                        "name": f"{component}.py",
                        "exists": False,
                        "status": "❌ 缺失"
                    })
                    layer_info["issues"].append(f"缺少组件文件: {component}.py")

        return layer_info

    def _generate_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """生成修复建议"""
        recommendations = []

        # 按问题类型分组
        issue_types = {}
        for issue in issues:
            if issue["type"] not in issue_types:
                issue_types[issue["type"]] = []
            issue_types[issue["type"]].append(issue)

        # 生成针对性建议
        if "missing_layer" in issue_types:
            recommendations.append(f"🔴 紧急修复: 创建缺少的架构层级目录 ({len(issue_types['missing_layer'])}个)")

        if "redundant_directory" in issue_types:
            recommendations.append(
                f"🟡 中等优先级: 迁移冗余目录到正确位置 ({len(issue_types['redundant_directory'])}个)")

        if "unexpected_directory" in issue_types:
            recommendations.append(
                f"🟢 低优先级: 对未分类目录进行架构定位 ({len(issue_types['unexpected_directory'])}个)")

        # 具体修复建议
        if issue_types.get("redundant_directory"):
            for issue in issue_types["redundant_directory"]:
                recommendations.append(
                    f"  - 迁移 `{issue['directory']}` 到 `{issue['expected_location']}` ({issue['reason']})")

        return recommendations

    def generate_consistency_report(self, result: Dict[str, Any]) -> str:
        """生成一致性检查报告"""
        report = f"""# 架构一致性检查报告

## 📊 检查概览

**检查时间**: {result['timestamp']}
**检查范围**: src目录结构
**发现问题**: {result['summary']['total_issues']} 个

### 问题分布
| 问题类型 | 数量 | 严重程度 |
|---------|------|---------|
| 缺失层级 | {result['summary']['missing_layers']} | 🔴 高 |
| 冗余目录 | {result['summary']['redundant_directories']} | 🟡 中 |
| 未分类目录 | {result['summary']['unexpected_directories']} | 🟢 低 |

---

## 🏗️ 架构层级检查结果

"""

        # 各层级检查结果
        for layer, details in result['details'].items():
            report += f"### {layer.upper()} 层级\n"
            report += f"**描述**: {details['description']}\n"
            report += f"**状态**: {'✅ 存在' if details['exists'] else '❌ 缺失'}\n\n"

            if details['subdirs']:
                report += "**子目录检查**:\n"
                for subdir in details['subdirs']:
                    report += f"- {subdir['name']}: {subdir['status']}\n"
                report += "\n"

            if details['files']:
                report += "**组件文件检查**:\n"
                for file_info in details['files']:
                    report += f"- {file_info['name']}: {file_info['status']}\n"
                report += "\n"

            if details['issues']:
                report += "**发现问题**:\n"
                for issue in details['issues']:
                    report += f"- ⚠️ {issue}\n"
                report += "\n"

        # 问题详情
        if result['issues']:
            report += "## 🔍 详细问题列表\n\n"

            for issue in result['issues']:
                severity_emoji = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(issue['severity'], "⚪")

                report += f"### {severity_emoji} {issue['type'].replace('_', ' ').title()}\n"
                report += f"**目录**: `{issue.get('directory', issue.get('layer', 'N/A'))}`\n"
                report += f"**描述**: {issue['description']}\n"

                if 'expected_location' in issue:
                    report += f"**建议位置**: `{issue['expected_location']}`\n"
                if 'reason' in issue:
                    report += f"**原因**: {issue['reason']}\n"
                if 'expected' in issue:
                    report += f"**预期内容**: {issue['expected']}\n"

                report += "\n"

        # 修复建议
        if result['recommendations']:
            report += "## 💡 修复建议\n\n"
            for rec in result['recommendations']:
                report += f"- {rec}\n"
            report += "\n"

        # 一致性评分
        total_layers = result['summary']['total_layers']
        existing_layers = result['summary']['existing_layers']
        consistency_score = (existing_layers / total_layers) * 100 if total_layers > 0 else 0

        report += f"""## 📈 一致性评分

### 架构一致性
- **总层级数**: {total_layers} 个
- **存在层级**: {existing_layers} 个
- **缺失层级**: {result['summary']['missing_layers']} 个
- **一致性得分**: {consistency_score:.1f}%

### 问题统计
- **总问题数**: {result['summary']['total_issues']} 个
- **高优先级**: {result['summary']['missing_layers']} 个
- **中优先级**: {result['summary']['redundant_directories']} 个
- **低优先级**: {result['summary']['unexpected_directories']} 个

---

**检查工具**: scripts/architecture_consistency_check.py
**检查标准**: 基于架构设计文档 v5.0
**建议处理**: 按严重程度从高到低修复问题
"""

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='架构一致性检查工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--output', help='输出报告文件')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='报告格式')

    args = parser.parse_args()

    checker = ArchitectureConsistencyChecker(args.project)
    result = checker.check_directory_structure()

    if args.format == 'json':
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        report = checker.generate_consistency_report(result)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(report)
        else:
            print(report)


if __name__ == "__main__":
    main()

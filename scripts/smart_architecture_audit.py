#!/usr/bin/env python3
"""
智能架构审计工具

更准确地检查src目录与架构设计的符合性：
1. 区分业务概念与架构职责
2. 识别真正的架构层级混乱
3. 分析接口设计的合理性
4. 检查依赖关系的规范性
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import re


class SmartArchitectureAuditor:
    """智能架构审计器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"

        # 架构层级定义和职责
        self.layer_definitions = {
            "core": {
                "name": "核心服务层",
                "primary_responsibilities": [
                    "事件总线", "依赖注入", "业务流程编排", "系统集成"
                ],
                "allowed_concepts": ["event", "bus", "container", "orchestrator", "process"],
                "restricted_concepts": []  # 核心层可以包含任何业务概念
            },
            "infrastructure": {
                "name": "基础设施层",
                "primary_responsibilities": [
                    "配置管理", "缓存系统", "日志系统", "安全管理",
                    "错误处理", "资源管理", "健康检查", "工具组件"
                ],
                "allowed_concepts": ["config", "cache", "logging", "security", "error", "resource", "health", "utils"],
                "restricted_concepts": []  # 基础设施层服务于所有上层
            },
            "data": {
                "name": "数据采集层",
                "primary_responsibilities": [
                    "数据源适配", "实时数据采集", "数据验证", "数据质量监控"
                ],
                "allowed_concepts": ["adapter", "collector", "validator", "quality", "loader", "source", "data"],
                "restricted_concepts": ["trading", "order", "execution"]  # 不应直接处理交易逻辑
            },
            "gateway": {
                "name": "API网关层",
                "primary_responsibilities": [
                    "路由转发", "认证授权", "限流熔断", "请求处理"
                ],
                "allowed_concepts": ["api", "gateway", "route", "auth", "rate", "limit", "middleware"],
                "restricted_concepts": ["trading", "model", "strategy"]  # 网关层不处理业务逻辑
            },
            "features": {
                "name": "特征处理层",
                "primary_responsibilities": [
                    "特征工程", "特征提取", "特征选择", "特征变换", "特征存储"
                ],
                "allowed_concepts": ["feature", "engineering", "distributed", "acceleration", "processor", "extract"],
                "restricted_concepts": ["trading", "order", "execution"]  # 不直接处理交易
            },
            "ml": {
                "name": "模型推理层",
                "primary_responsibilities": [
                    "模型管理", "模型推理", "模型训练", "模型评估", "集成学习"
                ],
                "allowed_concepts": ["model", "ml", "predict", "train", "inference", "ensemble", "tuning"],
                "restricted_concepts": ["trading", "order", "execution"]  # 不直接处理交易
            },
            "backtest": {
                "name": "策略决策层",
                "primary_responsibilities": [
                    "策略生成", "策略评估", "回测执行", "策略优化", "风险评估"
                ],
                "allowed_concepts": ["strategy", "backtest", "analyzer", "evaluation", "performance", "simulation"],
                "restricted_concepts": []  # 策略层可以包含各种业务概念
            },
            "risk": {
                "name": "风控合规层",
                "primary_responsibilities": [
                    "风险检查", "风险监控", "合规验证", "风险评估", "风险控制"
                ],
                "allowed_concepts": ["risk", "compliance", "checker", "monitor", "limit", "threshold", "validation"],
                "restricted_concepts": []  # 风控层服务于所有业务
            },
            "trading": {
                "name": "交易执行层",
                "primary_responsibilities": [
                    "订单管理", "交易执行", "执行监控", "交易记录", "智能路由"
                ],
                "allowed_concepts": ["trading", "order", "execution", "executor", "manager", "broker", "exchange"],
                "restricted_concepts": ["backtest", "simulation"]  # 不应包含回测逻辑
            },
            "engine": {
                "name": "监控反馈层",
                "primary_responsibilities": [
                    "系统监控", "业务监控", "性能监控", "日志聚合", "告警管理"
                ],
                "allowed_concepts": ["monitor", "logging", "alert", "dashboard", "metric", "performance", "health"],
                "restricted_concepts": []  # 监控层服务于所有业务
            }
        }

        # 业务概念词典（这些词在多个层级中都是合理的）
        self.business_concepts = {
            "data", "model", "strategy", "trading", "risk", "feature", "order",
            "config", "cache", "logging", "security", "error", "resource", "health",
            "api", "gateway", "route", "auth", "rate", "limit", "middleware",
            "ml", "predict", "train", "inference", "ensemble", "tuning",
            "backtest", "analyzer", "evaluation", "performance", "simulation",
            "compliance", "checker", "monitor", "limit", "threshold", "validation",
            "execution", "executor", "manager", "broker", "exchange",
            "alert", "dashboard", "metric"
        }

    def perform_smart_audit(self) -> Dict[str, Any]:
        """执行智能架构审计"""
        print("🔍 执行智能架构审计...")

        audit_result = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "layer_audit": {},
            "architecture_issues": [],
            "interface_analysis": {},
            "dependency_analysis": {},
            "quality_metrics": {}
        }

        # 1. 层级架构审计
        print("📊 步骤1: 层级架构审计")
        audit_result["layer_audit"] = self._audit_layer_architecture()

        # 2. 架构问题识别
        print("🔍 步骤2: 架构问题识别")
        audit_result["architecture_issues"] = self._identify_architecture_issues()

        # 3. 接口设计分析
        print("🔗 步骤3: 接口设计分析")
        audit_result["interface_analysis"] = self._analyze_interfaces()

        # 4. 依赖关系分析
        print("⚡ 步骤4: 依赖关系分析")
        audit_result["dependency_analysis"] = self._analyze_dependencies()

        # 5. 质量指标计算
        print("📈 步骤5: 质量指标计算")
        audit_result["quality_metrics"] = self._calculate_quality_metrics(audit_result)

        # 6. 生成摘要
        audit_result["summary"] = self._generate_summary(audit_result)

        print(f"✅ 智能审计完成，发现 {len(audit_result['architecture_issues'])} 个架构问题")
        return audit_result

    def _audit_layer_architecture(self) -> Dict[str, Any]:
        """审计层级架构"""
        layer_audit = {}

        for layer, definition in self.layer_definitions.items():
            layer_path = self.src_dir / layer
            if not layer_path.exists():
                layer_audit[layer] = {
                    "exists": False,
                    "status": "missing",
                    "files": 0,
                    "subdirs": 0
                }
                continue

            layer_audit[layer] = {
                "exists": True,
                "status": "present",
                "files": 0,
                "subdirs": 0,
                "file_list": [],
                "subdir_list": [],
                "primary_responsibility_match": 0,
                "cross_layer_concepts": 0,
                "architecture_violations": []
            }

            # 统计文件和目录
            for item in layer_path.rglob("*"):
                if item.is_file() and item.name.endswith('.py') and not item.name.startswith('_'):
                    layer_audit[layer]["files"] += 1
                    layer_audit[layer]["file_list"].append(str(item.relative_to(self.src_dir)))

                    # 分析文件内容
                    try:
                        file_analysis = self._analyze_file_content(item, layer, definition)
                        layer_audit[layer]["primary_responsibility_match"] += file_analysis["responsibility_match"]
                        layer_audit[layer]["cross_layer_concepts"] += file_analysis["cross_layer_concepts"]

                        if file_analysis["violations"]:
                            layer_audit[layer]["architecture_violations"].extend(
                                file_analysis["violations"])

                    except Exception as e:
                        layer_audit[layer]["architecture_violations"].append({
                            "file": str(item.relative_to(self.src_dir)),
                            "type": "analysis_error",
                            "description": f"文件分析错误: {e}"
                        })

                elif item.is_dir() and not item.name.startswith('_') and item != layer_path:
                    layer_audit[layer]["subdirs"] += 1
                    layer_audit[layer]["subdir_list"].append(item.name)

        return layer_audit

    def _analyze_file_content(self, file_path: Path, layer: str, definition: Dict[str, Any]) -> Dict[str, Any]:
        """分析文件内容"""
        analysis = {
            "responsibility_match": 0,
            "cross_layer_concepts": 0,
            "violations": []
        }

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()

            # 检查主要职责匹配
            for responsibility in definition["primary_responsibilities"]:
                if responsibility.lower() in content:
                    analysis["responsibility_match"] += 1

            # 检查允许的概念
            for concept in definition["allowed_concepts"]:
                if concept in content:
                    analysis["responsibility_match"] += 0.5

            # 检查受限概念（真正的架构违规）
            for concept in definition["restricted_concepts"]:
                if concept.lower() in content:
                    analysis["violations"].append({
                        "type": "restricted_concept",
                        "concept": concept,
                        "description": f"文件包含受限概念 '{concept}'，违反 {definition['name']} 的架构约束"
                    })

            # 检查是否有明显的层级混乱
            cross_layer_indicators = self._check_cross_layer_indicators(content, layer)
            if cross_layer_indicators:
                analysis["cross_layer_concepts"] += len(cross_layer_indicators)
                analysis["violations"].extend(cross_layer_indicators)

        except Exception as e:
            analysis["violations"].append({
                "type": "file_error",
                "description": f"文件读取错误: {e}"
            })

        return analysis

    def _check_cross_layer_indicators(self, content: str, current_layer: str) -> List[Dict[str, Any]]:
        """检查跨层级指标"""
        violations = []

        # 定义层级特定的违规模式
        layer_violations = {
            "data": [
                (r"trading\.order", "数据层直接处理交易订单"),
                (r"trading\.execution", "数据层直接执行交易"),
                (r"model\.predict", "数据层直接调用模型预测")
            ],
            "features": [
                (r"trading\.execute", "特征层直接执行交易"),
                (r"order\.submit", "特征层直接提交订单")
            ],
            "ml": [
                (r"trading\.execute", "模型层直接执行交易"),
                (r"order\.routing", "模型层直接处理订单路由")
            ],
            "backtest": [
                (r"live\.trading", "回测层包含实盘交易代码"),
                (r"production\.order", "回测层包含生产订单处理")
            ],
            "trading": [
                (r"backtest\.engine", "交易层包含回测引擎"),
                (r"simulation\.mode", "交易层包含模拟模式")
            ]
        }

        if current_layer in layer_violations:
            for pattern, description in layer_violations[current_layer]:
                if re.search(pattern, content, re.IGNORECASE):
                    violations.append({
                        "type": "cross_layer_violation",
                        "pattern": pattern,
                        "description": description
                    })

        return violations

    def _identify_architecture_issues(self) -> List[Dict[str, Any]]:
        """识别架构问题"""
        issues = []

        # 检查缺失的架构层级
        for layer, definition in self.layer_definitions.items():
            layer_path = self.src_dir / layer
            if not layer_path.exists():
                issues.append({
                    "type": "missing_layer",
                    "layer": layer,
                    "severity": "high",
                    "description": f"缺少架构层级: {definition['name']}",
                    "impact": "破坏架构完整性"
                })

        # 检查是否有文件放在错误的层级
        for layer, layer_audit in self._audit_layer_architecture().items():
            if not layer_audit["exists"]:
                continue

            for violation in layer_audit["architecture_violations"]:
                if violation["type"] == "restricted_concept":
                    issues.append({
                        "type": "wrong_layer_placement",
                        "layer": layer,
                        "file": violation.get("file", "unknown"),
                        "severity": "medium",
                        "description": violation["description"],
                        "impact": "架构职责混乱"
                    })
                elif violation["type"] == "cross_layer_violation":
                    issues.append({
                        "type": "cross_layer_coupling",
                        "layer": layer,
                        "file": violation.get("file", "unknown"),
                        "severity": "high",
                        "description": violation["description"],
                        "impact": "层级耦合过紧"
                    })

        return issues

    def _analyze_interfaces(self) -> Dict[str, Any]:
        """分析接口设计"""
        interface_analysis = {}

        for layer in self.layer_definitions.keys():
            layer_path = self.src_dir / layer
            if not layer_path.exists():
                continue

            interface_analysis[layer] = {
                "interface_files": 0,
                "base_implementation_files": 0,
                "standard_interfaces": 0,
                "interface_issues": []
            }

            # 检查接口文件
            for py_file in layer_path.rglob("*.py"):
                if py_file.name.startswith("_"):
                    continue

                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if "interfaces.py" in py_file.name:
                        interface_analysis[layer]["interface_files"] += 1

                        # 检查标准接口定义
                        if re.search(r"class\s+I[A-Z]\w+Component\(ABC\):", content):
                            interface_analysis[layer]["standard_interfaces"] += 1
                        else:
                            interface_analysis[layer]["interface_issues"].append({
                                "file": str(py_file.relative_to(self.src_dir)),
                                "issue": "接口命名不符合标准规范 I{Name}Component"
                            })

                    elif "base.py" in py_file.name:
                        interface_analysis[layer]["base_implementation_files"] += 1

                        # 检查基础实现模式
                        if re.search(r"class\s+Base[A-Z]\w+Component\(", content):
                            pass  # 符合标准
                        else:
                            interface_analysis[layer]["interface_issues"].append({
                                "file": str(py_file.relative_to(self.src_dir)),
                                "issue": "基础实现类不符合标准模式 Base{Name}Component"
                            })

                except Exception as e:
                    interface_analysis[layer]["interface_issues"].append({
                        "file": str(py_file.relative_to(self.src_dir)),
                        "issue": f"接口分析错误: {e}"
                    })

        return interface_analysis

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """分析依赖关系"""
        dependency_analysis = {}

        for layer in self.layer_definitions.keys():
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
                        lines = f.read().split('\n')

                    for line in lines:
                        line = line.strip()
                        if line.startswith('import ') or line.startswith('from '):
                            # 分析导入语句
                            if 'src.' in line:
                                if f'src.{layer}' in line:
                                    dependency_analysis[layer]["internal_imports"] += 1
                                else:
                                    dependency_analysis[layer]["cross_layer_imports"] += 1

                                    # 检查是否是合理的跨层级导入
                                    if not self._is_reasonable_cross_layer_import(line, layer):
                                        dependency_analysis[layer]["dependency_issues"].append({
                                            "file": str(py_file.relative_to(self.src_dir)),
                                            "import": line,
                                            "issue": "不合理的跨层级导入"
                                        })
                            else:
                                dependency_analysis[layer]["external_imports"] += 1

                except Exception as e:
                    dependency_analysis[layer]["dependency_issues"].append({
                        "file": str(py_file.relative_to(self.src_dir)),
                        "issue": f"依赖分析错误: {e}"
                    })

        return dependency_analysis

    def _is_reasonable_cross_layer_import(self, import_line: str, current_layer: str) -> bool:
        """检查是否是合理的跨层级导入"""
        # 定义合理的跨层级导入规则
        reasonable_imports = {
            "infrastructure": ["*"],  # 基础设施层可以被任何层级导入
            "core": ["*"],  # 核心层可以被任何层级导入
            "data": ["features", "ml", "backtest", "risk", "trading"],  # 数据层可以被上层业务层导入
            "gateway": ["*"],  # 网关层可以被任何业务层使用
            "features": ["ml", "backtest", "risk", "trading"],  # 特征层可以被上层业务层导入
            "ml": ["backtest", "risk", "trading"],  # 模型层可以被业务层导入
            "backtest": ["risk", "trading"],  # 回测层可以被风控和交易层使用
            "risk": ["trading"],  # 风控层可以被交易层使用
            "trading": ["engine"],  # 交易层可以被监控层监控
            "engine": []  # 监控层通常不被其他层级直接导入
        }

        if current_layer not in reasonable_imports:
            return False

        allowed_targets = reasonable_imports[current_layer]

        if "*" in allowed_targets:
            return True

        # 检查导入目标是否在允许列表中
        for target in allowed_targets:
            if f"src.{target}" in import_line:
                return True

        return False

    def _calculate_quality_metrics(self, audit_result: Dict[str, Any]) -> Dict[str, Any]:
        """计算质量指标"""
        metrics = {
            "architecture_completeness": 0,
            "layer_responsibility_score": 0,
            "interface_compliance_score": 0,
            "dependency_health_score": 0,
            "overall_architecture_score": 0
        }

        # 1. 架构完整性评分
        total_layers = len(self.layer_definitions)
        existing_layers = len([l for l in audit_result["layer_audit"].values() if l["exists"]])
        metrics["architecture_completeness"] = (existing_layers / total_layers) * 100

        # 2. 层级职责评分
        total_responsibility_score = 0
        total_files = 0
        for layer_audit in audit_result["layer_audit"].values():
            if layer_audit["exists"]:
                total_responsibility_score += layer_audit.get("primary_responsibility_match", 0)
                total_files += layer_audit.get("files", 0)

        if total_files > 0:
            metrics["layer_responsibility_score"] = (total_responsibility_score / total_files) * 100

        # 3. 接口符合性评分
        total_interfaces = 0
        standard_interfaces = 0
        for interface_analysis in audit_result["interface_analysis"].values():
            total_interfaces += interface_analysis["interface_files"]
            standard_interfaces += interface_analysis["standard_interfaces"]

        if total_interfaces > 0:
            metrics["interface_compliance_score"] = (standard_interfaces / total_interfaces) * 100
        else:
            metrics["interface_compliance_score"] = 100  # 如果没有接口文件，认为符合要求

        # 4. 依赖健康评分
        total_dependency_issues = 0
        for dep_analysis in audit_result["dependency_analysis"].values():
            total_dependency_issues += len(dep_analysis["dependency_issues"])

        # 简单的依赖健康评分算法
        if total_dependency_issues == 0:
            metrics["dependency_health_score"] = 100
        elif total_dependency_issues < 5:
            metrics["dependency_health_score"] = 80
        elif total_dependency_issues < 10:
            metrics["dependency_health_score"] = 60
        else:
            metrics["dependency_health_score"] = 40

        # 5. 综合架构评分
        weights = {
            "architecture_completeness": 0.3,
            "layer_responsibility_score": 0.3,
            "interface_compliance_score": 0.2,
            "dependency_health_score": 0.2
        }

        overall_score = 0
        for metric, weight in weights.items():
            overall_score += metrics[metric] * weight

        metrics["overall_architecture_score"] = overall_score

        return metrics

    def _generate_summary(self, audit_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成审计摘要"""
        summary = {
            "audit_time": audit_result["timestamp"],
            "total_issues": len(audit_result["architecture_issues"]),
            "critical_issues": len([i for i in audit_result["architecture_issues"] if i["severity"] == "high"]),
            "warning_issues": len([i for i in audit_result["architecture_issues"] if i["severity"] == "medium"]),
            "info_issues": len([i for i in audit_result["architecture_issues"] if i["severity"] == "low"]),
            "architecture_score": audit_result["quality_metrics"]["overall_architecture_score"]
        }

        # 生成状态描述
        if summary["architecture_score"] >= 90:
            summary["status"] = "excellent"
            summary["status_description"] = "架构质量优秀，完全符合设计要求"
        elif summary["architecture_score"] >= 75:
            summary["status"] = "good"
            summary["status_description"] = "架构质量良好，基本符合设计要求"
        elif summary["architecture_score"] >= 60:
            summary["status"] = "fair"
            summary["status_description"] = "架构质量一般，存在一些需要改进的地方"
        else:
            summary["status"] = "poor"
            summary["status_description"] = "架构质量较差，需要重点改进"

        return summary

    def generate_audit_report(self, audit_result: Dict[str, Any]) -> str:
        """生成审计报告"""
        report = f"""# 智能架构审计报告

## 📊 审计概览

**审计时间**: {audit_result['summary']['audit_time']}
**架构评分**: {audit_result['summary']['architecture_score']:.1f}/100
**发现问题**: {audit_result['summary']['total_issues']} 个
**状态**: {audit_result['summary']['status_description']}

### 问题统计
| 问题类型 | 数量 | 严重程度 |
|---------|------|---------|
| 关键问题 | {audit_result['summary']['critical_issues']} | 🔴 高 |
| 警告问题 | {audit_result['summary']['warning_issues']} | 🟡 中 |
| 信息问题 | {audit_result['summary']['info_issues']} | 🟢 低 |

---

## 🏗️ 层级架构审计结果

"""

        # 各层级审计结果
        for layer, layer_audit in audit_result['layer_audit'].items():
            layer_name = self.layer_definitions[layer]['name']
            report += f"### {layer.upper()} 层级 ({layer_name})\n"

            if not layer_audit["exists"]:
                report += "❌ **状态**: 层级不存在\n\n"
                continue

            report += f"**文件数量**: {layer_audit['files']} 个\n"
            report += f"**子目录数量**: {layer_audit['subdirs']} 个\n"
            report += f"**职责匹配度**: {layer_audit.get('primary_responsibility_match', 0)} 个主要职责匹配\n"
            report += f"**跨层级概念**: {layer_audit.get('cross_layer_concepts', 0)} 个\n\n"

            if layer_audit["subdir_list"]:
                report += "**子目录列表**:\n"
                for subdir in layer_audit["subdir_list"]:
                    report += f"- `{subdir}`\n"
                report += "\n"

            if layer_audit["architecture_violations"]:
                report += "**架构问题**:\n"
                for violation in layer_audit["architecture_violations"]:
                    severity_emoji = "🔴" if violation.get(
                        "type") == "cross_layer_violation" else "🟡"
                    report += f"- {severity_emoji} {violation.get('description', '未知问题')}\n"
                report += "\n"

        # 架构问题详情
        if audit_result['architecture_issues']:
            report += "## 🔍 架构问题详情\n\n"

            for issue in audit_result['architecture_issues']:
                severity_emoji = {
                    "high": "🔴",
                    "medium": "🟡",
                    "low": "🟢"
                }.get(issue['severity'], "⚪")

                report += f"### {severity_emoji} {issue['type'].replace('_', ' ').title()}\n"
                report += f"**层级**: {issue.get('layer', 'N/A')}\n"
                report += f"**严重程度**: {issue['severity']}\n"
                report += f"**影响**: {issue.get('impact', 'N/A')}\n"
                report += f"**描述**: {issue['description']}\n"

                if 'file' in issue:
                    report += f"**文件**: `{issue['file']}`\n"

                report += "\n"

        # 质量指标
        report += "## 📈 质量指标\n\n"
        metrics = audit_result['quality_metrics']

        report += "### 详细指标\n"
        report += f"- **架构完整性**: {metrics['architecture_completeness']:.1f}% - 层级完整程度\n"
        report += f"- **职责匹配度**: {metrics['layer_responsibility_score']:.1f}% - 文件职责符合度\n"
        report += f"- **接口符合性**: {metrics['interface_compliance_score']:.1f}% - 标准接口使用率\n"
        report += f"- **依赖健康度**: {metrics['dependency_health_score']:.1f}% - 依赖关系健康度\n"
        report += f"- **综合评分**: {metrics['overall_architecture_score']:.1f}% - 整体架构质量\n\n"

        # 评分标准解释
        report += "### 评分标准\n"
        report += "- **90-100%**: 架构质量优秀，完全符合设计要求\n"
        report += "- **75-89%**: 架构质量良好，基本符合设计要求\n"
        report += "- **60-74%**: 架构质量一般，存在一些需要改进的地方\n"
        report += "- **0-59%**: 架构质量较差，需要重点改进\n\n"

        # 接口分析
        report += "## 🔗 接口设计分析\n\n"

        for layer, interface_analysis in audit_result['interface_analysis'].items():
            if interface_analysis['interface_files'] > 0 or interface_analysis['base_implementation_files'] > 0:
                layer_name = self.layer_definitions[layer]['name']
                report += f"### {layer.upper()} 层级接口 ({layer_name})\n"
                report += f"- **接口文件**: {interface_analysis['interface_files']} 个\n"
                report += f"- **基础实现**: {interface_analysis['base_implementation_files']} 个\n"
                report += f"- **标准接口**: {interface_analysis['standard_interfaces']} 个\n\n"

                if interface_analysis['interface_issues']:
                    report += "**接口问题**:\n"
                    for issue in interface_analysis['interface_issues']:
                        report += f"- ⚠️ {issue.get('file', 'N/A')}: {issue.get('issue', 'N/A')}\n"
                    report += "\n"

        # 依赖分析
        report += "## ⚡ 依赖关系分析\n\n"

        for layer, dep_analysis in audit_result['dependency_analysis'].items():
            layer_name = self.layer_definitions[layer]['name']
            report += f"### {layer.upper()} 层级依赖 ({layer_name})\n"
            report += f"- **内部导入**: {dep_analysis['internal_imports']} 个\n"
            report += f"- **外部导入**: {dep_analysis['external_imports']} 个\n"
            report += f"- **跨层级导入**: {dep_analysis['cross_layer_imports']} 个\n\n"

            if dep_analysis['dependency_issues']:
                report += "**依赖问题**:\n"
                for issue in dep_analysis['dependency_issues']:
                    report += f"- ⚠️ {issue.get('file', 'N/A')}: {issue.get('issue', 'N/A')}\n"
                report += "\n"

        report += f"""---

**审计工具**: scripts/smart_architecture_audit.py
**审计标准**: 基于架构设计文档 v5.0
**建议处理**: 按严重程度从高到低修复问题
"""

        return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='智能架构审计工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--output', help='输出报告文件')
    parser.add_argument('--format', choices=['text', 'json'], default='text', help='报告格式')

    args = parser.parse_args()

    auditor = SmartArchitectureAuditor(args.project)
    audit_result = auditor.perform_smart_audit()

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

#!/usr/bin/env python3
"""
全面架构一致性检查工具

检查所有架构层的目录结构是否与架构设计文档一致
包括：基础设施层、数据采集层、特征处理层、模型推理层、策略决策层、风控合规层、交易执行层、监控反馈层等
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime


class ComprehensiveArchitectureConsistencyChecker:
    """全面架构一致性检查器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        # 架构层级定义（根据业务流程驱动的架构设计）
        self.architecture_layers = {
            "infrastructure": {
                "name": "基础设施层",
                "path": "infrastructure",
                "description": "提供系统的基础设施服务",
                "sub_components": {
                    "config": "配置管理",
                    "cache": "缓存系统",
                    "logging": "日志系统",
                    "security": "安全管理",
                    "error": "错误处理",
                    "resource": "资源管理",
                    "health": "健康检查",
                    "utils": "工具组件"
                },
                "expected_files_range": (600, 700),  # 文件数量范围
                "critical_files": ["__init__.py", "interfaces.py", "base.py"]
            },
            "data": {
                "name": "数据采集层",
                "path": "data",
                "description": "负责数据采集、处理和存储",
                "sub_components": {
                    "adapters": "数据源适配器",
                    "loader": "数据加载器",
                    "processing": "数据处理",
                    "quality": "数据质量",
                    "validation": "数据验证",
                    "cache": "数据缓存",
                    "monitoring": "数据监控"
                },
                "expected_files_range": (400, 500),
                "critical_files": ["__init__.py", "data_manager.py", "interfaces.py"]
            },
            "features": {
                "name": "特征处理层",
                "path": "features",
                "description": "特征工程和处理",
                "sub_components": {
                    "engineering": "特征工程",
                    "processors": "特征处理器",
                    "acceleration": "硬件加速",
                    "monitoring": "特征监控",
                    "store": "特征存储"
                },
                "expected_files_range": (200, 300),
                "critical_files": ["__init__.py", "feature_manager.py"]
            },
            "ml": {
                "name": "模型推理层",
                "path": "ml",
                "description": "机器学习模型推理",
                "sub_components": {
                    "models": "模型定义",
                    "engine": "推理引擎",
                    "ensemble": "模型集成",
                    "tuning": "模型调优"
                },
                "expected_files_range": (100, 200),
                "critical_files": ["__init__.py", "models/model_manager.py"]
            },
            "core": {
                "name": "策略决策层",
                "path": "core",
                "description": "核心业务逻辑和策略决策",
                "sub_components": {
                    "business_process": "业务流程",
                    "event_bus": "事件总线",
                    "service_container": "服务容器",
                    "integration": "集成管理"
                },
                "expected_files_range": (50, 150),
                "critical_files": ["__init__.py", "business_process_orchestrator.py", "event_bus.py"]
            },
            "risk": {
                "name": "风控合规层",
                "path": "risk",
                "description": "风险控制和合规检查",
                "sub_components": {
                    "checker": "风险检查",
                    "monitor": "风险监控",
                    "compliance": "合规检查",
                    "alert": "风险告警"
                },
                "expected_files_range": (30, 80),
                "critical_files": ["__init__.py", "risk_manager.py", "compliance_checker.py"]
            },
            "trading": {
                "name": "交易执行层",
                "path": "trading",
                "description": "交易执行和订单管理",
                "sub_components": {
                    "execution": "交易执行",
                    "order": "订单管理",
                    "position": "仓位管理",
                    "account": "账户管理"
                },
                "expected_files_range": (100, 150),
                "critical_files": ["__init__.py", "trading_engine.py", "order_manager.py"]
            },
            "backtest": {
                "name": "回测分析层",
                "path": "backtest",
                "description": "策略回测和分析",
                "sub_components": {
                    "engine": "回测引擎",
                    "analysis": "回测分析",
                    "evaluation": "策略评估",
                    "optimization": "参数优化"
                },
                "expected_files_range": (50, 100),
                "critical_files": ["__init__.py", "backtest_engine.py", "analyzer.py"]
            },
            "engine": {
                "name": "引擎层",
                "path": "engine",
                "description": "实时引擎和Web服务",
                "sub_components": {
                    "web": "Web服务",
                    "realtime": "实时引擎",
                    "optimization": "性能优化",
                    "monitoring": "引擎监控"
                },
                "expected_files_range": (100, 200),
                "critical_files": ["__init__.py", "realtime_engine.py"]
            },
            "gateway": {
                "name": "API网关层",
                "path": "gateway",
                "description": "API网关和路由",
                "sub_components": {
                    "api_gateway": "API网关",
                    "routing": "路由管理",
                    "auth": "认证授权"
                },
                "expected_files_range": (10, 50),
                "critical_files": ["__init__.py", "api_gateway.py"]
            }
        }

        # 检查结果
        self.check_results = {}

    def run_comprehensive_check(self) -> Dict[str, Any]:
        """运行全面架构一致性检查"""
        print("🔍 开始全面架构一致性检查...")

        self.check_results = {
            "timestamp": datetime.now(),
            "summary": {},
            "layer_checks": {},
            "issues": [],
            "recommendations": [],
            "overall_score": 0
        }

        # 检查所有架构层
        for layer_key, layer_config in self.architecture_layers.items():
            print(f"📋 检查 {layer_config['name']}...")
            layer_result = self._check_layer_consistency(layer_key, layer_config)
            self.check_results["layer_checks"][layer_key] = layer_result

        # 生成总结报告
        self._generate_summary_report()

        # 计算总体评分
        self._calculate_overall_score()

        print(f"✅ 全面架构一致性检查完成，总评分: {self.check_results['overall_score']}/100")

        return self.check_results

    def _check_layer_consistency(self, layer_key: str, layer_config: Dict[str, Any]) -> Dict[str, Any]:
        """检查单个架构层的目录结构一致性"""
        layer_result = {
            "layer_name": layer_config["name"],
            "layer_path": layer_config["path"],
            "exists": False,
            "issues": [],
            "warnings": [],
            "score": 0,
            "file_count": 0,
            "sub_components": {},
            "missing_critical_files": [],
            "naming_issues": []
        }

        layer_path = self.src_dir / layer_config["path"]

        # 检查层级目录是否存在
        if not layer_path.exists():
            layer_result["issues"].append(f"架构层目录不存在: {layer_path}")
            return layer_result

        layer_result["exists"] = True

        # 统计文件数量
        python_files = list(layer_path.rglob("*.py"))
        layer_result["file_count"] = len(python_files)

        # 检查文件数量是否在预期范围内
        min_files, max_files = layer_config["expected_files_range"]

        # 允许特定层的文件数量超出限制
        allowed_excess_layers = ["infrastructure", "features", "trading"]
        allow_excess = layer_key in allowed_excess_layers

        if layer_result["file_count"] < min_files:
            layer_result["warnings"].append(f"文件数量过少: {layer_result['file_count']} < {min_files}")
        elif layer_result["file_count"] > max_files and not allow_excess:
            layer_result["warnings"].append(f"文件数量过多: {layer_result['file_count']} > {max_files}")
        elif layer_result["file_count"] > max_files and allow_excess:
            # 对于允许超出的层，只记录信息，不作为警告
            layer_result["info"] = layer_result.get("info", [])
            layer_result["info"].append(
                f"文件数量超出预期但已允许: {layer_result['file_count']} > {max_files} (允许超出)")

        # 检查关键文件
        for critical_file in layer_config["critical_files"]:
            critical_path = layer_path / critical_file
            if not critical_path.exists():
                layer_result["missing_critical_files"].append(critical_file)

        # 检查子组件
        if layer_config["sub_components"]:
            for sub_component, description in layer_config["sub_components"].items():
                sub_path = layer_path / sub_component
                if sub_path.exists():
                    sub_files = list(sub_path.rglob("*.py"))
                    layer_result["sub_components"][sub_component] = {
                        "exists": True,
                        "file_count": len(sub_files),
                        "description": description
                    }
                else:
                    layer_result["sub_components"][sub_component] = {
                        "exists": False,
                        "file_count": 0,
                        "description": description
                    }

        # 检查命名规范
        layer_result["naming_issues"] = self._check_naming_conventions(layer_path, layer_config)

        # 检查接口一致性
        interface_issues = self._check_interface_consistency(layer_path)
        layer_result["interface_issues"] = interface_issues

        # 检查依赖关系
        dependency_issues = self._check_dependency_consistency(layer_path, layer_config)
        layer_result["dependency_issues"] = dependency_issues

        # 计算层级评分
        layer_result["score"] = self._calculate_layer_score(layer_result)

        return layer_result

    def _check_naming_conventions(self, layer_path: Path, layer_config: Dict[str, Any]) -> List[str]:
        """检查命名规范"""
        issues = []

        # 检查类命名
        for py_file in layer_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查接口命名（I{Name}Component格式）
                interface_matches = re.findall(r'class I[A-Z]\w*Component', content)
                if not interface_matches and 'interface' in py_file.name.lower():
                    issues.append(f"{py_file}: 接口文件缺少标准接口定义")

                # 检查类命名规范
                class_matches = re.findall(r'class ([A-Z]\w+)', content)
                for class_name in class_matches:
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                        issues.append(f"{py_file}: 类名不符合规范: {class_name}")

            except Exception as e:
                issues.append(f"{py_file}: 无法检查命名规范: {e}")

        return issues

    def _check_interface_consistency(self, layer_path: Path) -> List[str]:
        """检查接口一致性"""
        issues = []

        # 查找接口文件
        interface_files = list(layer_path.rglob("*interface*.py")) + \
            list(layer_path.rglob("interfaces.py"))

        for interface_file in interface_files:
            try:
                with open(interface_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查是否包含标准接口定义
                if not re.search(r'class I[A-Z]\w*Component', content):
                    issues.append(f"{interface_file}: 缺少标准接口定义 (I{{Name}}Component)")

                # 检查是否有详细的文档字符串
                if not re.search(r'""".*?"""', content, re.DOTALL):
                    issues.append(f"{interface_file}: 缺少接口文档")

            except Exception as e:
                issues.append(f"{interface_file}: 无法检查接口一致性: {e}")

        return issues

    def _check_dependency_consistency(self, layer_path: Path, layer_config: Dict[str, Any]) -> List[str]:
        """检查依赖关系一致性"""
        issues = []

        # 检查循环依赖
        dependency_graph = self._build_dependency_graph(layer_path)
        cycles = self._detect_cycles(dependency_graph)
        if cycles:
            issues.append(f"检测到循环依赖: {cycles}")

        # 检查不合理的跨层依赖
        cross_layer_imports = self._find_cross_layer_imports(layer_path, layer_config["path"])
        for import_issue in cross_layer_imports:
            issues.append(import_issue)

        return issues

    def _build_dependency_graph(self, layer_path: Path) -> Dict[str, List[str]]:
        """构建依赖图"""
        graph = {}

        for py_file in layer_path.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue

            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                imports = re.findall(
                    r'^from ([\w.]+) import|^import ([\w.]+)', content, re.MULTILINE)
                dependencies = []
                for imp in imports:
                    module = imp[0] or imp[1]
                    if module.startswith('.'):
                        # 相对导入
                        dependencies.append(str(py_file.parent / module.replace('.', '/')) + '.py')
                    elif module.startswith('src.'):
                        # 绝对导入
                        dependencies.append(
                            str(self.src_dir / module.replace('src.', '').replace('.', '/')) + '.py')

                graph[str(py_file)] = dependencies

            except Exception:
                continue

        return graph

    def _detect_cycles(self, graph: Dict[str, List[str]]) -> List[List[str]]:
        """检测循环依赖"""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor, path):
                        return True
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:])
                    return True

            rec_stack.remove(node)
            path.pop()
            return False

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _find_cross_layer_imports(self, layer_path: Path, current_layer: str) -> List[str]:
        """查找不合理的跨层依赖"""
        issues = []

        for py_file in layer_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # 检查导入语句及其上下文
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    # 检查导入语句
                    import_match = re.match(
                        r'^from src\.(\w+) import|^import src\.(\w+)', line.strip())
                    if import_match:
                        imported_layer = import_match.group(1) or import_match.group(2)

                        # 检查是否为不合理的跨层导入
                        if self._is_unreasonable_cross_layer_import(current_layer, imported_layer):
                            # 检查整个文件是否有"合理跨层级导入"的注释
                            has_justification = False
                            justification_patterns = [
                                "合理跨层级导入",
                                "合理跨层导入",
                                "Reasonable cross-layer import",
                                "# 跨层级导入：",
                                "# 跨层导入："
                            ]

                            for pattern in justification_patterns:
                                if pattern in content:
                                    has_justification = True
                                    break

                            # 如果没有合理理由，则报告问题
                            if not has_justification:
                                issues.append(f"{py_file}: 不合理的跨层导入: {imported_layer}")

            except Exception:
                continue

        return issues

    def _is_unreasonable_cross_layer_import(self, from_layer: str, to_layer: str) -> bool:
        """判断是否为不合理的跨层导入"""
        # 定义合理的依赖关系
        reasonable_dependencies = {
            "infrastructure": [],  # 基础设施层不应该依赖其他层
            "data": ["infrastructure"],
            "features": ["infrastructure", "data"],
            "ml": ["infrastructure", "data", "features"],
            "core": ["infrastructure", "data", "features", "ml"],
            "risk": ["infrastructure", "data", "features", "ml", "core"],
            "trading": ["infrastructure", "data", "features", "ml", "core", "risk"],
            "backtest": ["infrastructure", "data", "features", "ml", "core", "risk", "trading"],
            "engine": ["infrastructure", "data", "features", "ml", "core", "risk", "trading"],
            "gateway": ["infrastructure", "data", "features", "ml", "core", "risk", "trading"]
        }

        return to_layer not in reasonable_dependencies.get(from_layer, [])

    def _calculate_layer_score(self, layer_result: Dict[str, Any]) -> float:
        """计算层级评分"""
        score = 100.0

        # 目录存在性 (-50分)
        if not layer_result["exists"]:
            return 0.0

        # 关键文件缺失 (-10分/个)
        score -= len(layer_result["missing_critical_files"]) * 10

        # 命名问题 (-5分/个)
        score -= len(layer_result["naming_issues"]) * 5

        # 接口问题 (-5分/个)
        score -= len(layer_result["interface_issues"]) * 5

        # 依赖问题 (-10分/个)
        score -= len(layer_result["dependency_issues"]) * 10

        # 子组件完整性
        total_sub_components = len(layer_result.get("sub_components", {}))
        if total_sub_components > 0:
            existing_components = sum(
                1 for comp in layer_result["sub_components"].values() if comp["exists"])
            component_score = (existing_components / total_sub_components) * 20
            score += component_score

        # 文件数量合理性 - 允许特定层超出限制
        if "warnings" in layer_result and layer_result["warnings"]:
            # 检查是否只是文件数量过多且该层允许超出
            file_count_warnings = [w for w in layer_result["warnings"] if "文件数量过多" in w]
            other_warnings = [w for w in layer_result["warnings"] if "文件数量过多" not in w]

            # 如果只有文件数量过多且该层允许超出，则不扣分
            allowed_excess_layers = ["infrastructure", "features", "trading"]
            if len(file_count_warnings) == len(layer_result["warnings"]) and layer_key in allowed_excess_layers:
                score -= 0  # 不扣分
            else:
                score -= 10 * (len(other_warnings) + len(file_count_warnings))  # 对其他警告和不允许超出的层扣分

        return max(0, min(100, score))

    def _generate_summary_report(self):
        """生成总结报告"""
        summary = {
            "total_layers": len(self.architecture_layers),
            "existing_layers": 0,
            "total_files": 0,
            "total_issues": 0,
            "total_warnings": 0,
            "missing_critical_files": 0,
            "naming_issues": 0,
            "interface_issues": 0,
            "dependency_issues": 0
        }

        for layer_result in self.check_results["layer_checks"].values():
            if layer_result["exists"]:
                summary["existing_layers"] += 1

            summary["total_files"] += layer_result["file_count"]
            summary["total_issues"] += len(layer_result["issues"])
            summary["total_warnings"] += len(layer_result["warnings"])
            summary["missing_critical_files"] += len(layer_result["missing_critical_files"])
            summary["naming_issues"] += len(layer_result["naming_issues"])
            summary["interface_issues"] += len(layer_result.get("interface_issues", []))
            summary["dependency_issues"] += len(layer_result.get("dependency_issues", []))

        self.check_results["summary"] = summary

        # 生成问题列表
        for layer_key, layer_result in self.check_results["layer_checks"].items():
            for issue in layer_result["issues"]:
                self.check_results["issues"].append({
                    "layer": layer_result["layer_name"],
                    "type": "error",
                    "description": issue
                })

            for warning in layer_result["warnings"]:
                self.check_results["issues"].append({
                    "layer": layer_result["layer_name"],
                    "type": "warning",
                    "description": warning
                })

    def _calculate_overall_score(self):
        """计算总体评分"""
        if not self.check_results["layer_checks"]:
            self.check_results["overall_score"] = 0
            return

        total_score = 0
        layer_count = 0

        for layer_result in self.check_results["layer_checks"].values():
            total_score += layer_result["score"]
            layer_count += 1

        if layer_count > 0:
            self.check_results["overall_score"] = round(total_score / layer_count, 2)
        else:
            self.check_results["overall_score"] = 0

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合报告"""
        report_data = self.run_comprehensive_check()

        # 保存详细报告
        report_path = self.reports_dir / \
            f"comprehensive_architecture_consistency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)

        # 生成HTML报告
        html_report_path = self._generate_html_report(report_data)

        # 生成修复建议
        recommendations = self._generate_recommendations(report_data)

        return {
            "success": True,
            "overall_score": report_data["overall_score"],
            "json_report": str(report_path),
            "html_report": str(html_report_path),
            "recommendations": recommendations,
            "summary": report_data["summary"],
            "issues": report_data["issues"]
        }

    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """生成HTML报告"""
        # 计算分数颜色
        if report_data['overall_score'] >= 80:
            score_color = "green"
        elif report_data['overall_score'] >= 60:
            score_color = "orange"
        else:
            score_color = "red"

        html_content = ".2f"","f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>全面架构一致性检查报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .score {{ font-size: 48px; font-weight: bold; color: {score_color}; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .layer {{ margin-bottom: 20px; border: 1px solid #dee2e6; border-radius: 8px; overflow: hidden; }}
        .layer-header {{ background: #007bff; color: white; padding: 15px; font-weight: bold; }}
        .layer-content {{ padding: 20px; }}
        .issues {{ margin-top: 20px; }}
        .issue {{ padding: 10px; margin-bottom: 10px; border-radius: 4px; }}
        .issue.error {{ background: #f8d7da; border-left: 4px solid #dc3545; }}
        .issue.warning {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .recommendations {{ background: #d1ecf1; padding: 20px; border-radius: 8px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📋 全面架构一致性检查报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <div class="score">{report_data['overall_score']}/100</div>
            <p>总体一致性评分</p>
        </div>

        <div class="summary">
            <div class="card">
                <h3>总览</h3>
                <p>检查层数: {report_data['summary']['total_layers']}</p>
                <p>现有层数: {report_data['summary']['existing_layers']}</p>
                <p>总文件数: {report_data['summary']['total_files']}</p>
            </div>
            <div class="card">
                <h3>问题统计</h3>
                <p>总问题数: {report_data['summary']['total_issues']}</p>
                <p>警告数: {report_data['summary']['total_warnings']}</p>
                <p>关键文件缺失: {report_data['summary']['missing_critical_files']}</p>
            </div>
            <div class="card">
                <h3>规范检查</h3>
                <p>命名问题: {report_data['summary']['naming_issues']}</p>
                <p>接口问题: {report_data['summary']['interface_issues']}</p>
                <p>依赖问题: {report_data['summary']['dependency_issues']}</p>
            </div>
        </div>

        <h2>📊 各层级检查结果</h2>
"""

        # 添加各层级结果
        for layer_key, layer_result in report_data["layer_checks"].items():
            layer_status = "✅" if layer_result["exists"] else "❌"
            score_color = "green" if layer_result["score"] >= 80 else "orange" if layer_result["score"] >= 60 else "red"

            html_content += ".2f"","f"""
        <div class="layer">
            <div class="layer-header">
                {layer_status} {layer_result['layer_name']} - 评分: {layer_result['score']}/100
            </div>
            <div class="layer-content">
                <p><strong>文件数量:</strong> {layer_result['file_count']}</p>
                <p><strong>子组件:</strong> {len(layer_result['sub_components'])}</p>
"""

            if layer_result["missing_critical_files"]:
                html_content += f"<p><strong>缺失关键文件:</strong> {', '.join(layer_result['missing_critical_files'])}</p>"

            if layer_result["warnings"]:
                html_content += "<p><strong>警告:</strong></p><ul>"
                for warning in layer_result["warnings"]:
                    html_content += f"<li>{warning}</li>"
                html_content += "</ul>"

            if layer_result.get("info"):
                html_content += "<p><strong>信息:</strong></p><ul>"
                for info in layer_result["info"]:
                    html_content += f"<li style='color: blue;'>{info}</li>"
                html_content += "</ul>"

            html_content += "</div></div>"

        # 添加问题列表
        html_content += """
        <div class="issues">
            <h2>🔍 详细问题列表</h2>
"""

        for issue in report_data["issues"]:
            html_content += ".2f"f"""
            <div class="issue {issue['type']}">
                <strong>{issue['layer']}</strong>: {issue['description']}
            </div>
"""

        html_content += """
        </div>
    </div>
</body>
</html>
"""

        # 保存HTML报告
        html_report_path = self.reports_dir / \
            f"comprehensive_architecture_consistency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        with open(html_report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(html_report_path)

    def _generate_recommendations(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成修复建议"""
        recommendations = []

        # 基于问题类型生成建议
        if report_data["summary"]["missing_critical_files"] > 0:
            recommendations.append({
                "priority": "high",
                "category": "关键文件修复",
                "description": f"创建缺失的 {report_data['summary']['missing_critical_files']} 个关键文件",
                "implementation": "根据架构设计文档创建相应的 __init__.py, interfaces.py 等文件"
            })

        if report_data["summary"]["naming_issues"] > 0:
            recommendations.append({
                "priority": "medium",
                "category": "命名规范修复",
                "description": f"修复 {report_data['summary']['naming_issues']} 个命名规范问题",
                "implementation": "统一使用 I{Name}Component 接口命名规范和标准的类命名规范"
            })

        if report_data["summary"]["interface_issues"] > 0:
            recommendations.append({
                "priority": "medium",
                "category": "接口一致性修复",
                "description": f"修复 {report_data['summary']['interface_issues']} 个接口一致性问题",
                "implementation": "完善接口文档，统一接口定义格式"
            })

        if report_data["summary"]["dependency_issues"] > 0:
            recommendations.append({
                "priority": "high",
                "category": "依赖关系优化",
                "description": f"修复 {report_data['summary']['dependency_issues']} 个依赖关系问题",
                "implementation": "解决循环依赖，优化跨层依赖关系"
            })

        # 基于层级评分生成建议
        for layer_key, layer_result in report_data["layer_checks"].items():
            if layer_result["score"] < 60:
                recommendations.append({
                    "priority": "high",
                    "category": f"{layer_result['layer_name']} 重构",
                    "description": f"{layer_result['layer_name']} 评分过低 ({layer_result['score']}/100)",
                    "implementation": "进行深度重构，完善目录结构和组件划分"
                })
            elif layer_result["score"] < 80:
                recommendations.append({
                    "priority": "medium",
                    "category": f"{layer_result['layer_name']} 优化",
                    "description": f"{layer_result['layer_name']} 评分中等 ({layer_result['score']}/100)",
                    "implementation": "优化现有结构，完善缺失的组件和文件"
                })

        return recommendations


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='全面架构一致性检查工具')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--check', action='store_true', help='运行全面一致性检查')
    parser.add_argument('--report', action='store_true', help='生成完整报告')
    parser.add_argument('--fix', action='store_true', help='自动修复问题')
    parser.add_argument('--output', help='输出报告路径')

    args = parser.parse_args()

    checker = ComprehensiveArchitectureConsistencyChecker(args.project)

    if args.check or args.report:
        result = checker.generate_comprehensive_report()

        print(f"🎯 总体评分: {result['overall_score']}/100")
        print(f"📊 问题总数: {len(result['issues'])}")
        print(f"📋 建议数量: {len(result['recommendations'])}")

        if result['issues']:
            print("\n🔍 主要问题:")
            for issue in result['issues'][:10]:  # 显示前10个问题
                print(f"  - {issue['layer']}: {issue['description']}")

        if result['recommendations']:
            print("\n💡 修复建议:")
            for rec in result['recommendations'][:5]:  # 显示前5个建议
                print(f"  - [{rec['priority']}] {rec['category']}: {rec['description']}")

    elif args.fix:
        print("🔧 自动修复功能待实现...")

    else:
        print("🏗️ 全面架构一致性检查工具")
        print("使用 --help 查看可用命令")


if __name__ == "__main__":
    main()

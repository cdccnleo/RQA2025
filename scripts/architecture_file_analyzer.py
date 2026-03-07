#!/usr/bin/env python3
"""
架构文件分析器

深入分析各架构层的文件数量和结构，识别缺失的组件
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict


class ArchitectureFileAnalyzer:
    """架构文件分析器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.src_dir = self.project_root / "src"
        self.reports_dir = self.project_root / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        # 架构层级详细定义
        self.layer_definitions = {
            "infrastructure": {
                "name": "基础设施层",
                "expected_files": 600,
                "min_files": 500,
                "components": {
                    "config": {
                        "name": "配置管理",
                        "expected_files": 80,
                        "file_types": ["config", "manager", "loader", "validator", "strategy", "service"]
                    },
                    "cache": {
                        "name": "缓存系统",
                        "expected_files": 95,
                        "file_types": ["cache", "manager", "service", "strategy", "optimizer", "client"]
                    },
                    "logging": {
                        "name": "日志系统",
                        "expected_files": 75,
                        "file_types": ["logger", "handler", "formatter", "service", "manager", "config"]
                    },
                    "security": {
                        "name": "安全管理",
                        "expected_files": 60,
                        "file_types": ["security", "auth", "encrypt", "audit", "policy", "manager"]
                    },
                    "error": {
                        "name": "错误处理",
                        "expected_files": 55,
                        "file_types": ["error", "exception", "handler", "manager", "recovery", "fallback"]
                    },
                    "resource": {
                        "name": "资源管理",
                        "expected_files": 65,
                        "file_types": ["resource", "manager", "monitor", "quota", "optimizer", "pool"]
                    },
                    "health": {
                        "name": "健康检查",
                        "expected_files": 70,
                        "file_types": ["health", "checker", "monitor", "status", "probe", "alert"]
                    },
                    "utils": {
                        "name": "工具组件",
                        "expected_files": 90,
                        "file_types": ["util", "helper", "tool", "common", "base", "factory"]
                    }
                }
            },
            "data": {
                "name": "数据采集层",
                "expected_files": 400,
                "min_files": 350,
                "components": {
                    "adapters": {
                        "name": "数据源适配器",
                        "expected_files": 30,
                        "file_types": ["adapter", "connector", "client", "source", "provider"]
                    },
                    "loader": {
                        "name": "数据加载器",
                        "expected_files": 50,
                        "file_types": ["loader", "importer", "reader", "fetcher", "collector"]
                    },
                    "processing": {
                        "name": "数据处理",
                        "expected_files": 40,
                        "file_types": ["processor", "transformer", "cleaner", "validator", "filter"]
                    },
                    "quality": {
                        "name": "数据质量",
                        "expected_files": 70,
                        "file_types": ["quality", "validator", "checker", "monitor", "assurance"]
                    },
                    "validation": {
                        "name": "数据验证",
                        "expected_files": 35,
                        "file_types": ["validator", "checker", "verifier", "tester", "assertion"]
                    },
                    "cache": {
                        "name": "数据缓存",
                        "expected_files": 25,
                        "file_types": ["cache", "buffer", "store", "repository"]
                    },
                    "monitoring": {
                        "name": "数据监控",
                        "expected_files": 45,
                        "file_types": ["monitor", "watcher", "tracker", "observer", "metrics"]
                    }
                }
            },
            "features": {
                "name": "特征处理层",
                "expected_files": 200,
                "min_files": 180,
                "components": {
                    "engineering": {
                        "name": "特征工程",
                        "expected_files": 40,
                        "file_types": ["engineer", "extractor", "generator", "builder", "creator"]
                    },
                    "processors": {
                        "name": "特征处理器",
                        "expected_files": 80,
                        "file_types": ["processor", "transformer", "normalizer", "scaler", "encoder"]
                    },
                    "acceleration": {
                        "name": "硬件加速",
                        "expected_files": 30,
                        "file_types": ["gpu", "accelerator", "parallel", "distributed", "optimization"]
                    },
                    "monitoring": {
                        "name": "特征监控",
                        "expected_files": 25,
                        "file_types": ["monitor", "tracker", "analyzer", "profiler", "metrics"]
                    },
                    "store": {
                        "name": "特征存储",
                        "expected_files": 25,
                        "file_types": ["store", "repository", "database", "cache", "persistence"]
                    }
                }
            },
            "ml": {
                "name": "模型推理层",
                "expected_files": 100,
                "min_files": 80,
                "components": {
                    "models": {
                        "name": "模型定义",
                        "expected_files": 30,
                        "file_types": ["model", "network", "architecture", "definition", "structure"]
                    },
                    "engine": {
                        "name": "推理引擎",
                        "expected_files": 25,
                        "file_types": ["engine", "inference", "predictor", "classifier", "regressor"]
                    },
                    "ensemble": {
                        "name": "模型集成",
                        "expected_files": 20,
                        "file_types": ["ensemble", "voting", "stacking", "bagging", "boosting"]
                    },
                    "tuning": {
                        "name": "模型调优",
                        "expected_files": 25,
                        "file_types": ["tuner", "optimizer", "hyperparameter", "search", "grid"]
                    }
                }
            },
            "core": {
                "name": "策略决策层",
                "expected_files": 50,
                "min_files": 40,
                "components": {
                    "business_process": {
                        "name": "业务流程",
                        "expected_files": 15,
                        "file_types": ["process", "workflow", "orchestrator", "coordinator", "manager"]
                    },
                    "event_bus": {
                        "name": "事件总线",
                        "expected_files": 10,
                        "file_types": ["event", "bus", "publisher", "subscriber", "dispatcher"]
                    },
                    "service_container": {
                        "name": "服务容器",
                        "expected_files": 15,
                        "file_types": ["container", "registry", "locator", "resolver", "factory"]
                    },
                    "integration": {
                        "name": "集成管理",
                        "expected_files": 10,
                        "file_types": ["integration", "adapter", "bridge", "connector", "middleware"]
                    }
                }
            },
            "risk": {
                "name": "风控合规层",
                "expected_files": 30,
                "min_files": 25,
                "components": {
                    "checker": {
                        "name": "风险检查",
                        "expected_files": 12,
                        "file_types": ["checker", "validator", "assessor", "evaluator", "analyzer"]
                    },
                    "monitor": {
                        "name": "风险监控",
                        "expected_files": 8,
                        "file_types": ["monitor", "watcher", "tracker", "observer", "alert"]
                    },
                    "compliance": {
                        "name": "合规检查",
                        "expected_files": 10,
                        "file_types": ["compliance", "regulator", "policy", "rule", "standard"]
                    }
                }
            },
            "trading": {
                "name": "交易执行层",
                "expected_files": 150,
                "min_files": 120,
                "components": {
                    "execution": {
                        "name": "交易执行",
                        "expected_files": 50,
                        "file_types": ["execution", "executor", "trader", "order", "trade"]
                    },
                    "order": {
                        "name": "订单管理",
                        "expected_files": 40,
                        "file_types": ["order", "management", "manager", "handler", "processor"]
                    },
                    "position": {
                        "name": "仓位管理",
                        "expected_files": 30,
                        "file_types": ["position", "portfolio", "inventory", "holding", "balance"]
                    },
                    "account": {
                        "name": "账户管理",
                        "expected_files": 30,
                        "file_types": ["account", "balance", "fund", "capital", "margin"]
                    }
                }
            },
            "backtest": {
                "name": "回测分析层",
                "expected_files": 50,
                "min_files": 40,
                "components": {
                    "engine": {
                        "name": "回测引擎",
                        "expected_files": 20,
                        "file_types": ["engine", "backtest", "simulator", "runner", "executor"]
                    },
                    "analysis": {
                        "name": "回测分析",
                        "expected_files": 15,
                        "file_types": ["analysis", "analyzer", "metrics", "statistics", "report"]
                    },
                    "evaluation": {
                        "name": "策略评估",
                        "expected_files": 10,
                        "file_types": ["evaluation", "evaluator", "scorer", "judge", "assessor"]
                    },
                    "optimization": {
                        "name": "参数优化",
                        "expected_files": 5,
                        "file_types": ["optimization", "optimizer", "parameter", "tuning", "search"]
                    }
                }
            },
            "engine": {
                "name": "引擎层",
                "expected_files": 100,
                "min_files": 80,
                "components": {
                    "web": {
                        "name": "Web服务",
                        "expected_files": 40,
                        "file_types": ["web", "api", "http", "server", "endpoint", "route"]
                    },
                    "realtime": {
                        "name": "实时引擎",
                        "expected_files": 30,
                        "file_types": ["realtime", "engine", "live", "stream", "real"]
                    },
                    "optimization": {
                        "name": "性能优化",
                        "expected_files": 20,
                        "file_types": ["optimization", "optimizer", "performance", "speed", "efficiency"]
                    },
                    "monitoring": {
                        "name": "引擎监控",
                        "expected_files": 10,
                        "file_types": ["monitoring", "monitor", "metrics", "health", "status"]
                    }
                }
            },
            "gateway": {
                "name": "API网关层",
                "expected_files": 10,
                "min_files": 8,
                "components": {
                    "api_gateway": {
                        "name": "API网关",
                        "expected_files": 10,
                        "file_types": ["gateway", "api", "proxy", "router", "entry", "access"]
                    }
                }
            }
        }

    def analyze_file_structure(self) -> Dict[str, Any]:
        """分析文件结构"""
        print("🔍 开始深入分析文件结构...")

        analysis_result = {
            "timestamp": datetime.now(),
            "layers": {},
            "summary": {},
            "recommendations": [],
            "missing_components": []
        }

        # 分析每个架构层
        for layer_key, layer_config in self.layer_definitions.items():
            print(f"📋 分析 {layer_config['name']}...")
            layer_analysis = self._analyze_layer_files(layer_key, layer_config)
            analysis_result["layers"][layer_key] = layer_analysis

        # 生成总结报告
        analysis_result["summary"] = self._generate_file_summary(analysis_result)
        analysis_result["recommendations"] = self._generate_file_recommendations(analysis_result)

        print(f"✅ 文件结构分析完成")
        return analysis_result

    def _analyze_layer_files(self, layer_key: str, layer_config: Dict[str, Any]) -> Dict[str, Any]:
        """分析单个层的文件结构"""
        layer_result = {
            "layer_name": layer_config["name"],
            "expected_files": layer_config["expected_files"],
            "actual_files": 0,
            "components": {},
            "missing_files": 0,
            "file_distribution": {},
            "file_types": defaultdict(int),
            "issues": []
        }

        layer_path = self.src_dir / layer_key
        if not layer_path.exists():
            layer_result["issues"].append(f"架构层目录不存在: {layer_path}")
            return layer_result

        # 统计总文件数
        python_files = list(layer_path.rglob("*.py"))
        layer_result["actual_files"] = len(python_files)

        # 分析组件
        for comp_key, comp_config in layer_config["components"].items():
            component_analysis = self._analyze_component_files(
                layer_path / comp_key, comp_config
            )
            layer_result["components"][comp_key] = component_analysis

            # 统计文件类型分布
            for file_type, count in component_analysis["file_types"].items():
                layer_result["file_types"][file_type] += count

        # 检查文件数量是否充足
        if layer_result["actual_files"] < layer_config.get("min_files", 0):
            shortage = layer_config.get("min_files", 0) - layer_result["actual_files"]
            layer_result["issues"].append(f"文件数量严重不足: 缺少 {shortage} 个文件")

        # 生成组件文件分布
        layer_result["file_distribution"] = {
            comp_key: comp_result["actual_files"]
            for comp_key, comp_result in layer_result["components"].items()
        }

        return layer_result

    def _analyze_component_files(self, component_path: Path, component_config: Dict[str, Any]) -> Dict[str, Any]:
        """分析组件的文件结构"""
        component_result = {
            "component_name": component_config["name"],
            "expected_files": component_config["expected_files"],
            "actual_files": 0,
            "file_types": defaultdict(int),
            "missing_files": 0,
            "files": []
        }

        if not component_path.exists():
            component_result["missing_files"] = component_config["expected_files"]
            return component_result

        # 统计文件
        python_files = list(component_path.rglob("*.py"))
        component_result["actual_files"] = len(python_files)
        component_result["files"] = [str(f.relative_to(self.src_dir)) for f in python_files]

        # 分析文件类型分布
        file_type_patterns = component_config["file_types"]
        for py_file in python_files:
            file_name = py_file.stem.lower()

            # 匹配文件类型
            for file_type in file_type_patterns:
                if file_type in file_name:
                    component_result["file_types"][file_type] += 1
                    break
            else:
                # 未匹配的文件类型
                component_result["file_types"]["other"] += 1

        # 计算缺失文件数
        expected_types = len(file_type_patterns)
        actual_types = len([t for t in component_result["file_types"].keys() if t != "other"])
        component_result["missing_files"] = max(0, expected_types - actual_types)

        return component_result

    def _generate_file_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成文件分析总结"""
        summary = {
            "total_expected_files": 0,
            "total_actual_files": 0,
            "total_missing_files": 0,
            "layers_below_minimum": [],
            "most_complete_layers": [],
            "least_complete_layers": [],
            "file_type_distribution": defaultdict(int)
        }

        for layer_key, layer_result in analysis_result["layers"].items():
            layer_config = self.layer_definitions[layer_key]

            summary["total_expected_files"] += layer_config["expected_files"]
            summary["total_actual_files"] += layer_result["actual_files"]

            # 检查是否低于最低要求
            min_files = layer_config.get("min_files", 0)
            if layer_result["actual_files"] < min_files:
                summary["layers_below_minimum"].append({
                    "layer": layer_result["layer_name"],
                    "actual": layer_result["actual_files"],
                    "minimum": min_files,
                    "shortage": min_files - layer_result["actual_files"]
                })

            # 统计文件类型分布
            for file_type, count in layer_result["file_types"].items():
                summary["file_type_distribution"][file_type] += count

        # 按完成度排序
        layer_completions = []
        for layer_key, layer_result in analysis_result["layers"].items():
            layer_config = self.layer_definitions[layer_key]
            completion_rate = layer_result["actual_files"] / layer_config["expected_files"]
            layer_completions.append({
                "layer": layer_result["layer_name"],
                "completion_rate": completion_rate,
                "actual_files": layer_result["actual_files"],
                "expected_files": layer_config["expected_files"]
            })

        layer_completions.sort(key=lambda x: x["completion_rate"], reverse=True)
        summary["most_complete_layers"] = layer_completions[:3]
        summary["least_complete_layers"] = layer_completions[-3:]

        return summary

    def _generate_file_recommendations(self, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成文件结构建议"""
        recommendations = []

        # 基于严重不足的层级生成建议
        for layer_info in analysis_result["summary"]["layers_below_minimum"]:
            recommendations.append({
                "priority": "high",
                "category": f"{layer_info['layer']} 文件补全",
                "description": f"{layer_info['layer']} 文件数量严重不足 ({layer_info['actual']}/{layer_info['minimum']})",
                "recommendation": f"需要补充 {layer_info['shortage']} 个文件",
                "implementation": "根据架构设计文档，创建缺失的核心组件和功能模块"
            })

        # 基于组件缺失生成建议
        for layer_key, layer_result in analysis_result["layers"].items():
            for comp_key, comp_result in layer_result["components"].items():
                if comp_result["missing_files"] > 0:
                    recommendations.append({
                        "priority": "medium",
                        "category": f"{comp_result['component_name']} 组件补全",
                        "description": f"缺少 {comp_result['missing_files']} 个文件类型",
                        "recommendation": f"为 {comp_result['component_name']} 创建标准文件类型",
                        "implementation": f"根据组件需求，创建相应的 {', '.join(comp_result['file_types'].keys())} 类型文件"
                    })

        # 基于文件类型分布生成建议
        for file_type, count in analysis_result["summary"]["file_type_distribution"].items():
            if count == 0:
                recommendations.append({
                    "priority": "low",
                    "category": "文件类型补全",
                    "description": f"缺少 {file_type} 类型文件",
                    "recommendation": "创建相应的功能模块",
                    "implementation": f"根据业务需求，创建 {file_type} 相关的功能实现"
                })

        return recommendations

    def generate_file_analysis_report(self) -> Dict[str, Any]:
        """生成文件分析报告"""
        analysis_result = self.analyze_file_structure()

        # 保存JSON报告
        json_report_path = self.reports_dir / \
            f"architecture_file_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(json_report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2, default=str)

        # 生成HTML报告
        html_report_path = self._generate_file_analysis_html(analysis_result)

        return {
            "success": True,
            "json_report": str(json_report_path),
            "html_report": str(html_report_path),
            "analysis": analysis_result,
            "summary": analysis_result["summary"],
            "recommendations": analysis_result["recommendations"]
        }

    def _generate_file_analysis_html(self, analysis_result: Dict[str, Any]) -> str:
        """生成HTML文件分析报告"""
        html_content = ".2f"","".2f"","f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>架构文件分析报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; border-radius: 8px; padding: 30px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .score {{ font-size: 48px; font-weight: bold; color: #28a745; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }}
        .layer {{ margin-bottom: 20px; border: 1px solid #dee2e6; border-radius: 8px; overflow: hidden; }}
        .layer-header {{ background: #007bff; color: white; padding: 15px; font-weight: bold; }}
        .layer-content {{ padding: 20px; }}
        .component {{ background: #f8f9fa; margin: 10px 0; padding: 15px; border-radius: 5px; border-left: 4px solid #17a2b8; }}
        .missing {{ color: #dc3545; font-weight: bold; }}
        .warning {{ color: #ffc107; }}
        .success {{ color: #28a745; }}
        .chart {{ display: flex; align-items: center; margin: 10px 0; }}
        .chart-bar {{ height: 20px; background: #007bff; border-radius: 3px; margin-right: 10px; }}
        .recommendations {{ background: #d1ecf1; padding: 20px; border-radius: 8px; margin-top: 20px; }}
        .recommendation {{ padding: 15px; margin-bottom: 15px; border-radius: 5px; border-left: 4px solid #17a2b8; }}
        .priority-high {{ border-color: #dc3545; background: #f8d7da; }}
        .priority-medium {{ border-color: #ffc107; background: #fff3cd; }}
        .priority-low {{ border-color: #28a745; background: #d4edda; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 架构文件分析报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <div class="score">{analysis_result['summary']['total_actual_files']}/{analysis_result['summary']['total_expected_files']}</div>
            <p>总体文件完成情况</p>
        </div>

        <div class="summary">
            <div class="card">
                <h3>📈 总览</h3>
                <p>预期文件总数: {analysis_result['summary']['total_expected_files']}</p>
                <p>实际文件总数: {analysis_result['summary']['total_actual_files']}</p>
                <p>完成率: {analysis_result['summary']['total_actual_files'] / analysis_result['summary']['total_expected_files'] * 100:.1f}%</p>
            </div>
            <div class="card">
                <h3>⚠️ 问题统计</h3>
                <p>严重不足层数: {len(analysis_result['summary']['layers_below_minimum'])}</p>
                <p>最完整层级: {len(analysis_result['summary']['most_complete_layers'])}</p>
                <p>最不完整层级: {len(analysis_result['summary']['least_complete_layers'])}</p>
            </div>
            <div class="card">
                <h3>📁 文件类型</h3>
"""

        # 添加文件类型统计
        for file_type, count in sorted(analysis_result["summary"]["file_type_distribution"].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                html_content += f"<p>{file_type}: {count}</p>"

        html_content += ".2f"","f"""
        </div>

        <h2>📋 各层级文件分析</h2>
"""

        # 添加各层级分析
        for layer_key, layer_result in analysis_result["layers"].items():
            layer_config = self.layer_definitions[layer_key]
            completion_rate = layer_result["actual_files"] / layer_config["expected_files"] * 100

            # 设置层级状态颜色
            if completion_rate >= 80:
                status_class = "success"
            elif completion_rate >= 60:
                status_class = "warning"
            else:
                status_class = "missing"

            html_content += ".2f"","f"""
        <div class="layer">
            <div class="layer-header">
                📁 {layer_result['layer_name']} - {completion_rate:.1f}% ({layer_result['actual_files']}/{layer_config['expected_files']})
            </div>
            <div class="layer-content">
                <div class="chart">
                    <div class="chart-bar" style="width: {completion_rate}%;"></div>
                    <span>{completion_rate:.1f}%</span>
                </div>
"""

            # 添加组件分析
            for comp_key, comp_result in layer_result["components"].items():
                comp_completion = comp_result["actual_files"] / \
                    comp_result["expected_files"] * 100 if comp_result["expected_files"] > 0 else 0

                html_content += ".2f"","f"""
                <div class="component">
                    <strong>{comp_result['component_name']}</strong><br>
                    文件数: {comp_result['actual_files']}/{comp_result['expected_files']} ({comp_completion:.1f}%)
                    <div class="chart">
                        <div class="chart-bar" style="width: {comp_completion}%; background: #17a2b8;"></div>
                        <span>{comp_completion:.1f}%</span>
                    </div>
"""

                # 显示文件类型分布
                if comp_result["file_types"]:
                    html_content += "<small>文件类型: "
                    type_info = []
                    for file_type, count in comp_result["file_types"].items():
                        if count > 0:
                            type_info.append(f"{file_type}({count})")
                    html_content += ", ".join(type_info) + "</small>"

                html_content += "</div>"

            html_content += "</div></div>"

        # 添加严重不足层级列表
        if analysis_result["summary"]["layers_below_minimum"]:
            html_content += """
        <div class="issues">
            <h2>⚠️ 严重不足的层级</h2>
"""

            for layer_info in analysis_result["summary"]["layers_below_minimum"]:
                html_content += ".2f"f"""
            <div class="issue warning">
                <strong>{layer_info['layer']}</strong>: 缺少 {layer_info['shortage']} 个文件
                (实际 {layer_info['actual']} / 最低要求 {layer_info['minimum']})
            </div>
"""

            html_content += "</div>"

        # 添加建议
        html_content += """
        <div class="recommendations">
            <h2>💡 改进建议</h2>
"""

        for rec in analysis_result["recommendations"]:
            priority_class = f"priority-{rec['priority']}"
            html_content += ".2f"f"""
            <div class="recommendation {priority_class}">
                <strong>[{rec['priority'].upper()}] {rec['category']}</strong><br>
                <strong>问题:</strong> {rec['description']}<br>
                <strong>建议:</strong> {rec['recommendation']}<br>
                <strong>实施:</strong> {rec['implementation']}
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
            f"architecture_file_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        with open(html_report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(html_report_path)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='架构文件分析器')
    parser.add_argument('--project', default='.', help='项目根目录')
    parser.add_argument('--analyze', action='store_true', help='分析文件结构')
    parser.add_argument('--report', action='store_true', help='生成文件分析报告')

    args = parser.parse_args()

    analyzer = ArchitectureFileAnalyzer(args.project)

    if args.analyze or args.report:
        result = analyzer.generate_file_analysis_report()

        print(f"📊 文件分析完成")
        print(
            f"📈 总体完成率: {result['analysis']['summary']['total_actual_files'] / result['analysis']['summary']['total_expected_files'] * 100:.1f}%")
        print(f"⚠️ 严重不足层数: {len(result['analysis']['summary']['layers_below_minimum'])}")
        print(f"💡 建议数量: {len(result['recommendations'])}")

        if result['analysis']['summary']['layers_below_minimum']:
            print("\n🔍 严重不足的层级:")
            for layer_info in result['analysis']['summary']['layers_below_minimum']:
                print(f"  - {layer_info['layer']}: 缺少 {layer_info['shortage']} 个文件")

    else:
        print("📊 架构文件分析器")
        print("使用 --help 查看可用命令")


if __name__ == "__main__":
    main()

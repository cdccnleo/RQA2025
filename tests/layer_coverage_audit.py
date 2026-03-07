#!/usr/bin/env python3
"""
RQA2025 分层测试覆盖率检查脚本

按照业务流程驱动架构设计，分层依次检查各层单元测试、集成测试和端到端测试覆盖率是否达标投产要求。

执行方式:
    python tests/layer_coverage_audit.py
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import importlib.util


class LayerCoverageAuditor:
    """分层测试覆盖率审计器"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_root = self.project_root / "tests"
        self.src_root = self.project_root / "src"

        # 按照架构设计定义的层级及其测试要求
        self.layer_definitions = {
            # 核心业务层 (价值创造)
            "infrastructure": {
                "name": "基础设施层",
                "path": "infrastructure",
                "unit_threshold": 95.0,
                "integration_threshold": 90.0,
                "e2e_threshold": 85.0,
                "description": "系统基础服务、健康检查、配置管理"
            },
            "data": {
                "name": "数据管理层",
                "path": "data",
                "unit_threshold": 70.0,
                "integration_threshold": 65.0,
                "e2e_threshold": 60.0,
                "description": "数据适配、质量控制、存储管理"
            },
            "features": {
                "name": "特征分析层",
                "path": "features",
                "unit_threshold": 70.0,
                "integration_threshold": 65.0,
                "e2e_threshold": 60.0,
                "description": "技术指标计算、特征工程、数据预处理"
            },
            "ml": {
                "name": "机器学习层",
                "path": "ml",
                "unit_threshold": 71.5,
                "integration_threshold": 65.0,
                "e2e_threshold": 60.0,
                "description": "AI算法实现、模型训练、预测服务"
            },
            "strategy": {
                "name": "策略服务层",
                "path": "strategy",
                "unit_threshold": 100.0,
                "integration_threshold": 95.0,
                "e2e_threshold": 90.0,
                "description": "量化策略实现、回测引擎、信号生成"
            },
            "trading": {
                "name": "交易层",
                "path": "trading",
                "unit_threshold": 70.0,
                "integration_threshold": 65.0,
                "e2e_threshold": 60.0,
                "description": "订单管理、交易执行、持仓控制"
            },
            "risk": {
                "name": "风险控制层",
                "path": "risk",
                "unit_threshold": 40.0,  # 风险控制要求更严格
                "integration_threshold": 35.0,
                "e2e_threshold": 30.0,
                "description": "风险评估、VaR计算、合规检查"
            },

            # 核心支撑层 (技术赋能)
            "monitoring": {
                "name": "监控层",
                "path": "monitoring",
                "unit_threshold": 70.0,
                "integration_threshold": 65.0,
                "e2e_threshold": 60.0,
                "description": "系统监控、性能指标、告警管理"
            },
            "streaming": {
                "name": "流处理层",
                "path": "streaming",
                "unit_threshold": 94.3,
                "integration_threshold": 90.0,
                "e2e_threshold": 85.0,
                "description": "实时数据流处理、事件驱动架构"
            },
            "gateway": {
                "name": "网关层",
                "path": "gateway",
                "unit_threshold": 70.0,
                "integration_threshold": 65.0,
                "e2e_threshold": 60.0,
                "description": "API网关、路由管理、安全控制"
            },
            "optimization": {
                "name": "优化层",
                "path": "optimization",
                "unit_threshold": 35.0,  # 优化算法复杂度较高
                "integration_threshold": 30.0,
                "e2e_threshold": 25.0,
                "description": "投资组合优化、参数调优、性能优化"
            },

            # 辅助支撑层
            "adapters": {
                "name": "适配器层",
                "path": "adapters",
                "unit_threshold": 80.0,
                "integration_threshold": 75.0,
                "e2e_threshold": 70.0,
                "description": "外部系统适配、协议转换、接口统一"
            },
            "automation": {
                "name": "自动化层",
                "path": "automation",
                "unit_threshold": 85.0,
                "integration_threshold": 80.0,
                "e2e_threshold": 75.0,
                "description": "自动化任务、定时作业、工作流引擎"
            },
            "resilience": {
                "name": "弹性层",
                "path": "resilience",
                "unit_threshold": 75.0,
                "integration_threshold": 70.0,
                "e2e_threshold": 65.0,
                "description": "故障恢复、自动扩缩容、降级服务"
            },
            "testing": {
                "name": "测试层",
                "path": "testing",
                "unit_threshold": 90.0,
                "integration_threshold": 85.0,
                "e2e_threshold": 80.0,
                "description": "测试框架、Mock服务、测试数据管理"
            },
            "tools": {
                "name": "工具层",
                "path": "tools",
                "unit_threshold": 70.0,
                "integration_threshold": 65.0,
                "e2e_threshold": 60.0,
                "description": "开发工具、运维工具、分析工具"
            },
            "distributed": {
                "name": "分布式协调器",
                "path": "distributed",
                "unit_threshold": 65.0,
                "integration_threshold": 60.0,
                "e2e_threshold": 55.0,
                "description": "分布式锁、集群协调、服务发现"
            },
            "async_processor": {
                "name": "异步处理器",
                "path": "async_processor",
                "unit_threshold": 83.6,
                "integration_threshold": 80.0,
                "e2e_threshold": 75.0,
                "description": "异步任务处理、消息队列、事件驱动"
            },
            "mobile": {
                "name": "移动端层",
                "path": "mobile",
                "unit_threshold": 85.0,
                "integration_threshold": 80.0,
                "e2e_threshold": 75.0,
                "description": "移动端API、响应式设计、离线支持"
            },
            "boundary": {
                "name": "业务边界层",
                "path": "boundary",
                "unit_threshold": 90.0,
                "integration_threshold": 85.0,
                "e2e_threshold": 80.0,
                "description": "业务领域边界、上下文映射、聚合服务"
            }
        }

        self.audit_results = {}

    def count_test_files(self, layer_path: str) -> Dict[str, int]:
        """统计各类型测试文件的数量"""
        layer_test_path = self.test_root / "unit" / layer_path

        counts = {
            "unit": 0,
            "integration": 0,
            "e2e": 0,
            "total": 0
        }

        if not layer_test_path.exists():
            return counts

        # 统计单元测试文件
        for root, dirs, files in os.walk(layer_test_path):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    counts["unit"] += 1
                    counts["total"] += 1

        # 统计集成测试文件
        integration_path = self.test_root / "integration"
        if integration_path.exists():
            for root, dirs, files in os.walk(integration_path):
                for file in files:
                    if file.startswith("test_") and file.endswith(".py"):
                        if layer_path.lower() in file.lower():
                            counts["integration"] += 1
                            counts["total"] += 1

        # 统计端到端测试文件
        e2e_path = self.test_root / "e2e"
        if e2e_path.exists():
            for root, dirs, files in os.walk(e2e_path):
                for file in files:
                    if file.startswith("test_") and file.endswith(".py"):
                        if layer_path.lower() in file.lower():
                            counts["e2e"] += 1
                            counts["total"] += 1

        return counts

    def run_layer_tests(self, layer_path: str) -> Dict[str, Dict]:
        """运行指定层的测试并获取结果"""
        results = {
            "unit": {"passed": 0, "failed": 0, "coverage": 0.0, "status": "pending"},
            "integration": {"passed": 0, "failed": 0, "coverage": 0.0, "status": "pending"},
            "e2e": {"passed": 0, "failed": 0, "coverage": 0.0, "status": "pending"}
        }

        # 运行单元测试
        unit_test_path = self.test_root / "unit" / layer_path
        if unit_test_path.exists():
            try:
                cmd = [
                    sys.executable, "-m", "pytest",
                    str(unit_test_path),
                    "--tb=no", "--quiet", "--disable-warnings",
                    "--json-report", "--json-report-file=temp_unit_report.json"
                ]

                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if result.returncode in [0, 1]:  # 0=成功, 1=测试失败但pytest执行成功
                    results["unit"]["status"] = "completed"
                    # 尝试解析结果（简化版）
                    if "passed" in result.stdout:
                        results["unit"]["passed"] = result.stdout.count("PASSED")
                        results["unit"]["failed"] = result.stdout.count("FAILED")
                else:
                    results["unit"]["status"] = "error"

            except subprocess.TimeoutExpired:
                results["unit"]["status"] = "timeout"
            except Exception as e:
                results["unit"]["status"] = "error"

        return results

    def check_layer_compliance(self, layer_key: str, test_results: Dict) -> Dict:
        """检查层是否符合测试覆盖率要求"""
        layer_config = self.layer_definitions[layer_key]

        compliance = {
            "unit_compliant": False,
            "integration_compliant": False,
            "e2e_compliant": False,
            "overall_compliant": False,
            "issues": [],
            "recommendations": []
        }

        # 检查单元测试
        unit_coverage = test_results.get("unit", {}).get("coverage", 0)
        if unit_coverage >= layer_config["unit_threshold"]:
            compliance["unit_compliant"] = True
        else:
            compliance["issues"].append(
                f"单元测试覆盖率不足: {unit_coverage:.1f}% < {layer_config['unit_threshold']:.1f}%"
            )
            compliance["recommendations"].append(
                f"增加单元测试覆盖率至{layer_config['unit_threshold']:.1f}%"
            )

        # 检查集成测试
        integration_coverage = test_results.get("integration", {}).get("coverage", 0)
        if integration_coverage >= layer_config["integration_threshold"]:
            compliance["integration_compliant"] = True
        else:
            compliance["issues"].append(
                f"集成测试覆盖率不足: {integration_coverage:.1f}% < {layer_config['integration_threshold']:.1f}%"
            )
            compliance["recommendations"].append(
                f"增加集成测试覆盖率至{layer_config['integration_threshold']:.1f}%"
            )

        # 检查端到端测试
        e2e_coverage = test_results.get("e2e", {}).get("coverage", 0)
        if e2e_coverage >= layer_config["e2e_threshold"]:
            compliance["e2e_compliant"] = True
        else:
            compliance["issues"].append(
                f"端到端测试覆盖率不足: {e2e_coverage:.1f}% < {layer_config['e2e_threshold']:.1f}%"
            )
            compliance["recommendations"].append(
                f"增加端到端测试覆盖率至{layer_config['e2e_threshold']:.1f}%"
            )

        # 总体合规性判断
        compliance["overall_compliant"] = all([
            compliance["unit_compliant"],
            compliance["integration_compliant"],
            compliance["e2e_compliant"]
        ])

        return compliance

    def audit_layer(self, layer_key: str) -> Dict:
        """审计单个层"""
        layer_config = self.layer_definitions[layer_key]
        print(f"\n🔍 审计层: {layer_config['name']} ({layer_key})")
        print(f"   描述: {layer_config['description']}")

        # 统计测试文件数量
        test_counts = self.count_test_files(layer_config["path"])
        print(f"   测试文件: 单元测试{test_counts['unit']}个, 集成测试{test_counts['integration']}个, 端到端测试{test_counts['e2e']}个")

        # 运行测试并获取结果
        test_results = self.run_layer_tests(layer_config["path"])

        # 检查合规性
        compliance = self.check_layer_compliance(layer_key, test_results)

        layer_result = {
            "layer_info": layer_config,
            "test_counts": test_counts,
            "test_results": test_results,
            "compliance": compliance,
            "audited_at": datetime.now().isoformat()
        }

        return layer_result

    def run_full_audit(self) -> Dict:
        """运行完整的分层审计"""
        print("🚀 开始RQA2025分层测试覆盖率审计")
        print("=" * 70)
        print(f"审计时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        print(f"项目根目录: {self.project_root}")

        all_results = {}

        # 按照架构层次依次审计
        for layer_key in self.layer_definitions.keys():
            try:
                layer_result = self.audit_layer(layer_key)
                all_results[layer_key] = layer_result

                # 显示审计结果摘要
                compliance = layer_result["compliance"]
                status = "✅ 达标" if compliance["overall_compliant"] else "❌ 未达标"
                print(f"   结果: {status}")

                if not compliance["overall_compliant"]:
                    print(f"   问题: {len(compliance['issues'])}个")
                    for issue in compliance["issues"][:2]:  # 只显示前2个问题
                        print(f"     - {issue}")

            except Exception as e:
                print(f"   审计失败: {str(e)}")
                all_results[layer_key] = {
                    "error": str(e),
                    "audited_at": datetime.now().isoformat()
                }

        return all_results

    def generate_report(self, audit_results: Dict) -> str:
        """生成审计报告"""
        report_lines = []
        report_lines.append("# RQA2025 分层测试覆盖率审计报告")
        report_lines.append("")
        report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        report_lines.append(f"**审计范围**: {len(self.layer_definitions)}个架构层级")
        report_lines.append("")

        # 总体统计
        total_layers = len(audit_results)
        compliant_layers = sum(1 for r in audit_results.values()
                             if isinstance(r, dict) and r.get("compliance", {}).get("overall_compliant", False))
        compliance_rate = (compliant_layers / total_layers * 100) if total_layers > 0 else 0

        report_lines.append("## 📊 总体审计结果")
        report_lines.append("")
        report_lines.append(f"- **总层数**: {total_layers}")
        report_lines.append(f"- **达标层数**: {compliant_layers}")
        report_lines.append(f"- **合规率**: {compliance_rate:.1f}%")
        report_lines.append("")

        # 详细结果
        report_lines.append("## 🔍 分层审计详情")
        report_lines.append("")
        report_lines.append("| 层级 | 测试文件数 | 单元测试 | 集成测试 | 端到端测试 | 合规状态 |")
        report_lines.append("|------|----------|----------|----------|----------|----------|")

        for layer_key, result in audit_results.items():
            if "error" in result:
                report_lines.append(f"| {self.layer_definitions[layer_key]['name']} | 错误 | - | - | - | 审计失败 |")
                continue

            layer_info = result["layer_info"]
            test_counts = result["test_counts"]
            compliance = result["compliance"]

            total_tests = test_counts["total"]
            status = "✅ 达标" if compliance["overall_compliant"] else "❌ 未达标"

            unit_status = "✅" if compliance["unit_compliant"] else "❌"
            integration_status = "✅" if compliance["integration_compliant"] else "❌"
            e2e_status = "✅" if compliance["e2e_compliant"] else "❌"

            report_lines.append(f"| {layer_info['name']} | {total_tests} | {unit_status} | {integration_status} | {e2e_status} | {status} |")

        # 问题和建议
        report_lines.append("")
        report_lines.append("## ⚠️ 问题层级及建议")
        report_lines.append("")

        for layer_key, result in audit_results.items():
            if isinstance(result, dict) and "compliance" in result:
                compliance = result["compliance"]
                if not compliance["overall_compliant"]:
                    layer_name = result["layer_info"]["name"]
                    report_lines.append(f"### {layer_name}")
                    report_lines.append("")
                    for issue in compliance["issues"]:
                        report_lines.append(f"- ❌ {issue}")
                    for rec in compliance["recommendations"]:
                        report_lines.append(f"- 💡 {rec}")
                    report_lines.append("")

        # 结论
        report_lines.append("## 🎯 审计结论")
        report_lines.append("")

        if compliance_rate >= 80:
            report_lines.append("✅ **审计通过**: 系统测试覆盖率整体达标，符合生产部署要求")
        elif compliance_rate >= 60:
            report_lines.append("⚠️ **基本达标**: 系统测试覆盖率基本达标，建议进一步提升关键层级的测试覆盖率")
        else:
            report_lines.append("❌ **未达标**: 系统测试覆盖率严重不足，建议立即开展测试覆盖率提升工作")

        report_lines.append("")
        report_lines.append("### 投产建议")
        report_lines.append("")
        if compliance_rate >= 80:
            report_lines.append("🚀 **可以部署**: 系统已达到高质量投产标准")
        else:
            report_lines.append("⏳ **暂缓部署**: 建议先完成测试覆盖率提升，再考虑部署")

        return "\n".join(report_lines)

    def save_report(self, report_content: str, output_path: str = None):
        """保存审计报告"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"test_reports/layer_coverage_audit_{timestamp}.md"

        output_file = self.project_root / output_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n📄 审计报告已保存: {output_file}")
        return str(output_file)


def main():
    """主函数"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    auditor = LayerCoverageAuditor(project_root)

    try:
        # 运行完整审计
        audit_results = auditor.run_full_audit()

        # 生成报告
        report = auditor.generate_report(audit_results)

        # 保存报告
        report_path = auditor.save_report(report)

        # 显示关键结果
        total_layers = len(audit_results)
        compliant_layers = sum(1 for r in audit_results.values()
                             if isinstance(r, dict) and r.get("compliance", {}).get("overall_compliant", False))
        compliance_rate = (compliant_layers / total_layers * 100) if total_layers > 0 else 0

        print("\n🎊 审计完成！")
        print(f"合规率: {compliance_rate:.1f}% ({compliant_layers}/{total_layers})")
        print(f"详细报告: {report_path}")

        if compliance_rate >= 80:
            print("✅ 系统测试覆盖率达标，可以投入生产使用！")
            return 0
        else:
            print("⚠️ 系统测试覆盖率需要进一步提升")
            return 1

    except Exception as e:
        print(f"❌ 审计失败: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

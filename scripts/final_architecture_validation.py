#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 最终架构验证报告

基于分层架构重构后的最终验证
"""

import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def get_directory_structure():
    """获取目录结构"""
    src_path = project_root / "src"
    structure = {}

    for item in src_path.iterdir():
        if item.is_dir() and not item.name.startswith('.') and not item.name.startswith('__'):
            structure[item.name] = {
                "type": "package" if (item / "__init__.py").exists() else "directory",
                "path": str(item.relative_to(project_root)),
                "files": []
            }

            # 统计文件数量
            py_files = list(item.glob("**/*.py"))
            structure[item.name]["file_count"] = len(py_files)
            structure[item.name]["has_init"] = (item / "__init__.py").exists()

    return structure


def analyze_layer_completeness():
    """分析各层完整性"""
    layers = {
        "core_services": {
            "name": "核心服务层",
            "expected_packages": ["core"],
            "description": "事件总线、依赖注入、业务流程编排"
        },
        "infrastructure": {
            "name": "基础设施层",
            "expected_packages": ["infrastructure"],
            "description": "配置、缓存、日志、安全、错误处理"
        },
        "data_collection": {
            "name": "数据采集层",
            "expected_packages": ["data"],
            "description": "数据源适配、实时采集、数据验证"
        },
        "api_gateway": {
            "name": "API网关层",
            "expected_packages": ["gateway"],
            "description": "路由转发、认证授权、限流熔断"
        },
        "feature_processing": {
            "name": "特征处理层",
            "expected_packages": ["features"],
            "description": "特征工程、分布式处理、硬件加速"
        },
        "model_inference": {
            "name": "模型推理层",
            "expected_packages": ["ml"],
            "description": "集成学习、模型管理、实时推理"
        },
        "strategy_decision": {
            "name": "策略决策层",
            "expected_packages": ["backtest"],
            "description": "策略生成、策略框架、投资组合管理"
        },
        "risk_compliance": {
            "name": "风控合规层",
            "expected_packages": ["risk"],
            "description": "风控API、中国市场规则、风险控制器"
        },
        "trading_execution": {
            "name": "交易执行层",
            "expected_packages": ["trading"],
            "description": "订单管理、执行引擎、智能路由"
        },
        "monitoring_feedback": {
            "name": "监控反馈层",
            "expected_packages": ["engine"],
            "description": "系统监控、业务监控、性能监控"
        }
    }

    structure = get_directory_structure()

    analysis = {}
    for layer_key, layer_info in layers.items():
        layer_result = {
            "name": layer_info["name"],
            "description": layer_info["description"],
            "expected_packages": layer_info["expected_packages"],
            "found_packages": [],
            "missing_packages": [],
            "status": "unknown"
        }

        for expected_pkg in layer_info["expected_packages"]:
            if expected_pkg in structure:
                pkg_info = structure[expected_pkg]
                layer_result["found_packages"].append({
                    "name": expected_pkg,
                    "file_count": pkg_info["file_count"],
                    "has_init": pkg_info["has_init"],
                    "type": pkg_info["type"]
                })
            else:
                layer_result["missing_packages"].append(expected_pkg)

        # 计算状态
        if layer_result["missing_packages"]:
            layer_result["status"] = "incomplete"
        elif all(pkg["has_init"] for pkg in layer_result["found_packages"]):
            layer_result["status"] = "complete"
        else:
            layer_result["status"] = "partial"

        analysis[layer_key] = layer_result

    return analysis


def check_test_coverage():
    """检查测试覆盖率"""
    tests_path = project_root / "tests"
    if not tests_path.exists():
        return {"status": "missing", "message": "tests目录不存在"}

    test_files = list(tests_path.glob("**/*.py"))
    unit_tests = list(tests_path.glob("**/unit/**/*.py"))
    integration_tests = list(tests_path.glob("**/integration/**/*.py"))

    return {
        "status": "exists",
        "total_test_files": len(test_files),
        "unit_test_files": len(unit_tests),
        "integration_test_files": len(integration_tests),
        "test_structure": {
            "has_unit_tests": len(unit_tests) > 0,
            "has_integration_tests": len(integration_tests) > 0
        }
    }


def generate_architecture_report():
    """生成架构报告"""
    print("🚀 RQA2025 架构验证报告生成")
    print("=" * 60)

    # 获取目录结构
    structure = get_directory_structure()

    # 分析各层完整性
    layer_analysis = analyze_layer_completeness()

    # 检查测试覆盖率
    test_coverage = check_test_coverage()

    # 生成报告
    report = "# RQA2025 最终架构验证报告\n\n"
    report += f"**验证时间**: {datetime.now().isoformat()}\n\n"

    # 总体概述
    report += "## 📊 总体概述\n\n"
    total_packages = len([p for p in structure.values() if p["type"] == "package"])
    total_files = sum(p["file_count"] for p in structure.values())

    report += f"- **总包数**: {total_packages}\n"
    report += f"- **总文件数**: {total_files}\n"
    report += f"- **测试文件数**: {test_coverage.get('total_test_files', 0)}\n\n"

    # 各层状态
    report += "## 🏗️ 各层架构状态\n\n"

    status_counts = {"complete": 0, "partial": 0, "incomplete": 0}

    for layer_key, layer_info in layer_analysis.items():
        status_emoji = {
            "complete": "✅",
            "partial": "⚠️",
            "incomplete": "❌"
        }

        report += f"### {status_emoji[layer_info['status']]} {layer_info['name']}\n\n"
        report += f"**描述**: {layer_info['description']}\n"
        report += f"**状态**: {layer_info['status'].upper()}\n"

        if layer_info['found_packages']:
            report += "**发现包**:\n"
            for pkg in layer_info['found_packages']:
                init_status = "✅" if pkg['has_init'] else "❌"
                report += f"- {pkg['name']}: {pkg['file_count']} 个文件, __init__.py {init_status}\n"

        if layer_info['missing_packages']:
            report += "**缺失包**:\n"
            for pkg in layer_info['missing_packages']:
                report += f"- {pkg}\n"

        report += "\n"
        status_counts[layer_info['status']] += 1

    # 统计信息
    report += "## 📈 统计信息\n\n"
    total_layers = len(layer_analysis)
    report += f"- **总层数**: {total_layers}\n"
    report += f"- **完整层数**: {status_counts['complete']}\n"
    report += f"- **部分层数**: {status_counts['partial']}\n"
    report += f"- **不完整层数**: {status_counts['incomplete']}\n"

    if total_layers > 0:
        completeness_rate = status_counts['complete'] / total_layers
        report += ".1f"
    # 测试覆盖情况
    report += "## 🧪 测试覆盖情况\n\n"
    if test_coverage['status'] == 'exists':
        report += f"- **测试文件总数**: {test_coverage['total_test_files']}\n"
        report += f"- **单元测试文件**: {test_coverage['unit_test_files']}\n"
        report += f"- **集成测试文件**: {test_coverage['integration_test_files']}\n"

        if test_coverage['test_structure']['has_unit_tests']:
            report += "- **单元测试**: ✅ 存在\n"
        else:
            report += "- **单元测试**: ❌ 不存在\n"

        if test_coverage['test_structure']['has_integration_tests']:
            report += "- **集成测试**: ✅ 存在\n"
        else:
            report += "- **集成测试**: ❌ 不存在\n"
    else:
        report += f"- **测试状态**: {test_coverage['message']}\n"

    # 结论和建议
    report += "## 🎯 结论和建议\n\n"

    if status_counts['complete'] == total_layers:
        report += "### ✅ 架构验证通过\n\n"
        report += "所有架构层级都已完整实现，系统架构稳定可靠。\n\n"
    else:
        report += "### ⚠️ 架构验证结果\n\n"
        report += f"当前架构完整性: {completeness_rate:.1%}\n\n"

        if status_counts['incomplete'] > 0:
            report += "**需要重点关注的层级**:\n"
            for layer_key, layer_info in layer_analysis.items():
                if layer_info['status'] == 'incomplete':
                    report += f"- {layer_info['name']}: 缺失包 {', '.join(layer_info['missing_packages'])}\n"

        if status_counts['partial'] > 0:
            report += "**需要完善__init__.py的层级**:\n"
            for layer_key, layer_info in layer_analysis.items():
                if layer_info['status'] == 'partial':
                    incomplete_pkgs = [pkg['name']
                                       for pkg in layer_info['found_packages'] if not pkg['has_init']]
                    report += f"- {layer_info['name']}: {', '.join(incomplete_pkgs)}\n"

    report += f"\n---\n\n**验证完成时间**: {datetime.now().isoformat()}\n"
    report += "**验证脚本**: scripts/final_architecture_validation.py\n"

    return report


def save_report(report_content):
    """保存报告"""
    output_path = project_root / "reports" / "FINAL_ARCHITECTURE_VALIDATION_REPORT.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)

    print(f"📄 报告已保存到: {output_path}")


def main():
    """主函数"""
    try:
        report = generate_architecture_report()
        save_report(report)

        print("\n" + "=" * 60)
        print("🎉 架构验证报告生成完成!")
        print("📄 详细报告请查看: reports/FINAL_ARCHITECTURE_VALIDATION_REPORT.md")

        return 0

    except Exception as e:
        print(f"❌ 报告生成过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

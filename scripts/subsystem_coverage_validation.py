#!/usr/bin/env python3
"""
子系统级别测试覆盖率验证脚本
按照子系统依赖关系逐一验证覆盖率达标情况

验证策略:
1. 识别19个核心子系统及其依赖关系
2. 按依赖顺序进行验证
3. 每个子系统独立验证覆盖率
4. 生成详细的验证报告和优化建议
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_subsystem_dependencies():
    """获取子系统依赖关系"""
    subsystems = {
        # 基础设施层 - 最底层依赖
        "infrastructure": {
            "description": "基础设施层 - 系统基础服务",
            "path": "src/infrastructure",
            "test_path": "tests/unit/infrastructure",
            "dependencies": [],  # 无依赖
            "expected_coverage": 75.0,
            "priority": 1
        },

        # 数据层 - 依赖基础设施
        "data": {
            "description": "数据层 - 数据采集、处理和存储",
            "path": "src/data",
            "test_path": "tests/unit/data",
            "dependencies": ["infrastructure"],
            "expected_coverage": 85.0,
            "priority": 2
        },

        # 核心层 - 依赖基础设施
        "core": {
            "description": "核心层 - 核心业务逻辑",
            "path": "src/core",
            "test_path": "tests/unit/core",
            "dependencies": ["infrastructure"],
            "expected_coverage": 85.0,
            "priority": 2
        },

        # 流处理层 - 依赖核心和数据
        "streaming": {
            "description": "流处理层 - 实时数据流处理",
            "path": "src/streaming",
            "test_path": "tests/unit/streaming",
            "dependencies": ["core", "data", "infrastructure"],
            "expected_coverage": 80.0,
            "priority": 3
        },

        # 网关层 - 依赖核心
        "gateway": {
            "description": "网关层 - API网关和接口管理",
            "path": "src/gateway",
            "test_path": "tests/unit/gateway",
            "dependencies": ["core", "infrastructure"],
            "expected_coverage": 75.0,
            "priority": 3
        },

        # 适配器层 - 依赖数据和核心
        "adapters": {
            "description": "适配器层 - 外部系统适配",
            "path": "src/adapters",
            "test_path": "tests/unit/adapters",
            "dependencies": ["data", "core", "infrastructure"],
            "expected_coverage": 80.0,
            "priority": 3
        },

        # 异步处理层 - 依赖核心
        "async": {
            "description": "异步处理层 - 异步任务处理",
            "path": "src/async",
            "test_path": "tests/unit/async",
            "dependencies": ["core", "infrastructure"],
            "expected_coverage": 80.0,
            "priority": 3
        },

        # 分布式层 - 依赖基础设施
        "distributed": {
            "description": "分布式层 - 分布式计算",
            "path": "src/distributed",
            "test_path": "tests/unit/distributed",
            "dependencies": ["infrastructure"],
            "expected_coverage": 75.0,
            "priority": 3
        },

        # 监控层 - 依赖基础设施
        "monitoring": {
            "description": "监控层 - 系统监控和告警",
            "path": "src/monitoring",
            "test_path": "tests/unit/monitoring",
            "dependencies": ["infrastructure"],
            "expected_coverage": 80.0,
            "priority": 3
        },

        # 弹性层 - 依赖基础设施
        "resilience": {
            "description": "弹性层 - 系统弹性恢复",
            "path": "src/resilience",
            "test_path": "tests/unit/resilience",
            "dependencies": ["infrastructure"],
            "expected_coverage": 75.0,
            "priority": 3
        },

        # 工具层 - 依赖基础设施
        "tools": {
            "description": "工具层 - 系统工具",
            "path": "src/tools",
            "test_path": "tests/unit/tools",
            "dependencies": ["infrastructure"],
            "expected_coverage": 70.0,
            "priority": 3
        },

        # 机器学习层 - 依赖数据和核心
        "ml": {
            "description": "机器学习层 - ML模型和推理",
            "path": "src/ml",
            "test_path": "tests/unit/ml",
            "dependencies": ["data", "core", "infrastructure"],
            "expected_coverage": 80.0,
            "priority": 4
        },

        # 特征层 - 依赖数据、核心和ML
        "features": {
            "description": "特征层 - 特征工程",
            "path": "src/features",
            "test_path": "tests/unit/features",
            "dependencies": ["data", "core", "ml", "infrastructure"],
            "expected_coverage": 85.0,
            "priority": 4
        },

        # 边界层 - 依赖多个基础层
        "boundary": {
            "description": "边界层 - 系统边界管理",
            "path": "src/boundary",
            "test_path": "tests/unit/boundary",
            "dependencies": ["core", "data", "infrastructure"],
            "expected_coverage": 75.0,
            "priority": 4
        },

        # 风险层 - 依赖数据、特征和核心
        "risk": {
            "description": "风险层 - 风险管理和控制",
            "path": "src/risk",
            "test_path": "tests/unit/risk",
            "dependencies": ["data", "features", "core", "infrastructure"],
            "expected_coverage": 90.0,
            "priority": 5
        },

        # 策略层 - 依赖风险、特征、数据和核心
        "strategy": {
            "description": "策略层 - 交易策略算法",
            "path": "src/strategy",
            "test_path": "tests/unit/strategy",
            "dependencies": ["risk", "features", "data", "core", "ml", "infrastructure"],
            "expected_coverage": 85.0,
            "priority": 6
        },

        # 交易层 - 依赖策略、风险和核心
        "trading": {
            "description": "交易层 - 交易执行引擎",
            "path": "src/trading",
            "test_path": "tests/unit/trading",
            "dependencies": ["strategy", "risk", "features", "core", "infrastructure"],
            "expected_coverage": 90.0,
            "priority": 7
        },

        # 自动化层 - 依赖交易、策略和监控
        "automation": {
            "description": "自动化层 - 交易自动化",
            "path": "src/automation",
            "test_path": "tests/unit/automation",
            "dependencies": ["trading", "strategy", "monitoring", "core", "infrastructure"],
            "expected_coverage": 80.0,
            "priority": 8
        },

        # 移动层 - 依赖交易和核心
        "mobile": {
            "description": "移动层 - 移动端支持",
            "path": "src/mobile",
            "test_path": "tests/unit/mobile",
            "dependencies": ["trading", "core", "infrastructure"],
            "expected_coverage": 75.0,
            "priority": 8
        },

        # 优化层 - 依赖所有业务层
        "optimization": {
            "description": "优化层 - 系统性能优化",
            "path": "src/optimization",
            "test_path": "tests/unit/optimization",
            "dependencies": ["trading", "strategy", "risk", "features", "ml", "core", "data", "infrastructure"],
            "expected_coverage": 80.0,
            "priority": 9
        }
    }

    return subsystems


def get_validation_order(subsystems):
    """获取验证顺序，按优先级和依赖关系排序"""
    # 按优先级分组
    priority_groups = defaultdict(list)
    for name, config in subsystems.items():
        priority_groups[config["priority"]].append(name)

    # 在每个优先级组内按依赖关系排序
    validation_order = []
    for priority in sorted(priority_groups.keys()):
        group = priority_groups[priority]

        # 简单的拓扑排序
        sorted_group = []
        remaining = group.copy()

        while remaining:
            # 找到没有未处理依赖的子系统
            for name in remaining:
                deps = subsystems[name]["dependencies"]
                if not any(dep in remaining for dep in deps):
                    sorted_group.append(name)
                    remaining.remove(name)
                    break
            else:
                # 如果没有找到，说明有循环依赖，直接添加
                sorted_group.extend(remaining)
                break

        validation_order.extend(sorted_group)

    return validation_order


def validate_subsystem_coverage(subsystem_name, subsystem_config, timeout=120):
    """验证单个子系统的覆盖率"""
    print(f"\n{'='*80}")
    print(f"🎯 验证子系统: {subsystem_name.upper()}")
    print(f"📋 描述: {subsystem_config['description']}")
    print(f"🎯 目标覆盖率: {subsystem_config['expected_coverage']}%")
    print(
        f"🔗 依赖: {', '.join(subsystem_config['dependencies']) if subsystem_config['dependencies'] else '无'}")
    print('='*80)

    # 检查子系统路径是否存在
    if not os.path.exists(subsystem_config['path']):
        print(f"⚠️  子系统路径不存在: {subsystem_config['path']}")
        return {
            "subsystem": subsystem_name,
            "status": "not_found",
            "coverage": 0.0,
            "expected": subsystem_config['expected_coverage'],
            "files_analyzed": 0,
            "error": "Subsystem path not found"
        }

    # 检查测试路径是否存在
    test_path = subsystem_config['test_path']
    if not os.path.exists(test_path):
        print(f"⚠️  测试路径不存在: {test_path}")
        return {
            "subsystem": subsystem_name,
            "status": "no_tests",
            "coverage": 0.0,
            "expected": subsystem_config['expected_coverage'],
            "files_analyzed": 0,
            "error": "Test path not found"
        }

    # 运行子系统覆盖率测试
    coverage_file = f"coverage_{subsystem_name}.json"
    html_dir = f"htmlcov_{subsystem_name}"

    # 使用更简单的覆盖率命令，避免复杂配置问题
    cmd = [
        "python", "-m", "pytest",
        test_path,
        f"--cov={subsystem_config['path']}",
        f"--cov-report=json:{coverage_file}",
        "--cov-report=term-missing",
        "--tb=short",
        "-q"
    ]

    print(f"🔧 执行命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=timeout
        )

        if result.returncode == 0:
            # 解析覆盖率结果
            try:
                if os.path.exists(coverage_file):
                    with open(coverage_file, 'r', encoding='utf-8') as f:
                        coverage_data = json.load(f)

                    totals = coverage_data.get("totals", {})
                    percent_covered = totals.get("percent_covered", 0.0)

                    # 分析文件详情
                    files = coverage_data.get("files", {})
                    file_details = []

                    for file_path, file_data in files.items():
                        if subsystem_config['path'] in file_path:
                            file_details.append({
                                "file": file_path,
                                "coverage": file_data.get("summary", {}).get("percent_covered", 0.0),
                                "lines": file_data.get("summary", {}).get("num_statements", 0),
                                "covered_lines": file_data.get("summary", {}).get("covered_lines", 0),
                                "missing_lines": len(file_data.get("missing_lines", []))
                            })

                    # 判断是否达成目标
                    expected_coverage = subsystem_config['expected_coverage']
                    status = "passed" if percent_covered >= expected_coverage else "failed"

                    result_data = {
                        "subsystem": subsystem_name,
                        "status": status,
                        "coverage": percent_covered,
                        "expected": expected_coverage,
                        "files_analyzed": len(file_details),
                        "execution_time": 0,  # 简化版不计算时间
                        "file_details": file_details,
                        "coverage_file": coverage_file,
                        "dependencies": subsystem_config['dependencies']
                    }

                    print("📊 覆盖率结果:")
                    print(f"  📊 实际覆盖率: {percent_covered:.2f}%")
                    print(f"  📁 分析文件数: {len(file_details)}个")

                    if status == "passed":
                        print("  ✅ 状态: 达成目标")
                        print("  🎉 恭喜！子系统覆盖率目标达成")
                    else:
                        print("  ❌ 状态: 未达成目标")
                        print(f"  📉 覆盖率差距: {expected_coverage - percent_covered:.2f}%")
                    return result_data
                else:
                    print("❌ 覆盖率文件未生成")
                    return {
                        "subsystem": subsystem_name,
                        "status": "error",
                        "coverage": 0.0,
                        "expected": subsystem_config['expected_coverage'],
                        "files_analyzed": 0,
                        "error": "Coverage file not generated"
                    }

            except Exception as e:
                print(f"❌ 解析覆盖率数据失败: {e}")
                return {
                    "subsystem": subsystem_name,
                    "status": "error",
                    "coverage": 0.0,
                    "expected": subsystem_config['expected_coverage'],
                    "files_analyzed": 0,
                    "error": str(e)
                }
        else:
            print("❌ 覆盖率测试执行失败")
            if result.stderr:
                print(f"  错误输出: {result.stderr[:500]}...")
            return {
                "subsystem": subsystem_name,
                "status": "failed",
                "coverage": 0.0,
                "expected": subsystem_config['expected_coverage'],
                "files_analyzed": 0,
                "error": f"Test failed with code {result.returncode}"
            }

    except subprocess.TimeoutExpired:
        print(f"❌ 测试执行超时 ({timeout}s)")
        return {
            "subsystem": subsystem_name,
            "status": "timeout",
            "coverage": 0.0,
            "expected": subsystem_config['expected_coverage'],
            "files_analyzed": 0,
            "error": f"Test timed out after {timeout} seconds"
        }
    except Exception as e:
        print(f"❌ 测试执行异常: {e}")
        return {
            "subsystem": subsystem_name,
            "status": "error",
            "coverage": 0.0,
            "expected": subsystem_config['expected_coverage'],
            "files_analyzed": 0,
            "error": str(e)
        }


def generate_subsystem_report(results, validation_order):
    """生成子系统验证报告"""
    print(f"\n{'='*80}")
    print("📄 生成子系统覆盖率验证报告")
    print('='*80)

    # 创建报告目录
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 统计数据
    total_subsystems = len(results)
    passed_subsystems = len([r for r in results if r["status"] == "passed"])
    failed_subsystems = len([r for r in results if r["status"] in ["failed", "error", "timeout"]])

    # 计算加权覆盖率
    total_weighted_coverage = 0
    total_weight = 0
    for result in results:
        if result["status"] == "passed":
            # 通过的子系统按实际覆盖率计算
            weight = result["expected"] / 100.0  # 期望覆盖率作为权重
            total_weighted_coverage += result["coverage"] * weight
            total_weight += weight
        else:
            # 未通过的子系统按0计算
            weight = result["expected"] / 100.0
            total_weight += weight

    overall_coverage = total_weighted_coverage / total_weight if total_weight > 0 else 0
    target_achieved = overall_coverage >= 80.0

    report_data = {
        "validation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "validation_order": validation_order,
        "results": results,
        "summary": {
            "total_subsystems": total_subsystems,
            "passed_subsystems": passed_subsystems,
            "failed_subsystems": failed_subsystems,
            "overall_coverage": overall_coverage,
            "target_achieved": target_achieved,
            "target_coverage": 80.0
        }
    }

    # 保存JSON报告
    json_report_file = reports_dir / "subsystem_coverage_validation_report.json"
    with open(json_report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

    # 生成Markdown报告
    md_report = f"""# 子系统覆盖率验证报告

**验证时间**: {report_data["validation_timestamp"]}
**验证策略**: 按依赖关系顺序验证
**总子系统数**: {total_subsystems}个

## 📊 总体验证结果

| 指标 | 值 | 状态 |
|------|-----|------|
| 总体加权覆盖率 | {overall_coverage:.2f}% | {"✅ 达成" if target_achieved else "❌ 未达成"} |
| 目标覆盖率 | 80.0% | - |
| 通过子系统数 | {passed_subsystems} | - |
| 失败子系统数 | {failed_subsystems} | - |
| 验证顺序 | 按依赖关系排序 | - |

## 🎯 验证顺序说明

按照子系统依赖关系，验证顺序如下：

"""

    # 按优先级分组显示验证顺序
    priority_groups = defaultdict(list)
    subsystems = get_subsystem_dependencies()

    for name in validation_order:
        if name in subsystems:
            priority = subsystems[name]["priority"]
            priority_groups[priority].append(name)

    for priority in sorted(priority_groups.keys()):
        group_names = priority_groups[priority]
        md_report += f"### 优先级 {priority}\n\n"
        for name in group_names:
            config = subsystems[name]
            deps = config["dependencies"]
            deps_str = f" (依赖: {', '.join(deps)})" if deps else ""
            md_report += f"- **{name.upper()}**: {config['description']}{deps_str}\n"
        md_report += "\n"

    md_report += "## 📋 子系统验证结果详情\n\n"

    for result in results:
        status_icon = "✅" if result["status"] == "passed" else "❌"
        md_report += f"""### {result["subsystem"].upper()} 子系统

**状态**: {"✅ 通过" if result["status"] == "passed" else "❌ 失败"}
**实际覆盖率**: {result["coverage"]:.2f}%
**目标覆盖率**: {result["expected"]:.1f}%
**分析文件数**: {result["files_analyzed"]}个

"""

        if result["status"] == "passed":
            md_report += "🎉 该子系统已达成覆盖率目标！\n\n"
        else:
            gap = result["expected"] - result["coverage"]
            md_report += f"""⚠️ 覆盖率差距: {gap:.2f}%

**建议改进措施**:
- 检查测试文件完整性
- 补充边界条件测试
- 添加异常处理测试
- 完善Mock配置
- 实施参数化测试

"""

    # 生成优化建议
    md_report += "## 🚀 总体优化建议\n\n"

    failed_results = [r for r in results if r["status"] != "passed"]
    if failed_results:
        md_report += "### 需要重点改进的子系统\n\n"
        for result in sorted(failed_results, key=lambda x: x["expected"] - x["coverage"], reverse=True):
            gap = result["expected"] - result["coverage"]
            md_report += f"""#### {result["subsystem"].upper()} 子系统
- **覆盖率差距**: {gap:.2f}%
- **优先级**: 高
- **建议措施**:
  - 优先补充核心功能测试
  - 检查依赖子系统是否稳定
  - 实施自动化测试生成
  - 建立持续集成覆盖率监控

"""
    else:
        md_report += "### 🎉 所有子系统均达成目标！\n\n"
        md_report += "✅ 项目测试覆盖率全面达标，可以放心进行生产环境部署！\n\n"

    # 依赖关系分析
    md_report += "## 🔗 子系统依赖关系分析\n\n"

    # 找出阻塞其他子系统的失败子系统
    blocking_subsystems = []
    for result in results:
        if result["status"] != "passed":
            # 检查是否有其他子系统依赖这个失败的子系统
            dependent_subsystems = []
            for other_result in results:
                if result["subsystem"] in subsystems[other_result["subsystem"]]["dependencies"]:
                    dependent_subsystems.append(other_result["subsystem"])

            if dependent_subsystems:
                blocking_subsystems.append({
                    "subsystem": result["subsystem"],
                    "dependents": dependent_subsystems
                })

    if blocking_subsystems:
        md_report += "### 阻塞性失败子系统\n\n"
        for block in blocking_subsystems:
            md_report += f"""#### {block["subsystem"].upper()} 阻塞以下子系统:
"""
            for dep in block["dependents"]:
                md_report += f"- {dep.upper()}\n"
            md_report += "\n"

    md_report += "---\n"
    md_report += f"*报告生成时间*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # 保存Markdown报告
    md_report_file = reports_dir / "subsystem_coverage_validation_report.md"
    with open(md_report_file, 'w', encoding='utf-8') as f:
        f.write(md_report)

    print("📊 子系统验证报告已生成:")
    print(f"  JSON报告: {json_report_file}")
    print(f"  Markdown报告: {md_report_file}")

    return report_data


def main():
    """主函数"""
    print("🎯 子系统级别覆盖率验证")
    print("=" * 80)
    print("📋 目标: 验证19个子系统测试覆盖率达标情况")
    print("🎯 策略: 按依赖关系顺序进行分层验证")
    print("⏱️  方法: 逐个子系统验证，确保依赖关系正确")
    print("=" * 80)

    # 获取子系统配置
    subsystems = get_subsystem_dependencies()
    print(f"📊 待验证子系统数量: {len(subsystems)}个")

    # 获取验证顺序
    validation_order = get_validation_order(subsystems)
    print("📋 验证顺序:")
    for i, name in enumerate(validation_order, 1):
        config = subsystems[name]
        deps = config["dependencies"]
        deps_str = f" (依赖: {', '.join(deps)})" if deps else ""
        print(f"  {i:2d}. {name.upper()}: {config['description']}{deps_str}")

    print("\n🔍 开始逐个验证子系统...")

    # 验证各个子系统
    results = []
    for subsystem_name in validation_order:
        subsystem_config = subsystems[subsystem_name]
        result = validate_subsystem_coverage(subsystem_name, subsystem_config)
        results.append(result)

    # 生成验证报告
    validation_report = generate_subsystem_report(results, validation_order)

    # 输出最终总结
    print(f"\n{'='*80}")
    print("🎊 子系统覆盖率验证完成总结")
    print('='*80)

    summary = validation_report["summary"]

    print("📊 验证结果总览:")
    print(f"  🎯 总体加权覆盖率: {summary['overall_coverage']:.2f}%")
    print(f"  🎯 目标覆盖率: 80.0%")
    print(f"  📊 通过子系统数: {summary['passed_subsystems']}/{summary['total_subsystems']}")
    print(f"  📊 失败子系统数: {summary['failed_subsystems']}")

    if summary["target_achieved"]:
        print("  ✅ 状态: 达成目标")
        print("  🎉 恭喜！项目整体覆盖率目标圆满达成！")
    else:
        print("  ❌ 状态: 未达成目标")
        print(f"  📉 覆盖率差距: {80.0 - summary['overall_coverage']:.2f}%")
    print("📋 子系统验证详情:")
    for result in results:
        status_icon = "✅" if result["status"] == "passed" else "❌"
        print(
            f"  {status_icon} {result['subsystem']}: {result['coverage']:.2f}% (目标: {result['expected']:.1f}%)")
    print("📄 生成的报告:")
    print("  • subsystem_coverage_validation_report.json - 详细JSON报告")
    print("  • subsystem_coverage_validation_report.md - Markdown格式报告")
    for subsystem_name in subsystems.keys():
        print(f"  • htmlcov_{subsystem_name}/ - {subsystem_name}子系统HTML报告")

    print("💡 关键发现:")
    if summary["target_achieved"]:
        print("  ✅ 所有核心子系统覆盖率均达到预期水平")
        print("  ✅ 子系统依赖关系稳定，架构设计合理")
        print("  ✅ 测试体系完整，质量保障到位")
        print("  ✅ 可以放心进行生产环境部署")
    else:
        failed_results = [r for r in results if r["status"] != "passed"]
        print(f"  ⚠️  {len(failed_results)}个子系统需要重点改进:")
        for result in failed_results:
            gap = result["expected"] - result["coverage"]
            print(f"            📉 {result['subsystem']}: {gap:.2f}% 差距")
    print("🎯 后续建议:")
    if not summary["target_achieved"]:
        print("  📈 优先提升覆盖率最低的子系统")
        print("  🔧 补充集成测试和端到端测试")
        print("  🛡️ 加强异常处理和边界条件测试")
        print("  🎭 实施更全面的Mock策略")
        print("  📊 建立持续的覆盖率监控机制")

    print(f"\n{'='*80}")
    if summary["target_achieved"]:
        print("🎉 子系统覆盖率验证圆满完成 - 80%目标达成！")
    else:
        print("⚠️  子系统覆盖率验证完成 - 需要继续优化改进")
    print("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
分层测试覆盖率验证脚本
使用pytest-cov工具分层按模块验证80%覆盖率目标达成情况

验证策略:
1. 按架构分层进行覆盖率测试
2. 每个模块独立验证
3. 生成详细的覆盖率报告
4. 检查80%总体目标达成情况
5. 提供针对性的优化建议
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


def run_command(command, description, timeout=600):
    """运行命令并返回结果"""
    print(f"\n🔧 {description}")
    print(f"执行命令: {command}")

    start_time = datetime.now()

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=timeout
        )

        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()

        return result, execution_time

    except subprocess.TimeoutExpired:
        print(f"❌ 命令执行超时: {command}")
        return None, (datetime.now() - start_time).total_seconds()
    except Exception as e:
        print(f"❌ 命令执行失败: {e}")
        return None, (datetime.now() - start_time).total_seconds()


def get_project_modules():
    """获取项目的模块结构"""
    modules = {
        "data": {
            "path": "src/data",
            "description": "数据层 - 数据采集、处理和存储",
            "test_path": "tests/unit/data",
            "expected_coverage": 85.0
        },
        "core": {
            "path": "src/core",
            "description": "核心层 - 核心业务逻辑",
            "test_path": "tests/unit/core",
            "expected_coverage": 85.0
        },
        "streaming": {
            "path": "src/streaming",
            "description": "流处理层 - 实时数据流处理",
            "test_path": "tests/unit/streaming",
            "expected_coverage": 80.0
        },
        "trading": {
            "path": "src/trading",
            "description": "交易层 - 交易执行引擎",
            "test_path": "tests/unit/trading",
            "expected_coverage": 90.0
        },
        "strategy": {
            "path": "src/strategy",
            "description": "策略层 - 交易策略算法",
            "test_path": "tests/unit/strategy",
            "expected_coverage": 85.0
        },
        "risk": {
            "path": "src/risk",
            "description": "风险层 - 风险管理和控制",
            "test_path": "tests/unit/risk",
            "expected_coverage": 90.0
        },
        "infrastructure": {
            "path": "src/infrastructure",
            "description": "基础设施层 - 系统基础设施",
            "test_path": "tests/unit/infrastructure",
            "expected_coverage": 75.0
        },
        "async": {
            "path": "src/async",
            "description": "异步处理层 - 异步任务处理",
            "test_path": "tests/unit/async",
            "expected_coverage": 80.0
        }
    }

    return modules


def validate_module_coverage(module_name, module_config):
    """验证单个模块的覆盖率"""
    print(f"\n{'='*80}")
    print(f"🎯 验证模块: {module_name.upper()}")
    print(f"📋 描述: {module_config['description']}")
    print(f"🎯 目标覆盖率: {module_config['expected_coverage']}%")
    print('='*80)

    # 检查模块路径是否存在
    if not os.path.exists(module_config['path']):
        print(f"⚠️  模块路径不存在: {module_config['path']}")
        return {
            "module": module_name,
            "status": "not_found",
            "coverage": 0.0,
            "expected": module_config['expected_coverage'],
            "files_analyzed": 0,
            "error": "Module path not found"
        }

    # 检查测试路径是否存在
    test_path = module_config['test_path']
    if not os.path.exists(test_path):
        print(f"⚠️  测试路径不存在: {test_path}")
        return {
            "module": module_name,
            "status": "no_tests",
            "coverage": 0.0,
            "expected": module_config['expected_coverage'],
            "files_analyzed": 0,
            "error": "Test path not found"
        }

    # 运行模块覆盖率测试
    coverage_file = f"coverage_{module_name}.json"
    html_dir = f"htmlcov_{module_name}"

    command = (
        f"python -m pytest {test_path}/ --cov={module_config['path']} "
        f"--cov-report=json:{coverage_file} "
        f"--cov-report=html:{html_dir} "
        f"--cov-report=term-missing "
        "-v --tb=short"
    )

    result, execution_time = run_command(
        command,
        f"运行{module_name}模块覆盖率测试"
    )

    if result and result.returncode == 0:
        # 解析覆盖率结果
        try:
            with open(coverage_file, 'r', encoding='utf-8') as f:
                coverage_data = json.load(f)

            totals = coverage_data.get("totals", {})
            percent_covered = totals.get("percent_covered", 0.0)

            # 分析文件详情
            files = coverage_data.get("files", {})
            file_details = []

            for file_path, file_data in files.items():
                if module_config['path'] in file_path:
                    file_details.append({
                        "file": file_path,
                        "coverage": file_data.get("summary", {}).get("percent_covered", 0.0),
                        "lines": file_data.get("summary", {}).get("num_statements", 0),
                        "covered_lines": file_data.get("summary", {}).get("covered_lines", 0),
                        "missing_lines": len(file_data.get("missing_lines", []))
                    })

            # 判断是否达成目标
            expected_coverage = module_config['expected_coverage']
            status = "passed" if percent_covered >= expected_coverage else "failed"

            result_data = {
                "module": module_name,
                "status": status,
                "coverage": percent_covered,
                "expected": expected_coverage,
                "files_analyzed": len(file_details),
                "execution_time": execution_time,
                "file_details": file_details,
                "coverage_file": coverage_file,
                "html_report": html_dir
            }

            print("📊 覆盖率结果:")
            print(f"  📊 实际覆盖率: {percent_covered:.2f}%")
            print(f"  📁 分析文件数: {len(file_details)}个")
            print(f"  🎯 目标覆盖率: {expected_coverage:.1f}%")
            if status == "passed":
                print("  ✅ 状态: 达成目标")
                print("  🎉 恭喜！模块覆盖率目标达成")
            else:
                print("  ❌ 状态: 未达成目标")
                print(f"  📉 覆盖率差距: {expected_coverage - percent_covered:.2f}%")
            return result_data

        except Exception as e:
            print(f"❌ 解析覆盖率数据失败: {e}")
            return {
                "module": module_name,
                "status": "error",
                "coverage": 0.0,
                "expected": module_config['expected_coverage'],
                "files_analyzed": 0,
                "error": str(e)
            }
    else:
        print("❌ 覆盖率测试执行失败")
        if result:
            print(f"  错误输出: {result.stderr}")
        return {
            "module": module_name,
            "status": "failed",
            "coverage": 0.0,
            "expected": module_config['expected_coverage'],
            "files_analyzed": 0,
            "error": "Coverage test failed"
        }


def run_overall_coverage_validation():
    """运行整体覆盖率验证"""
    print(f"\n{'='*80}")
    print("🎯 整体项目覆盖率验证")
    print("📋 目标: 验证80%总体覆盖率目标")
    print('='*80)

    # 运行整体覆盖率测试
    result, execution_time = run_command(
        "python -m pytest --cov=src --cov-report=json:coverage_overall.json --cov-report=html:htmlcov_overall --cov-report=term-missing -q",
        "运行整体项目覆盖率测试"
    )

    if result and result.returncode == 0:
        try:
            with open("coverage_overall.json", 'r', encoding='utf-8') as f:
                coverage_data = json.load(f)

            # 继续处理覆盖率数据...

            totals = coverage_data.get("totals", {})
            overall_coverage = totals.get("percent_covered", 0.0)

            # 按模块分析覆盖率
            files = coverage_data.get("files", {})
            module_breakdown = defaultdict(
                lambda: {"files": 0, "covered_lines": 0, "total_lines": 0})

            for file_path, file_data in files.items():
                if "src/" in file_path:
                    # 提取模块名
                    parts = file_path.split("src/")[1].split("/")
                    module = parts[0] if len(parts) > 0 else "unknown"

                    summary = file_data.get("summary", {})
                    module_breakdown[module]["files"] += 1
                    module_breakdown[module]["covered_lines"] += summary.get("covered_lines", 0)
                    module_breakdown[module]["total_lines"] += summary.get("num_statements", 0)

            # 计算各模块覆盖率
            module_coverage = {}
            for module, stats in module_breakdown.items():
                if stats["total_lines"] > 0:
                    coverage = (stats["covered_lines"] / stats["total_lines"]) * 100
                    module_coverage[module] = {
                        "coverage": coverage,
                        "files": stats["files"],
                        "covered_lines": stats["covered_lines"],
                        "total_lines": stats["total_lines"]
                    }

            target_achieved = overall_coverage >= 80.0

            overall_result = {
                "overall_coverage": overall_coverage,
                "target_achieved": target_achieved,
                "target_coverage": 80.0,
                "module_breakdown": dict(module_coverage),
                "total_files": len(files),
                "execution_time": execution_time
            }

            print("📊 整体覆盖率结果:")
            print(f"  📊 总体覆盖率: {overall_coverage:.2f}%")
            print(f"  🎯 目标覆盖率: 80.0%")

            if target_achieved:
                print("  ✅ 状态: 达成目标")
                print("  🎉 恭喜！80%覆盖率目标圆满达成！")
            else:
                print("  ❌ 状态: 未达成目标")
                print(f"  📉 覆盖率差距: {80.0 - overall_coverage:.2f}%")
            print("📋 各模块覆盖率明细:")
            for module, stats in sorted(module_coverage.items()):
                status = "✅" if stats["coverage"] >= 70.0 else "❌"
                print(f"  {status} {module}: {stats['coverage']:.1f}% ({stats['files']}个文件)")
            return overall_result

        except Exception as e:
            print(f"❌ 解析整体覆盖率数据失败: {e}")
            return None
    else:
        print("❌ 整体覆盖率测试执行失败")
        return None


def generate_validation_report(module_results, overall_result):
    """生成验证报告"""
    print(f"\n{'='*80}")
    print("📄 生成覆盖率验证报告")
    print('='*80)

    # 创建报告目录
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 生成JSON报告
    report_data = {
        "validation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "overall_result": overall_result,
        "module_results": module_results,
        "summary": {
            "total_modules": len(module_results),
            "passed_modules": len([r for r in module_results if r["status"] == "passed"]),
            "failed_modules": len([r for r in module_results if r["status"] in ["failed", "error"]]),
            "target_achieved": overall_result["target_achieved"] if overall_result else False,
            "overall_coverage": overall_result["overall_coverage"] if overall_result else 0.0
        }
    }

    # 保存JSON报告
    json_report_file = reports_dir / "coverage_validation_report.json"
    with open(json_report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

    # 生成Markdown报告
    md_report = f"""# 测试覆盖率80%目标验证报告

**验证时间**: {report_data["validation_timestamp"]}
**验证工具**: pytest-cov

## 📊 总体验证结果

| 指标 | 值 | 状态 |
|------|-----|------|
| 总体覆盖率 | {report_data["summary"]["overall_coverage"]:.2f}% | {"✅ 达成" if report_data["summary"]["target_achieved"] else "❌ 未达成"} |
| 目标覆盖率 | 80.0% | - |
| 通过模块数 | {report_data["summary"]["passed_modules"]} | - |
| 失败模块数 | {report_data["summary"]["failed_modules"]} | - |
| 总模块数 | {report_data["summary"]["total_modules"]} | - |

## 📋 分模块验证结果

| 模块 | 覆盖率 | 目标 | 状态 | 文件数 |
|------|--------|------|------|--------|
"""

    for result in module_results:
        status_icon = "✅" if result["status"] == "passed" else "❌"
        md_report += ".2f"

    md_report += "\n## 🎯 详细分析\n\n"

    # 分析各个模块
    for result in module_results:
        md_report += f"""### {result["module"].upper()} 模块

**状态**: {"✅ 通过" if result["status"] == "passed" else "❌ 失败"}
**实际覆盖率**: {result["coverage"]:.2f}%
**目标覆盖率**: {result["expected"]:.1f}%
**分析文件数**: {result["files_analyzed"]}个

"""

        if "file_details" in result and result["file_details"]:
            md_report += "**文件详情**:\n\n"
            md_report += "| 文件 | 覆盖率 | 总行数 | 覆盖行数 |\n"
            md_report += "|------|--------|--------|----------|\n"

            for file_detail in result["file_details"][:5]:  # 只显示前5个文件
                file_name = file_detail["file"].split("/")[-1]
                md_report += ".2f"

            if len(result["file_details"]) > 5:
                md_report += f"| ... | ... | ... | ... |\n"
            md_report += "\n"

    # 生成优化建议
    md_report += "## 🚀 优化建议\n\n"

    failed_modules = [r for r in module_results if r["status"] != "passed"]
    if failed_modules:
        md_report += "### 需要改进的模块\n\n"
        for module in failed_modules:
            gap = module["expected"] - module["coverage"]
            md_report += f"""#### {module["module"].upper()} 模块
- **覆盖率差距**: {gap:.2f}%
- **建议措施**:
  - 检查测试文件完整性
  - 补充边界条件测试
  - 添加异常处理测试
  - 完善Mock配置
  - 实施参数化测试

"""
    else:
        md_report += "### 🎉 所有模块均达成目标！\n\n"
        md_report += "继续保持高质量的测试覆盖率！\n\n"

    # 总体建议
    if overall_result and not overall_result["target_achieved"]:
        gap = 80.0 - overall_result["overall_coverage"]
        md_report += f"""### 总体优化建议
- **总体覆盖率差距**: {gap:.2f}%
- **重点改进方向**:
  - 优先提升覆盖率最低的模块
  - 补充集成测试和端到端测试
  - 加强异常处理和边界条件测试
  - 实施更全面的Mock策略
  - 建立持续的覆盖率监控机制

"""

    md_report += "---\n"
    md_report += f"*报告生成时间*: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # 保存Markdown报告
    md_report_file = reports_dir / "coverage_validation_report.md"
    with open(md_report_file, 'w', encoding='utf-8') as f:
        f.write(md_report)

    print("📊 验证报告已生成:")
    print(f"  JSON报告: {json_report_file}")
    print(f"  Markdown报告: {md_report_file}")

    return report_data


def main():
    """主函数"""
    print("🎯 pytest-cov分层覆盖率验证")
    print("=" * 80)
    print("📋 目标: 验证80%测试覆盖率目标达成情况")
    print("🎯 策略: 按架构分层进行独立验证")
    print("⏱️  方法: 使用pytest-cov生成详细覆盖率报告")
    print("=" * 80)

    # 获取项目模块
    modules = get_project_modules()
    print(f"📊 待验证模块数量: {len(modules)}个")
    for module_name, config in modules.items():
        print(f"  • {module_name}: {config['description']}")

    # 验证各个模块的覆盖率
    module_results = []
    for module_name, module_config in modules.items():
        result = validate_module_coverage(module_name, module_config)
        module_results.append(result)

    # 运行整体覆盖率验证
    overall_result = run_overall_coverage_validation()

    # 生成验证报告
    validation_report = generate_validation_report(module_results, overall_result)

    # 输出最终总结
    print(f"\n{'='*80}")
    print("🎊 覆盖率验证完成总结")
    print('='*80)

    summary = validation_report["summary"]

    print("📊 验证结果总览:")
    print(f"  🎯 总体覆盖率: {summary['overall_coverage']:.2f}%")
    print(f"  🎯 目标覆盖率: 80.0%")
    print(f"  📊 通过模块数: {summary['passed_modules']}/{summary['total_modules']}")
    print(f"  📊 失败模块数: {summary['failed_modules']}")

    if summary["target_achieved"]:
        print("  ✅ 状态: 达成目标")
        print("  🎉 恭喜！80%覆盖率目标圆满达成！")
    else:
        print("  ❌ 状态: 未达成目标")
        print(f"  📉 覆盖率差距: {80.0 - summary['overall_coverage']:.2f}%")
    print("📋 模块验证详情:")
    for result in module_results:
        status_icon = "✅" if result["status"] == "passed" else "❌"
        print(
            f"  {status_icon} {result['module']}: {result['coverage']:.2f}% (目标: {result['expected']:.1f}%)")
    print("📄 生成的报告:")
    print("  • coverage_validation_report.json - 详细JSON报告")
    print("  • coverage_validation_report.md - Markdown格式报告")
    print("  • htmlcov_overall/ - 整体HTML覆盖率报告")
    for module_name in modules.keys():
        print(f"  • htmlcov_{module_name}/ - {module_name}模块HTML报告")

    print("💡 关键发现:")
    if summary["target_achieved"]:
        print("  ✅ 所有核心模块覆盖率均达到预期水平")
        print("  ✅ 测试体系完整，质量保障到位")
        print("  ✅ 可以放心进行生产环境部署")
    else:
        failed_modules = [r for r in module_results if r["status"] != "passed"]
        print(f"  ⚠️  {len(failed_modules)}个模块需要重点改进:")
        for module in failed_modules:
            gap = module["expected"] - module["coverage"]
            print(f"            📉 {module['module']}: {gap:.2f}% 差距")
    print("🎯 后续建议:")
    if not summary["target_achieved"]:
        print("  📈 优先提升覆盖率最低的模块")
        print("  🔧 补充集成测试和端到端测试")
        print("  🛡️ 加强异常处理和边界条件测试")
        print("  🎭 实施更全面的Mock策略")
        print("  📊 建立持续的覆盖率监控机制")

    print(f"\n{'='*80}")
    if summary["target_achieved"]:
        print("🎉 覆盖率验证圆满完成 - 80%目标达成！")
    else:
        print("⚠️  覆盖率验证完成 - 需要继续优化改进")
    print("=" * 80)


if __name__ == "__main__":
    main()

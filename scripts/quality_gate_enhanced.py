#!/usr/bin/env python3
"""
增强型质量门禁系统
自动化检查代码质量、测试覆盖率和性能基准
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime


class QualityGate:
    """质量门禁系统"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {}
        self.thresholds = {
            "coverage": {
                "overall": 85.0,
                "infrastructure": 95.0,
                "data": 87.0,
                "features": 75.0,
                "ml": 82.0,
                "strategy": 78.0,
                "trading": 80.0,
                "risk": 76.0,
                "core": 80.0
            },
            "test_success_rate": 95.0,
            "performance_degradation": 20.0,  # 百分比
            "code_quality_score": 80.0,
            "security_vulnerabilities": 0,
            "complexity_score": 10.0  # 圈复杂度阈值
        }

    def run_comprehensive_check(self) -> Dict[str, Any]:
        """运行全面质量检查"""
        print("🔍 开始全面质量门禁检查...")
        print("=" * 60)

        checks = [
            ("coverage_check", "测试覆盖率检查", self._check_coverage),
            ("test_quality_check", "测试质量检查", self._check_test_quality),
            ("performance_check", "性能基准检查", self._check_performance),
            ("code_quality_check", "代码质量检查", self._check_code_quality),
            ("security_check", "安全检查", self._check_security),
            ("complexity_check", "复杂度检查", self._check_complexity)
        ]

        all_passed = True
        summary = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "overall_status": "UNKNOWN",
            "recommendations": []
        }

        for check_id, check_name, check_func in checks:
            print(f"\\n🔍 执行 {check_name}...")
            try:
                result = check_func()
                summary["checks"][check_id] = result
                status = "✅ 通过" if result["passed"] else "❌ 失败"
                print(f"{status} {check_name}")

                if not result["passed"]:
                    all_passed = False
                    summary["recommendations"].extend(result.get("recommendations", []))

            except Exception as e:
                print(f"❌ {check_name} 执行失败: {e}")
                summary["checks"][check_id] = {
                    "passed": False,
                    "error": str(e),
                    "recommendations": [f"修复 {check_name} 执行错误"]
                }
                all_passed = False

        summary["overall_status"] = "PASSED" if all_passed else "FAILED"

        # 生成报告
        self._generate_quality_report(summary)

        print("\\n" + "=" * 60)
        if all_passed:
            print("🎉 所有质量门禁检查通过！")
        else:
            print("⚠️ 部分质量门禁检查失败，请查看详细报告")

        return summary

    def _check_coverage(self) -> Dict[str, Any]:
        """检查测试覆盖率"""
        print("  📊 运行覆盖率测试...")

        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "--cov=src", "--cov-report=json:coverage.json",
                "--cov-fail-under=0",  # 不自动失败，由我们自己检查
                "tests/unit/"
            ], capture_output=True, text=True, timeout=300)

            # 读取覆盖率报告
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r', encoding='utf-8') as f:
                    coverage_data = json.load(f)

                totals = coverage_data.get("totals", {})
                overall_coverage = totals.get("percent_covered", 0)

                # 检查各层覆盖率
                files = coverage_data.get("files", {})
                layer_coverage = self._analyze_layer_coverage(files)

                passed = overall_coverage >= self.thresholds["coverage"]["overall"]

                result = {
                    "passed": passed,
                    "overall_coverage": overall_coverage,
                    "layer_coverage": layer_coverage,
                    "details": {
                        "covered_lines": totals.get("covered_lines", 0),
                        "num_statements": totals.get("num_statements", 0),
                        "percent_covered": overall_coverage
                    },
                    "recommendations": []
                }

                if not passed:
                    result["recommendations"].append(
                        ".1f"
                    )

                # 检查各层
                for layer, coverage in layer_coverage.items():
                    threshold = self.thresholds["coverage"].get(layer, 75.0)
                    if coverage < threshold:
                        result["recommendations"].append(
                            ".1f"
                        )

                return result
            else:
                return {
                    "passed": False,
                    "error": "无法生成覆盖率报告",
                    "recommendations": ["检查pytest-cov配置"]
                }

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "error": "覆盖率测试执行超时",
                "recommendations": ["优化测试执行时间", "考虑分批运行测试"]
            }
        except Exception as e:
            return {
                "passed": False,
                "error": f"覆盖率检查失败: {e}",
                "recommendations": ["检查测试环境配置"]
            }

    def _analyze_layer_coverage(self, files: Dict[str, Any]) -> Dict[str, float]:
        """分析各层覆盖率"""
        layer_stats = {
            "infrastructure": {"covered": 0, "total": 0},
            "data": {"covered": 0, "total": 0},
            "features": {"covered": 0, "total": 0},
            "ml": {"covered": 0, "total": 0},
            "strategy": {"covered": 0, "total": 0},
            "trading": {"covered": 0, "total": 0},
            "risk": {"covered": 0, "total": 0},
            "core": {"covered": 0, "total": 0}
        }

        for file_path, file_data in files.items():
            if not file_path.startswith("src/"):
                continue

            # 确定层级
            path_parts = file_path.split("/")
            if len(path_parts) < 3:
                continue

            layer = path_parts[2]  # src/layer/...
            if layer not in layer_stats:
                continue

            # 累积统计
            summary = file_data.get("summary", {})
            layer_stats[layer]["covered"] += summary.get("covered_lines", 0)
            layer_stats[layer]["total"] += summary.get("num_statements", 0)

        # 计算覆盖率
        layer_coverage = {}
        for layer, stats in layer_stats.items():
            if stats["total"] > 0:
                layer_coverage[layer] = (stats["covered"] / stats["total"]) * 100
            else:
                layer_coverage[layer] = 0.0

        return layer_coverage

    def _check_test_quality(self) -> Dict[str, Any]:
        """检查测试质量"""
        print("  🧪 分析测试质量...")

        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/", "--collect-only", "-q"
            ], capture_output=True, text=True, timeout=60)

            # 分析测试数量和分布
            test_count = 0
            test_files = 0
            layer_distribution = {}

            lines = result.stdout.split('\\n')
            for line in lines:
                if "::" in line and ("test_" in line or "Test" in line):
                    test_count += 1
                    # 分析层级分布
                    if "tests/unit/" in line:
                        parts = line.split("/")
                        if len(parts) > 3:
                            layer = parts[3]
                            layer_distribution[layer] = layer_distribution.get(layer, 0) + 1

                elif line.endswith(".py"):
                    test_files += 1

            # 计算测试成功率（运行测试）
            success_result = subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/unit/", "--tb=no", "-q"
            ], capture_output=True, text=True, timeout=180)

            passed_tests = 0
            failed_tests = 0

            lines = success_result.stdout.split('\\n')
            for line in lines:
                if "passed" in line:
                    try:
                        passed_tests += int(line.split()[0])
                    except:
                        pass
                elif "failed" in line:
                    try:
                        failed_tests += int(line.split()[0])
                    except:
                        pass

            total_tests = passed_tests + failed_tests
            success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

            passed = success_rate >= self.thresholds["test_success_rate"]

            result = {
                "passed": passed,
                "test_count": test_count,
                "test_files": test_files,
                "success_rate": success_rate,
                "layer_distribution": layer_distribution,
                "recommendations": []
            }

            if not passed:
                result["recommendations"].append(
                    ".1f"
                )

            if test_count < 100:
                result["recommendations"].append(f"测试用例数量偏少({test_count})，建议增加到200+")

            return result

        except Exception as e:
            return {
                "passed": False,
                "error": f"测试质量检查失败: {e}",
                "recommendations": ["检查测试环境配置"]
            }

    def _check_performance(self) -> Dict[str, Any]:
        """检查性能基准"""
        print("  ⚡ 检查性能基准...")

        try:
            # 运行性能基准测试
            result = subprocess.run([
                sys.executable, "scripts/simple_performance_test.py"
            ], capture_output=True, text=True, timeout=120)

            # 读取性能基准结果
            perf_file = self.project_root / "test_logs" / "simple_performance_results.json"
            if perf_file.exists():
                with open(perf_file, 'r', encoding='utf-8') as f:
                    perf_data = json.load(f)

                # 读取基准线
                baseline_file = self.project_root / "test_logs" / "performance_baselines.json"
                if baseline_file.exists():
                    with open(baseline_file, 'r', encoding='utf-8') as f:
                        baseline_data = json.load(f)

                    # 比较性能
                    performance_issues = []
                    for operation, current_time in perf_data.items():
                        baseline_time = baseline_data.get(operation, {}).get("avg_time", current_time)

                        if baseline_time > 0:
                            degradation = ((current_time - baseline_time) / baseline_time) * 100
                            if degradation > self.thresholds["performance_degradation"]:
                                performance_issues.append({
                                    "operation": operation,
                                    "degradation": degradation,
                                    "current": current_time,
                                    "baseline": baseline_time
                                })

                    passed = len(performance_issues) == 0

                    result = {
                        "passed": passed,
                        "performance_issues": performance_issues,
                        "current_performance": perf_data,
                        "baselines": baseline_data,
                        "recommendations": []
                    }

                    if not passed:
                        result["recommendations"].extend([
                            f"性能退化: {issue['operation']} 下降{issue['degradation']:.1f}%"
                            for issue in performance_issues
                        ])

                    return result

            return {
                "passed": True,  # 如果没有基准线，默认通过
                "message": "性能基准数据不完整",
                "recommendations": ["建立完整的性能基准线"]
            }

        except Exception as e:
            return {
                "passed": False,
                "error": f"性能检查失败: {e}",
                "recommendations": ["检查性能测试配置"]
            }

    def _check_code_quality(self) -> Dict[str, Any]:
        """检查代码质量"""
        print("  📏 检查代码质量...")

        try:
            # 检查代码行数和复杂度
            total_lines = 0
            python_files = 0
            quality_issues = []

            for py_file in self.project_root.rglob("src/**/*.py"):
                python_files += 1
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_lines += len(lines)

                        # 简单的复杂度检查（行数>100的函数）
                        in_function = False
                        function_lines = 0

                        for line in lines:
                            stripped = line.strip()
                            if stripped.startswith("def ") or stripped.startswith("    def "):
                                if in_function and function_lines > 100:
                                    quality_issues.append({
                                        "type": "complex_function",
                                        "file": str(py_file),
                                        "lines": function_lines
                                    })
                                in_function = True
                                function_lines = 0
                            elif in_function:
                                if stripped and not stripped.startswith("#"):
                                    function_lines += 1
                                elif not stripped:  # 空行
                                    function_lines += 1

                        # 检查最后的函数
                        if in_function and function_lines > 100:
                            quality_issues.append({
                                "type": "complex_function",
                                "file": str(py_file),
                                "lines": function_lines
                            })

                except Exception:
                    continue

            # 计算质量评分
            quality_score = 100.0
            if quality_issues:
                quality_score -= len(quality_issues) * 5  # 每个问题扣5分

            if total_lines > 50000:  # 代码量过大
                quality_score -= 10

            if python_files < 50:  # 文件数量过少
                quality_score -= 10

            passed = quality_score >= self.thresholds["code_quality_score"]

            result = {
                "passed": passed,
                "quality_score": quality_score,
                "total_lines": total_lines,
                "python_files": python_files,
                "quality_issues": quality_issues,
                "recommendations": []
            }

            if not passed:
                result["recommendations"].append(f"代码质量评分过低: {quality_score:.1f}")

            if quality_issues:
                result["recommendations"].extend([
                    f"重构复杂函数: {issue['file']} ({issue['lines']}行)"
                    for issue in quality_issues[:3]  # 只显示前3个
                ])

            return result

        except Exception as e:
            return {
                "passed": False,
                "error": f"代码质量检查失败: {e}",
                "recommendations": ["检查代码分析工具配置"]
            }

    def _check_security(self) -> Dict[str, Any]:
        """检查安全漏洞"""
        print("  🔒 检查安全漏洞...")

        try:
            vulnerabilities = []

            # 检查危险的代码模式
            dangerous_patterns = [
                ("eval(", "使用eval()可能存在安全风险"),
                ("exec(", "使用exec()可能存在安全风险"),
                ("subprocess.call.*shell=True", "shell=True可能存在命令注入风险"),
                ("input(", "使用input()可能存在安全风险"),
                ("pickle.load", "使用pickle可能存在反序列化攻击风险")
            ]

            for py_file in self.project_root.rglob("src/**/*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                        for pattern, risk in dangerous_patterns:
                            if pattern in content:
                                vulnerabilities.append({
                                    "file": str(py_file),
                                    "pattern": pattern,
                                    "risk": risk
                                })

                except Exception:
                    continue

            passed = len(vulnerabilities) <= self.thresholds["security_vulnerabilities"]

            result = {
                "passed": passed,
                "vulnerabilities": vulnerabilities,
                "recommendations": []
            }

            if not passed:
                result["recommendations"].extend([
                    f"修复安全漏洞: {v['file']} - {v['risk']}"
                    for v in vulnerabilities
                ])

            return result

        except Exception as e:
            return {
                "passed": False,
                "error": f"安全检查失败: {e}",
                "recommendations": ["检查安全扫描工具配置"]
            }

    def _check_complexity(self) -> Dict[str, Any]:
        """检查代码复杂度"""
        print("  🧠 检查代码复杂度...")

        try:
            complexity_issues = []

            for py_file in self.project_root.rglob("src/**/*.py"):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 简单的圈复杂度计算
                        # 每个条件语句+1，每个循环+1，每个异常处理+1
                        complexity = 1  # 基础复杂度

                        complexity_indicators = [
                            "if ", "elif ", "else:",
                            "for ", "while ",
                            "try:", "except ", "finally:",
                            "and ", "or ", "not ",
                            "case ", "match "  # Python 3.10+模式匹配
                        ]

                        for indicator in complexity_indicators:
                            complexity += content.count(indicator)

                        # 检查文件级复杂度
                        if complexity > self.thresholds["complexity_score"] * 10:  # 文件级阈值
                            complexity_issues.append({
                                "file": str(py_file),
                                "complexity": complexity,
                                "type": "file_complexity"
                            })

                except Exception:
                    continue

            passed = len(complexity_issues) == 0

            result = {
                "passed": passed,
                "complexity_issues": complexity_issues,
                "recommendations": []
            }

            if not passed:
                result["recommendations"].extend([
                    f"重构高复杂度文件: {issue['file']} (复杂度: {issue['complexity']})"
                    for issue in complexity_issues[:3]
                ])

            return result

        except Exception as e:
            return {
                "passed": False,
                "error": f"复杂度检查失败: {e}",
                "recommendations": ["检查复杂度分析工具配置"]
            }

    def _generate_quality_report(self, summary: Dict[str, Any]):
        """生成质量门禁报告"""
        print("  📄 生成质量门禁报告...")

        report_content = f"""# 质量门禁检查报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**整体状态**: {"✅ 通过" if summary["overall_status"] == "PASSED" else "❌ 失败"}

## 检查结果汇总

"""

        for check_id, result in summary["checks"].items():
            status = "✅ 通过" if result.get("passed", False) else "❌ 失败"
            report_content += f"### {check_id.replace('_', ' ').title()}\n"
            report_content += f"- **状态**: {status}\\n"

            if "error" in result:
                report_content += f"- **错误**: {result['error']}\\n"

            # 添加具体指标
            if check_id == "coverage_check":
                if "overall_coverage" in result:
                    report_content += f"- **整体覆盖率**: {result['overall_coverage']:.1f}%\\n"
                if "layer_coverage" in result:
                    report_content += "- **分层覆盖率**:\\n"
                    for layer, coverage in result["layer_coverage"].items():
                        report_content += f"  - {layer}: {coverage:.1f}%\\n"

            elif check_id == "test_quality_check":
                if "success_rate" in result:
                    report_content += f"- **测试成功率**: {result['success_rate']:.1f}%\\n"
                if "test_count" in result:
                    report_content += f"- **测试用例数**: {result['test_count']}\\n"

            elif check_id == "performance_check":
                if "performance_issues" in result:
                    report_content += f"- **性能问题数**: {len(result['performance_issues'])}\\n"

            elif check_id == "code_quality_check":
                if "quality_score" in result:
                    report_content += f"- **质量评分**: {result['quality_score']:.1f}\\n"

            elif check_id == "security_check":
                if "vulnerabilities" in result:
                    report_content += f"- **安全漏洞数**: {len(result['vulnerabilities'])}\\n"

            elif check_id == "complexity_check":
                if "complexity_issues" in result:
                    report_content += f"- **复杂度问题数**: {len(result['complexity_issues'])}\\n"

            report_content += "\\n"

        # 改进建议
        if summary["recommendations"]:
            report_content += "## 📋 改进建议\\n\\n"
            for rec in summary["recommendations"]:
                report_content += f"- {rec}\\n"
        else:
            report_content += "## 🎉 无改进建议\\n\\n"
            report_content += "所有质量门禁检查均已通过！\\n"

        report_content += f"""
---
**质量门禁系统自动生成**
**检查阈值**: 覆盖率≥85%, 测试成功率≥95%, 质量评分≥80
**下次检查**: 建议在每次代码提交前运行
"""

        # 保存报告
        report_file = self.project_root / "test_logs" / "quality_gate_report.md"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report_content, encoding='utf-8')

        print(f"✅ 质量门禁报告已生成: {report_file}")


def main():
    """主函数"""
    gate = QualityGate(".")
    summary = gate.run_comprehensive_check()

    # 保存JSON结果
    result_file = Path("test_logs/quality_gate_results.json")
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 返回退出码
    sys.exit(0 if summary["overall_status"] == "PASSED" else 1)


if __name__ == "__main__":
    main()

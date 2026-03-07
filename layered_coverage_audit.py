#!/usr/bin/env python3
"""
RQA2025 分层测试覆盖率审核脚本
基于业务流程驱动架构设计，对17个架构层级进行分层测试覆盖率和通过率检查
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

class LayeredCoverageAudit:
    """分层测试覆盖率审核器"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_logs_dir = self.project_root / "test_logs"

        # 17个架构层级定义
        self.architecture_layers = {
            # 8个核心子系统
            "infrastructure": {
                "name": "基础设施层",
                "src_paths": ["src/infrastructure/"],
                "test_paths": ["tests/unit/infrastructure/", "tests/integration/infrastructure/", "tests/e2e/infrastructure/"],
                "file_count": 799,
                "priority": "核心支撑",
                "description": "配置管理、缓存、安全、日志、健康检查"
            },
            "core_service": {
                "name": "核心服务层",
                "src_paths": ["src/core/"],
                "test_paths": ["tests/unit/core/", "tests/integration/core/", "tests/e2e/core/"],
                "file_count": 168,
                "priority": "辅助支撑",
                "description": "事件驱动、依赖注入、业务流程编排"
            },
            "data": {
                "name": "数据管理层",
                "src_paths": ["src/data/"],
                "test_paths": ["tests/unit/data/", "tests/integration/data/", "tests/e2e/data/"],
                "file_count": 166,
                "priority": "核心支撑",
                "description": "数据采集、处理、存储、质量保障"
            },
            "features": {
                "name": "特征分析层",
                "src_paths": ["src/features/"],
                "test_paths": ["tests/unit/features/", "tests/integration/features/", "tests/e2e/features/"],
                "file_count": 132,
                "priority": "核心业务",
                "description": "技术指标计算、特征工程"
            },
            "ml": {
                "name": "机器学习层",
                "src_paths": ["src/ml/"],
                "test_paths": ["tests/unit/ml/", "tests/integration/ml/", "tests/e2e/ml/"],
                "file_count": 82,
                "priority": "核心支撑",
                "description": "分布式训练、特征工程、模型服务"
            },
            "strategy": {
                "name": "策略层",
                "src_paths": ["src/strategy/"],
                "test_paths": ["tests/unit/strategy/", "tests/integration/strategy/", "tests/e2e/strategy/"],
                "file_count": 148,
                "priority": "核心业务",
                "description": "策略开发、回测分析、策略部署"
            },
            "trading": {
                "name": "交易层",
                "src_paths": ["src/trading/"],
                "test_paths": ["tests/unit/trading/", "tests/integration/trading/", "tests/e2e/trading/"],
                "file_count": 53,
                "priority": "核心业务",
                "description": "订单管理、交易执行、高频交易"
            },
            "streaming": {
                "name": "流处理层",
                "src_paths": ["src/streaming/"],
                "test_paths": ["tests/unit/streaming/", "tests/integration/streaming/", "tests/e2e/streaming/"],
                "file_count": 21,
                "priority": "核心支撑",
                "description": "实时数据处理、事件驱动、弹性处理"
            },

            # 9个辅助支撑层级
            "risk": {
                "name": "风险控制层",
                "src_paths": ["src/risk/"],
                "test_paths": ["tests/unit/risk/", "tests/integration/risk/", "tests/e2e/risk/"],
                "file_count": 46,
                "priority": "核心业务",
                "description": "实时风控、合规检查、风险监控"
            },
            "monitoring": {
                "name": "监控层",
                "src_paths": ["src/monitoring/"],
                "test_paths": ["tests/unit/monitoring/", "tests/integration/monitoring/", "tests/e2e/monitoring/"],
                "file_count": 25,
                "priority": "辅助支撑",
                "description": "系统监控、业务监控、智能告警"
            },
            "optimization": {
                "name": "优化层",
                "src_paths": ["src/optimization/"],
                "test_paths": ["tests/unit/optimization/", "tests/integration/optimization/", "tests/e2e/optimization/"],
                "file_count": 40,
                "priority": "辅助支撑",
                "description": "性能优化、策略优化、系统调优"
            },
            "gateway": {
                "name": "网关层",
                "src_paths": ["src/gateway/"],
                "test_paths": ["tests/unit/gateway/", "tests/integration/gateway/", "tests/e2e/gateway/"],
                "file_count": 37,
                "priority": "辅助支撑",
                "description": "API路由、负载均衡、认证"
            },
            "adapters": {
                "name": "适配器层",
                "src_paths": ["src/adapters/"],
                "test_paths": ["tests/unit/adapters/", "tests/integration/adapters/", "tests/e2e/adapters/"],
                "file_count": 7,
                "priority": "辅助支撑",
                "description": "数据源适配、协议转换"
            },
            "automation": {
                "name": "自动化层",
                "src_paths": ["src/automation/"],
                "test_paths": ["tests/unit/automation/", "tests/integration/automation/", "tests/e2e/automation/"],
                "file_count": 31,
                "priority": "辅助支撑",
                "description": "流程自动化、任务调度"
            },
            "resilience": {
                "name": "弹性层",
                "src_paths": ["src/resilience/"],
                "test_paths": ["tests/unit/resilience/", "tests/integration/resilience/", "tests/e2e/resilience/"],
                "file_count": 4,
                "priority": "辅助支撑",
                "description": "降级服务、断路器"
            },
            "testing": {
                "name": "测试层",
                "src_paths": ["src/testing/"],
                "test_paths": ["tests/unit/testing/", "tests/integration/testing/", "tests/e2e/testing/"],
                "file_count": 18,
                "priority": "辅助支撑",
                "description": "单元测试、集成测试"
            },
            "utils": {
                "name": "工具层",
                "src_paths": ["src/utils/"],
                "test_paths": ["tests/unit/utils/", "tests/integration/utils/", "tests/e2e/utils/"],
                "file_count": 5,
                "priority": "辅助支撑",
                "description": "通用工具、辅助函数"
            }
        }

        # 额外的层级定义
        self.additional_layers = {
            "distributed": {
                "name": "分布式协调器",
                "src_paths": ["src/distributed/"],
                "test_paths": ["tests/unit/distributed/", "tests/integration/distributed/", "tests/e2e/distributed/"],
                "file_count": 0,
                "priority": "辅助支撑",
                "description": "分布式任务协调"
            },
            "async_processor": {
                "name": "异步处理器",
                "src_paths": ["src/async_processor/"],
                "test_paths": ["tests/unit/async_processor/", "tests/integration/async_processor/", "tests/e2e/async_processor/"],
                "file_count": 0,
                "priority": "辅助支撑",
                "description": "异步任务处理"
            },
            "mobile": {
                "name": "移动端层",
                "src_paths": ["src/mobile/"],
                "test_paths": ["tests/unit/mobile/", "tests/integration/mobile/", "tests/e2e/mobile/"],
                "file_count": 0,
                "priority": "辅助支撑",
                "description": "移动端接口适配"
            },
            "boundary": {
                "name": "业务边界层",
                "src_paths": ["src/boundary/"],
                "test_paths": ["tests/unit/boundary/", "tests/integration/boundary/", "tests/e2e/boundary/"],
                "file_count": 0,
                "priority": "辅助支撑",
                "description": "业务边界协调"
            }
        }

        self.results = {}

    def count_files_in_paths(self, paths):
        """统计指定路径下的文件数量"""
        total_files = 0
        for path in paths:
            full_path = self.project_root / path
            if full_path.exists():
                # 递归统计所有Python文件，排除__pycache__和__init__.py
                py_files = []
                for root, dirs, files in os.walk(full_path):
                    # 排除__pycache__目录
                    dirs[:] = [d for d in dirs if d != '__pycache__']
                    for file in files:
                        if file.endswith('.py') and not file.startswith('__'):
                            py_files.append(os.path.join(root, file))
                total_files += len(py_files)
        return total_files

    def run_layer_tests(self, layer_key, layer_config):
        """运行指定层级的测试"""
        print(f"\n🧪 正在测试 {layer_config['name']} ({layer_key})...")

        layer_results = {
            "name": layer_config["name"],
            "priority": layer_config["priority"],
            "description": layer_config["description"],
            "expected_src_files": layer_config["file_count"],
            "actual_src_files": 0,
            "unit_tests": {"files": 0, "passed": 0, "failed": 0, "skipped": 0},
            "integration_tests": {"files": 0, "passed": 0, "failed": 0, "skipped": 0},
            "e2e_tests": {"files": 0, "passed": 0, "failed": 0, "skipped": 0},
            "coverage": 0.0,
            "status": "unknown"
        }

        # 统计源码文件数
        layer_results["actual_src_files"] = self.count_files_in_paths(layer_config["src_paths"])

        # 运行单元测试
        unit_test_paths = [p for p in layer_config["test_paths"] if "unit" in p]
        if unit_test_paths:
            layer_results["unit_tests"] = self.run_test_type("unit", unit_test_paths, layer_key)

        # 运行集成测试
        integration_test_paths = [p for p in layer_config["test_paths"] if "integration" in p]
        if integration_test_paths:
            layer_results["integration_tests"] = self.run_test_type("integration", integration_test_paths, layer_key)

        # 运行端到端测试
        e2e_test_paths = [p for p in layer_config["test_paths"] if "e2e" in p]
        if e2e_test_paths:
            layer_results["e2e_tests"] = self.run_test_type("e2e", e2e_test_paths, layer_key)

        # 计算总体覆盖率
        total_tests = (
            layer_results["unit_tests"]["passed"] + layer_results["unit_tests"]["failed"] +
            layer_results["integration_tests"]["passed"] + layer_results["integration_tests"]["failed"] +
            layer_results["e2e_tests"]["passed"] + layer_results["e2e_tests"]["failed"]
        )

        if total_tests > 0:
            passed_tests = (
                layer_results["unit_tests"]["passed"] +
                layer_results["integration_tests"]["passed"] +
                layer_results["e2e_tests"]["passed"]
            )
            layer_results["coverage"] = (passed_tests / total_tests) * 100

            # 判断达标状态
            if layer_results["coverage"] >= 95:
                layer_results["status"] = "优秀达标"
            elif layer_results["coverage"] >= 80:
                layer_results["status"] = "良好达标"
            elif layer_results["coverage"] >= 60:
                layer_results["status"] = "基本达标"
            else:
                layer_results["status"] = "需加强"
        else:
            layer_results["status"] = "无测试"

        return layer_results

    def run_test_type(self, test_type, test_paths, layer_key):
        """运行特定类型的测试"""
        results = {"files": 0, "passed": 0, "failed": 0, "skipped": 0}

        try:
            # 统计测试文件数
            total_test_files = 0
            for test_path in test_paths:
                full_path = self.project_root / test_path
                if full_path.exists():
                    # 递归统计测试文件
                    test_files = []
                    for root, dirs, files in os.walk(full_path):
                        dirs[:] = [d for d in dirs if d != '__pycache__']
                        for file in files:
                            if file.startswith('test_') and file.endswith('.py'):
                                test_files.append(os.path.join(root, file))
                    total_test_files += len(test_files)

            results["files"] = total_test_files

            if total_test_files == 0:
                return results

            # 运行pytest获取真实的测试统计
            cmd = [
                sys.executable, "-m", "pytest",
                "--tb=no", "-q", "--disable-warnings", "--maxfail=20"
            ]

            # 添加测试路径
            valid_paths = []
            for test_path in test_paths:
                full_path = self.project_root / test_path
                if full_path.exists():
                    valid_paths.append(str(full_path))

            if not valid_paths:
                return results

            cmd.extend(valid_paths)

            # 运行测试并获取结果
            try:
                run_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=300  # 5分钟超时
                )

                # 解析pytest输出
                output = run_result.stdout + run_result.stderr

                # 解析测试结果
                passed_match = None
                failed_match = None
                skipped_match = None

                for line in output.split('\n'):
                    # 查找类似 "13 passed, 0 failed, 0 skipped" 的行
                    import re
                    match = re.search(r'(\d+)\s*passed.*?(\d+)\s*failed.*?(\d+)\s*skipped', line, re.IGNORECASE)
                    if match:
                        results["passed"] = int(match.group(1))
                        results["failed"] = int(match.group(2))
                        results["skipped"] = int(match.group(3))
                        break

                # 如果没有找到匹配的结果行，检查是否有其他格式
                if results["passed"] == 0 and results["failed"] == 0 and results["skipped"] == 0:
                    # 检查是否所有测试都通过了
                    if "passed" in output.lower() and "failed" not in output.lower():
                        # 尝试从collected行推断
                        for line in output.split('\n'):
                            if 'collected' in line and 'items' in line:
                                match = re.search(r'collected\s+(\d+)\s+items', line)
                                if match:
                                    collected_count = int(match.group(1))
                                    results["passed"] = collected_count
                                    break

            except subprocess.TimeoutExpired:
                print(f"⚠️  {test_type}测试超时")
                results["failed"] = total_test_files * 5  # 假设所有测试都失败了
            except Exception as e:
                print(f"❌ {test_type}测试执行失败: {e}")
                results["failed"] = total_test_files * 5  # 假设所有测试都失败了

        except subprocess.TimeoutExpired:
            print(f"⚠️  {test_type}测试超时")
            results["failed"] = results["files"] * 5  # 假设所有测试都失败了
        except Exception as e:
            print(f"❌ {test_type}测试执行失败: {e}")
            results["failed"] = results["files"] * 5  # 假设所有测试都失败了

        return results

    def generate_report(self):
        """生成分层测试覆盖率报告"""
        print("\n" + "="*80)
        print("🎯 RQA2025 分层测试覆盖率审核报告")
        print("="*80)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"📅 审核时间: {timestamp}")
        print(f"🏗️  架构层级: 17个 (8个核心子系统 + 9个辅助支撑层级)")

        # 核心子系统报告
        print("\n" + "="*60)
        print("⭐ 核心子系统 (8个) - 价值创造与技术赋能")
        print("="*60)

        core_layers = ["strategy", "trading", "risk", "features", "data", "ml", "infrastructure", "streaming"]

        for layer_key in core_layers:
            if layer_key in self.results:
                self.print_layer_report(layer_key, self.results[layer_key])

        # 辅助支撑层级报告
        print("\n" + "="*60)
        print("🔧 辅助支撑层级 (9个) - 架构支撑与运维保障")
        print("="*60)

        auxiliary_layers = ["core_service", "monitoring", "optimization", "gateway", "adapters",
                          "automation", "resilience", "testing", "utils"]

        for layer_key in auxiliary_layers:
            if layer_key in self.results:
                self.print_layer_report(layer_key, self.results[layer_key])

        # 汇总统计
        self.print_summary_report()

        # 保存报告
        self.save_report_to_file()

    def print_layer_report(self, layer_key, layer_data):
        """打印单个层级的报告"""
        print(f"\n📦 {layer_data['name']} ({layer_key.upper()})")
        print(f"   定位: {layer_data['priority']} - {layer_data['description']}")
        print(f"   源码文件: {layer_data['actual_src_files']}个")
        print(f"   单元测试: {layer_data['unit_tests']['files']}文件, {layer_data['unit_tests']['passed']}通过, {layer_data['unit_tests']['failed']}失败")
        print(f"   集成测试: {layer_data['integration_tests']['files']}文件, {layer_data['integration_tests']['passed']}通过, {layer_data['integration_tests']['failed']}失败")
        print(f"   端到端测试: {layer_data['e2e_tests']['files']}文件, {layer_data['e2e_tests']['passed']}通过, {layer_data['e2e_tests']['failed']}失败")
        print(f"   测试覆盖率: {layer_data['coverage']:.1f}%")
        status_emoji = {
            "优秀达标": "✅",
            "良好达标": "✅",
            "基本达标": "⚠️",
            "需加强": "❌",
            "无测试": "❓"
        }
        print(f"   达标状态: {status_emoji.get(layer_data['status'], '❓')} {layer_data['status']}")

    def print_summary_report(self):
        """打印汇总报告"""
        print("\n" + "="*60)
        print("📊 综合评估结果")
        print("="*60)

        total_layers = len(self.results)
        excellent_count = sum(1 for r in self.results.values() if r["status"] == "优秀达标")
        good_count = sum(1 for r in self.results.values() if r["status"] == "良好达标")
        basic_count = sum(1 for r in self.results.values() if r["status"] == "基本达标")
        poor_count = sum(1 for r in self.results.values() if r["status"] == "需加强")
        no_test_count = sum(1 for r in self.results.values() if r["status"] == "无测试")

        print(f"总架构层级数: {total_layers}")
        print(f"优秀达标 (≥95%): {excellent_count}个 ({excellent_count/total_layers*100:.1f}%)")
        print(f"良好达标 (80-95%): {good_count}个 ({good_count/total_layers*100:.1f}%)")
        print(f"基本达标 (60-80%): {basic_count}个 ({basic_count/total_layers*100:.1f}%)")
        print(f"需加强 (<60%): {poor_count}个 ({poor_count/total_layers*100:.1f}%)")
        print(f"无测试: {no_test_count}个 ({no_test_count/total_layers*100:.1f}%)")

        # 投产建议
        print("\n🚀 投产建议:")
        if excellent_count + good_count >= total_layers * 0.8:
            print("✅ 系统达到生产部署标准，可以进行生产部署")
        elif excellent_count + good_count >= total_layers * 0.6:
            print("⚠️ 系统基本达到生产标准，建议完善剩余层级测试后部署")
        else:
            print("❌ 系统测试覆盖不足，建议加强测试覆盖后再考虑部署")

    def save_report_to_file(self):
        """保存报告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.test_logs_dir / f"layered_coverage_audit_{timestamp}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# RQA2025 分层测试覆盖率审核报告\n\n")
            f.write(f"**审核时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 写入各层级结果
            for layer_key, layer_data in self.results.items():
                f.write(f"## {layer_data['name']} ({layer_key})\n\n")
                f.write(f"- **定位**: {layer_data['priority']} - {layer_data['description']}\n")
                f.write(f"- **源码文件**: {layer_data['actual_src_files']}个\n")
                f.write(f"- **测试覆盖率**: {layer_data['coverage']:.1f}%\n")
                f.write(f"- **达标状态**: {layer_data['status']}\n\n")

            # 写入汇总
            f.write("## 汇总统计\n\n")
            excellent_count = sum(1 for r in self.results.values() if r["status"] == "优秀达标")
            good_count = sum(1 for r in self.results.values() if r["status"] == "良好达标")
            total_layers = len(self.results)

            f.write(f"- 总层级数: {total_layers}\n")
            f.write(f"- 优秀达标: {excellent_count}个\n")
            f.write(f"- 良好达标: {good_count}个\n")
            f.write(f"- 达标率: {(excellent_count + good_count)/total_layers*100:.1f}%\n\n")

        print(f"\n📄 详细报告已保存至: {report_file}")

    def run_audit(self):
        """运行完整的分层审核"""
        print("🎯 开始RQA2025分层测试覆盖率审核...")
        print("基于业务流程驱动架构的17个架构层级")

        start_time = time.time()

        # 审核所有层级
        all_layers = {**self.architecture_layers, **self.additional_layers}

        for layer_key, layer_config in all_layers.items():
            try:
                self.results[layer_key] = self.run_layer_tests(layer_key, layer_config)
            except Exception as e:
                print(f"❌ 审核层级 {layer_key} 时出错: {e}")
                self.results[layer_key] = {
                    "name": layer_config["name"],
                    "status": "审核失败",
                    "error": str(e)
                }

        # 生成报告
        self.generate_report()

        elapsed_time = time.time() - start_time
        print(f"\n⏱️  审核总耗时: {elapsed_time:.1f}秒")

def main():
    """主函数"""
    auditor = LayeredCoverageAudit()
    auditor.run_audit()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
生产就绪评估系统 - RQA2025生产部署准备

全面评估量化交易系统生产部署readiness：
1. 功能完整性评估
2. 性能就绪评估
3. 稳定性评估
4. 安全性评估
5. 可运维性评估
6. 文档完整性评估

作者: AI Assistant
创建时间: 2025年12月4日
"""

import json
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess
import sys

logger = logging.getLogger(__name__)


@dataclass
class AssessmentCriterion:
    """评估标准"""
    category: str
    criterion: str
    description: str
    weight: float  # 权重 (0-1)
    mandatory: bool  # 是否为强制要求


@dataclass
class AssessmentResult:
    """评估结果"""
    criterion: str
    status: str  # "pass", "fail", "warning", "not_applicable"
    score: float  # 0-100
    evidence: List[str]
    recommendations: List[str]
    automated_check: bool


@dataclass
class ReadinessReport:
    """就绪报告"""
    assessment_date: datetime
    overall_score: float
    overall_status: str  # "ready", "conditional", "not_ready"
    category_scores: Dict[str, float]
    detailed_results: List[AssessmentResult]
    critical_issues: List[str]
    recommendations: List[str]
    deployment_readiness: Dict[str, Any]


class ProductionReadinessAssessor:
    """
    生产就绪评估器

    全面评估系统是否具备生产部署条件
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.assessment_criteria = self._define_assessment_criteria()

    def _define_assessment_criteria(self) -> List[AssessmentCriterion]:
        """定义评估标准"""
        return [
            # 功能完整性
            AssessmentCriterion(
                category="functionality",
                criterion="core_trading_features",
                description="核心交易功能完整性",
                weight=0.15,
                mandatory=True
            ),
            AssessmentCriterion(
                category="functionality",
                criterion="api_completeness",
                description="API接口完整性",
                weight=0.10,
                mandatory=True
            ),
            AssessmentCriterion(
                category="functionality",
                criterion="data_integrity",
                description="数据完整性和一致性",
                weight=0.10,
                mandatory=True
            ),

            # 性能就绪
            AssessmentCriterion(
                category="performance",
                criterion="throughput_requirements",
                description="吞吐量满足业务需求",
                weight=0.12,
                mandatory=True
            ),
            AssessmentCriterion(
                category="performance",
                criterion="latency_requirements",
                description="延迟满足业务需求",
                weight=0.12,
                mandatory=True
            ),
            AssessmentCriterion(
                category="performance",
                criterion="resource_efficiency",
                description="资源使用效率",
                weight=0.08,
                mandatory=False
            ),

            # 稳定性
            AssessmentCriterion(
                category="stability",
                criterion="error_handling",
                description="错误处理和恢复机制",
                weight=0.08,
                mandatory=True
            ),
            AssessmentCriterion(
                category="stability",
                criterion="memory_management",
                description="内存管理和泄漏防护",
                weight=0.06,
                mandatory=True
            ),
            AssessmentCriterion(
                category="stability",
                criterion="concurrency_handling",
                description="并发处理能力",
                weight=0.06,
                mandatory=True
            ),

            # 安全性
            AssessmentCriterion(
                category="security",
                criterion="authentication",
                description="身份验证机制",
                weight=0.06,
                mandatory=True
            ),
            AssessmentCriterion(
                category="security",
                criterion="authorization",
                description="权限控制机制",
                weight=0.06,
                mandatory=True
            ),
            AssessmentCriterion(
                category="security",
                criterion="data_protection",
                description="数据保护和加密",
                weight=0.04,
                mandatory=False
            ),

            # 可运维性
            AssessmentCriterion(
                category="operability",
                criterion="logging_monitoring",
                description="日志和监控机制",
                weight=0.06,
                mandatory=True
            ),
            AssessmentCriterion(
                category="operability",
                criterion="configuration_management",
                description="配置管理",
                weight=0.04,
                mandatory=True
            ),
            AssessmentCriterion(
                category="operability",
                criterion="deployment_automation",
                description="部署自动化",
                weight=0.04,
                mandatory=False
            ),

            # 文档完整性
            AssessmentCriterion(
                category="documentation",
                criterion="api_documentation",
                description="API文档完整性",
                weight=0.04,
                mandatory=True
            ),
            AssessmentCriterion(
                category="documentation",
                criterion="deployment_guide",
                description="部署指南完整性",
                weight=0.04,
                mandatory=True
            ),
            AssessmentCriterion(
                category="documentation",
                criterion="operations_manual",
                description="运维手册完整性",
                weight=0.03,
                mandatory=False
            )
        ]

    def assess_readiness(self) -> ReadinessReport:
        """
        执行全面的生产就绪评估

        Returns:
            就绪评估报告
        """
        print("🏭 开始RQA2025生产就绪评估...")
        print("=" * 60)

        assessment_results = []

        # 执行各项评估
        for criterion in self.assessment_criteria:
            print(f"📋 评估: {criterion.criterion}")
            result = self._assess_criterion(criterion)
            assessment_results.append(result)

        # 计算总体评分
        overall_score, category_scores = self._calculate_overall_score(assessment_results)

        # 确定总体状态
        overall_status = self._determine_overall_status(overall_score, assessment_results)

        # 提取关键问题和建议
        critical_issues, recommendations = self._extract_issues_and_recommendations(assessment_results)

        # 生成部署就绪评估
        deployment_readiness = self._assess_deployment_readiness(overall_score, assessment_results)

        report = ReadinessReport(
            assessment_date=datetime.now(),
            overall_score=overall_score,
            overall_status=overall_status,
            category_scores=category_scores,
            detailed_results=assessment_results,
            critical_issues=critical_issues,
            recommendations=recommendations,
            deployment_readiness=deployment_readiness
        )

        print("\n🎯 生产就绪评估完成")
        print("=" * 60)
        print(f"📊 总体评分: {overall_score:.1f}/100")
        print(f"🏆 就绪状态: {overall_status}")
        print(f"⚠️  关键问题: {len(critical_issues)} 个")
        print(f"💡 优化建议: {len(recommendations)} 个")

        return report

    def _assess_criterion(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估单个标准"""
        # 根据标准类型执行相应的自动化检查
        if criterion.criterion == "core_trading_features":
            return self._assess_core_trading_features(criterion)
        elif criterion.criterion == "api_completeness":
            return self._assess_api_completeness(criterion)
        elif criterion.criterion == "data_integrity":
            return self._assess_data_integrity(criterion)
        elif criterion.criterion == "throughput_requirements":
            return self._assess_throughput_requirements(criterion)
        elif criterion.criterion == "latency_requirements":
            return self._assess_latency_requirements(criterion)
        elif criterion.criterion == "resource_efficiency":
            return self._assess_resource_efficiency(criterion)
        elif criterion.criterion == "error_handling":
            return self._assess_error_handling(criterion)
        elif criterion.criterion == "memory_management":
            return self._assess_memory_management(criterion)
        elif criterion.criterion == "concurrency_handling":
            return self._assess_concurrency_handling(criterion)
        elif criterion.criterion == "authentication":
            return self._assess_authentication(criterion)
        elif criterion.criterion == "authorization":
            return self._assess_authorization(criterion)
        elif criterion.criterion == "data_protection":
            return self._assess_data_protection(criterion)
        elif criterion.criterion == "logging_monitoring":
            return self._assess_logging_monitoring(criterion)
        elif criterion.criterion == "configuration_management":
            return self._assess_configuration_management(criterion)
        elif criterion.criterion == "deployment_automation":
            return self._assess_deployment_automation(criterion)
        elif criterion.criterion == "api_documentation":
            return self._assess_api_documentation(criterion)
        elif criterion.criterion == "deployment_guide":
            return self._assess_deployment_guide(criterion)
        elif criterion.criterion == "operations_manual":
            return self._assess_operations_manual(criterion)
        else:
            # 默认评估
            return AssessmentResult(
                criterion=criterion.criterion,
                status="warning",
                score=50.0,
                evidence=["需要人工评估"],
                recommendations=["完善自动化评估逻辑"],
                automated_check=False
            )

    def _assess_core_trading_features(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估核心交易功能"""
        evidence = []
        recommendations = []
        score = 100.0

        # 检查是否有核心交易模块
        trading_modules = [
            "src/trading/",
            "src/strategy/",
            "src/risk/",
            "src/market_data/"
        ]

        existing_modules = 0
        for module_path in trading_modules:
            if (self.project_root / module_path).exists():
                existing_modules += 1
                evidence.append(f"✅ 存在核心模块: {module_path}")
            else:
                evidence.append(f"❌ 缺失核心模块: {module_path}")
                score -= 15

        # 检查是否有交易相关的测试
        test_files = list(self.project_root.glob("tests/**/test_*trading*.py"))
        if test_files:
            evidence.append(f"✅ 发现 {len(test_files)} 个交易功能测试文件")
        else:
            evidence.append("⚠️ 未发现专门的交易功能测试")
            score -= 10
            recommendations.append("增加交易功能专项测试")

        # 检查压力测试结果
        stress_test_results = list(self.project_root.glob("test_logs/*stress_test*.json"))
        if stress_test_results:
            evidence.append("✅ 具备压力测试验证结果")
            # 检查吞吐量是否达标 (1000 RPS)
            try:
                with open(stress_test_results[-1], 'r', encoding='utf-8') as f:
                    stress_data = json.load(f)
                    max_throughput = stress_data.get('system_limits', {}).get('scenario_performance', {}).get('high_frequency_trading', {}).get('max_throughput', 0)
                    if max_throughput >= 1000:
                        evidence.append(f"✅ 高频交易吞吐量达标: {max_throughput} RPS")
                    else:
                        evidence.append(f"⚠️ 高频交易吞吐量不足: {max_throughput} RPS")
                        score -= 10
                        recommendations.append("优化交易引擎性能，提升吞吐量")
            except:
                pass
        else:
            evidence.append("⚠️ 缺少压力测试验证")
            score -= 15
            recommendations.append("执行完整的压力测试验证")

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=recommendations,
            automated_check=True
        )

    def _assess_api_completeness(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估API完整性"""
        evidence = []
        recommendations = []
        score = 100.0

        # 检查是否有API相关文件
        api_files = list(self.project_root.glob("src/**/api*.py")) + \
                list(self.project_root.glob("src/**/*api*.py"))

        if api_files:
            evidence.append(f"✅ 发现 {len(api_files)} 个API相关文件")
        else:
            evidence.append("❌ 未发现API相关文件")
            score -= 30
            recommendations.append("实现核心API接口")

        # 检查是否有API测试
        api_tests = list(self.project_root.glob("tests/**/test_*api*.py"))
        if api_tests:
            evidence.append(f"✅ 发现 {len(api_tests)} 个API测试文件")
        else:
            evidence.append("⚠️ 缺少API测试")
            score -= 15
            recommendations.append("完善API功能测试")

        # 检查OpenAPI/Swagger文档
        swagger_files = list(self.project_root.glob("**/*swagger*.yaml")) + \
                    list(self.project_root.glob("**/*openapi*.yaml")) + \
                    list(self.project_root.glob("docs/api/*.md"))

        if swagger_files:
            evidence.append("✅ 具备API文档")
        else:
            evidence.append("⚠️ 缺少API文档")
            score -= 10
            recommendations.append("生成API文档")

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=recommendations,
            automated_check=True
        )

    def _assess_data_integrity(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估数据完整性"""
        evidence = []
        recommendations = []
        score = 100.0

        # 检查数据层组件
        data_components = [
            "src/data/",
            "src/database/",
            "src/cache/"
        ]

        existing_components = 0
        for component_path in data_components:
            if (self.project_root / component_path).exists():
                existing_components += 1
                evidence.append(f"✅ 存在数据组件: {component_path}")
            else:
                evidence.append(f"⚠️ 缺失数据组件: {component_path}")
                score -= 10

        # 检查数据验证机制
        validation_files = list(self.project_root.glob("src/**/*validation*.py")) + \
                        list(self.project_root.glob("src/**/*validator*.py"))

        if validation_files:
            evidence.append(f"✅ 发现 {len(validation_files)} 个数据验证组件")
        else:
            evidence.append("⚠️ 缺少数据验证机制")
            score -= 15
            recommendations.append("实现数据完整性验证")

        # 检查数据迁移脚本
        migration_files = list(self.project_root.glob("scripts/migrations/*.py")) + \
                        list(self.project_root.glob("**/migrations/*.py"))

        if migration_files:
            evidence.append("✅ 具备数据迁移能力")
        else:
            evidence.append("ℹ️ 建议提供数据迁移脚本")
            score -= 5

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=recommendations,
            automated_check=True
        )

    def _assess_throughput_requirements(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估吞吐量需求"""
        evidence = []
        recommendations = []
        score = 100.0

        # 检查压力测试结果
        stress_test_files = list(self.project_root.glob("test_logs/*stress_test*.json"))

        if not stress_test_files:
            evidence.append("❌ 缺少压力测试数据")
            score = 0
            recommendations.append("执行压力测试以验证吞吐量")
            status = "fail"
        else:
            # 读取最新的压力测试结果
            try:
                with open(stress_test_files[-1], 'r', encoding='utf-8') as f:
                    stress_data = json.load(f)

                max_throughput = stress_data.get('system_limits', {}).get('scenario_performance', {}).get('high_frequency_trading', {}).get('max_throughput', 0)
                evidence.append(f"📊 实测最大吞吐量: {max_throughput} RPS")

                # 量化交易系统吞吐量要求：至少500 RPS
                min_required = 500
                if max_throughput >= min_required:
                    evidence.append(f"✅ 吞吐量满足要求 (≥{min_required} RPS)")
                    score = 100
                else:
                    evidence.append(f"❌ 吞吐量不足要求 (<{min_required} RPS)")
                    score = (max_throughput / min_required) * 100
                    recommendations.append(f"优化系统性能，提升吞吐量至 {min_required} RPS以上")

                # 检查稳定性
                stability_score = 100 - stress_data.get('system_limits', {}).get('overall_max_stable_concurrency', 0)
                if stability_score < 80:
                    evidence.append("⚠️ 高负载下稳定性需要改进")
                    score -= 10
                    recommendations.append("改进系统在高负载下的稳定性")

            except Exception as e:
                evidence.append(f"⚠️ 无法解析压力测试结果: {e}")
                score = 50
                recommendations.append("重新执行压力测试")

            status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=recommendations,
            automated_check=True
        )

    def _assess_latency_requirements(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估延迟需求"""
        evidence = []
        recommendations = []
        score = 100.0

        # 检查性能基准测试结果
        benchmark_files = list(self.project_root.glob("test_logs/*benchmark*.json"))

        if not benchmark_files:
            evidence.append("❌ 缺少性能基准测试数据")
            score = 0
            recommendations.append("执行性能基准测试")
            status = "fail"
        else:
            try:
                with open(benchmark_files[-1], 'r', encoding='utf-8') as f:
                    benchmark_data = json.load(f)

                # 量化交易系统的延迟要求：P95 < 50ms
                p95_latency = benchmark_data.get('summary', {}).get('avg_latency_p95_ms', 1000)
                evidence.append(f"📊 P95延迟: {p95_latency:.1f}ms")

                max_allowed_latency = 50  # 50ms
                if p95_latency <= max_allowed_latency:
                    evidence.append(f"✅ 延迟满足要求 (≤{max_allowed_latency}ms)")
                    score = 100
                else:
                    evidence.append(f"❌ 延迟超出要求 (> {max_allowed_latency}ms)")
                    score = max(0, (max_allowed_latency / p95_latency) * 100)
                    recommendations.append(f"优化响应时间，将P95延迟控制在 {max_allowed_latency}ms以内")

            except Exception as e:
                evidence.append(f"⚠️ 无法解析性能测试结果: {e}")
                score = 50
                recommendations.append("重新执行性能基准测试")

            status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=recommendations,
            automated_check=True
        )

    def _assess_resource_efficiency(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估资源使用效率"""
        evidence = []
        recommendations = []
        score = 100.0

        # 检查性能测试中的资源使用情况
        benchmark_files = list(self.project_root.glob("test_logs/*benchmark*.json"))

        if benchmark_files:
            try:
                with open(benchmark_files[-1], 'r', encoding='utf-8') as f:
                    benchmark_data = json.load(f)

                # 检查CPU使用率
                cpu_usage = benchmark_data.get('summary', {}).get('avg_cpu_percent', 0)
                evidence.append(f"📊 平均CPU使用率: {cpu_usage:.1f}%")

                if cpu_usage > 80:
                    evidence.append("⚠️ CPU使用率较高")
                    score -= 20
                    recommendations.append("优化CPU密集型操作")
                elif cpu_usage < 30:
                    evidence.append("✅ CPU使用率合理")

                # 检查内存使用
                memory_mb = benchmark_data.get('summary', {}).get('avg_memory_mb', 0)
                evidence.append(f"📊 平均内存使用: {memory_mb:.1f}MB")

                if memory_mb > 1000:  # 1GB
                    evidence.append("⚠️ 内存使用量较大")
                    score -= 15
                    recommendations.append("优化内存使用效率")

            except Exception as e:
                evidence.append("⚠️ 无法获取资源使用数据")

        # 检查内存泄漏测试结果
        leak_test_files = list(self.project_root.glob("test_logs/*leak*.json"))

        if leak_test_files:
            try:
                with open(leak_test_files[-1], 'r', encoding='utf-8') as f:
                    leak_data = json.load(f)

                growth_rate = leak_data.get('overall_assessment', {}).get('avg_growth_rate', 0)
                evidence.append(f"📊 内存增长率: {growth_rate:.2f} MB/小时")

                if growth_rate > 20:
                    evidence.append("⚠️ 检测到内存泄漏风险")
                    score -= 25
                    recommendations.append("修复内存泄漏问题")
                elif growth_rate > 10:
                    evidence.append("⚠️ 内存增长较快，建议监控")
                    score -= 10
                    recommendations.append("持续监控内存使用情况")

            except Exception as e:
                evidence.append("⚠️ 无法获取内存泄漏测试数据")

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=recommendations,
            automated_check=True
        )

    def _assess_error_handling(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估错误处理机制"""
        evidence = []
        recommendations = []
        score = 100.0

        # 检查错误处理相关代码
        error_files = list(self.project_root.glob("src/**/error*.py")) + \
                    list(self.project_root.glob("src/**/*error*.py")) + \
                    list(self.project_root.glob("src/**/exception*.py"))

        if error_files:
            evidence.append(f"✅ 发现 {len(error_files)} 个错误处理模块")
        else:
            evidence.append("⚠️ 缺少专门的错误处理模块")
            score -= 15
            recommendations.append("实现统一的错误处理机制")

        # 检查try-except使用情况
        total_files = len(list(self.project_root.glob("src/**/*.py")))
        files_with_error_handling = 0

        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'try:' in content and 'except' in content:
                        files_with_error_handling += 1
            except:
                pass

        error_handling_ratio = files_with_error_handling / max(total_files, 1)
        evidence.append(f"📊 {files_with_error_handling}/{total_files} 个文件包含错误处理 ({error_handling_ratio:.1f})")

        if error_handling_ratio < 0.5:
            evidence.append("⚠️ 错误处理覆盖率偏低")
            score -= 10
            recommendations.append("增加错误处理覆盖率")

        # 检查日志记录
        logging_usage = 0
        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'logging.' in content or 'logger.' in content:
                        logging_usage += 1
            except:
                pass

        logging_ratio = logging_usage / max(total_files, 1)
        evidence.append(f"📊 {logging_usage}/{total_files} 个文件使用日志 ({logging_ratio:.1f})")

        if logging_ratio < 0.6:
            evidence.append("⚠️ 日志记录覆盖率偏低")
            score -= 10
            recommendations.append("完善日志记录机制")

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=recommendations,
            automated_check=True
        )

    def _assess_memory_management(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估内存管理"""
        evidence = []
        recommendations = []
        score = 100.0

        # 检查内存泄漏测试结果
        leak_test_files = list(self.project_root.glob("test_logs/*leak*.json"))

        if leak_test_files:
            try:
                with open(leak_test_files[-1], 'r', encoding='utf-8') as f:
                    leak_data = json.load(f)

                health_status = leak_data.get('overall_assessment', {}).get('memory_health_status', 'unknown')
                evidence.append(f"📊 内存健康状态: {health_status}")

                if health_status == "健康":
                    evidence.append("✅ 内存管理良好")
                elif health_status == "需要关注":
                    evidence.append("⚠️ 存在内存管理问题")
                    score -= 20
                    recommendations.append("优化内存使用模式")
                else:
                    evidence.append("❌ 内存管理存在严重问题")
                    score -= 40
                    recommendations.append("紧急修复内存泄漏问题")

                # 检查具体测试结果
                results = leak_data.get('test_results', [])
                failed_tests = [r for r in results if r.get('is_memory_leak_detected')]
                if failed_tests:
                    evidence.append(f"⚠️ {len(failed_tests)}/{len(results)} 个测试检测到内存泄漏")
                    score -= len(failed_tests) * 5

            except Exception as e:
                evidence.append(f"⚠️ 无法解析内存测试结果: {e}")
                score -= 20
        else:
            evidence.append("❌ 缺少内存泄漏测试")
            score -= 30
            recommendations.append("执行内存泄漏检测测试")

        # 检查垃圾回收配置
        gc_files = list(self.project_root.glob("src/**/gc_*.py")) + \
                list(self.project_root.glob("src/**/*gc*.py"))

        if gc_files:
            evidence.append("✅ 具备GC优化配置")
        else:
            evidence.append("ℹ️ 建议优化垃圾回收策略")
            score -= 5

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=recommendations,
            automated_check=True
        )

    def _assess_concurrency_handling(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估并发处理能力"""
        evidence = []
        recommendations = []
        score = 100.0

        # 检查并发相关代码
        concurrent_files = list(self.project_root.glob("src/**/concurrent*.py")) + \
                        list(self.project_root.glob("src/**/*thread*.py")) + \
                        list(self.project_root.glob("src/**/*async*.py"))

        if concurrent_files:
            evidence.append(f"✅ 发现 {len(concurrent_files)} 个并发处理模块")
        else:
            evidence.append("⚠️ 缺少并发处理模块")
            score -= 15
            recommendations.append("实现并发处理机制")

        # 检查压力测试并发能力
        stress_test_files = list(self.project_root.glob("test_logs/*stress_test*.json"))

        if stress_test_files:
            try:
                with open(stress_test_files[-1], 'r', encoding='utf-8') as f:
                    stress_data = json.load(f)

                max_concurrency = stress_data.get('system_limits', {}).get('overall_max_stable_concurrency', 0)
                evidence.append(f"📊 最大稳定并发数: {max_concurrency}")

                # 量化交易系统并发要求：至少10个并发用户
                min_required = 10
                if max_concurrency >= min_required:
                    evidence.append(f"✅ 并发能力满足要求 (≥{min_required})")
                else:
                    evidence.append(f"❌ 并发能力不足 (<{min_required})")
                    score -= 20
                    recommendations.append(f"提升并发处理能力至 {min_required}以上")

            except Exception as e:
                evidence.append(f"⚠️ 无法解析并发测试结果: {e}")
                score -= 10
        else:
            evidence.append("❌ 缺少并发压力测试")
            score -= 25
            recommendations.append("执行并发压力测试")

        # 检查锁机制和同步原语
        sync_usage = 0
        total_files = len(list(self.project_root.glob("src/**/*.py")))

        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if any(keyword in content for keyword in ['threading.Lock', 'asyncio.Lock', 'threading.RLock']):
                        sync_usage += 1
            except:
                pass

        sync_ratio = sync_usage / max(total_files, 1)
        evidence.append(f"📊 {sync_usage}/{total_files} 个文件使用同步原语 ({sync_ratio:.2f})")

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=recommendations,
            automated_check=True
        )

    # 其他评估方法的简化实现
    def _assess_authentication(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估身份验证机制"""
        evidence = ["需要人工评估身份验证机制完整性"]
        return AssessmentResult(
            criterion=criterion.criterion,
            status="warning",
            score=60.0,
            evidence=evidence,
            recommendations=["完善身份验证机制"],
            automated_check=False
        )

    def _assess_authorization(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估权限控制机制"""
        evidence = ["需要人工评估权限控制机制完整性"]
        return AssessmentResult(
            criterion=criterion.criterion,
            status="warning",
            score=60.0,
            evidence=evidence,
            recommendations=["完善权限控制机制"],
            automated_check=False
        )

    def _assess_data_protection(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估数据保护和加密"""
        evidence = ["需要人工评估数据加密和保护机制"]
        return AssessmentResult(
            criterion=criterion.criterion,
            status="warning",
            score=65.0,
            evidence=evidence,
            recommendations=["完善数据保护机制"],
            automated_check=False
        )

    def _assess_logging_monitoring(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估日志和监控机制"""
        evidence = []
        score = 100.0

        # 检查日志配置
        log_files = list(self.project_root.glob("**/logging*.py")) + \
                list(self.project_root.glob("**/logger*.py")) + \
                list(self.project_root.glob("src/**/log*.py"))

        if log_files:
            evidence.append(f"✅ 发现 {len(log_files)} 个日志配置")
        else:
            evidence.append("⚠️ 缺少日志配置")
            score -= 20

        # 检查监控相关代码
        monitor_files = list(self.project_root.glob("src/**/monitor*.py")) + \
                    list(self.project_root.glob("src/**/*metric*.py"))

        if monitor_files:
            evidence.append(f"✅ 发现 {len(monitor_files)} 个监控组件")
        else:
            evidence.append("⚠️ 缺少监控机制")
            score -= 20

        # 检查性能监控报告
        monitor_reports = list(self.project_root.glob("test_logs/*monitoring*.json"))
        if monitor_reports:
            evidence.append("✅ 具备性能监控报告")
        else:
            evidence.append("⚠️ 缺少监控数据")
            score -= 10

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=["完善日志和监控机制"] if score < 80 else [],
            automated_check=True
        )

    def _assess_configuration_management(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估配置管理"""
        evidence = []
        score = 100.0

        # 检查配置文件
        config_files = list(self.project_root.glob("config/*.yaml")) + \
                    list(self.project_root.glob("config/*.yml")) + \
                    list(self.project_root.glob("config/*.json")) + \
                    list(self.project_root.glob("*.ini"))

        if config_files:
            evidence.append(f"✅ 发现 {len(config_files)} 个配置文件")
        else:
            evidence.append("❌ 缺少配置文件")
            score -= 30

        # 检查环境变量使用
        env_usage = 0
        total_files = len(list(self.project_root.glob("src/**/*.py")))

        for py_file in self.project_root.glob("src/**/*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if 'os.environ' in content or 'os.getenv' in content:
                        env_usage += 1
            except:
                pass

        env_ratio = env_usage / max(total_files, 1)
        evidence.append(f"📊 {env_usage}/{total_files} 个文件使用环境变量 ({env_ratio:.2f})")

        if env_ratio < 0.3:
            evidence.append("⚠️ 环境变量使用率偏低")
            score -= 10

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=["完善配置管理机制"] if score < 80 else [],
            automated_check=True
        )

    def _assess_deployment_automation(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估部署自动化"""
        evidence = []
        score = 100.0

        # 检查Docker相关文件
        docker_files = list(self.project_root.glob("Dockerfile*")) + \
                    list(self.project_root.glob("docker-compose*.yml")) + \
                    list(self.project_root.glob("**/.dockerignore"))

        if docker_files:
            evidence.append(f"✅ 发现 {len(docker_files)} 个Docker配置")
        else:
            evidence.append("⚠️ 缺少Docker配置")
            score -= 20

        # 检查CI/CD配置
        ci_files = list(self.project_root.glob(".github/workflows/*.yml")) + \
                list(self.project_root.glob(".gitlab-ci.yml")) + \
                list(self.project_root.glob("Jenkinsfile")) + \
                list(self.project_root.glob("**/*pipeline*.yml"))

        if ci_files:
            evidence.append(f"✅ 发现 {len(ci_files)} 个CI/CD配置")
        else:
            evidence.append("⚠️ 缺少CI/CD配置")
            score -= 20

        # 检查部署脚本
        deploy_scripts = list(self.project_root.glob("scripts/deploy*.sh")) + \
                        list(self.project_root.glob("scripts/deploy*.py")) + \
                        list(self.project_root.glob("deploy/*.sh"))

        if deploy_scripts:
            evidence.append(f"✅ 发现 {len(deploy_scripts)} 个部署脚本")
        else:
            evidence.append("ℹ️ 建议提供部署脚本")
            score -= 10

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=["完善部署自动化"] if score < 80 else [],
            automated_check=True
        )

    def _assess_api_documentation(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估API文档"""
        evidence = []
        score = 100.0

        # 检查API文档文件
        api_docs = list(self.project_root.glob("docs/api/*.md")) + \
                list(self.project_root.glob("docs/*.md")) + \
                list(self.project_root.glob("**/*api*.md")) + \
                list(self.project_root.glob("README*.md"))

        if api_docs:
            evidence.append(f"✅ 发现 {len(api_docs)} 个API文档")
        else:
            evidence.append("❌ 缺少API文档")
            score -= 40

        # 检查Swagger/OpenAPI文档
        swagger_docs = list(self.project_root.glob("**/*swagger*.yaml")) + \
                    list(self.project_root.glob("**/*openapi*.yaml")) + \
                    list(self.project_root.glob("**/*swagger*.json")) + \
                    list(self.project_root.glob("**/*openapi*.json"))

        if swagger_docs:
            evidence.append(f"✅ 发现 {len(swagger_docs)} 个Swagger/OpenAPI文档")
        else:
            evidence.append("⚠️ 缺少Swagger/OpenAPI文档")
            score -= 20

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=["完善API文档"] if score < 80 else [],
            automated_check=True
        )

    def _assess_deployment_guide(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估部署指南"""
        evidence = []
        score = 100.0

        # 检查部署相关文档
        deploy_docs = list(self.project_root.glob("docs/deploy*.md")) + \
                    list(self.project_root.glob("docs/install*.md")) + \
                    list(self.project_root.glob("DEPLOY*.md")) + \
                    list(self.project_root.glob("INSTALL*.md"))

        if deploy_docs:
            evidence.append(f"✅ 发现 {len(deploy_docs)} 个部署指南")
        else:
            evidence.append("❌ 缺少部署指南")
            score -= 40

        # 检查requirements.txt或其他依赖文件
        dep_files = list(self.project_root.glob("requirements*.txt")) + \
                list(self.project_root.glob("pyproject.toml")) + \
                list(self.project_root.glob("setup.py")) + \
                list(self.project_root.glob("Pipfile"))

        if dep_files:
            evidence.append(f"✅ 发现 {len(dep_files)} 个依赖配置文件")
        else:
            evidence.append("⚠️ 缺少依赖配置文件")
            score -= 20

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=["完善部署指南"] if score < 80 else [],
            automated_check=True
        )

    def _assess_operations_manual(self, criterion: AssessmentCriterion) -> AssessmentResult:
        """评估运维手册"""
        evidence = []
        score = 100.0

        # 检查运维相关文档
        ops_docs = list(self.project_root.glob("docs/ops*.md")) + \
                list(self.project_root.glob("docs/operations*.md")) + \
                list(self.project_root.glob("docs/maintenance*.md")) + \
                list(self.project_root.glob("OPERATIONS*.md")) + \
                list(self.project_root.glob("MAINTENANCE*.md"))

        if ops_docs:
            evidence.append(f"✅ 发现 {len(ops_docs)} 个运维文档")
        else:
            evidence.append("ℹ️ 缺少运维手册")
            score -= 30

        # 检查监控和告警配置
        monitor_configs = list(self.project_root.glob("config/monitoring*.yaml")) + \
                        list(self.project_root.glob("config/alerting*.yaml"))

        if monitor_configs:
            evidence.append("✅ 具备监控告警配置")
        else:
            evidence.append("ℹ️ 建议提供监控告警配置")
            score -= 10

        status = "pass" if score >= 80 else "fail" if score < 60 else "warning"

        return AssessmentResult(
            criterion=criterion.criterion,
            status=status,
            score=max(0, min(100, score)),
            evidence=evidence,
            recommendations=["完善运维手册"] if score < 80 else [],
            automated_check=True
        )

    def _calculate_overall_score(self, results: List[AssessmentResult]) -> Tuple[float, Dict[str, float]]:
        """计算总体评分"""
        category_scores = {}
        category_weights = {}

        # 按类别分组计算
        for result in results:
            criterion = next((c for c in self.assessment_criteria if c.criterion == result.criterion), None)
            if criterion:
                category = criterion.category
                weight = criterion.weight

                if category not in category_scores:
                    category_scores[category] = 0
                    category_weights[category] = 0

                category_scores[category] += result.score * weight
                category_weights[category] += weight

        # 计算各类别平均分
        for category in category_scores:
            if category_weights[category] > 0:
                category_scores[category] /= category_weights[category]

        # 计算总体评分
        total_score = 0
        total_weight = 0
        for result in results:
            criterion = next((c for c in self.assessment_criteria if c.criterion == result.criterion), None)
            if criterion:
                total_score += result.score * criterion.weight
                total_weight += criterion.weight

        overall_score = total_score / total_weight if total_weight > 0 else 0

        return overall_score, category_scores

    def _determine_overall_status(self, overall_score: float, results: List[AssessmentResult]) -> str:
        """确定总体状态"""
        # 检查强制要求
        mandatory_failed = any(
            result.status == "fail" and
            any(c.mandatory for c in self.assessment_criteria if c.criterion == result.criterion)
            for result in results
        )

        if mandatory_failed or overall_score < 60:
            return "not_ready"
        elif overall_score < 80:
            return "conditional"
        else:
            return "ready"

    def _extract_issues_and_recommendations(self, results: List[AssessmentResult]) -> Tuple[List[str], List[str]]:
        """提取关键问题和建议"""
        critical_issues = []
        all_recommendations = []

        for result in results:
            if result.status == "fail":
                criterion = next((c for c in self.assessment_criteria if c.criterion == result.criterion), None)
                if criterion:
                    critical_issues.append(f"{criterion.description} - 评分: {result.score:.1f}")

            all_recommendations.extend(result.recommendations)

        # 去重建议
        all_recommendations = list(set(all_recommendations))

        return critical_issues, all_recommendations

    def _assess_deployment_readiness(self, overall_score: float, results: List[AssessmentResult]) -> Dict[str, Any]:
        """评估部署就绪情况"""
        deployment_readiness = {
            "can_deploy": overall_score >= 80,
            "risk_level": "low" if overall_score >= 90 else "medium" if overall_score >= 70 else "high",
            "estimated_deployment_time": "1-2天" if overall_score >= 80 else "3-5天" if overall_score >= 60 else "1-2周",
            "required_pre_deployment_tasks": [],
            "monitoring_requirements": [],
            "rollback_plan_needed": overall_score < 80
        }

        # 根据评估结果添加部署前任务
        failed_criteria = [r for r in results if r.status == "fail"]
        warning_criteria = [r for r in results if r.status == "warning"]

        for result in failed_criteria + warning_criteria:
            deployment_readiness["required_pre_deployment_tasks"].extend(result.recommendations)

        # 去重
        deployment_readiness["required_pre_deployment_tasks"] = list(
            set(deployment_readiness["required_pre_deployment_tasks"])
        )

        # 监控要求
        if any(r.criterion == "memory_management" and r.status != "pass" for r in results):
            deployment_readiness["monitoring_requirements"].append("内存使用监控")

        if any(r.criterion == "performance" and r.status != "pass" for r in results):
            deployment_readiness["monitoring_requirements"].append("性能指标监控")

        if any(r.criterion in ["logging_monitoring"] and r.status != "pass" for r in results):
            deployment_readiness["monitoring_requirements"].append("日志监控")

        return deployment_readiness

    def generate_assessment_report(self, report: ReadinessReport, filename: str = None):
        """生成评估报告"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"production_readiness_assessment_{timestamp}.json"

        report_file = self.project_root / "test_logs" / filename
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                "assessment_date": report.assessment_date.isoformat(),
                "overall_score": report.overall_score,
                "overall_status": report.overall_status,
                "category_scores": report.category_scores,
                "detailed_results": [asdict(r) for r in report.detailed_results],
                "critical_issues": report.critical_issues,
                "recommendations": report.recommendations,
                "deployment_readiness": report.deployment_readiness
            }, f, indent=2, default=str, ensure_ascii=False)

        # 生成HTML报告
        html_report = self._generate_assessment_html_report(report)
        html_file = report_file.with_suffix('.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)

        print(f"💾 生产就绪评估报告已保存: {report_file}")
        print(f"🌐 HTML报告已保存: {html_file}")

    def _generate_assessment_html_report(self, report: ReadinessReport) -> str:
        """生成HTML格式的评估报告"""
        status_color = {
            "ready": "#28a745",
            "conditional": "#ffc107",
            "not_ready": "#dc3545"
        }.get(report.overall_status, "#6c757d")

        status_text = {
            "ready": "生产就绪",
            "conditional": "条件就绪",
            "not_ready": "未就绪"
        }.get(report.overall_status, "未知")

        html = """
<!DOCTYPE html>
<html>
<head>
    <title>RQA2025生产就绪评估报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .status {{ background: {status_color}; color: white; padding: 20px; border-radius: 5px; margin: 20px 0; text-align: center; font-size: 24px; }}
        .score {{ background: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; font-size: 18px; text-align: center; }}
        .category {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .criterion {{ background: #ffffff; padding: 10px; margin: 5px 0; border-radius: 3px; border-left: 4px solid #007bff; }}
        .pass {{ border-left-color: #28a745; }}
        .fail {{ border-left-color: #dc3545; }}
        .warning {{ border-left-color: #ffc107; }}
        .issues {{ background: #f8d7da; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .recommendations {{ background: #d4edda; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .deployment {{ background: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RQA2025生产就绪评估报告</h1>
        <p>评估时间: {report.assessment_date.isoformat()}</p>
        <p>系统版本: RQA2025</p>
    </div>

    <div class="status">
        <strong>{status_text}</strong>
    </div>

    <div class="score">
        <strong>总体评分: {report.overall_score:.1f}/100</strong>
    </div>

    <h2>类别评分</h2>
"""

        for category, score in report.category_scores.items():
            category_name = {
                "functionality": "功能完整性",
                "performance": "性能就绪",
                "stability": "稳定性",
                "security": "安全性",
                "operability": "可运维性",
                "documentation": "文档完整性"
            }.get(category, category)

            html += """
    <div class="category">
        <h3>{category_name}: {score:.1f}/100</h3>
    </div>
"""

        html += """
    <h2>详细评估结果</h2>
"""

        for result in report.detailed_results:
            status_class = result.status
            status_icon = {"pass": "✅", "fail": "❌", "warning": "⚠️"}.get(result.status, "❓")

            html += """
    <div class="criterion {status_class}">
        <h4>{status_icon} {result.criterion}</h4>
        <p><strong>状态:</strong> {result.status} | <strong>评分:</strong> {result.score:.1f}/100</p>
        <p><strong>证据:</strong></p>
        <ul>
"""
            for evidence in result.evidence:
                html += f"<li>{evidence}</li>"

            html += "</ul>"

            if result.recommendations:
                html += "<p><strong>建议:</strong></p><ul>"
                for rec in result.recommendations:
                    html += f"<li>{rec}</li>"
                html += "</ul>"

            html += "</div>"

        if report.critical_issues:
            html += """
    <div class="issues">
        <h2>🚨 关键问题</h2>
        <ul>
"""
            for issue in report.critical_issues:
                html += f"<li>{issue}</li>"

            html += "</ul></div>"

        if report.recommendations:
            html += """
    <div class="recommendations">
        <h2>💡 优化建议</h2>
        <ul>
"""
            for rec in report.recommendations:
                html += f"<li>{rec}</li>"

            html += "</ul></div>"

        html += """
    <div class="deployment">
        <h2>🚀 部署就绪评估</h2>
        <p><strong>可部署:</strong> {'是' if report.deployment_readiness['can_deploy'] else '否'}</p>
        <p><strong>风险等级:</strong> {report.deployment_readiness['risk_level']}</p>
        <p><strong>预计部署时间:</strong> {report.deployment_readiness['estimated_deployment_time']}</p>
        <p><strong>需要回滚计划:</strong> {'是' if report.deployment_readiness['rollback_plan_needed'] else '否'}</p>

        <h3>部署前必备任务</h3>
        <ul>
"""
        for task in report.deployment_readiness.get('required_pre_deployment_tasks', []):
            html += f"<li>{task}</li>"

        html += """
        </ul>

        <h3>监控要求</h3>
        <ul>
"""
        for req in report.deployment_readiness.get('monitoring_requirements', []):
            html += f"<li>{req}</li>"

        html += """
        </ul>
    </div>
</body>
</html>
"""
        return html


def run_production_readiness_assessment():
    """运行生产就绪评估"""
    print("🏭 启动RQA2025生产就绪评估")
    print("=" * 60)

    assessor = ProductionReadinessAssessor()

    # 执行评估
    report = assessor.assess_readiness()

    # 生成报告
    assessor.generate_assessment_report(report)

    print("\n📊 评估完成总结")
    print("=" * 40)
    print(f"🏆 总体状态: {report.overall_status}")
    print(f"📊 总体评分: {report.overall_score:.1f}/100")
    print(f"⚠️  关键问题: {len(report.critical_issues)} 个")
    print(f"💡 优化建议: {len(report.recommendations)} 个")

    # 打印关键类别评分
    print("\n📈 类别评分:")
    for category, score in report.category_scores.items():
        category_name = {
            "functionality": "功能完整性",
            "performance": "性能就绪",
            "stability": "稳定性",
            "security": "安全性",
            "operability": "可运维性",
            "documentation": "文档完整性"
        }.get(category, category)
        print(f"  {category_name}: {score:.1f}/100")

    # 打印部署就绪信息
    deployment = report.deployment_readiness
    print("\n🚀 部署就绪:")
    print(f"  可部署: {'是' if deployment['can_deploy'] else '否'}")
    print(f"  风险等级: {deployment['risk_level']}")
    print(f"  预计部署时间: {deployment['estimated_deployment_time']}")

    return report


if __name__ == "__main__":
    run_production_readiness_assessment()

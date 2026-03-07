#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试覆盖率强制要求机制
确保RQA2025量化交易系统达到投产标准
"""

import argparse
import json
import sys
import os
from typing import Dict, Any
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoverageEnforcer:
    """测试覆盖率强制要求执行器"""

    def __init__(self, minimum_coverage: float = 80.0, target_coverage: float = 90.0,
                 critical_modules_coverage: float = 95.0):
        self.minimum_coverage = minimum_coverage
        self.target_coverage = target_coverage
        self.critical_modules_coverage = critical_modules_coverage

        # 关键模块定义
        self.critical_modules = [
            'src.core',
            'src.trading',
            'src.risk_management',
            'src.data_management',
            'src.strategies'
        ]

        # 核心业务模块
        self.core_business_modules = [
            'src.core.event_bus',
            'src.core.container',
            'src.core.business_process_orchestrator',
            'src.trading.order_management',
            'src.trading.execution_engine',
            'src.risk_management.risk_calculator'
        ]

    def load_coverage_data(self, coverage_file: str) -> Dict[str, Any]:
        """加载覆盖率数据"""
        try:
            with open(coverage_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"✅ 成功加载覆盖率数据: {coverage_file}")
                return data
        except FileNotFoundError:
            logger.error(f"❌ 覆盖率文件未找到: {coverage_file}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"❌ 覆盖率文件格式错误: {e}")
            return {}

    def analyze_coverage(self, coverage_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析覆盖率数据"""
        if not coverage_data:
            return {
                'overall_coverage': 0.0,
                'module_coverage': {},
                'critical_modules_coverage': {},
                'status': 'FAILED',
                'recommendations': ['无法获取覆盖率数据']
            }

        # 获取总体覆盖率
        totals = coverage_data.get('totals', {})
        overall_coverage = totals.get('percent_covered', 0.0)

        # 分析模块级覆盖率
        files = coverage_data.get('files', {})
        module_coverage = self._calculate_module_coverage(files)

        # 分析关键模块覆盖率
        critical_coverage = self._calculate_critical_module_coverage(files)

        # 生成状态和建议
        status, recommendations = self._generate_status_and_recommendations(
            overall_coverage, critical_coverage
        )

        return {
            'overall_coverage': round(overall_coverage, 2),
            'module_coverage': module_coverage,
            'critical_modules_coverage': critical_coverage,
            'status': status,
            'recommendations': recommendations,
            'details': {
                'minimum_required': self.minimum_coverage,
                'target_coverage': self.target_coverage,
                'critical_required': self.critical_modules_coverage
            }
        }

    def _calculate_module_coverage(self, files: Dict[str, Any]) -> Dict[str, float]:
        """计算模块级覆盖率"""
        module_stats = {}

        for file_path, file_data in files.items():
            # 提取模块名
            if file_path.startswith('src/'):
                parts = file_path.split('/')
                if len(parts) >= 2:
                    module = f"src.{parts[1]}"

                    if module not in module_stats:
                        module_stats[module] = {
                            'total_statements': 0,
                            'covered_statements': 0
                        }

                    # 累计统计
                    summary = file_data.get('summary', {})
                    total = summary.get('num_statements', 0)
                    covered = total - summary.get('missing_lines', 0)

                    module_stats[module]['total_statements'] += total
                    module_stats[module]['covered_statements'] += covered

        # 计算每个模块的覆盖率
        module_coverage = {}
        for module, stats in module_stats.items():
            if stats['total_statements'] > 0:
                coverage = (stats['covered_statements'] / stats['total_statements']) * 100
                module_coverage[module] = round(coverage, 2)
            else:
                module_coverage[module] = 0.0

        return module_coverage

    def _calculate_critical_module_coverage(self, files: Dict[str, Any]) -> Dict[str, float]:
        """计算关键模块覆盖率"""
        critical_coverage = {}

        for module in self.critical_modules:
            module_files = {k: v for k, v in files.items()
                            if k.startswith(module.replace('.', '/'))}

            total_statements = 0
            covered_statements = 0

            for file_data in module_files.values():
                summary = file_data.get('summary', {})
                total = summary.get('num_statements', 0)
                covered = total - summary.get('missing_lines', 0)

                total_statements += total
                covered_statements += covered

            if total_statements > 0:
                coverage = (covered_statements / total_statements) * 100
                critical_coverage[module] = round(coverage, 2)
            else:
                critical_coverage[module] = 0.0

        return critical_coverage

    def _generate_status_and_recommendations(self, overall_coverage: float,
                                             critical_coverage: Dict[str, float]) -> tuple:
        """生成状态和建议"""
        recommendations = []

        # 检查整体覆盖率
        if overall_coverage < self.minimum_coverage:
            status = 'FAILED'
            recommendations.append(
                f'整体覆盖率({overall_coverage:.1f}%)低于最低要求({self.minimum_coverage}%)')
        elif overall_coverage < self.target_coverage:
            status = 'WARNING'
            recommendations.append(
                f'整体覆盖率({overall_coverage:.1f}%)达到最低要求但低于目标({self.target_coverage}%)')
        else:
            status = 'PASSED'
            recommendations.append(f'整体覆盖率({overall_coverage:.1f}%)达到目标要求')

        # 检查关键模块覆盖率
        for module, coverage in critical_coverage.items():
            if coverage < self.critical_modules_coverage:
                recommendations.append(
                    f'关键模块{module}覆盖率({coverage:.1f}%)低于要求({self.critical_modules_coverage}%)')

        # 生成改进建议
        if overall_coverage < self.target_coverage:
            recommendations.extend([
                '建议优先提升核心业务模块测试覆盖',
                '增加边界条件和异常情况测试',
                '完善集成测试和端到端测试'
            ])

        return status, recommendations

    def enforce_coverage_requirements(self, analysis_result: Dict[str, Any]) -> bool:
        """强制执行覆盖率要求"""
        overall_coverage = analysis_result['overall_coverage']
        critical_coverage = analysis_result['critical_modules_coverage']
        status = analysis_result['status']

        logger.info("🎯 执行覆盖率强制要求检查")
        logger.info(f"📊 整体覆盖率: {overall_coverage}%")
        logger.info(f"🎯 最低要求: {self.minimum_coverage}%")
        logger.info(f"🏆 目标覆盖率: {self.target_coverage}%")

        # 检查关键模块
        critical_failed = []
        for module, coverage in critical_coverage.items():
            if coverage < self.critical_modules_coverage:
                critical_failed.append(f"{module}({coverage:.1f}%)")

        if critical_failed:
            logger.error(f"❌ 关键模块覆盖率不足: {', '.join(critical_failed)}")

        # 最终判定
        if status == 'FAILED':
            logger.error("❌ 覆盖率强制要求检查失败")
            logger.error("🚫 不允许部署到生产环境")
            return False
        elif status == 'WARNING':
            logger.warning("⚠️ 覆盖率强制要求检查通过(警告)")
            logger.warning("✅ 允许部署但建议提升覆盖率")
            return True
        else:
            logger.info("✅ 覆盖率强制要求检查通过")
            logger.info("🎉 准备就绪，可以部署到生产环境")
            return True

    def generate_enforcement_report(self, analysis_result: Dict[str, Any],
                                    output_file: str) -> None:
        """生成强制要求报告"""
        report = {
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else str(datetime.now()),
            'enforcement_result': analysis_result,
            'quality_gate': {
                'minimum_coverage': self.minimum_coverage,
                'target_coverage': self.target_coverage,
                'critical_modules_coverage': self.critical_modules_coverage
            },
            'critical_modules': self.critical_modules,
            'deployment_decision': 'APPROVED' if analysis_result['status'] != 'FAILED' else 'BLOCKED'
        }

        # 保存报告
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"📄 强制要求报告已生成: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='RQA2025测试覆盖率强制要求机制')
    parser.add_argument('--minimum-coverage', type=float, default=80.0,
                        help='最低覆盖率要求 (默认: 80%)')
    parser.add_argument('--target-coverage', type=float, default=90.0,
                        help='目标覆盖率 (默认: 90%)')
    parser.add_argument('--critical-modules-coverage', type=float, default=95.0,
                        help='关键模块覆盖率要求 (默认: 95%)')
    parser.add_argument('--coverage-file', required=True,
                        help='覆盖率数据文件路径 (JSON格式)')
    parser.add_argument('--output-report', required=True,
                        help='输出报告文件路径')

    args = parser.parse_args()

    # 创建强制要求执行器
    enforcer = CoverageEnforcer(
        minimum_coverage=args.minimum_coverage,
        target_coverage=args.target_coverage,
        critical_modules_coverage=args.critical_modules_coverage
    )

    # 加载和分析覆盖率数据
    coverage_data = enforcer.load_coverage_data(args.coverage_file)
    analysis_result = enforcer.analyze_coverage(coverage_data)

    # 执行强制要求检查
    passed = enforcer.enforce_coverage_requirements(analysis_result)

    # 生成报告
    enforcer.generate_enforcement_report(analysis_result, args.output_report)

    # 输出结果
    if passed:
        print("✅ 覆盖率强制要求检查通过")
        sys.exit(0)
    else:
        print("❌ 覆盖率强制要求检查失败")
        sys.exit(1)


if __name__ == "__main__":
    main()

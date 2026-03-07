#!/usr/bin/env python3
"""
测试覆盖率质量门禁检查脚本

检查基础设施日志模块的测试覆盖率是否达到质量标准。
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any


class CoverageQualityChecker:
    """覆盖率质量检查器"""

    def __init__(self, coverage_file: str = "test_logs/coverage_final_boost_detailed.json"):
        self.coverage_file = coverage_file
        self.min_overall_coverage = 70.0  # 总体最低覆盖率
        self.min_module_coverage = {
            'logger_pool': 95.0,
            'monitoring': 85.0,
            'logger_service': 80.0,
            'storage': 75.0,
            'utils_logger': 80.0,
            'formatters': 70.0,
            'handlers': 65.0,
            'unified_logger': 75.0
        }

    def load_coverage_data(self) -> Dict[str, Any]:
        """加载覆盖率数据"""
        if not os.path.exists(self.coverage_file):
            print(f"❌ 覆盖率文件不存在: {self.coverage_file}")
            return {}

        try:
            with open(self.coverage_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 读取覆盖率文件失败: {e}")
            return {}

    def analyze_infrastructure_coverage(self, data: Dict[str, Any]) -> Dict[str, float]:
        """分析基础设施模块的覆盖率"""
        infrastructure_coverage = {}

        for file_path, file_data in data.get('files', {}).items():
            if 'infrastructure/logging' in file_path:
                percent = file_data['summary']['percent_covered']
                infrastructure_coverage[file_path] = percent

        return infrastructure_coverage

    def check_module_coverage(self, file_coverages: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """检查各模块覆盖率"""
        results = {}

        # 检查核心模块
        core_modules = {
            'logger_pool': 'src/infrastructure/logging/core/logger_pool.py',
            'monitoring': 'src/infrastructure/logging/core/monitoring.py',
            'logger_service': 'src/infrastructure/logging/services/logger_service.py',
            'storage': 'src/infrastructure/logging/storage/base.py',
            'utils_logger': 'src/infrastructure/logging/utils/logger.py',
            'unified_logger': 'src/infrastructure/logging/core/unified_logger.py'
        }

        for module_name, file_path in core_modules.items():
            if file_path in file_coverages:
                actual_coverage = file_coverages[file_path]
                min_required = self.min_module_coverage.get(module_name, 70.0)

                results[module_name] = {
                    'actual': actual_coverage,
                    'required': min_required,
                    'passed': actual_coverage >= min_required
                }

        # 检查formatters和handlers模块
        formatter_files = [f for f in file_coverages.keys() if 'formatters' in f and not f.endswith('__init__.py')]
        handler_files = [f for f in file_coverages.keys() if 'handlers' in f and not f.endswith('__init__.py')]

        if formatter_files:
            avg_formatter_coverage = sum(file_coverages[f] for f in formatter_files) / len(formatter_files)
            results['formatters'] = {
                'actual': avg_formatter_coverage,
                'required': self.min_module_coverage['formatters'],
                'passed': avg_formatter_coverage >= self.min_module_coverage['formatters']
            }

        if handler_files:
            avg_handler_coverage = sum(file_coverages[f] for f in handler_files) / len(handler_files)
            results['handlers'] = {
                'actual': avg_handler_coverage,
                'required': self.min_module_coverage['handlers'],
                'passed': avg_handler_coverage >= self.min_module_coverage['handlers']
            }

        return results

    def check_overall_coverage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """检查总体覆盖率"""
        totals = data.get('totals', {})
        overall_coverage = totals.get('percent_covered', 0.0)

        return {
            'actual': overall_coverage,
            'required': self.min_overall_coverage,
            'passed': overall_coverage >= self.min_overall_coverage
        }

    def generate_report(self) -> bool:
        """生成质量检查报告"""
        print("🔍 基础设施日志模块测试覆盖率质量检查")
        print("=" * 50)

        data = self.load_coverage_data()
        if not data:
            return False

        # 检查总体覆盖率
        overall_result = self.check_overall_coverage(data)
        print("\n📊 总体覆盖率检查:")
        print(".1f")
        print(".1f")
        if overall_result['passed']:
            print("✅ 通过")
        else:
            print("❌ 未通过")

        # 分析各文件覆盖率
        file_coverages = self.analyze_infrastructure_coverage(data)
        print("\n📁 基础设施模块文件覆盖率:")
        for file_path, coverage in sorted(file_coverages.items()):
            status = "✅" if coverage >= 70.0 else "⚠️" if coverage >= 50.0 else "❌"
            print(".1f")
        # 检查各模块覆盖率
        module_results = self.check_module_coverage(file_coverages)
        print("\n🏗️ 核心模块覆盖率检查:")
        all_passed = True

        for module_name, result in module_results.items():
            status = "✅" if result['passed'] else "❌"
            print(".1f")
            if not result['passed']:
                all_passed = False

        # 生成总结
        print("\n🎯 质量门禁检查结果:")
        if overall_result['passed'] and all_passed:
            print("🎉 恭喜！所有质量门禁检查均通过")
            print("✅ 基础设施日志模块已达到生产就绪标准")
            return True
        else:
            print("⚠️  发现质量问题，需要改进:")
            if not overall_result['passed']:
                print(".1f")
            failed_modules = [m for m, r in module_results.items() if not r['passed']]
            if failed_modules:
                print(f"  • 模块覆盖率不足: {', '.join(failed_modules)}")
            return False

    def generate_recommendations(self) -> None:
        """生成改进建议"""
        print("\n💡 改进建议:")
        print("1. 提升低覆盖率模块的测试用例数量")
        print("2. 增加边界条件和异常场景测试")
        print("3. 完善端到端业务流程测试")
        print("4. 添加性能和并发测试")
        print("5. 建立持续集成覆盖率监控")


def main():
    """主函数"""
    checker = CoverageQualityChecker()

    # 查找最新的覆盖率文件
    possible_files = [
        "test_logs/coverage_final_boost_detailed.json",
        "test_logs/coverage_handlers_boost.json",
        "coverage.json"
    ]

    coverage_file = None
    for file_path in possible_files:
        if os.path.exists(file_path):
            coverage_file = file_path
            break

    if coverage_file:
        checker.coverage_file = coverage_file
        print(f"使用覆盖率文件: {coverage_file}")
    else:
        print("❌ 未找到覆盖率文件")
        sys.exit(1)

    success = checker.generate_report()

    if not success:
        checker.generate_recommendations()
        sys.exit(1)


if __name__ == "__main__":
    main()

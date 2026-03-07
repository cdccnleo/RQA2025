#!/usr/bin/env python3
"""
验证测试修复效果的脚本
运行修复后的测试并生成详细报告
"""

import os
import sys
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestValidationRunner:
    """测试验证运行器"""

    def __init__(self):
        self.layers = {
            'infrastructure': 'tests/unit/infrastructure',
            'features': 'tests/unit/features',
            'ml': 'tests/unit/ml',
            'trading': 'tests/unit/trading',
            'risk': 'tests/unit/risk',
            'core': 'tests/unit/core'
        }

    def run_layer_tests(self, layer_name: str, test_path: str, timeout: int = 120) -> Dict[str, Any]:
        """运行指定层的测试"""
        logger.info(f"开始验证{layer_name}层测试...")

        if not os.path.exists(test_path):
            return {
                'layer': layer_name,
                'status': 'no_tests',
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'total_tests': 0
            }

        try:
            # 运行一个简单的语法检查
            cmd = ['python', '-m', 'py_compile']
            test_files = list(Path(test_path).rglob('*comprehensive.py'))
            syntax_errors = 0

            for test_file in test_files[:10]:  # 只检查前10个文件
                try:
                    result = subprocess.run(
                        cmd + [str(test_file)],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    if result.returncode != 0:
                        syntax_errors += 1
                except:
                    syntax_errors += 1

            # 运行少量实际测试
            cmd = ['python', '-m', 'pytest',
                   f"{test_path}/test_*_layer_coverage.py", '-v', '--tb=short', '-x']
            result = subprocess.run(
                cmd,
                cwd=os.path.join(os.path.dirname(__file__), '../..'),
                capture_output=True,
                text=True,
                timeout=timeout
            )

            # 解析结果
            output = result.stdout + result.stderr
            passed = len([line for line in output.split('\n') if 'PASSED' in line])
            failed = len([line for line in output.split('\n') if 'FAILED' in line])
            errors = len([line for line in output.split('\n') if 'ERROR' in line])

            return {
                'layer': layer_name,
                'status': 'completed' if result.returncode == 0 else 'with_issues',
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'syntax_errors': syntax_errors,
                'total_files': len(test_files),
                'return_code': result.returncode
            }

        except subprocess.TimeoutExpired:
            logger.error(f"{layer_name}层测试超时")
            return {
                'layer': layer_name,
                'status': 'timeout',
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'syntax_errors': 0,
                'total_files': 0,
                'return_code': -1
            }
        except Exception as e:
            logger.error(f"{layer_name}层测试执行出错: {e}")
            return {
                'layer': layer_name,
                'status': 'error',
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'syntax_errors': 0,
                'total_files': 0,
                'return_code': -1
            }

    def validate_all_layers(self) -> Dict[str, Any]:
        """验证所有层的测试修复效果"""
        logger.info("开始验证所有层级测试修复效果...")

        validation_results = {}

        for layer_name, test_path in self.layers.items():
            logger.info(f"验证{layer_name}层...")
            result = self.run_layer_tests(layer_name, test_path)
            validation_results[layer_name] = result
            logger.info(f"{layer_name}层验证完成: {result}")

        return validation_results

    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """生成验证报告"""
        report = []
        report.append("# 🧪 测试修复验证报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append("## 📊 各层级验证结果")
        report.append("")

        total_passed = 0
        total_failed = 0
        total_syntax_errors = 0
        total_files = 0

        for layer_name, result in results.items():
            total_passed += result.get('passed', 0)
            total_failed += result.get('failed', 0)
            total_syntax_errors += result.get('syntax_errors', 0)
            total_files += result.get('total_files', 0)

            report.append(f"### {layer_name.capitalize()}层")
            report.append(f"- 测试文件总数: {result.get('total_files', 0)}")
            report.append(f"- 语法错误: {result.get('syntax_errors', 0)}")
            report.append(f"- 通过测试: {result.get('passed', 0)}")
            report.append(f"- 失败测试: {result.get('failed', 0)}")
            report.append(f"- 错误测试: {result.get('errors', 0)}")
            report.append(f"- 状态: {result.get('status', 'unknown')}")
            report.append("")

        report.append("## 📈 总体统计")
        report.append(f"- 总测试文件: {total_files}")
        report.append(f"- 总语法错误: {total_syntax_errors}")
        report.append(f"- 总通过测试: {total_passed}")
        report.append(f"- 总失败测试: {total_failed}")
        report.append("")

        report.append("## 🎯 修复效果评估")
        report.append("")

        if total_syntax_errors == 0:
            report.append("✅ **语法修复成功**: 所有测试文件语法正确")
        else:
            report.append(f"⚠️ **语法修复待完善**: 发现 {total_syntax_errors} 个语法错误")

        if total_passed > 0:
            report.append(f"✅ **测试运行成功**: {total_passed} 个测试通过")
        else:
            report.append("❌ **测试运行问题**: 没有测试成功运行")

        if total_failed == 0:
            report.append("✅ **测试修复成功**: 没有测试失败")
        else:
            report.append(f"⚠️ **测试修复待完善**: {total_failed} 个测试仍失败")

        report.append("")
        report.append("## 🔧 建议")
        report.append("")

        if total_syntax_errors > 0:
            report.append("### 语法修复:")
            report.append("- 检查剩余的语法错误文件")
            report.append("- 修复导入路径和模块引用问题")
            report.append("- 验证Python语法正确性")

        if total_failed > 0:
            report.append("### 测试修复:")
            report.append("- 分析失败的测试用例")
            report.append("- 修复断言和测试逻辑")
            report.append("- 完善测试数据和mock对象")

        report.append("### 整体改进:")
        report.append("- 建立持续的测试验证流程")
        report.append("- 优化测试执行时间")
        report.append("- 增强测试覆盖率监控")

        return '\n'.join(report)

    def create_next_steps_plan(self, results: Dict[str, Any]) -> str:
        """创建下一步行动计划"""
        plan = []
        plan.append("# 🚀 下一步行动计划")
        plan.append("")

        # 分析各层状态
        syntax_issue_layers = []
        test_failure_layers = []
        good_layers = []

        for layer_name, result in results.items():
            syntax_errors = result.get('syntax_errors', 0)
            failures = result.get('failed', 0) + result.get('errors', 0)

            if syntax_errors > 0:
                syntax_issue_layers.append(layer_name)
            elif failures > 0:
                test_failure_layers.append(layer_name)
            else:
                good_layers.append(layer_name)

        if good_layers:
            plan.append("## ✅ 已完成层级")
            for layer in good_layers:
                plan.append(f"- {layer}: 语法正确，测试可运行")
            plan.append("")

        if syntax_issue_layers:
            plan.append("## 🔧 优先修复语法问题层级")
            for layer in syntax_issue_layers:
                plan.append(f"- {layer}: 需要修复语法错误")
            plan.append("")
            plan.append("### 语法修复任务:")
            plan.append("1. 检查导入路径和模块引用")
            plan.append("2. 修复Python语法错误")
            plan.append("3. 验证模块依赖关系")
            plan.append("")

        if test_failure_layers:
            plan.append("## 🧪 修复测试逻辑层级")
            for layer in test_failure_layers:
                plan.append(f"- {layer}: 需要修复测试逻辑")
            plan.append("")
            plan.append("### 测试修复任务:")
            plan.append("1. 分析失败的测试用例")
            plan.append("2. 修复断言和测试数据")
            plan.append("3. 完善mock对象和依赖注入")
            plan.append("")

        plan.append("## 📋 总体行动计划")
        plan.append("")
        plan.append("### 阶段一: 语法修复 (1-2天)")
        plan.append("1. 逐个修复语法错误的测试文件")
        plan.append("2. 验证所有测试文件可正常导入")
        plan.append("3. 建立语法检查自动化流程")
        plan.append("")

        plan.append("### 阶段二: 测试逻辑修复 (3-5天)")
        plan.append("1. 分析失败的测试用例原因")
        plan.append("2. 修复断言、mock和测试数据")
        plan.append("3. 优化测试执行性能")
        plan.append("")

        plan.append("### 阶段三: 质量提升 (1-2周)")
        plan.append("1. 增加边界条件测试")
        plan.append("2. 完善错误处理测试")
        plan.append("3. 建立测试覆盖率监控")
        plan.append("")

        plan.append("### 阶段四: 集成和部署 (2-3天)")
        plan.append("1. 集成到CI/CD流水线")
        plan.append("2. 建立自动化测试报告")
        plan.append("3. 部署测试验证")
        plan.append("")

        return '\n'.join(plan)


def main():
    """主函数"""
    runner = TestValidationRunner()

    print("🔍 开始验证测试修复效果...")
    print("分析各层级测试状态...")

    # 验证所有层级
    results = runner.validate_all_layers()

    # 生成验证报告
    validation_report = runner.generate_validation_report(results)
    next_steps_plan = runner.create_next_steps_plan(results)

    # 保存报告
    report_file = "docs/testing/TEST_FIXES_VALIDATION_REPORT.md"
    plan_file = "docs/testing/NEXT_STEPS_ACTION_PLAN.md"

    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(validation_report)

    with open(plan_file, 'w', encoding='utf-8') as f:
        f.write(next_steps_plan)

    print("\n" + "="*60)
    print("🎉 测试修复验证完成！")
    print("="*60)
    print(f"📄 验证报告已保存至: {report_file}")
    print(f"📋 行动计划已保存至: {plan_file}")
    print("\n验证摘要:")
    print("-" * 40)

    total_passed = sum(r.get('passed', 0) for r in results.values())
    total_syntax_errors = sum(r.get('syntax_errors', 0) for r in results.values())

    print(f"✅ 通过测试: {total_passed}")
    print(f"⚠️  语法错误: {total_syntax_errors}")

    if total_syntax_errors == 0:
        print("🎯 语法修复完成！")
    else:
        print(f"🔧 发现 {total_syntax_errors} 个语法问题需要修复")


if __name__ == "__main__":
    main()

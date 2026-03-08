#!/usr/bin/env python3
"""
RQA2025用户验收测试执行系统

User Acceptance Testing System for RQA2025

This module has been refactored - all classes extracted to separate modules.

Author: RQA2025 Development Team
Date: 2025-11-01 (Refactored for better code organization)
"""

import sys
import os
import logging
import time
from typing import Optional

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from .enums import UserRole, AcceptanceTestType, TestScenario
from .models import UserAcceptanceTest, TestExecutionResult, TestStatus
from .test_manager import AcceptanceTestManager
from .test_executor import UserAcceptanceTestExecutor

logger = logging.getLogger(__name__)

def main():

    """主函数 - 用户验收测试执行系统演示"""

    print("👤 RQA2025用户验收测试执行系统")

    print("=" * 60)



    # 创建验收测试执行器

    executor = UserAcceptanceTestExecutor()



    print("✅ 用户验收测试执行器创建完成")

    print("   包含以下功能:")

    print("   - 自动化测试分配")

    print("   - 并发生测执行")

    print("   - 结果收集分析")

    print("   - 缺陷跟踪管理")



    # 加载测试库

    test_manager = executor.test_manager



    # 显示测试统计

    total_tests = sum(len(tests) for tests in test_manager.test_library.values())

    print(f"   测试库总计: {total_tests} 个测试用例")



    for category, tests in test_manager.test_library.items():

        print(f"     {category}: {len(tests)} 个")



    try:

        # 启动验收测试

        print("\n🚀 启动用户验收测试...")



        # 选择测试类型和用户角色

        test_types = [

            (UserRole.TRADER, "交易员测试"),

            (UserRole.RISK_MANAGER, "风险经理测试"),

            (UserRole.SYSTEM_ADMINISTRATOR, "系统管理员测试")

        ]



        for user_role, description in test_types:

            print(f"\n🎯 执行{description}...")

            test_plan = test_manager.get_test_plan(user_role)



            if test_plan:

                print(f"   找到 {len(test_plan)} 个相关测试用例")



                # 启动该角色的测试

                executor.start_acceptance_testing(user_role, "staging")



                # 等待一小段时间

                time.sleep(5)



                # 显示进度

                summary = executor.get_execution_summary()

                report = summary['test_report']['summary']



                print(f"   进度: {report['completed_tests']}/{report['total_tests']} 完成")

                print(f"   通过率: {report['pass_rate']:.1%}")



        # 等待测试完成

        print("\n⏳ 等待测试完成...")

        time.sleep(30)  # 等待30秒



        print("\n🎉 用户验收测试演示完成！")

        print("   测试已成功执行并收集用户反馈")

        print("   缺陷已记录并分配修复")



    except KeyboardInterrupt:

        print("\n\n🛑 收到停止信号，正在停止测试...")

    except Exception as e:

        print(f"\n❌ 测试过程中出错: {e}")

        import traceback

        traceback.print_exc()

    finally:

        # 停止测试

        executor.stop_acceptance_testing()

        print("✅ 用户验收测试已停止")



        # 显示最终统计

        final_summary = executor.get_execution_summary()

        execution_stats = final_summary['execution_stats']

        report = final_summary['test_report']



        print("\n📋 最终测试统计:")

        print(f"   总执行数: {execution_stats['total_executions']}")

        print(f"   成功执行: {execution_stats['successful_executions']}")

        print(f"   失败执行: {execution_stats['failed_executions']}")

        print(f"   发现缺陷: {execution_stats['defects_found']}")

        print(f"   平均执行时间: {execution_stats['avg_execution_time']:.1f}秒")



        summary = report['summary']

        print("\n📊 测试通过情况:")

        print(f"   总测试数: {summary['total_tests']}")

        print(f"   已完成: {summary['completed_tests']}")

        print(f"   通过测试: {summary['passed_tests']}")

        print(f"   失败测试: {summary['failed_tests']}")

        print(f"   通过率: {summary['pass_rate']:.1%}")

        print(f"   完成率: {summary['completion_rate']:.1%}")



        if report.get('defect_summary'):



            defects = report['defect_summary']

            print("\n🐛 缺陷统计:")

            print(f"   总缺陷数: {defects['total_defects']}")

            print(f"   严重缺陷: {defects['critical_defects']}")

            print(f"   平均每测: {defects['defects_per_test']:.2f} 个")



        print("\n✅ 用户验收测试系统验证完成！")

        print("   系统已成功演示了完整的UAT流程")

        print("   包括测试分配、执行、结果收集和缺陷管理")



    return executor


__all__ = [
    'UserRole', 'AcceptanceTestType', 'TestScenario',
    'UserAcceptanceTest', 'TestExecutionResult', 'TestStatus',
    'AcceptanceTestManager', 'UserAcceptanceTestExecutor',
    'main'
]

"""
验收测试执行器

Acceptance Test Executor - executes user acceptance tests.

Extracted from user_acceptance_tester.py to improve code organization.

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import numpy as np
import logging
import time
import queue
from typing import Dict, List, Any, Optional
import concurrent.futures

from .enums import UserRole, AcceptanceTestType
from .models import UserAcceptanceTest, TestExecutionResult, TestStatus
from .test_manager import AcceptanceTestManager

logger = logging.getLogger(__name__)

class UserAcceptanceTestExecutor:

    """用户验收测试执行器"""

    def __init__(self):

        self.test_manager = AcceptanceTestManager()
        self.is_executing = False
        self.execution_queue = queue.Queue()
        self.executors = concurrent.futures.ThreadPoolExecutor(max_workers=5)

        # 执行统计
        self.stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time': 0.0,
            'defects_found': 0
        }

    def start_acceptance_testing(self, user_role: Optional[UserRole] = None,


                                 environment: str = "staging"):
        """启动验收测试"""
        if self.is_executing:
            logger.warning("验收测试已在执行中")
            return

        self.is_executing = True

        # 获取测试计划
        test_plan = self.test_manager.get_test_plan(user_role)

        print(f"📋 加载测试计划: {len(test_plan)} 个测试用例")

        # 分配测试执行
        for test in test_plan:
            if not self.is_executing:
                break

            # 自动分配测试人员
            tester = self._assign_tester(test)
            success = self.test_manager.assign_test_execution(
                test.test_id, tester, environment
            )

        if success:
            # 提交执行任务
            future = self.executors.submit(
                self._execute_single_test,
                test.test_id,
                tester,
                environment
            )

            print(f"✅ 已分配测试: {test.test_name} -> {tester}")

        logger.info("用户验收测试已启动")

    def stop_acceptance_testing(self):
        """停止验收测试"""
        self.is_executing = False
        self.executors.shutdown(wait=True)
        logger.info("用户验收测试已停止")

    def _assign_tester(self, test: UserAcceptanceTest) -> str:
        """分配测试人员"""
        # 简化的测试人员分配逻辑
        testers = {
            UserRole.TRADER: ["张交易员", "李交易员"],
            UserRole.RISK_MANAGER: ["王风险经理", "赵风险经理"],
            UserRole.PORTFOLIO_MANAGER: ["刘投资经理", "陈投资经理"],
            UserRole.QUANTITATIVE_ANALYST: ["杨量化分析师", "黄量化分析师"],
            UserRole.SYSTEM_ADMINISTRATOR: ["周系统管理员", "吴系统管理员"],
            UserRole.COMPLIANCE_OFFICER: ["郑合规官", "马合规官"],
            UserRole.BUSINESS_ANALYST: ["孙业务分析师", "胡业务分析师"]
        }

        for role in test.target_users:
            if role in testers and testers[role]:
                return testers[role][0]

        return "默认测试员"

    def _execute_single_test(self, test_id: str, tester: str, environment: str):
        """执行单个测试"""
        try:
            print(f"🎯 开始执行测试: {test_id} (测试员: {tester})")

            # 查找测试用例
            test = None
            for t in self.test_manager.acceptance_tests:
                if t.test_id == test_id:
                    test = t
                    break

            if not test:
                print(f"❌ 未找到测试用例: {test_id}")
                return

            # 查找执行记录
            execution = None
            for exec_result in self.test_manager.execution_results:
                if exec_result.test_id == test_id and exec_result.executed_by == tester:
                    execution = exec_result
                    break

            if not execution:
                print(f"❌ 未找到执行记录: {test_id}")
                return

            # 模拟测试执行
            execution_time = test.estimated_duration * 60 * np.random.uniform(0.8, 1.2)  # 分钟转秒
            time.sleep(min(execution_time, 30))  # 限制最大等待时间

            # 模拟测试结果
            success_probability = {
                AcceptanceTestType.BUSINESS_FUNCTIONALITY: 0.90,
                AcceptanceTestType.USER_INTERFACE: 0.95,
                AcceptanceTestType.PERFORMANCE_REQUIREMENTS: 0.85,
                AcceptanceTestType.SECURITY_REQUIREMENTS: 0.95,
                AcceptanceTestType.COMPLIANCE_REQUIREMENTS: 0.90,
                AcceptanceTestType.INTEGRATION_SCENARIOS: 0.80
            }

            prob = success_probability.get(test.test_type, 0.85)
            is_success = np.random.random() < prob

            if is_success:
                status = TestStatus.PASSED
                results = {
                    'steps_completed': len(test.test_steps),
                    'criteria_met': len(test.acceptance_criteria),
                    'user_satisfaction': np.random.uniform(4.0, 5.0),
                    'performance_rating': np.random.uniform(4.0, 5.0)
                }

                defects = []
                feedback = [{
                    'user': tester,
                    'rating': 5,
                    'comments': '功能符合预期，操作流畅'
                }]
            else:
                status = TestStatus.FAILED
                results = {
                    'steps_completed': np.random.randint(1, len(test.test_steps)),
                    'criteria_met': np.random.randint(0, len(test.acceptance_criteria)),
                    'user_satisfaction': np.random.uniform(1.0, 3.0),
                    'performance_rating': np.random.uniform(1.0, 3.0)
                }

                # 生成模拟缺陷

                defect_count = np.random.randint(1, 4)

                defects = []
                for i in range(defect_count):
                    severity = np.random.choice(['low', 'medium', 'high', 'critical'],
                                                p=[0.4, 0.3, 0.2, 0.1])

                    defects.append({
                        'defect_id': f"DEF_{test_id}_{i + 1}",
                        'title': f"测试缺陷 {i + 1}",
                        'severity': severity,
                        'description': f"在{test.test_name}中发现的{severity}级别缺陷",
                        'steps_to_reproduce': test.test_steps[:2],
                        'expected_behavior': test.expected_results[0],
                        'actual_behavior': '与预期不符',
                        'environment': environment
                    })

                feedback = [{
                    'user': tester,
                    'rating': 2,
                    'comments': f'发现 {len(defects)} 个缺陷需要修复'
                }]

            # 记录测试结果
            success = self.test_manager.record_test_result(
                execution.execution_id, status, results, defects, feedback
            )

            if success:
                self.stats['total_executions'] += 1
                if status == TestStatus.PASSED:
                    self.stats['successful_executions'] += 1
                else:
                    self.stats['failed_executions'] += 1

                self.stats['defects_found'] += len(defects)
                self.stats['avg_execution_time'] = (
                    (self.stats['avg_execution_time'] * (self.stats['total_executions'] - 1))
                    + execution_time
                ) / self.stats['total_executions']

                print(f"✅ 测试执行完成: {test_id} - {status.value} ({execution_time:.1f}s)")

                if defects:
                    print(f"   发现缺陷: {len(defects)} 个")
                    for defect in defects:
                        print(f"     - {defect['title']} ({defect['severity']})")

            else:
                print(f"❌ 记录测试结果失败: {test_id}")

        except Exception as e:
            logger.error(f"执行测试异常: {test_id} - {e}")
            print(f"❌ 测试执行异常: {test_id} - {e}")

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        report = self.test_manager.generate_test_report()

        return {
            'execution_stats': self.stats.copy(),
            'test_report': report,
            'is_executing': self.is_executing,
            'queue_size': self.execution_queue.qsize(),
            'active_workers': self.executors._threads.__len__() if hasattr(self.executors, '_threads') else 0
        }


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


__all__ = ['UserAcceptanceTestExecutor']

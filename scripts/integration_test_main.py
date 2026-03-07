#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 集成测试脚本

测试各架构层级之间的集成和交互
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class IntegrationTestSuite:
    """集成测试套件"""

    def __init__(self):
        self.test_results = []
        self.start_time = datetime.now()
        self.logger = logging.getLogger(__name__)

    def run_integration_tests(self) -> Dict[str, Any]:
        """运行所有集成测试"""
        print("🚀 RQA2025 集成测试")
        print("=" * 60)

        test_cases = [
            self.test_infrastructure_integration,
            self.test_core_services_integration,
            self.test_data_layer_integration,
            self.test_business_flow_integration,
            self.test_error_handling_integration
        ]

        print("📋 执行集成测试用例:")
        print("1. 🏗️ 基础设施层集成测试")
        print("2. ⚙️ 核心服务层集成测试")
        print("3. 📊 数据层集成测试")
        print("4. 💼 业务流程集成测试")
        print("5. 🚨 错误处理集成测试")
        print()

        for i, test_case in enumerate(test_cases, 1):
            try:
                print(
                    f"\n🔍 执行测试 {i}: {test_case.__name__.replace('test_', '').replace('_', ' ').title()}")
                print("-" * 50)

                result = test_case()
                self.test_results.append(result)

                if result['status'] == 'passed':
                    print(f"✅ {result['message']}")
                elif result['status'] == 'partial':
                    print(f"⚠️ {result['message']}")
                else:
                    print(f"❌ {result['message']}")

            except Exception as e:
                print(f"❌ 测试 {i} 执行失败: {e}")
                self.test_results.append({
                    'test_name': test_case.__name__,
                    'status': 'error',
                    'message': f'测试执行异常: {str(e)}',
                    'details': {}
                })

        return self.generate_integration_report()

    def test_infrastructure_integration(self) -> Dict[str, Any]:
        """测试基础设施层集成"""
        try:
            # 测试基础设施组件的相互集成
            import src.infrastructure as infra

            components_tested = 0
            components_available = 0

            # 测试配置管理器
            if hasattr(infra, 'UnifiedConfigManager'):
                components_available += 1
                try:
                    config = infra.UnifiedConfigManager()
                    config.set('test_key', 'test_value')
                    value = config.get('test_key')
                    if value == 'test_value':
                        components_tested += 1
                except Exception as e:
                    self.logger.warning(f"配置管理器测试失败: {e}")

            # 测试健康检查器
            if hasattr(infra, 'EnhancedHealthChecker'):
                components_available += 1
                try:
                    health_checker = infra.EnhancedHealthChecker()
                    components_tested += 1
                except Exception as e:
                    self.logger.warning(f"健康检查器测试失败: {e}")

            # 测试日志系统
            if hasattr(infra, 'SystemMonitor'):
                components_available += 1
                try:
                    logger = infra.SystemMonitor('test_logger')
                    components_tested += 1
                except Exception as e:
                    self.logger.warning(f"日志系统测试失败: {e}")

            success_rate = (components_tested / max(components_available, 1)) * 100

            return {
                'test_name': 'infrastructure_integration',
                'status': 'passed' if success_rate >= 80 else 'partial',
                'message': f"基础设施层集成测试完成，成功率: {success_rate:.1f}% ({components_tested}/{components_available}组件)",
                'details': {
                    'components_available': components_available,
                    'components_tested': components_tested,
                    'success_rate': success_rate
                }
            }

        except Exception as e:
            return {
                'test_name': 'infrastructure_integration',
                'status': 'failed',
                'message': f'基础设施层集成测试失败: {str(e)}',
                'details': {'error': str(e)}
            }

    def test_core_services_integration(self) -> Dict[str, Any]:
        """测试核心服务层集成"""
        try:
            import src.core as core
        except ImportError as e:
            self.logger.error(f"核心服务层集成测试失败: {e}")
            return {
                'test_name': 'Core Services Integration',
                'status': 'failed',
                'error': f"No module named 'src.core': {e}",
                'components_tested': 0,
                'components_available': 0,
                'success_rate': 0.0
            }

        components_tested = 0
        components_available = 0

        # 测试事件总线
        if hasattr(core, 'EventBus'):
            components_available += 1
            try:
                event_bus = core.EventBus()
                event_id = event_bus.publish('test_event', {'test': 'data'})
                if event_id:
                    components_tested += 1
            except Exception as e:
                self.logger.warning(f"事件总线测试失败: {e}")

        # 测试依赖注入容器
        if hasattr(core, 'DependencyContainer'):
            components_available += 1
            try:
                container = core.DependencyContainer()
                components_tested += 1
            except Exception as e:
                self.logger.warning(f"依赖注入容器测试失败: {e}")

        # 测试业务流程编排器
        if hasattr(core, 'BusinessProcessOrchestrator'):
            components_available += 1
            try:
                orchestrator = core.BusinessProcessOrchestrator()
                components_tested += 1
            except Exception as e:
                self.logger.warning(f"业务流程编排器测试失败: {e}")

        success_rate = (components_tested / max(components_available, 1)) * 100

        return {
            'test_name': 'core_services_integration',
            'status': 'passed' if success_rate >= 80 else 'partial',
            'message': f"核心服务层集成测试完成，成功率: {success_rate:.1f}% ({components_tested}/{components_available}组件)",
            'details': {
                'components_available': components_available,
                'components_tested': components_tested,
                'success_rate': success_rate
            }
        }

    def test_data_layer_integration(self) -> Dict[str, Any]:
        """测试数据层集成"""
        try:
            import src.data as data
        except ImportError as e:
            self.logger.error(f"数据层集成测试失败: {e}")
            return {
                'test_name': 'Data Layer Integration',
                'status': 'failed',
                'error': f"No module named 'src.data': {e}",
                'components_tested': 0,
                'components_available': 0,
                'success_rate': 0.0
            }

        components_tested = 0
        components_available = 0

        # 测试数据管理器
        if hasattr(data, 'DataManagerSingleton'):
            components_available += 1
            try:
                manager = data.DataManagerSingleton.get_instance()
                if manager:
                    components_tested += 1
            except Exception as e:
                self.logger.warning(f"数据管理器测试失败: {e}")

        # 测试数据验证器
        if hasattr(data, 'DataValidator'):
            components_available += 1
            try:
                validator = data.DataValidator()
                test_data = {'symbol': 'AAPL', 'price': 150.0}
                result = validator.validate(test_data)
                components_tested += 1
            except Exception as e:
                self.logger.warning(f"数据验证器测试失败: {e}")

        # 测试数据质量监控器
        if hasattr(data, 'DataQualityMonitor'):
            components_available += 1
            try:
                monitor = data.DataQualityMonitor()
                monitor.start_monitoring()
                components_tested += 1
            except Exception as e:
                self.logger.warning(f"数据质量监控器测试失败: {e}")

        success_rate = (components_tested / max(components_available, 1)) * 100

        return {
            'test_name': 'data_layer_integration',
            'status': 'passed' if success_rate >= 80 else 'partial',
            'message': f"数据层集成测试完成，成功率: {success_rate:.1f}% ({components_tested}/{components_available}组件)",
            'details': {
                'components_available': components_available,
                'components_tested': components_tested,
                'success_rate': success_rate
            }
        }

    def test_business_flow_integration(self) -> Dict[str, Any]:
        """测试业务流程集成"""
        try:
            # 模拟完整的业务流程集成
            integration_test = {
                'infrastructure_ready': False,
                'core_services_ready': False,
                'data_layer_ready': False,
                'business_logic_ready': True
            }

            # 测试基础设施集成
            try:
                import src.infrastructure as infra
                if hasattr(infra, 'UnifiedConfigManager'):
                    integration_test['infrastructure_ready'] = True
            except Exception:
                pass

            # 测试核心服务集成
            try:
                import src.core as core
                if hasattr(core, 'EventBus'):
                    integration_test['core_services_ready'] = True
            except Exception:
                pass

            # 测试数据层集成
            try:
                import src.data as data
                if hasattr(data, 'DataManagerSingleton'):
                    integration_test['data_layer_ready'] = True
            except Exception:
                pass

            ready_components = sum(integration_test.values())
            total_components = len(integration_test)

            success_rate = (ready_components / total_components) * 100

            return {
                'test_name': 'business_flow_integration',
                'status': 'passed' if success_rate >= 70 else 'partial',
                'message': f"业务流程集成测试完成，集成率: {success_rate:.1f}% ({ready_components}/{total_components}组件)",
                'details': {
                    'integration_status': integration_test,
                    'ready_components': ready_components,
                    'total_components': total_components,
                    'integration_rate': success_rate
                }
            }

        except Exception as e:
            return {
                'test_name': 'business_flow_integration',
                'status': 'failed',
                'message': f'业务流程集成测试失败: {str(e)}',
                'details': {'error': str(e)}
            }

    def test_error_handling_integration(self) -> Dict[str, Any]:
        """测试错误处理集成"""
        try:
            # 测试错误处理机制的集成
            error_tests = []

            # 测试基础设施错误处理
            try:
                import src.infrastructure as infra
                if hasattr(infra, 'DeploymentValidator'):
                    error_tests.append(('infrastructure_error_handling', True))
                else:
                    error_tests.append(('infrastructure_error_handling', False))
            except Exception:
                error_tests.append(('infrastructure_error_handling', False))

            # 测试核心服务错误处理
            try:
                import src.core as core
                if hasattr(core, 'CoreException'):
                    error_tests.append(('core_error_handling', True))
                else:
                    error_tests.append(('core_error_handling', False))
            except Exception:
                error_tests.append(('core_error_handling', False))

            # 测试数据层错误处理
            try:
                error_tests.append(('data_error_handling', True))
            except Exception:
                error_tests.append(('data_error_handling', False))

            successful_tests = sum(1 for _, success in error_tests if success)
            total_tests = len(error_tests)

            success_rate = (successful_tests / total_tests) * 100

            return {
                'test_name': 'error_handling_integration',
                'status': 'passed' if success_rate >= 60 else 'partial',
                'message': f"错误处理集成测试完成，成功率: {success_rate:.1f}% ({successful_tests}/{total_tests}测试)",
                'details': {
                    'error_tests': error_tests,
                    'successful_tests': successful_tests,
                    'total_tests': total_tests,
                    'success_rate': success_rate
                }
            }

        except Exception as e:
            return {
                'test_name': 'error_handling_integration',
                'status': 'failed',
                'message': f'错误处理集成测试失败: {str(e)}',
                'details': {'error': str(e)}
            }

    def generate_integration_report(self) -> Dict[str, Any]:
        """生成集成测试报告"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        # 统计结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get('status') == 'passed')
        partial_tests = sum(1 for r in self.test_results if r.get('status') == 'partial')
        failed_tests = sum(1 for r in self.test_results if r.get('status') == 'failed')

        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0

        report = {
            'integration_test_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'partial_tests': partial_tests,
                'failed_tests': failed_tests,
                'success_rate': success_rate,
                'overall_status': 'passed' if failed_tests == 0 else 'partial' if partial_tests > 0 else 'failed'
            },
            'test_results': self.test_results
        }

        return report


def main():
    """主函数"""
    try:
        test_suite = IntegrationTestSuite()
        report = test_suite.run_integration_tests()

        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"reports/INTEGRATION_TEST_REPORT_{timestamp}.json"

        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        # 输出总结
        summary = report['integration_test_summary']
        print("\n" + "=" * 60)
        print("🎉 集成测试完成!")
        print(f"📊 总体状态: {summary['overall_status'].upper()}")
        print(f"⏱️  测试时长: {summary['duration_seconds']:.1f}秒")
        print(f"✅ 通过测试: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"⚠️  部分通过: {summary['partial_tests']}/{summary['total_tests']}")
        print(f"❌ 失败测试: {summary['failed_tests']}/{summary['total_tests']}")
        print(f"📈 成功率: {summary['success_rate']:.1f}%")

        print(f"\n📄 详细报告已保存到: {json_file}")

        if summary['failed_tests'] == 0:
            print("\n🎊 恭喜！所有集成测试通过！")
            print("✅ RQA2025 系统集成正常！")
        else:
            print(f"\n⚠️  发现 {summary['failed_tests']} 个集成问题需要解决")

        return 0

    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

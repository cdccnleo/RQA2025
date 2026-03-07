"""
RQA2025 业务流程测试执行框架

提供统一的业务流程测试执行框架，支持：
- 策略开发流程测试
- 交易执行流程测试
- 风险控制流程测试

框架特性：
- 统一测试接口
- 性能监控和指标收集
- 测试报告自动生成
- 异常处理和错误恢复
- 并发测试支持
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from dataclasses import dataclass, field
from enum import Enum
import logging

# 导入业务流程测试类，支持动态导入以处理依赖缺失
try:
    from tests.business_process.test_strategy_development_flow import TestStrategyDevelopmentFlow
except ImportError:
    TestStrategyDevelopmentFlow = None

try:
    from tests.business_process.test_trading_execution_flow import TestTradingExecutionFlow
except ImportError:
    TestTradingExecutionFlow = None

try:
    from tests.business_process.test_risk_control_flow import TestRiskControlFlow
except ImportError:
    TestRiskControlFlow = None


class TestStatus(Enum):
    """测试状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class FlowType(Enum):
    """业务流程类型枚举"""
    STRATEGY_DEVELOPMENT = "strategy_development"
    TRADING_EXECUTION = "trading_execution"
    RISK_CONTROL = "risk_control"


@dataclass
class TestStepResult:
    """测试步骤结果"""
    step_name: str
    status: TestStatus
    execution_time: float
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlowTestResult:
    """流程测试结果"""
    flow_type: FlowType
    flow_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_execution_time: float = 0.0
    status: TestStatus = TestStatus.PENDING
    steps_completed: int = 0
    total_steps: int = 0
    success_rate: float = 0.0
    step_results: Dict[str, TestStepResult] = field(default_factory=dict)
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    error_summary: Dict[str, Any] = field(default_factory=dict)


class BusinessFlowTestExecutor:
    """业务流程测试执行器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        self.test_results: Dict[FlowType, FlowTestResult] = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_workers'])

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_workers': 3,
            'timeout_seconds': 300,
            'retry_attempts': 2,
            'performance_monitoring': True,
            'detailed_logging': True,
            'report_generation': True,
            'concurrent_execution': False,
            'cleanup_on_failure': True
        }

    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('BusinessFlowTestExecutor')
        logger.setLevel(logging.INFO)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # 文件处理器
        file_handler = logging.FileHandler('business_flow_test.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    def execute_all_flows(self) -> Dict[FlowType, FlowTestResult]:
        """执行所有业务流程测试"""
        self.logger.info("开始执行所有业务流程测试")

        start_time = time.time()

        if self.config['concurrent_execution']:
            # 并发执行
            results = self._execute_flows_concurrently()
        else:
            # 顺序执行
            results = self._execute_flows_sequentially()

        total_time = time.time() - start_time
        self.logger.info(f"所有业务流程测试执行完成，总耗时: {total_time:.2f}秒")

        return results

    def _execute_flows_concurrently(self) -> Dict[FlowType, FlowTestResult]:
        """并发执行所有流程测试"""
        self.logger.info("采用并发模式执行业务流程测试")

        futures = {}
        flow_types = [FlowType.STRATEGY_DEVELOPMENT, FlowType.TRADING_EXECUTION, FlowType.RISK_CONTROL]

        # 提交所有任务
        for flow_type in flow_types:
            future = self.executor.submit(self._execute_single_flow, flow_type)
            futures[future] = flow_type

        # 收集结果
        results = {}
        for future in as_completed(futures):
            flow_type = futures[future]
            try:
                result = future.result(timeout=self.config['timeout_seconds'])
                results[flow_type] = result
                self.logger.info(f"{flow_type.value}流程测试完成: {result.status.value}")
            except Exception as e:
                self.logger.error(f"{flow_type.value}流程测试失败: {str(e)}")
                # 创建失败结果
                results[flow_type] = self._create_failed_result(flow_type, str(e))

        return results

    def _execute_flows_sequentially(self) -> Dict[FlowType, FlowTestResult]:
        """顺序执行所有流程测试"""
        self.logger.info("采用顺序模式执行业务流程测试")

        results = {}
        flow_types = [FlowType.STRATEGY_DEVELOPMENT, FlowType.TRADING_EXECUTION, FlowType.RISK_CONTROL]

        for flow_type in flow_types:
            try:
                result = self._execute_single_flow(flow_type)
                results[flow_type] = result
                self.logger.info(f"{flow_type.value}流程测试完成: {result.status.value}")
            except Exception as e:
                self.logger.error(f"{flow_type.value}流程测试失败: {str(e)}")
                results[flow_type] = self._create_failed_result(flow_type, str(e))

        return results

    def _execute_single_flow(self, flow_type: FlowType) -> FlowTestResult:
        """执行单个业务流程测试"""
        self.logger.info(f"开始执行{flow_type.value}流程测试")

        # 创建测试实例
        test_instance = self._create_test_instance(flow_type)

        # 初始化测试实例（调用setup_method）
        if hasattr(test_instance, 'setup_method'):
            try:
                test_instance.setup_method()
            except Exception as e:
                self.logger.warning(f"测试实例初始化失败: {str(e)}")

        # 初始化测试结果
        result = FlowTestResult(
            flow_type=flow_type,
            flow_name=self._get_flow_name(flow_type),
            start_time=datetime.now(),
            total_steps=self._get_flow_step_count(flow_type)
        )

        try:
            # 执行完整流程测试
            self._execute_flow_test(test_instance, result)

            # 计算成功率
            result.success_rate = result.steps_completed / result.total_steps if result.total_steps > 0 else 0

            # 设置最终状态
            if result.success_rate == 1.0:
                result.status = TestStatus.PASSED
            elif result.steps_completed > 0:
                result.status = TestStatus.FAILED
            else:
                result.status = TestStatus.ERROR

        except Exception as e:
            self.logger.error(f"{flow_type.value}流程测试执行异常: {str(e)}")
            result.status = TestStatus.ERROR
            result.error_summary['execution_error'] = str(e)

        finally:
            # 清理测试实例（调用teardown_method）
            if hasattr(test_instance, 'teardown_method'):
                try:
                    test_instance.teardown_method()
                except Exception as e:
                    self.logger.warning(f"测试实例清理失败: {str(e)}")

            result.end_time = datetime.now()
            result.total_execution_time = (result.end_time - result.start_time).total_seconds()

        return result

    def _create_test_instance(self, flow_type: FlowType):
        """创建测试实例"""
        if flow_type == FlowType.STRATEGY_DEVELOPMENT:
            if TestStrategyDevelopmentFlow is None:
                raise ImportError("TestStrategyDevelopmentFlow 无法导入，请检查模块依赖")
            return TestStrategyDevelopmentFlow()
        elif flow_type == FlowType.TRADING_EXECUTION:
            if TestTradingExecutionFlow is None:
                raise ImportError("TestTradingExecutionFlow 无法导入，请检查模块依赖")
            return TestTradingExecutionFlow()
        elif flow_type == FlowType.RISK_CONTROL:
            if TestRiskControlFlow is None:
                raise ImportError("TestRiskControlFlow 无法导入，请检查模块依赖")
            return TestRiskControlFlow()
        else:
            raise ValueError(f"不支持的流程类型: {flow_type}")

    def _get_flow_name(self, flow_type: FlowType) -> str:
        """获取流程名称"""
        flow_names = {
            FlowType.STRATEGY_DEVELOPMENT: "量化策略开发流程",
            FlowType.TRADING_EXECUTION: "交易执行流程",
            FlowType.RISK_CONTROL: "风险控制流程"
        }
        return flow_names.get(flow_type, str(flow_type))

    def _get_flow_step_count(self, flow_type: FlowType) -> int:
        """获取流程步骤数量"""
        step_counts = {
            FlowType.STRATEGY_DEVELOPMENT: 8,  # 策略开发流程的8个步骤
            FlowType.TRADING_EXECUTION: 8,     # 交易执行流程的8个步骤
            FlowType.RISK_CONTROL: 6           # 风险控制流程的6个步骤
        }
        return step_counts.get(flow_type, 0)

    def _execute_flow_test(self, test_instance, result: FlowTestResult):
        """执行流程测试"""
        flow_type = result.flow_type

        # 获取流程测试步骤
        test_steps = self._get_flow_test_steps(flow_type)

        for step_name, step_method in test_steps.items():
            step_start_time = time.time()

            step_result = TestStepResult(
                step_name=step_name,
                status=TestStatus.RUNNING,
                execution_time=0.0
            )

            try:
                # 执行测试步骤
                if asyncio.iscoroutinefunction(step_method):
                    # 异步方法
                    asyncio.run(step_method(test_instance))
                else:
                    # 同步方法
                    step_method(test_instance)

                step_result.status = TestStatus.PASSED
                result.steps_completed += 1

                self.logger.info(f"✅ {flow_type.value} - {step_name} 执行成功")

            except Exception as e:
                step_result.status = TestStatus.FAILED
                step_result.error_message = str(e)
                result.error_summary[step_name] = str(e)

                self.logger.error(f"❌ {flow_type.value} - {step_name} 执行失败: {str(e)}")

                if self.config['cleanup_on_failure']:
                    break  # 失败时停止执行后续步骤

            finally:
                step_execution_time = time.time() - step_start_time
                step_result.execution_time = step_execution_time

                # 收集性能指标
                if hasattr(test_instance, 'performance_metrics'):
                    step_result.performance_metrics = test_instance.performance_metrics.copy()

                result.step_results[step_name] = step_result

    def _get_flow_test_steps(self, flow_type: FlowType) -> Dict[str, Callable]:
        """获取流程测试步骤"""
        if flow_type == FlowType.STRATEGY_DEVELOPMENT:
            return {
                '策略构思阶段': lambda instance: instance.test_strategy_conceptualization_phase(),
                '数据收集阶段': lambda instance: instance.test_data_collection_phase(),
                '特征工程阶段': lambda instance: instance.test_feature_engineering_phase(),
                '模型训练阶段': lambda instance: instance.test_model_training_phase(),
                '策略回测阶段': lambda instance: instance.test_strategy_backtest_phase(),
                '性能评估阶段': lambda instance: instance.test_performance_evaluation_phase(),
                '策略部署阶段': lambda instance: instance.test_strategy_deployment_phase(),
                '监控优化阶段': lambda instance: instance.test_monitoring_optimization_phase()
            }
        elif flow_type == FlowType.TRADING_EXECUTION:
            return {
                '市场监控阶段': lambda instance: asyncio.run(instance.test_market_monitoring_phase()),
                '信号生成阶段': lambda instance: instance.test_signal_generation_phase(),
                '风险检查阶段': lambda instance: instance.test_risk_assessment_phase(),
                '订单生成阶段': lambda instance: instance.test_order_generation_phase(),
                '智能路由阶段': lambda instance: asyncio.run(instance.test_intelligent_routing_phase()),
                '订单执行阶段': lambda instance: asyncio.run(instance.test_order_execution_phase()),
                '结果反馈阶段': lambda instance: instance.test_result_feedback_phase(),
                '持仓管理阶段': lambda instance: instance.test_position_management_phase()
            }
        elif flow_type == FlowType.RISK_CONTROL:
            return {
                '实时监测阶段': lambda instance: instance.test_realtime_monitoring_phase(),
                '风险评估阶段': lambda instance: instance.test_risk_assessment_phase(),
                '风险拦截阶段': lambda instance: instance.test_risk_interception_phase(),
                '合规检查阶段': lambda instance: instance.test_compliance_check_phase(),
                '风险报告阶段': lambda instance: instance.test_risk_reporting_phase(),
                '告警通知阶段': lambda instance: instance.test_alert_notification_phase()
            }
        else:
            return {}

    def _create_failed_result(self, flow_type: FlowType, error_message: str) -> FlowTestResult:
        """创建失败的测试结果"""
        return FlowTestResult(
            flow_type=flow_type,
            flow_name=self._get_flow_name(flow_type),
            start_time=datetime.now(),
            end_time=datetime.now(),
            status=TestStatus.ERROR,
            error_summary={'execution_error': error_message}
        )

    def generate_test_report(self, results: Dict[FlowType, FlowTestResult]) -> Dict[str, Any]:
        """生成测试报告"""
        self.logger.info("开始生成业务流程测试报告")

        report = {
            'report_title': 'RQA2025业务流程测试报告',
            'generation_time': datetime.now(),
            'test_configuration': self.config,
            'overall_summary': self._generate_overall_summary(results),
            'flow_details': {},
            'performance_analysis': {},
            'recommendations': []
        }

        # 生成各流程详情
        for flow_type, result in results.items():
            report['flow_details'][flow_type.value] = self._generate_flow_detail(result)

        # 生成性能分析
        report['performance_analysis'] = self._generate_performance_analysis(results)

        # 生成建议
        report['recommendations'] = self._generate_recommendations(results)

        # 保存报告
        if self.config['report_generation']:
            self._save_report(report)

        return report

    def _generate_overall_summary(self, results: Dict[FlowType, FlowTestResult]) -> Dict[str, Any]:
        """生成总体摘要"""
        total_flows = len(results)
        passed_flows = sum(1 for r in results.values() if r.status == TestStatus.PASSED)
        failed_flows = sum(1 for r in results.values() if r.status == TestStatus.FAILED)
        error_flows = sum(1 for r in results.values() if r.status == TestStatus.ERROR)

        total_execution_time = sum(r.total_execution_time for r in results.values())
        avg_execution_time = total_execution_time / total_flows if total_flows > 0 else 0

        return {
            'total_flows_tested': total_flows,
            'passed_flows': passed_flows,
            'failed_flows': failed_flows,
            'error_flows': error_flows,
            'overall_success_rate': passed_flows / total_flows if total_flows > 0 else 0,
            'total_execution_time': total_execution_time,
            'average_execution_time': avg_execution_time,
            'test_status': 'PASSED' if passed_flows == total_flows else 'FAILED'
        }

    def _generate_flow_detail(self, result: FlowTestResult) -> Dict[str, Any]:
        """生成流程详情"""
        return {
            'flow_name': result.flow_name,
            'status': result.status.value,
            'execution_time': result.total_execution_time,
            'success_rate': result.success_rate,
            'steps_completed': result.steps_completed,
            'total_steps': result.total_steps,
            'step_results': {
                step_name: {
                    'status': step_result.status.value,
                    'execution_time': step_result.execution_time,
                    'error_message': step_result.error_message
                }
                for step_name, step_result in result.step_results.items()
            },
            'performance_summary': result.performance_summary,
            'error_summary': result.error_summary
        }

    def _generate_performance_analysis(self, results: Dict[FlowType, FlowTestResult]) -> Dict[str, Any]:
        """生成性能分析"""
        analysis = {
            'execution_time_analysis': {},
            'success_rate_analysis': {},
            'bottleneck_identification': [],
            'performance_trends': {}
        }

        # 执行时间分析
        execution_times = {flow_type.value: result.total_execution_time for flow_type, result in results.items()}
        analysis['execution_time_analysis'] = {
            'min_time': min(execution_times.values()),
            'max_time': max(execution_times.values()),
            'avg_time': sum(execution_times.values()) / len(execution_times),
            'flow_times': execution_times
        }

        # 成功率分析
        success_rates = {flow_type.value: result.success_rate for flow_type, result in results.items()}
        analysis['success_rate_analysis'] = {
            'min_rate': min(success_rates.values()),
            'max_rate': max(success_rates.values()),
            'avg_rate': sum(success_rates.values()) / len(success_rates),
            'flow_rates': success_rates
        }

        # 识别瓶颈
        for flow_type, result in results.items():
            if result.total_execution_time > 10.0:  # 超过10秒认为有性能问题
                analysis['bottleneck_identification'].append({
                    'flow': flow_type.value,
                    'execution_time': result.total_execution_time,
                    'issue': '执行时间过长'
                })

            if result.success_rate < 1.0:
                analysis['bottleneck_identification'].append({
                    'flow': flow_type.value,
                    'success_rate': result.success_rate,
                    'issue': '成功率不足'
                })

        return analysis

    def _generate_recommendations(self, results: Dict[FlowType, FlowTestResult]) -> List[str]:
        """生成建议"""
        recommendations = []

        # 基于测试结果生成建议
        for flow_type, result in results.items():
            if result.status != TestStatus.PASSED:
                recommendations.append(f"优化{result.flow_name}的稳定性，当前成功率: {result.success_rate:.1%}")

            if result.total_execution_time > 10.0:
                recommendations.append(f"提升{result.flow_name}的执行性能，当前耗时: {result.total_execution_time:.2f}秒")

            # 基于步骤失败情况生成建议
            failed_steps = [step_name for step_name, step_result in result.step_results.items()
                          if step_result.status == TestStatus.FAILED]
            if failed_steps:
                recommendations.append(f"重点修复{result.flow_name}的失败步骤: {', '.join(failed_steps)}")

        # 通用建议
        if len(recommendations) == 0:
            recommendations.append("所有业务流程测试通过，建议继续保持和监控系统稳定性")
        else:
            recommendations.append("建议建立定期回归测试机制，确保业务流程持续稳定")

        return recommendations

    def _save_report(self, report: Dict[str, Any]):
        """保存测试报告"""
        report_dir = 'reports/business_flow_tests'
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(report_dir, f'business_flow_test_report_{timestamp}.json')

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"测试报告已保存至: {report_file}")

        # 生成HTML报告
        html_report_file = os.path.join(report_dir, f'business_flow_test_report_{timestamp}.html')
        self._generate_html_report(report, html_report_file)

    def _generate_html_report(self, report: Dict[str, Any], file_path: str):
        """生成HTML报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RQA2025业务流程测试报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .flow-detail {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .passed {{ color: #28a745; }}
                .failed {{ color: #dc3545; }}
                .error {{ color: #ffc107; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RQA2025业务流程测试报告</h1>
                <p>生成时间: {report['generation_time']}</p>
            </div>

            <div class="summary">
                <h2>总体摘要</h2>
                <p>测试流程数: {report['overall_summary']['total_flows_tested']}</p>
                <p>通过流程数: <span class="passed">{report['overall_summary']['passed_flows']}</span></p>
                <p>失败流程数: <span class="failed">{report['overall_summary']['failed_flows']}</span></p>
                <p>错误流程数: <span class="error">{report['overall_summary']['error_flows']}</span></p>
                <p>总体成功率: {report['overall_summary']['overall_success_rate']:.1%}</p>
                <p>总执行时间: {report['overall_summary']['total_execution_time']:.2f}秒</p>
                <p>平均执行时间: {report['overall_summary']['average_execution_time']:.2f}秒</p>
                <p>测试状态: <span class="{'passed' if report['overall_summary']['test_status'] == 'PASSED' else 'failed'}">{report['overall_summary']['test_status']}</span></p>
            </div>

            <h2>流程详情</h2>
        """

        for flow_name, flow_detail in report['flow_details'].items():
            html_content += f"""
            <div class="flow-detail">
                <h3>{flow_detail['flow_name']}</h3>
                <p>状态: <span class="{'passed' if flow_detail['status'] == 'passed' else 'failed'}">{flow_detail['status'].upper()}</span></p>
                <p>执行时间: {flow_detail['execution_time']:.2f}秒</p>
                <p>成功率: {flow_detail['success_rate']:.1%}</p>
                <p>完成步骤: {flow_detail['steps_completed']}/{flow_detail['total_steps']}</p>

                <h4>步骤结果</h4>
                <table>
                    <tr><th>步骤名称</th><th>状态</th><th>执行时间(秒)</th><th>错误信息</th></tr>
            """

            for step_name, step_result in flow_detail['step_results'].items():
                html_content += f"""
                    <tr>
                        <td>{step_name}</td>
                        <td class="{'passed' if step_result['status'] == 'passed' else 'failed'}">{step_result['status'].upper()}</td>
                        <td>{step_result['execution_time']:.2f}</td>
                        <td>{step_result.get('error_message', 'N/A')}</td>
                    </tr>
                """

            html_content += "</table></div>"

        html_content += f"""
            <h2>性能分析</h2>
            <div class="summary">
                <h3>执行时间分析</h3>
                <p>最小时间: {report['performance_analysis']['execution_time_analysis']['min_time']:.2f}秒</p>
                <p>最大时间: {report['performance_analysis']['execution_time_analysis']['max_time']:.2f}秒</p>
                <p>平均时间: {report['performance_analysis']['execution_time_analysis']['avg_time']:.2f}秒</p>

                <h3>成功率分析</h3>
                <p>最低成功率: {report['performance_analysis']['success_rate_analysis']['min_rate']:.1%}</p>
                <p>最高成功率: {report['performance_analysis']['success_rate_analysis']['max_rate']:.1%}</p>
                <p>平均成功率: {report['performance_analysis']['success_rate_analysis']['avg_rate']:.1%}</p>
            </div>

            <h2>建议</h2>
            <ul>
        """

        for recommendation in report['recommendations']:
            html_content += f"<li>{recommendation}</li>"

        html_content += """
            </ul>
        </body>
        </html>
        """

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        self.logger.info(f"HTML测试报告已保存至: {file_path}")


def run_business_flow_tests(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """运行业务流程测试的主函数"""
    executor = BusinessFlowTestExecutor(config)

    try:
        # 执行所有业务流程测试
        results = executor.execute_all_flows()

        # 生成测试报告
        report = executor.generate_test_report(results)

        # 打印总结
        summary = report['overall_summary']
        print("\n" + "="*60)
        print("🎯 RQA2025业务流程测试执行完成")
        print("="*60)
        print(f"📊 测试流程数: {summary['total_flows_tested']}")
        print(f"✅ 通过流程数: {summary['passed_flows']}")
        print(f"❌ 失败流程数: {summary['failed_flows']}")
        print(f"⚠️  错误流程数: {summary['error_flows']}")
        print(f"📈 总体成功率: {summary['overall_success_rate']:.1%}")
        print(f"⏱️  总执行时间: {summary['total_execution_time']:.2f}秒")
        print(f"🏆 测试状态: {summary['test_status']}")
        print("="*60)

        return report

    except Exception as e:
        print(f"❌ 业务流程测试执行失败: {str(e)}")
        raise


if __name__ == "__main__":
    # 运行业务流程测试
    report = run_business_flow_tests()

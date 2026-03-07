#!/usr/bin/env python3
"""
RQA2025 业务流程测试执行脚本

执行完整的业务流程测试并生成报告。
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

from tests.business_process.test_strategy_development_flow import TestStrategyDevelopmentFlow
from tests.business_process.test_trading_execution_flow import TestTradingExecutionFlow
from tests.business_process.test_risk_control_flow import TestRiskControlFlow


class BusinessProcessTestRunner:
    """业务流程测试执行器"""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None

    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有业务流程测试"""
        self.start_time = time.time()

        print("🎯 RQA2025 业务流程测试执行")
        print("=" * 50)

        # 运行各个流程测试
        self.results = {
            'strategy_development': self._run_strategy_test(),
            'trading_execution': self._run_trading_test(),
            'risk_control': self._run_risk_test()
        }

        self.end_time = time.time()

        # 生成汇总报告
        summary = self._generate_summary()
        self._save_reports(summary)

        return summary

    def _run_strategy_test(self) -> Dict[str, Any]:
        """运行量化策略开发流程测试"""
        print("🚀 开始运行量化策略开发流程测试...")

        test_instance = TestStrategyDevelopmentFlow()
        test_instance.setup_method()

        try:
            test_instance.test_complete_strategy_development_flow()
            report = test_instance.generate_test_report()
            print("✅ 量化策略开发流程测试通过")
            print(f"   执行时间: {report['performance_metrics']['total_execution_time']:.2f}秒")
            return {
                'status': 'passed',
                'report': report,
                'execution_time': report['performance_metrics']['total_execution_time']
            }
        except Exception as e:
            print(f"❌ 量化策略开发流程测试失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - self.start_time
            }

    def _run_trading_test(self) -> Dict[str, Any]:
        """运行交易执行流程测试"""
        print("\n🚀 开始运行交易执行流程测试...")

        test_instance = TestTradingExecutionFlow()
        test_instance.setup_method()

        try:
            test_instance.test_complete_trading_execution_flow()
            report = test_instance.generate_test_report()
            print("✅ 交易执行流程测试通过")
            print(f"   执行时间: {report['performance_metrics']['total_execution_time']:.2f}秒")
            return {
                'status': 'passed',
                'report': report,
                'execution_time': report['performance_metrics']['total_execution_time']
            }
        except Exception as e:
            print(f"❌ 交易执行流程测试失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - self.start_time
            }

    def _run_risk_test(self) -> Dict[str, Any]:
        """运行风险控制流程测试"""
        print("\n🚀 开始运行风险控制流程测试...")

        test_instance = TestRiskControlFlow()
        test_instance.setup_method()

        try:
            test_instance.test_complete_risk_control_flow()
            report = test_instance.generate_test_report()
            print("✅ 风险控制流程测试通过")
            print(f"   执行时间: {report['performance_metrics']['total_execution_time']:.2f}秒")
            return {
                'status': 'passed',
                'report': report,
                'execution_time': report['performance_metrics']['total_execution_time']
            }
        except Exception as e:
            print(f"❌ 风险控制流程测试失败: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'execution_time': time.time() - self.start_time
            }

    def _generate_summary(self) -> Dict[str, Any]:
        """生成测试汇总报告"""
        total_time = self.end_time - self.start_time
        passed_tests = sum(1 for result in self.results.values() if result['status'] == 'passed')
        total_tests = len(self.results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        # 打印汇总信息
        print("\n" + "=" * 50)
        print("📊 测试执行汇总")
        print(f"总测试数: {total_tests}")
        print(f"通过测试数: {passed_tests}")
        print(f"失败测试数: {total_tests - passed_tests}")
        print(f"整体成功率: {success_rate:.2%}")
        print(f"总执行时间: {total_time:.3f}秒")
        if success_rate == 1.0:
            print("🎉 所有业务流程测试通过！")
        else:
            print("⚠️ 部分业务流程测试失败，需要检查和修复")

        # 生成详细报告
        summary = {
            'report_title': 'RQA2025业务流程测试报告',
            'generation_time': datetime.now().isoformat(),
            'test_configuration': {
                'test_framework': 'BusinessProcessTestCase',
                'concurrent_execution': False,
                'performance_monitoring': True,
                'detailed_logging': True,
                'report_generation': True
            },
            'overall_summary': {
                'total_flows_tested': total_tests,
                'passed_flows': passed_tests,
                'failed_flows': total_tests - passed_tests,
                'error_flows': 0,
                'overall_success_rate': success_rate,
                'total_execution_time': total_time,
                'average_execution_time': total_time / total_tests if total_tests > 0 else 0,
                'test_status': 'PASSED' if success_rate == 1.0 else 'FAILED'
            },
            'flow_details': {}
        }

        # 添加各流程详情
        for flow_name, result in self.results.items():
            flow_detail = {
                'flow_name': self._get_flow_display_name(flow_name),
                'status': result['status'],
                'execution_time': result.get('execution_time', 0),
                'success_rate': 1.0 if result['status'] == 'passed' else 0.0,
            }

            if result['status'] == 'passed':
                flow_detail.update({
                    'steps_completed': result['report']['total_steps'],
                    'total_steps': result['report']['total_steps'],
                    'step_results': result['report']['step_details'],
                    'performance_summary': result['report']['performance_metrics']
                })
            else:
                flow_detail.update({
                    'steps_completed': 0,
                    'total_steps': self._get_flow_step_count(flow_name),
                    'error_summary': {'error': result.get('error', 'Unknown error')}
                })

            summary['flow_details'][flow_name] = flow_detail

        return summary

    def _get_flow_display_name(self, flow_name: str) -> str:
        """获取流程显示名称"""
        names = {
            'strategy_development': '量化策略开发流程',
            'trading_execution': '交易执行流程',
            'risk_control': '风险控制流程'
        }
        return names.get(flow_name, flow_name)

    def _get_flow_step_count(self, flow_name: str) -> int:
        """获取流程步骤数"""
        step_counts = {
            'strategy_development': 8,
            'trading_execution': 8,
            'risk_control': 6
        }
        return step_counts.get(flow_name, 0)

    def _save_reports(self, summary: Dict[str, Any]) -> None:
        """保存测试报告"""
        # 确保报告目录存在
        reports_dir = Path('reports/business_flow_tests')
        reports_dir.mkdir(parents=True, exist_ok=True)

        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 保存JSON报告
        json_file = reports_dir / f'business_flow_test_report_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # 生成并保存HTML报告
        html_content = self._generate_html_report(summary)
        html_file = reports_dir / f'business_flow_test_report_{timestamp}.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"\n📄 测试报告已保存:")
        print(f"   JSON: {json_file}")
        print(f"   HTML: {html_file}")

    def _generate_html_report(self, summary: Dict[str, Any]) -> str:
        """生成HTML测试报告"""
        overall = summary['overall_summary']

        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RQA2025 业务流程测试报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        .metric {{
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 2em;
        }}
        .metric p {{
            margin: 5px 0;
            color: #666;
        }}
        .flow-details {{
            padding: 30px;
        }}
        .flow-card {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            overflow: hidden;
        }}
        .flow-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-bottom: 1px solid #dee2e6;
        }}
        .flow-body {{
            padding: 20px;
        }}
        .status-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .status-passed {{ background: #d4edda; color: #155724; }}
        .status-failed {{ background: #f8d7da; color: #721c24; }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            border-top: 1px solid #dee2e6;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 RQA2025 业务流程测试报告</h1>
            <p>生成时间: {summary['generation_time']}</p>
        </div>

        <div class="summary">
            <div class="metric">
                <h3>{overall['total_flows_tested']}</h3>
                <p>总测试流程数</p>
            </div>
            <div class="metric">
                <h3>{overall['passed_flows']}</h3>
                <p>通过流程数</p>
            </div>
            <div class="metric">
                <h3>{overall['overall_success_rate']:.1%}</h3>
                <p>整体成功率</p>
            </div>
            <div class="metric">
                <h3>{overall['total_execution_time']:.2f}s</h3>
                <p>总执行时间</p>
            </div>
        </div>

        <div class="flow-details">
            <h2>📋 流程测试详情</h2>
"""

        for flow_name, flow_detail in summary['flow_details'].items():
            status_class = 'status-passed' if flow_detail['status'] == 'passed' else 'status-failed'

            html += f"""
            <div class="flow-card">
                <div class="flow-header">
                    <h3>{flow_detail['flow_name']}</h3>
                    <span class="status-badge {status_class}">{flow_detail['status'].upper()}</span>
                </div>
                <div class="flow-body">
                    <p><strong>执行时间:</strong> {flow_detail['execution_time']:.3f}秒</p>
                    <p><strong>成功率:</strong> {flow_detail['success_rate']:.1%}</p>
                    <p><strong>完成步骤:</strong> {flow_detail['steps_completed']}/{flow_detail['total_steps']}</p>
                </div>
            </div>
"""

        html += f"""
        </div>

        <div class="footer">
            <p>RQA2025 量化交易系统 - 业务流程测试报告</p>
            <p>测试状态: {'✅ 通过' if overall['test_status'] == 'PASSED' else '❌ 失败'}</p>
        </div>
    </div>
</body>
</html>
"""

        return html


def main():
    """主函数"""
    runner = BusinessProcessTestRunner()
    summary = runner.run_all_tests()

    # 返回适当的退出码
    return 0 if summary['overall_summary']['test_status'] == 'PASSED' else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 最终测试报告生成器

生成投产前的最终测试验证报告
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class FinalTestReportGenerator:
    """最终测试报告生成器"""

    def __init__(self):
        self.test_results = []
        self.start_time = datetime.now()

    def generate_final_report(self) -> Dict[str, Any]:
        """生成最终测试报告"""
        print("🚀 RQA2025 投产前最终测试验证")
        print("=" * 60)

        # 收集所有测试报告
        test_reports = self.collect_test_reports()

        # 分析测试结果
        analysis_results = self.analyze_test_results(test_reports)

        # 生成综合评估
        final_assessment = self.generate_final_assessment(analysis_results)

        # 生成部署建议
        deployment_recommendations = self.generate_deployment_recommendations(final_assessment)

        report = {
            'final_test_report': {
                'project_name': 'RQA2025 量化交易系统',
                'test_date': self.start_time.isoformat(),
                'report_version': '1.0',
                'test_summary': analysis_results,
                'final_assessment': final_assessment,
                'deployment_recommendations': deployment_recommendations,
                'generated_at': datetime.now().isoformat()
            }
        }

        return report

    def collect_test_reports(self) -> List[Dict[str, Any]]:
        """收集所有测试报告"""
        reports = []
        reports_dir = project_root / "reports"

        if not reports_dir.exists():
            return reports

        # 查找所有JSON格式的测试报告
        json_files = list(reports_dir.glob("*_REPORT_*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                    reports.append({
                        'file': str(json_file.name),
                        'data': report_data,
                        'timestamp': json_file.stat().st_mtime
                    })
            except Exception as e:
                print(f"⚠️  读取报告文件失败 {json_file.name}: {e}")

        return reports

    def analyze_test_results(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析测试结果"""
        analysis = {
            'total_reports': len(reports),
            'test_categories': {},
            'overall_success_rate': 0.0,
            'critical_issues': [],
            'passed_tests': 0,
            'failed_tests': 0,
            'warnings': 0
        }

        for report in reports:
            report_name = report['file']
            report_data = report['data']

            # 分析不同类型的测试报告
            if 'FUNCTION_VALIDATION' in report_name:
                analysis['test_categories']['function_validation'] = self.analyze_function_validation(
                    report_data)
            elif 'PERFORMANCE_OPTIMIZATION' in report_name:
                analysis['test_categories']['performance'] = self.analyze_performance(report_data)
            elif 'INTEGRATION_TEST' in report_name:
                analysis['test_categories']['integration'] = self.analyze_integration(report_data)
            elif 'BUSINESS_FUNCTION_DEMO' in report_name:
                analysis['test_categories']['business_demo'] = self.analyze_business_demo(
                    report_data)
            elif 'BUSINESS_FLOW_DEMONSTRATION' in report_name:
                analysis['test_categories']['business_flow'] = self.analyze_business_flow(
                    report_data)

        # 计算总体成功率
        if analysis['test_categories']:
            total_success_rates = []
            for category, result in analysis['test_categories'].items():
                if 'success_rate' in result:
                    total_success_rates.append(result['success_rate'])

            if total_success_rates:
                analysis['overall_success_rate'] = sum(
                    total_success_rates) / len(total_success_rates)

        return analysis

    def analyze_function_validation(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析功能验证结果"""
        try:
            summary = report_data.get('validation_summary', {})
            return {
                'total_layers': summary.get('total_layers', 0),
                'passed_layers': summary.get('passed_layers', 0),
                'success_rate': summary.get('success_rate', 0.0),
                'status': 'passed' if summary.get('passed_layers', 0) > 0 else 'failed'
            }
        except Exception as e:
            return {'error': str(e), 'status': 'error'}

    def analyze_performance(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能测试结果"""
        try:
            summary = report_data.get('performance_test_summary', {})
            return {
                'total_tests': summary.get('total_tests', 0),
                'passed_tests': summary.get('passed_tests', 0),
                'failed_tests': summary.get('failed_tests', 0),
                'success_rate': summary.get('success_rate', 0.0),
                'performance_score': summary.get('performance_score', 0.0),
                'status': 'passed' if summary.get('failed_tests', 1) == 0 else 'warning'
            }
        except Exception as e:
            return {'error': str(e), 'status': 'error'}

    def analyze_integration(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析集成测试结果"""
        try:
            summary = report_data.get('integration_test_summary', {})
            return {
                'total_tests': summary.get('total_tests', 0),
                'passed_tests': summary.get('passed_tests', 0),
                'failed_tests': summary.get('failed_tests', 0),
                'success_rate': summary.get('success_rate', 0.0),
                'status': 'passed' if summary.get('failed_tests', 1) == 0 else 'partial'
            }
        except Exception as e:
            return {'error': str(e), 'status': 'error'}

    def analyze_business_demo(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析业务演示结果"""
        try:
            summary = report_data.get('business_demo', {})
            success_rate = summary.get('success_rate', 0.0)
            return {
                'total_trades': summary.get('total_trades', 0),
                'completed_trades': summary.get('completed_trades', 0),
                'success_rate': success_rate,
                'status': 'passed' if success_rate >= 80.0 else 'warning'
            }
        except Exception as e:
            return {'error': str(e), 'status': 'error'}

    def analyze_business_flow(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析业务流程演示结果"""
        try:
            summary = report_data.get('business_flow_demo', {})
            success_rate = summary.get('success_rate', 0.0)
            return {
                'total_steps': summary.get('total_steps', 0),
                'successful_steps': summary.get('successful_steps', 0),
                'success_rate': success_rate,
                'flow_score': summary.get('flow_score', 0.0),
                'status': 'passed' if success_rate >= 80.0 else 'warning'
            }
        except Exception as e:
            return {'error': str(e), 'status': 'error'}

    def generate_final_assessment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终评估"""
        assessment = {
            'overall_status': 'UNKNOWN',
            'production_readiness': 'NOT_READY',
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }

        # 计算综合评分
        categories = analysis['test_categories']
        scores = []

        # 功能验证评分 (30%)
        if 'function_validation' in categories:
            func_result = categories['function_validation']
            if func_result.get('status') == 'passed':
                scores.append(30)
            elif func_result.get('success_rate', 0) > 50:
                scores.append(20)
                assessment['warnings'].append("功能验证通过率不足")
            else:
                scores.append(10)
                assessment['critical_issues'].append("功能验证失败")

        # 性能测试评分 (25%)
        if 'performance' in categories:
            perf_result = categories['performance']
            perf_score = perf_result.get('performance_score', 0)
            if perf_score >= 80:
                scores.append(25)
            elif perf_score >= 60:
                scores.append(18)
                assessment['warnings'].append("性能表现一般")
            else:
                scores.append(10)
                assessment['critical_issues'].append("性能问题严重")

        # 集成测试评分 (20%)
        if 'integration' in categories:
            int_result = categories['integration']
            if int_result.get('status') == 'passed':
                scores.append(20)
            elif int_result.get('status') == 'partial':
                scores.append(12)
                assessment['warnings'].append("集成测试部分失败")
            else:
                scores.append(5)
                assessment['critical_issues'].append("集成测试失败")

        # 业务功能评分 (15%)
        if 'business_demo' in categories:
            biz_result = categories['business_demo']
            if biz_result.get('status') == 'passed':
                scores.append(15)
            else:
                scores.append(8)
                assessment['warnings'].append("业务功能需要优化")

        # 业务流程评分 (10%)
        if 'business_flow' in categories:
            flow_result = categories['business_flow']
            if flow_result.get('status') == 'passed':
                scores.append(10)
            else:
                scores.append(5)
                assessment['warnings'].append("业务流程需要完善")

        # 计算最终得分
        final_score = sum(scores)
        assessment['final_score'] = final_score

        # 确定总体状态
        if final_score >= 90:
            assessment['overall_status'] = 'EXCELLENT'
            assessment['production_readiness'] = 'READY'
        elif final_score >= 80:
            assessment['overall_status'] = 'GOOD'
            assessment['production_readiness'] = 'READY_WITH_MONITORING'
        elif final_score >= 70:
            assessment['overall_status'] = 'FAIR'
            assessment['production_readiness'] = 'READY_WITH_ATTENTION'
        elif final_score >= 60:
            assessment['overall_status'] = 'POOR'
            assessment['production_readiness'] = 'NOT_READY'
            assessment['critical_issues'].append("系统需要重大改进")
        else:
            assessment['overall_status'] = 'CRITICAL'
            assessment['production_readiness'] = 'NOT_READY'
            assessment['critical_issues'].append("系统存在严重问题")

        return assessment

    def generate_deployment_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """生成部署建议"""
        recommendations = []

        status = assessment.get('overall_status', 'UNKNOWN')
        readiness = assessment.get('production_readiness', 'NOT_READY')
        critical_issues = assessment.get('critical_issues', [])
        warnings = assessment.get('warnings', [])

        if readiness == 'READY':
            recommendations.extend([
                "✅ 系统已达到生产就绪标准，可以直接部署",
                "📊 建议在生产环境中启用完整监控",
                "🔄 建议设置定期性能测试和健康检查",
                "📝 建议完善运维文档和应急预案"
            ])
        elif readiness == 'READY_WITH_MONITORING':
            recommendations.extend([
                "⚠️ 系统基本满足生产要求，但需要加强监控",
                "📊 建议部署后前3天进行24小时监控",
                "🔄 建议设置告警阈值和自动恢复机制",
                "📝 建议准备回滚计划和应急响应流程"
            ])
        elif readiness == 'READY_WITH_ATTENTION':
            recommendations.extend([
                "⚠️ 系统需要重点关注，建议先在测试环境验证",
                "🔍 建议进行额外的集成测试和压力测试",
                "📊 建议完善监控和日志收集系统",
                "👥 建议增加运维人员支持"
            ])
        else:
            recommendations.extend([
                "❌ 系统暂不满足生产要求",
                "🔧 建议解决所有关键问题后再考虑部署",
                "🧪 建议在测试环境中进行全面验证",
                "📋 建议重新评估架构和实现方案"
            ])

        # 添加具体问题建议
        if critical_issues:
            recommendations.append(f"🚨 关键问题 ({len(critical_issues)}个):")
            for issue in critical_issues:
                recommendations.append(f"   - {issue}")

        if warnings:
            recommendations.append(f"⚠️ 需要注意的问题 ({len(warnings)}个):")
            for warning in warnings:
                recommendations.append(f"   - {warning}")

        return recommendations


def main():
    """主函数"""
    try:
        generator = FinalTestReportGenerator()
        report = generator.generate_final_report()

        # 保存最终报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = f"reports/FINAL_PRODUCTION_TEST_REPORT_{timestamp}.json"

        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        # 格式化输出
        final_report = report['final_test_report']
        assessment = final_report['final_assessment']

        print("\n" + "=" * 80)
        print("🎯 RQA2025 投产前最终测试验证报告")
        print("=" * 80)
        print(
            f"📅 测试日期: {datetime.fromisoformat(final_report['test_date'].replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 最终得分: {assessment.get('final_score', 0)}/100")
        print(f"🎯 总体状态: {assessment.get('overall_status', 'UNKNOWN')}")
        print(f"🚀 生产就绪: {assessment.get('production_readiness', 'UNKNOWN')}")

        print(f"\n📋 测试总结:")
        summary = final_report['test_summary']
        categories = summary.get('test_categories', {})

        if 'function_validation' in categories:
            func = categories['function_validation']
            print(f"   ✅ 功能验证: {func.get('passed_layers', 0)}/{func.get('total_layers', 0)} 层通过")

        if 'performance' in categories:
            perf = categories['performance']
            print(f"   ✅ 性能测试: {perf.get('performance_score', 0):.1f}分")

        if 'integration' in categories:
            integ = categories['integration']
            print(f"   ✅ 集成测试: {integ.get('success_rate', 0):.1f}% 成功率")

        if 'business_demo' in categories:
            biz = categories['business_demo']
            print(f"   ✅ 业务演示: {biz.get('success_rate', 0):.1f}% 成功率")

        if 'business_flow' in categories:
            flow = categories['business_flow']
            print(f"   ✅ 业务流程: {flow.get('flow_score', 0):.1f}分")

        print(f"\n🚨 关键问题 ({len(assessment.get('critical_issues', []))}个):")
        for issue in assessment.get('critical_issues', []):
            print(f"   ❌ {issue}")

        print(f"\n⚠️ 需要注意 ({len(assessment.get('warnings', []))}个):")
        for warning in assessment.get('warnings', []):
            print(f"   ⚠️ {warning}")

        print(f"\n📋 部署建议:")
        recommendations = final_report['deployment_recommendations']
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

        print(f"\n📄 详细报告已保存到: {json_file}")

        # 最终结论
        readiness = assessment.get('production_readiness', 'NOT_READY')
        if readiness == 'READY':
            print("\n🎉 恭喜！系统已达到生产就绪标准！")
            print("✅ 可以进行生产环境部署！")
        elif readiness in ['READY_WITH_MONITORING', 'READY_WITH_ATTENTION']:
            print(f"\n⚠️ 系统{readiness}，建议在严格监控下部署！")
        else:
            print("\n❌ 系统暂不满足生产要求，建议继续完善！")
        return 0

    except Exception as e:
        print(f"❌ 生成最终报告时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

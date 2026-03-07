#!/usr/bin/env python3
"""
性能优化脚本
基于系统集成测试的性能数据来优化系统性能
"""

from src.utils.logger import get_logger
import sys
import os
import json
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


class PerformanceOptimizer:
    """性能优化器"""

    def __init__(self):
        self.logger = get_logger(__name__)
        self.optimization_results = {}

    def analyze_performance_data(self, test_report_path: str) -> Dict[str, Any]:
        """分析性能测试数据"""
        try:
            with open(test_report_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)

            self.logger.info("📊 开始分析性能测试数据")

            # 提取性能指标
            performance_data = test_data.get('performance', {})
            if not performance_data:
                self.logger.error("未找到性能测试数据")
                return {}

            # 分析并发性能
            concurrency_analysis = self._analyze_concurrency_performance(performance_data)

            # 分析响应时间
            response_time_analysis = self._analyze_response_time(performance_data)

            # 分析吞吐量
            throughput_analysis = self._analyze_throughput(performance_data)

            # 生成性能分析报告
            analysis_report = {
                'timestamp': datetime.now().isoformat(),
                'concurrency_analysis': concurrency_analysis,
                'response_time_analysis': response_time_analysis,
                'throughput_analysis': throughput_analysis,
                'optimization_recommendations': self._generate_optimization_recommendations(
                    concurrency_analysis, response_time_analysis, throughput_analysis
                )
            }

            self.logger.info("✅ 性能数据分析完成")
            return analysis_report

        except Exception as e:
            self.logger.error(f"❌ 分析性能数据失败: {e}")
            return {}

    def _analyze_concurrency_performance(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析并发性能"""
        details = performance_data.get('details', {})

        analysis = {
            'low_concurrency': details.get('低并发测试', {}),
            'medium_concurrency': details.get('中并发测试', {}),
            'high_concurrency': details.get('高并发测试', {}),
            'scalability_analysis': {},
            'bottleneck_identification': {}
        }

        # 分析可扩展性
        low_rps = analysis['low_concurrency'].get('requests_per_second', 0)
        medium_rps = analysis['medium_concurrency'].get('requests_per_second', 0)
        high_rps = analysis['high_concurrency'].get('requests_per_second', 0)

        # 计算扩展效率
        if low_rps > 0:
            medium_efficiency = (medium_rps / 5) / (low_rps / 1)  # 5倍线程，1倍线程
            high_efficiency = (high_rps / 10) / (low_rps / 1)     # 10倍线程，1倍线程

            analysis['scalability_analysis'] = {
                'medium_concurrency_efficiency': medium_efficiency,
                'high_concurrency_efficiency': high_efficiency,
                'scalability_score': (medium_efficiency + high_efficiency) / 2
            }

        # 识别性能瓶颈
        if medium_rps < low_rps * 4:  # 中并发性能未达到预期
            analysis['bottleneck_identification']['medium_concurrency'] = "可能存在资源竞争或锁竞争"

        if high_rps < low_rps * 8:  # 高并发性能未达到预期
            analysis['bottleneck_identification']['high_concurrency'] = "可能存在内存瓶颈或CPU瓶颈"

        return analysis

    def _analyze_response_time(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析响应时间"""
        details = performance_data.get('details', {})
        response_time_data = details.get('响应时间测试', {})

        analysis = {
            'avg_response_time': response_time_data.get('avg_response_time', 0),
            'min_response_time': response_time_data.get('min_response_time', 0),
            'max_response_time': response_time_data.get('max_response_time', 0),
            'response_time_stability': {},
            'performance_grade': ''
        }

        # 分析响应时间稳定性
        if analysis['avg_response_time'] > 0:
            stability_ratio = (analysis['max_response_time'] -
                               analysis['min_response_time']) / analysis['avg_response_time']
            analysis['response_time_stability'] = {
                'stability_ratio': stability_ratio,
                'is_stable': stability_ratio < 2.0,  # 最大响应时间不超过平均值的2倍
                'variability': 'low' if stability_ratio < 1.0 else 'medium' if stability_ratio < 2.0 else 'high'
            }

        # 性能评级
        avg_time = analysis['avg_response_time']
        if avg_time < 0.01:  # 10ms
            analysis['performance_grade'] = 'A+'
        elif avg_time < 0.05:  # 50ms
            analysis['performance_grade'] = 'A'
        elif avg_time < 0.1:   # 100ms
            analysis['performance_grade'] = 'B'
        elif avg_time < 0.5:   # 500ms
            analysis['performance_grade'] = 'C'
        else:
            analysis['performance_grade'] = 'D'

        return analysis

    def _analyze_throughput(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析吞吐量"""
        details = performance_data.get('details', {})
        throughput_data = details.get('吞吐量测试', {})

        analysis = {
            'requests_per_second': throughput_data.get('requests_per_second', 0),
            'total_requests': throughput_data.get('total_requests', 0),
            'test_duration': throughput_data.get('execution_time', 0),
            'throughput_efficiency': {},
            'capacity_analysis': {}
        }

        # 分析吞吐量效率
        if analysis['test_duration'] > 0:
            actual_rps = analysis['total_requests'] / analysis['test_duration']
            efficiency = actual_rps / \
                analysis['requests_per_second'] if analysis['requests_per_second'] > 0 else 0

            analysis['throughput_efficiency'] = {
                'actual_rps': actual_rps,
                'efficiency_ratio': efficiency,
                'is_efficient': efficiency > 0.8  # 效率超过80%视为高效
            }

        # 容量分析
        if analysis['requests_per_second'] > 1000:
            analysis['capacity_analysis']['level'] = 'high'
            analysis['capacity_analysis']['description'] = '高吞吐量系统'
        elif analysis['requests_per_second'] > 100:
            analysis['capacity_analysis']['level'] = 'medium'
            analysis['capacity_analysis']['description'] = '中等吞吐量系统'
        else:
            analysis['capacity_analysis']['level'] = 'low'
            analysis['capacity_analysis']['description'] = '低吞吐量系统'

        return analysis

    def _generate_optimization_recommendations(self, concurrency_analysis: Dict[str, Any],
                                               response_time_analysis: Dict[str, Any],
                                               throughput_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成优化建议"""
        recommendations = []

        # 基于并发性能的优化建议
        scalability_score = concurrency_analysis.get(
            'scalability_analysis', {}).get('scalability_score', 0)
        if scalability_score < 0.8:
            recommendations.append({
                'category': '并发性能',
                'priority': 'high',
                'issue': '系统可扩展性不足',
                'recommendation': '优化锁机制，减少资源竞争，考虑使用无锁数据结构',
                'expected_improvement': '可扩展性提升20-30%'
            })

        bottlenecks = concurrency_analysis.get('bottleneck_identification', {})
        for bottleneck_type, description in bottlenecks.items():
            recommendations.append({
                'category': '性能瓶颈',
                'priority': 'high',
                'issue': f'{bottleneck_type}: {description}',
                'recommendation': '进行性能分析，识别具体瓶颈点，优化关键路径',
                'expected_improvement': '性能提升15-25%'
            })

        # 基于响应时间的优化建议
        response_stability = response_time_analysis.get('response_time_stability', {})
        if not response_stability.get('is_stable', True):
            recommendations.append({
                'category': '响应时间稳定性',
                'priority': 'medium',
                'issue': '响应时间不稳定，存在较大波动',
                'recommendation': '优化算法复杂度，减少异常情况处理时间，增加缓存机制',
                'expected_improvement': '响应时间稳定性提升30-40%'
            })

        performance_grade = response_time_analysis.get('performance_grade', '')
        if performance_grade in ['C', 'D']:
            recommendations.append({
                'category': '响应时间性能',
                'priority': 'high',
                'issue': f'响应时间性能评级较低: {performance_grade}',
                'recommendation': '优化算法实现，使用更高效的数据结构，考虑异步处理',
                'expected_improvement': '响应时间减少40-60%'
            })

        # 基于吞吐量的优化建议
        throughput_efficiency = throughput_analysis.get('throughput_efficiency', {})
        if not throughput_efficiency.get('is_efficient', True):
            recommendations.append({
                'category': '吞吐量效率',
                'priority': 'medium',
                'issue': '吞吐量效率不足，实际性能低于理论值',
                'recommendation': '优化I/O操作，减少系统调用开销，优化内存分配策略',
                'expected_improvement': '吞吐量效率提升20-30%'
            })

        # 通用优化建议
        recommendations.append({
            'category': '系统优化',
            'priority': 'low',
            'issue': '系统整体性能良好，可进行微调优化',
            'recommendation': '定期进行性能监控，持续优化热点代码，保持系统性能',
            'expected_improvement': '性能持续提升5-10%'
        })

        return recommendations

    def generate_optimization_report(self, analysis_report: Dict[str, Any]) -> str:
        """生成优化报告"""
        report_path = f"reports/performance_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_report, f, ensure_ascii=False, indent=2)

            self.logger.info(f"📊 性能优化报告已生成: {report_path}")
            return report_path

        except Exception as e:
            self.logger.error(f"❌ 生成优化报告失败: {e}")
            return ""

    def print_optimization_summary(self, analysis_report: Dict[str, Any]):
        """打印优化摘要"""
        if not analysis_report:
            self.logger.error("分析报告为空，无法生成摘要")
            return

        print("\n" + "="*80)
        print("🚀 性能优化分析摘要")
        print("="*80)
        print(f"📅 分析时间: {analysis_report.get('timestamp', 'N/A')}")

        # 并发性能分析
        concurrency_analysis = analysis_report.get('concurrency_analysis', {})
        scalability_analysis = concurrency_analysis.get('scalability_analysis', {})
        if scalability_analysis:
            print(f"\n📊 并发性能分析:")
            print(f"  可扩展性评分: {scalability_analysis.get('scalability_score', 0):.2f}")
            print(f"  中并发效率: {scalability_analysis.get('medium_concurrency_efficiency', 0):.2f}")
            print(f"  高并发效率: {scalability_analysis.get('high_concurrency_efficiency', 0):.2f}")

        # 响应时间分析
        response_time_analysis = analysis_report.get('response_time_analysis', {})
        print(f"\n⏱️  响应时间分析:")
        print(f"  平均响应时间: {response_time_analysis.get('avg_response_time', 0):.3f}秒")
        print(f"  性能评级: {response_time_analysis.get('performance_grade', 'N/A')}")
        stability = response_time_analysis.get('response_time_stability', {})
        print(f"  稳定性: {'稳定' if stability.get('is_stable', False) else '不稳定'}")

        # 吞吐量分析
        throughput_analysis = analysis_report.get('throughput_analysis', {})
        print(f"\n📈 吞吐量分析:")
        print(f"  请求/秒: {throughput_analysis.get('requests_per_second', 0):.2f}")
        print(f"  容量等级: {throughput_analysis.get('capacity_analysis', {}).get('description', 'N/A')}")

        # 优化建议
        recommendations = analysis_report.get('optimization_recommendations', [])
        print(f"\n🔧 优化建议 ({len(recommendations)}项):")
        for i, rec in enumerate(recommendations, 1):
            priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            print(f"  {i}. {priority_icon} [{rec['category']}] {rec['issue']}")
            print(f"      💡 建议: {rec['recommendation']}")
            print(f"      📈 预期改进: {rec['expected_improvement']}")

        print("="*80)


def main():
    """主函数"""
    print("🚀 开始性能优化分析")

    # 查找最新的集成测试报告
    reports_dir = "reports"
    test_reports = [f for f in os.listdir(reports_dir) if f.startswith(
        'system_integration_test_report_')]

    if not test_reports:
        print("❌ 未找到系统集成测试报告")
        return 1

    # 使用最新的报告
    latest_report = sorted(test_reports)[-1]
    report_path = os.path.join(reports_dir, latest_report)

    print(f"📋 使用测试报告: {latest_report}")

    # 创建性能优化器
    optimizer = PerformanceOptimizer()

    try:
        # 分析性能数据
        analysis_report = optimizer.analyze_performance_data(report_path)

        if not analysis_report:
            print("❌ 性能数据分析失败")
            return 1

        # 生成优化报告
        report_path = optimizer.generate_optimization_report(analysis_report)

        # 打印优化摘要
        optimizer.print_optimization_summary(analysis_report)

        if report_path:
            print(f"\n📊 详细优化报告已保存到: {report_path}")

        print("\n🎉 性能优化分析完成！")
        return 0

    except Exception as e:
        print(f"\n❌ 性能优化分析失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

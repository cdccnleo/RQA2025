#!/usr/bin/env python3
"""
RQA2025 AI质量保障系统使用示例

展示如何使用AI质量保障系统的核心功能：
1. 质量数据分析
2. 异常预测
3. 性能优化建议
4. 质量趋势分析
5. 智能决策支持
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def create_sample_quality_data(days=30):
    """创建示例质量数据"""
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='D')

    # 模拟质量指标数据
    quality_data = pd.DataFrame({
        'timestamp': dates,
        'test_coverage': np.random.normal(85, 5, days),
        'test_success_rate': np.random.normal(92, 3, days),
        'code_quality_score': np.random.normal(8.0, 0.5, days),
        'error_rate': np.random.normal(0.05, 0.02, days),
        'response_time': np.random.normal(200, 50, days),
        'throughput': np.random.normal(1000, 200, days),
        'cpu_usage': np.random.normal(70, 10, days),
        'memory_usage': np.random.normal(75, 8, days),
        'disk_usage': np.random.normal(60, 5, days)
    })

    return quality_data

def demo_anomaly_prediction():
    """异常预测功能演示"""
    print("\n🔍 === 异常预测功能演示 ===")

    try:
        from ai_quality.anomaly_prediction import AnomalyPredictionEngine

        # 创建引擎
        engine = AnomalyPredictionEngine()

        # 生成训练数据
        training_data = create_sample_quality_data(60)
        print(f"生成训练数据: {len(training_data)} 条记录")

        # 训练模型
        print("正在训练异常预测模型...")
        training_result = engine.train_model(training_data)
        print(f"训练完成: {training_result}")

        # 模拟当前系统状态
        current_metrics = {
            'test_coverage': 78.5,  # 偏低
            'test_success_rate': 88.2,  # 偏低
            'error_rate': 0.12,  # 偏高
            'response_time': 350,  # 偏高
            'cpu_usage': 92,  # 偏高
            'memory_usage': 88  # 偏高
        }

        print(f"当前系统指标: {current_metrics}")

        # 预测异常
        prediction = engine.predict_anomalies(current_metrics)
        print(f"异常预测结果: {prediction}")

        if prediction.get('anomaly_detected'):
            print("🚨 检测到系统异常!")
            print(f"异常严重程度: {prediction.get('severity')}")
            print(f"异常指标: {prediction.get('anomalous_metrics')}")
            print(f"置信度: {prediction.get('confidence', 0):.2%}")

    except Exception as e:
        print(f"异常预测演示失败: {e}")

def demo_performance_optimization():
    """性能优化功能演示"""
    print("\n⚡ === 性能优化功能演示 ===")

    try:
        from ai_quality.performance_optimization import PerformanceAnalyzer

        # 创建分析器
        analyzer = PerformanceAnalyzer()

        # 模拟性能数据
        performance_data = {
            'response_time': 450,  # 毫秒，偏高
            'throughput': 650,     # 请求/分钟，偏低
            'cpu_usage': 85,       # %，偏高
            'memory_usage': 82,    # %，偏高
            'disk_io': 1200,       # KB/s，偏高
            'network_io': 800,     # KB/s，正常
            'active_connections': 1250,  # 偏高
            'queue_length': 45     # 偏高
        }

        print(f"当前性能指标: {performance_data}")

        # 分析性能
        analysis_result = analyzer.analyze_performance(performance_data)
        print("性能分析结果:")
        print(f"- 性能分数: {analysis_result.get('performance_score', 0):.2f}")
        print(f"- 瓶颈识别: {analysis_result.get('bottlenecks', [])}")
        print(f"- 优化建议: {len(analysis_result.get('optimization_suggestions', []))} 条")

        # 显示优化建议
        suggestions = analysis_result.get('optimization_suggestions', [])
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"  {i}. {suggestion.get('title', 'Unknown')}")
            print(f"     优先级: {suggestion.get('priority', 'medium')}")
            print(f"     预期改善: {suggestion.get('expected_improvement', 'unknown')}")

    except Exception as e:
        print(f"性能优化演示失败: {e}")

def demo_quality_trend_analysis():
    """质量趋势分析功能演示"""
    print("\n📈 === 质量趋势分析功能演示 ===")

    try:
        from ai_quality.quality_trend_analysis import QualityTrendAnalyzer

        # 创建分析器
        analyzer = QualityTrendAnalyzer()

        # 生成趋势数据
        trend_data = create_sample_quality_data(90)  # 90天的数据
        print(f"生成趋势分析数据: {len(trend_data)} 条记录")

        # 分析质量趋势
        analysis_result = analyzer.analyze_quality_trends(trend_data)

        print("质量趋势分析结果:")
        print(f"- 整体质量趋势: {analysis_result.get('overall_trend', 'unknown')}")
        print(f"- 预测准确性: {analysis_result.get('prediction_accuracy', 0):.2%}")
        print(f"- 检测到的异常: {len(analysis_result.get('anomalies', []))} 个")

        # 显示趋势详情
        trends = analysis_result.get('metric_trends', {})
        print("各指标趋势:")
        for metric, trend_info in list(trends.items())[:5]:
            print(f"- {metric}: {trend_info.get('trend', 'unknown')} "
                  f"(当前值: {trend_info.get('current_value', 0):.2f})")

    except Exception as e:
        print(f"质量趋势分析演示失败: {e}")

def demo_test_generation():
    """自动化测试生成功能演示"""
    print("\n🧪 === 自动化测试生成功能演示 ===")

    try:
        from ai_quality.test_generation import AutomatedTestGenerator

        # 创建测试生成器
        generator = AutomatedTestGenerator()

        # 模拟业务场景
        business_scenario = {
            'module': 'trading_engine',
            'functionality': 'order_execution',
            'complexity': 'high',
            'dependencies': ['market_data', 'risk_management', 'portfolio'],
            'edge_cases': ['market_volatility', 'connection_failure', 'data_anomaly']
        }

        print(f"业务场景: {business_scenario}")

        # 生成测试用例
        test_cases = generator.generate_test_cases(business_scenario)

        print("生成的测试用例:")
        print(f"- 单元测试: {len(test_cases.get('unit_tests', []))} 个")
        print(f"- 集成测试: {len(test_cases.get('integration_tests', []))} 个")
        print(f"- 端到端测试: {len(test_cases.get('e2e_tests', []))} 个")

        # 显示部分测试用例
        unit_tests = test_cases.get('unit_tests', [])
        if unit_tests:
            print("\\n示例单元测试用例:")
            for i, test in enumerate(unit_tests[:3], 1):
                print(f"{i}. {test.get('name', 'Unknown')}")
                print(f"   描述: {test.get('description', '')}")
                print(f"   优先级: {test.get('priority', 'medium')}")

    except Exception as e:
        print(f"自动化测试生成演示失败: {e}")

def demo_decision_support():
    """智能决策支持功能演示"""
    print("\n🧠 === 智能决策支持功能演示 ===")

    try:
        from ai_quality.decision_support_system import QualityAIDecisionSupportSystem

        # 创建决策支持系统
        dss = QualityAIDecisionSupportSystem()

        # 模拟当前质量状态
        quality_metrics = {
            'test_coverage': 82.5,
            'test_success_rate': 89.2,
            'error_rate': 0.08,
            'response_time': 280,
            'cpu_usage': 78,
            'memory_usage': 75,
            'code_quality_score': 7.8,
            'security_score': 8.2
        }

        # 模拟历史数据
        historical_data = create_sample_quality_data(30)

        # 模拟风险告警
        risk_alerts = [
            {
                'alert_id': 'alert_001',
                'title': '测试覆盖率下降',
                'severity': 'medium',
                'probability': 0.8,
                'affected_components': ['testing_framework']
            }
        ]

        print(f"当前质量指标: {quality_metrics}")

        # 执行综合质量分析
        analysis_result = dss.perform_comprehensive_quality_analysis(
            quality_metrics, historical_data, risk_alerts
        )

        print("综合质量分析结果:")
        assessment = analysis_result.get('quality_assessment', {})
        print(f"- 整体质量分数: {assessment.get('overall_score', 0):.2f}")
        print(f"- 风险等级: {assessment.get('risk_level', 'unknown')}")
        print(f"- 质量趋势: {assessment.get('trend_direction', 'unknown')}")

        recommendations = analysis_result.get('decision_recommendations', [])
        print(f"- 决策建议数量: {len(recommendations)}")

        # 显示前3个建议
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"  {i}. {rec.get('title', 'Unknown')} (优先级: {rec.get('priority', 'medium')})")

    except Exception as e:
        print(f"智能决策支持演示失败: {e}")

def main():
    """主函数"""
    print("🎯 RQA2025 AI质量保障系统使用示例")
    print("=" * 50)

    # 演示各个功能
    demo_anomaly_prediction()
    demo_performance_optimization()
    demo_quality_trend_analysis()
    demo_test_generation()
    demo_decision_support()

    print("\n" + "=" * 50)
    print("✅ AI质量保障系统功能演示完成!")
    print("\\n💡 提示:")
    print("- 使用 start_ai_quality_system.py 启动完整系统")
    print("- 查看 test_logs/ 目录中的报告")
    print("- 参考 src/ai_quality/ 目录中的详细实现")
    print("- 根据实际需求定制和扩展功能")

if __name__ == "__main__":
    main()

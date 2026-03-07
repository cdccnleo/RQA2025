#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能Logger使用示例

演示基于AI和使用模式的智能Logger优化功能。
"""

from infrastructure.logging.intelligent import (
    UsageAnalyzer,
    AdaptiveLogger,
    SmartLogger,
    LogAnomalyDetector,
    PredictiveOptimizer,
    LogPerformancePredictor
)
import time
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def create_intelligent_logger(logger_name: str):
    """创建智能Logger实例"""
    print(f"创建智能Logger: {logger_name}")

    # 创建使用模式分析器
    usage_analyzer = UsageAnalyzer(
        analysis_window=1800,  # 30分钟分析窗口
        update_interval=60     # 每分钟更新一次
    )

    # 创建异常检测器
    anomaly_detector = LogAnomalyDetector(
        detection_window=900,  # 15分钟检测窗口
        sensitivity=0.7        # 检测灵敏度
    )

    # 创建预测优化器
    predictive_optimizer = PredictiveOptimizer(
        usage_analyzer=usage_analyzer,
        prediction_window=3600  # 1小时预测窗口
    )

    # 创建自适应Logger
    adaptive_logger = AdaptiveLogger(
        name=logger_name,
        usage_analyzer=usage_analyzer,
        adaptation_strategy="balanced",
        adaptation_interval=300  # 5分钟调整一次
    )

    # 创建智能Logger
    smart_logger = SmartLogger(
        name=f"smart_{logger_name}",
        usage_analyzer=usage_analyzer,
        anomaly_detector=anomaly_detector,
        predictive_optimizer=predictive_optimizer,
        adaptation_strategy="balanced"
    )

    return {
        'usage_analyzer': usage_analyzer,
        'anomaly_detector': anomaly_detector,
        'predictive_optimizer': predictive_optimizer,
        'adaptive_logger': adaptive_logger,
        'smart_logger': smart_logger
    }


def simulate_logger_usage(logger_components, duration_minutes: int = 10):
    """模拟Logger使用"""
    print(f"\n开始模拟Logger使用 ({duration_minutes}分钟)...")

    adaptive_logger = logger_components['adaptive_logger']
    smart_logger = logger_components['smart_logger']

    # 启动组件
    logger_components['usage_analyzer'].start()
    adaptive_logger.start()
    smart_logger.start()

    # 模拟不同类型的日志记录
    log_patterns = [
        # (级别, 消息, 分类, 频率, 持续时间)
        ("INFO", "用户登录", "BUSINESS", 5, duration_minutes * 60),      # 每12秒一次
        ("DEBUG", "缓存命中", "PERFORMANCE", 10, duration_minutes * 60),  # 每6秒一次
        ("WARNING", "高负载警告", "SYSTEM", 2, duration_minutes * 60),    # 每30秒一次
        ("ERROR", "数据库连接失败", "DATABASE", 0.5, duration_minutes * 60),  # 每2分钟一次
        ("INFO", "交易执行", "TRADING", 8, duration_minutes * 60),       # 每7.5秒一次
    ]

    start_time = time.time()

    try:
        while time.time() - start_time < duration_minutes * 60:
            current_time = time.time()

            for level_name, message, category, frequency, end_time in log_patterns:
                if current_time > start_time + end_time:
                    continue

                # 计算是否应该记录此类型的日志
                time_since_start = current_time - start_time
                expected_logs = time_since_start * (frequency / 60)  # 每分钟频率转换为累计数量

                # 简单的泊松过程模拟
                import random
                import logging
                if random.random() < (frequency / 60) * 0.1:  # 每秒10%概率
                    try:
                        # 将字符串级别映射到整数级别
                        level_map = {
                            "DEBUG": logging.DEBUG,
                            "INFO": logging.INFO,
                            "WARNING": logging.WARNING,
                            "ERROR": logging.ERROR,
                            "CRITICAL": logging.CRITICAL
                        }
                        level_value = level_map.get(level_name, logging.INFO)

                        # 自适应Logger记录
                        adaptive_logger.log(level_value, message)

                        # 智能Logger记录（带异常检测）
                        smart_logger.log(level_value, message)

                    except Exception as e:
                        print(f"日志记录失败: {e}")

            time.sleep(0.1)  # 短暂延迟

    except KeyboardInterrupt:
        print("模拟被用户中断")

    finally:
        # 停止组件
        smart_logger.stop()
        adaptive_logger.stop()
        logger_components['usage_analyzer'].stop()

    print("Logger使用模拟完成")


def analyze_usage_patterns(logger_components):
    """分析使用模式"""
    print("\n=== 使用模式分析 ===")

    usage_analyzer = logger_components['usage_analyzer']
    analytics = usage_analyzer.get_usage_analytics()

    print(f"总日志条目: {analytics.total_log_entries}")
    print(f"活跃Logger数量: {len(analytics.patterns_by_logger)}")

    print("\n各Logger使用统计:")
    for logger_name, pattern in analytics.patterns_by_logger.items():
        print(f"  {logger_name}:")
        print(f"    频率: {pattern.frequency:.1f} 次/分钟")
        print(f"    错误率: {pattern.error_rate:.2%}")
        print(f"    性能影响: {pattern.performance_impact:.2f}")
        print(f"    常见级别: {pattern.level}")
        print(f"    峰值时段: {pattern.peak_hours}")

    print(f"\n优化建议: {len(analytics.recommendations)} 条")
    for i, rec in enumerate(analytics.recommendations, 1):
        print(f"  {i}. {rec}")


def demonstrate_anomaly_detection(logger_components):
    """演示异常检测"""
    print("\n=== 异常检测演示 ===")

    anomaly_detector = logger_components['anomaly_detector']

    # 模拟一些异常日志
    test_logs = [
        {"level": 20, "message": "正常的用户登录", "logger_name": "test_logger"},
        {"level": 30, "message": "重复的错误: 连接超时", "logger_name": "test_logger"},
        {"level": 30, "message": "重复的错误: 连接超时", "logger_name": "test_logger"},
        {"level": 30, "message": "重复的错误: 连接超时", "logger_name": "test_logger"},
        {"level": 30, "message": "重复的错误: 连接超时", "logger_name": "test_logger"},
        {"level": 30, "message": "重复的错误: 连接超时", "logger_name": "test_logger"},
        {"level": 40, "message": "严重错误: 系统崩溃", "logger_name": "test_logger"},
        {"level": 20, "message": "非常长的错误消息" + "x" * 1000, "logger_name": "test_logger"},
    ]

    anomalies_detected = 0

    for log_entry in test_logs:
        is_anomaly = anomaly_detector.detect_anomaly(log_entry)
        if is_anomaly:
            anomalies_detected += 1
            print(f"✓ 检测到异常: {log_entry['message'][:50]}...")

        time.sleep(0.1)

    print(f"\n检测到 {anomalies_detected} 个异常")

    # 显示异常统计
    anomaly_stats = anomaly_detector.get_anomaly_stats()
    print(f"活跃异常: {anomaly_stats['active_anomalies']}")
    print(f"异常检测总数: {anomaly_stats['detection_count']}")


def demonstrate_adaptive_behavior(logger_components):
    """演示自适应行为"""
    print("\n=== 自适应行为演示 ===")

    adaptive_logger = logger_components['adaptive_logger']
    smart_logger = logger_components['smart_logger']

    print("初始配置:")
    adaptive_stats = adaptive_logger.get_adaptation_stats()
    smart_stats = smart_logger.get_smart_stats()

    print(
        f"自适应Logger - 采样率: {adaptive_stats['current_config']['sampling_rate']}, 批处理: {adaptive_stats['current_config']['batch_size']}")
    print(
        f"智能Logger - 采样率: {smart_stats['current_config']['sampling_rate']}, 批处理: {smart_stats['current_config']['batch_size']}")

    # 等待适应调整
    print("\n等待适应调整 (10秒)...")
    time.sleep(10)

    print("调整后配置:")
    adaptive_stats = adaptive_logger.get_adaptation_stats()
    smart_stats = smart_logger.get_smart_stats()

    print(
        f"自适应Logger - 采样率: {adaptive_stats['current_config']['sampling_rate']}, 批处理: {adaptive_stats['current_config']['batch_size']}")
    print(
        f"智能Logger - 采样率: {smart_stats['current_config']['sampling_rate']}, 批处理: {smart_stats['current_config']['batch_size']}")

    # 显示适应历史
    adaptive_history = adaptive_logger.get_adaptation_history(limit=3)
    if adaptive_history:
        print(f"\n适应调整历史: {len(adaptive_history)} 次调整")
        for i, record in enumerate(adaptive_history, 1):
            print(f"  {i}. {record['reason']} - 改善: {record['expected_impact']:.1f}%")


def demonstrate_predictive_optimization(logger_components):
    """演示预测优化"""
    print("\n=== 预测优化演示 ===")

    predictive_optimizer = logger_components['predictive_optimizer']

    # 模拟性能记录
    test_configs = [
        {'sampling_rate': 1.0, 'batch_size': 1},
        {'sampling_rate': 0.8, 'batch_size': 1},
        {'sampling_rate': 0.8, 'batch_size': 3},
        {'sampling_rate': 0.5, 'batch_size': 5},
    ]

    test_metrics = [
        {'processing_time': 0.002, 'memory_usage': 1024*1024, 'throughput': 500},
        {'processing_time': 0.0016, 'memory_usage': 1024*1024, 'throughput': 400},
        {'processing_time': 0.0014, 'memory_usage': 1536*1024, 'throughput': 600},
        {'processing_time': 0.001, 'memory_usage': 2048*1024, 'throughput': 300},
    ]

    # 记录性能数据
    for config, metrics in zip(test_configs, test_metrics):
        predictive_optimizer.record_performance(
            'test_logger', config, metrics
        )

    # 创建一个模拟的使用模式
    from infrastructure.logging.intelligent.usage_analyzer import LogUsagePattern
    mock_pattern = LogUsagePattern(
        logger_name='test_logger',
        level='INFO',
        category='TEST',
        frequency=100,  # 每分钟100次
        error_rate=0.05,  # 5%错误率
        performance_impact=0.6  # 中等性能影响
    )

    # 获取预测
    prediction = predictive_optimizer.predict_optimal_config(mock_pattern)

    if prediction:
        print("预测的最优配置:")
        print(f"  采样率: {prediction.get('sampling_rate', 'N/A')}")
        print(f"  批处理大小: {prediction.get('batch_size', 'N/A')}")
        # 注意：这里返回的是配置字典，不是PredictionResult对象
        print("  (预测配置已生成，但详细信息不可用)")
    else:
        print("无法生成预测，可能是因为数据不足")


def demonstrate_performance_prediction(logger_components):
    """演示性能预测"""
    print("\n=== 性能预测演示 ===")

    # 创建性能预测器
    performance_predictor = LogPerformancePredictor(
        usage_analyzer=logger_components['usage_analyzer'],
        prediction_window=3600
    )

    # 预测不同配置的性能
    test_configs = [
        {'sampling_rate': 1.0, 'batch_size': 1, 'compression_enabled': False},
        {'sampling_rate': 0.7, 'batch_size': 3, 'compression_enabled': True},
        {'sampling_rate': 0.5, 'batch_size': 5, 'compression_enabled': True},
    ]

    print("性能预测结果:")
    for i, config in enumerate(test_configs, 1):
        prediction = performance_predictor.predict_log_performance('test_logger', config)

        print(f"\n配置 {i}:")
        print(f"  配置: {config}")
        print(f"  预期改善: {prediction['base_prediction']['expected_improvement']:.1f}%")
        print(f"  风险评估: {prediction['base_prediction']['risk_assessment']}")
        print(f"  日志大小: {prediction['log_specific_metrics']['estimated_log_size']:.0f} 字节/条")
        print(f"  存储成本: ${prediction['log_specific_metrics']['estimated_storage_cost']:.3f}/月")


def main():
    """主函数"""
    print("RQA2025 智能Logger系统演示")
    print("=" * 50)

    # 创建智能Logger组件
    logger_name = "intelligent_demo"
    components = create_intelligent_logger(logger_name)

    try:
        # 运行各种演示
        simulate_logger_usage(components, duration_minutes=2)  # 缩短演示时间
        analyze_usage_patterns(components)
        demonstrate_anomaly_detection(components)
        demonstrate_adaptive_behavior(components)
        demonstrate_predictive_optimization(components)
        demonstrate_performance_prediction(components)

        print(f"\n=== 演示完成 ===")
        print(f"智能Logger系统展示了完整的AI驱动优化能力:")
        print(f"• 使用模式分析和异常检测")
        print(f"• 自适应配置调整")
        print(f"• 预测优化和性能评估")
        print(f"• 基于机器学习的持续学习")

    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        try:
            for component in components.values():
                if hasattr(component, 'stop'):
                    component.stop()
        except Exception as e:
            print(f"清理资源时出错: {e}")

    print("\n智能Logger演示完成")


if __name__ == "__main__":
    main()

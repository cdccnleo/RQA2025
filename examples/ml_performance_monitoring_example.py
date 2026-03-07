#!/usr/bin/env python3
"""
RQA2025 ML性能监控示例

演示ML性能监控系统的功能，包括：
1. 实时性能指标收集
2. 监控面板展示
3. 告警机制
4. 性能报告生成
"""

import logging
import time
import random
from datetime import datetime

from src.ml.performance_monitor import (
    start_ml_monitoring, stop_ml_monitoring,
    record_inference_performance, record_model_performance,
    get_ml_performance_stats
)
from src.ml.monitoring_dashboard import (
    start_ml_dashboard, stop_ml_dashboard,
    get_ml_dashboard_data, get_ml_health_score
)
from src.ml.process_orchestrator import (
    create_ml_process, submit_ml_process,
    get_ml_process_status
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simulate_inference_workload():
    """模拟推理工作负载"""
    logger.info("开始模拟推理工作负载...")

    # 模拟不同的延迟分布
    latency_patterns = [
        (50, 10),   # 正常延迟: 50ms ± 10ms
        (200, 50),  # 中等延迟: 200ms ± 50ms
        (1000, 200),  # 高延迟: 1000ms ± 200ms
        (5000, 1000)  # 异常延迟: 5000ms ± 1000ms
    ]

    for i in range(100):
        # 随机选择延迟模式
        pattern = random.choice(latency_patterns)
        base_latency, variance = pattern

        # 生成延迟值
        latency = random.gauss(base_latency, variance)
        latency = max(10, min(10000, latency))  # 限制在合理范围内

        # 模拟偶尔出现的错误
        if random.random() < 0.05:  # 5%错误率
            record_inference_performance(latency, "test_model", "SimulatedError")
        else:
            record_inference_performance(latency, "test_model")

        # 模拟吞吐量变化
        time.sleep(random.uniform(0.01, 0.1))  # 10-100ms间隔

    logger.info("推理工作负载模拟完成")


def simulate_model_evaluation():
    """模拟模型评估过程"""
    logger.info("开始模拟模型评估...")

    # 模拟不同模型的性能指标
    models = ["rf_v1", "xgb_v2", "lgb_v1", "nn_v3"]

    for model_id in models:
        # 生成随机的但合理的性能指标
        accuracy = random.uniform(0.75, 0.95)
        precision = random.uniform(0.70, 0.92)
        recall = random.uniform(0.72, 0.94)
        f1_score = random.uniform(0.71, 0.93)

        record_model_performance(accuracy, precision, recall, f1_score, model_id)

        logger.info(f"记录模型 {model_id} 性能: Acc={accuracy:.3f}, F1={f1_score:.3f}")
        time.sleep(0.5)

    logger.info("模型评估模拟完成")


def demonstrate_monitoring_dashboard():
    """演示监控面板功能"""
    logger.info("开始演示监控面板...")

    # 获取监控面板数据
    dashboard_data = get_ml_dashboard_data()
    logger.info(f"监控面板数据时间戳: {dashboard_data['timestamp']}")

    # 显示当前统计
    current_stats = dashboard_data['current_stats']
    if 'inference' in current_stats:
        inference = current_stats['inference']
        logger.info(f"推理统计: 请求数={inference.get('total_requests', 0)}, "
                    f"平均延迟={inference.get('avg_latency_ms', 0):.2f}ms")

    if 'model' in current_stats:
        model = current_stats['model']
        logger.info(f"模型统计: 评估次数={model.get('total_evaluations', 0)}, "
                    f"平均准确率={model.get('avg_accuracy', 0):.3f}")

    # 获取健康评分
    health_score = get_ml_health_score()
    logger.info(f"系统健康评分: {health_score:.1f}/100")

    logger.info("监控面板演示完成")


def create_sample_ml_process():
    """创建示例ML流程"""
    logger.info("创建示例ML流程...")

    process = create_ml_process(
        process_type="model_training",
        process_name="性能监控示例流程",
        steps={
            'data_load': {
                'step_id': 'data_load',
                'step_name': '数据加载',
                'step_type': 'data_loading',
                'config': {
                    'data_source': 'file',
                    'data_format': 'csv',
                    'data_path': 'data/sample.csv'
                }
            },
            'feature_eng': {
                'step_id': 'feature_eng',
                'step_name': '特征工程',
                'step_type': 'feature_engineering',
                'dependencies': ['data_load'],
                'config': {
                    'target_column': 'target',
                    'scaling_method': 'standard'
                }
            },
            'train': {
                'step_id': 'train',
                'step_name': '模型训练',
                'step_type': 'model_training',
                'dependencies': ['feature_eng'],
                'config': {
                    'model_type': 'random_forest',
                    'model_config': {'n_estimators': 50}
                }
            }
        },
        config={'experiment_name': 'performance_monitoring_demo'},
        priority=2,
        timeout=300
    )

    # 提交流程
    process_id = submit_ml_process(process)
    logger.info(f"已提交流程: {process_id}")

    return process_id


def monitor_process_execution(process_id):
    """监控流程执行"""
    logger.info(f"开始监控流程 {process_id}...")

    max_wait = 60  # 最多等待60秒
    waited = 0

    while waited < max_wait:
        status = get_ml_process_status(process_id)

        if status:
            logger.info(f"流程状态: {status['status']} "
                        f"(进度: {status['progress']:.1%})")

            if status['status'] in ['completed', 'failed']:
                logger.info(f"流程最终状态: {status['status']}")
                break

        time.sleep(5)
        waited += 5

    if waited >= max_wait:
        logger.warning(f"流程 {process_id} 监控超时")


def demonstrate_alert_system():
    """演示告警系统"""
    logger.info("演示告警系统...")

    # 模拟一些可能触发告警的情况
    for i in range(10):
        # 模拟高延迟
        high_latency = random.uniform(1500, 3000)  # 1.5-3秒
        record_inference_performance(high_latency, "high_latency_model")

        # 模拟错误
        if i % 3 == 0:  # 每3次就有1次错误
            record_inference_performance(500, "error_model", "SimulatedTimeoutError")

        time.sleep(1)

    logger.info("告警系统演示完成，请检查监控面板的告警信息")


def generate_performance_report():
    """生成性能报告"""
    logger.info("生成性能报告...")

    # 获取当前性能统计
    stats = get_ml_performance_stats()

    # 简单的报告输出
    logger.info("=" * 60)
    logger.info("ML性能监控报告")
    logger.info("=" * 60)
    logger.info(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if 'inference' in stats:
        inference = stats['inference']
        logger.info("\n推理性能:")
        logger.info(f"  总请求数: {inference.get('total_requests', 0)}")
        logger.info(f"  平均延迟: {inference.get('avg_latency_ms', 0):.2f}ms")
        logger.info(f"  P95延迟: {inference.get('p95_latency_ms', 0):.2f}ms")
        logger.info(f"  错误率: {inference.get('error_rate', 0):.2%}")

    if 'model' in stats:
        model = stats['model']
        logger.info("\n模型性能:")
        logger.info(f"  评估次数: {model.get('total_evaluations', 0)}")
        logger.info(f"  平均准确率: {model.get('avg_accuracy', 0):.3f}")
        logger.info(f"  平均F1得分: {model.get('avg_f1_score', 0):.3f}")

    if 'resources' in stats:
        resources = stats['resources']
        logger.info("\n资源使用:")
        logger.info(f"  CPU使用率: {resources.get('cpu_avg_percent', 0):.1f}%")
        logger.info(f"  内存使用率: {resources.get('memory_avg_percent', 0):.1f}%")

    health_score = get_ml_health_score()
    logger.info(f"\n系统健康评分: {health_score:.1f}/100")

    logger.info("=" * 60)


def main():
    """主函数"""
    logger.info("RQA2025 ML性能监控示例开始")

    try:
        # 启动性能监控
        logger.info("启动ML性能监控...")
        start_ml_monitoring()

        # 启动监控面板
        logger.info("启动ML监控面板...")
        start_ml_dashboard()

        # 等待系统初始化
        time.sleep(2)

        # 1. 模拟推理工作负载
        simulate_inference_workload()

        # 2. 模拟模型评估
        simulate_model_evaluation()

        # 3. 演示监控面板
        demonstrate_monitoring_dashboard()

        # 4. 创建并监控ML流程
        process_id = create_sample_ml_process()
        monitor_process_execution(process_id)

        # 5. 演示告警系统
        demonstrate_alert_system()

        # 6. 生成性能报告
        time.sleep(5)  # 等待数据收集
        generate_performance_report()

        logger.info("所有ML性能监控示例完成")

    except Exception as e:
        logger.error(f"ML性能监控示例执行失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源
        try:
            stop_ml_dashboard()
            stop_ml_monitoring()
            logger.info("ML性能监控资源清理完成")
        except Exception as e:
            logger.error(f"资源清理失败: {e}")


if __name__ == "__main__":
    main()

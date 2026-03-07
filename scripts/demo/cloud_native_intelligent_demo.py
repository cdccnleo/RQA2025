#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
云原生和智能化功能演示脚本
"""

import asyncio
import time
import logging
from src.backtest.cloud_native_features import (
    ScalingPolicy, AutoScaler, CircuitBreaker, BlueGreenDeployment,
    BlueGreenConfig
)
from src.backtest.intelligent_features import (
    MLModelConfig, AutoTuningConfig, PredictiveMaintenanceConfig,
    MLModel, AutoTuner, PredictiveMaintenance, IntelligentMonitor as IntelligentMonitor2
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_cloud_native_features():
    """演示云原生功能"""
    logger.info("=== 云原生功能演示 ===")

    # 1. 自动扩缩容演示
    logger.info("1. 自动扩缩容演示")
    scaling_policy = ScalingPolicy(
        min_replicas=1,
        max_replicas=5,
        target_cpu_utilization=70.0,
        target_memory_utilization=80.0
    )
    auto_scaler = AutoScaler(scaling_policy)

    # 模拟高负载
    logger.info("模拟高负载情况...")
    for i in range(10):
        auto_scaler.record_metrics(85.0, 90.0, 150, 200.0)
        time.sleep(0.1)

    decision = auto_scaler.get_scaling_decision()
    logger.info(f"扩缩容决策: {decision}")

    if decision:
        auto_scaler.execute_scaling(decision)
        logger.info(f"当前副本数: {auto_scaler.current_replicas}")

    # 2. 熔断器演示
    logger.info("\n2. 熔断器演示")
    circuit_breaker = CircuitBreaker(threshold=3, timeout=5)

    def failing_function():
        raise Exception("模拟服务故障")

    # 测试熔断器
    for i in range(4):
        try:
            circuit_breaker.call(failing_function)
        except Exception as e:
            logger.info(f"调用 {i+1}: {e}")
            logger.info(f"熔断器状态: {circuit_breaker.state}")

    # 3. 蓝绿部署演示
    logger.info("\n3. 蓝绿部署演示")
    blue_green_config = BlueGreenConfig()
    blue_green = BlueGreenDeployment(blue_green_config)

    # 记录健康检查
    for i in range(10):
        blue_green.record_health_check("blue", True)
        blue_green.record_health_check("green", True)

    logger.info(f"蓝版本健康率: {blue_green.get_health_rate('blue')}")
    logger.info(f"绿版本健康率: {blue_green.get_health_rate('green')}")
    logger.info(f"当前活跃版本: {blue_green.active_version}")

    # 切换流量
    blue_green.switch_traffic("green")
    logger.info(f"切换后活跃版本: {blue_green.active_version}")


async def demo_intelligent_features():
    """演示智能化功能"""
    logger.info("\n=== 智能化功能演示 ===")

    # 1. 机器学习模型演示
    logger.info("1. 机器学习模型演示")
    ml_config = MLModelConfig()
    ml_model = MLModel(ml_config)

    # 添加训练数据
    logger.info("添加训练数据...")
    for i in range(15):
        features = {
            "cpu": 50.0 + i * 2,
            "memory": 60.0 + i * 2,
            "requests": 100 + i * 5
        }
        target = 80.0 + i * 1.5
        ml_model.add_training_data(features, target)

    # 训练模型
    logger.info("训练模型...")
    success = ml_model.train()
    logger.info(f"模型训练成功: {success}")

    # 进行预测
    test_features = {"cpu": 70.0, "memory": 75.0, "requests": 120}
    prediction = ml_model.predict(test_features)
    logger.info(f"性能预测: {prediction:.2f}")

    # 2. 自动调优演示
    logger.info("\n2. 自动调优演示")
    tuning_config = AutoTuningConfig()
    auto_tuner = AutoTuner(tuning_config)

    # 设置初始参数
    initial_params = {"param1": 1.0, "param2": 2.0}
    auto_tuner.set_parameters(initial_params)

    # 记录性能数据
    logger.info("记录性能数据...")
    for i in range(15):
        performance_score = 80.0 + i * 2
        auto_tuner.record_performance(performance_score)

    # 优化参数
    logger.info("优化参数...")
    optimized_params = auto_tuner.optimize_parameters()
    logger.info(f"优化后参数: {optimized_params}")

    # 3. 预测性维护演示
    logger.info("\n3. 预测性维护演示")
    maintenance_config = PredictiveMaintenanceConfig()
    maintenance = PredictiveMaintenance(maintenance_config)

    # 记录系统健康数据
    logger.info("记录系统健康数据...")
    for i in range(15):
        health_score = 0.9 - (i * 0.02)  # 健康分数逐渐下降
        metrics = {"cpu": 70.0 + i, "memory": 75.0 + i}
        maintenance.record_system_health("test-component", health_score, metrics)

    # 预测故障
    failure_prob = maintenance.predict_failure("test-component")
    logger.info(f"故障预测概率: {failure_prob:.2f}")

    # 检查是否需要维护
    should_maintain = maintenance.should_schedule_maintenance("test-component")
    logger.info(f"是否需要安排维护: {should_maintain}")

    # 4. 智能监控演示
    logger.info("\n4. 智能监控演示")
    intelligent_monitor = IntelligentMonitor2()

    # 添加ML模型和自动调优器
    intelligent_monitor.add_ml_model("test-service", ml_config)
    intelligent_monitor.add_auto_tuner("test-service", tuning_config)

    # 记录指标
    metrics = {"cpu": 75.0, "memory": 80.0, "requests": 100}
    performance_score = 85.0
    intelligent_monitor.record_metrics("test-service", metrics, performance_score)

    # 训练模型和优化参数
    intelligent_monitor.train_models()
    intelligent_monitor.optimize_parameters()

    # 获取预测和优化参数
    prediction = intelligent_monitor.get_predictions("test-service", metrics)
    optimized_params = intelligent_monitor.get_optimized_parameters("test-service")

    logger.info(f"智能预测: {prediction:.2f}")
    logger.info(f"智能优化参数: {optimized_params}")


async def main():
    """主演示函数"""
    logger.info("开始云原生和智能化功能演示...")

    try:
        # 演示云原生功能
        await demo_cloud_native_features()

        # 演示智能化功能
        await demo_intelligent_features()

        logger.info("\n=== 演示完成 ===")
        logger.info("云原生和智能化功能已成功实现！")

    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

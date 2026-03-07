#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分布式计算示例

演示如何使用特征层分布式计算功能。
"""

import time
import pandas as pd
import numpy as np
from src.features.distributed import (
    LoadBalancingStrategy,
    TaskPriority,
    create_distributed_processor
)
from src.features.core.config import FeatureProcessingConfig, DefaultConfigs


def create_sample_data(rows: int = 1000) -> pd.DataFrame:
    """创建示例数据"""
    np.random.seed(42)

    # 股票数据
    dates = pd.date_range('2024-01-01', periods=rows, freq='D')
    prices = np.random.randn(rows).cumsum() + 100

    data = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(rows) * 0.5,
        'high': prices + np.random.randn(rows) * 1.0,
        'low': prices - np.random.randn(rows) * 1.0,
        'close': prices,
        'volume': np.random.randint(1000, 10000, rows),
        'title': [f'新闻标题{i}' for i in range(rows)],
        'content': [f'这是第{i}条新闻内容，包含一些情感词汇。' for i in range(rows)]
    })

    return data


def example_basic_distributed_processing():
    """基础分布式处理示例"""
    print("=== 基础分布式处理示例 ===")

    # 创建分布式处理器
    processor = create_distributed_processor(
        max_workers=4,
        executor_type="thread",
        load_balancing_strategy=LoadBalancingStrategy.SIMPLE
    )

    # 启动处理器
    processor.start()

    try:
        # 创建示例数据
        data = create_sample_data(100)
        print(f"原始数据形状: {data.shape}")

        # 创建配置
        config = DefaultConfigs.basic_technical()
        processing_config = FeatureProcessingConfig(
            batch_size=50,
            timeout=10.0,
            retry_count=2
        )

        # 提交任务
        task_id = processor.process_features(
            data=data,
            config=config,
            processing_config=processing_config,
            priority=TaskPriority.NORMAL
        )
        print(f"提交任务: {task_id}")

        # 等待任务完成
        print("等待任务完成...")
        result = processor.wait_for_task(task_id, timeout=30.0)

        if result:
            print(f"任务完成: {result.task_id}")
            print(f"处理时间: {result.processing_time:.2f}s")
            print(f"内存使用: {result.memory_usage:.2f}MB")
            print(f"成功: {result.success}")
            if result.error_message:
                print(f"错误: {result.error_message}")
        else:
            print("任务超时")

    finally:
        processor.stop()


def example_batch_processing():
    """批量处理示例"""
    print("\n=== 批量处理示例 ===")

    processor = create_distributed_processor(max_workers=4)
    processor.start()

    try:
        # 创建多个数据批次
        data_batches = []
        configs = [
            DefaultConfigs.basic_technical(),
            DefaultConfigs.comprehensive_technical(),
            DefaultConfigs.sentiment_analysis()
        ]

        for i in range(3):
            data = create_sample_data(50 + i * 20)
            config = configs[i]
            processing_config = FeatureProcessingConfig(batch_size=25)

            data_batches.append((data, config, processing_config))

        # 批量提交任务
        task_ids = processor.process_features_batch(
            data_batch=data_batches,
            priority=TaskPriority.NORMAL
        )
        print(f"提交了 {len(task_ids)} 个批量任务")

        # 等待所有任务完成
        print("等待批量任务完成...")
        results = processor.wait_for_batch(task_ids, timeout=60.0)

        # 统计结果
        successful = sum(1 for r in results if r and r.success)
        total_time = sum(r.processing_time for r in results if r)

        print(f"批量处理完成: {successful}/{len(results)} 成功")
        print(f"总处理时间: {total_time:.2f}s")

    finally:
        processor.stop()


def example_priority_tasks():
    """优先级任务示例"""
    print("\n=== 优先级任务示例 ===")

    processor = create_distributed_processor(max_workers=2)
    processor.start()

    try:
        data = create_sample_data(100)
        config = DefaultConfigs.basic_technical()
        processing_config = FeatureProcessingConfig()

        # 提交不同优先级的任务
        priorities = [TaskPriority.LOW, TaskPriority.NORMAL, TaskPriority.HIGH]
        task_ids = []

        for priority in priorities:
            task_id = processor.process_features(
                data=data.copy(),
                config=config,
                processing_config=processing_config,
                priority=priority
            )
            task_ids.append(task_id)
            print(f"提交 {priority.value} 优先级任务: {task_id}")

        # 等待任务完成
        print("等待优先级任务完成...")
        results = processor.wait_for_batch(task_ids, timeout=45.0)

        for i, (priority, result) in enumerate(zip(priorities, results)):
            if result:
                print(f"{priority.value} 优先级任务完成: {result.processing_time:.2f}s")
            else:
                print(f"{priority.value} 优先级任务超时")

    finally:
        processor.stop()


def example_worker_simulation():
    """工作节点模拟示例"""
    print("\n=== 工作节点模拟示例 ===")

    processor = create_distributed_processor(max_workers=3)
    processor.start()

    try:
        # 模拟工作节点注册
        worker_manager = processor.worker_manager

        # 注册模拟工作节点
        worker_ids = ["worker_1", "worker_2", "worker_3"]
        for worker_id in worker_ids:
            worker_manager.register_worker(
                worker_id=worker_id,
                capabilities={"cpu": 4, "memory": 8192, "performance_score": 0.8}
            )
            print(f"注册工作节点: {worker_id}")

        # 提交任务
        data = create_sample_data(80)
        config = DefaultConfigs.basic_technical()
        processing_config = FeatureProcessingConfig()

        task_ids = []
        for i in range(5):
            task_id = processor.process_features(
                data=data.copy(),
                config=config,
                processing_config=processing_config
            )
            task_ids.append(task_id)

        print(f"提交了 {len(task_ids)} 个任务")

        # 等待任务完成
        results = processor.wait_for_batch(task_ids, timeout=30.0)

        # 显示工作节点统计
        worker_stats = worker_manager.get_worker_stats()
        print(f"工作节点统计: {worker_stats}")

    finally:
        processor.stop()


def example_performance_monitoring():
    """性能监控示例"""
    print("\n=== 性能监控示例 ===")

    processor = create_distributed_processor(max_workers=4)
    processor.start()

    try:
        # 提交多个任务
        data = create_sample_data(60)
        config = DefaultConfigs.comprehensive_technical()
        processing_config = FeatureProcessingConfig()

        task_ids = []
        for i in range(8):
            task_id = processor.process_features(
                data=data.copy(),
                config=config,
                processing_config=processing_config
            )
            task_ids.append(task_id)

        # 等待任务完成
        processor.wait_for_batch(task_ids, timeout=60.0)

        # 获取性能统计
        stats = processor.get_processing_stats()
        print("性能统计:")
        print(f"  总任务数: {stats['total_tasks']}")
        print(f"  成功任务数: {stats['successful_tasks']}")
        print(f"  失败任务数: {stats['failed_tasks']}")
        print(f"  成功率: {stats['success_rate']:.2%}")
        print(f"  平均处理时间: {stats['avg_processing_time']:.2f}s")
        print(f"  平均内存使用: {stats['avg_memory_usage']:.2f}MB")

    finally:
        processor.stop()


def example_task_cancellation():
    """任务取消示例"""
    print("\n=== 任务取消示例 ===")

    processor = create_distributed_processor(max_workers=2)
    processor.start()

    try:
        data = create_sample_data(200)
        config = DefaultConfigs.basic_technical()
        processing_config = FeatureProcessingConfig()

        # 提交任务
        task_ids = []
        for i in range(5):
            task_id = processor.process_features(
                data=data.copy(),
                config=config,
                processing_config=processing_config
            )
            task_ids.append(task_id)

        print(f"提交了 {len(task_ids)} 个任务")

        # 等待一段时间后取消部分任务
        time.sleep(2.0)

        # 取消前3个任务
        cancel_ids = task_ids[:3]
        cancel_results = processor.cancel_batch(cancel_ids)

        cancelled = sum(cancel_results)
        print(f"取消了 {cancelled}/{len(cancel_ids)} 个任务")

        # 等待剩余任务完成
        remaining_ids = task_ids[3:]
        if remaining_ids:
            results = processor.wait_for_batch(remaining_ids, timeout=30.0)
            completed = sum(1 for r in results if r and r.success)
            print(f"剩余任务完成: {completed}/{len(remaining_ids)}")

    finally:
        processor.stop()


def main():
    """主函数"""
    print("分布式计算示例")
    print("=" * 50)

    try:
        # 基础分布式处理
        example_basic_distributed_processing()

        # 批量处理
        example_batch_processing()

        # 优先级任务
        example_priority_tasks()

        # 工作节点模拟
        example_worker_simulation()

        # 性能监控
        example_performance_monitoring()

        # 任务取消
        example_task_cancellation()

        print("\n所有示例执行完成!")

    except Exception as e:
        print(f"示例执行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

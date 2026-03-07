#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化组件性能测试

测试各个优化组件的性能表现，包括模型管理、数据处理、性能监控和缓存优化。
"""

import time
import random
import logging
import statistics
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入优化组件
from src.ml.core.ml_service import MLService
from src.data.validation.validator import DataValidator
from src.monitoring.core.real_time_monitor import start_monitoring, update_business_metric, get_monitor
from src.data.cache.cache_manager import CacheManager
from src.data.cache.hot_data_cache import get_hot_data_cache
from src.data.cache.enhanced_cache_manager import EnhancedCacheManager
from src.data.cache.enhanced_cache_strategy import get_cache_strategy


class PerformanceTest:
    """性能测试基类"""

    def __init__(self, test_name: str, iterations: int = 1000):
        """
        初始化性能测试

        Args:
            test_name: 测试名称
            iterations: 测试迭代次数
        """
        self.test_name = test_name
        self.iterations = iterations
        self.results = []
        self.start_time = 0
        self.end_time = 0

    def start(self):
        """开始测试"""
        logger.info(f"开始性能测试: {self.test_name} (迭代 {self.iterations} 次)")
        self.start_time = time.time()

    def end(self):
        """结束测试"""
        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        avg_time = total_time / self.iterations if self.iterations > 0 else 0
        
        logger.info(f"结束性能测试: {self.test_name}")
        logger.info(f"总耗时: {total_time:.4f} 秒")
        logger.info(f"平均耗时: {avg_time:.6f} 秒/次")
        logger.info(f"吞吐量: {self.iterations / total_time:.2f} 次/秒")
        
        if self.results:
            logger.info(f"最大耗时: {max(self.results):.6f} 秒")
            logger.info(f"最小耗时: {min(self.results):.6f} 秒")
            logger.info(f"标准差: {statistics.stdev(self.results):.6f} 秒")
        
        return {
            'test_name': self.test_name,
            'iterations': self.iterations,
            'total_time': total_time,
            'avg_time': avg_time,
            'throughput': self.iterations / total_time,
            'max_time': max(self.results) if self.results else 0,
            'min_time': min(self.results) if self.results else 0,
            'std_dev': statistics.stdev(self.results) if len(self.results) > 1 else 0
        }

    def add_result(self, elapsed: float):
        """添加测试结果"""
        self.results.append(elapsed)


class ModelPerformanceTest(PerformanceTest):
    """模型性能测试"""

    def __init__(self, iterations: int = 1000):
        super().__init__("模型管理性能测试", iterations)
        self.ml_service = MLService()

    def run(self):
        """运行测试"""
        self.start()
        
        for i in range(self.iterations):
            start = time.time()
            
            # 测试模型注册和获取
            model_name = f"test_model_{i}"
            model_config = {
                'model_type': 'test',
                'model_path': f'test_model_{i}',
                'params': {'param1': i, 'param2': i * 2}
            }
            
            try:
                # 注册模型
                self.ml_service.register_model(model_name, model_config)
                # 获取模型
                model = self.ml_service.get_model(model_name)
            except Exception as e:
                logger.error(f"模型操作失败: {e}")
            
            end = time.time()
            self.add_result(end - start)
        
        return self.end()


class DataValidationPerformanceTest(PerformanceTest):
    """数据验证性能测试"""

    def __init__(self, iterations: int = 1000):
        super().__init__("数据验证性能测试", iterations)
        self.validator = DataValidator()
        # 准备测试数据
        self.test_data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'feature3': ['a', 'b', 'c', 'd', 'e'],
            'feature4': [True, False, True, False, True]
        }
        self.validation_rules = {
            'feature1': {'type': 'list', 'required': True},
            'feature2': {'type': 'list', 'required': True},
            'feature3': {'type': 'list', 'required': True},
            'feature4': {'type': 'list', 'required': True}
        }

    def run(self):
        """运行测试"""
        self.start()
        
        for i in range(self.iterations):
            start = time.time()
            
            # 测试数据验证
            try:
                result = self.validator.validate_dict(self.test_data, self.validation_rules)
            except Exception as e:
                logger.error(f"数据验证失败: {e}")
            
            end = time.time()
            self.add_result(end - start)
        
        return self.end()


class CachePerformanceTest(PerformanceTest):
    """缓存性能测试"""

    def __init__(self, iterations: int = 1000):
        super().__init__("缓存性能测试", iterations)
        self.cache_manager = CacheManager()
        # 准备测试数据
        self.test_keys = [f"test_key_{i}" for i in range(100)]
        self.test_values = [f"test_value_{i}" for i in range(100)]

    def run(self):
        """运行测试"""
        self.start()
        
        for i in range(self.iterations):
            start = time.time()
            
            # 随机选择键值对
            key = random.choice(self.test_keys)
            value = random.choice(self.test_values)
            
            # 测试缓存操作
            try:
                # 设置缓存
                self.cache_manager.set(key, value)
                # 获取缓存
                get_value = self.cache_manager.get(key)
                # 检查缓存存在
                exists = self.cache_manager.exists(key)
            except Exception as e:
                logger.error(f"缓存操作失败: {e}")
            
            end = time.time()
            self.add_result(end - start)
        
        # 清理缓存
        for key in self.test_keys:
            self.cache_manager.delete(key)
        
        return self.end()


class HotDataCachePerformanceTest(PerformanceTest):
    """热点数据缓存性能测试"""

    def __init__(self, iterations: int = 1000):
        super().__init__("热点数据缓存性能测试", iterations)
        self.hot_cache = get_hot_data_cache()
        # 准备测试数据
        self.hot_key = "hot_data_key"
        self.hot_value = "hot_data_value"

    def run(self):
        """运行测试"""
        # 首先设置热点数据
        self.hot_cache.set(self.hot_key, self.hot_value)
        
        self.start()
        
        for i in range(self.iterations):
            start = time.time()
            
            # 测试热点数据访问
            try:
                value = self.hot_cache.get(self.hot_key)
            except Exception as e:
                logger.error(f"热点数据访问失败: {e}")
            
            end = time.time()
            self.add_result(end - start)
        
        # 清理缓存
        self.hot_cache.delete(self.hot_key)
        
        return self.end()


class EnhancedCachePerformanceTest(PerformanceTest):
    """增强缓存管理器性能测试"""

    def __init__(self, iterations: int = 1000):
        super().__init__("增强缓存管理器性能测试", iterations)
        self.enhanced_cache = EnhancedCacheManager(cache_dir="test_perf_cache")
        # 准备测试数据
        self.test_keys = [f"enhanced_key_{i}" for i in range(100)]
        self.test_values = [f"enhanced_value_{i}" for i in range(100)]

    def run(self):
        """运行测试"""
        self.start()
        
        for i in range(self.iterations):
            start = time.time()
            
            # 随机选择键值对
            key = random.choice(self.test_keys)
            value = random.choice(self.test_values)
            
            # 测试增强缓存操作
            try:
                # 设置缓存
                self.enhanced_cache.set(key, value)
                # 获取缓存
                get_value = self.enhanced_cache.get(key)
                # 检查缓存存在
                exists = self.enhanced_cache.exists(key)
            except Exception as e:
                logger.error(f"增强缓存操作失败: {e}")
            
            end = time.time()
            self.add_result(end - start)
        
        # 清理缓存
        for key in self.test_keys:
            self.enhanced_cache.delete(key)
        
        # 关闭缓存管理器
        self.enhanced_cache.shutdown()
        
        return self.end()


class MonitoringPerformanceTest(PerformanceTest):
    """监控系统性能测试"""

    def __init__(self, iterations: int = 1000):
        super().__init__("监控系统性能测试", iterations)
        start_monitoring()
        self.monitor = get_monitor()

    def run(self):
        """运行测试"""
        self.start()
        
        for i in range(self.iterations):
            start = time.time()
            
            # 测试监控操作
            try:
                # 更新业务指标
                update_business_metric('request', 1.0)
                update_business_metric('response_time', random.uniform(50, 200))
                # 收集指标
                metrics = self.monitor.metrics_collector.collect_all_metrics()
            except Exception as e:
                logger.error(f"监控操作失败: {e}")
            
            end = time.time()
            self.add_result(end - start)
        
        return self.end()


class BatchOperationPerformanceTest(PerformanceTest):
    """批量操作性能测试"""

    def __init__(self, iterations: int = 100):
        super().__init__("批量操作性能测试", iterations)
        self.enhanced_cache = EnhancedCacheManager(cache_dir="test_batch_cache")
        # 准备测试数据
        self.batch_size = 100
        self.batch_items = [(f"batch_key_{i}", f"batch_value_{i}") for i in range(self.batch_size)]

    def run(self):
        """运行测试"""
        self.start()
        
        for i in range(self.iterations):
            start = time.time()
            
            # 测试批量操作
            try:
                # 批量设置
                batch_set_result = self.enhanced_cache.batch_set(self.batch_items)
                # 批量获取
                batch_get_result = self.enhanced_cache.batch_get([item[0] for item in self.batch_items])
                # 批量删除
                batch_delete_result = self.enhanced_cache.batch_delete([item[0] for item in self.batch_items])
            except Exception as e:
                logger.error(f"批量操作失败: {e}")
            
            end = time.time()
            self.add_result(end - start)
        
        # 关闭缓存管理器
        self.enhanced_cache.shutdown()
        
        return self.end()


def run_all_performance_tests():
    """运行所有性能测试"""
    logger.info("=== 开始全面性能测试 ===")
    
    # 初始化测试结果
    all_results = []
    
    # 运行模型性能测试
    model_test = ModelPerformanceTest(iterations=1000)
    model_test.run()
    
    # 运行数据验证性能测试
    validation_test = DataValidationPerformanceTest(iterations=1000)
    validation_test.run()
    
    # 运行缓存性能测试
    cache_test = CachePerformanceTest(iterations=10000)
    cache_test.run()
    
    # 运行热点数据缓存性能测试
    hot_cache_test = HotDataCachePerformanceTest(iterations=10000)
    hot_cache_test.run()
    
    # 运行增强缓存性能测试
    enhanced_cache_test = EnhancedCachePerformanceTest(iterations=10000)
    enhanced_cache_test.run()
    
    # 运行监控系统性能测试
    monitoring_test = MonitoringPerformanceTest(iterations=1000)
    monitoring_test.run()
    
    # 运行批量操作性能测试
    batch_test = BatchOperationPerformanceTest(iterations=100)
    batch_test.run()
    
    logger.info("=== 全面性能测试完成 ===")


if __name__ == "__main__":
    # 运行所有性能测试
    run_all_performance_tests()

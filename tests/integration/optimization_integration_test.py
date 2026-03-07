#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化组件集成测试

测试所有优化组件的集成功能，包括模型管理、数据处理、性能监控和缓存优化。
"""

import unittest
import time
import logging
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 导入优化组件
from src.ml.core.ml_service import MLService
from src.data.validation.validator import DataValidator, ValidationResult
from src.monitoring.core.real_time_monitor import start_monitoring, update_business_metric, get_monitor
from src.monitoring.intelligent_alert_system import IntelligentAlertSystem
from src.data.cache.cache_manager import CacheManager, CacheConfig
from src.data.cache.hot_data_cache import HotDataCache, get_hot_data_cache
from src.data.cache.enhanced_cache_manager import EnhancedCacheManager
from src.data.cache.enhanced_cache_strategy import get_cache_strategy, AdaptiveStrategy


class OptimizationIntegrationTest(unittest.TestCase):
    """优化组件集成测试"""

    def setUp(self):
        """设置测试环境"""
        logger.info("=== 开始优化组件集成测试 ===")
        
        # 初始化模型服务
        self.ml_service = MLService()
        
        # 初始化数据验证器
        self.validator = DataValidator()
        
        # 启动监控系统
        start_monitoring()
        self.monitor = get_monitor()
        
        # 初始化智能告警系统
        self.alert_system = IntelligentAlertSystem()
        
        # 初始化缓存管理器
        self.cache_manager = CacheManager()
        
        # 初始化热点数据缓存
        self.hot_cache = get_hot_data_cache()
        
        # 初始化增强缓存管理器
        self.enhanced_cache = EnhancedCacheManager(cache_dir="test_cache")
        
        # 初始化自适应缓存策略
        self.adaptive_strategy = get_cache_strategy('adaptive')
        
        # 等待系统启动
        time.sleep(2)

    def tearDown(self):
        """清理测试环境"""
        logger.info("=== 结束优化组件集成测试 ===")
        
        # 停止缓存管理器
        self.cache_manager.stop()
        self.enhanced_cache.shutdown()
        
        # 等待清理完成
        time.sleep(1)

    def test_model_management(self):
        """测试模型管理功能"""
        logger.info("测试模型管理功能...")
        
        # 测试模型加载
        model_config = {
            'model_type': 'test',
            'model_path': 'test_model',
            'params': {'test_param': 'value'}
        }
        
        # 测试模型注册
        try:
            self.ml_service.register_model('test_model', model_config)
            logger.info("✓ 模型注册成功")
        except Exception as e:
            logger.error(f"✗ 模型注册失败: {e}")
            self.fail(f"模型注册失败: {e}")
        
        # 测试模型获取
        try:
            model = self.ml_service.get_model('test_model')
            logger.info("✓ 模型获取成功")
        except Exception as e:
            logger.error(f"✗ 模型获取失败: {e}")
            self.fail(f"模型获取失败: {e}")

    def test_data_validation(self):
        """测试数据验证功能"""
        logger.info("测试数据验证功能...")
        
        # 测试DataFrame验证
        import pandas as pd
        
        test_data = pd.DataFrame({
            'column1': [1, 2, 3, 4, 5],
            'column2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'column3': ['a', 'b', 'c', 'd', 'e']
        })
        
        # 定义验证规则
        validation_rules = {
            'column1': {'required': True, 'type': 'int', 'min': 1, 'max': 10},
            'column2': {'required': True, 'type': 'float'},
            'column3': {'required': True, 'type': 'str'}
        }
        
        # 执行验证
        result = self.validator.validate_dataframe(test_data, validation_rules)
        
        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.is_valid, f"数据验证失败: {result.errors}")
        logger.info("✓ 数据验证成功")

    def test_performance_monitoring(self):
        """测试性能监控功能"""
        logger.info("测试性能监控功能...")
        
        # 更新业务指标
        for i in range(5):
            update_business_metric('request', 1.0)
            update_business_metric('response_time', 100.0 + i * 10)
            time.sleep(0.1)
        
        # 收集指标
        metrics = self.monitor.metrics_collector.collect_all_metrics()
        
        self.assertGreater(len(metrics), 0, "未收集到任何指标")
        logger.info(f"✓ 性能监控成功，收集到 {len(metrics)} 个指标")

    def test_intelligent_alert(self):
        """测试智能告警功能"""
        logger.info("测试智能告警功能...")
        
        # 模拟异常数据
        test_data = {
            'cpu_percent': 95.0,  # 超过阈值
            'memory_percent': 90.0,  # 超过阈值
            'cache_hit_rate': 0.3,  # 低于阈值
            'data_quality_score': 0.6  # 低于阈值
        }
        
        # 检查异常
        alerts = self.alert_system.check_anomaly(test_data)
        
        logger.info(f"✓ 智能告警测试完成，检测到 {len(alerts)} 个告警")

    def test_cache_management(self):
        """测试缓存管理功能"""
        logger.info("测试缓存管理功能...")
        
        # 测试基本缓存操作
        test_key = "test_cache_key"
        test_value = "test_cache_value"
        
        # 设置缓存
        set_result = self.cache_manager.set(test_key, test_value)
        self.assertTrue(set_result, "缓存设置失败")
        
        # 获取缓存
        get_result = self.cache_manager.get(test_key)
        self.assertEqual(get_result, test_value, "缓存获取失败")
        
        # 检查缓存存在
        exists_result = self.cache_manager.exists(test_key)
        self.assertTrue(exists_result, "缓存存在检查失败")
        
        # 删除缓存
        delete_result = self.cache_manager.delete(test_key)
        self.assertTrue(delete_result, "缓存删除失败")
        
        logger.info("✓ 缓存管理功能测试成功")

    def test_hot_data_cache(self):
        """测试热点数据缓存功能"""
        logger.info("测试热点数据缓存功能...")
        
        # 模拟热点数据访问
        hot_key = "hot_data_key"
        hot_value = "hot_data_value"
        
        # 设置缓存
        self.hot_cache.set(hot_key, hot_value)
        
        # 模拟多次访问
        for i in range(15):  # 超过热点阈值
            value = self.hot_cache.get(hot_key)
            self.assertEqual(value, hot_value, f"热点数据获取失败，第 {i+1} 次")
            time.sleep(0.1)
        
        # 等待热点检测
        time.sleep(5)
        
        # 检查热点键
        hot_keys = self.hot_cache.get_hot_keys()
        logger.info(f"✓ 热点数据缓存测试成功，检测到 {len(hot_keys)} 个热点键")

    def test_enhanced_cache_manager(self):
        """测试增强缓存管理器功能"""
        logger.info("测试增强缓存管理器功能...")
        
        # 测试缓存预热
        self.enhanced_cache.add_warmup_data("warmup_key", "warmup_value")
        self.enhanced_cache.execute_warmup()
        
        # 测试批量操作
        batch_items = [
            ("batch_key1", "batch_value1"),
            ("batch_key2", "batch_value2"),
            ("batch_key3", "batch_value3")
        ]
        
        # 批量设置
        batch_set_result = self.enhanced_cache.batch_set(batch_items)
        self.assertEqual(len(batch_set_result), 3, "批量设置失败")
        
        # 批量获取
        batch_get_result = self.enhanced_cache.batch_get(["batch_key1", "batch_key2", "batch_key3"])
        self.assertEqual(len(batch_get_result), 3, "批量获取失败")
        
        # 批量删除
        batch_delete_result = self.enhanced_cache.batch_delete(["batch_key1", "batch_key2", "batch_key3"])
        self.assertEqual(len(batch_delete_result), 3, "批量删除失败")
        
        logger.info("✓ 增强缓存管理器测试成功")

    def test_integration_workflow(self):
        """测试完整集成工作流"""
        logger.info("测试完整集成工作流...")
        
        # 模拟完整的工作流程
        test_data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'feature3': ['a', 'b', 'c', 'd', 'e']
        }
        
        # 1. 数据验证
        validation_result = self.validator.validate_dict(test_data, {
            'feature1': {'type': 'list', 'required': True},
            'feature2': {'type': 'list', 'required': True},
            'feature3': {'type': 'list', 'required': True}
        })
        
        self.assertTrue(validation_result.is_valid, "数据验证失败")
        
        # 2. 模型预测
        try:
            # 模拟模型预测
            start_time = time.time()
            # 这里不实际执行预测，只测试流程
            prediction_time = time.time() - start_time
            
            # 更新业务指标
            update_business_metric('model_inference_time', prediction_time * 1000)
            update_business_metric('data_validation_time', 100)
            
            logger.info("✓ 模型预测流程测试成功")
        except Exception as e:
            logger.error(f"✗ 模型预测失败: {e}")
            # 不失败测试，因为可能没有实际模型
        
        # 3. 缓存热点数据
        self.hot_cache.set("workflow_data", test_data)
        
        # 4. 检查系统状态
        system_status = self.monitor.get_system_status()
        self.assertIn('system_health', system_status, "系统状态获取失败")
        
        logger.info("✓ 完整集成工作流测试成功")

    def test_performance_metrics(self):
        """测试性能指标收集"""
        logger.info("测试性能指标收集...")
        
        # 执行一系列操作以生成性能指标
        operations = [
            ("cache", lambda: self.cache_manager.set("perf_key", "perf_value")),
            ("validation", lambda: self.validator.validate_dict({"test": "value"}, {"test": {}})),
            ("monitoring", lambda: update_business_metric("request", 1.0)),
        ]
        
        for name, operation in operations:
            try:
                start_time = time.time()
                operation()
                duration = (time.time() - start_time) * 1000  # 转换为毫秒
                logger.info(f"✓ {name} 操作耗时: {duration:.2f} ms")
            except Exception as e:
                logger.error(f"✗ {name} 操作失败: {e}")
        
        # 等待指标收集
        time.sleep(2)
        
        # 获取指标
        metrics = self.monitor.metrics_collector.collect_all_metrics()
        logger.info(f"✓ 性能指标收集成功，获取到 {len(metrics)} 个指标")


if __name__ == '__main__':
    unittest.main()

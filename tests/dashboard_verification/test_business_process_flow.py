"""
业务流程数据流测试
按照业务流程顺序测试数据流连通性
"""

import pytest
import requests
from typing import Dict, Any, List


BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


class TestStrategyDevelopmentFlow:
    """量化策略开发流程数据流测试"""
    
    def test_data_collection_to_feature_engineering(self):
        """测试数据收集 → 特征工程数据流"""
        # 1. 检查数据收集API
        data_sources_res = requests.get(f"{API_BASE}/data/sources")
        print(f"数据收集API状态: {data_sources_res.status_code}")
        
        # 2. 检查特征工程API
        feature_tasks_res = requests.get(f"{API_BASE}/features/engineering/tasks")
        print(f"特征工程API状态: {feature_tasks_res.status_code}")
        
        # 验证数据流
        if data_sources_res.status_code == 200 and feature_tasks_res.status_code == 200:
            print("✅ 数据收集 → 特征工程数据流正常")
        else:
            print("⚠️ 数据收集 → 特征工程数据流存在问题")
    
    def test_feature_engineering_to_model_training(self):
        """测试特征工程 → 模型训练数据流"""
        # 1. 检查特征工程API
        features_res = requests.get(f"{API_BASE}/features/engineering/features")
        print(f"特征工程API状态: {features_res.status_code}")
        
        # 2. 检查模型训练API
        training_jobs_res = requests.get(f"{API_BASE}/ml/training/jobs")
        print(f"模型训练API状态: {training_jobs_res.status_code}")
        
        # 验证数据流
        if features_res.status_code == 200 and training_jobs_res.status_code == 200:
            print("✅ 特征工程 → 模型训练数据流正常")
        else:
            print("⚠️ 特征工程 → 模型训练数据流存在问题")
    
    def test_model_training_to_backtest(self):
        """测试模型训练 → 策略回测数据流"""
        # 1. 检查模型训练API
        training_jobs_res = requests.get(f"{API_BASE}/ml/training/jobs")
        print(f"模型训练API状态: {training_jobs_res.status_code}")
        
        # 2. 检查策略回测API
        backtest_res = requests.get(f"{API_BASE}/strategy/conceptions")
        print(f"策略回测API状态: {backtest_res.status_code}")
        
        # 验证数据流
        if training_jobs_res.status_code == 200 and backtest_res.status_code == 200:
            print("✅ 模型训练 → 策略回测数据流正常")
        else:
            print("⚠️ 模型训练 → 策略回测数据流存在问题")
    
    def test_backtest_to_performance_evaluation(self):
        """测试策略回测 → 性能评估数据流"""
        # 1. 检查策略回测API
        backtest_res = requests.get(f"{API_BASE}/strategy/conceptions")
        print(f"策略回测API状态: {backtest_res.status_code}")
        
        # 2. 检查性能评估API
        performance_res = requests.get(f"{API_BASE}/strategy/performance/comparison")
        print(f"性能评估API状态: {performance_res.status_code}")
        
        # 验证数据流
        if backtest_res.status_code == 200 and performance_res.status_code == 200:
            print("✅ 策略回测 → 性能评估数据流正常")
        else:
            print("⚠️ 策略回测 → 性能评估数据流存在问题")


class TestTradingExecutionFlow:
    """交易执行流程数据流测试"""
    
    def test_market_monitoring_to_signal_generation(self):
        """测试市场监控 → 信号生成数据流"""
        # 1. 检查市场数据API（通过数据源API）
        market_data_res = requests.get(f"{API_BASE}/data/sources")
        print(f"市场数据API状态: {market_data_res.status_code}")
        
        # 2. 检查信号生成API
        signals_res = requests.get(f"{API_BASE}/trading/signals/realtime")
        print(f"信号生成API状态: {signals_res.status_code}")
        
        # 验证数据流
        if market_data_res.status_code == 200 and signals_res.status_code == 200:
            print("✅ 市场监控 → 信号生成数据流正常")
        else:
            print("⚠️ 市场监控 → 信号生成数据流存在问题")
    
    def test_signal_generation_to_order_routing(self):
        """测试信号生成 → 订单路由数据流"""
        # 1. 检查信号生成API
        signals_res = requests.get(f"{API_BASE}/trading/signals/realtime")
        print(f"信号生成API状态: {signals_res.status_code}")
        
        # 2. 检查订单路由API
        routing_res = requests.get(f"{API_BASE}/trading/routing/decisions")
        print(f"订单路由API状态: {routing_res.status_code}")
        
        # 验证数据流
        if signals_res.status_code == 200 and routing_res.status_code == 200:
            print("✅ 信号生成 → 订单路由数据流正常")
        else:
            print("⚠️ 信号生成 → 订单路由数据流存在问题")


class TestRiskControlFlow:
    """风险控制流程数据流测试"""
    
    def test_risk_monitoring_to_reporting(self):
        """测试风险监测 → 风险报告数据流"""
        # 1. 检查风险控制API（通过风险服务）
        risk_res = requests.get(f"{API_BASE}/risk/reporting/templates")
        print(f"风险控制API状态: {risk_res.status_code}")
        
        # 2. 检查风险报告API
        reporting_res = requests.get(f"{API_BASE}/risk/reporting/history")
        print(f"风险报告API状态: {reporting_res.status_code}")
        
        # 验证数据流
        if risk_res.status_code == 200 and reporting_res.status_code == 200:
            print("✅ 风险监测 → 风险报告数据流正常")
        else:
            print("⚠️ 风险监测 → 风险报告数据流存在问题")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


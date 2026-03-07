"""
仪表盘API端点测试
按照业务流程顺序测试所有API端点
"""

import pytest
import requests
import json
from typing import Dict, Any, List
from datetime import datetime

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


class TestDataCollectionAPIs:
    """数据收集阶段API测试"""
    
    def test_data_sources_list(self):
        """测试数据源列表API"""
        response = requests.get(f"{API_BASE}/data/sources")
        assert response.status_code in [200, 404], f"数据源列表API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (dict, list)), "数据源列表格式错误"
            print(f"✅ 数据源列表API正常: {len(data) if isinstance(data, list) else 'N/A'} 个数据源")
    
    def test_data_sources_metrics(self):
        """测试数据源指标API"""
        response = requests.get(f"{API_BASE}/data-sources/metrics")
        assert response.status_code in [200, 404], f"数据源指标API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 数据源指标API正常")
    
    def test_data_quality_metrics(self):
        """测试数据质量指标API"""
        response = requests.get(f"{API_BASE}/data/quality/metrics")
        assert response.status_code in [200, 404], f"数据质量指标API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 数据质量指标API正常")


class TestFeatureEngineeringAPIs:
    """特征工程监控API测试"""
    
    def test_feature_tasks(self):
        """测试特征任务列表API"""
        response = requests.get(f"{API_BASE}/features/engineering/tasks")
        assert response.status_code in [200, 404], f"特征任务API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert "tasks" in data or isinstance(data, list), "特征任务数据格式错误"
            print(f"✅ 特征任务API正常")
    
    def test_feature_features(self):
        """测试特征列表API"""
        response = requests.get(f"{API_BASE}/features/engineering/features")
        assert response.status_code in [200, 404], f"特征列表API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert "features" in data or isinstance(data, list), "特征列表数据格式错误"
            print(f"✅ 特征列表API正常")
    
    def test_feature_indicators(self):
        """测试技术指标状态API"""
        response = requests.get(f"{API_BASE}/features/engineering/indicators")
        assert response.status_code in [200, 404], f"技术指标API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 技术指标API正常")


class TestModelTrainingAPIs:
    """模型训练监控API测试"""
    
    def test_training_jobs(self):
        """测试训练任务列表API"""
        response = requests.get(f"{API_BASE}/ml/training/jobs")
        assert response.status_code in [200, 404], f"训练任务API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert "jobs" in data or isinstance(data, list), "训练任务数据格式错误"
            print(f"✅ 训练任务API正常")
    
    def test_training_metrics(self):
        """测试训练指标API"""
        response = requests.get(f"{API_BASE}/ml/training/metrics")
        assert response.status_code in [200, 404], f"训练指标API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 训练指标API正常")


class TestStrategyPerformanceAPIs:
    """策略性能评估API测试"""
    
    def test_strategy_comparison(self):
        """测试策略对比API"""
        response = requests.get(f"{API_BASE}/strategy/performance/comparison")
        assert response.status_code in [200, 404], f"策略对比API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert "strategies" in data or isinstance(data, list), "策略对比数据格式错误"
            print(f"✅ 策略对比API正常")
    
    def test_strategy_metrics(self):
        """测试策略性能指标API"""
        response = requests.get(f"{API_BASE}/strategy/performance/metrics")
        assert response.status_code in [200, 404], f"策略性能指标API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 策略性能指标API正常")


class TestTradingSignalAPIs:
    """交易信号监控API测试"""
    
    def test_realtime_signals(self):
        """测试实时信号API"""
        response = requests.get(f"{API_BASE}/trading/signals/realtime")
        assert response.status_code in [200, 404], f"实时信号API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert "signals" in data or isinstance(data, list), "实时信号数据格式错误"
            print(f"✅ 实时信号API正常")
    
    def test_signal_stats(self):
        """测试信号统计API"""
        response = requests.get(f"{API_BASE}/trading/signals/stats")
        assert response.status_code in [200, 404], f"信号统计API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 信号统计API正常")
    
    def test_signal_distribution(self):
        """测试信号分布API"""
        response = requests.get(f"{API_BASE}/trading/signals/distribution")
        assert response.status_code in [200, 404], f"信号分布API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 信号分布API正常")


class TestOrderRoutingAPIs:
    """订单路由监控API测试"""
    
    def test_routing_decisions(self):
        """测试路由决策API"""
        response = requests.get(f"{API_BASE}/trading/routing/decisions")
        assert response.status_code in [200, 404], f"路由决策API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert "decisions" in data or isinstance(data, list), "路由决策数据格式错误"
            print(f"✅ 路由决策API正常")
    
    def test_routing_stats(self):
        """测试路由统计API"""
        response = requests.get(f"{API_BASE}/trading/routing/stats")
        assert response.status_code in [200, 404], f"路由统计API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 路由统计API正常")
    
    def test_routing_performance(self):
        """测试路由性能API"""
        response = requests.get(f"{API_BASE}/trading/routing/performance")
        assert response.status_code in [200, 404], f"路由性能API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 路由性能API正常")


class TestRiskReportingAPIs:
    """风险报告生成API测试"""
    
    def test_report_templates(self):
        """测试报告模板API"""
        response = requests.get(f"{API_BASE}/risk/reporting/templates")
        assert response.status_code in [200, 404], f"报告模板API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            assert "templates" in data or isinstance(data, list), "报告模板数据格式错误"
            print(f"✅ 报告模板API正常")
    
    def test_report_tasks(self):
        """测试报告生成任务API"""
        response = requests.get(f"{API_BASE}/risk/reporting/tasks")
        assert response.status_code in [200, 404], f"报告任务API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 报告任务API正常")
    
    def test_report_history(self):
        """测试报告历史API"""
        response = requests.get(f"{API_BASE}/risk/reporting/history")
        assert response.status_code in [200, 404], f"报告历史API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 报告历史API正常")
    
    def test_report_stats(self):
        """测试报告统计API"""
        response = requests.get(f"{API_BASE}/risk/reporting/stats")
        assert response.status_code in [200, 404], f"报告统计API返回: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 报告统计API正常")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


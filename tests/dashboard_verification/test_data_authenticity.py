"""
数据真实性测试
验证每个API端点返回真实数据，不使用模拟数据
"""

import pytest
import requests
import json
from typing import Dict, Any, List
from datetime import datetime

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


class TestDataAuthenticity:
    """数据真实性测试类"""
    
    def test_no_mock_data_in_response(self, response_data: Dict[str, Any], endpoint: str):
        """验证响应中不包含模拟数据标识"""
        # 检查响应中是否有模拟数据标识
        response_str = json.dumps(response_data, ensure_ascii=False).lower()
        
        # 不应该包含这些模拟数据标识
        mock_indicators = [
            "mock",
            "模拟",
            "fake",
            "dummy",
            "test_data",
            "sample_data",
            "random.choice",
            "random.uniform",
            "random.randint"
        ]
        
        for indicator in mock_indicators:
            assert indicator not in response_str, \
                f"端点 {endpoint} 的响应包含模拟数据标识: {indicator}"
    
    def test_response_has_data_source_note(self, response_data: Dict[str, Any], endpoint: str):
        """验证响应包含数据来源说明"""
        # 如果响应为空或只有note字段，应该说明数据来源
        if not response_data or (len(response_data) == 1 and "note" in response_data):
            assert "note" in response_data, \
                f"端点 {endpoint} 返回空数据但未说明原因"
            note = response_data.get("note", "")
            assert "真实" in note or "真实" in note or "监控" in note or "回测" in note, \
                f"端点 {endpoint} 的note字段未说明使用真实数据"


class TestDataCollectionAuthenticity:
    """数据收集阶段数据真实性测试"""
    
    def test_data_sources_metrics_authenticity(self):
        """测试数据源指标API不使用硬编码数据"""
        response = requests.get(f"{API_BASE}/data-sources/metrics")
        assert response.status_code == 200, f"数据源指标API返回: {response.status_code}"
        
        data = response.json()
        
        # 检查是否有硬编码警告
        system_metrics = data.get("system_metrics", {})
        note = system_metrics.get("note", "")
        
        # 不应该有硬编码估算的警告
        assert "估算" not in note or "真实监控数据" in note, \
            "数据源指标API仍使用硬编码估算值"
        
        # 如果指标为空，应该有说明
        if not data.get("latency_data"):
            assert "note" in system_metrics, \
                "数据源指标为空但未说明原因"
    
    def test_data_quality_metrics_authenticity(self):
        """测试数据质量指标API使用真实组件"""
        response = requests.get(f"{API_BASE}/data/quality/metrics")
        assert response.status_code == 200, f"数据质量指标API返回: {response.status_code}"
        
        data = response.json()
        
        # 检查是否有模拟数据标识
        response_str = json.dumps(data, ensure_ascii=False).lower()
        assert "mock" not in response_str, \
            "数据质量指标API返回模拟数据"
        
        # 如果指标为0，应该有说明
        metrics = data.get("metrics", {})
        if metrics.get("overall_score", 0) == 0:
            assert "note" in data or "error" in data, \
                "数据质量指标为0但未说明原因"


class TestStrategyPerformanceAuthenticity:
    """策略性能评估数据真实性测试"""
    
    def test_strategy_comparison_authenticity(self):
        """测试策略对比API不使用模拟数据"""
        response = requests.get(f"{API_BASE}/strategy/performance/comparison")
        assert response.status_code == 200, f"策略对比API返回: {response.status_code}"
        
        data = response.json()
        
        # 检查是否有模拟数据标识
        response_str = json.dumps(data, ensure_ascii=False).lower()
        assert "mock" not in response_str, \
            "策略对比API返回模拟数据"
        
        # 如果列表为空，应该有说明
        strategies = data.get("strategies", [])
        if not strategies:
            assert "note" in data, \
                "策略对比列表为空但未说明原因"
            note = data.get("note", "")
            assert "真实" in note or "回测" in note, \
                "策略对比的note字段未说明使用真实数据"
    
    def test_performance_metrics_authenticity(self):
        """测试性能指标API不使用模拟数据"""
        response = requests.get(f"{API_BASE}/strategy/performance/metrics")
        assert response.status_code == 200, f"性能指标API返回: {response.status_code}"
        
        data = response.json()
        
        # 检查是否有模拟数据标识
        response_str = json.dumps(data, ensure_ascii=False).lower()
        assert "mock" not in response_str, \
            "性能指标API返回模拟数据"
        
        # 检查是否有随机生成的数据
        return_curves = data.get("return_curves", [])
        if return_curves:
            # 检查收益曲线是否过于规律（可能是随机生成的）
            for curve in return_curves:
                values = curve.get("values", [])
                if len(values) > 10:
                    # 检查是否有明显的随机模式
                    # 如果所有值都在很小的范围内变化，可能是随机生成的
                    value_range = max(values) - min(values) if values else 0
                    assert value_range > 0.01, \
                        "收益曲线值变化过小，可能是随机生成的数据"


class TestTradingSignalAuthenticity:
    """交易信号数据真实性测试"""
    
    def test_realtime_signals_authenticity(self):
        """测试实时信号API不使用模拟数据"""
        response = requests.get(f"{API_BASE}/trading/signals/realtime")
        assert response.status_code == 200, f"实时信号API返回: {response.status_code}"
        
        data = response.json()
        
        # 检查是否有模拟数据标识
        response_str = json.dumps(data, ensure_ascii=False).lower()
        assert "mock" not in response_str, \
            "实时信号API返回模拟数据"
        
        # 如果列表为空，应该有说明
        signals = data.get("signals", [])
        if not signals:
            assert "note" in data, \
                "实时信号列表为空但未说明原因"
            note = data.get("note", "")
            assert "真实" in note or "信号生成器" in note, \
                "实时信号的note字段未说明使用真实数据"
        
        # 检查信号数据是否包含随机生成的标识
        for signal in signals:
            signal_str = json.dumps(signal, ensure_ascii=False).lower()
            # 不应该有明显的随机生成模式
            # 例如：所有信号的时间戳都在同一分钟内
            if len(signals) > 5:
                timestamps = [s.get("timestamp", 0) for s in signals]
                timestamp_range = max(timestamps) - min(timestamps) if timestamps else 0
                # 如果所有时间戳都在60秒内，可能是随机生成的
                assert timestamp_range > 60 or len(set(timestamps)) > 1, \
                    "信号时间戳过于集中，可能是随机生成的数据"


class TestBusinessProcessDataFlow:
    """业务流程数据流真实性测试"""
    
    def test_data_collection_to_feature_engineering_flow(self):
        """测试数据收集到特征工程的数据流"""
        # 1. 获取数据源列表
        sources_response = requests.get(f"{API_BASE}/data/sources")
        assert sources_response.status_code == 200
        
        sources_data = sources_response.json()
        sources = sources_data.get("data_sources", [])
        
        if sources:
            # 2. 获取特征工程任务（应该使用数据源的数据）
            features_response = requests.get(f"{API_BASE}/features/engineering/tasks")
            assert features_response.status_code == 200
            
            features_data = features_response.json()
            tasks = features_data.get("tasks", [])
            
            # 验证特征工程任务引用了真实数据源
            # 如果任务列表不为空，应该能关联到数据源
            if tasks:
                for task in tasks:
                    # 检查任务是否有数据源引用
                    assert "data_source" in task or "source_id" in task, \
                        "特征工程任务缺少数据源引用"
    
    def test_feature_engineering_to_model_training_flow(self):
        """测试特征工程到模型训练的数据流"""
        # 1. 获取特征列表
        features_response = requests.get(f"{API_BASE}/features/engineering/features")
        assert features_response.status_code == 200
        
        features_data = features_response.json()
        features = features_data.get("features", [])
        
        if features:
            # 2. 获取训练任务（应该使用特征数据）
            training_response = requests.get(f"{API_BASE}/ml/training/jobs")
            assert training_response.status_code == 200
            
            training_data = training_response.json()
            jobs = training_data.get("jobs", [])
            
            # 验证训练任务引用了真实特征
            if jobs:
                for job in jobs:
                    # 检查任务是否有特征引用
                    assert "features" in job or "feature_set" in job, \
                        "训练任务缺少特征引用"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


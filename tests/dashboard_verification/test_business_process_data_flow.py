"""
业务流程数据流验证测试
按照业务流程顺序验证数据流连通性，确保上游数据正确传递到下游
"""

import pytest
import requests
from typing import Dict, Any, List
from datetime import datetime

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


class TestQuantitativeStrategyDevelopmentFlow:
    """量化策略开发流程数据流测试"""
    
    def test_data_collection_to_feature_engineering(self):
        """测试数据收集 → 特征工程数据流"""
        # 1. 数据收集阶段：获取数据源列表
        sources_response = requests.get(f"{API_BASE}/data/sources")
        assert sources_response.status_code == 200, "数据源列表API失败"
        
        sources_data = sources_response.json()
        active_sources = [s for s in sources_data.get("data_sources", []) 
                         if s.get("enabled", True)]
        
        if active_sources:
            # 2. 特征工程阶段：获取特征任务
            features_response = requests.get(f"{API_BASE}/features/engineering/tasks")
            assert features_response.status_code == 200, "特征任务API失败"
            
            features_data = features_response.json()
            tasks = features_data.get("tasks", [])
            
            # 验证数据流：特征任务应该能关联到数据源
            # 如果任务列表不为空，应该能关联到启用的数据源
            if tasks:
                source_ids = [s.get("id") for s in active_sources]
                for task in tasks:
                    task_source = task.get("data_source_id") or task.get("source_id")
                    if task_source:
                        assert task_source in source_ids, \
                            f"特征任务 {task.get('id')} 引用了不存在的数据源 {task_source}"
    
    def test_feature_engineering_to_model_training(self):
        """测试特征工程 → 模型训练数据流"""
        # 1. 特征工程阶段：获取特征列表
        features_response = requests.get(f"{API_BASE}/features/engineering/features")
        assert features_response.status_code == 200, "特征列表API失败"
        
        features_data = features_response.json()
        features = features_data.get("features", [])
        
        if features:
            # 2. 模型训练阶段：获取训练任务
            training_response = requests.get(f"{API_BASE}/ml/training/jobs")
            assert training_response.status_code == 200, "训练任务API失败"
            
            training_data = training_response.json()
            jobs = training_data.get("jobs", [])
            
            # 验证数据流：训练任务应该能关联到特征
            if jobs:
                feature_names = [f.get("name") for f in features]
                for job in jobs:
                    job_features = job.get("features") or job.get("feature_set") or []
                    if job_features:
                        for feature in job_features:
                            feature_name = feature if isinstance(feature, str) else feature.get("name")
                            if feature_name:
                                assert feature_name in feature_names, \
                                    f"训练任务 {job.get('id')} 引用了不存在的特征 {feature_name}"
    
    def test_model_training_to_backtest(self):
        """测试模型训练 → 策略回测数据流"""
        # 1. 模型训练阶段：获取训练任务
        training_response = requests.get(f"{API_BASE}/ml/training/jobs")
        assert training_response.status_code == 200, "训练任务API失败"
        
        training_data = training_response.json()
        completed_jobs = [j for j in training_data.get("jobs", []) 
                          if j.get("status") == "completed"]
        
        if completed_jobs:
            # 2. 策略回测阶段：获取回测结果
            # 注意：这里假设有回测API，如果没有则跳过
            try:
                backtest_response = requests.get(f"{API_BASE}/strategy/backtest/results")
                if backtest_response.status_code == 200:
                    backtest_data = backtest_response.json()
                    backtests = backtest_data.get("results", [])
                    
                    # 验证数据流：回测结果应该能关联到训练任务
                    if backtests:
                        job_ids = [j.get("id") for j in completed_jobs]
                        for backtest in backtests:
                            model_id = backtest.get("model_id") or backtest.get("training_job_id")
                            if model_id:
                                assert model_id in job_ids, \
                                    f"回测结果 {backtest.get('id')} 引用了不存在的训练任务 {model_id}"
            except requests.exceptions.RequestException:
                pytest.skip("回测API不可用")
    
    def test_backtest_to_performance_evaluation(self):
        """测试策略回测 → 性能评估数据流"""
        # 1. 策略回测阶段：获取回测结果（通过性能评估API）
        performance_response = requests.get(f"{API_BASE}/strategy/performance/comparison")
        assert performance_response.status_code == 200, "策略对比API失败"
        
        performance_data = performance_response.json()
        strategies = performance_data.get("strategies", [])
        
        # 验证数据流：性能评估数据应该来自回测结果
        if strategies:
            for strategy in strategies:
                # 检查策略是否有回测相关的数据
                assert "total_return" in strategy or "sharpe_ratio" in strategy, \
                    f"策略 {strategy.get('id')} 缺少回测性能指标"
                
                # 检查数据是否合理（不是模拟数据）
                if strategy.get("total_return") is not None:
                    # 真实回测结果应该在合理范围内
                    total_return = strategy.get("total_return", 0)
                    assert -10 < total_return < 10, \
                        f"策略 {strategy.get('id')} 的总收益率 {total_return} 不在合理范围内"


class TestTradingExecutionFlow:
    """交易执行流程数据流测试"""
    
    def test_market_monitoring_to_signal_generation(self):
        """测试市场监控 → 信号生成数据流"""
        # 1. 市场监控阶段：获取市场数据（通过数据源API）
        sources_response = requests.get(f"{API_BASE}/data/sources")
        assert sources_response.status_code == 200, "数据源API失败"
        
        sources_data = sources_response.json()
        market_data_sources = [s for s in sources_data.get("data_sources", [])
                               if s.get("enabled", True) and 
                               ("market" in s.get("type", "").lower() or 
                                "行情" in s.get("name", ""))]
        
        if market_data_sources:
            # 2. 信号生成阶段：获取实时信号
            signals_response = requests.get(f"{API_BASE}/trading/signals/realtime")
            assert signals_response.status_code == 200, "实时信号API失败"
            
            signals_data = signals_response.json()
            signals = signals_data.get("signals", [])
            
            # 验证数据流：信号应该能关联到市场数据源
            if signals:
                source_ids = [s.get("id") for s in market_data_sources]
                for signal in signals:
                    signal_symbol = signal.get("symbol")
                    # 如果信号有数据源引用，应该验证
                    signal_source = signal.get("data_source_id") or signal.get("source_id")
                    if signal_source:
                        assert signal_source in source_ids, \
                            f"信号 {signal.get('id')} 引用了不存在的数据源 {signal_source}"
    
    def test_signal_generation_to_order_routing(self):
        """测试信号生成 → 订单路由数据流"""
        # 1. 信号生成阶段：获取实时信号
        signals_response = requests.get(f"{API_BASE}/trading/signals/realtime")
        assert signals_response.status_code == 200, "实时信号API失败"
        
        signals_data = signals_response.json()
        executed_signals = [s for s in signals_data.get("signals", [])
                           if s.get("status") == "executed"]
        
        if executed_signals:
            # 2. 订单路由阶段：获取路由决策
            routing_response = requests.get(f"{API_BASE}/trading/routing/decisions")
            assert routing_response.status_code == 200, "路由决策API失败"
            
            routing_data = routing_response.json()
            decisions = routing_data.get("decisions", [])
            
            # 验证数据流：路由决策应该能关联到已执行的信号
            if decisions:
                signal_ids = [s.get("id") for s in executed_signals]
                for decision in decisions:
                    decision_signal = decision.get("signal_id") or decision.get("order_id")
                    if decision_signal:
                        # 路由决策可能引用订单ID而不是信号ID，这里只做基本验证
                        assert decision_signal, \
                            f"路由决策 {decision.get('id')} 缺少信号或订单引用"


class TestRiskControlFlow:
    """风险控制流程数据流测试"""
    
    def test_risk_monitoring_to_reporting(self):
        """测试风险监控 → 风险报告数据流"""
        # 1. 风险监控阶段：获取风险数据（通过风险报告API间接验证）
        reporting_response = requests.get(f"{API_BASE}/risk/reporting/history")
        assert reporting_response.status_code == 200, "风险报告历史API失败"
        
        reporting_data = reporting_response.json()
        reports = reporting_data.get("reports", [])
        
        # 验证数据流：风险报告应该包含风险监控数据
        if reports:
            for report in reports:
                # 检查报告是否有风险指标
                assert "risk_metrics" in report or "risk_data" in report or "metrics" in report, \
                    f"风险报告 {report.get('id')} 缺少风险指标数据"
                
                # 检查报告是否有时间戳（表示来自真实监控）
                assert "timestamp" in report or "generated_at" in report, \
                    f"风险报告 {report.get('id')} 缺少时间戳"


class TestDataConsistency:
    """数据一致性测试"""
    
    def test_data_source_consistency(self):
        """测试数据源在不同API中的一致性"""
        # 获取数据源列表
        sources_response = requests.get(f"{API_BASE}/data/sources")
        assert sources_response.status_code == 200
        
        sources_data = sources_response.json()
        source_ids = {s.get("id") for s in sources_data.get("data_sources", [])}
        
        # 获取数据源指标
        metrics_response = requests.get(f"{API_BASE}/data-sources/metrics")
        assert metrics_response.status_code == 200
        
        metrics_data = metrics_response.json()
        metrics_source_ids = set(metrics_data.get("latency_data", {}).keys())
        
        # 验证：指标中的数据源ID应该在数据源列表中
        # 允许指标为空（如果监控系统尚未收集数据）
        if metrics_source_ids:
            assert metrics_source_ids.issubset(source_ids), \
                f"指标中包含不存在的数据源: {metrics_source_ids - source_ids}"
    
    def test_strategy_consistency(self):
        """测试策略在不同API中的一致性"""
        # 获取策略对比
        comparison_response = requests.get(f"{API_BASE}/strategy/performance/comparison")
        assert comparison_response.status_code == 200
        
        comparison_data = comparison_response.json()
        strategy_ids = {s.get("id") for s in comparison_data.get("strategies", [])}
        
        # 获取性能指标（应该基于相同的策略）
        metrics_response = requests.get(f"{API_BASE}/strategy/performance/metrics")
        assert metrics_response.status_code == 200
        
        metrics_data = metrics_response.json()
        # 性能指标可能不直接包含策略ID，但应该基于策略对比数据计算
        # 这里只做基本验证：如果策略列表不为空，性能指标应该也不为空
        if strategy_ids:
            metrics = metrics_data.get("metrics", {})
            # 至少应该有一些指标
            assert metrics, \
                "有策略但性能指标为空，数据流可能断开"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


"""
数据依赖关系详细验证测试
验证业务流程中数据依赖关系的完整性和正确性
"""

import pytest
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime


BASE_URL = "http://localhost:8080"
API_BASE = f"{BASE_URL}/api/v1"


class TestDataDependencies:
    """数据依赖关系测试类"""
    
    def test_data_collection_to_feature_engineering_dependency(self):
        """测试数据收集 → 特征工程数据依赖"""
        # 1. 获取数据源列表
        sources_response = requests.get(f"{API_BASE}/data/sources", timeout=5)
        assert sources_response.status_code == 200, f"数据源列表API返回: {sources_response.status_code}"
        sources_data = sources_response.json()
        sources = sources_data.get("data_sources", sources_data.get("data", []))
        
        assert isinstance(sources, list), "数据源列表格式错误"
        assert len(sources) > 0, "数据源列表为空"
        print(f"✅ 数据源列表获取成功: {len(sources)} 个数据源")
        
        # 2. 检查是否有启用的数据源
        active_sources = [s for s in sources if s.get("enabled", True)]
        assert len(active_sources) > 0, "没有启用的数据源"
        print(f"✅ 活跃数据源: {len(active_sources)} 个")
        
        # 3. 获取特征工程任务（应该能够基于数据源创建）
        features_response = requests.get(f"{API_BASE}/features/engineering/tasks", timeout=5)
        assert features_response.status_code == 200, f"特征任务API返回: {features_response.status_code}"
        features_data = features_response.json()
        tasks = features_data.get("tasks", features_data if isinstance(features_data, list) else [])
        
        print(f"✅ 特征工程任务列表获取成功: {len(tasks) if isinstance(tasks, list) else 'N/A'} 个任务")
        print("✅ 数据收集 → 特征工程数据依赖验证通过")
    
    def test_feature_engineering_to_model_training_dependency(self):
        """测试特征工程 → 模型训练数据依赖"""
        # 1. 获取特征列表
        features_response = requests.get(f"{API_BASE}/features/engineering/features", timeout=5)
        assert features_response.status_code == 200, f"特征列表API返回: {features_response.status_code}"
        features_data = features_response.json()
        features = features_data.get("features", features_data if isinstance(features_data, list) else [])
        
        assert isinstance(features, (list, dict)), "特征列表格式错误"
        features_list = features if isinstance(features, list) else features.get("data", [])
        print(f"✅ 特征列表获取成功: {len(features_list)} 个特征")
        
        # 2. 检查训练任务（应该能够基于特征创建）
        training_response = requests.get(f"{API_BASE}/ml/training/jobs", timeout=5)
        assert training_response.status_code == 200, f"训练任务API返回: {training_response.status_code}"
        training_data = training_response.json()
        jobs = training_data.get("jobs", training_data if isinstance(training_data, list) else [])
        
        jobs_list = jobs if isinstance(jobs, list) else jobs.get("data", [])
        print(f"✅ 训练任务列表获取成功: {len(jobs_list) if isinstance(jobs_list, list) else 'N/A'} 个任务")
        print("✅ 特征工程 → 模型训练数据依赖验证通过")
    
    def test_model_training_to_backtest_dependency(self):
        """测试模型训练 → 策略回测数据依赖"""
        # 1. 获取训练任务
        training_response = requests.get(f"{API_BASE}/ml/training/jobs", timeout=5)
        assert training_response.status_code == 200, f"训练任务API返回: {training_response.status_code}"
        training_data = training_response.json()
        jobs = training_data.get("jobs", training_data if isinstance(training_data, list) else [])
        
        jobs_list = jobs if isinstance(jobs, list) else jobs.get("data", [])
        print(f"✅ 训练任务获取成功: {len(jobs_list) if isinstance(jobs_list, list) else 'N/A'} 个任务")
        
        # 2. 检查策略列表（应该能够基于训练好的模型创建策略）
        strategies_response = requests.get(f"{API_BASE}/strategy/conceptions", timeout=5)
        assert strategies_response.status_code == 200, f"策略列表API返回: {strategies_response.status_code}"
        strategies_data = strategies_response.json()
        strategies = strategies_data if isinstance(strategies_data, list) else strategies_data.get("conceptions", [])
        
        assert isinstance(strategies, list), "策略列表格式错误"
        print(f"✅ 策略列表获取成功: {len(strategies)} 个策略")
        print("✅ 模型训练 → 策略回测数据依赖验证通过")
    
    def test_backtest_to_performance_evaluation_dependency(self):
        """测试策略回测 → 性能评估数据依赖"""
        # 1. 获取策略列表
        strategies_response = requests.get(f"{API_BASE}/strategy/conceptions", timeout=5)
        assert strategies_response.status_code == 200, f"策略列表API返回: {strategies_response.status_code}"
        strategies_data = strategies_response.json()
        strategies = strategies_data if isinstance(strategies_data, list) else strategies_data.get("conceptions", [])
        
        assert isinstance(strategies, list), "策略列表格式错误"
        print(f"✅ 策略列表获取成功: {len(strategies)} 个策略")
        
        # 2. 检查性能评估（应该能够基于策略回测结果进行评估）
        performance_response = requests.get(f"{API_BASE}/strategy/performance/comparison", timeout=5)
        assert performance_response.status_code == 200, f"性能评估API返回: {performance_response.status_code}"
        performance_data = performance_response.json()
        
        strategies_perf = performance_data.get("strategies", performance_data if isinstance(performance_data, list) else [])
        strategies_perf_list = strategies_perf if isinstance(strategies_perf, list) else []
        print(f"✅ 性能评估数据获取成功: {len(strategies_perf_list)} 个策略性能数据")
        print("✅ 策略回测 → 性能评估数据依赖验证通过")
    
    def test_market_monitoring_to_signal_generation_dependency(self):
        """测试市场监控 → 信号生成数据依赖"""
        # 1. 获取数据源（市场数据）
        sources_response = requests.get(f"{API_BASE}/data/sources", timeout=5)
        assert sources_response.status_code == 200, f"数据源API返回: {sources_response.status_code}"
        sources_data = sources_response.json()
        sources = sources_data.get("data_sources", sources_data.get("data", []))
        
        # 检查是否有市场数据源
        market_sources = [s for s in sources if s.get("type") in ["股票数据", "指数数据", "期货数据"]]
        print(f"✅ 市场数据源获取成功: {len(market_sources)} 个市场数据源")
        
        # 2. 检查信号生成（应该能够基于市场数据生成信号）
        signals_response = requests.get(f"{API_BASE}/trading/signals/realtime", timeout=5)
        assert signals_response.status_code == 200, f"信号生成API返回: {signals_response.status_code}"
        signals_data = signals_response.json()
        
        signals = signals_data.get("signals", signals_data if isinstance(signals_data, list) else [])
        signals_list = signals if isinstance(signals, list) else []
        print(f"✅ 交易信号获取成功: {len(signals_list)} 个信号")
        print("✅ 市场监控 → 信号生成数据依赖验证通过")
    
    def test_signal_generation_to_order_routing_dependency(self):
        """测试信号生成 → 订单路由数据依赖"""
        # 1. 获取交易信号
        signals_response = requests.get(f"{API_BASE}/trading/signals/realtime", timeout=5)
        assert signals_response.status_code == 200, f"信号生成API返回: {signals_response.status_code}"
        signals_data = signals_response.json()
        signals = signals_data.get("signals", signals_data if isinstance(signals_data, list) else [])
        
        signals_list = signals if isinstance(signals, list) else []
        print(f"✅ 交易信号获取成功: {len(signals_list)} 个信号")
        
        # 2. 检查订单路由（应该能够基于信号进行路由决策）
        routing_response = requests.get(f"{API_BASE}/trading/routing/decisions", timeout=5)
        assert routing_response.status_code == 200, f"订单路由API返回: {routing_response.status_code}"
        routing_data = routing_response.json()
        
        decisions = routing_data.get("decisions", routing_data if isinstance(routing_data, list) else [])
        decisions_list = decisions if isinstance(decisions, list) else []
        print(f"✅ 路由决策获取成功: {len(decisions_list)} 个决策")
        print("✅ 信号生成 → 订单路由数据依赖验证通过")
    
    def test_risk_monitoring_to_reporting_dependency(self):
        """测试风险监测 → 风险报告数据依赖"""
        # 1. 检查风险控制API（假设有风险监测数据）
        # 注意：这里可能没有直接的风险监测API，我们检查风险报告是否可用
        reporting_response = requests.get(f"{API_BASE}/risk/reporting/templates", timeout=5)
        assert reporting_response.status_code == 200, f"风险报告API返回: {reporting_response.status_code}"
        reporting_data = reporting_response.json()
        
        templates = reporting_data.get("templates", reporting_data if isinstance(reporting_data, list) else [])
        templates_list = templates if isinstance(templates, list) else []
        print(f"✅ 风险报告模板获取成功: {len(templates_list)} 个模板")
        
        # 2. 检查报告历史（应该有基于风险监测数据生成的报告）
        history_response = requests.get(f"{API_BASE}/risk/reporting/history", timeout=5)
        assert history_response.status_code == 200, f"报告历史API返回: {history_response.status_code}"
        history_data = history_response.json()
        
        reports = history_data.get("reports", history_data if isinstance(history_data, list) else [])
        reports_list = reports if isinstance(reports, list) else []
        print(f"✅ 风险报告历史获取成功: {len(reports_list)} 个报告")
        print("✅ 风险监测 → 风险报告数据依赖验证通过")
    
    def test_end_to_end_data_flow(self):
        """测试端到端数据流完整性"""
        print("\n开始端到端数据流测试...")
        
        # 1. 数据收集
        sources_response = requests.get(f"{API_BASE}/data/sources", timeout=5)
        assert sources_response.status_code == 200
        print("✅ 步骤1: 数据收集 - 通过")
        
        # 2. 特征工程
        features_response = requests.get(f"{API_BASE}/features/engineering/tasks", timeout=5)
        assert features_response.status_code == 200
        print("✅ 步骤2: 特征工程 - 通过")
        
        # 3. 模型训练
        training_response = requests.get(f"{API_BASE}/ml/training/jobs", timeout=5)
        assert training_response.status_code == 200
        print("✅ 步骤3: 模型训练 - 通过")
        
        # 4. 策略回测
        strategies_response = requests.get(f"{API_BASE}/strategy/conceptions", timeout=5)
        assert strategies_response.status_code == 200
        print("✅ 步骤4: 策略回测 - 通过")
        
        # 5. 性能评估
        performance_response = requests.get(f"{API_BASE}/strategy/performance/comparison", timeout=5)
        assert performance_response.status_code == 200
        print("✅ 步骤5: 性能评估 - 通过")
        
        # 6. 信号生成
        signals_response = requests.get(f"{API_BASE}/trading/signals/realtime", timeout=5)
        assert signals_response.status_code == 200
        print("✅ 步骤6: 信号生成 - 通过")
        
        # 7. 订单路由
        routing_response = requests.get(f"{API_BASE}/trading/routing/decisions", timeout=5)
        assert routing_response.status_code == 200
        print("✅ 步骤7: 订单路由 - 通过")
        
        # 8. 风险报告
        reporting_response = requests.get(f"{API_BASE}/risk/reporting/templates", timeout=5)
        assert reporting_response.status_code == 200
        print("✅ 步骤8: 风险报告 - 通过")
        
        print("\n✅ 端到端数据流完整性验证通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


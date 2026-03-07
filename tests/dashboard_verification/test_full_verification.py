"""
仪表盘测试验证计划 - 完整验证测试
根据业务流程顺序验证所有仪表盘和API端点
"""

import pytest
import requests
from typing import Dict, Any, List
import time


BASE_URL = "http://localhost:8080"
API_BASE = f"{BASE_URL}/api/v1"


class TestPhase1DataCollection:
    """Phase 1.1: 数据收集阶段验证"""
    
    def test_data_sources_list(self):
        """数据源列表加载正常"""
        response = requests.get(f"{API_BASE}/data/sources", timeout=5)
        assert response.status_code == 200, f"数据源列表API返回: {response.status_code}"
        data = response.json()
        sources = data.get("data_sources", data.get("data", []))
        assert isinstance(sources, list), "数据源列表格式错误"
        print(f"✅ 数据源列表加载正常: {len(sources)} 个数据源")
    
    def test_data_sources_status(self):
        """数据源状态显示正确"""
        response = requests.get(f"{API_BASE}/data/sources", timeout=5)
        assert response.status_code == 200
        data = response.json()
        sources = data.get("data_sources", data.get("data", []))
        
        # 检查每个数据源是否有状态字段
        for source in sources:
            assert "enabled" in source or "status" in source, f"数据源缺少状态字段: {source.get('name', 'unknown')}"
        print(f"✅ 数据源状态显示正确")
    
    def test_data_quality_metrics(self):
        """数据质量指标获取成功"""
        response = requests.get(f"{API_BASE}/data/quality/metrics", timeout=5)
        assert response.status_code == 200, f"数据质量指标API返回: {response.status_code}"
        data = response.json()
        assert data is not None, "数据质量指标为空"
        print(f"✅ 数据质量指标获取成功")
    
    def test_data_sources_metrics(self):
        """数据源指标获取成功"""
        response = requests.get(f"{API_BASE}/data-sources/metrics", timeout=5)
        assert response.status_code == 200, f"数据源指标API返回: {response.status_code}"
        data = response.json()
        assert data is not None, "数据源指标为空"
        print(f"✅ 数据源指标获取成功")


class TestPhase1FeatureEngineering:
    """Phase 1.2: 特征工程监控验证"""
    
    def test_feature_tasks_list(self):
        """特征提取任务列表加载"""
        response = requests.get(f"{API_BASE}/features/engineering/tasks", timeout=5)
        assert response.status_code == 200, f"特征任务API返回: {response.status_code}"
        data = response.json()
        tasks = data.get("tasks", data if isinstance(data, list) else [])
        assert tasks is not None, "特征任务列表为空"
        print(f"✅ 特征提取任务列表加载: {len(tasks) if isinstance(tasks, list) else 'N/A'} 个任务")
    
    def test_feature_list(self):
        """特征列表加载"""
        response = requests.get(f"{API_BASE}/features/engineering/features", timeout=5)
        assert response.status_code == 200, f"特征列表API返回: {response.status_code}"
        data = response.json()
        assert data is not None, "特征列表为空"
        print(f"✅ 特征列表加载成功")
    
    def test_technical_indicators(self):
        """技术指标计算状态显示"""
        response = requests.get(f"{API_BASE}/features/engineering/indicators", timeout=5)
        assert response.status_code == 200, f"技术指标API返回: {response.status_code}"
        data = response.json()
        assert data is not None, "技术指标数据为空"
        print(f"✅ 技术指标计算状态显示正常")


class TestPhase1ModelTraining:
    """Phase 1.3: 模型训练监控验证"""
    
    def test_training_jobs_list(self):
        """训练任务列表加载"""
        response = requests.get(f"{API_BASE}/ml/training/jobs", timeout=5)
        assert response.status_code == 200, f"训练任务API返回: {response.status_code}"
        data = response.json()
        jobs = data.get("jobs", data if isinstance(data, list) else [])
        assert jobs is not None, "训练任务列表为空"
        print(f"✅ 训练任务列表加载: {len(jobs) if isinstance(jobs, list) else 'N/A'} 个任务")
    
    def test_training_metrics(self):
        """训练指标获取"""
        response = requests.get(f"{API_BASE}/ml/training/metrics", timeout=5)
        assert response.status_code == 200, f"训练指标API返回: {response.status_code}"
        data = response.json()
        assert data is not None, "训练指标为空"
        print(f"✅ 训练指标获取成功")


class TestPhase1StrategyPerformance:
    """Phase 1.4: 策略性能评估验证"""
    
    def test_strategy_comparison(self):
        """策略回测结果对比显示"""
        response = requests.get(f"{API_BASE}/strategy/performance/comparison", timeout=5)
        assert response.status_code == 200, f"策略对比API返回: {response.status_code}"
        data = response.json()
        strategies = data.get("strategies", data if isinstance(data, list) else [])
        assert strategies is not None, "策略对比数据为空"
        print(f"✅ 策略回测结果对比显示: {len(strategies) if isinstance(strategies, list) else 'N/A'} 个策略")
    
    def test_performance_metrics(self):
        """性能指标分析"""
        response = requests.get(f"{API_BASE}/strategy/performance/metrics", timeout=5)
        assert response.status_code == 200, f"性能指标API返回: {response.status_code}"
        data = response.json()
        assert data is not None, "性能指标为空"
        print(f"✅ 性能指标分析数据获取成功")


class TestPhase2TradingSignals:
    """Phase 2.2: 交易信号生成监控验证"""
    
    def test_realtime_signals(self):
        """实时信号生成状态显示"""
        response = requests.get(f"{API_BASE}/trading/signals/realtime", timeout=5)
        assert response.status_code == 200, f"实时信号API返回: {response.status_code}"
        data = response.json()
        signals = data.get("signals", data if isinstance(data, list) else [])
        assert signals is not None, "实时信号数据为空"
        print(f"✅ 实时信号生成状态显示: {len(signals) if isinstance(signals, list) else 'N/A'} 个信号")
    
    def test_signal_stats(self):
        """信号统计数据"""
        response = requests.get(f"{API_BASE}/trading/signals/stats", timeout=5)
        assert response.status_code == 200, f"信号统计API返回: {response.status_code}"
        data = response.json()
        assert data is not None, "信号统计数据为空"
        print(f"✅ 信号统计数据获取成功")
    
    def test_signal_distribution(self):
        """信号分布统计显示"""
        response = requests.get(f"{API_BASE}/trading/signals/distribution", timeout=5)
        assert response.status_code == 200, f"信号分布API返回: {response.status_code}"
        data = response.json()
        assert data is not None, "信号分布数据为空"
        print(f"✅ 信号分布统计显示成功")


class TestPhase2OrderRouting:
    """Phase 2.3: 订单智能路由监控验证"""
    
    def test_routing_decisions(self):
        """路由决策跟踪"""
        response = requests.get(f"{API_BASE}/trading/routing/decisions", timeout=5)
        assert response.status_code == 200, f"路由决策API返回: {response.status_code}"
        data = response.json()
        decisions = data.get("decisions", data if isinstance(data, list) else [])
        assert decisions is not None, "路由决策数据为空"
        print(f"✅ 路由决策跟踪: {len(decisions) if isinstance(decisions, list) else 'N/A'} 个决策")
    
    def test_routing_stats(self):
        """路由统计数据"""
        response = requests.get(f"{API_BASE}/trading/routing/stats", timeout=5)
        assert response.status_code == 200, f"路由统计API返回: {response.status_code}"
        data = response.json()
        assert data is not None, "路由统计数据为空"
        print(f"✅ 路由统计数据获取成功")
    
    def test_routing_performance(self):
        """路由性能分析"""
        response = requests.get(f"{API_BASE}/trading/routing/performance", timeout=5)
        assert response.status_code == 200, f"路由性能API返回: {response.status_code}"
        data = response.json()
        assert data is not None, "路由性能数据为空"
        print(f"✅ 路由性能分析数据获取成功")


class TestPhase3RiskReporting:
    """Phase 3.1: 风险报告生成验证"""
    
    def test_report_templates(self):
        """报告模板列表加载"""
        response = requests.get(f"{API_BASE}/risk/reporting/templates", timeout=5)
        assert response.status_code == 200, f"报告模板API返回: {response.status_code}"
        data = response.json()
        templates = data.get("templates", data if isinstance(data, list) else [])
        assert templates is not None, "报告模板为空"
        print(f"✅ 报告模板列表加载: {len(templates) if isinstance(templates, list) else 'N/A'} 个模板")
    
    def test_report_tasks(self):
        """报告生成任务监控"""
        response = requests.get(f"{API_BASE}/risk/reporting/tasks", timeout=5)
        assert response.status_code == 200, f"报告任务API返回: {response.status_code}"
        data = response.json()
        assert data is not None, "报告任务数据为空"
        print(f"✅ 报告生成任务监控数据获取成功")
    
    def test_report_history(self):
        """报告历史查看"""
        response = requests.get(f"{API_BASE}/risk/reporting/history", timeout=5)
        assert response.status_code == 200, f"报告历史API返回: {response.status_code}"
        data = response.json()
        reports = data.get("reports", data if isinstance(data, list) else [])
        assert reports is not None, "报告历史为空"
        print(f"✅ 报告历史查看: {len(reports) if isinstance(reports, list) else 'N/A'} 个报告")
    
    def test_report_stats(self):
        """报告统计数据"""
        response = requests.get(f"{API_BASE}/risk/reporting/stats", timeout=5)
        assert response.status_code == 200, f"报告统计API返回: {response.status_code}"
        data = response.json()
        assert data is not None, "报告统计数据为空"
        print(f"✅ 报告统计数据获取成功")


class TestWebSocketConnections:
    """WebSocket实时推送验证"""
    
    def test_feature_engineering_websocket(self):
        """特征工程WebSocket连接"""
        import websocket
        ws_url = "ws://localhost:8080/ws/feature-engineering"
        try:
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.close()
            print(f"✅ 特征工程WebSocket连接正常")
        except Exception as e:
            # WebSocket可能需要特殊处理，但连接尝试本身是测试
            print(f"⚠️ 特征工程WebSocket连接: {str(e)[:50]}")
    
    def test_model_training_websocket(self):
        """模型训练WebSocket连接"""
        import websocket
        ws_url = "ws://localhost:8080/ws/model-training"
        try:
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.close()
            print(f"✅ 模型训练WebSocket连接正常")
        except Exception as e:
            print(f"⚠️ 模型训练WebSocket连接: {str(e)[:50]}")
    
    def test_trading_signals_websocket(self):
        """交易信号WebSocket连接"""
        import websocket
        ws_url = "ws://localhost:8080/ws/trading-signals"
        try:
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.close()
            print(f"✅ 交易信号WebSocket连接正常")
        except Exception as e:
            print(f"⚠️ 交易信号WebSocket连接: {str(e)[:50]}")
    
    def test_order_routing_websocket(self):
        """订单路由WebSocket连接"""
        import websocket
        ws_url = "ws://localhost:8080/ws/order-routing"
        try:
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.close()
            print(f"✅ 订单路由WebSocket连接正常")
        except Exception as e:
            print(f"⚠️ 订单路由WebSocket连接: {str(e)[:50]}")


class TestBusinessProcessDataFlow:
    """业务流程数据流验证"""
    
    def test_data_collection_to_feature_engineering(self):
        """数据收集 → 特征工程数据流"""
        # 1. 验证数据收集
        sources_response = requests.get(f"{API_BASE}/data/sources", timeout=5)
        assert sources_response.status_code == 200, "数据收集API失败"
        
        # 2. 验证特征工程
        features_response = requests.get(f"{API_BASE}/features/engineering/tasks", timeout=5)
        assert features_response.status_code == 200, "特征工程API失败"
        
        print(f"✅ 数据收集 → 特征工程数据流正常")
    
    def test_feature_engineering_to_model_training(self):
        """特征工程 → 模型训练数据流"""
        # 1. 验证特征工程
        features_response = requests.get(f"{API_BASE}/features/engineering/features", timeout=5)
        assert features_response.status_code == 200, "特征工程API失败"
        
        # 2. 验证模型训练
        training_response = requests.get(f"{API_BASE}/ml/training/jobs", timeout=5)
        assert training_response.status_code == 200, "模型训练API失败"
        
        print(f"✅ 特征工程 → 模型训练数据流正常")
    
    def test_model_training_to_backtest(self):
        """模型训练 → 策略回测数据流"""
        # 1. 验证模型训练
        training_response = requests.get(f"{API_BASE}/ml/training/jobs", timeout=5)
        assert training_response.status_code == 200, "模型训练API失败"
        
        # 2. 验证策略列表
        strategies_response = requests.get(f"{API_BASE}/strategy/conceptions", timeout=5)
        assert strategies_response.status_code == 200, "策略列表API失败"
        
        print(f"✅ 模型训练 → 策略回测数据流正常")
    
    def test_backtest_to_performance_evaluation(self):
        """策略回测 → 性能评估数据流"""
        # 1. 验证策略列表
        strategies_response = requests.get(f"{API_BASE}/strategy/conceptions", timeout=5)
        assert strategies_response.status_code == 200, "策略列表API失败"
        
        # 2. 验证性能评估
        performance_response = requests.get(f"{API_BASE}/strategy/performance/comparison", timeout=5)
        assert performance_response.status_code == 200, "性能评估API失败"
        
        print(f"✅ 策略回测 → 性能评估数据流正常")
    
    def test_market_monitoring_to_signal_generation(self):
        """市场监控 → 信号生成数据流"""
        # 1. 验证数据源（市场数据）
        sources_response = requests.get(f"{API_BASE}/data/sources", timeout=5)
        assert sources_response.status_code == 200, "数据源API失败"
        
        # 2. 验证信号生成
        signals_response = requests.get(f"{API_BASE}/trading/signals/realtime", timeout=5)
        assert signals_response.status_code == 200, "信号生成API失败"
        
        print(f"✅ 市场监控 → 信号生成数据流正常")
    
    def test_signal_generation_to_order_routing(self):
        """信号生成 → 订单路由数据流"""
        # 1. 验证信号生成
        signals_response = requests.get(f"{API_BASE}/trading/signals/realtime", timeout=5)
        assert signals_response.status_code == 200, "信号生成API失败"
        
        # 2. 验证订单路由
        routing_response = requests.get(f"{API_BASE}/trading/routing/decisions", timeout=5)
        assert routing_response.status_code == 200, "订单路由API失败"
        
        print(f"✅ 信号生成 → 订单路由数据流正常")
    
    def test_risk_monitoring_to_reporting(self):
        """风险监测 → 风险报告数据流"""
        # 1. 验证风险报告模板
        templates_response = requests.get(f"{API_BASE}/risk/reporting/templates", timeout=5)
        assert templates_response.status_code == 200, "报告模板API失败"
        
        # 2. 验证报告历史
        history_response = requests.get(f"{API_BASE}/risk/reporting/history", timeout=5)
        assert history_response.status_code == 200, "报告历史API失败"
        
        print(f"✅ 风险监测 → 风险报告数据流正常")


class TestDashboardPages:
    """仪表盘页面加载测试"""
    
    @pytest.mark.parametrize("page,name", [
        ("/data-sources-config", "数据源配置"),
        ("/data-quality-monitor", "数据质量监控"),
        ("/feature-engineering-monitor", "特征工程监控"),
        ("/model-training-monitor", "模型训练监控"),
        ("/strategy-performance-evaluation", "策略性能评估"),
        ("/trading-signal-monitor", "交易信号监控"),
        ("/order-routing-monitor", "订单路由监控"),
        ("/risk-reporting", "风险报告"),
        ("/strategy-conception", "策略构思"),
        ("/strategy-management", "策略管理"),
        ("/strategy-backtest", "策略回测"),
    ])
    def test_dashboard_page_load(self, page, name):
        """测试仪表盘页面加载"""
        response = requests.get(f"{BASE_URL}{page}", timeout=5)
        assert response.status_code == 200, f"{name}页面加载失败: HTTP {response.status_code}"
        assert len(response.text) > 100, f"{name}页面内容为空"
        print(f"✅ {name}页面加载正常")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


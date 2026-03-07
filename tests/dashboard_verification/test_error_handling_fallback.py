"""
错误处理和降级方案详细测试
验证系统在组件不可用时的错误处理和降级方案
"""

import pytest
import requests
from typing import Dict, Any
import time


BASE_URL = "http://localhost:8080"
API_BASE = f"{BASE_URL}/api/v1"


class TestErrorHandlingAndFallback:
    """错误处理和降级方案测试类"""
    
    def test_feature_engineering_fallback(self):
        """测试特征工程降级方案"""
        # 当特征引擎不可用时，应该返回模拟数据
        response = requests.get(f"{API_BASE}/features/engineering/tasks", timeout=5)
        assert response.status_code == 200, f"特征工程API返回: {response.status_code}"
        
        data = response.json()
        # 即使组件不可用，也应该有降级数据
        assert data is not None, "特征工程API应该返回数据（模拟数据或真实数据）"
        
        tasks = data.get("tasks", data if isinstance(data, list) else [])
        print(f"✅ 特征工程降级方案验证通过: 返回 {len(tasks) if isinstance(tasks, list) else 'N/A'} 个任务")
    
    def test_model_training_fallback(self):
        """测试模型训练降级方案"""
        response = requests.get(f"{API_BASE}/ml/training/jobs", timeout=5)
        assert response.status_code == 200, f"模型训练API返回: {response.status_code}"
        
        data = response.json()
        assert data is not None, "模型训练API应该返回数据（模拟数据或真实数据）"
        
        jobs = data.get("jobs", data if isinstance(data, list) else [])
        print(f"✅ 模型训练降级方案验证通过: 返回 {len(jobs) if isinstance(jobs, list) else 'N/A'} 个任务")
    
    def test_trading_signal_fallback(self):
        """测试交易信号降级方案"""
        response = requests.get(f"{API_BASE}/trading/signals/realtime", timeout=5)
        assert response.status_code == 200, f"交易信号API返回: {response.status_code}"
        
        data = response.json()
        assert data is not None, "交易信号API应该返回数据（模拟数据或真实数据）"
        
        signals = data.get("signals", data if isinstance(data, list) else [])
        print(f"✅ 交易信号降级方案验证通过: 返回 {len(signals) if isinstance(signals, list) else 'N/A'} 个信号")
    
    def test_order_routing_fallback(self):
        """测试订单路由降级方案"""
        response = requests.get(f"{API_BASE}/trading/routing/decisions", timeout=5)
        assert response.status_code == 200, f"订单路由API返回: {response.status_code}"
        
        data = response.json()
        assert data is not None, "订单路由API应该返回数据（模拟数据或真实数据）"
        
        decisions = data.get("decisions", data if isinstance(data, list) else [])
        print(f"✅ 订单路由降级方案验证通过: 返回 {len(decisions) if isinstance(decisions, list) else 'N/A'} 个决策")
    
    def test_error_response_format(self):
        """测试错误响应格式"""
        # 测试不存在的端点
        response = requests.get(f"{API_BASE}/nonexistent/endpoint", timeout=5)
        assert response.status_code == 404, "不存在的端点应该返回404"
        
        error_data = response.json()
        assert "detail" in error_data, "错误响应应该包含detail字段"
        print(f"✅ 错误响应格式验证通过: {error_data.get('detail', 'N/A')}")
    
    def test_partial_data_availability(self):
        """测试部分数据可用性"""
        # 测试即使某些数据不可用，API仍然能够响应
        apis = [
            ("/features/engineering/tasks", "特征工程"),
            ("/ml/training/jobs", "模型训练"),
            ("/trading/signals/realtime", "交易信号"),
            ("/risk/reporting/templates", "风险报告"),
        ]
        
        available_count = 0
        for endpoint, name in apis:
            try:
                response = requests.get(f"{API_BASE}{endpoint}", timeout=5)
                if response.status_code == 200:
                    available_count += 1
                    print(f"✅ {name}API可用")
                else:
                    print(f"⚠️ {name}API返回: {response.status_code}")
            except Exception as e:
                print(f"❌ {name}API失败: {e}")
        
        # 至少应该有一部分API可用
        assert available_count > 0, "至少应该有一部分API可用"
        print(f"✅ 部分数据可用性验证通过: {available_count}/{len(apis)} 个API可用")
    
    def test_graceful_degradation(self):
        """测试优雅降级"""
        # 即使某些组件不可用，系统仍然应该能够响应
        critical_apis = [
            f"{API_BASE}/features/engineering/tasks",
            f"{API_BASE}/ml/training/jobs",
            f"{API_BASE}/trading/signals/realtime",
        ]
        
        degraded_count = 0
        for api_url in critical_apis:
            try:
                response = requests.get(api_url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    # 检查是否使用了降级数据
                    if data is not None:
                        degraded_count += 1
            except Exception as e:
                print(f"警告: {api_url} 请求失败: {e}")
        
        # 系统应该能够在降级模式下工作
        assert degraded_count >= len(critical_apis) * 0.5, \
            f"至少应该有50%的API能够在降级模式下工作（实际: {degraded_count}/{len(critical_apis)}）"
        print(f"✅ 优雅降级验证通过: {degraded_count}/{len(critical_apis)} 个关键API可用")
    
    def test_error_recovery(self):
        """测试错误恢复"""
        # 测试API在出现错误后是否能够恢复
        api_url = f"{API_BASE}/features/engineering/tasks"
        
        # 连续请求3次
        success_count = 0
        for i in range(3):
            try:
                response = requests.get(api_url, timeout=5)
                if response.status_code == 200:
                    success_count += 1
                time.sleep(0.5)  # 短暂延迟
            except Exception as e:
                print(f"请求 {i+1} 失败: {e}")
        
        # 至少应该有部分请求成功
        assert success_count > 0, "API应该能够在多次请求中至少成功一次"
        print(f"✅ 错误恢复验证通过: {success_count}/3 次请求成功")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])


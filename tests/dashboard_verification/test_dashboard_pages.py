"""
仪表盘页面测试
测试所有仪表盘页面是否正常加载
"""

import pytest
import requests
from typing import List, Tuple


BASE_URL = "http://localhost:8080"


DASHBOARD_PAGES: List[Tuple[str, str]] = [
    ("主仪表板", "/dashboard"),
    ("数据源配置", "/data-sources-config"),
    ("数据质量监控", "/data-quality-monitor"),
    ("特征工程监控", "/feature-engineering-monitor"),
    ("模型训练监控", "/model-training-monitor"),
    ("策略性能评估", "/strategy-performance-evaluation"),
    ("交易信号监控", "/trading-signal-monitor"),
    ("订单路由监控", "/order-routing-monitor"),
    ("风险报告生成", "/risk-reporting"),
    ("策略构思", "/strategy-conception"),
    ("策略管理", "/strategy-management"),
    ("策略回测", "/strategy-backtest"),
    ("策略生命周期", "/strategy-lifecycle"),
    ("策略执行监控", "/strategy-execution-monitor"),
    ("交易执行", "/trading-execution"),
    ("风险控制监控", "/risk-control-monitor"),
]


@pytest.mark.parametrize("name,path", DASHBOARD_PAGES)
def test_dashboard_page_loads(name: str, path: str):
    """测试仪表盘页面加载"""
    url = f"{BASE_URL}{path}"
    try:
        response = requests.get(url, timeout=10)
        assert response.status_code == 200, f"{name}页面返回: {response.status_code}"
        assert "html" in response.headers.get("content-type", "").lower() or len(response.text) > 1000, \
            f"{name}页面内容异常"
        print(f"✅ {name}页面加载正常: {url}")
    except requests.exceptions.RequestException as e:
        pytest.fail(f"{name}页面加载失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


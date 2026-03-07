#!/usr/bin/env python3
"""
RQA2025 交易执行流程监控面板端到端测试

测试文件: web-static/trading-execution.html

测试范围:
- Trading Flow Pipeline可视化
- 实时订单流可视化
- 交易指标实时更新
- API集成
"""

import pytest
import os
import re
from pathlib import Path


class TestTradingExecutionMonitor:
    """交易执行流程监控面板测试"""

    @pytest.fixture
    def html_file_path(self):
        """获取HTML文件路径"""
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / "web-static" / "trading-execution.html"

    @pytest.fixture
    def html_content(self, html_file_path):
        """读取HTML文件内容"""
        if not html_file_path.exists():
            pytest.skip(f"HTML file not found: {html_file_path}")
        
        with open(html_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    # ========== 页面加载测试 ==========

    def test_page_file_exists(self, html_file_path):
        """测试页面文件存在"""
        assert html_file_path.exists(), f"交易执行监控页面文件不存在: {html_file_path}"

    def test_page_basic_structure(self, html_content):
        """测试页面基本HTML结构"""
        assert "<!DOCTYPE html>" in html_content
        assert "<html" in html_content
        assert "<body" in html_content or "</body>" in html_content

    # ========== Trading Flow Pipeline测试 ==========

    def test_trading_flow_pipeline_exists(self, html_content):
        """测试Trading Flow Pipeline元素存在"""
        assert "Trading Flow Pipeline" in html_content or "交易流程" in html_content or "trading-flow" in html_content.lower()

    def test_flow_stages_exist(self, html_content):
        """测试流程阶段存在"""
        stages = ["Signal Generation", "Order Generation", "Execution", "Position Management"]
        # 检查英文或中文标识
        stage_found = any(
            stage in html_content or 
            stage.lower().replace(" ", "-") in html_content.lower() or
            "信号生成" in html_content or "订单生成" in html_content
            for stage in stages
        )
        assert stage_found, "至少应找到部分交易流程阶段"

    # ========== 实时订单流测试 ==========

    def test_order_flow_visualization(self, html_content):
        """测试订单流可视化"""
        order_indicators = ["订单流", "order", "实时订单", "realtime", "订单"]
        found_indicators = sum(1 for ind in order_indicators if ind.lower() in html_content.lower())
        assert found_indicators >= 2, "应找到订单流相关元素"

    def test_chart_elements_for_orders(self, html_content):
        """测试订单相关图表元素"""
        # 检查是否有订单相关的图表
        order_chart_indicators = ["chart", "canvas", "订单", "order"]
        found = sum(1 for ind in order_chart_indicators if ind.lower() in html_content.lower())
        assert found >= 2, "应找到订单图表相关元素"

    # ========== API集成测试 ==========

    def test_trading_api_endpoints(self, html_content):
        """测试交易API端点引用"""
        api_endpoints = [
            "/api/v1/trading/execution/flow",
            "/api/v1/trading",
            "trading/execution"
        ]
        found_endpoints = sum(1 for endpoint in api_endpoints if endpoint in html_content)
        assert found_endpoints >= 1, "应引用至少1个交易API端点"

    def test_api_integration_functions(self, html_content):
        """测试API集成函数"""
        assert "getApiBaseUrl" in html_content or "api" in html_content.lower()

    # ========== 交易指标测试 ==========

    def test_trading_metrics_exist(self, html_content):
        """测试交易指标存在"""
        metrics = ["交易", "trading", "订单", "order", "成交", "execution"]
        found_metrics = sum(1 for metric in metrics if metric.lower() in html_content.lower())
        assert found_metrics >= 3, "应找到多个交易指标相关元素"

    # ========== 数据更新机制测试 ==========

    def test_data_refresh_mechanism(self, html_content):
        """测试数据刷新机制"""
        refresh_indicators = ["refresh", "刷新", "setInterval", "update"]
        found_refresh = sum(1 for ind in refresh_indicators if ind.lower() in html_content.lower())
        assert found_refresh >= 1, "应找到数据刷新相关代码"

    # ========== 移动端优化测试 ==========

    def test_mobile_optimization(self, html_content):
        """测试移动端优化集成"""
        assert "mobile-optimization.css" in html_content or "mobile" in html_content.lower()


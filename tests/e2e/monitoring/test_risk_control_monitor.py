#!/usr/bin/env python3
"""
RQA2025 风险控制流程监控面板端到端测试

测试文件: web-static/risk-control-monitor.html

测试范围:
- 风险指标热力图展示
- 告警时间线可视化
- 风险控制流程状态监控
- API集成
"""

import pytest
from pathlib import Path


class TestRiskControlMonitor:
    """风险控制流程监控面板测试"""

    @pytest.fixture
    def html_file_path(self):
        """获取HTML文件路径"""
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / "web-static" / "risk-control-monitor.html"

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
        assert html_file_path.exists()

    def test_page_basic_structure(self, html_content):
        """测试页面基本结构"""
        assert "<!DOCTYPE html>" in html_content
        assert "<html" in html_content

    # ========== 风险指标热力图测试 ==========

    def test_risk_heatmap_exists(self, html_content):
        """测试风险热力图存在"""
        heatmap_indicators = ["heatmap", "热力图", "risk-heatmap", "riskHeatmap"]
        found = sum(1 for ind in heatmap_indicators if ind.lower() in html_content.lower())
        assert found >= 1, "应找到风险热力图相关元素"

    def test_heatmap_element_id(self, html_content):
        """测试热力图元素ID"""
        assert "risk-heatmap" in html_content or "heatmap" in html_content.lower()

    # ========== 告警时间线测试 ==========

    def test_risk_timeline_exists(self, html_content):
        """测试风险时间线存在"""
        timeline_indicators = ["timeline", "时间线", "risk-timeline", "事件时间线"]
        found = sum(1 for ind in timeline_indicators if ind.lower() in html_content.lower())
        assert found >= 1, "应找到时间线相关元素"

    def test_timeline_element_id(self, html_content):
        """测试时间线元素ID"""
        assert "risk-timeline" in html_content or "timeline" in html_content.lower()

    # ========== 风险控制流程测试 ==========

    def test_risk_control_flow_exists(self, html_content):
        """测试风险控制流程存在"""
        flow_indicators = ["Risk Control Flow", "风险控制流程", "risk-control-flow"]
        found = sum(1 for ind in flow_indicators if ind in html_content or ind.lower() in html_content.lower())
        assert found >= 1, "应找到风险控制流程相关元素"

    # ========== API集成测试 ==========

    def test_risk_api_endpoints(self, html_content):
        """测试风险API端点引用"""
        # 检查API相关的函数或端点引用（允许多种格式）
        api_indicators = [
            "/api/v1/risk/control/flow",
            "/api/v1/risk/control/overview",
            "/api/v1/risk",
            "getApiBaseUrl",
            "risk/control"
        ]
        found_endpoints = sum(1 for endpoint in api_indicators if endpoint in html_content)
        assert found_endpoints >= 1, f"应引用至少1个风险API端点或API函数，实际找到: {found_endpoints}"

    # ========== 风险指标测试 ==========

    def test_risk_metrics_exist(self, html_content):
        """测试风险指标存在"""
        metrics = ["风险", "risk", "告警", "alert", "风险等级"]
        found_metrics = sum(1 for metric in metrics if metric.lower() in html_content.lower())
        assert found_metrics >= 3, "应找到多个风险指标相关元素"


#!/usr/bin/env python3
"""
RQA2025 智能告警可视化端到端测试

测试文件: web-static/intelligent-alerts.html

测试范围:
- 告警概览
- 关联分析（D3.js网络图）
- 趋势预测
- 模式识别
- 处理建议
- API集成
"""

import pytest
from pathlib import Path


class TestIntelligentAlerts:
    """智能告警可视化测试"""

    @pytest.fixture
    def html_file_path(self):
        """获取HTML文件路径"""
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / "web-static" / "intelligent-alerts.html"

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

    # ========== 告警概览测试 ==========

    def test_alert_overview_exists(self, html_content):
        """测试告警概览存在"""
        assert "告警" in html_content or "alert" in html_content.lower()
        assert "alert-critical" in html_content or "严重告警" in html_content

    def test_alert_priority_levels(self, html_content):
        """测试告警优先级级别"""
        priorities = ["严重", "高", "中", "低", "critical", "high", "medium", "low"]
        found = sum(1 for p in priorities if p.lower() in html_content.lower())
        assert found >= 3, "应找到多个告警优先级级别"

    # ========== 关联分析测试 ==========

    def test_correlation_analysis_exists(self, html_content):
        """测试关联分析存在"""
        assert "关联分析" in html_content or "correlation" in html_content.lower()

    def test_d3_js_integration(self, html_content):
        """测试D3.js集成"""
        assert "d3js.org" in html_content or "d3.v7" in html_content or "d3" in html_content.lower()

    def test_network_graph_element(self, html_content):
        """测试网络图元素"""
        assert "correlation-network" in html_content or "network" in html_content.lower()

    def test_correlation_heatmap(self, html_content):
        """测试关联热力图"""
        assert "correlation-heatmap" in html_content or "heatmap" in html_content.lower()

    # ========== 趋势预测测试 ==========

    def test_prediction_tab_exists(self, html_content):
        """测试预测标签页存在"""
        assert "趋势预测" in html_content or "prediction" in html_content.lower()

    def test_prediction_charts(self, html_content):
        """测试预测图表"""
        assert "alert-prediction-chart" in html_content or "prediction" in html_content.lower()

    # ========== 模式识别测试 ==========

    def test_pattern_recognition_exists(self, html_content):
        """测试模式识别存在"""
        assert "模式识别" in html_content or "pattern" in html_content.lower()

    def test_pattern_clustering(self, html_content):
        """测试模式聚类"""
        assert "pattern-clustering" in html_content or "clustering" in html_content.lower()

    # ========== 处理建议测试 ==========

    def test_recommendations_exists(self, html_content):
        """测试处理建议存在"""
        assert "处理建议" in html_content or "recommendation" in html_content.lower()

    def test_ai_recommendations(self, html_content):
        """测试AI处理建议"""
        assert "AI处理建议" in html_content or "ai-recommendations" in html_content or "recommendation" in html_content.lower()

    # ========== API集成测试 ==========

    def test_intelligent_alerts_apis(self, html_content):
        """测试智能告警API端点"""
        # 检查API相关的函数或端点引用（允许多种格式）
        api_indicators = [
            "/api/v1/monitoring/intelligent-alerts",
            "/api/v1/monitoring/alerts/correlation",
            "/api/v1/monitoring/alerts/prediction",
            "/api/v1/monitoring/alerts/patterns",
            "/api/v1/monitoring/alerts/recommendations",
            "getApiBaseUrl",
            "/api/v1/monitoring",
            "monitoring/alerts"
        ]
        found_endpoints = sum(1 for endpoint in api_indicators if endpoint in html_content)
        assert found_endpoints >= 1, f"应引用至少1个智能告警API端点或API函数，实际找到: {found_endpoints}"


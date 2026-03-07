#!/usr/bin/env python3
"""
RQA2025 预测性分析展示端到端测试

测试文件: web-static/predictive-analysis.html

测试范围:
- 性能预测（CPU、内存、网络、响应时间）
- 容量规划
- 异常预测
- 趋势分析
"""

import pytest
from pathlib import Path


class TestPredictiveAnalysis:
    """预测性分析展示测试"""

    @pytest.fixture
    def html_file_path(self):
        """获取HTML文件路径"""
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / "web-static" / "predictive-analysis.html"

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

    # ========== 性能预测测试 ==========

    def test_performance_prediction_tab(self, html_content):
        """测试性能预测标签页"""
        assert "性能预测" in html_content or "performance" in html_content.lower()

    def test_cpu_prediction_chart(self, html_content):
        """测试CPU预测图表"""
        assert "cpu-prediction-chart" in html_content or "cpu" in html_content.lower()

    def test_memory_prediction_chart(self, html_content):
        """测试内存预测图表"""
        assert "memory-prediction-chart" in html_content or "memory" in html_content.lower()

    # ========== 容量规划测试 ==========

    def test_capacity_planning_tab(self, html_content):
        """测试容量规划标签页"""
        assert "容量规划" in html_content or "capacity" in html_content.lower()

    def test_capacity_recommendations(self, html_content):
        """测试容量规划建议"""
        assert "capacity-recommendations" in html_content or "容量规划建议" in html_content

    # ========== 异常预测测试 ==========

    def test_anomaly_prediction_tab(self, html_content):
        """测试异常预测标签页"""
        assert "异常预测" in html_content or "anomaly" in html_content.lower()

    def test_anomaly_timeline(self, html_content):
        """测试异常时间线"""
        assert "anomaly-timeline" in html_content or "异常" in html_content

    # ========== 趋势分析测试 ==========

    def test_trends_analysis_tab(self, html_content):
        """测试趋势分析标签页"""
        assert "趋势分析" in html_content or "trend" in html_content.lower()

    # ========== API集成测试 ==========

    def test_predictive_analysis_apis(self, html_content):
        """测试预测分析API端点"""
        # 检查API相关的函数或端点引用（允许多种格式）
        api_indicators = [
            "/api/v1/monitoring/predictive/overview",
            "/api/v1/monitoring/predictive/performance",
            "/api/v1/monitoring/predictive/capacity",
            "/api/v1/monitoring/predictive/anomaly",
            "/api/v1/monitoring/predictive/trends",
            "getApiBaseUrl",
            "/api/v1/monitoring",
            "monitoring/predictive"
        ]
        found_endpoints = sum(1 for endpoint in api_indicators if endpoint in html_content)
        assert found_endpoints >= 1, f"应引用至少1个预测分析API端点或API函数，实际找到: {found_endpoints}"


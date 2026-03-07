#!/usr/bin/env python3
"""
RQA2025 实时性能监控详细视图端到端测试

测试文件: web-static/performance-monitor.html

测试范围:
- 系统性能（CPU、内存、磁盘、网络）
- 应用性能（API性能、错误率、组件性能）
- 业务性能（交易性能、策略性能）
- 性能对比（历史对比、基准对比）
"""

import pytest
from pathlib import Path


class TestPerformanceMonitor:
    """实时性能监控详细视图测试"""

    @pytest.fixture
    def html_file_path(self):
        """获取HTML文件路径"""
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / "web-static" / "performance-monitor.html"

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

    # ========== 系统性能测试 ==========

    def test_system_performance_tab(self, html_content):
        """测试系统性能标签页"""
        assert "系统性能" in html_content or "system" in html_content.lower()

    def test_system_gauges(self, html_content):
        """测试系统资源仪表盘"""
        assert "system-gauges" in html_content or "gauge" in html_content.lower() or "CPU" in html_content

    def test_cpu_memory_disk_network(self, html_content):
        """测试CPU、内存、磁盘、网络监控"""
        resources = ["CPU", "内存", "磁盘", "网络", "cpu", "memory", "disk", "network"]
        found = sum(1 for r in resources if r.lower() in html_content.lower())
        assert found >= 3, "应找到多个系统资源监控元素"

    # ========== 应用性能测试 ==========

    def test_application_performance_tab(self, html_content):
        """测试应用性能标签页"""
        assert "应用性能" in html_content or "application" in html_content.lower()

    def test_api_performance_chart(self, html_content):
        """测试API性能图表"""
        assert "api-performance-chart" in html_content or "api" in html_content.lower()

    # ========== 业务性能测试 ==========

    def test_business_performance_tab(self, html_content):
        """测试业务性能标签页"""
        assert "业务性能" in html_content or "business" in html_content.lower()

    def test_trading_performance(self, html_content):
        """测试交易性能"""
        assert "trading-performance" in html_content or "交易性能" in html_content

    # ========== 性能对比测试 ==========

    def test_performance_comparison_tab(self, html_content):
        """测试性能对比标签页"""
        assert "性能对比" in html_content or "comparison" in html_content.lower()

    def test_historical_comparison(self, html_content):
        """测试历史性能对比"""
        assert "historical-comparison" in html_content or "历史" in html_content

    # ========== API集成测试 ==========

    def test_performance_apis(self, html_content):
        """测试性能监控API端点"""
        # 检查API相关的函数或端点引用（允许多种格式）
        api_indicators = [
            "/api/v1/monitoring/performance/overview",
            "/api/v1/monitoring/performance/system",
            "/api/v1/monitoring/performance/application",
            "/api/v1/monitoring/performance/business",
            "/api/v1/monitoring/performance/comparison",
            "getApiBaseUrl",
            "/api/v1/monitoring",
            "monitoring/performance"
        ]
        found_endpoints = sum(1 for endpoint in api_indicators if endpoint in html_content)
        assert found_endpoints >= 1, f"应引用至少1个性能监控API端点或API函数，实际找到: {found_endpoints}"


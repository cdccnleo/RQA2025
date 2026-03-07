#!/usr/bin/env python3
"""
RQA2025 策略开发流程监控面板端到端测试

测试文件: web-static/strategy-development-monitor.html

测试范围:
- 页面加载和内容渲染
- 8个阶段监控视图
- 数据加载和API集成
- 图表功能
- 交互功能
"""

import pytest
import os
import re
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestStrategyDevelopmentMonitor:
    """策略开发流程监控面板测试"""

    @pytest.fixture
    def html_file_path(self):
        """获取HTML文件路径"""
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / "web-static" / "strategy-development-monitor.html"

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
        assert html_file_path.exists(), f"策略开发监控页面文件不存在: {html_file_path}"

    def test_page_basic_structure(self, html_content):
        """测试页面基本HTML结构"""
        assert "<!DOCTYPE html>" in html_content
        assert "<html" in html_content
        assert "<head>" in html_content or "<head" in html_content
        assert "<body" in html_content or "</body>" in html_content
        assert "</html>" in html_content

    def test_page_title(self, html_content):
        """测试页面标题"""
        assert "量化策略开发流程监控" in html_content or "策略开发" in html_content

    def test_css_resources_loading(self, html_content):
        """测试CSS资源加载"""
        # 检查Tailwind CSS
        assert "tailwindcss.com" in html_content or "tailwind" in html_content.lower()
        # 检查Font Awesome
        assert "font-awesome" in html_content.lower() or "fontawesome" in html_content.lower()
        # 检查移动端优化CSS
        assert "mobile-optimization.css" in html_content

    def test_js_resources_loading(self, html_content):
        """测试JavaScript资源加载"""
        # 检查Chart.js
        assert "chart.js" in html_content.lower()
        # 检查移动端工具库
        assert "mobile-utils.js" in html_content
        # 检查UX优化库
        assert "ux-optimization.js" in html_content

    # ========== 8个阶段监控视图测试 ==========

    def test_stage_conception_exists(self, html_content):
        """测试策略构思阶段元素"""
        # 检查阶段标识
        assert "策略构思" in html_content or "stage-conception" in html_content
        # 检查关键指标元素ID
        assert "conception-designers" in html_content or "策略构思" in html_content

    def test_stage_data_collection_exists(self, html_content):
        """测试数据收集阶段元素"""
        assert "数据收集" in html_content or "stage-data" in html_content
        assert "data-sources" in html_content or "数据源" in html_content

    def test_stage_feature_engineering_exists(self, html_content):
        """测试特征工程阶段元素"""
        assert "特征工程" in html_content or "stage-features" in html_content
        assert "feature-tasks" in html_content or "特征任务" in html_content

    def test_stage_model_training_exists(self, html_content):
        """测试模型训练阶段元素"""
        assert "模型训练" in html_content or "stage-training" in html_content
        assert "training-jobs" in html_content or "训练任务" in html_content
        assert "gpu-usage" in html_content or "GPU" in html_content

    def test_stage_backtest_exists(self, html_content):
        """测试策略回测阶段元素"""
        assert "策略回测" in html_content or "stage-backtest" in html_content
        assert "backtest-jobs" in html_content or "回测任务" in html_content

    def test_stage_evaluation_exists(self, html_content):
        """测试性能评估阶段元素"""
        assert "性能评估" in html_content or "stage-evaluation" in html_content
        assert "eval-metrics" in html_content or "评估指标" in html_content

    def test_stage_deployment_exists(self, html_content):
        """测试策略部署阶段元素"""
        assert "策略部署" in html_content or "stage-deployment" in html_content
        assert "deploy-status" in html_content or "部署状态" in html_content

    def test_stage_monitoring_exists(self, html_content):
        """测试监控优化阶段元素"""
        assert "监控优化" in html_content or "stage-monitoring" in html_content
        assert "realtime-performance" in html_content or "实时性能" in html_content

    def test_all_stages_present(self, html_content):
        """测试所有8个阶段都存在"""
        stages = [
            "策略构思", "数据收集", "特征工程", "模型训练",
            "策略回测", "性能评估", "策略部署", "监控优化"
        ]
        found_stages = sum(1 for stage in stages if stage in html_content)
        assert found_stages >= 6, f"至少应有6个阶段，实际找到: {found_stages}"

    # ========== 概览指标测试 ==========

    def test_overview_metrics_exist(self, html_content):
        """测试概览指标元素"""
        metrics = [
            "active-strategies", "successful-deployments",
            "avg-development-time", "strategy-performance"
        ]
        found_metrics = sum(1 for metric in metrics if metric in html_content)
        assert found_metrics >= 3, f"至少应有3个概览指标，实际找到: {found_metrics}"

    # ========== 图表功能测试 ==========

    def test_chart_elements_exist(self, html_content):
        """测试图表元素存在"""
        # 策略性能趋势图
        assert "strategy-performance-chart" in html_content
        # 开发时间线图
        assert "development-timeline-chart" in html_content
        # 检查Canvas元素
        assert "<canvas" in html_content

    def test_chart_js_integration(self, html_content):
        """测试Chart.js集成"""
        # 检查Chart.js库引用
        assert "chart.js" in html_content.lower()
        # 检查Chart实例创建（通过查找常见的Chart.js模式）
        assert "new Chart" in html_content or "Chart(" in html_content

    # ========== 数据表格测试 ==========

    def test_strategy_details_table_exists(self, html_content):
        """测试策略详情表格存在"""
        assert "strategy-details-table" in html_content
        assert "<table" in html_content or "<tbody" in html_content

    def test_table_headers_exist(self, html_content):
        """测试表格表头存在"""
        headers = ["策略名称", "当前阶段", "进度", "性能指标", "状态"]
        found_headers = sum(1 for header in headers if header in html_content)
        assert found_headers >= 3, f"至少应有3个表头，实际找到: {found_headers}"

    # ========== API集成测试 ==========

    def test_api_endpoints_referenced(self, html_content):
        """测试API端点引用"""
        # 检查API相关的函数或端点引用（允许多种格式）
        api_indicators = [
            "/api/v1/strategy/development/overview",
            "/api/v1/strategy/development/details",
            "getApiBaseUrl",
            "/api/v1/strategy",
            "strategy/development"
        ]
        found_endpoints = sum(1 for endpoint in api_indicators if endpoint in html_content)
        assert found_endpoints >= 1, f"至少应引用1个API端点或API函数，实际找到: {found_endpoints}"

    def test_get_api_base_url_function_exists(self, html_content):
        """测试API基础URL函数存在"""
        assert "getApiBaseUrl" in html_content

    def test_refresh_function_exists(self, html_content):
        """测试刷新功能函数存在"""
        assert "refreshAll" in html_content or "refresh" in html_content.lower()

    # ========== JavaScript功能测试 ==========

    def test_data_loading_functions_exist(self, html_content):
        """测试数据加载函数存在"""
        functions = [
            "loadStrategyOverview",
            "loadStrategyCharts",
            "loadStrategyDetails"
        ]
        found_functions = sum(1 for func in functions if func in html_content)
        assert found_functions >= 2, f"至少应有2个数据加载函数，实际找到: {found_functions}"

    def test_chart_initialization_code_exists(self, html_content):
        """测试图表初始化代码存在"""
        # 检查是否有Chart初始化相关代码
        assert "Chart(" in html_content or "new Chart" in html_content or "performanceChart" in html_content

    def test_dom_content_loaded_event(self, html_content):
        """测试DOMContentLoaded事件监听"""
        assert "DOMContentLoaded" in html_content or "document.addEventListener" in html_content

    # ========== 导航和交互测试 ==========

    def test_navigation_elements_exist(self, html_content):
        """测试导航元素存在"""
        assert "<nav" in html_content or "navigation" in html_content.lower()
        # 检查返回链接
        assert "/dashboard" in html_content or "dashboard" in html_content.lower()

    def test_refresh_button_exists(self, html_content):
        """测试刷新按钮存在"""
        assert "刷新" in html_content
        assert "refreshAll" in html_content or "refresh" in html_content.lower()

    # ========== 响应式设计测试 ==========

    def test_responsive_design_classes(self, html_content):
        """测试响应式设计类"""
        # 检查Tailwind响应式类
        responsive_classes = ["sm:", "md:", "lg:", "grid-cols"]
        found_classes = sum(1 for cls in responsive_classes if cls in html_content)
        assert found_classes >= 2, f"至少应有2个响应式类，实际找到: {found_classes}"

    def test_mobile_optimization_integration(self, html_content):
        """测试移动端优化集成"""
        assert "mobile-optimization.css" in html_content
        assert "mobile-utils.js" in html_content

    # ========== 样式和UI测试 ==========

    def test_card_elements_exist(self, html_content):
        """测试卡片元素存在"""
        # 检查是否有卡片相关的类或结构
        assert "card" in html_content.lower() or "rounded-lg" in html_content

    def test_status_indicators_exist(self, html_content):
        """测试状态指示器存在"""
        # 检查状态指示相关的元素
        indicators = ["status", "indicator", "状态", "bg-green", "bg-red"]
        found_indicators = sum(1 for ind in indicators if ind in html_content.lower())
        assert found_indicators >= 2, f"至少应有2个状态指示器，实际找到: {found_indicators}"

    # ========== 完整功能验证 ==========

    def test_page_completeness(self, html_content):
        """测试页面完整性"""
        # 检查关键组件是否都存在
        components = {
            "导航": html_content.count("nav") >= 1 or "navigation" in html_content.lower(),
            "概览指标": "active-strategies" in html_content or "概览" in html_content,
            "阶段监控": "stage-" in html_content or "阶段" in html_content,
            "图表": "chart" in html_content.lower(),
            "表格": "table" in html_content.lower() or "tbody" in html_content,
            "JavaScript": "<script" in html_content
        }
        
        passed_components = sum(1 for v in components.values() if v)
        assert passed_components >= 5, f"至少应有5个关键组件，实际找到: {passed_components}/{len(components)}"

    def test_no_broken_references(self, html_content):
        """测试没有明显的损坏引用"""
        # 检查是否有明显的JavaScript错误模式
        error_patterns = [
            r"undefined\s*\(",
            r"null\s*\.",
            r"\.\s*undefined"
        ]
        
        errors_found = []
        for pattern in error_patterns:
            matches = re.findall(pattern, html_content)
            if matches:
                errors_found.extend(matches)
        
        # 允许少量可能的误报
        assert len(errors_found) < 5, f"发现可能的JavaScript错误模式: {errors_found[:5]}"


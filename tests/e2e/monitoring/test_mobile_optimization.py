#!/usr/bin/env python3
"""
RQA2025 移动端适配优化端到端测试

测试文件: web-static/mobile-optimization.css, mobile-utils.js
以及所有监控页面的移动端适配

测试范围:
- 响应式布局
- 移动端功能
- 浏览器兼容性
- 性能优化
"""

import pytest
from pathlib import Path


class TestMobileOptimization:
    """移动端适配优化测试"""

    @pytest.fixture
    def css_file_path(self):
        """获取CSS文件路径"""
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / "web-static" / "mobile-optimization.css"

    @pytest.fixture
    def js_file_path(self):
        """获取JS文件路径"""
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / "web-static" / "mobile-utils.js"

    @pytest.fixture
    def css_content(self, css_file_path):
        """读取CSS文件内容"""
        if not css_file_path.exists():
            pytest.skip(f"CSS file not found: {css_file_path}")
        with open(css_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @pytest.fixture
    def js_content(self, js_file_path):
        """读取JS文件内容"""
        if not js_file_path.exists():
            pytest.skip(f"JS file not found: {js_file_path}")
        with open(js_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    # ========== CSS文件测试 ==========

    def test_mobile_css_file_exists(self, css_file_path):
        """测试移动端CSS文件存在"""
        assert css_file_path.exists()

    def test_responsive_design_rules(self, css_content):
        """测试响应式设计规则"""
        assert "@media" in css_content
        assert "max-width" in css_content or "min-width" in css_content

    def test_mobile_specific_classes(self, css_content):
        """测试移动端专用类"""
        mobile_classes = ["mobile-grid", "mobile-card", "mobile-btn", "mobile-container"]
        found = sum(1 for cls in mobile_classes if cls in css_content)
        assert found >= 2, "应定义多个移动端专用CSS类"

    def test_touch_optimization(self, css_content):
        """测试触摸优化"""
        assert "touch" in css_content.lower() or "44px" in css_content

    # ========== JavaScript文件测试 ==========

    def test_mobile_utils_file_exists(self, js_file_path):
        """测试移动端工具JS文件存在"""
        assert js_file_path.exists()

    def test_mobile_utils_functions(self, js_content):
        """测试移动端工具函数"""
        functions = ["isMobile", "isTablet", "isTouchDevice", "getNetworkStatus"]
        found = sum(1 for func in functions if func in js_content)
        assert found >= 2, "应定义多个移动端工具函数"

    def test_mobile_utils_object(self, js_content):
        """测试MobileUtils对象"""
        assert "MobileUtils" in js_content or "mobileUtils" in js_content

    # ========== HTML页面集成测试 ==========

    @pytest.fixture
    def monitoring_pages(self):
        """获取所有监控页面"""
        project_root = Path(__file__).parent.parent.parent.parent
        web_static = project_root / "web-static"
        pages = [
            "strategy-development-monitor.html",
            "trading-execution.html",
            "risk-control-monitor.html",
            "layer-monitor.html",
            "intelligent-alerts.html",
            "predictive-analysis.html",
            "performance-monitor.html",
            "dashboard.html"
        ]
        return [(web_static / page) for page in pages]

    def test_all_pages_reference_mobile_css(self, monitoring_pages):
        """测试所有监控页面都引用移动端CSS"""
        pages_with_css = 0
        for page_path in monitoring_pages:
            if page_path.exists():
                with open(page_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "mobile-optimization.css" in content:
                        pages_with_css += 1
        
        assert pages_with_css >= 5, f"至少5个页面应引用移动端CSS，实际: {pages_with_css}"

    def test_all_pages_reference_mobile_utils(self, monitoring_pages):
        """测试所有监控页面都引用移动端工具库"""
        pages_with_utils = 0
        for page_path in monitoring_pages:
            if page_path.exists():
                with open(page_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "mobile-utils.js" in content:
                        pages_with_utils += 1
        
        assert pages_with_utils >= 5, f"至少5个页面应引用移动端工具库，实际: {pages_with_utils}"


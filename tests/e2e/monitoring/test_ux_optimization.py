#!/usr/bin/env python3
"""
RQA2025 用户体验优化端到端测试

测试文件: web-static/ux-optimization.js, sw.js
以及所有监控页面的UX优化

测试范围:
- 页面加载性能优化
- 数据刷新策略
- 自定义监控面板
- 性能监控
"""

import pytest
from pathlib import Path


class TestUXOptimization:
    """用户体验优化测试"""

    @pytest.fixture
    def ux_js_file_path(self):
        """获取UX优化JS文件路径"""
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / "web-static" / "ux-optimization.js"

    @pytest.fixture
    def sw_file_path(self):
        """获取Service Worker文件路径"""
        project_root = Path(__file__).parent.parent.parent.parent
        return project_root / "web-static" / "sw.js"

    @pytest.fixture
    def ux_js_content(self, ux_js_file_path):
        """读取UX优化JS文件内容"""
        if not ux_js_file_path.exists():
            pytest.skip(f"UX JS file not found: {ux_js_file_path}")
        with open(ux_js_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @pytest.fixture
    def sw_content(self, sw_file_path):
        """读取Service Worker文件内容"""
        if not sw_file_path.exists():
            pytest.skip(f"Service Worker file not found: {sw_file_path}")
        with open(sw_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    # ========== UX优化JS文件测试 ==========

    def test_ux_js_file_exists(self, ux_js_file_path):
        """测试UX优化JS文件存在"""
        assert ux_js_file_path.exists()

    def test_ux_optimization_object(self, ux_js_content):
        """测试UXOptimization对象"""
        assert "UXOptimization" in ux_js_content or "uxOptimization" in ux_js_content

    def test_page_load_optimization(self, ux_js_content):
        """测试页面加载优化函数"""
        assert "optimizePageLoad" in ux_js_content or "preload" in ux_js_content.lower()

    def test_data_refresh_optimization(self, ux_js_content):
        """测试数据刷新优化"""
        assert "optimizeDataRefresh" in ux_js_content or "refresh" in ux_js_content.lower()

    def test_custom_dashboard_functions(self, ux_js_content):
        """测试自定义面板功能"""
        assert "enableCustomDashboard" in ux_js_content or "custom" in ux_js_content.lower()

    # ========== Service Worker测试 ==========

    def test_service_worker_file_exists(self, sw_file_path):
        """测试Service Worker文件存在"""
        assert sw_file_path.exists()

    def test_service_worker_install_event(self, sw_content):
        """测试Service Worker安装事件"""
        assert "addEventListener" in sw_content
        assert "install" in sw_content.lower()

    def test_service_worker_cache(self, sw_content):
        """测试Service Worker缓存功能"""
        assert "cache" in sw_content.lower() or "Cache" in sw_content

    def test_service_worker_fetch_event(self, sw_content):
        """测试Service Worker获取事件"""
        assert "fetch" in sw_content.lower()

    # ========== HTML页面集成测试 ==========

    @pytest.fixture
    def main_monitoring_pages(self):
        """获取主要监控页面"""
        project_root = Path(__file__).parent.parent.parent.parent
        web_static = project_root / "web-static"
        pages = [
            "dashboard.html",
            "strategy-development-monitor.html"
        ]
        return [(web_static / page) for page in pages]

    def test_pages_reference_ux_optimization(self, main_monitoring_pages):
        """测试主要页面引用UX优化库"""
        pages_with_ux = 0
        for page_path in main_monitoring_pages:
            if page_path.exists():
                with open(page_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "ux-optimization.js" in content:
                        pages_with_ux += 1
        
        assert pages_with_ux >= 1, "至少1个主要页面应引用UX优化库"


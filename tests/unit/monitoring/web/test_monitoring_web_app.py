#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控Web应用测试
测试monitoring_web_app.py的所有路由和方法
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# 处理依赖问题
try:
    from src.monitoring.web.monitoring_web_app import (
        MonitoringWebApp,
        get_web_app,
        start_web_app,
        stop_web_app,
        _web_app_instance
    )
except ImportError:
    pytest.skip("flask_cors模块不可用，跳过web应用测试", allow_module_level=True)


class TestMonitoringWebAppInitialization:
    """测试MonitoringWebApp初始化"""

    def test_init_default_values(self):
        """测试使用默认值初始化"""
        with patch('src.monitoring.web.monitoring_web_app.get_monitor') as mock_get_monitor:
            mock_monitor = MagicMock()
            mock_get_monitor.return_value = mock_monitor
            
            with patch('src.monitoring.web.monitoring_web_app.CORS'):
                app = MonitoringWebApp()
                
                assert app.host == '0.0.0.0'
                assert app.port == 5000
                assert app.monitor == mock_monitor

    def test_init_custom_values(self):
        """测试使用自定义值初始化"""
        with patch('src.monitoring.web.monitoring_web_app.get_monitor') as mock_get_monitor:
            mock_monitor = MagicMock()
            mock_get_monitor.return_value = mock_monitor
            
            with patch('src.monitoring.web.monitoring_web_app.CORS'):
                app = MonitoringWebApp(host='127.0.0.1', port=8080)
                
                assert app.host == '127.0.0.1'
                assert app.port == 8080

    def test_init_sets_up_logging(self):
        """测试初始化设置日志"""
        with patch('src.monitoring.web.monitoring_web_app.get_monitor'):
            with patch('src.monitoring.web.monitoring_web_app.CORS'):
                with patch('logging.basicConfig') as mock_logging:
                    MonitoringWebApp()
                    mock_logging.assert_called_once()

    def test_init_registers_routes(self):
        """测试初始化注册路由"""
        with patch('src.monitoring.web.monitoring_web_app.get_monitor') as mock_get_monitor:
            mock_monitor = MagicMock()
            mock_get_monitor.return_value = mock_monitor
            
            with patch('src.monitoring.web.monitoring_web_app.CORS'):
                app = MonitoringWebApp()
                
                # 验证路由被注册
                assert hasattr(app, 'app')
                # 检查是否有路由规则
                assert len(app.app.url_map._rules) > 0


class TestMonitoringWebAppRoutes:
    """测试MonitoringWebApp路由"""

    @pytest.fixture
    def web_app(self):
        """创建Web应用实例"""
        with patch('src.monitoring.web.monitoring_web_app.get_monitor') as mock_get_monitor:
            mock_monitor = MagicMock()
            mock_get_monitor.return_value = mock_monitor
            
            with patch('src.monitoring.web.monitoring_web_app.CORS'):
                app = MonitoringWebApp()
                app.monitor = mock_monitor
                app.app.config['TESTING'] = True
                return app

    def test_index_route(self, web_app):
        """测试主页路由"""
        with patch('src.monitoring.web.monitoring_web_app.render_template') as mock_render:
            mock_render.return_value = '<html>Dashboard</html>'
            
            with web_app.app.test_client() as client:
                response = client.get('/')
                assert response.status_code == 200
                mock_render.assert_called_once_with('monitoring_dashboard.html')

    def test_get_metrics_route_success(self, web_app):
        """测试获取指标API成功"""
        from dataclasses import dataclass
        
        @dataclass
        class MockMetric:
            value: float
            timestamp: datetime
        
        mock_metrics = {
            'cpu_percent': MockMetric(75.0, datetime.now()),
            'memory_percent': MockMetric(60.0, datetime.now())
        }
        
        web_app.monitor.get_current_metrics.return_value = mock_metrics
        web_app.monitor.get_alerts_summary.return_value = {'total': 0}
        web_app.monitor.get_system_status.return_value = {'system_health': 'healthy'}
        
        with web_app.app.test_client() as client:
            response = client.get('/api/monitoring/metrics')
            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True
            assert 'metrics' in data
            assert 'alerts' in data
            assert 'system_status' in data
            assert 'timestamp' in data

    def test_get_metrics_route_error(self, web_app):
        """测试获取指标API错误处理"""
        web_app.monitor.get_current_metrics.side_effect = Exception("Test error")
        
        with web_app.app.test_client() as client:
            response = client.get('/api/monitoring/metrics')
            assert response.status_code == 500
            data = response.get_json()
            assert data['success'] is False
            assert 'error' in data

    def test_get_status_route_success(self, web_app):
        """测试获取状态API成功"""
        web_app.monitor.get_system_status.return_value = {'system_health': 'healthy'}
        
        with web_app.app.test_client() as client:
            response = client.get('/api/monitoring/status')
            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True
            assert 'status' in data
            assert 'timestamp' in data

    def test_get_status_route_error(self, web_app):
        """测试获取状态API错误处理"""
        web_app.monitor.get_system_status.side_effect = Exception("Test error")
        
        with web_app.app.test_client() as client:
            response = client.get('/api/monitoring/status')
            assert response.status_code == 500
            data = response.get_json()
            assert data['success'] is False

    def test_get_alerts_route_success(self, web_app):
        """测试获取告警API成功"""
        web_app.monitor.get_alerts_summary.return_value = {'total': 2, 'critical': 1}
        
        with web_app.app.test_client() as client:
            response = client.get('/api/monitoring/alerts')
            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True
            assert 'alerts' in data

    def test_get_alerts_route_error(self, web_app):
        """测试获取告警API错误处理"""
        web_app.monitor.get_alerts_summary.side_effect = Exception("Test error")
        
        with web_app.app.test_client() as client:
            response = client.get('/api/monitoring/alerts')
            assert response.status_code == 500
            data = response.get_json()
            assert data['success'] is False

    def test_get_history_route_success(self, web_app):
        """测试获取历史指标API成功"""
        from dataclasses import dataclass
        
        @dataclass
        class MockMetric:
            value: float
            timestamp: datetime
            name: str
        
        mock_metrics = {
            'cpu_percent': MockMetric(75.0, datetime.now(), 'cpu_percent')
        }
        
        web_app.monitor.get_current_metrics.return_value = mock_metrics
        
        with web_app.app.test_client() as client:
            response = client.get('/api/monitoring/history?hours=1')
            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True
            assert 'history' in data
            assert data['hours'] == 1

    def test_get_history_route_default_hours(self, web_app):
        """测试获取历史指标API默认hours"""
        web_app.monitor.get_current_metrics.return_value = {}
        
        with web_app.app.test_client() as client:
            response = client.get('/api/monitoring/history')
            assert response.status_code == 200
            data = response.get_json()
            assert data['hours'] == 1

    def test_get_history_route_error(self, web_app):
        """测试获取历史指标API错误处理"""
        web_app.monitor.get_current_metrics.side_effect = Exception("Test error")
        
        with web_app.app.test_client() as client:
            response = client.get('/api/monitoring/history')
            assert response.status_code == 500
            data = response.get_json()
            assert data['success'] is False

    def test_update_metric_route_success(self, web_app):
        """测试更新指标API成功"""
        with web_app.app.test_client() as client:
            response = client.post(
                '/api/monitoring/update',
                json={'name': 'test_metric', 'value': 100.0},
                content_type='application/json'
            )
            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True
            web_app.monitor.update_business_metric.assert_called_once_with('test_metric', 100.0)

    def test_update_metric_route_no_data(self, web_app):
        """测试更新指标API无数据"""
        with web_app.app.test_client() as client:
            response = client.post(
                '/api/monitoring/update',
                content_type='application/json'
            )
            assert response.status_code == 400
            data = response.get_json()
            assert data['success'] is False

    def test_update_metric_route_missing_fields(self, web_app):
        """测试更新指标API缺少字段"""
        with web_app.app.test_client() as client:
            response = client.post(
                '/api/monitoring/update',
                json={'name': 'test_metric'},
                content_type='application/json'
            )
            assert response.status_code == 400
            data = response.get_json()
            assert data['success'] is False

    def test_update_metric_route_error(self, web_app):
        """测试更新指标API错误处理"""
        web_app.monitor.update_business_metric.side_effect = Exception("Test error")
        
        with web_app.app.test_client() as client:
            response = client.post(
                '/api/monitoring/update',
                json={'name': 'test_metric', 'value': 100.0},
                content_type='application/json'
            )
            assert response.status_code == 500
            data = response.get_json()
            assert data['success'] is False

    def test_health_check_route_healthy(self, web_app):
        """测试健康检查端点健康状态"""
        web_app.monitor.get_system_status.return_value = {'system_health': 'healthy'}
        
        with web_app.app.test_client() as client:
            response = client.get('/health')
            assert response.status_code == 200
            data = response.get_json()
            assert data['status'] == 'healthy'

    def test_health_check_route_unhealthy(self, web_app):
        """测试健康检查端点不健康状态"""
        web_app.monitor.get_system_status.return_value = {'system_health': 'unhealthy'}
        
        with web_app.app.test_client() as client:
            response = client.get('/health')
            assert response.status_code == 200
            data = response.get_json()
            assert data['status'] == 'unhealthy'

    def test_health_check_route_error(self, web_app):
        """测试健康检查端点错误处理"""
        web_app.monitor.get_system_status.side_effect = Exception("Test error")
        
        with web_app.app.test_client() as client:
            response = client.get('/health')
            assert response.status_code == 500
            data = response.get_json()
            assert data['status'] == 'unhealthy'
            assert 'error' in data


class TestMonitoringWebAppMethods:
    """测试MonitoringWebApp方法"""

    def test_start_method(self):
        """测试启动Web应用方法"""
        with patch('src.monitoring.web.monitoring_web_app.get_monitor') as mock_get_monitor:
            mock_monitor = MagicMock()
            mock_get_monitor.return_value = mock_monitor
            
            with patch('src.monitoring.web.monitoring_web_app.CORS'):
                app = MonitoringWebApp()
                
                with patch.object(app.app, 'run') as mock_run:
                    mock_run.side_effect = KeyboardInterrupt()
                    
                    app.start()
                    
                    mock_monitor.start_monitoring.assert_called_once()
                    mock_monitor.stop_monitoring.assert_called()

    def test_stop_method(self):
        """测试停止Web应用方法"""
        with patch('src.monitoring.web.monitoring_web_app.get_monitor') as mock_get_monitor:
            mock_monitor = MagicMock()
            mock_get_monitor.return_value = mock_monitor
            
            with patch('src.monitoring.web.monitoring_web_app.CORS'):
                app = MonitoringWebApp()
                app.stop()
                
                mock_monitor.stop_monitoring.assert_called_once()


class TestWebAppGlobalFunctions:
    """测试Web应用全局函数"""

    def test_get_web_app_first_call(self):
        """测试获取Web应用首次调用"""
        # 重置全局实例
        import src.monitoring.web.monitoring_web_app as web_app_module
        web_app_module._web_app_instance = None
        
        with patch('src.monitoring.web.monitoring_web_app.MonitoringWebApp') as mock_app_class:
            mock_app = MagicMock()
            mock_app_class.return_value = mock_app
            
            result = get_web_app()
            
            assert result == mock_app
            mock_app_class.assert_called_once_with('0.0.0.0', 5000)

    def test_get_web_app_subsequent_calls(self):
        """测试获取Web应用后续调用"""
        # 重置全局实例
        import src.monitoring.web.monitoring_web_app as web_app_module
        existing_app = MagicMock()
        web_app_module._web_app_instance = existing_app
        
        result = get_web_app()
        
        assert result == existing_app

    def test_start_web_app(self):
        """测试启动Web应用函数"""
        # 重置全局实例
        import src.monitoring.web.monitoring_web_app as web_app_module
        web_app_module._web_app_instance = None
        
        with patch('src.monitoring.web.monitoring_web_app.get_web_app') as mock_get:
            mock_app = MagicMock()
            mock_get.return_value = mock_app
            
            with patch('src.monitoring.web.monitoring_web_app.MonitoringWebApp'):
                start_web_app()
                
                mock_app.start.assert_called_once()

    def test_stop_web_app_with_instance(self):
        """测试停止Web应用函数（有实例）"""
        # 设置全局实例
        import src.monitoring.web.monitoring_web_app as web_app_module
        mock_app = MagicMock()
        web_app_module._web_app_instance = mock_app
        
        stop_web_app()
        
        mock_app.stop.assert_called_once()
        assert web_app_module._web_app_instance is None

    def test_stop_web_app_without_instance(self):
        """测试停止Web应用函数（无实例）"""
        # 重置全局实例
        import src.monitoring.web.monitoring_web_app as web_app_module
        web_app_module._web_app_instance = None
        
        # 应该不会抛出异常
        stop_web_app()
        
        assert web_app_module._web_app_instance is None




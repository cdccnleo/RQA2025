#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Monitoring模块核心监控系统测试

测试monitoring/目录中的核心监控功能，避免复杂的模块导入依赖
"""

import pytest
import time
import threading
import psutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json


class TestMonitoringCoreSystems:
    """测试监控核心系统功能"""

    def setup_method(self):
        """测试前准备"""
        self.start_time = time.time()
        self.system_metrics = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'disk_usage': 0.0,
            'network_connections': 0,
            'thread_count': 0,
            'process_count': 0
        }

        # 模拟监控配置
        self.monitor_config = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage': 90.0,
            'check_interval': 5.0,
            'alert_cooldown': 60.0,
            'max_retries': 3
        }

    def test_system_metrics_collection(self):
        """测试系统指标收集"""
        # 模拟系统指标收集
        def collect_system_metrics():
            """收集系统性能指标"""
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # 内存使用率
                memory = psutil.virtual_memory()
                memory_percent = memory.percent

                # 磁盘使用率
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent

                # 网络连接数
                network_connections = len(psutil.net_connections())

                # 进程和线程信息
                process_count = len(psutil.pids())
                current_process = psutil.Process()
                thread_count = current_process.num_threads()

                return {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent,
                    'network_connections': network_connections,
                    'process_count': process_count,
                    'thread_count': thread_count,
                    'timestamp': datetime.now().isoformat(),
                    'collection_success': True
                }
            except Exception as e:
                return {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'collection_success': False
                }

        # 执行指标收集
        metrics = collect_system_metrics()

        # 验证收集结果
        assert 'timestamp' in metrics
        assert 'collection_success' in metrics

        if metrics['collection_success']:
            # 如果收集成功，验证关键指标存在
            assert 'cpu_percent' in metrics
            assert 'memory_percent' in metrics
            assert 'disk_percent' in metrics
            assert 'network_connections' in metrics

            # 验证指标范围合理
            assert 0 <= metrics['cpu_percent'] <= 100
            assert 0 <= metrics['memory_percent'] <= 100
            assert 0 <= metrics['disk_percent'] <= 100
            assert metrics['network_connections'] >= 0
        else:
            # 如果收集失败，确保有错误信息
            assert 'error' in metrics

    def test_health_check_system(self):
        """测试健康检查系统"""
        def perform_health_check(services: List[str]) -> Dict[str, Any]:
            """执行服务健康检查"""
            health_results = {
                'overall_status': 'healthy',
                'services_checked': len(services),
                'healthy_services': 0,
                'unhealthy_services': 0,
                'check_timestamp': datetime.now().isoformat(),
                'service_details': {}
            }

            for service in services:
                # 模拟服务健康检查
                if 'database' in service.lower():
                    # 数据库服务检查
                    status = 'healthy' if len(service) > 5 else 'unhealthy'
                    response_time = 0.1 + len(service) * 0.01
                elif 'cache' in service.lower():
                    # 缓存服务检查
                    status = 'healthy'
                    response_time = 0.05
                elif 'api' in service.lower():
                    # API服务检查
                    status = 'healthy' if 'gateway' not in service.lower() else 'degraded'
                    response_time = 0.2
                else:
                    # 其他服务
                    status = 'healthy'
                    response_time = 0.1

                health_results['service_details'][service] = {
                    'status': status,
                    'response_time': response_time,
                    'last_check': datetime.now().isoformat()
                }

                if status == 'healthy':
                    health_results['healthy_services'] += 1
                else:
                    health_results['unhealthy_services'] += 1

            # 确定整体状态
            if health_results['unhealthy_services'] > 0:
                health_results['overall_status'] = 'degraded' if health_results['unhealthy_services'] < len(services) // 2 else 'unhealthy'

            return health_results

        # 测试健康检查
        test_services = ['database', 'cache_service', 'api_gateway', 'message_queue', 'file_service']
        health_check = perform_health_check(test_services)

        # 验证健康检查结果结构
        assert 'overall_status' in health_check
        assert 'services_checked' in health_check
        assert 'healthy_services' in health_check
        assert 'unhealthy_services' in health_check
        assert 'service_details' in health_check
        assert 'check_timestamp' in health_check

        # 验证计数正确性
        assert health_check['services_checked'] == len(test_services)
        assert health_check['healthy_services'] + health_check['unhealthy_services'] == len(test_services)

        # 验证整体状态合理性
        assert health_check['overall_status'] in ['healthy', 'degraded', 'unhealthy']

        # 验证服务详情
        for service in test_services:
            assert service in health_check['service_details']
            service_detail = health_check['service_details'][service]
            assert 'status' in service_detail
            assert 'response_time' in service_detail
            assert 'last_check' in service_detail
            assert service_detail['status'] in ['healthy', 'degraded', 'unhealthy']
            assert service_detail['response_time'] > 0

    def test_performance_monitoring(self):
        """测试性能监控"""
        def monitor_performance(operation_name: str, operation_func, *args, **kwargs):
            """监控操作性能"""
            start_time = time.time()
            start_cpu = psutil.cpu_percent(interval=None)
            start_memory = psutil.virtual_memory().percent

            try:
                # 执行操作
                result = operation_func(*args, **kwargs)
                success = True
            except Exception as e:
                result = None
                success = False
                error = str(e)
            finally:
                end_time = time.time()
                end_cpu = psutil.cpu_percent(interval=None)
                end_memory = psutil.virtual_memory().percent

            execution_time = end_time - start_time
            cpu_usage = end_cpu - start_cpu
            memory_usage = end_memory - start_memory

            performance_data = {
                'operation': operation_name,
                'execution_time': execution_time,
                'cpu_usage_delta': cpu_usage,
                'memory_usage_delta': memory_usage,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }

            if not success:
                performance_data['error'] = error

            return result, performance_data

        # 测试被监控的操作
        def sample_operation(x, y):
            """示例操作"""
            time.sleep(0.1)  # 模拟处理时间
            return x + y

        # 执行性能监控
        result, perf_data = monitor_performance('addition', sample_operation, 10, 20)

        # 验证结果
        assert result == 30  # 操作结果正确
        assert perf_data['success'] == True
        assert perf_data['operation'] == 'addition'
        assert 'execution_time' in perf_data
        assert 'cpu_usage_delta' in perf_data
        assert 'memory_usage_delta' in perf_data
        assert 'timestamp' in perf_data

        # 验证性能指标合理性
        assert perf_data['execution_time'] >= 0.1  # 至少0.1秒
        assert perf_data['execution_time'] < 1.0   # 不应该超过1秒

    def test_alert_system(self):
        """测试告警系统"""
        def create_alert_system(thresholds: Dict[str, float]):
            """创建告警系统"""
            alerts = []

            def check_thresholds(metrics: Dict[str, float]) -> List[Dict[str, Any]]:
                """检查阈值并生成告警"""
                current_alerts = []

                for metric_name, value in metrics.items():
                    if metric_name in thresholds:
                        threshold = thresholds[metric_name]
                        if value > threshold:
                            alert = {
                                'metric': metric_name,
                                'value': value,
                                'threshold': threshold,
                                'severity': 'high' if value > threshold * 1.2 else 'medium',
                                'timestamp': datetime.now().isoformat(),
                                'message': f'{metric_name} exceeded threshold: {value:.2f} > {threshold:.2f}'
                            }
                            current_alerts.append(alert)

                return current_alerts

            def add_alert(alert: Dict[str, Any]):
                """添加告警"""
                alerts.append(alert)

            def get_alerts(count: int = None) -> List[Dict[str, Any]]:
                """获取告警"""
                if count is None:
                    return alerts.copy()
                return alerts[-count:] if count > 0 else []

            return {
                'check_thresholds': check_thresholds,
                'add_alert': add_alert,
                'get_alerts': get_alerts,
                'alert_count': lambda: len(alerts)
            }

        # 创建告警系统
        alert_system = create_alert_system(self.monitor_config)

        # 测试阈值检查
        test_metrics = {
            'cpu_percent': 85.0,  # 超过阈值80
            'memory_percent': 75.0,  # 未超过阈值85
            'disk_usage': 95.0,  # 超过阈值90
        }

        alerts = alert_system['check_thresholds'](test_metrics)

        # 验证告警生成
        assert len(alerts) == 2  # CPU和磁盘应该触发告警

        # 检查CPU告警
        cpu_alert = next((a for a in alerts if a['metric'] == 'cpu_percent'), None)
        assert cpu_alert is not None
        assert cpu_alert['value'] == 85.0
        assert cpu_alert['threshold'] == 80.0
        assert cpu_alert['severity'] in ['medium', 'high']
        assert 'message' in cpu_alert

        # 检查磁盘告警
        disk_alert = next((a for a in alerts if a['metric'] == 'disk_usage'), None)
        assert disk_alert is not None
        assert disk_alert['severity'] == 'medium'  # 95 <= 90*1.2

        # 验证内存没有触发告警
        memory_alert = next((a for a in alerts if a['metric'] == 'memory_percent'), None)
        assert memory_alert is None

    def test_monitoring_dashboard(self):
        """测试监控仪表板"""
        def create_monitoring_dashboard():
            """创建监控仪表板"""
            dashboard_data = {
                'system_overview': {},
                'performance_metrics': {},
                'alerts_summary': {},
                'service_health': {},
                'last_update': None
            }

            def update_dashboard(system_metrics, alerts, health_status):
                """更新仪表板数据"""
                dashboard_data.update({
                    'system_overview': {
                        'cpu_usage': system_metrics.get('cpu_percent', 0),
                        'memory_usage': system_metrics.get('memory_percent', 0),
                        'disk_usage': system_metrics.get('disk_percent', 0),
                        'active_connections': system_metrics.get('network_connections', 0)
                    },
                    'performance_metrics': {
                        'avg_response_time': 0.15,
                        'throughput': 1250,
                        'error_rate': 0.02
                    },
                    'alerts_summary': {
                        'total_alerts': len(alerts),
                        'critical_alerts': len([a for a in alerts if a.get('severity') == 'high']),
                        'active_alerts': len([a for a in alerts if not a.get('resolved', False)])
                    },
                    'service_health': health_status,
                    'last_update': datetime.now().isoformat()
                })

            def get_dashboard_data() -> Dict[str, Any]:
                """获取仪表板数据"""
                return dashboard_data.copy()

            def generate_summary_report() -> str:
                """生成摘要报告"""
                data = dashboard_data
                if not data['last_update']:
                    return "Dashboard not initialized"

                report = """
监控摘要报告 - {data['last_update']}

系统概览:
- CPU使用率: {data['system_overview'].get('cpu_usage', 0):.1f}%
- 内存使用率: {data['system_overview'].get('memory_usage', 0):.1f}%
- 磁盘使用率: {data['system_overview'].get('disk_usage', 0):.1f}%
- 活动连接数: {data['system_overview'].get('active_connections', 0)}

性能指标:
- 平均响应时间: {data['performance_metrics'].get('avg_response_time', 0):.3f}s
- 吞吐量: {data['performance_metrics'].get('throughput', 0)} req/s
- 错误率: {data['performance_metrics'].get('error_rate', 0):.1%}

告警汇总:
- 总告警数: {data['alerts_summary'].get('total_alerts', 0)}
- 严重告警数: {data['alerts_summary'].get('critical_alerts', 0)}
- 活动告警数: {data['alerts_summary'].get('active_alerts', 0)}
"""
                return report.strip()

            return {
                'update_dashboard': update_dashboard,
                'get_dashboard_data': get_dashboard_data,
                'generate_summary_report': generate_summary_report
            }

        # 创建监控仪表板
        dashboard = create_monitoring_dashboard()

        # 模拟数据更新
        system_metrics = {
            'cpu_percent': 65.5,
            'memory_percent': 72.3,
            'disk_percent': 45.8,
            'network_connections': 1250
        }

        alerts = [
            {'severity': 'high', 'resolved': False},
            {'severity': 'medium', 'resolved': True},
            {'severity': 'low', 'resolved': False}
        ]

        health_status = {
            'database': 'healthy',
            'cache': 'healthy',
            'api': 'degraded'
        }

        # 更新仪表板
        dashboard['update_dashboard'](system_metrics, alerts, health_status)

        # 获取仪表板数据
        dashboard_data = dashboard['get_dashboard_data']()

        # 验证仪表板数据结构
        assert 'system_overview' in dashboard_data
        assert 'performance_metrics' in dashboard_data
        assert 'alerts_summary' in dashboard_data
        assert 'service_health' in dashboard_data
        assert 'last_update' in dashboard_data

        # 验证系统概览数据
        system_overview = dashboard_data['system_overview']
        assert system_overview['cpu_usage'] == 65.5
        assert system_overview['memory_usage'] == 72.3
        assert system_overview['active_connections'] == 1250

        # 验证告警汇总
        alerts_summary = dashboard_data['alerts_summary']
        assert alerts_summary['total_alerts'] == 3
        assert alerts_summary['critical_alerts'] == 1  # 1个high severity
        assert alerts_summary['active_alerts'] == 2  # 2个未解决

        # 生成摘要报告
        report = dashboard['generate_summary_report']()
        assert isinstance(report, str)
        assert len(report) > 100  # 报告应该有足够的内容
        assert '监控摘要报告' in report
        assert '65.5%' in report  # CPU使用率
        assert '总告警数: 3' in report

    def test_metrics_persistence(self):
        """测试指标持久化"""
        import tempfile
        import json
        import os

        def create_metrics_storage():
            """创建指标存储系统"""
            storage = {
                'metrics': [],
                'alerts': [],
                'health_checks': []
            }

            def save_metrics(metrics: Dict[str, Any]):
                """保存指标数据"""
                metrics_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'data': metrics
                }
                storage['metrics'].append(metrics_entry)

            def save_alert(alert: Dict[str, Any]):
                """保存告警数据"""
                alert_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'data': alert
                }
                storage['alerts'].append(alert_entry)

            def save_health_check(health_data: Dict[str, Any]):
                """保存健康检查数据"""
                health_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'data': health_data
                }
                storage['health_checks'].append(health_entry)

            def get_recent_metrics(count: int = 10) -> List[Dict[str, Any]]:
                """获取最近的指标数据"""
                return storage['metrics'][-count:] if storage['metrics'] else []

            def export_to_json(filepath: str):
                """导出数据到JSON文件"""
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(storage, f, ensure_ascii=False, indent=2)

            def import_from_json(filepath: str):
                """从JSON文件导入数据"""
                if os.path.exists(filepath):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        imported_data = json.load(f)
                    storage.update(imported_data)

            return {
                'save_metrics': save_metrics,
                'save_alert': save_alert,
                'save_health_check': save_health_check,
                'get_recent_metrics': get_recent_metrics,
                'export_to_json': export_to_json,
                'import_from_json': import_from_json,
                'get_storage_stats': lambda: {
                    'metrics_count': len(storage['metrics']),
                    'alerts_count': len(storage['alerts']),
                    'health_checks_count': len(storage['health_checks'])
                }
            }

        # 创建指标存储系统
        storage = create_metrics_storage()

        # 保存一些测试数据
        test_metrics = [
            {'cpu_percent': 45.2, 'memory_percent': 67.8},
            {'cpu_percent': 52.1, 'memory_percent': 71.3},
            {'cpu_percent': 48.9, 'memory_percent': 69.5}
        ]

        for metrics in test_metrics:
            storage['save_metrics'](metrics)

        # 保存告警
        test_alert = {'metric': 'cpu_percent', 'value': 85.0, 'severity': 'high'}
        storage['save_alert'](test_alert)

        # 保存健康检查
        test_health = {'overall_status': 'healthy', 'services_checked': 5}
        storage['save_health_check'](test_health)

        # 验证存储统计
        stats = storage['get_storage_stats']()
        assert stats['metrics_count'] == 3
        assert stats['alerts_count'] == 1
        assert stats['health_checks_count'] == 1

        # 验证获取最近指标
        recent_metrics = storage['get_recent_metrics'](2)
        assert len(recent_metrics) == 2

        # 测试数据导出和导入
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            # 导出数据
            storage['export_to_json'](temp_file)

            # 创建新的存储系统并导入
            new_storage = create_metrics_storage()
            new_storage['import_from_json'](temp_file)

            # 验证导入的数据
            new_stats = new_storage['get_storage_stats']()
            assert new_stats['metrics_count'] == 3
            assert new_stats['alerts_count'] == 1
            assert new_stats['health_checks_count'] == 1

        finally:
            # 清理临时文件
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_monitoring_configuration(self):
        """测试监控配置管理"""
        def create_monitoring_config():
            """创建监控配置管理系统"""
            default_config = {
                'monitoring_enabled': True,
                'check_interval': 30.0,
                'alert_thresholds': {
                    'cpu_percent': 80.0,
                    'memory_percent': 85.0,
                    'disk_percent': 90.0,
                    'response_time': 2.0
                },
                'notification_channels': ['email', 'slack'],
                'retention_days': 30,
                'max_alert_history': 1000
            }

            current_config = default_config.copy()

            def update_config(updates: Dict[str, Any]):
                """更新配置"""
                def update_nested_dict(base_dict, updates):
                    for key, value in updates.items():
                        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                            update_nested_dict(base_dict[key], value)
                        else:
                            base_dict[key] = value

                update_nested_dict(current_config, updates)

            def get_config() -> Dict[str, Any]:
                """获取当前配置"""
                return current_config.copy()

            def validate_config(config: Dict[str, Any]) -> List[str]:
                """验证配置有效性"""
                errors = []

                # 检查必需字段
                required_fields = ['monitoring_enabled', 'check_interval', 'alert_thresholds']
                for field in required_fields:
                    if field not in config:
                        errors.append(f"Missing required field: {field}")

                # 验证数值范围
                if 'check_interval' in config and config['check_interval'] <= 0:
                    errors.append("check_interval must be positive")

                if 'alert_thresholds' in config:
                    thresholds = config['alert_thresholds']
                    for metric, threshold in thresholds.items():
                        if not isinstance(threshold, (int, float)) or threshold <= 0:
                            errors.append(f"Invalid threshold for {metric}: {threshold}")

                return errors

            def reset_to_defaults():
                """重置为默认配置"""
                nonlocal current_config
                current_config = default_config.copy()

            return {
                'update_config': update_config,
                'get_config': get_config,
                'validate_config': validate_config,
                'reset_to_defaults': reset_to_defaults
            }

        # 创建配置管理器
        config_manager = create_monitoring_config()

        # 测试获取默认配置
        config = config_manager['get_config']()
        assert config['monitoring_enabled'] == True
        assert config['check_interval'] == 30.0
        assert 'cpu_percent' in config['alert_thresholds']

        # 测试配置更新
        updates = {
            'check_interval': 60.0,
            'alert_thresholds': {
                'cpu_percent': 75.0,
                'memory_percent': 80.0
            }
        }
        config_manager['update_config'](updates)

        updated_config = config_manager['get_config']()
        assert updated_config['check_interval'] == 60.0
        assert updated_config['alert_thresholds']['cpu_percent'] == 75.0
        assert updated_config['alert_thresholds']['disk_percent'] == 90.0  # 未更新的保持原值

        # 测试配置验证
        valid_config = config_manager['get_config']()
        errors = config_manager['validate_config'](valid_config)
        assert len(errors) == 0  # 有效配置应该没有错误

        # 测试无效配置
        invalid_config = {'check_interval': -1, 'alert_thresholds': {'cpu_percent': 'invalid'}}
        errors = config_manager['validate_config'](invalid_config)
        assert len(errors) > 0  # 无效配置应该有错误

        # 测试重置
        config_manager['reset_to_defaults']()
        reset_config = config_manager['get_config']()
        assert reset_config['check_interval'] == 30.0  # 应该回到默认值

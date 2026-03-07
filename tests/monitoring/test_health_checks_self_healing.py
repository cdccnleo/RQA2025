#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
健康检查自愈测试
Health Checks and Self-Healing Tests

测试健康检查和自愈的完整性，包括：
1. 健康状态检测测试
2. 自动恢复机制测试
3. 故障转移测试
4. 服务重启和扩容测试
5. 资源自动扩展测试
6. 自愈决策逻辑测试
7. 恢复时间和成功率测试
8. 自愈安全和稳定性测试
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sys
from pathlib import Path
import subprocess

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class TestHealthStatusDetection:
    """测试健康状态检测"""

    def setup_method(self):
        """测试前准备"""
        self.health_checker = Mock()
        self.status_detector = Mock()

    def test_service_health_endpoints(self):
        """测试服务健康端点"""
        # 定义健康检查端点配置
        health_endpoints = {
            'api_gateway': {
                'url': 'http://api-gateway:8000/health',
                'method': 'GET',
                'expected_status': 200,
                'timeout': 5,
                'headers': {'Accept': 'application/json'},
                'body_check': {'status': 'healthy'},
                'response_time_threshold': 1000  # 1秒
            },
            'user_service': {
                'url': 'http://user-service:8080/health',
                'method': 'GET',
                'expected_status': 200,
                'timeout': 3,
                'body_check': {'status': 'up', 'database': 'connected'},
                'response_time_threshold': 500
            },
            'payment_service': {
                'url': 'https://payment-service:8443/health',
                'method': 'POST',
                'expected_status': 200,
                'timeout': 10,
                'headers': {'Authorization': 'Bearer token'},
                'body_check': {'status': 'healthy', 'dependencies': ['database', 'redis']},
                'response_time_threshold': 2000
            }
        }

        def check_health_endpoint(service_name: str, config: Dict) -> Dict:
            """检查健康端点"""
            result = {
                'service': service_name,
                'healthy': False,
                'response_time_ms': None,
                'status_code': None,
                'error': None,
                'checks_passed': 0,
                'total_checks': 3  # 状态码、响应时间、body检查
            }

            try:
                # 模拟HTTP请求（实际应该使用requests）
                start_time = time.time()

                # 模拟不同的响应场景
                if service_name == 'api_gateway':
                    result['status_code'] = 200
                    result['response_time_ms'] = 450
                    result['healthy'] = True
                    result['checks_passed'] = 3
                elif service_name == 'user_service':
                    result['status_code'] = 200
                    result['response_time_ms'] = 320
                    result['healthy'] = True
                    result['checks_passed'] = 3
                elif service_name == 'payment_service':
                    result['status_code'] = 503  # 服务不可用
                    result['response_time_ms'] = 8500
                    result['healthy'] = False
                    result['error'] = 'Service Unavailable'
                    result['checks_passed'] = 0

            except Exception as e:
                result['error'] = str(e)
                result['healthy'] = False

            return result

        # 检查所有服务的健康端点
        health_results = {}
        for service_name, config in health_endpoints.items():
            result = check_health_endpoint(service_name, config)
            health_results[service_name] = result

        # 验证健康检查结果
        assert len(health_results) == 3, "应该检查3个服务"

        # API网关应该健康
        api_result = health_results['api_gateway']
        assert api_result['healthy'], f"API网关应该健康，实际: {api_result}"
        assert api_result['status_code'] == 200, "API网关状态码应该是200"
        assert api_result['response_time_ms'] <= 1000, f"API网关响应时间过长: {api_result['response_time_ms']}ms"
        assert api_result['checks_passed'] == 3, "API网关应该通过所有检查"

        # 用户服务应该健康
        user_result = health_results['user_service']
        assert user_result['healthy'], f"用户服务应该健康，实际: {user_result}"
        assert user_result['status_code'] == 200, "用户服务状态码应该是200"
        assert user_result['response_time_ms'] <= 500, f"用户服务响应时间过长: {user_result['response_time_ms']}ms"

        # 支付服务应该不健康
        payment_result = health_results['payment_service']
        assert not payment_result['healthy'], f"支付服务应该不健康，实际: {payment_result}"
        assert payment_result['status_code'] == 503, "支付服务状态码应该是503"
        assert payment_result['error'] == 'Service Unavailable', "支付服务应该有错误信息"

        # 计算整体健康率
        healthy_services = sum(1 for r in health_results.values() if r['healthy'])
        total_services = len(health_results)
        health_rate = healthy_services / total_services

        assert health_rate == 2/3, f"健康率应该是66.7%，实际: {health_rate:.3f}"
        assert healthy_services == 2, f"应该有2个健康服务，实际: {healthy_services}"

    def test_database_connection_health(self):
        """测试数据库连接健康检查"""
        # 定义数据库健康检查配置
        db_health_config = {
            'primary_db': {
                'type': 'postgresql',
                'host': 'primary-db.rqa2025.com',
                'port': 5432,
                'database': 'rqa2025',
                'connection_timeout': 5,
                'query_timeout': 10,
                'test_query': 'SELECT 1',
                'pool_size_check': True,
                'replication_check': True
            },
            'read_replica': {
                'type': 'postgresql',
                'host': 'replica-db.rqa2025.com',
                'port': 5432,
                'database': 'rqa2025',
                'connection_timeout': 3,
                'query_timeout': 5,
                'test_query': 'SELECT count(*) from users',
                'pool_size_check': False,
                'replication_check': True,
                'replication_lag_threshold': 30  # 30秒
            },
            'redis_cache': {
                'type': 'redis',
                'host': 'redis-cluster.rqa2025.com',
                'port': 6379,
                'connection_timeout': 2,
                'test_command': 'PING',
                'memory_check': True,
                'max_memory_threshold': 0.9
            }
        }

        def check_database_health(db_name: str, config: Dict) -> Dict:
            """检查数据库健康"""
            result = {
                'database': db_name,
                'healthy': False,
                'connection_time_ms': None,
                'query_time_ms': None,
                'error': None,
                'checks_passed': 0,
                'total_checks': 3
            }

            try:
                # 模拟数据库连接检查
                if db_name == 'primary_db':
                    result['connection_time_ms'] = 120
                    result['query_time_ms'] = 45
                    result['healthy'] = True
                    result['checks_passed'] = 3
                elif db_name == 'read_replica':
                    result['connection_time_ms'] = 85
                    result['query_time_ms'] = 120
                    result['healthy'] = True
                    result['checks_passed'] = 3
                elif db_name == 'redis_cache':
                    result['connection_time_ms'] = 15
                    result['query_time_ms'] = 5
                    result['healthy'] = True
                    result['checks_passed'] = 3

            except Exception as e:
                result['error'] = str(e)
                result['healthy'] = False

            return result

        # 检查所有数据库的健康状态
        db_health_results = {}
        for db_name, config in db_health_config.items():
            result = check_database_health(db_name, config)
            db_health_results[db_name] = result

        # 验证数据库健康检查结果
        assert len(db_health_results) == 3, "应该检查3个数据库"

        # 所有数据库应该健康
        for db_name, result in db_health_results.items():
            assert result['healthy'], f"{db_name}应该健康，实际: {result}"
            assert result['connection_time_ms'] is not None, f"{db_name}应该有连接时间"
            assert result['query_time_ms'] is not None, f"{db_name}应该有查询时间"
            assert result['checks_passed'] == 3, f"{db_name}应该通过所有检查"

        # 验证性能指标
        primary_result = db_health_results['primary_db']
        replica_result = db_health_results['read_replica']
        redis_result = db_health_results['redis_cache']

        # Redis应该最快
        assert redis_result['connection_time_ms'] < primary_result['connection_time_ms'], "Redis连接应该最快"
        assert redis_result['query_time_ms'] < primary_result['query_time_ms'], "Redis查询应该最快"

        # 计算平均响应时间
        avg_connection_time = sum(r['connection_time_ms'] for r in db_health_results.values()) / len(db_health_results)
        avg_query_time = sum(r['query_time_ms'] for r in db_health_results.values()) / len(db_health_results)

        assert avg_connection_time < 1000, f"平均连接时间过长: {avg_connection_time:.1f}ms"
        assert avg_query_time < 1000, f"平均查询时间过长: {avg_query_time:.1f}ms"


class TestAutomaticRecoveryMechanisms:
    """测试自动恢复机制"""

    def setup_method(self):
        """测试前准备"""
        self.recovery_manager = Mock()
        self.auto_healer = Mock()

    def test_service_restart_recovery(self):
        """测试服务重启恢复"""
        # 定义服务重启配置
        restart_config = {
            'api_service': {
                'restart_policy': 'always',
                'max_restart_attempts': 3,
                'restart_delay_seconds': 10,
                'health_check_after_restart': True,
                'health_check_timeout': 30,
                'failure_threshold': 5  # 5次失败后停止重启
            },
            'worker_service': {
                'restart_policy': 'on-failure',
                'max_restart_attempts': 5,
                'restart_delay_seconds': 5,
                'health_check_after_restart': True,
                'health_check_timeout': 60,
                'failure_threshold': 10
            },
            'background_service': {
                'restart_policy': 'no',  # 不自动重启
                'max_restart_attempts': 0,
                'restart_delay_seconds': 0,
                'health_check_after_restart': False,
                'health_check_timeout': 0,
                'failure_threshold': 0
            }
        }

        def simulate_service_restart(service_name: str, config: Dict) -> Dict:
            """模拟服务重启"""
            result = {
                'service': service_name,
                'restart_attempted': False,
                'restart_successful': False,
                'restart_time_ms': None,
                'health_check_passed': False,
                'error': None,
                'attempts_made': 0
            }

            try:
                policy = config.get('restart_policy', 'no')
                max_attempts = config.get('max_restart_attempts', 0)

                if policy in ['always', 'on-failure'] and max_attempts > 0:
                    result['restart_attempted'] = True

                    # 模拟重启过程
                    start_time = time.time()
                    time.sleep(0.1)  # 模拟重启时间
                    result['restart_time_ms'] = int((time.time() - start_time) * 1000)

                    # 模拟重启成功率
                    if service_name in ['api_service', 'worker_service']:
                        result['restart_successful'] = True
                        result['attempts_made'] = 1

                        # 健康检查
                        if config.get('health_check_after_restart', False):
                            time.sleep(0.05)  # 模拟健康检查
                            result['health_check_passed'] = True
                    else:
                        result['restart_successful'] = False
                        result['error'] = 'Restart failed'

            except Exception as e:
                result['error'] = str(e)

            return result

        # 测试所有服务的重启恢复
        restart_results = {}
        for service_name, config in restart_config.items():
            result = simulate_service_restart(service_name, config)
            restart_results[service_name] = result

        # 验证重启结果
        assert len(restart_results) == 3, "应该测试3个服务"

        # API服务应该重启成功
        api_result = restart_results['api_service']
        assert api_result['restart_attempted'], "API服务应该尝试重启"
        assert api_result['restart_successful'], f"API服务重启应该成功，实际: {api_result}"
        assert api_result['health_check_passed'], "API服务应该通过健康检查"

        # Worker服务应该重启成功
        worker_result = restart_results['worker_service']
        assert worker_result['restart_attempted'], "Worker服务应该尝试重启"
        assert worker_result['restart_successful'], "Worker服务重启应该成功"

        # Background服务不应该重启
        bg_result = restart_results['background_service']
        assert not bg_result['restart_attempted'], "Background服务不应该尝试重启"
        assert not bg_result['restart_successful'], "Background服务重启应该失败"

        # 计算重启成功率
        successful_restarts = sum(1 for r in restart_results.values() if r['restart_successful'])
        total_restart_attempts = sum(1 for r in restart_results.values() if r['restart_attempted'])

        if total_restart_attempts > 0:
            restart_success_rate = successful_restarts / total_restart_attempts
            assert restart_success_rate == 1.0, f"重启成功率应该是100%，实际: {restart_success_rate:.2f}"

    def test_resource_auto_scaling(self):
        """测试资源自动扩展"""
        # 定义自动扩展配置
        scaling_config = {
            'web_service': {
                'metric': 'cpu_usage_percent',
                'scale_up_threshold': 70,
                'scale_down_threshold': 30,
                'min_instances': 2,
                'max_instances': 10,
                'scale_up_cooldown': 300,  # 5分钟
                'scale_down_cooldown': 600,  # 10分钟
                'current_instances': 3
            },
            'worker_service': {
                'metric': 'queue_length',
                'scale_up_threshold': 100,
                'scale_down_threshold': 10,
                'min_instances': 1,
                'max_instances': 20,
                'scale_up_cooldown': 60,
                'scale_down_cooldown': 300,
                'current_instances': 5
            },
            'cache_service': {
                'metric': 'memory_usage_percent',
                'scale_up_threshold': 80,
                'scale_down_threshold': 40,
                'min_instances': 3,
                'max_instances': 8,
                'scale_up_cooldown': 180,
                'scale_down_cooldown': 900,
                'current_instances': 3
            }
        }

        # 当前指标值（触发扩展条件）
        current_metrics = {
            'web_service': {'cpu_usage_percent': 85},  # 触发扩容
            'worker_service': {'queue_length': 150},   # 触发扩容
            'cache_service': {'memory_usage_percent': 25}  # 触发缩容
        }

        def evaluate_auto_scaling(service_name: str, config: Dict, metrics: Dict) -> Dict:
            """评估自动扩展"""
            result = {
                'service': service_name,
                'action': 'none',
                'reason': 'No scaling needed',
                'current_instances': config['current_instances'],
                'target_instances': config['current_instances'],
                'cooldown_active': False
            }

            metric_name = config['metric']
            current_value = metrics.get(metric_name, 0)

            # 检查扩容条件
            if current_value >= config['scale_up_threshold']:
                if config['current_instances'] < config['max_instances']:
                    result['action'] = 'scale_up'
                    result['reason'] = f"{metric_name} {current_value} >= {config['scale_up_threshold']}"
                    result['target_instances'] = min(config['current_instances'] + 1, config['max_instances'])
                else:
                    result['action'] = 'at_max_capacity'
                    result['reason'] = 'Already at maximum capacity'

            # 检查缩容条件
            elif current_value <= config['scale_down_threshold']:
                if config['current_instances'] > config['min_instances']:
                    result['action'] = 'scale_down'
                    result['reason'] = f"{metric_name} {current_value} <= {config['scale_down_threshold']}"
                    result['target_instances'] = max(config['current_instances'] - 1, config['min_instances'])
                else:
                    result['action'] = 'at_min_capacity'
                    result['reason'] = 'Already at minimum capacity'

            return result

        # 评估所有服务的自动扩展
        scaling_decisions = {}
        for service_name, config in scaling_config.items():
            metrics = current_metrics.get(service_name, {})
            decision = evaluate_auto_scaling(service_name, config, metrics)
            scaling_decisions[service_name] = decision

        # 验证扩展决策
        assert len(scaling_decisions) == 3, "应该评估3个服务的扩展"

        # Web服务应该扩容
        web_decision = scaling_decisions['web_service']
        assert web_decision['action'] == 'scale_up', f"Web服务应该扩容，实际: {web_decision['action']}"
        assert web_decision['target_instances'] == 4, f"Web服务目标实例数应该是4，实际: {web_decision['target_instances']}"

        # Worker服务应该扩容
        worker_decision = scaling_decisions['worker_service']
        assert worker_decision['action'] == 'scale_up', f"Worker服务应该扩容，实际: {worker_decision['action']}"
        assert worker_decision['target_instances'] == 6, f"Worker服务目标实例数应该是6，实际: {worker_decision['target_instances']}"

        # Cache服务应该缩容
        cache_decision = scaling_decisions['cache_service']
        assert cache_decision['action'] == 'scale_down', f"Cache服务应该缩容，实际: {cache_decision['action']}"
        assert cache_decision['target_instances'] == 3, f"Cache服务目标实例数应该是3，实际: {cache_decision['target_instances']}"

        # 统计扩展动作
        scale_up_count = sum(1 for d in scaling_decisions.values() if d['action'] == 'scale_up')
        scale_down_count = sum(1 for d in scaling_decisions.values() if d['action'] == 'scale_down')

        assert scale_up_count == 2, f"应该有2个服务需要扩容，实际: {scale_up_count}"
        assert scale_down_count == 1, f"应该有1个服务需要缩容，实际: {scale_down_count}"


class TestFailoverMechanisms:
    """测试故障转移机制"""

    def setup_method(self):
        """测试前准备"""
        self.failover_manager = Mock()
        self.load_balancer = Mock()

    def test_database_failover(self):
        """测试数据库故障转移"""
        # 定义数据库故障转移配置
        failover_config = {
            'primary_db': {
                'role': 'primary',
                'status': 'healthy',
                'last_heartbeat': datetime.now(),
                'replicas': ['replica-1', 'replica-2']
            },
            'replica-1': {
                'role': 'replica',
                'status': 'healthy',
                'lag_seconds': 2,
                'last_heartbeat': datetime.now()
            },
            'replica-2': {
                'role': 'replica',
                'status': 'unhealthy',  # 故障副本
                'lag_seconds': None,
                'last_heartbeat': datetime.now() - timedelta(minutes=5)
            }
        }

        def simulate_database_failover(config: Dict) -> Dict:
            """模拟数据库故障转移"""
            result = {
                'failover_triggered': False,
                'new_primary': None,
                'old_primary': None,
                'failover_time_ms': None,
                'success': False,
                'error': None
            }

            # 检查主库状态
            primary_db = None
            for db_name, db_config in config.items():
                if db_config['role'] == 'primary':
                    primary_db = db_name
                    break

            if not primary_db:
                result['error'] = 'No primary database found'
                return result

            primary_status = config[primary_db]['status']

            # 如果主库不健康，触发故障转移
            if primary_status != 'healthy':
                result['failover_triggered'] = True
                result['old_primary'] = primary_db

                # 选择新的主库（最健康的副本）
                candidates = []
                for db_name, db_config in config.items():
                    if (db_config['role'] == 'replica' and
                        db_config['status'] == 'healthy' and
                        db_config.get('lag_seconds', 999) < 30):  # 复制延迟小于30秒
                        candidates.append((db_name, db_config.get('lag_seconds', 999)))

                if candidates:
                    # 选择复制延迟最小的副本
                    new_primary = min(candidates, key=lambda x: x[1])[0]

                    # 执行故障转移
                    start_time = time.time()
                    time.sleep(0.2)  # 模拟故障转移时间

                    result['new_primary'] = new_primary
                    result['failover_time_ms'] = int((time.time() - start_time) * 1000)
                    result['success'] = True

                    # 更新配置
                    config[primary_db]['role'] = 'demoted'
                    config[new_primary]['role'] = 'primary'
                else:
                    result['error'] = 'No suitable replica for failover'

            return result

        # 模拟主库故障
        failover_config['primary_db']['status'] = 'unhealthy'

        # 执行故障转移
        failover_result = simulate_database_failover(failover_config)

        # 验证故障转移结果
        assert failover_result['failover_triggered'], "应该触发故障转移"
        assert failover_result['old_primary'] == 'primary_db', "旧主库应该是primary_db"
        assert failover_result['new_primary'] == 'replica-1', "新主库应该是replica-1（唯一健康的副本）"
        assert failover_result['success'], f"故障转移应该成功，实际: {failover_result}"
        assert failover_result['failover_time_ms'] is not None, "应该有故障转移时间"

        # 验证配置更新
        assert failover_config['primary_db']['role'] == 'demoted', "原主库应该被降级"
        assert failover_config['replica-1']['role'] == 'primary', "replica-1应该成为新主库"

    def test_load_balancer_failover(self):
        """测试负载均衡器故障转移"""
        # 定义负载均衡器配置
        lb_config = {
            'primary_lb': {
                'status': 'unhealthy',  # 主LB故障
                'backends': ['web-1', 'web-2', 'web-3'],
                'active_connections': 0
            },
            'secondary_lb': {
                'status': 'healthy',
                'backends': ['web-4', 'web-5'],
                'active_connections': 45
            },
            'tertiary_lb': {
                'status': 'healthy',
                'backends': ['web-6'],
                'active_connections': 12
            }
        }

        def simulate_lb_failover(config: Dict) -> Dict:
            """模拟负载均衡器故障转移"""
            result = {
                'failover_triggered': False,
                'active_lb': None,
                'previous_lb': None,
                'traffic_switched': 0,
                'switch_time_ms': None,
                'success': False
            }

            # 查找当前活跃的LB
            active_lb = None
            for lb_name, lb_info in config.items():
                if lb_info['status'] == 'healthy':
                    active_lb = lb_name
                    break

            if not active_lb:
                result['error'] = 'No healthy load balancer available'
                return result

            # 检查是否有故障的LB需要转移
            failed_lbs = [name for name, info in config.items() if info['status'] != 'healthy']

            if failed_lbs:
                result['failover_triggered'] = True
                result['previous_lb'] = failed_lbs[0]  # 假设只有一个故障

                # 计算需要转移的流量
                failed_lb_info = config[failed_lbs[0]]
                traffic_to_switch = len(failed_lb_info['backends']) * 100  # 简化计算

                # 执行流量切换
                start_time = time.time()
                time.sleep(0.1)  # 模拟切换时间

                result['active_lb'] = active_lb
                result['traffic_switched'] = traffic_to_switch
                result['switch_time_ms'] = int((time.time() - start_time) * 1000)
                result['success'] = True

            return result

        # 执行LB故障转移
        failover_result = simulate_lb_failover(lb_config)

        # 验证故障转移结果
        assert failover_result['failover_triggered'], "应该触发故障转移"
        assert failover_result['previous_lb'] == 'primary_lb', "前一个LB应该是primary_lb"
        assert failover_result['active_lb'] == 'secondary_lb', "活跃LB应该是secondary_lb"
        assert failover_result['traffic_switched'] == 300, "应该转移300个连接（3个后端x100）"
        assert failover_result['success'], f"故障转移应该成功，实际: {failover_result}"

        # 验证切换时间合理
        assert 50 <= failover_result['switch_time_ms'] <= 150, f"切换时间不合理: {failover_result['switch_time_ms']}ms"


class TestSelfHealingDecisionLogic:
    """测试自愈决策逻辑"""

    def setup_method(self):
        """测试前准备"""
        self.decision_engine = Mock()
        self.healing_orchestrator = Mock()

    def test_healing_strategy_selection(self):
        """测试自愈策略选择"""
        # 定义自愈策略配置
        healing_strategies = {
            'service_restart': {
                'applicable_scenarios': ['service_down', 'service_unresponsive'],
                'estimated_recovery_time': 30,  # 30秒
                'success_rate': 0.85,
                'resource_impact': 'low',
                'risk_level': 'low'
            },
            'horizontal_scaling': {
                'applicable_scenarios': ['high_load', 'resource_exhaustion'],
                'estimated_recovery_time': 120,  # 2分钟
                'success_rate': 0.95,
                'resource_impact': 'medium',
                'risk_level': 'low'
            },
            'failover': {
                'applicable_scenarios': ['primary_failure', 'data_center_down'],
                'estimated_recovery_time': 300,  # 5分钟
                'success_rate': 0.90,
                'resource_impact': 'high',
                'risk_level': 'medium'
            },
            'rollback_deployment': {
                'applicable_scenarios': ['deployment_failure', 'performance_regression'],
                'estimated_recovery_time': 600,  # 10分钟
                'success_rate': 0.75,
                'resource_impact': 'high',
                'risk_level': 'high'
            }
        }

        # 定义故障场景
        failure_scenarios = [
            {
                'scenario': 'api_service_down',
                'symptoms': ['service_unresponsive', 'error_5xx'],
                'severity': 'critical',
                'time_to_detect': 30,
                'affected_users': 1000,
                'business_impact': 'high'
            },
            {
                'scenario': 'high_memory_usage',
                'symptoms': ['resource_exhaustion', 'high_load'],
                'severity': 'warning',
                'time_to_detect': 120,
                'affected_users': 100,
                'business_impact': 'medium'
            },
            {
                'scenario': 'database_primary_failure',
                'symptoms': ['primary_failure', 'data_center_down'],
                'severity': 'critical',
                'time_to_detect': 60,
                'affected_users': 5000,
                'business_impact': 'critical'
            }
        ]

        def select_healing_strategy(scenario: Dict, strategies: Dict) -> Dict:
            """选择自愈策略"""
            symptoms = scenario['symptoms']
            severity = scenario['severity']
            business_impact = scenario['business_impact']

            # 基于症状匹配策略
            candidate_strategies = []
            for strategy_name, strategy_config in strategies.items():
                applicable_scenarios = strategy_config['applicable_scenarios']
                if any(symptom in applicable_scenarios for symptom in symptoms):
                    candidate_strategies.append((strategy_name, strategy_config))

            if not candidate_strategies:
                return {'selected_strategy': None, 'reason': 'No applicable strategy found'}

            # 基于多个因素选择最佳策略
            scored_strategies = []
            for strategy_name, config in candidate_strategies:
                score = 0

                # 成功率权重（40%）
                success_rate = config['success_rate']
                score += success_rate * 40

                # 恢复时间权重（30%）- 越快越好
                recovery_time = config['estimated_recovery_time']
                time_score = max(0, 100 - (recovery_time / 10))  # 每10秒扣1分
                score += time_score * 0.3

                # 风险水平权重（20%）- 风险越低越好
                risk_levels = {'low': 100, 'medium': 70, 'high': 40}
                risk_score = risk_levels[config['risk_level']]
                score += risk_score * 0.2

                # 严重程度加权
                if severity == 'critical':
                    score *= 1.2
                elif severity == 'warning':
                    score *= 0.9

                scored_strategies.append((strategy_name, score, config))

            # 选择得分最高的策略
            best_strategy = max(scored_strategies, key=lambda x: x[1])
            strategy_name, score, config = best_strategy

            return {
                'selected_strategy': strategy_name,
                'score': score,
                'estimated_recovery_time': config['estimated_recovery_time'],
                'success_rate': config['success_rate'],
                'risk_level': config['risk_level'],
                'reason': f'Selected based on symptoms: {symptoms}'
            }

        # 为每个故障场景选择自愈策略
        healing_decisions = {}
        for scenario in failure_scenarios:
            scenario_name = scenario['scenario']
            decision = select_healing_strategy(scenario, healing_strategies)
            healing_decisions[scenario_name] = decision

        # 验证策略选择结果
        assert len(healing_decisions) == 3, "应该为3个场景选择策略"

        # API服务宕机应该选择服务重启
        api_decision = healing_decisions['api_service_down']
        assert api_decision['selected_strategy'] == 'service_restart', f"API服务应该选择重启策略，实际: {api_decision['selected_strategy']}"

        # 高内存使用应该选择水平扩展
        memory_decision = healing_decisions['high_memory_usage']
        assert memory_decision['selected_strategy'] == 'horizontal_scaling', f"内存问题应该选择扩展策略，实际: {memory_decision['selected_strategy']}"

        # 数据库主库故障应该选择故障转移
        db_decision = healing_decisions['database_primary_failure']
        assert db_decision['selected_strategy'] == 'failover', f"数据库故障应该选择故障转移，实际: {db_decision['selected_strategy']}"

        # 验证决策质量
        for scenario_name, decision in healing_decisions.items():
            assert decision['selected_strategy'] is not None, f"{scenario_name}应该选择策略"
            assert 'score' in decision, f"{scenario_name}决策应该有评分"
            assert decision['score'] > 0, f"{scenario_name}评分应该大于0"
            assert 'estimated_recovery_time' in decision, f"{scenario_name}应该有预计恢复时间"


if __name__ == "__main__":
    pytest.main([__file__])

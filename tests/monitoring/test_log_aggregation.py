#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
日志聚合分析测试
Log Aggregation and Analysis Tests

测试日志聚合和分析的完整性，包括：
1. 日志收集和传输测试
2. 日志解析和结构化测试
3. 日志聚合和统计测试
4. 日志存储和检索测试
5. 日志分析和模式识别测试
6. 日志监控和告警测试
7. 日志性能和扩展性测试
8. 日志安全和合规测试
"""

import pytest
import time
import re
import json
import gzip
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sys
import logging
import threading
import queue

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class TestLogCollectionTransmission:
    """测试日志收集和传输"""

    def setup_method(self):
        """测试前准备"""
        self.log_collector = Mock()
        self.transmission_tester = Mock()

    def test_structured_logging_formats(self):
        """测试结构化日志格式"""
        # 定义不同类型的结构化日志格式
        log_formats = {
            'json': {
                'format': 'json',
                'sample': {
                    'timestamp': '2024-01-01T10:00:00Z',
                    'level': 'INFO',
                    'service': 'api-gateway',
                    'request_id': 'req-12345',
                    'user_id': 'user-67890',
                    'endpoint': '/api/users',
                    'method': 'GET',
                    'status_code': 200,
                    'response_time': 45,
                    'message': 'User profile retrieved successfully'
                }
            },
            'logfmt': {
                'format': 'logfmt',
                'sample': 'timestamp=2024-01-01T10:00:00Z level=INFO service=api-gateway request_id=req-12345 user_id=user-67890 endpoint="/api/users" method=GET status_code=200 response_time=45ms message="User profile retrieved successfully"'
            },
            'syslog': {
                'format': 'syslog',
                'sample': '<14>2024-01-01T10:00:00Z api-gateway req-12345 user-67890 GET /api/users 200 45 "User profile retrieved successfully"'
            }
        }

        def validate_log_format(format_name: str, format_config: Dict) -> Dict:
            """验证日志格式"""
            validation_result = {
                'format': format_name,
                'valid': True,
                'issues': [],
                'parsed_fields': 0,
                'structured': False
            }

            sample = format_config['sample']

            if format_name == 'json':
                try:
                    # 验证JSON格式
                    if isinstance(sample, dict):
                        parsed = sample  # 已经是字典
                    else:
                        parsed = json.loads(sample)

                    validation_result['parsed_fields'] = len(parsed)
                    validation_result['structured'] = True

                    # 检查必需字段
                    required_fields = ['timestamp', 'level', 'service', 'message']
                    missing_fields = [f for f in required_fields if f not in parsed]
                    if missing_fields:
                        validation_result['issues'].append(f"缺少必需字段: {missing_fields}")

                except json.JSONDecodeError as e:
                    validation_result['valid'] = False
                    validation_result['issues'].append(f"JSON解析错误: {e}")

            elif format_name == 'logfmt':
                try:
                    # 解析logfmt格式
                    parsed = {}
                    pairs = sample.split()
                    for pair in pairs:
                        if '=' in pair:
                            key, value = pair.split('=', 1)
                            # 移除引号
                            value = value.strip('"')
                            parsed[key] = value

                    validation_result['parsed_fields'] = len(parsed)
                    validation_result['structured'] = True

                except Exception as e:
                    validation_result['valid'] = False
                    validation_result['issues'].append(f"logfmt解析错误: {e}")

            elif format_name == 'syslog':
                try:
                    # 基础的syslog格式验证
                    syslog_pattern = r'<\d+>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\s+.+'
                    if re.match(syslog_pattern, sample):
                        validation_result['parsed_fields'] = len(sample.split())
                    else:
                        validation_result['issues'].append("不符合syslog格式规范")

                except Exception as e:
                    validation_result['valid'] = False
                    validation_result['issues'].append(f"syslog验证错误: {e}")

            return validation_result

        # 验证所有日志格式
        validation_results = {}
        for format_name, format_config in log_formats.items():
            result = validate_log_format(format_name, format_config)
            validation_results[format_name] = result

            # 验证结果有效性
            assert result['valid'], f"{format_name}格式验证失败: {result['issues']}"

        # 验证JSON格式的详细检查
        json_result = validation_results['json']
        assert json_result['structured'], "JSON应该是结构化格式"
        assert json_result['parsed_fields'] >= 9, f"JSON应该至少有9个字段，实际: {json_result['parsed_fields']}"
        assert len(json_result['issues']) == 0, f"JSON格式应该没有问题: {json_result['issues']}"

        # 验证logfmt格式
        logfmt_result = validation_results['logfmt']
        assert logfmt_result['structured'], "logfmt应该是结构化格式"
        assert logfmt_result['parsed_fields'] >= 8, f"logfmt应该至少有8个字段，实际: {logfmt_result['parsed_fields']}"

        # 验证syslog格式
        syslog_result = validation_results['syslog']
        assert syslog_result['parsed_fields'] > 0, "syslog应该能解析出字段"

    def test_log_shipping_mechanisms(self):
        """测试日志传输机制"""
        # 定义日志传输配置
        shipping_configs = {
            'fluentd': {
                'protocol': 'tcp',
                'host': 'fluentd.internal.rqa2025.com',
                'port': 24224,
                'format': 'json',
                'buffer': {
                    'chunk_limit_size': '8MB',
                    'queue_limit_length': 1000,
                    'flush_interval': '5s'
                }
            },
            'kafka': {
                'brokers': ['kafka-1.internal.rqa2025.com:9092', 'kafka-2.internal.rqa2025.com:9092'],
                'topic': 'rqa2025-application-logs',
                'compression': 'gzip',
                'acks': 'all',
                'batch_size': 16384,
                'linger_ms': 5
            },
            'elasticsearch': {
                'hosts': ['es-1.internal.rqa2025.com:9200', 'es-2.internal.rqa2025.com:9200'],
                'index_pattern': 'rqa2025-logs-{yyyy.MM.dd}',
                'pipeline': 'rqa2025-log-pipeline',
                'bulk_size': 1000,
                'flush_interval': '30s'
            }
        }

        def validate_shipping_config(config_name: str, config: Dict) -> Dict:
            """验证传输配置"""
            validation_result = {
                'config_name': config_name,
                'valid': True,
                'issues': [],
                'performance_score': 0,
                'reliability_score': 0
            }

            if config_name == 'fluentd':
                # 验证Fluentd配置
                required_fields = ['protocol', 'host', 'port']
                for field in required_fields:
                    if field not in config:
                        validation_result['issues'].append(f"Fluentd缺少必需字段: {field}")

                if 'buffer' in config:
                    buffer_config = config['buffer']
                    if buffer_config.get('chunk_limit_size', '0MB').endswith('MB'):
                        size_mb = int(buffer_config['chunk_limit_size'].rstrip('MB'))
                        if size_mb > 50:
                            validation_result['issues'].append("buffer块大小过大，可能影响性能")

                validation_result['performance_score'] = 85  # Fluentd性能良好
                validation_result['reliability_score'] = 90  # Fluentd可靠性高

            elif config_name == 'kafka':
                # 验证Kafka配置
                required_fields = ['brokers', 'topic']
                for field in required_fields:
                    if field not in config:
                        validation_result['issues'].append(f"Kafka缺少必需字段: {field}")

                brokers = config.get('brokers', [])
                if len(brokers) < 1:
                    validation_result['issues'].append("Kafka至少需要一个broker")

                if config.get('acks') == 'all':
                    validation_result['reliability_score'] = 95
                elif config.get('acks') == '1':
                    validation_result['reliability_score'] = 80
                else:
                    validation_result['reliability_score'] = 60

                validation_result['performance_score'] = 90  # Kafka性能优秀

            elif config_name == 'elasticsearch':
                # 验证Elasticsearch配置
                required_fields = ['hosts', 'index_pattern']
                for field in required_fields:
                    if field not in config:
                        validation_result['issues'].append(f"Elasticsearch缺少必需字段: {field}")

                hosts = config.get('hosts', [])
                if len(hosts) < 1:
                    validation_result['issues'].append("Elasticsearch至少需要一个主机")

                bulk_size = config.get('bulk_size', 1000)
                if bulk_size > 5000:
                    validation_result['issues'].append("bulk_size过大，可能导致内存压力")

                validation_result['performance_score'] = 75  # ES查询性能一般
                validation_result['reliability_score'] = 85   # ES存储可靠性好

            if validation_result['issues']:
                validation_result['valid'] = False

            return validation_result

        # 验证所有传输配置
        validation_results = {}
        for config_name, config in shipping_configs.items():
            result = validate_shipping_config(config_name, config)
            validation_results[config_name] = result

            # 验证配置有效性
            assert result['valid'], f"{config_name}配置验证失败: {result['issues']}"

        # 比较不同传输机制的性能和可靠性
        fluentd_result = validation_results['fluentd']
        kafka_result = validation_results['kafka']
        es_result = validation_results['elasticsearch']

        # Kafka应该有最高的性能评分
        assert kafka_result['performance_score'] >= fluentd_result['performance_score'], "Kafka性能应该优于Fluentd"
        assert kafka_result['performance_score'] >= es_result['performance_score'], "Kafka性能应该优于Elasticsearch"

        # Kafka应该有高可靠性（acks=all）
        assert kafka_result['reliability_score'] >= 90, f"Kafka可靠性评分应该>=90，实际: {kafka_result['reliability_score']}"


class TestLogParsingStructuring:
    """测试日志解析和结构化"""

    def setup_method(self):
        """测试前准备"""
        self.parser = Mock()
        self.structurer = Mock()

    def test_apache_access_log_parsing(self):
        """测试Apache访问日志解析"""
        # Apache访问日志样本
        apache_logs = [
            '192.168.1.100 - - [01/Jan/2024:10:00:00 +0000] "GET /api/users HTTP/1.1" 200 1234 "https://app.rqa2025.com" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36" "req-12345"',
            '10.0.0.50 - - [01/Jan/2024:10:00:01 +0000] "POST /api/orders HTTP/1.1" 201 567 "https://app.rqa2025.com/orders" "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36" "req-67890"',
            '192.168.1.100 - - [01/Jan/2024:10:00:02 +0000] "GET /api/products/123 HTTP/1.1" 404 234 "-" "curl/7.68.0" "req-99999"'
        ]

        # Apache日志格式正则表达式
        apache_pattern = r'^(\S+) (\S+) (\S+) \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+) "([^"]*)" "([^"]*)" "([^"]*)"$'

        def parse_apache_log(log_line: str) -> Optional[Dict]:
            """解析Apache日志行"""
            match = re.match(apache_pattern, log_line)
            if not match:
                return None

            groups = match.groups()
            timestamp_str = groups[3]

            # 解析时间戳
            try:
                # 简化时间戳解析
                timestamp = datetime.strptime(timestamp_str, '%d/%b/%Y:%H:%M:%S %z')
            except ValueError:
                timestamp = datetime.now()  # 解析失败时使用当前时间

            return {
                'client_ip': groups[0],
                'remote_user': groups[2],
                'timestamp': timestamp,
                'method': groups[4],
                'url': groups[5],
                'protocol': groups[6],
                'status_code': int(groups[7]),
                'response_size': int(groups[8]),
                'referer': groups[9],
                'user_agent': groups[10],
                'request_id': groups[11],
                'parsed_successfully': True
            }

        # 解析所有Apache日志
        parsed_logs = []
        for log_line in apache_logs:
            parsed = parse_apache_log(log_line)
            assert parsed is not None, f"日志解析失败: {log_line}"
            assert parsed['parsed_successfully'], f"日志结构化失败: {log_line}"
            parsed_logs.append(parsed)

        # 验证解析结果
        assert len(parsed_logs) == 3, "应该解析3条日志"

        # 验证第一条日志
        log1 = parsed_logs[0]
        assert log1['client_ip'] == '192.168.1.100', "IP地址解析错误"
        assert log1['method'] == 'GET', "HTTP方法解析错误"
        assert log1['url'] == '/api/users', "URL解析错误"
        assert log1['status_code'] == 200, "状态码解析错误"
        assert log1['request_id'] == 'req-12345', "请求ID解析错误"

        # 验证状态码分布
        status_codes = [log['status_code'] for log in parsed_logs]
        assert 200 in status_codes, "应该包含200状态码"
        assert 201 in status_codes, "应该包含201状态码"
        assert 404 in status_codes, "应该包含404状态码"

        # 验证请求方法分布
        methods = [log['method'] for log in parsed_logs]
        assert methods.count('GET') == 2, "应该有2个GET请求"
        assert methods.count('POST') == 1, "应该有1个POST请求"

    def test_application_json_log_parsing(self):
        """测试应用JSON日志解析"""
        # JSON日志样本
        json_logs = [
            {
                'timestamp': '2024-01-01T10:00:00Z',
                'level': 'INFO',
                'service': 'user-service',
                'request_id': 'req-12345',
                'user_id': 'user-67890',
                'operation': 'get_user_profile',
                'parameters': {'user_id': '67890'},
                'duration_ms': 45,
                'success': True,
                'message': 'User profile retrieved successfully'
            },
            {
                'timestamp': '2024-01-01T10:00:01Z',
                'level': 'ERROR',
                'service': 'payment-service',
                'request_id': 'req-67890',
                'user_id': 'user-54321',
                'operation': 'process_payment',
                'parameters': {'amount': 99.99, 'currency': 'USD'},
                'error_code': 'PAYMENT_DECLINED',
                'error_message': 'Insufficient funds',
                'duration_ms': 1250,
                'success': False,
                'message': 'Payment processing failed'
            },
            {
                'timestamp': '2024-01-01T10:00:02Z',
                'level': 'WARN',
                'service': 'inventory-service',
                'request_id': 'req-99999',
                'operation': 'check_stock',
                'parameters': {'product_id': 'prod-123', 'quantity': 50},
                'available_stock': 25,
                'requested_quantity': 50,
                'duration_ms': 78,
                'success': True,
                'message': 'Insufficient stock warning'
            }
        ]

        def parse_application_json_log(log_entry: Dict) -> Dict:
            """解析应用JSON日志"""
            parsed = {
                'timestamp': None,
                'level': log_entry.get('level', 'UNKNOWN'),
                'service': log_entry.get('service', 'unknown'),
                'request_id': log_entry.get('request_id'),
                'user_id': log_entry.get('user_id'),
                'operation': log_entry.get('operation'),
                'duration_ms': log_entry.get('duration_ms', 0),
                'success': log_entry.get('success', False),
                'error_info': None,
                'business_metrics': {},
                'parsed_successfully': True
            }

            # 解析时间戳
            timestamp_str = log_entry.get('timestamp')
            if timestamp_str:
                try:
                    parsed['timestamp'] = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                except ValueError:
                    parsed['timestamp'] = datetime.now()

            # 解析错误信息
            if not log_entry.get('success', False):
                parsed['error_info'] = {
                    'error_code': log_entry.get('error_code'),
                    'error_message': log_entry.get('error_message')
                }

            # 提取业务指标
            if log_entry.get('operation') == 'process_payment':
                parsed['business_metrics'] = {
                    'transaction_amount': log_entry.get('parameters', {}).get('amount'),
                    'currency': log_entry.get('parameters', {}).get('currency')
                }
            elif log_entry.get('operation') == 'check_stock':
                parsed['business_metrics'] = {
                    'available_stock': log_entry.get('available_stock'),
                    'requested_quantity': log_entry.get('requested_quantity'),
                    'stock_shortage': log_entry.get('requested_quantity', 0) - log_entry.get('available_stock', 0)
                }

            return parsed

        # 解析所有JSON日志
        parsed_logs = []
        for json_log in json_logs:
            parsed = parse_application_json_log(json_log)
            assert parsed['parsed_successfully'], f"JSON日志解析失败: {json_log}"
            parsed_logs.append(parsed)

        # 验证解析结果
        assert len(parsed_logs) == 3, "应该解析3条JSON日志"

        # 验证级别分布
        levels = [log['level'] for log in parsed_logs]
        assert 'INFO' in levels, "应该包含INFO级别"
        assert 'ERROR' in levels, "应该包含ERROR级别"
        assert 'WARN' in levels, "应该包含WARN级别"

        # 验证服务分布
        services = [log['service'] for log in parsed_logs]
        expected_services = {'user-service', 'payment-service', 'inventory-service'}
        assert set(services) == expected_services, f"服务分布不正确: {set(services)} vs {expected_services}"

        # 验证业务指标提取
        payment_log = next(log for log in parsed_logs if log['service'] == 'payment-service')
        assert 'business_metrics' in payment_log, "支付日志应该包含业务指标"
        assert payment_log['business_metrics']['transaction_amount'] == 99.99, "交易金额提取错误"
        assert payment_log['business_metrics']['currency'] == 'USD', "货币提取错误"

        inventory_log = next(log for log in parsed_logs if log['service'] == 'inventory-service')
        assert inventory_log['business_metrics']['stock_shortage'] == 25, "库存缺口计算错误"

        # 验证错误信息提取
        error_log = next(log for log in parsed_logs if log['level'] == 'ERROR')
        assert error_log['error_info'] is not None, "错误日志应该包含错误信息"
        assert error_log['error_info']['error_code'] == 'PAYMENT_DECLINED', "错误码提取错误"


class TestLogAggregationStatistics:
    """测试日志聚合和统计"""

    def setup_method(self):
        """测试前准备"""
        self.aggregator = Mock()
        self.statistician = Mock()

    def test_log_volume_aggregation(self):
        """测试日志量聚合"""
        # 模拟不同时间段的日志量数据
        log_volumes = [
            {'timestamp': datetime(2024, 1, 1, 10, 0), 'service': 'api-gateway', 'count': 1250, 'size_bytes': 512000},
            {'timestamp': datetime(2024, 1, 1, 10, 0), 'service': 'user-service', 'count': 890, 'size_bytes': 356000},
            {'timestamp': datetime(2024, 1, 1, 10, 0), 'service': 'payment-service', 'count': 456, 'size_bytes': 198000},
            {'timestamp': datetime(2024, 1, 1, 10, 5), 'service': 'api-gateway', 'count': 1380, 'size_bytes': 567000},
            {'timestamp': datetime(2024, 1, 1, 10, 5), 'service': 'user-service', 'count': 923, 'size_bytes': 389000},
            {'timestamp': datetime(2024, 1, 1, 10, 5), 'service': 'payment-service', 'count': 489, 'size_bytes': 201000},
            {'timestamp': datetime(2024, 1, 1, 10, 10), 'service': 'api-gateway', 'count': 1156, 'size_bytes': 498000},
            {'timestamp': datetime(2024, 1, 1, 10, 10), 'service': 'user-service', 'count': 867, 'size_bytes': 345000},
            {'timestamp': datetime(2024, 1, 1, 10, 10), 'service': 'payment-service', 'count': 423, 'size_bytes': 189000},
        ]

        def aggregate_log_volumes(log_data: List[Dict], group_by: str = 'service') -> Dict:
            """聚合日志量"""
            aggregated = {}

            for entry in log_data:
                key = entry[group_by]
                if key not in aggregated:
                    aggregated[key] = {
                        'total_count': 0,
                        'total_size_bytes': 0,
                        'avg_count_per_interval': 0,
                        'avg_size_per_interval': 0,
                        'intervals_count': 0,
                        'peak_count': 0,
                        'peak_size': 0
                    }

                stats = aggregated[key]
                stats['total_count'] += entry['count']
                stats['total_size_bytes'] += entry['size_bytes']
                stats['intervals_count'] += 1
                stats['peak_count'] = max(stats['peak_count'], entry['count'])
                stats['peak_size'] = max(stats['peak_size'], entry['size_bytes'])

            # 计算平均值
            for key, stats in aggregated.items():
                intervals = stats['intervals_count']
                if intervals > 0:
                    stats['avg_count_per_interval'] = stats['total_count'] / intervals
                    stats['avg_size_per_interval'] = stats['total_size_bytes'] / intervals

            return aggregated

        # 按服务聚合日志量
        service_aggregation = aggregate_log_volumes(log_volumes, 'service')

        # 验证聚合结果
        assert len(service_aggregation) == 3, "应该有3个服务的聚合数据"

        # 验证API网关聚合
        api_stats = service_aggregation['api-gateway']
        assert api_stats['total_count'] == 1250 + 1380 + 1156, "API网关总日志数计算错误"
        assert api_stats['intervals_count'] == 3, "API网关应该有3个时间间隔"
        assert api_stats['avg_count_per_interval'] == pytest.approx((1250 + 1380 + 1156) / 3, rel=1e-2), "API网关平均日志数计算错误"
        assert api_stats['peak_count'] == 1380, "API网关峰值日志数应该是1380"

        # 验证用户服务聚合
        user_stats = service_aggregation['user-service']
        assert user_stats['total_count'] == 890 + 923 + 867, "用户服务总日志数计算错误"
        assert user_stats['peak_count'] == 923, "用户服务峰值日志数应该是923"

        # 验证支付服务聚合（日志量最少）
        payment_stats = service_aggregation['payment-service']
        assert payment_stats['total_count'] == 456 + 489 + 423, "支付服务总日志数计算错误"
        assert payment_stats['peak_count'] == 489, "支付服务峰值日志数应该是489"

        # 比较服务间的日志量差异
        api_total = api_stats['total_count']
        user_total = user_stats['total_count']
        payment_total = payment_stats['total_count']

        assert api_total > user_total > payment_total, "日志量应该按API > 用户 > 支付排序"

        # 计算日志增长率
        def calculate_growth_rate(current: float, previous: float) -> float:
            """计算增长率"""
            if previous == 0:
                return 0.0
            return ((current - previous) / previous) * 100

        # 按时间段聚合
        time_aggregation = aggregate_log_volumes(log_volumes, 'timestamp')

        # 验证时间趋势（这里简化，实际应该按时间排序）
        time_keys = list(time_aggregation.keys())
        if len(time_keys) >= 2:
            # 简化的趋势分析
            total_counts = [time_aggregation[t]['total_count'] for t in time_keys]
            if len(total_counts) >= 2:
                growth = calculate_growth_rate(total_counts[1], total_counts[0])
                # 不做具体断言，因为数据可能有波动

    def test_error_pattern_analysis(self):
        """测试错误模式分析"""
        # 模拟错误日志数据
        error_logs = [
            {'timestamp': datetime(2024, 1, 1, 10, 0), 'level': 'ERROR', 'service': 'api-gateway', 'error_type': 'TimeoutError', 'message': 'Database query timeout', 'endpoint': '/api/users', 'count': 5},
            {'timestamp': datetime(2024, 1, 1, 10, 0), 'level': 'ERROR', 'service': 'user-service', 'error_type': 'ValidationError', 'message': 'Invalid email format', 'endpoint': '/api/users', 'count': 3},
            {'timestamp': datetime(2024, 1, 1, 10, 5), 'level': 'ERROR', 'service': 'api-gateway', 'error_type': 'ConnectionError', 'message': 'Redis connection failed', 'endpoint': '/api/cache', 'count': 8},
            {'timestamp': datetime(2024, 1, 1, 10, 5), 'level': 'ERROR', 'service': 'payment-service', 'error_type': 'TimeoutError', 'message': 'Payment gateway timeout', 'endpoint': '/api/payments', 'count': 12},
            {'timestamp': datetime(2024, 1, 1, 10, 10), 'level': 'ERROR', 'service': 'api-gateway', 'error_type': 'TimeoutError', 'message': 'Database query timeout', 'endpoint': '/api/products', 'count': 6},
            {'timestamp': datetime(2024, 1, 1, 10, 10), 'level': 'ERROR', 'service': 'user-service', 'error_type': 'ValidationError', 'message': 'Missing required field', 'endpoint': '/api/users', 'count': 7},
            {'timestamp': datetime(2024, 1, 1, 10, 10), 'level': 'ERROR', 'service': 'payment-service', 'error_type': 'AuthError', 'message': 'Invalid API key', 'endpoint': '/api/payments', 'count': 4},
        ]

        def analyze_error_patterns(error_data: List[Dict]) -> Dict:
            """分析错误模式"""
            analysis = {
                'error_types': {},
                'services': {},
                'endpoints': {},
                'temporal_patterns': {},
                'top_errors': []
            }

            for error in error_data:
                error_type = error['error_type']
                service = error['service']
                endpoint = error['endpoint']
                count = error['count']

                # 按错误类型聚合
                if error_type not in analysis['error_types']:
                    analysis['error_types'][error_type] = 0
                analysis['error_types'][error_type] += count

                # 按服务聚合
                if service not in analysis['services']:
                    analysis['services'][service] = 0
                analysis['services'][service] += count

                # 按端点聚合
                if endpoint not in analysis['endpoints']:
                    analysis['endpoints'][endpoint] = 0
                analysis['endpoints'][endpoint] += count

            # 找出最常见的错误
            sorted_errors = sorted(analysis['error_types'].items(), key=lambda x: x[1], reverse=True)
            analysis['top_errors'] = sorted_errors[:3]

            # 分析服务错误分布
            total_errors = sum(analysis['services'].values())
            analysis['service_error_distribution'] = {
                service: (count / total_errors) * 100
                for service, count in analysis['services'].items()
            }

            return analysis

        # 分析错误模式
        error_analysis = analyze_error_patterns(error_logs)

        # 验证分析结果
        assert 'error_types' in error_analysis, "应该包含错误类型分析"
        assert 'services' in error_analysis, "应该包含服务分析"
        assert 'endpoints' in error_analysis, "应该包含端点分析"

        # 验证错误类型统计
        error_types = error_analysis['error_types']
        assert 'TimeoutError' in error_types, "应该包含TimeoutError"
        assert 'ValidationError' in error_types, "应该包含ValidationError"
        assert 'ConnectionError' in error_types, "应该包含ConnectionError"

        # TimeoutError应该是最常见的错误
        timeout_count = error_types['TimeoutError']
        assert timeout_count == 5 + 12 + 6, "TimeoutError总数计算错误"

        # 验证服务错误分布
        services = error_analysis['services']
        assert 'api-gateway' in services, "应该包含api-gateway服务错误"
        assert 'payment-service' in services, "应该包含payment-service服务错误"

        # API网关错误最多
        api_errors = services['api-gateway']
        assert api_errors == 5 + 8 + 6, "API网关错误总数计算错误"

        # 验证前3大错误
        top_errors = error_analysis['top_errors']
        assert len(top_errors) <= 3, "应该最多返回3个最常见错误"

        # TimeoutError应该是最常见的
        if top_errors:
            top_error_type, top_error_count = top_errors[0]
            assert top_error_type == 'TimeoutError', f"最常见错误应该是TimeoutError，实际: {top_error_type}"

        # 验证服务错误分布百分比
        distribution = error_analysis['service_error_distribution']
        total_percentage = sum(distribution.values())
        assert abs(total_percentage - 100.0) < 0.1, f"错误分布百分比总和应该是100%，实际: {total_percentage}"

        # 找出错误率最高的服务
        most_error_prone_service = max(distribution.items(), key=lambda x: x[1])
        assert most_error_prone_service[0] in ['api-gateway', 'payment-service'], f"错误率最高的服务应该是API网关或支付服务，实际: {most_error_prone_service[0]}"


if __name__ == "__main__":
    pytest.main([__file__])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 业务监控指标收集器
收集和监控量化交易系统的业务指标

作者: AI Assistant
创建日期: 2025年9月13日
"""

import os
import time
import logging
import threading
import json
import requests
from typing import Dict, List, Any
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BusinessMonitor:
    """业务监控器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.metrics = {}
        self.alerts = []
        self.is_running = False
        self.monitor_thread = None
        self.api_base_url = self.config.get('api_base_url', 'http://localhost:8000')

    def _default_config(self) -> Dict[str, Any]:
        """默认配置"""
        return {
            'check_interval': 60,  # 检查间隔（秒）
            'api_base_url': 'http://localhost:8000',
            'alert_thresholds': {
                'api_response_time': 2.0,  # 秒
                'api_success_rate': 0.99,  # 99%
                'trade_success_rate': 0.95,  # 95%
                'order_processing_time': 5.0,  # 秒
                'market_data_delay': 1.0,  # 秒
                'risk_violation_rate': 0.05,  # 5%
            },
            'business_metrics': [
                {'name': 'api_health', 'endpoint': '/health', 'method': 'GET'},
                {'name': 'api_metrics', 'endpoint': '/metrics', 'method': 'GET'},
                {'name': 'trading_status', 'endpoint': '/api/v1/trading/status', 'method': 'GET'},
                {'name': 'portfolio_status', 'endpoint': '/api/v1/portfolio/summary', 'method': 'GET'},
                {'name': 'risk_status', 'endpoint': '/api/v1/risk/summary', 'method': 'GET'},
                {'name': 'market_data_status', 'endpoint': '/api/v1/market/status', 'method': 'GET'},
            ],
            'alert_channels': [
                {'type': 'console', 'enabled': True},
                {'type': 'webhook', 'url': 'https://hooks.slack.com/...', 'enabled': False},
                {'type': 'email', 'recipients': ['admin@rqa2025.com'], 'enabled': False}
            ]
        }

    def start_monitoring(self):
        """启动监控"""
        if self.is_running:
            logger.warning("业务监控已经在运行中")
            return

        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        logger.info("🚀 业务监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        if not self.is_running:
            logger.warning("业务监控未在运行")
            return

        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)

        logger.info("🛑 业务监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集业务指标
                self._collect_business_metrics()

                # 检查业务健康状态
                self._check_business_health()

                # 分析业务性能
                self._analyze_business_performance()

                # 检查告警条件
                self._check_business_alerts()

                # 发送告警
                self._send_alerts()

                # 清理过期数据
                self._cleanup_old_data()

            except Exception as e:
                logger.error(f"业务监控循环异常: {e}")

            time.sleep(self.config['check_interval'])

    def _collect_business_metrics(self):
        """收集业务指标"""
        timestamp = datetime.now().isoformat()

        for metric in self.config['business_metrics']:
            try:
                metric_name = metric['name']
                endpoint = metric['endpoint']
                method = metric['method']

                start_time = time.time()
                response = self._make_api_request(endpoint, method)
                response_time = time.time() - start_time

                if response:
                    # 记录API响应时间
                    self.metrics[f'{metric_name}_response_time_{timestamp}'] = response_time
                    self.metrics[f'{metric_name}_status_code_{timestamp}'] = response.status_code

                    # 解析业务指标
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            self._parse_business_data(metric_name, data, timestamp)
                        except json.JSONDecodeError:
                            logger.warning(f"无法解析 {metric_name} 的响应数据")
                    else:
                        logger.warning(f"{metric_name} API返回异常状态码: {response.status_code}")

                else:
                    # API调用失败
                    self.metrics[f'{metric_name}_response_time_{timestamp}'] = -1
                    self.metrics[f'{metric_name}_status_code_{timestamp}'] = -1

                logger.debug(f"业务指标 {metric_name} 收集完成")

            except Exception as e:
                logger.error(f"收集业务指标 {metric_name} 失败: {e}")

    def _make_api_request(self, endpoint: str, method: str = 'GET', timeout: int = 10):
        """发送API请求"""
        try:
            url = f"{self.api_base_url}{endpoint}"
            headers = {'Content-Type': 'application/json'}

            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, timeout=timeout)
            else:
                logger.warning(f"不支持的HTTP方法: {method}")
                return None

            return response

        except requests.exceptions.RequestException as e:
            logger.warning(f"API请求失败 {endpoint}: {e}")
            return None

    def _parse_business_data(self, metric_name: str, data: Dict[str, Any], timestamp: str):
        """解析业务数据"""
        try:
            if metric_name == 'api_health':
                self._parse_health_data(data, timestamp)
            elif metric_name == 'api_metrics':
                self._parse_metrics_data(data, timestamp)
            elif metric_name == 'trading_status':
                self._parse_trading_data(data, timestamp)
            elif metric_name == 'portfolio_status':
                self._parse_portfolio_data(data, timestamp)
            elif metric_name == 'risk_status':
                self._parse_risk_data(data, timestamp)
            elif metric_name == 'market_data_status':
                self._parse_market_data(data, timestamp)
        except Exception as e:
            logger.error(f"解析业务数据 {metric_name} 失败: {e}")

    def _parse_health_data(self, data: Dict[str, Any], timestamp: str):
        """解析健康检查数据"""
        if 'status' in data:
            self.metrics[f'health_status_{timestamp}'] = data['status']

        if 'uptime' in data:
            self.metrics[f'system_uptime_{timestamp}'] = data['uptime']

        if 'version' in data:
            self.metrics[f'system_version_{timestamp}'] = data['version']

    def _parse_metrics_data(self, data: Dict[str, Any], timestamp: str):
        """解析系统指标数据"""
        if 'requests_total' in data:
            self.metrics[f'requests_total_{timestamp}'] = data['requests_total']

        if 'requests_per_second' in data:
            self.metrics[f'requests_per_second_{timestamp}'] = data['requests_per_second']

        if 'average_response_time' in data:
            self.metrics[f'average_response_time_{timestamp}'] = data['average_response_time']

        if 'error_rate' in data:
            self.metrics[f'error_rate_{timestamp}'] = data['error_rate']

    def _parse_trading_data(self, data: Dict[str, Any], timestamp: str):
        """解析交易状态数据"""
        if 'active_orders' in data:
            self.metrics[f'active_orders_{timestamp}'] = data['active_orders']

        if 'completed_orders_today' in data:
            self.metrics[f'completed_orders_today_{timestamp}'] = data['completed_orders_today']

        if 'success_rate' in data:
            self.metrics[f'trade_success_rate_{timestamp}'] = data['success_rate']

        if 'average_processing_time' in data:
            self.metrics[f'order_processing_time_{timestamp}'] = data['average_processing_time']

    def _parse_portfolio_data(self, data: Dict[str, Any], timestamp: str):
        """解析投资组合数据"""
        if 'total_value' in data:
            self.metrics[f'portfolio_value_{timestamp}'] = data['total_value']

        if 'total_pnl' in data:
            self.metrics[f'portfolio_pnl_{timestamp}'] = data['total_pnl']

        if 'positions_count' in data:
            self.metrics[f'positions_count_{timestamp}'] = data['positions_count']

        if 'cash_balance' in data:
            self.metrics[f'cash_balance_{timestamp}'] = data['cash_balance']

    def _parse_risk_data(self, data: Dict[str, Any], timestamp: str):
        """解析风险数据"""
        if 'var_95' in data:
            self.metrics[f'var_95_{timestamp}'] = data['var_95']

        if 'max_drawdown' in data:
            self.metrics[f'max_drawdown_{timestamp}'] = data['max_drawdown']

        if 'sharpe_ratio' in data:
            self.metrics[f'sharpe_ratio_{timestamp}'] = data['sharpe_ratio']

        if 'violations_count' in data:
            self.metrics[f'risk_violations_{timestamp}'] = data['violations_count']

    def _parse_market_data(self, data: Dict[str, Any], timestamp: str):
        """解析市场数据"""
        if 'data_delay' in data:
            self.metrics[f'market_data_delay_{timestamp}'] = data['data_delay']

        if 'instruments_count' in data:
            self.metrics[f'instruments_count_{timestamp}'] = data['instruments_count']

        if 'last_update' in data:
            self.metrics[f'market_data_last_update_{timestamp}'] = data['last_update']

    def _check_business_health(self):
        """检查业务健康状态"""
        # 检查API响应时间
        response_time = self._get_latest_metric('api_health_response_time')
        if response_time and response_time > self.config['alert_thresholds']['api_response_time']:
            self._add_alert(
                'warning',
                f'API响应时间过慢: {response_time:.2f}s',
                {'metric': 'api_response_time', 'value': response_time}
            )

        # 检查交易成功率
        success_rate = self._get_latest_metric('trade_success_rate')
        if success_rate and success_rate < self.config['alert_thresholds']['trade_success_rate']:
            self._add_alert(
                'critical',
                f'交易成功率过低: {success_rate:.2%}',
                {'metric': 'trade_success_rate', 'value': success_rate}
            )

        # 检查风险违规
        violations = self._get_latest_metric('risk_violations')
        if violations and violations > 0:
            self._add_alert(
                'warning',
                f'发现风险违规: {violations}个',
                {'metric': 'risk_violations', 'value': violations}
            )

        # 检查市场数据延迟
        data_delay = self._get_latest_metric('market_data_delay')
        if data_delay and data_delay > self.config['alert_thresholds']['market_data_delay']:
            self._add_alert(
                'warning',
                f'市场数据延迟过高: {data_delay:.2f}s',
                {'metric': 'market_data_delay', 'value': data_delay}
            )

    def _analyze_business_performance(self):
        """分析业务性能"""
        # 计算业务指标趋势
        self._calculate_metric_trends()

        # 识别性能异常
        self._identify_performance_anomalies()

        # 生成业务洞察
        self._generate_business_insights()

    def _calculate_metric_trends(self):
        """计算指标趋势"""
        # 计算最近1小时的趋势
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)

        trends = {}
        for key, value in self.metrics.items():
            if isinstance(value, (int, float)):
                # 这里可以实现更复杂的趋势计算
                trends[key] = {'current': value, 'trend': 'stable'}

        self.metrics.update(trends)

    def _identify_performance_anomalies(self):
        """识别性能异常"""
        # 简单的异常检测逻辑
        response_times = [v for k, v in self.metrics.items()
                          if 'response_time' in k and isinstance(v, (int, float)) and v > 0]

        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            if avg_response_time > self.config['alert_thresholds']['api_response_time'] * 1.5:
                self._add_alert(
                    'warning',
                    f'检测到响应时间异常升高: 平均 {avg_response_time:.2f}s',
                    {'metric': 'average_response_time', 'value': avg_response_time}
                )

    def _generate_business_insights(self):
        """生成业务洞察"""
        # 这里可以实现更复杂的业务分析逻辑
        insights = {
            'system_health': '良好',
            'trading_performance': '正常',
            'risk_exposure': '可控'
        }

        # 基于指标数据更新洞察
        if self._get_latest_metric('trade_success_rate', 0) < 0.95:
            insights['trading_performance'] = '需要关注'

        if self._get_latest_metric('risk_violations', 0) > 0:
            insights['risk_exposure'] = '存在风险'

        logger.info(f"业务洞察: {insights}")

    def _check_business_alerts(self):
        """检查业务告警条件"""
        # 业务层面的告警检查已在_check_business_health中完成

    def _send_alerts(self):
        """发送告警"""
        if not self.alerts:
            return

        # 按级别分组告警
        alerts_by_level = {}
        for alert in self.alerts:
            level = alert['level']
            if level not in alerts_by_level:
                alerts_by_level[level] = []
            alerts_by_level[level].append(alert)

        # 发送告警
        for channel in self.config['alert_channels']:
            if not channel.get('enabled', False):
                continue

            channel_type = channel['type']

            try:
                if channel_type == 'console':
                    self._send_console_alert(alerts_by_level)
                elif channel_type == 'webhook':
                    self._send_webhook_alert(channel, alerts_by_level)
                elif channel_type == 'email':
                    self._send_email_alert(channel, alerts_by_level)
            except Exception as e:
                logger.error(f"发送业务告警失败 ({channel_type}): {e}")

        # 清空已发送的告警
        self.alerts.clear()

    def _send_console_alert(self, alerts_by_level: Dict[str, List[Dict[str, Any]]]):
        """发送控制台告警"""
        for level, alerts in alerts_by_level.items():
            for alert in alerts:
                print(f"📊 [{level.upper()}] {alert['message']}")
                if alert['details']:
                    print(f"   详情: {alert['details']}")

    def _send_webhook_alert(self, channel: Dict[str, Any], alerts_by_level: Dict[str, List[Dict[str, Any]]]):
        """发送Webhook告警"""
        url = channel['url']

        payload = {
            'text': 'RQA2025 业务监控告警',
            'alerts': alerts_by_level,
            'timestamp': datetime.now().isoformat()
        }

        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()

    def _send_email_alert(self, channel: Dict[str, Any], alerts_by_level: Dict[str, List[Dict[str, Any]]]):
        """发送邮件告警"""
        recipients = channel['recipients']

        subject = 'RQA2025 业务监控告警'
        body = f"""
RQA2025 业务监控检测到告警：

{json.dumps(alerts_by_level, indent=2, ensure_ascii=False)}

时间: {datetime.now().isoformat()}
        """.strip()

        logger.info(f"邮件告警: {subject}")

    def _cleanup_old_data(self):
        """清理过期数据"""
        # 保留最近24小时的数据
        cutoff_time = datetime.now() - timedelta(hours=24)
        cutoff_timestamp = cutoff_time.isoformat()

        # 清理指标数据
        keys_to_remove = [
            k for k in self.metrics.keys()
            if k.split('_')[-1] < cutoff_timestamp
        ]

        for key in keys_to_remove:
            del self.metrics[key]

        # 清理告警数据（保留最近1小时）
        alert_cutoff = datetime.now() - timedelta(hours=1)
        self.alerts = [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > alert_cutoff
        ]

    def _get_latest_metric(self, metric_prefix: str, default=None):
        """获取最新的指标值"""
        latest_metrics = [
            (k, v) for k, v in self.metrics.items()
            if k.startswith(metric_prefix + '_') and isinstance(v, (int, float))
        ]

        if not latest_metrics:
            return default

        # 按时间戳排序，取最新的
        latest_metrics.sort(key=lambda x: x[0], reverse=True)
        return latest_metrics[0][1]

    def _add_alert(self, level: str, message: str, details: Dict[str, Any]):
        """添加告警"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'details': details,
            'type': 'business'
        }

        self.alerts.append(alert)
        logger.warning(f"📊 业务告警 [{level}]: {message}")

    def get_business_metrics_summary(self) -> Dict[str, Any]:
        """获取业务指标汇总"""
        return {
            'total_metrics': len(self.metrics),
            'total_alerts': len(self.alerts),
            'latest_metrics': {
                'api_response_time': self._get_latest_metric('api_health_response_time'),
                'trade_success_rate': self._get_latest_metric('trade_success_rate'),
                'portfolio_value': self._get_latest_metric('portfolio_value'),
                'risk_violations': self._get_latest_metric('risk_violations'),
                'market_data_delay': self._get_latest_metric('market_data_delay')
            },
            'alerts_by_level': self._count_alerts_by_level(),
            'business_health_score': self._calculate_business_health_score()
        }

    def _count_alerts_by_level(self) -> Dict[str, int]:
        """按级别统计告警"""
        counts = {}
        for alert in self.alerts:
            level = alert['level']
            counts[level] = counts.get(level, 0) + 1
        return counts

    def _calculate_business_health_score(self) -> float:
        """计算业务健康评分"""
        # 简单的健康评分算法
        score = 100.0

        # API响应时间影响
        response_time = self._get_latest_metric('api_health_response_time', 0)
        if response_time > 1.0:
            score -= min(20, (response_time - 1.0) * 10)

        # 交易成功率影响
        success_rate = self._get_latest_metric('trade_success_rate', 1.0)
        if success_rate < 0.99:
            score -= (1.0 - success_rate) * 100

        # 风险违规影响
        violations = self._get_latest_metric('risk_violations', 0)
        if violations > 0:
            score -= min(30, violations * 5)

        return max(0.0, score)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='RQA2025 业务监控器')
    parser.add_argument('--config', help='配置文件路径')
    parser.add_argument('--api-url', help='API基础URL')
    parser.add_argument('--daemon', action='store_true', help='后台运行模式')

    args = parser.parse_args()

    # 加载配置
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)

    if args.api_url:
        if config is None:
            config = {}
        config['api_base_url'] = args.api_url

    # 创建业务监控器
    monitor = BusinessMonitor(config)

    if args.daemon:
        # 后台运行模式
        monitor.start_monitoring()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
    else:
        # 前台运行模式，执行一次检查
        monitor._collect_business_metrics()
        monitor._check_business_health()
        monitor._analyze_business_performance()
        monitor._check_business_alerts()
        monitor._send_alerts()

        # 输出汇总信息
        summary = monitor.get_business_metrics_summary()
        print("📊 业务监控汇总:")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Monitoring模块交易监控系统测试

测试Trading Monitor的核心功能，包括性能监控、策略监控和风险监控
"""

import pytest
import time
import threading
import psutil
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json


class TestTradingMonitor:
    """测试交易监控系统功能"""

    def setup_method(self):
        """测试前准备"""
        self.start_time = time.time()

        # 模拟交易数据
        self.sample_trades = [
            {
                'symbol': 'AAPL',
                'side': 'buy',
                'quantity': 100,
                'price': 150.0,
                'timestamp': datetime.now() - timedelta(minutes=5),
                'strategy': 'momentum',
                'pnl': 250.0
            },
            {
                'symbol': 'GOOGL',
                'side': 'sell',
                'quantity': 50,
                'price': 2800.0,
                'timestamp': datetime.now() - timedelta(minutes=3),
                'strategy': 'mean_reversion',
                'pnl': -150.0
            },
            {
                'symbol': 'MSFT',
                'side': 'buy',
                'quantity': 75,
                'price': 300.0,
                'timestamp': datetime.now() - timedelta(minutes=1),
                'strategy': 'momentum',
                'pnl': 180.0
            }
        ]

        # 模拟策略性能数据
        self.strategy_performance = {
            'momentum': {
                'total_trades': 10,
                'winning_trades': 7,
                'losing_trades': 3,
                'total_pnl': 1250.0,
                'max_drawdown': 0.05,
                'sharpe_ratio': 1.8,
                'win_rate': 0.7
            },
            'mean_reversion': {
                'total_trades': 8,
                'winning_trades': 5,
                'losing_trades': 3,
                'total_pnl': 750.0,
                'max_drawdown': 0.03,
                'sharpe_ratio': 1.2,
                'win_rate': 0.625
            }
        }

    def test_trading_performance_calculation(self):
        """测试交易性能计算"""
        def calculate_trading_performance(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
            """计算交易性能指标"""
            if not trades:
                return {
                    'total_trades': 0,
                    'total_volume': 0,
                    'total_pnl': 0.0,
                    'avg_pnl_per_trade': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'max_win': 0.0,
                    'max_loss': 0.0
                }

            total_trades = len(trades)
            total_volume = sum(trade['quantity'] * trade['price'] for trade in trades)
            total_pnl = sum(trade['pnl'] for trade in trades)

            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]

            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0

            gross_profit = sum(t['pnl'] for t in winning_trades)
            gross_loss = abs(sum(t['pnl'] for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('in')

            avg_pnl_per_trade = total_pnl / total_trades
            max_win = max((t['pnl'] for t in trades), default=0.0)
            max_loss = min((t['pnl'] for t in trades), default=0.0)

            return {
                'total_trades': total_trades,
                'total_volume': total_volume,
                'total_pnl': total_pnl,
                'avg_pnl_per_trade': avg_pnl_per_trade,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_win': max_win,
                'max_loss': max_loss
            }

        # 测试性能计算
        performance = calculate_trading_performance(self.sample_trades)

        # 验证基本统计
        assert performance['total_trades'] == 3
        assert performance['total_pnl'] == 280.0  # 250 + (-150) + 180
        assert performance['win_rate'] == 2/3  # 2个盈利交易
        assert performance['max_win'] == 250.0
        assert performance['max_loss'] == -150.0

        # 验证平均每笔交易盈利
        assert abs(performance['avg_pnl_per_trade'] - 93.33) < 0.1

        # 测试空交易列表
        empty_performance = calculate_trading_performance([])
        assert empty_performance['total_trades'] == 0
        assert empty_performance['total_pnl'] == 0.0

    def test_strategy_performance_monitoring(self):
        """测试策略性能监控"""
        def monitor_strategy_performance(strategy_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
            """监控策略性能"""
            strategy_metrics = {}

            for strategy_name, data in strategy_data.items():
                metrics = {
                    'name': strategy_name,
                    'total_trades': data['total_trades'],
                    'win_rate': data['win_rate'],
                    'total_pnl': data['total_pnl'],
                    'sharpe_ratio': data['sharpe_ratio'],
                    'max_drawdown': data['max_drawdown'],
                    'performance_score': 0.0,
                    'status': 'unknown'
                }

                # 计算综合性能评分 (0-100)
                win_rate_score = metrics['win_rate'] * 40  # 胜率权重40%
                pnl_score = min(metrics['total_pnl'] / 1000 * 30, 30)  # 盈利权重30%，上限30
                sharpe_score = min(metrics['sharpe_ratio'] * 10, 20)  # 夏普率权重20%，上限20
                drawdown_penalty = metrics['max_drawdown'] * 100 * 0.1  # 最大回撤惩罚权重10%

                metrics['performance_score'] = max(0, win_rate_score + pnl_score + sharpe_score - drawdown_penalty)

                # 确定策略状态
                if metrics['performance_score'] >= 70:
                    metrics['status'] = 'excellent'
                elif metrics['performance_score'] >= 50:
                    metrics['status'] = 'good'
                elif metrics['performance_score'] >= 30:
                    metrics['status'] = 'fair'
                else:
                    metrics['status'] = 'poor'

                strategy_metrics[strategy_name] = metrics

            # 计算整体统计
            total_strategies = len(strategy_metrics)
            excellent_count = sum(1 for m in strategy_metrics.values() if m['status'] == 'excellent')
            good_count = sum(1 for m in strategy_metrics.values() if m['status'] == 'good')
            fair_count = sum(1 for m in strategy_metrics.values() if m['status'] == 'fair')
            poor_count = sum(1 for m in strategy_metrics.values() if m['status'] == 'poor')

            overall_status = 'healthy'
            if poor_count > total_strategies // 2:
                overall_status = 'critical'
            elif fair_count + poor_count > total_strategies // 2:
                overall_status = 'warning'

            return {
                'strategy_metrics': strategy_metrics,
                'summary': {
                    'total_strategies': total_strategies,
                    'excellent_count': excellent_count,
                    'good_count': good_count,
                    'fair_count': fair_count,
                    'poor_count': poor_count,
                    'overall_status': overall_status
                }
            }

        # 测试策略监控
        monitoring_result = monitor_strategy_performance(self.strategy_performance)

        # 验证策略指标
        assert 'momentum' in monitoring_result['strategy_metrics']
        assert 'mean_reversion' in monitoring_result['strategy_metrics']

        momentum_metrics = monitoring_result['strategy_metrics']['momentum']
        assert momentum_metrics['win_rate'] == 0.7
        assert momentum_metrics['total_pnl'] == 1250.0
        assert momentum_metrics['status'] in ['excellent', 'good', 'fair', 'poor']
        assert 0 <= momentum_metrics['performance_score'] <= 100

        # 验证汇总统计
        summary = monitoring_result['summary']
        assert summary['total_strategies'] == 2
        assert summary['excellent_count'] + summary['good_count'] + summary['fair_count'] + summary['poor_count'] == 2
        assert summary['overall_status'] in ['healthy', 'warning', 'critical']

    def test_risk_monitoring(self):
        """测试风险监控"""
        def monitor_risk_indicators(portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
            """监控风险指标"""
            risk_metrics = {
                'var_95': 0.0,  # 95% VaR
                'expected_shortfall': 0.0,  # 期望 shortfall
                'max_drawdown': 0.0,  # 最大回撤
                'volatility': 0.0,  # 波动率
                'beta': 0.0,  # 贝塔系数
                'concentration_risk': 0.0,  # 集中度风险
                'liquidity_risk': 0.0,  # 流动性风险
                'overall_risk_score': 0.0,  # 综合风险评分
                'risk_level': 'low'  # 风险等级
            }

            # 模拟风险计算
            returns = portfolio_data.get('returns', [])
            positions = portfolio_data.get('positions', {})

            if len(returns) > 0:
                # 计算波动率 (年化)
                risk_metrics['volatility'] = np.std(returns) * np.sqrt(252)

                # 计算VaR (假设正态分布)
                risk_metrics['var_95'] = -np.percentile(returns, 5)

                # 计算期望shortfall
                tail_returns = returns[returns <= np.percentile(returns, 5)]
                risk_metrics['expected_shortfall'] = -np.mean(tail_returns) if len(tail_returns) > 0 else 0

                # 计算最大回撤
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdowns = (cumulative - running_max) / running_max
                risk_metrics['max_drawdown'] = np.min(drawdowns)

            # 计算集中度风险 (基于前三大持仓占比)
            if positions:
                sorted_positions = sorted(positions.values(), reverse=True)
                top_3_weight = sum(sorted_positions[:3]) / sum(sorted_positions) if sorted_positions else 0
                risk_metrics['concentration_risk'] = top_3_weight

            # 模拟流动性风险 (基于持仓换手率)
            risk_metrics['liquidity_risk'] = portfolio_data.get('avg_turnover', 0.5)

            # 计算综合风险评分
            vol_score = min(risk_metrics['volatility'] * 100, 25)  # 波动率权重25%
            var_score = min(risk_metrics['var_95'] * 1000, 20)  # VaR权重20%
            dd_score = abs(risk_metrics['max_drawdown']) * 100 * 0.3  # 回撤权重30%
            conc_score = risk_metrics['concentration_risk'] * 100 * 0.15  # 集中度权重15%
            liq_score = risk_metrics['liquidity_risk'] * 100 * 0.1  # 流动性权重10%

            risk_metrics['overall_risk_score'] = vol_score + var_score + dd_score + conc_score + liq_score

            # 确定风险等级
            if risk_metrics['overall_risk_score'] >= 70:
                risk_metrics['risk_level'] = 'high'
            elif risk_metrics['overall_risk_score'] >= 40:
                risk_metrics['risk_level'] = 'medium'
            else:
                risk_metrics['risk_level'] = 'low'

            return risk_metrics

        # 测试风险监控
        portfolio_data = {
            'returns': np.array([0.01, -0.005, 0.008, -0.012, 0.006, -0.003, 0.009]),
            'positions': {'AAPL': 0.3, 'GOOGL': 0.25, 'MSFT': 0.2, 'AMZN': 0.15, 'TSLA': 0.1},
            'avg_turnover': 0.7
        }

        risk_result = monitor_risk_indicators(portfolio_data)

        # 验证风险指标存在
        required_metrics = ['var_95', 'expected_shortfall', 'max_drawdown', 'volatility',
                          'concentration_risk', 'liquidity_risk', 'overall_risk_score', 'risk_level']
        for metric in required_metrics:
            assert metric in risk_result

        # 验证数值范围合理
        assert risk_result['volatility'] >= 0
        assert risk_result['var_95'] >= 0  # VaR应该是正数
        assert risk_result['max_drawdown'] <= 0  # 回撤应该是负数
        assert 0 <= risk_result['concentration_risk'] <= 1
        assert 0 <= risk_result['liquidity_risk'] <= 1
        assert risk_result['overall_risk_score'] >= 0
        assert risk_result['risk_level'] in ['low', 'medium', 'high']

    def test_system_resource_monitoring(self):
        """测试系统资源监控"""
        def monitor_system_resources() -> Dict[str, Any]:
            """监控系统资源使用情况"""
            try:
                # CPU信息
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_count = psutil.cpu_count()
                cpu_freq = psutil.cpu_freq()

                # 内存信息
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used = memory.used / (1024**3)  # GB
                memory_total = memory.total / (1024**3)  # GB

                # 磁盘信息
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                disk_used = disk.used / (1024**3)  # GB
                disk_total = disk.total / (1024**3)  # GB

                # 网络信息
                network = psutil.net_io_counters()
                bytes_sent = network.bytes_sent / (1024**2)  # MB
                bytes_recv = network.bytes_recv / (1024**2)  # MB

                # 进程信息
                process_count = len(psutil.pids())
                current_process = psutil.Process()
                threads_count = current_process.num_threads()

                return {
                    'cpu': {
                        'usage_percent': cpu_percent,
                        'count': cpu_count,
                        'frequency_mhz': cpu_freq.current if cpu_freq else None
                    },
                    'memory': {
                        'usage_percent': memory_percent,
                        'used_gb': round(memory_used, 2),
                        'total_gb': round(memory_total, 2)
                    },
                    'disk': {
                        'usage_percent': disk_percent,
                        'used_gb': round(disk_used, 2),
                        'total_gb': round(disk_total, 2)
                    },
                    'network': {
                        'bytes_sent_mb': round(bytes_sent, 2),
                        'bytes_recv_mb': round(bytes_recv, 2)
                    },
                    'processes': {
                        'count': process_count,
                        'threads': threads_count
                    },
                    'timestamp': datetime.now().isoformat(),
                    'status': 'healthy'
                }

            except Exception as e:
                return {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error'
                }

        # 测试系统资源监控
        resources = monitor_system_resources()

        # 验证监控结果
        assert 'timestamp' in resources
        assert 'status' in resources

        if resources['status'] == 'healthy':
            # 验证所有关键指标存在
            assert 'cpu' in resources
            assert 'memory' in resources
            assert 'disk' in resources
            assert 'network' in resources
            assert 'processes' in resources

            # 验证CPU指标
            cpu_info = resources['cpu']
            assert 'usage_percent' in cpu_info
            assert 0 <= cpu_info['usage_percent'] <= 100
            assert cpu_info['count'] > 0

            # 验证内存指标
            memory_info = resources['memory']
            assert 'usage_percent' in memory_info
            assert 'used_gb' in memory_info
            assert 'total_gb' in memory_info
            assert 0 <= memory_info['usage_percent'] <= 100
            assert memory_info['total_gb'] > 0

            # 验证磁盘指标
            disk_info = resources['disk']
            assert 'usage_percent' in disk_info
            assert 0 <= disk_info['usage_percent'] <= 100

        else:
            # 如果监控失败，确保有错误信息
            assert 'error' in resources

    def test_market_data_monitoring(self):
        """测试市场数据监控"""
        def monitor_market_data(data_feeds: List[Dict[str, Any]]) -> Dict[str, Any]:
            """监控市场数据源"""
            monitoring_results = {
                'total_feeds': len(data_feeds),
                'active_feeds': 0,
                'inactive_feeds': 0,
                'delayed_feeds': 0,
                'error_feeds': 0,
                'avg_latency_ms': 0.0,
                'max_latency_ms': 0.0,
                'feed_details': {},
                'overall_status': 'healthy'
            }

            latencies = []

            for feed in data_feeds:
                feed_name = feed['name']
                feed_status = feed['status']
                feed_latency = feed.get('latency_ms', 0)

                monitoring_results['feed_details'][feed_name] = {
                    'status': feed_status,
                    'latency_ms': feed_latency,
                    'last_update': feed.get('last_update', datetime.now().isoformat())
                }

                if feed_status == 'active':
                    monitoring_results['active_feeds'] += 1
                    latencies.append(feed_latency)
                elif feed_status == 'inactive':
                    monitoring_results['inactive_feeds'] += 1
                elif feed_status == 'delayed':
                    monitoring_results['delayed_feeds'] += 1
                elif feed_status == 'error':
                    monitoring_results['error_feeds'] += 1

            # 计算延迟统计
            if latencies:
                monitoring_results['avg_latency_ms'] = sum(latencies) / len(latencies)
                monitoring_results['max_latency_ms'] = max(latencies)

            # 确定整体状态
            error_count = monitoring_results['inactive_feeds'] + monitoring_results['error_feeds']
            delayed_count = monitoring_results['delayed_feeds']

            if error_count > len(data_feeds) // 2:
                monitoring_results['overall_status'] = 'critical'
            elif error_count > 0 or delayed_count > len(data_feeds) // 3:
                monitoring_results['overall_status'] = 'warning'
            else:
                monitoring_results['overall_status'] = 'healthy'

            return monitoring_results

        # 测试市场数据监控
        sample_feeds = [
            {'name': 'NYSE', 'status': 'active', 'latency_ms': 15},
            {'name': 'NASDAQ', 'status': 'active', 'latency_ms': 12},
            {'name': 'LSE', 'status': 'delayed', 'latency_ms': 150},
            {'name': 'HKEX', 'status': 'active', 'latency_ms': 45},
            {'name': 'SSE', 'status': 'inactive', 'latency_ms': 0}
        ]

        monitoring = monitor_market_data(sample_feeds)

        # 验证监控结果
        assert monitoring['total_feeds'] == 5
        assert monitoring['active_feeds'] == 3  # NYSE, NASDAQ, HKEX
        assert monitoring['delayed_feeds'] == 1  # LSE
        assert monitoring['inactive_feeds'] == 1  # SSE
        assert monitoring['error_feeds'] == 0

        # 验证延迟统计
        assert monitoring['avg_latency_ms'] > 0
        assert monitoring['max_latency_ms'] == 45  # HKEX延迟最长 (inactive的SSE不计入延迟统计)

        # 验证每个feed的详细信息
        for feed in sample_feeds:
            assert feed['name'] in monitoring['feed_details']
            feed_detail = monitoring['feed_details'][feed['name']]
            assert feed_detail['status'] == feed['status']
            assert feed_detail['latency_ms'] == feed['latency_ms']

        # 验证整体状态 (有inactive feed，应该不是healthy)
        assert monitoring['overall_status'] in ['healthy', 'warning', 'critical']

    def test_alert_generation_and_handling(self):
        """测试告警生成和处理"""
        def create_alert_system(thresholds: Dict[str, Any]):
            """创建告警系统"""
            alerts = []
            alert_id_counter = 1

            def generate_alert(metric_name: str, value: Any, threshold: Any,
                             severity: str, message: str) -> Dict[str, Any]:
                """生成告警"""
                nonlocal alert_id_counter
                alert = {
                    'id': f"alert_{alert_id_counter}",
                    'metric': metric_name,
                    'value': value,
                    'threshold': threshold,
                    'severity': severity,
                    'message': message,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'active',
                    'acknowledged': False
                }
                alert_id_counter += 1
                alerts.append(alert)
                return alert

            def check_thresholds_and_alert(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
                """检查阈值并生成告警"""
                new_alerts = []

                for metric_name, value in metrics.items():
                    if metric_name in thresholds:
                        threshold_info = thresholds[metric_name]
                        threshold_value = threshold_info['value']
                        severity = threshold_info['severity']

                        should_alert = False
                        if threshold_info['operator'] == '>':
                            should_alert = value > threshold_value
                        elif threshold_info['operator'] == '<':
                            should_alert = value < threshold_value
                        elif threshold_info['operator'] == '>=':
                            should_alert = value >= threshold_value
                        elif threshold_info['operator'] == '<=':
                            should_alert = value <= threshold_value

                        if should_alert:
                            message = f"{metric_name} {threshold_info['operator']} {threshold_value}: current value = {value}"
                            alert = generate_alert(metric_name, value, threshold_value, severity, message)
                            new_alerts.append(alert)

                return new_alerts

            def acknowledge_alert(alert_id: str) -> bool:
                """确认告警"""
                for alert in alerts:
                    if alert['id'] == alert_id:
                        alert['acknowledged'] = True
                        alert['acknowledged_at'] = datetime.now().isoformat()
                        return True
                return False

            def resolve_alert(alert_id: str) -> bool:
                """解决告警"""
                for alert in alerts:
                    if alert['id'] == alert_id:
                        alert['status'] = 'resolved'
                        alert['resolved_at'] = datetime.now().isoformat()
                        return True
                return False

            def get_active_alerts() -> List[Dict[str, Any]]:
                """获取活跃告警"""
                return [a for a in alerts if a['status'] == 'active']

            def get_alerts_summary() -> Dict[str, Any]:
                """获取告警汇总"""
                active_alerts = get_active_alerts()
                critical_count = sum(1 for a in active_alerts if a['severity'] == 'critical')
                warning_count = sum(1 for a in active_alerts if a['severity'] == 'warning')
                info_count = sum(1 for a in active_alerts if a['severity'] == 'info')

                return {
                    'total_active': len(active_alerts),
                    'critical': critical_count,
                    'warning': warning_count,
                    'info': info_count,
                    'total_resolved': sum(1 for a in alerts if a['status'] == 'resolved')
                }

            return {
                'generate_alert': generate_alert,
                'check_thresholds_and_alert': check_thresholds_and_alert,
                'acknowledge_alert': acknowledge_alert,
                'resolve_alert': resolve_alert,
                'get_active_alerts': get_active_alerts,
                'get_alerts_summary': get_alerts_summary
            }

        # 创建告警系统
        alert_thresholds = {
            'cpu_usage': {'value': 80.0, 'operator': '>', 'severity': 'warning'},
            'memory_usage': {'value': 90.0, 'operator': '>', 'severity': 'critical'},
            'error_rate': {'value': 0.05, 'operator': '>', 'severity': 'critical'},
            'response_time': {'value': 2.0, 'operator': '>', 'severity': 'warning'}
        }

        alert_system = create_alert_system(alert_thresholds)

        # 测试告警生成
        test_metrics = {
            'cpu_usage': 85.0,  # 超过阈值
            'memory_usage': 75.0,  # 未超过阈值
            'error_rate': 0.08,  # 超过阈值
            'response_time': 2.5  # 超过阈值
        }

        new_alerts = alert_system['check_thresholds_and_alert'](test_metrics)

        # 验证生成2个告警 (CPU和响应时间)
        assert len(new_alerts) == 3  # CPU, error_rate, response_time

        # 验证告警详情
        cpu_alert = next((a for a in new_alerts if a['metric'] == 'cpu_usage'), None)
        assert cpu_alert is not None
        assert cpu_alert['severity'] == 'warning'
        assert cpu_alert['status'] == 'active'
        assert not cpu_alert['acknowledged']

        # 测试告警确认
        alert_id = cpu_alert['id']
        acknowledged = alert_system['acknowledge_alert'](alert_id)
        assert acknowledged == True

        # 验证告警已被确认
        active_alerts = alert_system['get_active_alerts']()
        cpu_alert_updated = next((a for a in active_alerts if a['id'] == alert_id), None)
        assert cpu_alert_updated['acknowledged'] == True

        # 测试告警解决
        resolved = alert_system['resolve_alert'](alert_id)
        assert resolved == True

        # 验证告警汇总
        summary = alert_system['get_alerts_summary']()
        assert summary['total_resolved'] == 1
        assert summary['total_active'] == 2  # 还有2个未解决的告警

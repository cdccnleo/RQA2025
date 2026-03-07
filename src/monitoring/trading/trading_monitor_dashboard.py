import logging
#!/usr/bin/env python3
"""
RQA2025 交易层监控面板

专门为交易系统设计的监控面板，提供交易性能、订单状态、风险指标等多维度监控
"""

import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

# Flask相关导入
try:
    from flask import Flask, render_template, jsonify, Response
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Plotly相关导入
try:
    import plotly
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# 导入相关模块

logger = logging.getLogger(__name__)


class TradingMetric(Enum):

    """交易指标枚举"""
    ORDER_LATENCY = "order_latency"          # 订单延迟
    ORDER_THROUGHPUT = "order_throughput"    # 订单吞吐量
    EXECUTION_RATE = "execution_rate"        # 执行率
    CANCEL_RATE = "cancel_rate"             # 取消率
    SLIPPAGE = "slippage"                   # 滑点
    MARKET_DATA_LATENCY = "market_data_latency"  # 行情延迟
    CONNECTION_STATUS = "connection_status"  # 连接状态
    RISK_EXPOSURE = "risk_exposure"         # 风险敞口
    PNL_REALIZED = "pnl_realized"           # 已实现盈亏
    POSITION_SIZE = "position_size"         # 持仓规模


@dataclass
class TradingStatus:

    """交易状态"""
    timestamp: datetime
    metrics: Dict[str, float] = field(default_factory=dict)
    orders: Dict[str, Any] = field(default_factory=dict)
    positions: Dict[str, Any] = field(default_factory=dict)
    connections: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)


class TradingMonitorDashboard:

    """交易层监控面板"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}

        # 配置参数
        self.update_interval = self.config.get('update_interval', 5.0)  # 更新间隔(秒)
        self.history_size = self.config.get('history_size', 3600)  # 历史数据大小(秒)
        self.alert_threshold = self.config.get('alert_threshold', 0.8)  # 告警阈值

        # 数据存储
        self.trading_history = []  # 交易历史
        self.current_status = TradingStatus(timestamp=datetime.now())

        # Web应用
        self.app = None
        self.cors = None

        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None

        # 回调函数
        self.status_callbacks: List[Callable[[TradingStatus], None]] = []

        # 初始化Web应用
        if FLASK_AVAILABLE:
            self._initialize_web_app()

        logger.info("交易层监控面板初始化完成")

    def _initialize_web_app(self):
        """初始化Web应用"""
        self.app = Flask(__name__,
                         template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                         static_folder=os.path.join(os.path.dirname(__file__), 'static'))

        if CORS:
            self.cors = CORS(self.app)

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        """注册路由"""

        @self.app.route('/')
        def index():
            """主页"""
            return render_template('trading_dashboard.html')

        @self.app.route('/api/trading/status')
        def get_trading_status():
            """获取交易状态"""
            try:
                status_data = self._get_current_status_data()
                return jsonify(status_data)
            except Exception as e:
                logger.error(f"获取交易状态失败: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/trading/metrics')
        def get_trading_metrics():
            """获取交易指标"""
            try:
                metrics_data = self._get_metrics_data()
                return jsonify(metrics_data)
            except Exception as e:
                logger.error(f"获取交易指标失败: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/trading/orders')
        def get_order_status():
            """获取订单状态"""
            try:
                order_data = self._get_order_status_data()
                return jsonify(order_data)
            except Exception as e:
                logger.error(f"获取订单状态失败: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/trading/positions')
        def get_position_status():
            """获取持仓状态"""
            try:
                position_data = self._get_position_status_data()
                return jsonify(position_data)
            except Exception as e:
                logger.error(f"获取持仓状态失败: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/trading/risk')
        def get_risk_metrics():
            """获取风险指标"""
            try:
                risk_data = self._get_risk_metrics_data()
                return jsonify(risk_data)
            except Exception as e:
                logger.error(f"获取风险指标失败: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/trading/connections')
        def get_connection_status():
            """获取连接状态"""
            try:
                connection_data = self._get_connection_status_data()
                return jsonify(connection_data)
            except Exception as e:
                logger.error(f"获取连接状态失败: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/trading/charts/<chart_type>')
        def get_chart_data(chart_type):
            """获取图表数据"""
            try:
                if chart_type == 'performance_overview':
                    return self._get_performance_overview_chart()
                elif chart_type == 'order_flow':
                    return self._get_order_flow_chart()
                elif chart_type == 'pnl_trend':
                    return self._get_pnl_trend_chart()
                elif chart_type == 'risk_exposure':
                    return self._get_risk_exposure_chart()
                elif chart_type == 'connection_health':
                    return self._get_connection_health_chart()
                else:
                    return jsonify({'error': 'Unknown chart type'}), 400
            except Exception as e:
                logger.error(f"获取图表数据失败: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/trading/alerts')
        def get_trading_alerts():
            """获取交易告警"""
            try:
                alert_data = self._get_trading_alerts_data()
                return jsonify(alert_data)
            except Exception as e:
                logger.error(f"获取交易告警失败: {e}")
                return jsonify({'error': str(e)}), 500

    def start_monitoring(self):
        """启动监控"""
        if self.is_monitoring:
            logger.warning("交易监控已在运行")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info("交易监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        logger.info("交易监控已停止")

    def _monitoring_loop(self):
        """监控主循环"""
        while self.is_monitoring:
            try:
                # 收集交易状态数据
                self._collect_trading_status()

                # 触发状态回调
                self._trigger_status_callbacks()

                # 检查告警条件
                self._check_alert_conditions()

            except Exception as e:
                logger.error(f"交易监控循环错误: {e}")

            time.sleep(self.update_interval)

    def _collect_trading_status(self):
        """收集交易状态数据"""
        # 这里应该从实际的交易系统中收集数据
        # 目前使用模拟数据

        timestamp = datetime.now()

        # 模拟交易指标
        metrics = {
            'order_latency': 4.2,  # 4.2ms
            'order_throughput': 1850,  # 1850 TPS
            'execution_rate': 0.95,  # 95%
            'cancel_rate': 0.03,  # 3%
            'slippage': 0.002,  # 0.2%
            'market_data_latency': 2.1,  # 2.1ms
            'connection_status': 0.98,  # 98%
            'risk_exposure': 1250000,  # $1.25M
            'pnl_realized': 45250,  # $45.25K
            'position_size': 87500  # 87500 shares
        }

        # 模拟订单状态
        orders = {
            'pending': 12,
            'executed': 1847,
            'cancelled': 58,
            'rejected': 3
        }

        # 模拟持仓状态
        positions = {
            'AAPL': {'size': 25000, 'avg_price': 175.50, 'current_price': 178.25},
            'GOOGL': {'size': 15000, 'avg_price': 2750.00, 'current_price': 2785.50},
            'MSFT': {'size': 30000, 'avg_price': 335.75, 'current_price': 342.80}
        }

        # 模拟连接状态
        connections = {
            'NYSE': {'status': 'connected', 'latency': 2.1, 'uptime': 0.997},
            'NASDAQ': {'status': 'connected', 'latency': 1.8, 'uptime': 0.995},
            'CME': {'status': 'connected', 'latency': 3.2, 'uptime': 0.999}
        }

        # 更新当前状态
        self.current_status = TradingStatus(
            timestamp=timestamp,
            metrics=metrics,
            orders=orders,
            positions=positions,
            connections=connections,
            alerts=[]
        )

        # 添加到历史记录
        self.trading_history.append({
            'timestamp': timestamp,
            'metrics': metrics,
            'orders': orders,
            'positions': positions,
            'connections': connections
        })

        # 限制历史记录数量
        if len(self.trading_history) > self.history_size:
            self.trading_history = self.trading_history[-self.history_size:]

    def _trigger_status_callbacks(self):
        """触发状态回调"""
        for callback in self.status_callbacks:
            try:
                callback(self.current_status)
            except Exception as e:
                logger.error(f"状态回调执行失败: {e}")

    def _check_alert_conditions(self):
        """检查告警条件"""
        alerts = []

        # 检查延迟告警
        if self.current_status.metrics.get('order_latency', 0) > 10:
            alerts.append({
                'level': 'warning',
                'message': '订单处理延迟过高',
                'value': self.current_status.metrics['order_latency'],
                'threshold': 10
            })

        # 检查连接状态告警
        for exchange, conn_info in self.current_status.connections.items():
            if conn_info.get('status') != 'connected':
                alerts.append({
                    'level': 'error',
                    'message': f'{exchange} 连接断开',
                    'exchange': exchange
                })

        # 检查风险敞口告警
        if self.current_status.metrics.get('risk_exposure', 0) > 2000000:
            alerts.append({
                'level': 'warning',
                'message': '风险敞口过高',
                'value': self.current_status.metrics['risk_exposure'],
                'threshold': 2000000
            })

        self.current_status.alerts = alerts

    def add_status_callback(self, callback: Callable[[TradingStatus], None]):
        """添加状态回调"""
        self.status_callbacks.append(callback)

    def _get_current_status_data(self) -> Dict[str, Any]:
        """获取当前状态数据"""
        return {
            'timestamp': self.current_status.timestamp.isoformat(),
            'metrics': self.current_status.metrics,
            'orders': self.current_status.orders,
            'positions': self.current_status.positions,
            'connections': self.current_status.connections,
            'alerts': self.current_status.alerts,
            'health_score': self._calculate_health_score(),
            'last_update': datetime.now().isoformat()
        }

    def _calculate_health_score(self) -> float:
        """计算健康评分"""
        score = 100.0

        # 基于各种指标计算健康分数
        metrics = self.current_status.metrics

        # 延迟影响
        if metrics.get('order_latency', 0) > 5:
            score -= 10

        # 执行率影响
        execution_rate = metrics.get('execution_rate', 1.0)
        if execution_rate < 0.95:
            score -= (1.0 - execution_rate) * 100

        # 连接状态影响
        connected_count = sum(1 for conn in self.current_status.connections.values()
                              if conn.get('status') == 'connected')
        total_count = len(self.current_status.connections)
        if total_count > 0:
            connection_rate = connected_count / total_count
            if connection_rate < 0.9:
                score -= (1.0 - connection_rate) * 50

        return max(0, min(100, score))

    def _get_metrics_data(self) -> Dict[str, Any]:
        """获取指标数据"""
        # 返回最近1小时的指标历史
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=1)

        recent_history = [
            entry for entry in self.trading_history
            if start_time <= entry['timestamp'] <= end_time
        ]

        if not recent_history:
            return {'error': 'No metrics data available'}

        return {
            'time_range': {'start': start_time.isoformat(), 'end': end_time.isoformat()},
            'data_points': len(recent_history),
            'current_metrics': self.current_status.metrics,
            'historical_data': recent_history[-100:],  # 返回最近100个数据点
            'summary': self._calculate_metrics_summary(recent_history)
        }

    def _calculate_metrics_summary(self, history: List[Dict]) -> Dict[str, Any]:
        """计算指标汇总"""
        if not history:
            return {}

        summary = {}

        # 计算每个指标的统计信息
        for metric_name in ['order_latency', 'order_throughput', 'execution_rate']:
            values = [entry['metrics'].get(metric_name, 0) for entry in history]
            if values:
                summary[metric_name] = {
                    'current': values[-1],
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'trend': 'up' if values[-1] > values[0] else 'down' if values[-1] < values[0] else 'stable'
                }

        return summary

    def _get_order_status_data(self) -> Dict[str, Any]:
        """获取订单状态数据"""
        return {
            'current_orders': self.current_status.orders,
            'order_distribution': self._calculate_order_distribution(),
            'execution_stats': self._calculate_execution_stats(),
            'recent_orders': self._get_recent_orders()
        }

    def _calculate_order_distribution(self) -> Dict[str, float]:
        """计算订单分布"""
        orders = self.current_status.orders
        total = sum(orders.values())

        if total == 0:
            return {}

        return {
            status: (count / total) * 100
            for status, count in orders.items()
        }

    def _calculate_execution_stats(self) -> Dict[str, Any]:
        """计算执行统计"""
        orders = self.current_status.orders
        total_orders = sum(orders.values())
        executed = orders.get('executed', 0)
        cancelled = orders.get('cancelled', 0)
        rejected = orders.get('rejected', 0)

        if total_orders == 0:
            return {}

        return {
            'execution_rate': executed / total_orders,
            'cancel_rate': cancelled / total_orders,
            'reject_rate': rejected / total_orders,
            'success_rate': (executed + cancelled) / total_orders
        }

    def _get_recent_orders(self) -> List[Dict[str, Any]]:
        """获取最近订单"""
        # 这里应该从实际的订单系统中获取
        # 目前返回模拟数据
        return [
            {
                'order_id': f'ORD_{i:06d}',
                'symbol': ['AAPL', 'GOOGL', 'MSFT'][i % 3],
                'side': ['BUY', 'SELL'][i % 2],
                'quantity': 100 * (i % 10 + 1),
                'price': 150 + i % 50,
                'status': ['EXECUTED', 'PENDING', 'CANCELLED'][i % 3],
                'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat()
            }
            for i in range(10)
        ]

    def _get_position_status_data(self) -> Dict[str, Any]:
        """获取持仓状态数据"""
        positions = self.current_status.positions

        return {
            'current_positions': positions,
            'position_summary': self._calculate_position_summary(positions),
            'pnl_analysis': self._calculate_pnl_analysis(positions),
            'risk_metrics': self._calculate_position_risk_metrics(positions)
        }

    def _calculate_position_summary(self, positions: Dict) -> Dict[str, Any]:
        """计算持仓汇总"""
        total_value = 0
        total_pnl = 0

        for symbol, pos_data in positions.items():
            size = pos_data['size']
            avg_price = pos_data['avg_price']
            current_price = pos_data['current_price']

            position_value = size * current_price
            position_pnl = size * (current_price - avg_price)

            total_value += position_value
            total_pnl += position_pnl

        return {
            'total_positions': len(positions),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'pnl_percentage': (total_pnl / max(total_value - total_pnl, 1)) * 100
        }

    def _calculate_pnl_analysis(self, positions: Dict) -> Dict[str, Any]:
        """计算盈亏分析"""
        profitable_positions = 0
        losing_positions = 0
        total_pnl = 0

        for symbol, pos_data in positions.items():
            size = pos_data['size']
            avg_price = pos_data['avg_price']
            current_price = pos_data['current_price']

            pnl = size * (current_price - avg_price)
            total_pnl += pnl

            if pnl > 0:
                profitable_positions += 1
            elif pnl < 0:
                losing_positions += 1

        return {
            'profitable_positions': profitable_positions,
            'losing_positions': losing_positions,
            'neutral_positions': len(positions) - profitable_positions - losing_positions,
            'total_pnl': total_pnl,
            'win_rate': profitable_positions / max(len(positions), 1)
        }

    def _calculate_position_risk_metrics(self, positions: Dict) -> Dict[str, Any]:
        """计算持仓风险指标"""
        # 简化风险计算
        return {
            'concentration_risk': len(positions),  # 持仓集中度
            'exposure_limit': 2000000,  # 风险敞口限额
            'current_exposure': sum(pos['size'] * pos['current_price'] for pos in positions.values()),
            'risk_score': 0.3  # 风险评分 (0 - 1)
        }

    def _get_risk_metrics_data(self) -> Dict[str, Any]:
        """获取风险指标数据"""
        return {
            'exposure_metrics': self._calculate_exposure_metrics(),
            'volatility_metrics': self._calculate_volatility_metrics(),
            'liquidity_metrics': self._calculate_liquidity_metrics(),
            'compliance_metrics': self._calculate_compliance_metrics()
        }

    def _calculate_exposure_metrics(self) -> Dict[str, Any]:
        """计算敞口指标"""
        return {
            'total_exposure': self.current_status.metrics.get('risk_exposure', 0),
            'exposure_limit': 2000000,
            'exposure_utilization': (self.current_status.metrics.get('risk_exposure', 0) / 2000000) * 100,
            'exposure_distribution': {
                'equity': 60,
                'fixed_income': 25,
                'derivatives': 15
            }
        }

    def _calculate_volatility_metrics(self) -> Dict[str, Any]:
        """计算波动性指标"""
        return {
            'portfolio_volatility': 0.15,
            'value_at_risk': 45000,
            'expected_shortfall': 65000,
            'beta_coefficient': 1.2
        }

    def _calculate_liquidity_metrics(self) -> Dict[str, Any]:
        """计算流动性指标"""
        return {
            'liquidity_ratio': 0.85,
            'cash_position': 125000,
            'liquid_assets': 875000,
            'illiquid_assets': 375000
        }

    def _calculate_compliance_metrics(self) -> Dict[str, Any]:
        """计算合规指标"""
        return {
            'compliance_score': 95,
            'breach_count': 0,
            'warning_count': 2,
            'last_audit_date': (datetime.now() - timedelta(days=7)).isoformat()
        }

    def _get_connection_status_data(self) -> Dict[str, Any]:
        """获取连接状态数据"""
        connections = self.current_status.connections

        return {
            'connections': connections,
            'overall_health': self._calculate_connection_health(connections),
            'connection_metrics': self._calculate_connection_metrics(connections),
            'recent_events': self._get_recent_connection_events()
        }

    def _calculate_connection_health(self, connections: Dict) -> Dict[str, Any]:
        """计算连接健康状态"""
        total = len(connections)
        connected = sum(1 for conn in connections.values() if conn.get('status') == 'connected')
        healthy = sum(1 for conn in connections.values()
                      if conn.get('status') == 'connected' and conn.get('uptime', 0) > 0.99)

        return {
            'total_connections': total,
            'connected_count': connected,
            'healthy_count': healthy,
            'health_percentage': (healthy / max(total, 1)) * 100,
            'overall_status': 'healthy' if healthy == total else 'degraded' if connected == total else 'critical'
        }

    def _calculate_connection_metrics(self, connections: Dict) -> Dict[str, Any]:
        """计算连接指标"""
        latencies = [conn.get('latency', 0) for conn in connections.values()
                     if conn.get('status') == 'connected']

        if not latencies:
            return {'average_latency': 0, 'min_latency': 0, 'max_latency': 0}

        return {
            'average_latency': sum(latencies) / len(latencies),
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'latency_distribution': {
                '< 1ms': sum(1 for l in latencies if l < 1),
                '1 - 5ms': sum(1 for l in latencies if 1 <= l < 5),
                '5 - 10ms': sum(1 for l in latencies if 5 <= l < 10),
                '> 10ms': sum(1 for l in latencies if l >= 10)
            }
        }

    def _get_recent_connection_events(self) -> List[Dict[str, Any]]:
        """获取最近连接事件"""
        # 这里应该从实际的连接日志中获取
        # 目前返回模拟数据
        return [
            {
                'timestamp': (datetime.now() - timedelta(minutes=i * 5)).isoformat(),
                'event': 'reconnected' if i % 3 == 0 else 'latency_spike' if i % 3 == 1 else 'status_check',
                'exchange': ['NYSE', 'NASDAQ', 'CME'][i % 3],
                'details': f'Connection event {i + 1}'
            }
            for i in range(10)
        ]

    def _get_trading_alerts_data(self) -> Dict[str, Any]:
        """获取交易告警数据"""
        alerts = self.current_status.alerts

        return {
            'active_alerts': alerts,
            'alert_summary': self._calculate_alert_summary(alerts),
            'alert_trends': self._calculate_alert_trends(),
            'critical_alerts': [alert for alert in alerts if alert.get('level') == 'error'],
            'warning_alerts': [alert for alert in alerts if alert.get('level') == 'warning']
        }

    def _calculate_alert_summary(self, alerts: List[Dict]) -> Dict[str, Any]:
        """计算告警汇总"""
        total = len(alerts)
        errors = sum(1 for alert in alerts if alert.get('level') == 'error')
        warnings = sum(1 for alert in alerts if alert.get('level') == 'warning')

        return {
            'total_alerts': total,
            'error_count': errors,
            'warning_count': warnings,
            'info_count': total - errors - warnings,
            'alert_severity': 'high' if errors > 0 else 'medium' if warnings > 0 else 'low'
        }

    def _calculate_alert_trends(self) -> Dict[str, Any]:
        """计算告警趋势"""
        # 这里应该分析历史告警数据
        return {
            'alert_trend': 'stable',  # stable, increasing, decreasing
            'alert_frequency': 2.5,  # alerts per hour
            'most_common_type': 'latency_warning',
            'resolution_rate': 0.85
        }

    # 图表生成方法

    def _get_performance_overview_chart(self) -> Response:
        """获取性能概览图表"""
        if not PLOTLY_AVAILABLE:
            return jsonify({'error': 'Plotly not available'}), 500

        # 创建性能指标概览图表
        fig = go.Figure()

        # 添加订单延迟指标
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=self.current_status.metrics.get('order_latency', 0),
            title={'text': "订单延迟 (ms)"},
            gauge={'axis': {'range': [None, 20]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 5], 'color': "lightgreen"},
                       {'range': [5, 10], 'color': "yellow"},
                       {'range': [10, 20], 'color': "red"}
            ]}
        ))

        return Response(
            plotly.io.to_json(fig),
            mimetype='application/json'
        )

    def _get_order_flow_chart(self) -> Response:
        """获取订单流图表"""
        if not PLOTLY_AVAILABLE:
            return jsonify({'error': 'Plotly not available'}), 500

        # 创建订单流图表
        orders = self.current_status.orders
        labels = list(orders.keys())
        values = list(orders.values())

        fig = go.Figure(data=[
            go.Pie(
                labels=labels,
                values=values,
                title="订单状态分布"
            )
        ])

        return Response(
            plotly.io.to_json(fig),
            mimetype='application/json'
        )

    def _get_pnl_trend_chart(self) -> Response:
        """获取盈亏趋势图表"""
        if not PLOTLY_AVAILABLE:
            return jsonify({'error': 'Plotly not available'}), 500

        # 创建盈亏趋势图表
        # 这里应该使用实际的历史数据
        dates = [(datetime.now() - timedelta(hours=i)).strftime('%H:%M') for i in range(24)]
        pnl_values = [45000 + (i - 12) * 500 + (hash(str(i)) % 1000 - 500) for i in range(24)]

        fig = go.Figure(data=[
            go.Scatter(
                x=dates,
                y=pnl_values,
                mode='lines+markers',
                name='已实现盈亏',
                line=dict(color='green', width=2)
            )
        ])

        fig.update_layout(
            title='盈亏趋势',
            xaxis_title='时间',
            yaxis_title='盈亏 ($)',
            template='plotly_white'
        )

        return Response(
            plotly.io.to_json(fig),
            mimetype='application/json'
        )

    def _get_risk_exposure_chart(self) -> Response:
        """获取风险敞口图表"""
        if not PLOTLY_AVAILABLE:
            return jsonify({'error': 'Plotly not available'}), 500

        # 创建风险敞口图表
        fig = go.Figure()

        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=self.current_status.metrics.get('risk_exposure', 0) / 1000,  # 转换为K
            title={'text': "风险敞口 (K$)"},
            gauge={'axis': {'range': [None, 2500]},
                   'bar': {'color': "darkred"},
                   'steps': [
                       {'range': [0, 1000], 'color': "lightgreen"},
                       {'range': [1000, 1500], 'color': "yellow"},
                       {'range': [1500, 2500], 'color': "red"}
            ]}
        ))

        return Response(
            plotly.io.to_json(fig),
            mimetype='application/json'
        )

    def _get_connection_health_chart(self) -> Response:
        """获取连接健康图表"""
        if not PLOTLY_AVAILABLE:
            return jsonify({'error': 'Plotly not available'}), 500

        # 创建连接健康图表
        connections = self.current_status.connections
        exchanges = list(connections.keys())
        latencies = [conn.get('latency', 0) for conn in connections.values()]

        fig = go.Figure(data=[
            go.Bar(
                x=exchanges,
                y=latencies,
                marker_color=['green' if conn.get('status') == 'connected' else 'red'
                              for conn in connections.values()],
                name='连接延迟'
            )
        ])

        fig.update_layout(
            title='市场连接延迟',
            xaxis_title='交易所',
            yaxis_title='延迟 (ms)',
            template='plotly_white'
        )

        return Response(
            plotly.io.to_json(fig),
            mimetype='application/json'
        )

    def start_server(self, host: str = 'localhost', port: int = 5002, debug: bool = False):
        """启动Web服务器"""
        if not self.app:
            logger.error("Web应用未初始化")
            return

        logger.info(f"启动交易层监控面板服务器: http://{host}:{port}")

        try:
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                threaded=True
            )
        except Exception as e:
            logger.error(f"启动服务器失败: {e}")

    def run_in_background(self, host: str = 'localhost', port: int = 5002, debug: bool = False):
        """在后台运行服务器"""
        server_thread = threading.Thread(
            target=self.start_server,
            args=(host, port, debug),
            daemon=True
        )
        server_thread.start()
        logger.info("交易层监控面板已在后台启动")

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """获取仪表板汇总信息"""
        return {
            'current_status': self._get_current_status_data(),
            'health_score': self._calculate_health_score(),
            'active_alerts': len(self.current_status.alerts),
            'total_positions': len(self.current_status.positions),
            'connected_exchanges': sum(1 for conn in self.current_status.connections.values()
                                       if conn.get('status') == 'connected'),
            'last_update': datetime.now().isoformat()
        }

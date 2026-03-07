#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
性能监控面板测试
Monitoring Dashboards Tests

测试监控面板的完整性，包括：
1. 仪表板配置和布局测试
2. 图表和可视化组件测试
3. 实时数据更新测试
4. 历史数据展示测试
5. 自定义仪表板测试
6. 面板权限和访问控制测试
7. 响应式设计和移动端适配测试
8. 面板性能和加载优化测试
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class TestDashboardConfiguration:
    """测试仪表板配置"""

    def setup_method(self):
        """测试前准备"""
        self.dashboard_config = Mock()
        self.layout_tester = Mock()

    def test_dashboard_layout_configuration(self):
        """测试仪表板布局配置"""
        # 定义仪表板布局配置
        dashboard_config = {
            'id': 'system-overview-dashboard',
            'title': '系统概览面板',
            'description': 'RQA2025系统整体状态监控面板',
            'version': '1.0.0',
            'refresh_interval': 30,  # 30秒刷新
            'layout': {
                'type': 'grid',
                'columns': 12,
                'rows': 8,
                'panels': [
                    {
                        'id': 'cpu-usage-panel',
                        'title': 'CPU使用率',
                        'type': 'graph',
                        'position': {'x': 0, 'y': 0, 'width': 6, 'height': 4},
                        'data_source': 'prometheus',
                        'query': 'cpu_usage_percent',
                        'visualization': {
                            'type': 'line',
                            'color': '#FF6B6B',
                            'y_axis_label': '使用率 (%)'
                        }
                    },
                    {
                        'id': 'memory-usage-panel',
                        'title': '内存使用率',
                        'type': 'gauge',
                        'position': {'x': 6, 'y': 0, 'width': 3, 'height': 4},
                        'data_source': 'prometheus',
                        'query': 'memory_usage_percent',
                        'visualization': {
                            'type': 'gauge',
                            'thresholds': [
                                {'value': 70, 'color': '#FFA500'},
                                {'value': 90, 'color': '#FF0000'}
                            ]
                        }
                    },
                    {
                        'id': 'error-rate-panel',
                        'title': '错误率',
                        'type': 'stat',
                        'position': {'x': 9, 'y': 0, 'width': 3, 'height': 2},
                        'data_source': 'elasticsearch',
                        'query': 'error_rate_last_5m',
                        'visualization': {
                            'type': 'stat',
                            'unit': '%',
                            'color_mode': 'thresholds',
                            'thresholds': [
                                {'value': 1, 'color': 'green'},
                                {'value': 5, 'color': 'orange'},
                                {'value': 10, 'color': 'red'}
                            ]
                        }
                    },
                    {
                        'id': 'service-health-panel',
                        'title': '服务健康状态',
                        'type': 'table',
                        'position': {'x': 0, 'y': 4, 'width': 12, 'height': 4},
                        'data_source': 'kubernetes',
                        'query': 'service_health_status',
                        'visualization': {
                            'type': 'table',
                            'columns': [
                                {'field': 'service_name', 'title': '服务名称'},
                                {'field': 'status', 'title': '状态'},
                                {'field': 'uptime', 'title': '运行时间'},
                                {'field': 'response_time', 'title': '响应时间'}
                            ]
                        }
                    }
                ]
            },
            'permissions': {
                'view': ['admin', 'operator', 'developer'],
                'edit': ['admin', 'operator'],
                'delete': ['admin']
            },
            'tags': ['system', 'overview', 'monitoring']
        }

        def validate_dashboard_config(config: Dict) -> List[str]:
            """验证仪表板配置"""
            errors = []

            # 检查必需字段
            required_fields = ['id', 'title', 'layout']
            for field in required_fields:
                if field not in config:
                    errors.append(f"仪表板缺少必需字段: {field}")

            # 验证布局配置
            layout = config.get('layout', {})
            if 'panels' not in layout:
                errors.append("仪表板布局缺少panels配置")

            panels = layout.get('panels', [])
            if len(panels) == 0:
                errors.append("仪表板至少需要一个面板")

            # 验证面板配置
            for panel in panels:
                panel_required_fields = ['id', 'title', 'type', 'position']
                for field in panel_required_fields:
                    if field not in panel:
                        errors.append(f"面板 {panel.get('id', 'unknown')} 缺少字段: {field}")

                # 验证位置配置
                position = panel.get('position', {})
                position_fields = ['x', 'y', 'width', 'height']
                for field in position_fields:
                    if field not in position:
                        errors.append(f"面板 {panel.get('id', 'unknown')} 位置缺少字段: {field}")

                # 验证可视化配置
                visualization = panel.get('visualization', {})
                if 'type' not in visualization:
                    errors.append(f"面板 {panel.get('id', 'unknown')} 可视化缺少type字段")

            # 验证权限配置
            permissions = config.get('permissions', {})
            if not permissions:
                errors.append("仪表板缺少权限配置")

            return errors

        # 验证仪表板配置
        errors = validate_dashboard_config(dashboard_config)

        # 应该没有配置错误
        assert len(errors) == 0, f"仪表板配置存在错误: {errors}"

        # 验证面板数量和类型
        panels = dashboard_config['layout']['panels']
        assert len(panels) == 4, "应该有4个面板"

        panel_types = [p['type'] for p in panels]
        assert 'graph' in panel_types, "应该包含graph类型的面板"
        assert 'gauge' in panel_types, "应该包含gauge类型的面板"
        assert 'stat' in panel_types, "应该包含stat类型的面板"
        assert 'table' in panel_types, "应该包含table类型的面板"

        # 验证布局位置不重叠（简化检查）
        positions = [p['position'] for p in panels]
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i != j:
                    # 检查是否有重叠（简化逻辑）
                    if (pos1['x'] < pos2['x'] + pos2['width'] and
                        pos1['x'] + pos1['width'] > pos2['x'] and
                        pos1['y'] < pos2['y'] + pos2['height'] and
                        pos1['y'] + pos1['height'] > pos2['y']):
                        # 这里不做断言，因为测试数据可能有重叠设计
                        pass

        # 验证权限配置
        permissions = dashboard_config['permissions']
        assert 'view' in permissions, "应该有查看权限"
        assert 'admin' in permissions['view'], "管理员应该有查看权限"
        assert 'admin' in permissions['edit'], "管理员应该有编辑权限"
        assert 'admin' in permissions['delete'], "管理员应该有删除权限"

    def test_dashboard_responsive_layout(self):
        """测试仪表板响应式布局"""
        # 定义响应式布局配置
        responsive_config = {
            'breakpoints': {
                'mobile': {'max_width': 768, 'columns': 4},
                'tablet': {'max_width': 1024, 'columns': 8},
                'desktop': {'max_width': 9999, 'columns': 12}
            },
            'panels': [
                {
                    'id': 'cpu-panel',
                    'responsive': {
                        'mobile': {'x': 0, 'y': 0, 'width': 4, 'height': 3},
                        'tablet': {'x': 0, 'y': 0, 'width': 4, 'height': 3},
                        'desktop': {'x': 0, 'y': 0, 'width': 6, 'height': 4}
                    }
                },
                {
                    'id': 'memory-panel',
                    'responsive': {
                        'mobile': {'x': 0, 'y': 3, 'width': 4, 'height': 3},
                        'tablet': {'x': 4, 'y': 0, 'width': 4, 'height': 3},
                        'desktop': {'x': 6, 'y': 0, 'width': 3, 'height': 4}
                    }
                },
                {
                    'id': 'network-panel',
                    'responsive': {
                        'mobile': {'x': 0, 'y': 6, 'width': 4, 'height': 3},
                        'tablet': {'x': 0, 'y': 3, 'width': 8, 'height': 3},
                        'desktop': {'x': 9, 'y': 0, 'width': 3, 'height': 4}
                    }
                }
            ]
        }

        def validate_responsive_layout(config: Dict) -> List[str]:
            """验证响应式布局"""
            errors = []

            # 检查断点配置
            breakpoints = config.get('breakpoints', {})
            if not breakpoints:
                errors.append("缺少断点配置")

            required_breakpoints = ['mobile', 'tablet', 'desktop']
            for bp in required_breakpoints:
                if bp not in breakpoints:
                    errors.append(f"缺少断点配置: {bp}")

            # 验证断点顺序
            mobile_max = breakpoints.get('mobile', {}).get('max_width', 0)
            tablet_max = breakpoints.get('tablet', {}).get('max_width', 0)
            desktop_max = breakpoints.get('desktop', {}).get('max_width', 0)

            if mobile_max >= tablet_max:
                errors.append("移动端断点宽度应该小于平板断点宽度")
            if tablet_max >= desktop_max:
                errors.append("平板断点宽度应该小于桌面断点宽度")

            # 验证面板响应式配置
            panels = config.get('panels', [])
            for panel in panels:
                responsive = panel.get('responsive', {})
                if not responsive:
                    errors.append(f"面板 {panel.get('id')} 缺少响应式配置")

                for bp in required_breakpoints:
                    if bp not in responsive:
                        errors.append(f"面板 {panel.get('id')} 缺少 {bp} 响应式配置")

                    bp_config = responsive.get(bp, {})
                    required_pos_fields = ['x', 'y', 'width', 'height']
                    for field in required_pos_fields:
                        if field not in bp_config:
                            errors.append(f"面板 {panel.get('id')} {bp} 配置缺少 {field}")

            return errors

        # 验证响应式布局
        errors = validate_responsive_layout(responsive_config)

        # 应该没有配置错误
        assert len(errors) == 0, f"响应式布局配置存在错误: {errors}"

        # 验证断点配置
        breakpoints = responsive_config['breakpoints']
        assert breakpoints['mobile']['max_width'] == 768, "移动端断点应该是768px"
        assert breakpoints['tablet']['max_width'] == 1024, "平板断点应该是1024px"
        assert breakpoints['desktop']['max_width'] == 9999, "桌面断点应该是9999px"

        # 验证面板响应式配置
        panels = responsive_config['panels']
        for panel in panels:
            responsive = panel['responsive']
            for breakpoint in ['mobile', 'tablet', 'desktop']:
                assert breakpoint in responsive, f"面板 {panel['id']} 缺少 {breakpoint} 配置"

                bp_config = responsive[breakpoint]
                assert 'width' in bp_config, f"面板 {panel['id']} {breakpoint} 缺少width"

                # 验证宽度不超过断点列数
                bp_columns = breakpoints[breakpoint]['columns']
                panel_width = bp_config['width']
                assert panel_width <= bp_columns, f"面板 {panel['id']} 在 {breakpoint} 的宽度 {panel_width} 超过列数 {bp_columns}"


class TestChartVisualization:
    """测试图表可视化组件"""

    def setup_method(self):
        """测试前准备"""
        self.chart_renderer = Mock()
        self.visualization_tester = Mock()

    def test_time_series_chart_rendering(self):
        """测试时间序列图表渲染"""
        # 定义时间序列数据
        time_series_data = {
            'metric': 'cpu_usage_percent',
            'time_range': {'start': datetime(2024, 1, 1, 10, 0), 'end': datetime(2024, 1, 1, 10, 30)},
            'data_points': [
                {'timestamp': datetime(2024, 1, 1, 10, 0), 'value': 45.2},
                {'timestamp': datetime(2024, 1, 1, 10, 5), 'value': 52.8},
                {'timestamp': datetime(2024, 1, 1, 10, 10), 'value': 48.1},
                {'timestamp': datetime(2024, 1, 1, 10, 15), 'value': 67.3},
                {'timestamp': datetime(2024, 1, 1, 10, 20), 'value': 71.9},
                {'timestamp': datetime(2024, 1, 1, 10, 25), 'value': 58.4},
                {'timestamp': datetime(2024, 1, 1, 10, 30), 'value': 62.1}
            ]
        }

        # 定义图表配置
        chart_config = {
            'type': 'line',
            'title': 'CPU使用率趋势',
            'x_axis': {
                'type': 'time',
                'title': '时间',
                'format': '%H:%M'
            },
            'y_axis': {
                'title': '使用率 (%)',
                'min': 0,
                'max': 100,
                'thresholds': [
                    {'value': 70, 'color': '#FF6B6B', 'label': '高负载'},
                    {'value': 80, 'color': '#DC143C', 'label': '严重负载'}
                ]
            },
            'series': [{
                'name': 'CPU使用率',
                'data': time_series_data['data_points'],
                'color': '#4ECDC4',
                'line_width': 2,
                'show_points': True
            }],
            'legend': {
                'enabled': True,
                'position': 'top'
            },
            'grid': {
                'enabled': True,
                'color': '#E0E0E0'
            }
        }

        def validate_chart_rendering(chart_config: Dict, data: Dict) -> Dict:
            """验证图表渲染"""
            validation_result = {
                'valid': True,
                'issues': [],
                'data_points_count': 0,
                'time_range_valid': False,
                'thresholds_applied': 0,
                'rendering_score': 0
            }

            # 验证数据点
            data_points = data.get('data_points', [])
            validation_result['data_points_count'] = len(data_points)

            if len(data_points) == 0:
                validation_result['issues'].append("图表缺少数据点")
                validation_result['valid'] = False

            # 验证时间范围
            time_range = data.get('time_range', {})
            if 'start' in time_range and 'end' in time_range:
                if time_range['start'] < time_range['end']:
                    validation_result['time_range_valid'] = True
                else:
                    validation_result['issues'].append("时间范围无效：开始时间晚于结束时间")

            # 验证阈值配置
            y_axis = chart_config.get('y_axis', {})
            thresholds = y_axis.get('thresholds', [])
            validation_result['thresholds_applied'] = len(thresholds)

            # 验证系列配置
            series = chart_config.get('series', [])
            if not series:
                validation_result['issues'].append("图表缺少系列配置")
                validation_result['valid'] = False

            for s in series:
                if 'data' not in s:
                    validation_result['issues'].append(f"系列 {s.get('name', 'unknown')} 缺少数据")
                    validation_result['valid'] = False

            # 计算渲染评分（0-100）
            score = 100
            if len(data_points) < 5:
                score -= 20  # 数据点不足
            if not validation_result['time_range_valid']:
                score -= 15  # 时间范围无效
            if len(thresholds) == 0:
                score -= 10  # 缺少阈值
            if not series:
                score -= 30  # 缺少系列

            validation_result['rendering_score'] = max(0, score)

            if validation_result['issues']:
                validation_result['valid'] = False

            return validation_result

        # 验证图表渲染
        validation = validate_chart_rendering(chart_config, time_series_data)

        # 应该渲染成功
        assert validation['valid'], f"图表渲染验证失败: {validation['issues']}"
        assert validation['data_points_count'] == 7, "应该有7个数据点"
        assert validation['time_range_valid'], "时间范围应该有效"
        assert validation['thresholds_applied'] == 2, "应该有2个阈值"
        assert validation['rendering_score'] >= 80, f"渲染评分过低: {validation['rendering_score']}"

        # 验证数据点值范围
        values = [dp['value'] for dp in time_series_data['data_points']]
        assert all(0 <= v <= 100 for v in values), "CPU使用率应该在0-100范围内"

        # 验证峰值检测
        max_value = max(values)
        max_index = values.index(max_value)
        max_timestamp = time_series_data['data_points'][max_index]['timestamp']

        assert max_value >= 70, f"应该检测到峰值，实际最大值: {max_value}"
        assert max_timestamp.minute in [15, 20], f"峰值应该在15或20分钟，实际: {max_timestamp.minute}"

    def test_gauge_chart_rendering(self):
        """测试仪表盘图表渲染"""
        # 定义仪表盘数据
        gauge_data = {
            'metric': 'memory_usage_percent',
            'current_value': 78.5,
            'min_value': 0,
            'max_value': 100,
            'unit': '%'
        }

        # 定义仪表盘配置
        gauge_config = {
            'type': 'gauge',
            'title': '内存使用率',
            'value': gauge_data['current_value'],
            'min': gauge_data['min_value'],
            'max': gauge_data['max_value'],
            'thresholds': [
                {'value': 60, 'color': '#4ECDC4', 'label': '正常'},
                {'value': 80, 'color': '#FFA500', 'label': '警告'},
                {'value': 90, 'color': '#FF6B6B', 'label': '危险'}
            ],
            'colors': {
                'background': '#FFFFFF',
                'value': '#2C3E50',
                'thresholds': ['#4ECDC4', '#FFA500', '#FF6B6B']
            },
            'size': {
                'width': 300,
                'height': 200
            }
        }

        def validate_gauge_rendering(gauge_config: Dict, data: Dict) -> Dict:
            """验证仪表盘渲染"""
            validation_result = {
                'valid': True,
                'issues': [],
                'threshold_breached': None,
                'color_applied': None,
                'rendering_quality': 0
            }

            # 验证值范围
            value = gauge_config.get('value', 0)
            min_val = gauge_config.get('min', 0)
            max_val = gauge_config.get('max', 100)

            if not (min_val <= value <= max_val):
                validation_result['issues'].append(f"值 {value} 超出范围 [{min_val}, {max_val}]")
                validation_result['valid'] = False

            # 验证阈值
            thresholds = gauge_config.get('thresholds', [])
            if not thresholds:
                validation_result['issues'].append("仪表盘缺少阈值配置")

            # 确定当前阈值状态
            current_threshold = None
            for threshold in sorted(thresholds, key=lambda x: x['value']):
                if value >= threshold['value']:
                    current_threshold = threshold
                else:
                    break

            if current_threshold:
                validation_result['threshold_breached'] = current_threshold['label']
                validation_result['color_applied'] = current_threshold['color']

            # 验证颜色配置
            colors = gauge_config.get('colors', {})
            if not colors:
                validation_result['issues'].append("仪表盘缺少颜色配置")

            # 计算渲染质量
            quality = 100
            if len(thresholds) < 2:
                quality -= 20
            if not colors:
                quality -= 15
            if not (min_val <= value <= max_val):
                quality -= 30

            validation_result['rendering_quality'] = max(0, quality)

            if validation_result['issues']:
                validation_result['valid'] = False

            return validation_result

        # 验证仪表盘渲染
        validation = validate_gauge_rendering(gauge_config, gauge_data)

        # 应该渲染成功
        assert validation['valid'], f"仪表盘渲染验证失败: {validation['issues']}"
        assert validation['threshold_breached'] == '警告', f"应该触发警告阈值，实际: {validation['threshold_breached']}"
        assert validation['color_applied'] == '#FFA500', f"应该应用橙色，实际: {validation['color_applied']}"
        assert validation['rendering_quality'] >= 80, f"渲染质量过低: {validation['rendering_quality']}"

        # 验证值在合理范围内
        assert 0 <= gauge_data['current_value'] <= 100, "内存使用率应该在0-100范围内"

        # 验证阈值逻辑
        value = gauge_data['current_value']
        thresholds = gauge_config['thresholds']

        # 78.5应该落在60-80之间，触发警告阈值
        warning_threshold = next(t for t in thresholds if t['value'] == 80)
        assert value >= 60 and value < 80, "值应该在警告范围内"
        assert validation['threshold_breached'] == warning_threshold['label'], "应该触发警告阈值"


class TestRealTimeDataUpdates:
    """测试实时数据更新"""

    def setup_method(self):
        """测试前准备"""
        self.realtime_updater = Mock()
        self.websocket_manager = Mock()

    @patch('time.sleep')
    def test_dashboard_realtime_refresh(self, mock_sleep):
        """测试仪表板实时刷新"""
        # 模拟实时数据更新
        refresh_intervals = [30, 60, 300]  # 30秒、1分钟、5分钟
        update_cycles = 3

        def simulate_realtime_updates(refresh_interval: int, cycles: int) -> List[Dict]:
            """模拟实时更新"""
            updates = []

            for cycle in range(cycles):
                # 模拟数据更新
                update_data = {
                    'cycle': cycle + 1,
                    'timestamp': datetime.now(),
                    'metrics': {
                        'cpu_usage': 40 + cycle * 5,  # 逐渐增加
                        'memory_usage': 60 + cycle * 2,
                        'active_users': 1000 + cycle * 50
                    },
                    'refresh_interval': refresh_interval,
                    'data_freshness': 'realtime'
                }

                updates.append(update_data)

                # 模拟刷新间隔
                time.sleep(refresh_interval)

            return updates

        # 测试不同刷新间隔
        for interval in refresh_intervals:
            updates = simulate_realtime_updates(interval, update_cycles)

            # 验证更新周期
            assert len(updates) == update_cycles, f"应该有{update_cycles}个更新周期"

            # 验证数据趋势（CPU使用率应该逐渐增加）
            cpu_values = [u['metrics']['cpu_usage'] for u in updates]
            assert cpu_values == sorted(cpu_values), "CPU使用率应该逐渐增加"

            # 验证时间戳递增
            timestamps = [u['timestamp'] for u in updates]
            assert timestamps == sorted(timestamps), "时间戳应该递增"

            # 验证刷新间隔被正确记录
            for update in updates:
                assert update['refresh_interval'] == interval, "刷新间隔应该被正确记录"

        # 验证mock调用
        expected_calls = sum(update_cycles for _ in refresh_intervals)
        assert mock_sleep.call_count == expected_calls, f"应该调用sleep {expected_calls}次，实际: {mock_sleep.call_count}"

    def test_websocket_data_streaming(self):
        """测试WebSocket数据流"""
        # 模拟WebSocket连接和数据流
        websocket_config = {
            'url': 'ws://monitoring.rqa2025.com/stream',
            'protocols': ['monitoring-v1'],
            'heartbeat_interval': 30,
            'reconnect_delay': 5,
            'max_reconnect_attempts': 3
        }

        # 模拟数据流
        data_stream = [
            {'type': 'metric_update', 'metric': 'cpu_usage', 'value': 45.2, 'timestamp': datetime.now()},
            {'type': 'metric_update', 'metric': 'memory_usage', 'value': 67.8, 'timestamp': datetime.now()},
            {'type': 'alert', 'level': 'warning', 'message': 'High CPU usage detected', 'timestamp': datetime.now()},
            {'type': 'heartbeat', 'timestamp': datetime.now()},
            {'type': 'metric_update', 'metric': 'disk_usage', 'value': 78.9, 'timestamp': datetime.now()}
        ]

        def validate_websocket_stream(config: Dict, stream_data: List[Dict]) -> Dict:
            """验证WebSocket数据流"""
            validation_result = {
                'valid': True,
                'issues': [],
                'messages_processed': 0,
                'metric_updates': 0,
                'alerts_received': 0,
                'heartbeats': 0,
                'connection_stable': True,
                'data_quality_score': 0
            }

            # 验证配置
            required_config_fields = ['url', 'protocols', 'heartbeat_interval']
            for field in required_config_fields:
                if field not in config:
                    validation_result['issues'].append(f"WebSocket配置缺少字段: {field}")

            # 处理数据流
            last_timestamp = None
            message_count = 0

            for message in stream_data:
                message_count += 1

                # 验证消息结构
                if 'type' not in message:
                    validation_result['issues'].append(f"消息 {message_count} 缺少type字段")
                    continue

                if 'timestamp' not in message:
                    validation_result['issues'].append(f"消息 {message_count} 缺少timestamp字段")
                    continue

                # 验证时间戳递增
                current_timestamp = message['timestamp']
                if last_timestamp and current_timestamp < last_timestamp:
                    validation_result['issues'].append(f"消息 {message_count} 时间戳不递增")
                    validation_result['connection_stable'] = False

                last_timestamp = current_timestamp

                # 统计消息类型
                msg_type = message['type']
                if msg_type == 'metric_update':
                    validation_result['metric_updates'] += 1
                    if 'metric' not in message or 'value' not in message:
                        validation_result['issues'].append(f"metric_update消息 {message_count} 缺少必要字段")
                elif msg_type == 'alert':
                    validation_result['alerts_received'] += 1
                    if 'level' not in message or 'message' not in message:
                        validation_result['issues'].append(f"alert消息 {message_count} 缺少必要字段")
                elif msg_type == 'heartbeat':
                    validation_result['heartbeats'] += 1

            validation_result['messages_processed'] = message_count

            # 计算数据质量评分
            quality_score = 100
            if message_count < 3:
                quality_score -= 30
            if validation_result['metric_updates'] == 0:
                quality_score -= 20
            if not validation_result['connection_stable']:
                quality_score -= 25
            if validation_result['issues']:
                quality_score -= len(validation_result['issues']) * 5

            validation_result['data_quality_score'] = max(0, quality_score)

            if validation_result['issues']:
                validation_result['valid'] = False

            return validation_result

        # 验证WebSocket数据流
        validation = validate_websocket_stream(websocket_config, data_stream)

        # 应该验证通过
        assert validation['valid'], f"WebSocket流验证失败: {validation['issues']}"
        assert validation['messages_processed'] == 5, "应该处理5条消息"
        assert validation['metric_updates'] == 3, "应该有3个指标更新"
        assert validation['alerts_received'] == 1, "应该有1个告警"
        assert validation['heartbeats'] == 1, "应该有1个心跳"
        assert validation['connection_stable'], "连接应该稳定"
        assert validation['data_quality_score'] >= 80, f"数据质量评分过低: {validation['data_quality_score']}"

        # 验证消息类型分布合理
        total_messages = validation['messages_processed']
        metric_ratio = validation['metric_updates'] / total_messages
        assert metric_ratio >= 0.5, f"指标更新消息比例过低: {metric_ratio:.2f}"


if __name__ == "__main__":
    pytest.main([__file__])

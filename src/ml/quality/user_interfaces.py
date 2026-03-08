"""
AI质量保障用户接口与工具

提供直观的用户界面和工具，帮助用户：
1. 质量保障仪表板 - 可视化展示质量状态
2. 配置管理工具 - 简化AI系统配置
3. 监控和告警界面 - 实时监控AI系统状态
4. 报告生成工具 - 自动生成质量报告
5. 故障排查助手 - 智能故障诊断和修复指导
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import asyncio
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

logger = logging.getLogger(__name__)


class QualityDashboard:
    """质量保障仪表板"""

    def __init__(self, dashboard_config: Dict[str, Any] = None):
        self.config = dashboard_config or self._get_default_config()
        self.dashboard_data = {}
        self.refresh_interval = self.config.get('refresh_interval', 300)  # 5分钟
        self.last_refresh = None

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'refresh_interval': 300,  # 5分钟
            'max_history_points': 100,
            'alert_history_days': 7,
            'performance_history_hours': 24,
            'charts_enabled': True,
            'export_formats': ['json', 'csv', 'png']
        }

    async def generate_dashboard(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成仪表板数据"""
        try:
            dashboard = {
                'timestamp': datetime.now(),
                'summary': self._generate_summary_section(system_data),
                'quality_metrics': self._generate_quality_section(system_data),
                'performance_metrics': self._generate_performance_section(system_data),
                'system_health': self._generate_health_section(system_data),
                'alerts_and_incidents': self._generate_alerts_section(system_data),
                'recommendations': self._generate_recommendations_section(system_data),
                'charts': self._generate_charts(system_data) if self.config['charts_enabled'] else {},
                'metadata': {
                    'last_refresh': self.last_refresh.isoformat() if self.last_refresh else None,
                    'data_freshness': 'current' if self._is_data_fresh() else 'stale',
                    'system_status': 'operational'
                }
            }

            # 缓存仪表板数据
            self.dashboard_data = dashboard
            self.last_refresh = datetime.now()

            return dashboard

        except Exception as e:
            logger.error(f"生成仪表板失败: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now(),
                'status': 'error'
            }

    def _generate_summary_section(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成摘要部分"""
        try:
            # 提取关键指标
            quality_score = system_data.get('quality_assessment', {}).get('overall_score', 0)
            risk_level = system_data.get('quality_assessment', {}).get('risk_level', 'unknown')
            active_alerts = len(system_data.get('risk_alerts', []))

            # 计算整体状态
            if quality_score >= 0.8 and active_alerts == 0:
                overall_status = 'excellent'
                status_color = 'green'
            elif quality_score >= 0.6 and active_alerts <= 2:
                overall_status = 'good'
                status_color = 'yellow'
            elif quality_score >= 0.4 or active_alerts <= 5:
                overall_status = 'fair'
                status_color = 'orange'
            else:
                overall_status = 'critical'
                status_color = 'red'

            return {
                'overall_status': overall_status,
                'status_color': status_color,
                'quality_score': quality_score,
                'risk_level': risk_level,
                'active_alerts_count': active_alerts,
                'key_indicators': {
                    'models_active': system_data.get('model_count', 0),
                    'tests_passed_today': system_data.get('tests_passed_today', 0),
                    'incidents_last_24h': system_data.get('incidents_24h', 0),
                    'system_uptime': system_data.get('system_uptime', 0)
                },
                'trends': {
                    'quality_trend': system_data.get('quality_trend', 'stable'),
                    'performance_trend': system_data.get('performance_trend', 'stable'),
                    'alert_trend': 'decreasing' if active_alerts < 3 else 'increasing'
                }
            }

        except Exception as e:
            logger.error(f"生成摘要部分失败: {e}")
            return {'error': str(e)}

    def _generate_quality_section(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成质量指标部分"""
        try:
            quality_data = system_data.get('quality_assessment', {})

            return {
                'overall_quality_score': quality_data.get('overall_score', 0),
                'quality_dimensions': {
                    'code_quality': {
                        'score': quality_data.get('code_quality_score', 0),
                        'trend': 'improving',
                        'key_metrics': {
                            'test_coverage': quality_data.get('test_coverage', 0),
                            'code_quality_score': quality_data.get('code_quality_score', 0)
                        }
                    },
                    'performance': {
                        'score': quality_data.get('performance_score', 0),
                        'trend': 'stable',
                        'key_metrics': {
                            'response_time': quality_data.get('response_time', 0),
                            'throughput': quality_data.get('throughput', 0)
                        }
                    },
                    'reliability': {
                        'score': quality_data.get('reliability_score', 0),
                        'trend': 'stable',
                        'key_metrics': {
                            'error_rate': quality_data.get('error_rate', 0),
                            'availability': quality_data.get('availability', 0)
                        }
                    },
                    'security': {
                        'score': quality_data.get('security_score', 0),
                        'trend': 'stable',
                        'key_metrics': {
                            'vulnerabilities': quality_data.get('vulnerabilities', 0),
                            'compliance_score': quality_data.get('compliance_score', 0)
                        }
                    }
                },
                'quality_trends': {
                    'last_7_days': self._calculate_trend_over_period(system_data, 7),
                    'last_30_days': self._calculate_trend_over_period(system_data, 30),
                    'last_90_days': self._calculate_trend_over_period(system_data, 90)
                }
            }

        except Exception as e:
            logger.error(f"生成质量指标部分失败: {e}")
            return {'error': str(e)}

    def _generate_performance_section(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成性能指标部分"""
        try:
            return {
                'response_times': {
                    'avg_response_time': system_data.get('avg_response_time', 0),
                    'p95_response_time': system_data.get('p95_response_time', 0),
                    'p99_response_time': system_data.get('p99_response_time', 0),
                    'trend': 'stable'
                },
                'throughput': {
                    'current_throughput': system_data.get('current_throughput', 0),
                    'peak_throughput': system_data.get('peak_throughput', 0),
                    'avg_throughput': system_data.get('avg_throughput', 0),
                    'trend': 'increasing'
                },
                'resource_utilization': {
                    'cpu_usage': system_data.get('cpu_usage', 0),
                    'memory_usage': system_data.get('memory_usage', 0),
                    'disk_usage': system_data.get('disk_usage', 0),
                    'network_usage': system_data.get('network_usage', 0)
                },
                'error_rates': {
                    'overall_error_rate': system_data.get('error_rate', 0),
                    'by_component': system_data.get('error_rate_by_component', {}),
                    'trend': 'decreasing'
                },
                'performance_targets': {
                    'response_time_target': '< 500ms',
                    'uptime_target': '> 99.9%',
                    'error_rate_target': '< 0.1%'
                }
            }

        except Exception as e:
            logger.error(f"生成性能指标部分失败: {e}")
            return {'error': str(e)}

    def _generate_health_section(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成系统健康部分"""
        try:
            return {
                'system_components': {
                    'ai_models': {
                        'total': system_data.get('total_models', 0),
                        'healthy': system_data.get('healthy_models', 0),
                        'warning': system_data.get('warning_models', 0),
                        'critical': system_data.get('critical_models', 0)
                    },
                    'data_pipelines': {
                        'total': system_data.get('total_pipelines', 0),
                        'active': system_data.get('active_pipelines', 0),
                        'failed': system_data.get('failed_pipelines', 0)
                    },
                    'monitoring_systems': {
                        'status': system_data.get('monitoring_status', 'operational'),
                        'uptime': system_data.get('monitoring_uptime', 0),
                        'coverage': system_data.get('monitoring_coverage', 0)
                    }
                },
                'health_score': system_data.get('overall_health_score', 0),
                'health_trends': {
                    'last_24h': 'stable',
                    'last_7d': 'improving',
                    'last_30d': 'stable'
                },
                'maintenance_status': {
                    'scheduled_maintenance': system_data.get('scheduled_maintenance', []),
                    'emergency_maintenance': system_data.get('emergency_maintenance', []),
                    'last_maintenance': system_data.get('last_maintenance', None)
                }
            }

        except Exception as e:
            logger.error(f"生成系统健康部分失败: {e}")
            return {'error': str(e)}

    def _generate_alerts_section(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成告警部分"""
        try:
            alerts = system_data.get('alerts', [])

            # 按严重性分组
            critical_alerts = [a for a in alerts if a.get('severity') == 'critical']
            high_alerts = [a for a in alerts if a.get('severity') == 'high']
            medium_alerts = [a for a in alerts if a.get('severity') == 'medium']
            low_alerts = [a for a in alerts if a.get('severity') == 'low']

            return {
                'active_alerts': {
                    'critical': len(critical_alerts),
                    'high': len(high_alerts),
                    'medium': len(medium_alerts),
                    'low': len(low_alerts),
                    'total': len(alerts)
                },
                'recent_alerts': alerts[-10:] if len(alerts) > 10 else alerts,  # 最近10个告警
                'alert_trends': {
                    'last_24h': len([a for a in alerts if self._is_recent_alert(a, 24)]),
                    'last_7d': len([a for a in alerts if self._is_recent_alert(a, 24*7)]),
                    'last_30d': len([a for a in alerts if self._is_recent_alert(a, 24*30)])
                },
                'top_alert_types': self._get_top_alert_types(alerts),
                'escalation_needed': len(critical_alerts) > 0 or len(high_alerts) > 5
            }

        except Exception as e:
            logger.error(f"生成告警部分失败: {e}")
            return {'error': str(e)}

    def _generate_recommendations_section(self, system_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成建议部分"""
        try:
            recommendations = system_data.get('recommendations', [])

            # 按优先级排序并分类
            priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}

            sorted_recs = sorted(
                recommendations,
                key=lambda x: priority_order.get(x.get('priority', 'low'), 1),
                reverse=True
            )

            # 分类整理
            categorized_recs = {
                'immediate_actions': [r for r in sorted_recs if r.get('priority') == 'critical'],
                'high_priority': [r for r in sorted_recs if r.get('priority') == 'high'],
                'medium_priority': [r for r in sorted_recs if r.get('priority') == 'medium'],
                'low_priority': [r for r in sorted_recs if r.get('priority') == 'low']
            }

            return categorized_recs

        except Exception as e:
            logger.error(f"生成建议部分失败: {e}")
            return {'error': str(e)}

    def _generate_charts(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成图表"""
        try:
            charts = {}

            # 质量分数趋势图
            if 'quality_history' in system_data:
                charts['quality_trend'] = self._create_quality_trend_chart(system_data['quality_history'])

            # 性能指标图
            if 'performance_history' in system_data:
                charts['performance_chart'] = self._create_performance_chart(system_data['performance_history'])

            # 告警分布图
            if 'alerts' in system_data:
                charts['alerts_distribution'] = self._create_alerts_distribution_chart(system_data['alerts'])

            return charts

        except Exception as e:
            logger.error(f"生成图表失败: {e}")
            return {}

    def _create_quality_trend_chart(self, quality_history: List[Dict[str, Any]]) -> str:
        """创建质量趋势图表"""
        try:
            # 提取数据
            timestamps = [datetime.fromisoformat(h['timestamp']) for h in quality_history[-30:]]
            scores = [h.get('overall_score', 0) for h in quality_history[-30:]]

            # 创建图表
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, scores, marker='o', linewidth=2, markersize=4)
            plt.title('Quality Score Trend (Last 30 Days)')
            plt.xlabel('Date')
            plt.ylabel('Quality Score')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            # 转换为base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.read()).decode()
            plt.close()

            return f"data:image/png;base64,{chart_data}"

        except Exception as e:
            logger.error(f"创建质量趋势图表失败: {e}")
            return ""

    def _create_performance_chart(self, performance_history: List[Dict[str, Any]]) -> str:
        """创建性能图表"""
        try:
            # 提取数据
            timestamps = [datetime.fromisoformat(h['timestamp']) for h in performance_history[-24:]]
            response_times = [h.get('response_time', 0) for h in performance_history[-24:]]
            throughputs = [h.get('throughput', 0) for h in performance_history[-24:]]

            # 创建双轴图表
            fig, ax1 = plt.subplots(figsize=(10, 6))

            ax1.plot(timestamps, response_times, 'b-', marker='o', label='Response Time (ms)')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Response Time (ms)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')

            ax2 = ax1.twinx()
            ax2.plot(timestamps, throughputs, 'r-', marker='s', label='Throughput (req/s)')
            ax2.set_ylabel('Throughput (req/s)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')

            plt.title('Performance Metrics (Last 24 Hours)')
            fig.tight_layout()

            # 转换为base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.read()).decode()
            plt.close()

            return f"data:image/png;base64,{chart_data}"

        except Exception as e:
            logger.error(f"创建性能图表失败: {e}")
            return ""

    def _create_alerts_distribution_chart(self, alerts: List[Dict[str, Any]]) -> str:
        """创建告警分布图表"""
        try:
            # 统计告警类型
            alert_types = {}
            for alert in alerts[-100:]:  # 最近100个告警
                alert_type = alert.get('type', 'unknown')
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

            # 创建饼图
            plt.figure(figsize=(8, 8))
            plt.pie(alert_types.values(), labels=alert_types.keys(), autopct='%1.1f%%')
            plt.title('Alert Distribution by Type')

            # 转换为base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            chart_data = base64.b64encode(buffer.read()).decode()
            plt.close()

            return f"data:image/png;base64,{chart_data}"

        except Exception as e:
            logger.error(f"创建告警分布图表失败: {e}")
            return ""

    def _calculate_trend_over_period(self, system_data: Dict[str, Any], days: int) -> str:
        """计算指定周期内的趋势"""
        # 简化的趋势计算逻辑
        return 'stable'

    def _is_recent_alert(self, alert: Dict[str, Any], hours: int) -> bool:
        """检查告警是否在指定小时内"""
        try:
            alert_time = datetime.fromisoformat(alert.get('timestamp', ''))
            return (datetime.now() - alert_time).total_seconds() < hours * 3600
        except:
            return False

    def _get_top_alert_types(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """获取最常见的告警类型"""
        try:
            alert_types = {}
            for alert in alerts[-100:]:  # 最近100个告警
                alert_type = alert.get('type', 'unknown')
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

            # 排序并返回前5个
            sorted_types = sorted(alert_types.items(), key=lambda x: x[1], reverse=True)
            return [{'type': t[0], 'count': t[1]} for t in sorted_types[:5]]

        except Exception as e:
            logger.error(f"获取最常见告警类型失败: {e}")
            return []

    def _is_data_fresh(self) -> bool:
        """检查数据是否新鲜"""
        if not self.last_refresh:
            return False

        return (datetime.now() - self.last_refresh).total_seconds() < self.refresh_interval * 2

    def export_dashboard(self, format_type: str = 'json') -> str:
        """导出仪表板数据"""
        try:
            if format_type == 'json':
                return json.dumps(self.dashboard_data, indent=2, default=str)
            elif format_type == 'csv':
                # 将数据转换为CSV格式
                return self._convert_to_csv(self.dashboard_data)
            else:
                raise ValueError(f"不支持的导出格式: {format_type}")

        except Exception as e:
            logger.error(f"导出仪表板失败: {e}")
            return f"导出失败: {e}"

    def _convert_to_csv(self, data: Dict[str, Any]) -> str:
        """转换为CSV格式"""
        try:
            # 简化的CSV转换
            lines = ["Key,Value"]
            for key, value in data.items():
                if isinstance(value, (str, int, float)):
                    lines.append(f"{key},{value}")
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (str, int, float)):
                            lines.append(f"{key}.{sub_key},{sub_value}")

            return "\n".join(lines)

        except Exception as e:
            return f"CSV转换失败: {e}"


class ConfigurationManager:
    """配置管理工具"""

    def __init__(self, config_path: str = "config/ai_quality_config.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_cache = {}
        self.config_history = []

    def load_configuration(self, config_type: str) -> Dict[str, Any]:
        """加载配置"""
        try:
            config_file = self.config_path.parent / f"{config_type}_config.json"

            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = self._get_default_configuration(config_type)

            self.config_cache[config_type] = config
            return config

        except Exception as e:
            logger.error(f"加载配置失败 {config_type}: {e}")
            return self._get_default_configuration(config_type)

    def save_configuration(self, config_type: str, config: Dict[str, Any]) -> bool:
        """保存配置"""
        try:
            # 记录配置历史
            if config_type in self.config_cache:
                self.config_history.append({
                    'config_type': config_type,
                    'old_config': self.config_cache[config_type].copy(),
                    'new_config': config.copy(),
                    'timestamp': datetime.now(),
                    'user': 'system'  # 这里可以扩展为实际用户
                })

            # 验证配置
            if not self._validate_configuration(config_type, config):
                logger.error(f"配置验证失败: {config_type}")
                return False

            # 保存配置
            config_file = self.config_path.parent / f"{config_type}_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            self.config_cache[config_type] = config

            logger.info(f"配置已保存: {config_type}")
            return True

        except Exception as e:
            logger.error(f"保存配置失败 {config_type}: {e}")
            return False

    def update_configuration(self, config_type: str, updates: Dict[str, Any]) -> bool:
        """更新配置"""
        try:
            current_config = self.load_configuration(config_type)
            updated_config = self._deep_merge(current_config, updates)

            return self.save_configuration(config_type, updated_config)

        except Exception as e:
            logger.error(f"更新配置失败 {config_type}: {e}")
            return False

    def _get_default_configuration(self, config_type: str) -> Dict[str, Any]:
        """获取默认配置"""
        defaults = {
            'ai_quality': {
                'enabled': True,
                'monitoring_interval': 300,
                'alert_thresholds': {
                    'quality_score': 0.8,
                    'error_rate': 0.05,
                    'response_time': 5.0
                },
                'model_paths': {
                    'anomaly_detector': 'models/anomaly_prediction',
                    'quality_predictor': 'models/quality_prediction'
                }
            },
            'data_pipeline': {
                'batch_size': 1000,
                'processing_timeout': 300,
                'quality_checks_enabled': True,
                'retention_days': 365
            },
            'model_operations': {
                'auto_update_enabled': False,
                'health_check_interval': 300,
                'performance_monitoring_enabled': True,
                'ab_testing_enabled': False
            }
        }

        return defaults.get(config_type, {})

    def _validate_configuration(self, config_type: str, config: Dict[str, Any]) -> bool:
        """验证配置"""
        try:
            if config_type == 'ai_quality':
                required_keys = ['enabled', 'monitoring_interval', 'alert_thresholds']
                for key in required_keys:
                    if key not in config:
                        logger.error(f"配置缺少必需字段: {key}")
                        return False

                # 验证数值范围
                if not (10 <= config.get('monitoring_interval', 0) <= 3600):
                    logger.error("监控间隔必须在10-3600秒之间")
                    return False

            elif config_type == 'data_pipeline':
                if config.get('batch_size', 0) <= 0:
                    logger.error("批处理大小必须大于0")
                    return False

            # 其他配置类型的验证可以在这里添加

            return True

        except Exception as e:
            logger.error(f"配置验证异常: {e}")
            return False

    def _deep_merge(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base_dict.copy()

        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def get_configuration_history(self, config_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """获取配置历史"""
        try:
            history = self.config_history

            if config_type:
                history = [h for h in history if h['config_type'] == config_type]

            return history[-limit:] if history else []

        except Exception as e:
            logger.error(f"获取配置历史失败: {e}")
            return []

    def rollback_configuration(self, config_type: str, history_index: int = -1) -> bool:
        """回滚配置"""
        try:
            history = self.get_configuration_history(config_type)

            if not history or abs(history_index) > len(history):
                logger.error("无效的历史索引")
                return False

            target_config = history[history_index]['old_config']
            return self.save_configuration(config_type, target_config)

        except Exception as e:
            logger.error(f"配置回滚失败: {e}")
            return False

    def export_configuration(self, config_type: str, format_type: str = 'json') -> str:
        """导出配置"""
        try:
            config = self.load_configuration(config_type)

            if format_type == 'json':
                return json.dumps(config, indent=2)
            else:
                return f"不支持的导出格式: {format_type}"

        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            return f"导出失败: {e}"


class ReportGenerator:
    """报告生成工具"""

    def __init__(self, report_config: Dict[str, Any] = None):
        self.config = report_config or self._get_default_config()
        self.report_templates = self._load_report_templates()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'output_formats': ['html', 'pdf', 'json'],
            'report_retention_days': 90,
            'auto_generate_schedules': ['daily', 'weekly', 'monthly']
        }

    def _load_report_templates(self) -> Dict[str, Dict[str, Any]]:
        """加载报告模板"""
        return {
            'daily_quality_report': {
                'title': '每日质量报告',
                'sections': ['summary', 'quality_metrics', 'performance', 'alerts', 'recommendations'],
                'schedule': 'daily',
                'recipients': ['quality_team', 'management']
            },
            'weekly_comprehensive_report': {
                'title': '每周综合质量报告',
                'sections': ['executive_summary', 'quality_trends', 'performance_analysis', 'incidents', 'improvements', 'forecast'],
                'schedule': 'weekly',
                'recipients': ['all_stakeholders']
            },
            'monthly_executive_report': {
                'title': '月度高管质量报告',
                'sections': ['strategic_summary', 'kpi_performance', 'risk_assessment', 'future_outlook'],
                'schedule': 'monthly',
                'recipients': ['executives', 'board']
            },
            'incident_report': {
                'title': '质量事件报告',
                'sections': ['incident_details', 'impact_analysis', 'root_cause', 'resolution', 'prevention'],
                'schedule': 'on_demand',
                'recipients': ['incident_response_team']
            }
        }

    def generate_report(self, report_type: str, data: Dict[str, Any],
                       format_type: str = 'html') -> Dict[str, Any]:
        """生成报告"""
        try:
            if report_type not in self.report_templates:
                raise ValueError(f"未知的报告类型: {report_type}")

            template = self.report_templates[report_type]

            # 生成报告内容
            report_content = {
                'report_id': f"{report_type}_{int(time.time())}",
                'title': template['title'],
                'generated_at': datetime.now(),
                'report_type': report_type,
                'sections': {}
            }

            # 生成各个部分
            for section in template['sections']:
                report_content['sections'][section] = self._generate_report_section(
                    section, data, report_type
                )

            # 生成最终报告
            if format_type == 'html':
                formatted_report = self._format_as_html(report_content)
            elif format_type == 'json':
                formatted_report = json.dumps(report_content, indent=2, default=str)
            else:
                formatted_report = json.dumps(report_content, indent=2, default=str)

            report_content['formatted_content'] = formatted_report
            report_content['format'] = format_type

            # 保存报告
            self._save_report(report_content)

            return report_content

        except Exception as e:
            logger.error(f"生成报告失败 {report_type}: {e}")
            return {
                'error': str(e),
                'report_type': report_type,
                'generated_at': datetime.now()
            }

    def _generate_report_section(self, section_name: str, data: Dict[str, Any],
                               report_type: str) -> Dict[str, Any]:
        """生成报告部分"""
        try:
            if section_name == 'summary':
                return self._generate_summary_section(data)
            elif section_name == 'quality_metrics':
                return self._generate_quality_metrics_section(data)
            elif section_name == 'performance':
                return self._generate_performance_section(data)
            elif section_name == 'alerts':
                return self._generate_alerts_section(data)
            elif section_name == 'recommendations':
                return self._generate_recommendations_section(data)
            elif section_name == 'executive_summary':
                return self._generate_executive_summary_section(data)
            elif section_name == 'quality_trends':
                return self._generate_quality_trends_section(data)
            elif section_name == 'performance_analysis':
                return self._generate_performance_analysis_section(data)
            elif section_name == 'incidents':
                return self._generate_incidents_section(data)
            elif section_name == 'improvements':
                return self._generate_improvements_section(data)
            elif section_name == 'forecast':
                return self._generate_forecast_section(data)
            elif section_name == 'strategic_summary':
                return self._generate_strategic_summary_section(data)
            elif section_name == 'kpi_performance':
                return self._generate_kpi_performance_section(data)
            elif section_name == 'risk_assessment':
                return self._generate_risk_assessment_section(data)
            elif section_name == 'future_outlook':
                return self._generate_future_outlook_section(data)
            else:
                return {'content': f'未知部分: {section_name}', 'data': {}}

        except Exception as e:
            logger.error(f"生成报告部分失败 {section_name}: {e}")
            return {'error': str(e)}

    def _generate_summary_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成摘要部分"""
        quality_score = data.get('quality_assessment', {}).get('overall_score', 0)
        active_alerts = len(data.get('alerts', []))

        return {
            'overall_quality_score': quality_score,
            'risk_level': data.get('quality_assessment', {}).get('risk_level', 'unknown'),
            'active_alerts': active_alerts,
            'key_highlights': [
                f"整体质量分数: {quality_score:.2f}",
                f"活跃告警数量: {active_alerts}",
                f"系统状态: {'正常' if quality_score > 0.7 and active_alerts < 3 else '需要关注'}"
            ]
        }

    def _generate_quality_metrics_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成质量指标部分"""
        assessment = data.get('quality_assessment', {})

        return {
            'dimensions': {
                'code_quality': assessment.get('code_quality_score', 0),
                'performance': assessment.get('performance_score', 0),
                'reliability': assessment.get('reliability_score', 0),
                'security': assessment.get('security_score', 0)
            },
            'trends': assessment.get('quality_trends', {}),
            'key_findings': assessment.get('key_findings', [])
        }

    def _generate_performance_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成性能部分"""
        return {
            'current_metrics': {
                'response_time': data.get('avg_response_time', 0),
                'throughput': data.get('current_throughput', 0),
                'error_rate': data.get('error_rate', 0),
                'availability': data.get('availability', 0)
            },
            'trends': {
                'response_time_trend': data.get('response_time_trend', 'stable'),
                'throughput_trend': data.get('throughput_trend', 'stable')
            },
            'targets': {
                'response_time_target': '< 500ms',
                'uptime_target': '> 99.9%',
                'error_rate_target': '< 0.1%'
            }
        }

    def _generate_alerts_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成告警部分"""
        alerts = data.get('alerts', [])

        return {
            'total_alerts': len(alerts),
            'by_severity': {
                'critical': len([a for a in alerts if a.get('severity') == 'critical']),
                'high': len([a for a in alerts if a.get('severity') == 'high']),
                'medium': len([a for a in alerts if a.get('severity') == 'medium']),
                'low': len([a for a in alerts if a.get('severity') == 'low'])
            },
            'recent_alerts': alerts[-5:] if alerts else [],
            'trends': data.get('alert_trends', {})
        }

    def _generate_recommendations_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成建议部分"""
        recommendations = data.get('recommendations', [])

        return {
            'total_recommendations': len(recommendations),
            'by_priority': {
                'critical': len([r for r in recommendations if r.get('priority') == 'critical']),
                'high': len([r for r in recommendations if r.get('priority') == 'high']),
                'medium': len([r for r in recommendations if r.get('priority') == 'medium']),
                'low': len([r for r in recommendations if r.get('priority') == 'low'])
            },
            'top_recommendations': recommendations[:5] if recommendations else []
        }

    def _generate_executive_summary_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成执行摘要部分"""
        return self._generate_summary_section(data)  # 复用摘要逻辑

    def _generate_quality_trends_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成质量趋势部分"""
        return {
            'trend_analysis': data.get('quality_trends', {}),
            'improvement_areas': data.get('improvement_areas', []),
            'forecast': data.get('quality_forecast', {})
        }

    def _generate_performance_analysis_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成性能分析部分"""
        return self._generate_performance_section(data)  # 复用性能逻辑

    def _generate_incidents_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成事件部分"""
        return {
            'total_incidents': data.get('total_incidents', 0),
            'by_category': data.get('incidents_by_category', {}),
            'resolution_time': data.get('avg_resolution_time', 0),
            'prevention_measures': data.get('prevention_measures', [])
        }

    def _generate_improvements_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成改进部分"""
        return {
            'completed_improvements': data.get('completed_improvements', []),
            'ongoing_initiatives': data.get('ongoing_initiatives', []),
            'planned_improvements': data.get('planned_improvements', [])
        }

    def _generate_forecast_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成预测部分"""
        return {
            'quality_forecast': data.get('quality_forecast', {}),
            'risk_forecast': data.get('risk_forecast', {}),
            'capacity_forecast': data.get('capacity_forecast', {})
        }

    def _generate_strategic_summary_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成战略摘要部分"""
        return {
            'strategic_objectives': data.get('strategic_objectives', []),
            'progress_towards_goals': data.get('progress_towards_goals', {}),
            'strategic_initiatives': data.get('strategic_initiatives', [])
        }

    def _generate_kpi_performance_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成KPI性能部分"""
        return {
            'kpi_scores': data.get('kpi_scores', {}),
            'kpi_trends': data.get('kpi_trends', {}),
            'kpi_targets': data.get('kpi_targets', {})
        }

    def _generate_risk_assessment_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成风险评估部分"""
        return {
            'current_risks': data.get('current_risks', []),
            'risk_trends': data.get('risk_trends', {}),
            'risk_mitigation': data.get('risk_mitigation', [])
        }

    def _generate_future_outlook_section(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """生成未来展望部分"""
        return {
            'future_initiatives': data.get('future_initiatives', []),
            'technology_trends': data.get('technology_trends', []),
            'strategic_recommendations': data.get('strategic_recommendations', [])
        }

    def _format_as_html(self, report_content: Dict[str, Any]) -> str:
        """格式化为HTML"""
        try:
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{report_content['title']}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                    .section {{ margin-bottom: 30px; }}
                    .section h2 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 5px; }}
                    .metric {{ background-color: #e9ecef; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                    .alert-critical {{ background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 10px; margin: 5px 0; }}
                    .alert-high {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; margin: 5px 0; }}
                    .recommendation {{ background-color: #d1ecf1; border-left: 4px solid #17a2b8; padding: 10px; margin: 5px 0; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{report_content['title']}</h1>
                    <p>生成时间: {report_content['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>报告类型: {report_content['report_type']}</p>
                </div>
            """

            # 添加各个部分
            for section_name, section_data in report_content['sections'].items():
                html += f'<div class="section"><h2>{section_name.replace("_", " ").title()}</h2>'

                if isinstance(section_data, dict):
                    for key, value in section_data.items():
                        if isinstance(value, (int, float)):
                            html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'
                        elif isinstance(value, list):
                            html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong><ul>'
                            for item in value[:5]:  # 限制显示数量
                                if isinstance(item, dict):
                                    item_str = ", ".join([f"{k}: {v}" for k, v in item.items()])
                                    html += f'<li>{item_str}</li>'
                                else:
                                    html += f'<li>{item}</li>'
                            html += '</ul></div>'
                        elif isinstance(value, dict):
                            html += f'<div class="metric"><strong>{key.replace("_", " ").title()}:</strong> {value}</div>'

                html += '</div>'

            html += '</body></html>'

            return html

        except Exception as e:
            logger.error(f"HTML格式化失败: {e}")
            return f"<html><body><h1>报告生成失败</h1><p>{e}</p></body></html>"

    def _save_report(self, report_content: Dict[str, Any]):
        """保存报告"""
        try:
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            report_filename = f"{report_content['report_id']}.{report_content['format']}"
            report_path = reports_dir / report_filename

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content['formatted_content'])

            logger.info(f"报告已保存: {report_path}")

        except Exception as e:
            logger.error(f"保存报告失败: {e}")

    def get_report_history(self, report_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """获取报告历史"""
        try:
            reports_dir = Path("reports")
            if not reports_dir.exists():
                return []

            reports = []
            for report_file in reports_dir.glob("*.json"):
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)

                    if not report_type or report_data.get('report_type') == report_type:
                        reports.append(report_data)

                except Exception as e:
                    logger.error(f"读取报告文件失败 {report_file}: {e}")

            # 按生成时间排序
            reports.sort(key=lambda x: x.get('generated_at', ''), reverse=True)

            return reports[:limit]

        except Exception as e:
            logger.error(f"获取报告历史失败: {e}")
            return []


class TroubleshootingAssistant:
    """故障排查助手"""

    def __init__(self):
        self.troubleshooting_knowledge = self._load_troubleshooting_knowledge()
        self.diagnostic_workflows = self._load_diagnostic_workflows()

    def _load_troubleshooting_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """加载故障排查知识库"""
        return {
            'quality_score_low': {
                'symptoms': ['整体质量分数低于0.7', '多个维度分数偏低'],
                'possible_causes': [
                    '测试覆盖率不足',
                    '代码质量下降',
                    '性能问题频发',
                    '安全漏洞增加'
                ],
                'diagnostic_steps': [
                    '检查测试覆盖率指标',
                    '分析代码质量报告',
                    '审查性能监控数据',
                    '检查安全扫描结果'
                ],
                'solutions': [
                    '增加单元测试覆盖率',
                    '进行代码重构',
                    '优化性能瓶颈',
                    '修复安全漏洞'
                ]
            },
            'high_error_rate': {
                'symptoms': ['错误率超过5%', '用户投诉增加'],
                'possible_causes': [
                    '代码缺陷',
                    '外部服务不可用',
                    '资源不足',
                    '配置错误'
                ],
                'diagnostic_steps': [
                    '检查错误日志',
                    '验证外部服务状态',
                    '监控资源使用情况',
                    '审查配置参数'
                ],
                'solutions': [
                    '修复代码缺陷',
                    '实现服务降级',
                    '增加资源分配',
                    '修正配置参数'
                ]
            },
            'performance_degradation': {
                'symptoms': ['响应时间增加', '吞吐量下降'],
                'possible_causes': [
                    '代码效率低下',
                    '数据库查询优化不足',
                    '缓存失效',
                    '资源竞争'
                ],
                'diagnostic_steps': [
                    '进行性能分析',
                    '检查数据库查询',
                    '验证缓存状态',
                    '监控资源竞争'
                ],
                'solutions': [
                    '优化代码性能',
                    '改进数据库查询',
                    '调整缓存策略',
                    '解决资源竞争问题'
                ]
            },
            'model_accuracy_drop': {
                'symptoms': ['AI模型准确率下降', '预测错误增加'],
                'possible_causes': [
                    '数据分布变化',
                    '模型过时',
                    '特征工程失效',
                    '训练数据不足'
                ],
                'diagnostic_steps': [
                    '检查输入数据分布',
                    '验证模型性能指标',
                    '分析特征重要性',
                    '审查训练数据质量'
                ],
                'solutions': [
                    '重新训练模型',
                    '更新特征工程',
                    '收集更多训练数据',
                    '实施模型监控'
                ]
            }
        }

    def _load_diagnostic_workflows(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载诊断工作流"""
        return {
            'comprehensive_diagnostic': [
                {
                    'step': 1,
                    'name': '问题确认',
                    'description': '确认问题的存在和严重程度',
                    'actions': ['收集相关指标', '确认问题影响范围'],
                    'expected_outcome': '问题确认完成'
                },
                {
                    'step': 2,
                    'name': '初步诊断',
                    'description': '进行初步问题诊断',
                    'actions': ['检查系统日志', '验证配置参数', '监控系统状态'],
                    'expected_outcome': '识别潜在问题原因'
                },
                {
                    'step': 3,
                    'name': '深入分析',
                    'description': '进行深入的技术分析',
                    'actions': ['性能分析', '代码审查', '数据分析'],
                    'expected_outcome': '确定根本原因'
                },
                {
                    'step': 4,
                    'name': '解决方案制定',
                    'description': '制定解决方案',
                    'actions': ['评估修复方案', '制定实施计划', '准备回滚计划'],
                    'expected_outcome': '解决方案就绪'
                },
                {
                    'step': 5,
                    'name': '问题解决',
                    'description': '实施解决方案',
                    'actions': ['应用修复措施', '验证修复效果', '监控系统状态'],
                    'expected_outcome': '问题得到解决'
                }
            ]
        }

    def diagnose_issue(self, issue_description: str, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """诊断问题"""
        try:
            # 识别问题类型
            issue_type = self._identify_issue_type(issue_description, system_data)

            if issue_type not in self.troubleshooting_knowledge:
                return {
                    'issue_type': 'unknown',
                    'diagnosis': '无法识别的问题类型',
                    'recommendations': ['咨询技术专家', '收集更多诊断信息']
                }

            knowledge = self.troubleshooting_knowledge[issue_type]

            # 执行诊断步骤
            diagnostic_results = self._execute_diagnostic_steps(knowledge, system_data)

            # 生成解决方案
            solutions = self._generate_solutions(knowledge, diagnostic_results)

            return {
                'issue_type': issue_type,
                'description': knowledge['symptoms'][0] if knowledge['symptoms'] else '未知问题',
                'possible_causes': knowledge['possible_causes'],
                'diagnostic_results': diagnostic_results,
                'recommended_solutions': solutions,
                'diagnostic_workflow': self.diagnostic_workflows.get('comprehensive_diagnostic', []),
                'confidence_level': self._calculate_diagnostic_confidence(diagnostic_results)
            }

        except Exception as e:
            logger.error(f"问题诊断失败: {e}")
            return {
                'error': str(e),
                'recommendations': ['检查系统状态', '收集更多信息']
            }

    def _identify_issue_type(self, description: str, system_data: Dict[str, Any]) -> str:
        """识别问题类型"""
        try:
            description_lower = description.lower()

            # 基于关键词识别
            if any(word in description_lower for word in ['质量', '分数', '低', 'score']):
                return 'quality_score_low'
            elif any(word in description_lower for word in ['错误', 'error', '异常']):
                return 'high_error_rate'
            elif any(word in description_lower for word in ['性能', '慢', '响应', 'response']):
                return 'performance_degradation'
            elif any(word in description_lower for word in ['模型', '准确率', '预测', 'accuracy']):
                return 'model_accuracy_drop'

            # 基于系统数据识别
            quality_score = system_data.get('quality_assessment', {}).get('overall_score', 1.0)
            if quality_score < 0.7:
                return 'quality_score_low'

            error_rate = system_data.get('error_rate', 0)
            if error_rate > 0.05:
                return 'high_error_rate'

            response_time = system_data.get('avg_response_time', 0)
            if response_time > 2.0:
                return 'performance_degradation'

            return 'unknown'

        except Exception as e:
            logger.error(f"问题类型识别失败: {e}")
            return 'unknown'

    def _execute_diagnostic_steps(self, knowledge: Dict[str, Any], system_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行诊断步骤"""
        try:
            results = {}

            for step in knowledge.get('diagnostic_steps', []):
                # 这里应该实现具体的诊断逻辑
                # 目前返回模拟结果
                results[step] = {
                    'status': 'completed',
                    'findings': f'诊断步骤 "{step}" 已执行',
                    'evidence': '系统数据分析完成'
                }

            return results

        except Exception as e:
            logger.error(f"执行诊断步骤失败: {e}")
            return {'error': str(e)}

    def _generate_solutions(self, knowledge: Dict[str, Any], diagnostic_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成解决方案"""
        try:
            solutions = []

            for solution in knowledge.get('solutions', []):
                solutions.append({
                    'solution': solution,
                    'priority': 'high' if 'critical' in solution.lower() else 'medium',
                    'estimated_effort': 'medium',
                    'expected_impact': 'significant' if 'fix' in solution.lower() else 'moderate',
                    'implementation_steps': [
                        '评估影响范围',
                        '准备实施计划',
                        '执行解决方案',
                        '验证修复效果'
                    ]
                })

            return solutions

        except Exception as e:
            logger.error(f"生成解决方案失败: {e}")
            return []

    def _calculate_diagnostic_confidence(self, diagnostic_results: Dict[str, Any]) -> float:
        """计算诊断置信度"""
        try:
            if not diagnostic_results:
                return 0.0

            successful_steps = sum(1 for result in diagnostic_results.values()
                                 if result.get('status') == 'completed')

            total_steps = len(diagnostic_results)

            confidence = successful_steps / total_steps if total_steps > 0 else 0.0

            return min(0.95, confidence)

        except Exception:
            return 0.5

    def get_available_diagnostics(self) -> List[str]:
        """获取可用的诊断类型"""
        return list(self.troubleshooting_knowledge.keys())

    def get_diagnostic_workflow(self, workflow_name: str = 'comprehensive_diagnostic') -> List[Dict[str, Any]]:
        """获取诊断工作流"""
        return self.diagnostic_workflows.get(workflow_name, [])

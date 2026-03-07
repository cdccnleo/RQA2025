#!/usr/bin/env python3
"""
监控增强脚本
扩展监控指标和告警规则
"""

import sys
import json
import psutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MonitoringEnhancer:
    """监控增强器"""

    def __init__(self):
        self.project_root = Path(project_root)
        self.report_dir = self.project_root / 'reports' / 'infrastructure'

        # 确保报告目录存在
        self.report_dir.mkdir(parents=True, exist_ok=True)

        self.enhancement_results = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': {},
            'new_metrics': {},
            'alert_rules': {},
            'performance_analysis': {},
            'recommendations': [],
            'generated_files': []
        }

    def analyze_current_monitoring(self) -> Dict[str, Any]:
        """分析当前监控状态"""
        print("分析当前监控状态...")

        current_metrics = {
            'system_metrics': self._get_system_metrics(),
            'application_metrics': self._get_application_metrics(),
            'business_metrics': self._get_business_metrics(),
            'infrastructure_metrics': self._get_infrastructure_metrics()
        }

        return current_metrics

    def _get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available': memory.available,
                'memory_total': memory.total,
                'disk_usage': disk.percent,
                'disk_free': disk.free,
                'disk_total': disk.total,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except Exception as e:
            print(f"获取系统指标失败: {e}")
            return {}

    def _get_application_metrics(self) -> Dict[str, Any]:
        """获取应用指标"""
        return {
            'process_count': len(psutil.pids()),
            'python_processes': len([p for p in psutil.process_iter(['pid', 'name']) if 'python' in p.info['name'].lower()]),
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.Process().cpu_percent(),
            'open_files': len(psutil.Process().open_files()),
            'threads': psutil.Process().num_threads()
        }

    def _get_business_metrics(self) -> Dict[str, Any]:
        """获取业务指标"""
        return {
            'active_connections': 0,  # 需要从实际应用中获取
            'requests_per_second': 0,
            'response_time_avg': 0,
            'error_rate': 0,
            'throughput': 0
        }

    def _get_infrastructure_metrics(self) -> Dict[str, Any]:
        """获取基础设施指标"""
        return {
            'config_files': len(list(self.project_root.rglob('*.json'))),
            'log_files': len(list(self.project_root.rglob('*.log'))),
            'test_files': len(list(self.project_root.rglob('test_*.py'))),
            'source_files': len(list(self.project_root.rglob('*.py'))),
            'documentation_files': len(list(self.project_root.rglob('*.md')))
        }

    def design_new_metrics(self) -> Dict[str, Any]:
        """设计新的监控指标"""
        print("设计新的监控指标...")

        new_metrics = {
            'performance_metrics': {
                'response_time_p95': '95%响应时间',
                'response_time_p99': '99%响应时间',
                'throughput_rps': '每秒请求数',
                'error_rate_percent': '错误率百分比',
                'availability_percent': '可用性百分比',
                'memory_leak_rate': '内存泄漏率',
                'gc_frequency': '垃圾回收频率',
                'thread_pool_utilization': '线程池利用率'
            },
            'security_metrics': {
                'failed_login_attempts': '失败登录尝试次数',
                'suspicious_activities': '可疑活动次数',
                'data_access_violations': '数据访问违规次数',
                'encryption_errors': '加密错误次数',
                'authentication_failures': '认证失败次数',
                'authorization_violations': '授权违规次数'
            },
            'business_metrics': {
                'active_users': '活跃用户数',
                'transaction_volume': '交易量',
                'profit_loss': '盈亏情况',
                'risk_exposure': '风险敞口',
                'compliance_score': '合规评分',
                'customer_satisfaction': '客户满意度'
            },
            'infrastructure_metrics': {
                'service_health_score': '服务健康评分',
                'dependency_health': '依赖服务健康状态',
                'resource_utilization': '资源利用率',
                'capacity_planning': '容量规划指标',
                'deployment_frequency': '部署频率',
                'rollback_rate': '回滚率'
            }
        }

        return new_metrics

    def design_alert_rules(self) -> Dict[str, Any]:
        """设计告警规则"""
        print("设计告警规则...")

        alert_rules = {
            'critical_alerts': {
                'high_cpu_usage': {
                    'condition': 'cpu_usage > 90',
                    'duration': '5m',
                    'severity': 'critical',
                    'action': '立即通知运维团队'
                },
                'high_memory_usage': {
                    'condition': 'memory_usage > 95',
                    'duration': '3m',
                    'severity': 'critical',
                    'action': '立即重启服务'
                },
                'service_down': {
                    'condition': 'service_health_score = 0',
                    'duration': '1m',
                    'severity': 'critical',
                    'action': '立即启动故障恢复流程'
                },
                'security_breach': {
                    'condition': 'suspicious_activities > 10',
                    'duration': '1m',
                    'severity': 'critical',
                    'action': '立即隔离系统并通知安全团队'
                }
            },
            'warning_alerts': {
                'moderate_cpu_usage': {
                    'condition': 'cpu_usage > 70',
                    'duration': '10m',
                    'severity': 'warning',
                    'action': '监控并准备扩容'
                },
                'moderate_memory_usage': {
                    'condition': 'memory_usage > 80',
                    'duration': '5m',
                    'severity': 'warning',
                    'action': '检查内存泄漏'
                },
                'high_error_rate': {
                    'condition': 'error_rate > 5',
                    'duration': '5m',
                    'severity': 'warning',
                    'action': '检查应用日志'
                },
                'low_availability': {
                    'condition': 'availability < 99.5',
                    'duration': '15m',
                    'severity': 'warning',
                    'action': '检查服务状态'
                }
            },
            'info_alerts': {
                'deployment_completed': {
                    'condition': 'deployment_status = success',
                    'duration': '1m',
                    'severity': 'info',
                    'action': '记录部署日志'
                },
                'backup_completed': {
                    'condition': 'backup_status = success',
                    'duration': '1m',
                    'severity': 'info',
                    'action': '记录备份日志'
                }
            }
        }

        return alert_rules

    def analyze_performance(self) -> Dict[str, Any]:
        """分析性能"""
        print("分析性能...")

        performance_analysis = {
            'current_performance': {
                'cpu_usage': self._get_system_metrics().get('cpu_usage', 0),
                'memory_usage': self._get_system_metrics().get('memory_usage', 0),
                'disk_usage': self._get_system_metrics().get('disk_usage', 0),
                'process_count': self._get_application_metrics().get('process_count', 0)
            },
            'performance_trends': {
                'cpu_trend': 'stable',
                'memory_trend': 'stable',
                'disk_trend': 'stable',
                'response_time_trend': 'stable'
            },
            'bottlenecks': [],
            'optimization_opportunities': []
        }

        # 识别瓶颈
        if performance_analysis['current_performance']['cpu_usage'] > 80:
            performance_analysis['bottlenecks'].append('CPU使用率过高')
            performance_analysis['optimization_opportunities'].append('考虑CPU扩容或优化算法')

        if performance_analysis['current_performance']['memory_usage'] > 85:
            performance_analysis['bottlenecks'].append('内存使用率过高')
            performance_analysis['optimization_opportunities'].append('检查内存泄漏或增加内存')

        if performance_analysis['current_performance']['disk_usage'] > 90:
            performance_analysis['bottlenecks'].append('磁盘使用率过高')
            performance_analysis['optimization_opportunities'].append('清理日志文件或扩容存储')

        return performance_analysis

    def generate_recommendations(self) -> List[str]:
        """生成监控增强建议"""
        print("生成监控增强建议...")

        recommendations = [
            "实现实时监控仪表板",
            "添加自定义业务指标",
            "完善告警通知机制",
            "建立监控数据存储策略",
            "实现监控数据可视化",
            "添加监控数据备份机制",
            "建立监控数据清理策略",
            "实现监控数据导出功能",
            "添加监控数据API接口",
            "建立监控数据安全策略"
        ]

        return recommendations

    def generate_monitoring_config(self) -> Dict[str, Any]:
        """生成监控配置"""
        print("生成监控配置...")

        monitoring_config = {
            'prometheus_config': {
                'global': {
                    'scrape_interval': '15s',
                    'evaluation_interval': '15s'
                },
                'rule_files': [
                    'rules/infrastructure_alerts.yml'
                ],
                'scrape_configs': [
                    {
                        'job_name': 'infrastructure',
                        'static_configs': [
                            {
                                'targets': ['localhost:8000']
                            }
                        ]
                    }
                ]
            },
            'grafana_dashboards': {
                'infrastructure_overview': {
                    'title': '基础设施概览',
                    'panels': [
                        'CPU使用率',
                        '内存使用率',
                        '磁盘使用率',
                        '网络流量',
                        '服务健康状态'
                    ]
                },
                'application_performance': {
                    'title': '应用性能',
                    'panels': [
                        '响应时间',
                        '吞吐量',
                        '错误率',
                        '活跃连接数'
                    ]
                }
            },
            'alertmanager_config': {
                'global': {
                    'smtp_smarthost': 'localhost:587',
                    'smtp_from': 'alertmanager@example.com'
                },
                'route': {
                    'group_by': ['alertname'],
                    'group_wait': '10s',
                    'group_interval': '10s',
                    'repeat_interval': '1h',
                    'receiver': 'web.hook'
                },
                'receivers': [
                    {
                        'name': 'web.hook',
                        'webhook_configs': [
                            {
                                'url': 'http://127.0.0.1:5001/'
                            }
                        ]
                    }
                ]
            }
        }

        return monitoring_config

    def save_enhancement_results(self):
        """保存增强结果"""
        print("保存监控增强结果...")

        # 保存监控配置
        monitoring_config_file = self.report_dir / 'monitoring_config.json'
        with open(monitoring_config_file, 'w', encoding='utf-8') as f:
            json.dump(self.enhancement_results['monitoring_config'],
                      f, ensure_ascii=False, indent=2)

        # 保存告警规则
        alert_rules_file = self.report_dir / 'alert_rules.json'
        with open(alert_rules_file, 'w', encoding='utf-8') as f:
            json.dump(self.enhancement_results['alert_rules'], f, ensure_ascii=False, indent=2)

        # 保存完整报告
        report_file = self.report_dir / 'monitoring_enhancement_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.enhancement_results, f, ensure_ascii=False, indent=2)

        self.enhancement_results['generated_files'].extend([
            str(monitoring_config_file),
            str(alert_rules_file),
            str(report_file)
        ])

        print(f"监控增强结果已保存到: {self.report_dir}")

    def run(self):
        """运行监控增强流程"""
        print("开始监控增强流程...")

        try:
            # 分析当前监控状态
            self.enhancement_results['current_metrics'] = self.analyze_current_monitoring()

            # 设计新的监控指标
            self.enhancement_results['new_metrics'] = self.design_new_metrics()

            # 设计告警规则
            self.enhancement_results['alert_rules'] = self.design_alert_rules()

            # 分析性能
            self.enhancement_results['performance_analysis'] = self.analyze_performance()

            # 生成建议
            self.enhancement_results['recommendations'] = self.generate_recommendations()

            # 生成监控配置
            self.enhancement_results['monitoring_config'] = self.generate_monitoring_config()

            # 保存结果
            self.save_enhancement_results()

            print("监控增强流程完成")

            # 输出摘要
            current_metrics = self.enhancement_results['current_metrics']
            new_metrics = self.enhancement_results['new_metrics']
            alert_rules = self.enhancement_results['alert_rules']
            performance = self.enhancement_results['performance_analysis']

            print(f"\n=== 监控增强报告 ===")
            print(f"当前系统指标: {len(current_metrics)}")
            print(f"新增监控指标: {sum(len(metrics) for metrics in new_metrics.values())}")
            print(f"告警规则: {sum(len(rules) for rules in alert_rules.values())}")
            print(f"性能瓶颈: {len(performance['bottlenecks'])}")
            print(f"优化机会: {len(performance['optimization_opportunities'])}")
            print(f"增强建议: {len(self.enhancement_results['recommendations'])}")

            # 输出当前系统状态
            system_metrics = current_metrics.get('system_metrics', {})
            print(f"\n当前系统状态:")
            print(f"CPU使用率: {system_metrics.get('cpu_usage', 0):.1f}%")
            print(f"内存使用率: {system_metrics.get('memory_usage', 0):.1f}%")
            print(f"磁盘使用率: {system_metrics.get('disk_usage', 0):.1f}%")

        except Exception as e:
            print(f"监控增强流程失败: {e}")
            raise


if __name__ == '__main__':
    enhancer = MonitoringEnhancer()
    enhancer.run()

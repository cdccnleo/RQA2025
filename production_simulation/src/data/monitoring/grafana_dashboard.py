#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 数据层Grafana仪表板管理

提供数据层专用的Grafana监控仪表板配置和管理，
实现可视化的数据层监控和告警展示。

设计模式：构建器模式 + 模板方法模式
职责：数据层监控仪表板配置生成和管理
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json


class PanelType(Enum):

    """面板类型"""
    GRAPH = "graph"
    SINGLSTAT = "singlestat"
    TABLE = "table"
    HEATMAP = "heatmap"
    BARGAUGE = "bargauge"
    GAUGE = "gauge"
    TEXT = "text"


class MetricType(Enum):

    """指标类型"""
    PROMETHEUS = "prometheus"
    ELASTICSEARCH = "elasticsearch"
    INFLUXDB = "influxdb"


@dataclass
class DashboardPanel:

    """仪表板面板"""
    id: int
    title: str
    type: PanelType
    grid_pos: Dict[str, int]
    targets: List[Dict[str, Any]]
    options: Dict[str, Any] = field(default_factory=dict)
    field_config: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    repeat: Optional[str] = None
    repeat_options: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        panel_dict = {
            'id': self.id,
            'title': self.title,
            'type': self.type.value,
            'gridPos': self.grid_pos,
            'targets': self.targets
        }

        if self.description:
            panel_dict['description'] = self.description

        if self.options:
            panel_dict['options'] = self.options

        if self.field_config:
            panel_dict['fieldConfig'] = self.field_config

        if self.repeat:
            panel_dict['repeat'] = self.repeat
            if self.repeat_options:
                panel_dict['repeatOptions'] = self.repeat_options

        return panel_dict


@dataclass
class DashboardTemplate:

    """仪表板模板"""
    name: str
    description: str
    panels: List[DashboardPanel]
    variables: List[Dict[str, Any]] = field(default_factory=list)
    time_settings: Dict[str, Any] = field(default_factory=lambda: {
        'from': 'now - 1h',
        'to': 'now'
    })
    refresh: str = '30s'
    tags: List[str] = field(default_factory=lambda: ['rqa2025', 'data - layer'])

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'dashboard': {
                'title': f'RQA2025 数据层监控 - {self.name}',
                'description': self.description,
                'tags': self.tags,
                'timezone': 'browser',
                'panels': [panel.to_dict() for panel in self.panels],
                'templating': {
                    'list': self.variables
                },
                'time': self.time_settings,
                'refresh': self.refresh,
                'schemaVersion': 27,
                'version': 1,
                'links': [],
                'annotations': {
                    'list': []
                }
            }
        }


class DataGrafanaDashboard:

    """
    数据层Grafana仪表板管理器

    提供完整的Grafana仪表板配置生成和管理：
    - 标准监控仪表板模板
    - 动态面板配置
    - 数据源集成
    - 仪表板部署管理
    """

    def __init__(self, data_source_type: MetricType = MetricType.PROMETHEUS,


                 data_source_name: str = 'Prometheus'):
        """
        初始化仪表板管理器

        Args:
            data_source_type: 数据源类型
            data_source_name: 数据源名称
        """
        self.data_source_type = data_source_type
        self.data_source_name = data_source_name
        self.templates: Dict[str, DashboardTemplate] = {}
        self.deployed_dashboards: Dict[str, Dict[str, Any]] = {}

        # 初始化标准模板
        self._initialize_standard_templates()

        self._log_operation('initialized', 'DataGrafanaDashboard', 'success')

    def _initialize_standard_templates(self):
        """初始化标准仪表板模板"""
        # 主要监控仪表板
        main_template = self._create_main_monitoring_dashboard()
        self.templates['main'] = main_template

        # 缓存性能仪表板
        cache_template = self._create_cache_performance_dashboard()
        self.templates['cache'] = cache_template

        # 数据质量仪表板
        quality_template = self._create_data_quality_dashboard()
        self.templates['quality'] = quality_template

        # 性能监控仪表板
        performance_template = self._create_performance_dashboard()
        self.templates['performance'] = performance_template

    def _create_main_monitoring_dashboard(self) -> DashboardTemplate:
        """创建主要监控仪表板"""
        panels = []

        # 面板1: 系统概览
        panels.append(DashboardPanel(
            id=1,
            title='系统概览',
            type=PanelType.TEXT,
            grid_pos={'h': 3, 'w': 24, 'x': 0, 'y': 0},
            targets=[],
            description='RQA2025数据层监控系统概览',
            options={
                'content': '# RQA2025 数据层监控概览\n\n'
                '该仪表板提供数据层的全面监控视图，包括缓存性能、数据质量、\n'
                '处理性能和告警统计等关键指标。\n\n'
                '**监控范围：**\n'
                '- 数据请求处理\n'
                '- 缓存性能\n'
                '- 数据质量\n'
                '- 错误统计\n'
                '- 告警监控',
                'mode': 'markdown'
            }
        ))

        # 面板2: 数据请求统计
        panels.append(DashboardPanel(
            id=2,
            title='数据请求统计',
            type=PanelType.GRAPH,
            grid_pos={'h': 8, 'w': 12, 'x': 0, 'y': 3},
            targets=[
                {
                    'expr': 'sum(rate(data_processing_count[5m])) by (operation)',
                    'legendFormat': '{{operation}} 请求数',
                    'refId': 'A'
                }
            ],
            description='数据处理请求统计，按操作类型分组'
        ))

        # 面板3: 缓存命中率
        panels.append(DashboardPanel(
            id=3,
            title='缓存命中率',
            type=PanelType.SINGLSTAT,
            grid_pos={'h': 8, 'w': 6, 'x': 12, 'y': 3},
            targets=[
                {
                    'expr': 'rate(data_cache_hits[5m]) / rate(data_cache_requests[5m]) * 100',
                    'refId': 'A'
                }
            ],
            options={
                'reduceOptions': {
                    'values': False,
                    'calcs': ['last'],
                    'fields': ''
                },
                'orientation': 'auto',
                'textMode': 'value_and_name',
                'colorMode': 'value',
                'graphMode': 'none',
                'justifyMode': 'auto'
            },
            field_config={
                'defaults': {
                    'mappings': [],
                    'thresholds': {
                        'mode': 'absolute',
                        'steps': [
                            {'color': 'red', 'value': None},
                            {'color': 'orange', 'value': 70},
                            {'color': 'green', 'value': 85}
                        ]
                    },
                    'unit': 'percent'
                }
            },
            description='缓存命中率百分比'
        ))

        # 面板4: 错误率统计
        panels.append(DashboardPanel(
            id=4,
            title='错误率统计',
            type=PanelType.GRAPH,
            grid_pos={'h': 8, 'w': 6, 'x': 18, 'y': 3},
            targets=[
                {
                    'expr': 'sum(rate(data_error_count[5m])) by (error_type) / sum(rate(data_processing_count[5m])) * 100',
                    'legendFormat': '{{error_type}} 错误率',
                    'refId': 'A'
                }
            ],
            description='各类型错误率统计'
        ))

        # 面板5: 数据质量指标
        panels.append(DashboardPanel(
            id=5,
            title='数据质量指标',
            type=PanelType.TABLE,
            grid_pos={'h': 8, 'w': 12, 'x': 0, 'y': 11},
            targets=[
                {
                    'expr': 'data_quality_completeness',
                    'legendFormat': '{{data_type}} 完整性',
                    'refId': 'A'
                },
                {
                    'expr': 'data_quality_accuracy',
                    'legendFormat': '{{data_type}} 准确性',
                    'refId': 'B'
                },
                {
                    'expr': 'data_quality_timeliness',
                    'legendFormat': '{{data_type}} 时效性',
                    'refId': 'C'
                }
            ],
            description='数据质量指标表格'
        ))

        # 面板6: 响应时间分布
        panels.append(DashboardPanel(
            id=6,
            title='响应时间分布',
            type=PanelType.HEATMAP,
            grid_pos={'h': 8, 'w': 12, 'x': 12, 'y': 11},
            targets=[
                {
                    'expr': 'data_processing_duration',
                    'legendFormat': '响应时间',
                    'refId': 'A'
                }
            ],
            description='数据处理响应时间分布热力图'
        ))

        # 面板7: 活跃告警
        panels.append(DashboardPanel(
            id=7,
            title='活跃告警',
            type=PanelType.TABLE,
            grid_pos={'h': 8, 'w': 24, 'x': 0, 'y': 19},
            targets=[
                {
                    'expr': 'ALERTS{alertstate="firing", alertname=~"data_.*"}',
                    'legendFormat': '{{alertname}}',
                    'refId': 'A'
                }
            ],
            description='当前活跃的数据层告警'
        ))

        return DashboardTemplate(
            name='主要监控',
            description='RQA2025数据层主要监控仪表板，提供核心指标和告警监控',
            panels=panels,
            variables=[
                {
                    'name': 'datasource',
                    'type': 'datasource',
                    'query': self.data_source_type.value,
                    'label': 'Data Source'
                },
                {
                    'name': 'data_type',
                    'type': 'custom',
                    'query': 'stock,bond,futures,options,forex,crypto,commodity',
                    'label': 'Data Type'
                }
            ]
        )

    def _create_cache_performance_dashboard(self) -> DashboardTemplate:
        """创建缓存性能仪表板"""
        panels = []

        # 缓存命中率趋势
        panels.append(DashboardPanel(
            id=1,
            title='缓存命中率趋势',
            type=PanelType.GRAPH,
            grid_pos={'h': 8, 'w': 24, 'x': 0, 'y': 0},
            targets=[
                {
                    'expr': 'rate(data_cache_hits[5m]) / rate(data_cache_requests[5m]) * 100',
                    'legendFormat': '缓存命中率 %',
                    'refId': 'A'
                }
            ]
        ))

        # 缓存操作统计
        panels.append(DashboardPanel(
            id=2,
            title='缓存操作统计',
            type=PanelType.BARGAUGE,
            grid_pos={'h': 8, 'w': 12, 'x': 0, 'y': 8},
            targets=[
                {
                    'expr': 'rate(data_cache_hits[5m])',
                    'legendFormat': '缓存命中',
                    'refId': 'A'
                },
                {
                    'expr': 'rate(data_cache_misses[5m])',
                    'legendFormat': '缓存未命中',
                    'refId': 'B'
                }
            ],
            options={
                'orientation': 'horizontal',
                'displayMode': 'basic',
                'showUnfilled': True
            }
        ))

        # 缓存大小监控
        panels.append(DashboardPanel(
            id=3,
            title='缓存大小监控',
            type=PanelType.GAUGE,
            grid_pos={'h': 8, 'w': 12, 'x': 12, 'y': 8},
            targets=[
                {
                    'expr': 'data_cache_size_bytes',
                    'refId': 'A'
                }
            ],
            field_config={
                'defaults': {
                    'unit': 'bytes',
                    'min': 0,
                    'max': 1073741824  # 1GB
                }
            }
        ))

        return DashboardTemplate(
            name='缓存性能',
            description='数据层缓存性能监控仪表板',
            panels=panels
        )

    def _create_data_quality_dashboard(self) -> DashboardTemplate:
        """创建数据质量仪表板"""
        panels = []

        # 数据质量总览
        panels.append(DashboardPanel(
            id=1,
            title='数据质量总览',
            type=PanelType.TABLE,
            grid_pos={'h': 8, 'w': 24, 'x': 0, 'y': 0},
            targets=[
                {
                    'expr': 'data_quality_overall',
                    'legendFormat': '{{data_type}} 综合质量',
                    'refId': 'A'
                },
                {
                    'expr': 'data_quality_completeness',
                    'legendFormat': '{{data_type}} 完整性',
                    'refId': 'B'
                },
                {
                    'expr': 'data_quality_accuracy',
                    'legendFormat': '{{data_type}} 准确性',
                    'refId': 'C'
                }
            ]
        ))

        # 质量趋势图
        panels.append(DashboardPanel(
            id=2,
            title='质量趋势图',
            type=PanelType.GRAPH,
            grid_pos={'h': 8, 'w': 24, 'x': 0, 'y': 8},
            targets=[
                {
                    'expr': 'data_quality_overall',
                    'legendFormat': '{{data_type}} 综合质量',
                    'refId': 'A'
                }
            ]
        ))

        return DashboardTemplate(
            name='数据质量',
            description='数据质量监控仪表板',
            panels=panels
        )

    def _create_performance_dashboard(self) -> DashboardTemplate:
        """创建性能监控仪表板"""
        panels = []

        # 响应时间监控
        panels.append(DashboardPanel(
            id=1,
            title='响应时间监控',
            type=PanelType.GRAPH,
            grid_pos={'h': 8, 'w': 12, 'x': 0, 'y': 0},
            targets=[
                {
                    'expr': 'histogram_quantile(0.95, rate(data_processing_duration_bucket[5m]))',
                    'legendFormat': 'P95 响应时间',
                    'refId': 'A'
                },
                {
                    'expr': 'histogram_quantile(0.50, rate(data_processing_duration_bucket[5m]))',
                    'legendFormat': 'P50 响应时间',
                    'refId': 'B'
                }
            ]
        ))

        # 吞吐量监控
        panels.append(DashboardPanel(
            id=2,
            title='吞吐量监控',
            type=PanelType.GRAPH,
            grid_pos={'h': 8, 'w': 12, 'x': 12, 'y': 0},
            targets=[
                {
                    'expr': 'rate(data_processing_count[5m])',
                    'legendFormat': '{{operation}} TPS',
                    'refId': 'A'
                }
            ]
        ))

        return DashboardTemplate(
            name='性能监控',
            description='数据层性能监控仪表板',
            panels=panels
        )

    # =========================================================================
    # 仪表板管理
    # =========================================================================

    def create_dashboard(self, template_name: str, custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        创建仪表板

        Args:
            template_name: 模板名称
            custom_config: 自定义配置

        Returns:
            仪表板配置
        """
        if template_name not in self.templates:
            raise ValueError(f"模板不存在: {template_name}")

        template = self.templates[template_name]

        # 应用自定义配置
        if custom_config:
            dashboard_config = self._apply_custom_config(template.to_dict(), custom_config)
        else:
            dashboard_config = template.to_dict()

        return dashboard_config

    def create_custom_dashboard(self, title: str, description: str,


                                panels: List[DashboardPanel],
                                variables: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        创建自定义仪表板

        Args:
            title: 仪表板标题
            description: 描述
            panels: 面板列表
            variables: 变量列表

        Returns:
            仪表板配置
        """
        template = DashboardTemplate(
            name=title,
            description=description,
            panels=panels,
            variables=variables or []
        )

        return template.to_dict()

    def deploy_dashboard(self, dashboard_config: Dict[str, Any], folder_name: str = 'RQA2025') -> bool:
        """
        部署仪表板到Grafana

        Args:
            dashboard_config: 仪表板配置
            folder_name: 文件夹名称

        Returns:
            是否部署成功
        """
        try:
            # 这里应该实现实际的Grafana API调用
            # 目前只是记录部署信息

            dashboard_title = dashboard_config['dashboard']['title']
            self.deployed_dashboards[dashboard_title] = {
                'config': dashboard_config,
                'folder': folder_name,
                'deployed_at': datetime.now().isoformat()
            }

            self._log_operation('deploy_dashboard', dashboard_title, 'success')
            return True

        except Exception as e:
            self._log_operation('deploy_dashboard', 'unknown', f'failed: {e}')
            return False

    def update_dashboard(self, dashboard_title: str, new_config: Dict[str, Any]) -> bool:
        """
        更新仪表板

        Args:
            dashboard_title: 仪表板标题
            new_config: 新配置

        Returns:
            是否更新成功
        """
        try:
            if dashboard_title not in self.deployed_dashboards:
                raise ValueError(f"仪表板不存在: {dashboard_title}")

            self.deployed_dashboards[dashboard_title]['config'] = new_config
            self.deployed_dashboards[dashboard_title]['updated_at'] = datetime.now().isoformat()

            self._log_operation('update_dashboard', dashboard_title, 'success')
            return True

        except Exception as e:
            self._log_operation('update_dashboard', dashboard_title, f'failed: {e}')
            return False

    def delete_dashboard(self, dashboard_title: str) -> bool:
        """
        删除仪表板

        Args:
            dashboard_title: 仪表板标题

        Returns:
            是否删除成功
        """
        try:
            if dashboard_title not in self.deployed_dashboards:
                raise ValueError(f"仪表板不存在: {dashboard_title}")

            del self.deployed_dashboards[dashboard_title]

            self._log_operation('delete_dashboard', dashboard_title, 'success')
            return True

        except Exception as e:
            self._log_operation('delete_dashboard', dashboard_title, f'failed: {e}')
            return False

    # =========================================================================
    # 面板管理
    # =========================================================================

    def add_panel_to_template(self, template_name: str, panel: DashboardPanel) -> bool:
        """
        向模板添加面板

        Args:
            template_name: 模板名称
            panel: 面板配置

        Returns:
            是否添加成功
        """
        try:
            if template_name not in self.templates:
                raise ValueError(f"模板不存在: {template_name}")

            self.templates[template_name].panels.append(panel)
            self._log_operation('add_panel', f"{template_name}:{panel.title}", 'success')
            return True

        except Exception as e:
            self._log_operation('add_panel', template_name, f'failed: {e}')
            return False

    def create_metric_panel(self, title: str, metric_expr: str,


                            panel_type: PanelType = PanelType.GRAPH,
                            grid_pos: Optional[Dict[str, int]] = None) -> DashboardPanel:
        """
        创建指标面板

        Args:
            title: 面板标题
            metric_expr: 指标表达式
            panel_type: 面板类型
            grid_pos: 网格位置

        Returns:
            面板配置
        """
        if grid_pos is None:
            grid_pos = {'h': 8, 'w': 12, 'x': 0, 'y': 0}

        # 生成面板ID
        panel_id = len(self.templates.get('main', DashboardTemplate('', '', [])).panels) + 1

        return DashboardPanel(
            id=panel_id,
            title=title,
            type=panel_type,
            grid_pos=grid_pos,
            targets=[
                {
                    'expr': metric_expr,
                    'legendFormat': title,
                    'refId': 'A'
                }
            ]
        )

    # =========================================================================
    # 配置导入导出
    # =========================================================================

    def export_dashboard(self, template_name: str) -> str:
        """
        导出仪表板配置

        Args:
            template_name: 模板名称

        Returns:
            JSON格式的配置
        """
        if template_name not in self.templates:
            raise ValueError(f"模板不存在: {template_name}")

        dashboard_config = self.templates[template_name].to_dict()
        return json.dumps(dashboard_config, indent=2, ensure_ascii=False)

    def import_dashboard(self, dashboard_json: str, template_name: str) -> bool:
        """
        导入仪表板配置

        Args:
            dashboard_json: JSON格式的配置
            template_name: 模板名称

        Returns:
            是否导入成功
        """
        try:
            dashboard_data = json.loads(dashboard_json)

            # 这里应该实现完整的模板重建逻辑
            # 暂时只记录导入信息

            self._log_operation('import_dashboard', template_name, 'success')
            return True

        except Exception as e:
            self._log_operation('import_dashboard', template_name, f'failed: {e}')
            return False

    # =========================================================================
    # 工具方法
    # =========================================================================

    def _apply_custom_config(self, base_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
        """应用自定义配置"""
        # 深度合并配置

        def deep_merge(base: Dict[str, Any], custom: Dict[str, Any]) -> Dict[str, Any]:

            result = base.copy()
            for key, value in custom.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        return deep_merge(base_config, custom_config)

    def _log_operation(self, operation: str, target: str, status: str) -> None:
        """
        记录操作日志

        Args:
            operation: 操作类型
            target: 操作目标
            status: 操作状态
        """
        try:
            message = f"Grafana仪表板管理器 - {operation}: {target}, 状态: {status}"
            print(f"[DataGrafanaDashboard] {message}")
        except Exception:
            print(f"[DataGrafanaDashboard] {operation}: {target} - {status}")

    def get_available_templates(self) -> List[str]:
        """
        获取可用模板列表

        Returns:
            模板名称列表
        """
        return list(self.templates.keys())

    def get_deployed_dashboards(self) -> List[str]:
        """
        获取已部署的仪表板列表

        Returns:
            仪表板标题列表
        """
        return list(self.deployed_dashboards.keys())

    def get_dashboard_stats(self) -> Dict[str, Any]:
        """
        获取仪表板统计信息

        Returns:
            统计信息
        """
        return {
            'available_templates': len(self.templates),
            'deployed_dashboards': len(self.deployed_dashboards),
            'total_panels': sum(len(template.panels) for template in self.templates.values()),
            'timestamp': datetime.now().isoformat()
        }

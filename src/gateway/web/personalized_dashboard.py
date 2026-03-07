"""
个性化仪表盘模块

功能：
- 用户自定义布局
- 拖拽式组件配置
- 实时数据刷新
- 多主题支持
- 组件库管理
- 仪表盘模板
- 分享与导出

技术栈：
- dataclasses: 数据模型
- json: 配置序列化
- datetime: 时间管理

作者: Claude
创建日期: 2026-02-21
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WidgetType(Enum):
    """组件类型"""
    CHART_LINE = "chart_line"              # 折线图
    CHART_BAR = "chart_bar"                # 柱状图
    CHART_PIE = "chart_pie"                # 饼图
    CHART_CANDLESTICK = "chart_candlestick"  # K线图
    METRIC_CARD = "metric_card"            # 指标卡片
    DATA_TABLE = "data_table"              # 数据表格
    NEWS_FEED = "news_feed"                # 新闻流
    ALERT_LIST = "alert_list"              # 告警列表
    PORTFOLIO_SUMMARY = "portfolio_summary"  # 组合摘要
    MARKET_OVERVIEW = "market_overview"    # 市场概览
    WATCHLIST = "watchlist"                # 自选股列表
    CALENDAR = "calendar"                  # 日历
    NOTE = "note"                          # 笔记
    IFRAME = "iframe"                      # 嵌入式页面


class RefreshInterval(Enum):
    """刷新间隔"""
    REALTIME = 0                           # 实时
    SECONDS_5 = 5                          # 5秒
    SECONDS_10 = 10                        # 10秒
    SECONDS_30 = 30                        # 30秒
    MINUTE_1 = 60                          # 1分钟
    MINUTES_5 = 300                        # 5分钟
    MINUTES_15 = 900                       # 15分钟
    MINUTES_30 = 1800                      # 30分钟
    HOUR_1 = 3600                          # 1小时
    MANUAL = -1                            # 手动刷新


class DashboardTheme(Enum):
    """仪表盘主题"""
    LIGHT = "light"                        # 浅色主题
    DARK = "dark"                          # 深色主题
    BLUE = "blue"                          # 蓝色主题
    GREEN = "green"                        # 绿色主题
    PURPLE = "purple"                      # 紫色主题
    AUTO = "auto"                          # 自动跟随系统


@dataclass
class WidgetPosition:
    """组件位置"""
    x: int                                 # X坐标
    y: int                                 # Y坐标
    width: int                             # 宽度
    height: int                            # 高度
    
    def to_dict(self) -> Dict[str, int]:
        return {'x': self.x, 'y': self.y, 'width': self.width, 'height': self.height}


@dataclass
class WidgetConfig:
    """组件配置"""
    widget_id: str
    widget_type: WidgetType
    title: str
    position: WidgetPosition
    data_source: str                       # 数据源ID或URL
    refresh_interval: RefreshInterval
    settings: Dict[str, Any] = field(default_factory=dict)
    theme: DashboardTheme = DashboardTheme.LIGHT
    is_visible: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'widget_id': self.widget_id,
            'widget_type': self.widget_type.value,
            'title': self.title,
            'position': self.position.to_dict(),
            'data_source': self.data_source,
            'refresh_interval': self.refresh_interval.value,
            'settings': self.settings,
            'theme': self.theme.value,
            'is_visible': self.is_visible,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class DashboardConfig:
    """仪表盘配置"""
    dashboard_id: str
    user_id: str
    name: str
    description: str
    theme: DashboardTheme
    layout_type: str                       # grid, free, tabs
    widgets: List[WidgetConfig]
    is_default: bool = False
    is_shared: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dashboard_id': self.dashboard_id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'theme': self.theme.value,
            'layout_type': self.layout_type,
            'widgets': [w.to_dict() for w in self.widgets],
            'is_default': self.is_default,
            'is_shared': self.is_shared,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


class WidgetLibrary:
    """组件库"""
    
    def __init__(self):
        self.widgets: Dict[WidgetType, Dict[str, Any]] = {
            WidgetType.CHART_LINE: {
                'name': '折线图',
                'icon': 'chart-line',
                'description': '展示数据趋势变化',
                'default_size': {'width': 6, 'height': 4},
                'supported_data': ['time_series', 'numerical'],
                'config_schema': {
                    'x_axis': {'type': 'string', 'default': 'date'},
                    'y_axis': {'type': 'string', 'default': 'value'},
                    'show_legend': {'type': 'boolean', 'default': True},
                    'smooth': {'type': 'boolean', 'default': False}
                }
            },
            WidgetType.CHART_BAR: {
                'name': '柱状图',
                'icon': 'chart-bar',
                'description': '对比不同类别的数据',
                'default_size': {'width': 6, 'height': 4},
                'supported_data': ['categorical', 'numerical'],
                'config_schema': {
                    'orientation': {'type': 'string', 'default': 'vertical'},
                    'stacked': {'type': 'boolean', 'default': False}
                }
            },
            WidgetType.CHART_PIE: {
                'name': '饼图',
                'icon': 'chart-pie',
                'description': '展示占比分布',
                'default_size': {'width': 4, 'height': 4},
                'supported_data': ['categorical', 'numerical'],
                'config_schema': {
                    'show_legend': {'type': 'boolean', 'default': True},
                    'donut': {'type': 'boolean', 'default': False}
                }
            },
            WidgetType.CHART_CANDLESTICK: {
                'name': 'K线图',
                'icon': 'chart-candlestick',
                'description': '展示股票价格走势',
                'default_size': {'width': 8, 'height': 5},
                'supported_data': ['ohlcv'],
                'config_schema': {
                    'show_volume': {'type': 'boolean', 'default': True},
                    'indicators': {'type': 'array', 'default': []}
                }
            },
            WidgetType.METRIC_CARD: {
                'name': '指标卡片',
                'icon': 'card-text',
                'description': '展示关键指标数值',
                'default_size': {'width': 3, 'height': 2},
                'supported_data': ['metric'],
                'config_schema': {
                    'prefix': {'type': 'string', 'default': ''},
                    'suffix': {'type': 'string', 'default': ''},
                    'decimals': {'type': 'number', 'default': 2},
                    'show_change': {'type': 'boolean', 'default': True}
                }
            },
            WidgetType.DATA_TABLE: {
                'name': '数据表格',
                'icon': 'table',
                'description': '展示详细数据列表',
                'default_size': {'width': 6, 'height': 6},
                'supported_data': ['tabular'],
                'config_schema': {
                    'page_size': {'type': 'number', 'default': 10},
                    'sortable': {'type': 'boolean', 'default': True},
                    'filterable': {'type': 'boolean', 'default': True}
                }
            },
            WidgetType.NEWS_FEED: {
                'name': '新闻流',
                'icon': 'newspaper',
                'description': '实时财经新闻',
                'default_size': {'width': 4, 'height': 6},
                'supported_data': ['news'],
                'config_schema': {
                    'categories': {'type': 'array', 'default': ['market', 'company']},
                    'max_items': {'type': 'number', 'default': 10}
                }
            },
            WidgetType.ALERT_LIST: {
                'name': '告警列表',
                'icon': 'bell',
                'description': '系统告警信息',
                'default_size': {'width': 4, 'height': 4},
                'supported_data': ['alerts'],
                'config_schema': {
                    'severity_filter': {'type': 'array', 'default': ['high', 'critical']},
                    'max_items': {'type': 'number', 'default': 5}
                }
            },
            WidgetType.PORTFOLIO_SUMMARY: {
                'name': '组合摘要',
                'icon': 'briefcase',
                'description': '投资组合概览',
                'default_size': {'width': 6, 'height': 4},
                'supported_data': ['portfolio'],
                'config_schema': {
                    'show_chart': {'type': 'boolean', 'default': True},
                    'show_positions': {'type': 'boolean', 'default': True}
                }
            },
            WidgetType.MARKET_OVERVIEW: {
                'name': '市场概览',
                'icon': 'globe',
                'description': '市场整体行情',
                'default_size': {'width': 8, 'height': 3},
                'supported_data': ['market_indices'],
                'config_schema': {
                    'indices': {'type': 'array', 'default': ['沪深300', '上证指数', '深证成指']}
                }
            },
            WidgetType.WATCHLIST: {
                'name': '自选股',
                'icon': 'star',
                'description': '关注的股票列表',
                'default_size': {'width': 4, 'height': 6},
                'supported_data': ['stocks'],
                'config_schema': {
                    'columns': {'type': 'array', 'default': ['price', 'change', 'volume']}
                }
            },
            WidgetType.CALENDAR: {
                'name': '日历',
                'icon': 'calendar',
                'description': '财经日历',
                'default_size': {'width': 4, 'height': 4},
                'supported_data': ['events'],
                'config_schema': {
                    'event_types': {'type': 'array', 'default': ['earnings', 'economic']}
                }
            },
            WidgetType.NOTE: {
                'name': '笔记',
                'icon': 'sticky-note',
                'description': '个人笔记',
                'default_size': {'width': 4, 'height': 3},
                'supported_data': ['text'],
                'config_schema': {
                    'content': {'type': 'string', 'default': ''},
                    'color': {'type': 'string', 'default': 'yellow'}
                }
            },
            WidgetType.IFRAME: {
                'name': '嵌入页面',
                'icon': 'window',
                'description': '嵌入外部网页',
                'default_size': {'width': 6, 'height': 6},
                'supported_data': ['url'],
                'config_schema': {
                    'url': {'type': 'string', 'default': ''},
                    'allow_scripts': {'type': 'boolean', 'default': False}
                }
            }
        }
    
    def get_widget_info(self, widget_type: WidgetType) -> Optional[Dict[str, Any]]:
        """获取组件信息"""
        return self.widgets.get(widget_type)
    
    def list_widgets(self) -> List[Dict[str, Any]]:
        """列出所有可用组件"""
        return [
            {
                'type': wt.value,
                **info
            }
            for wt, info in self.widgets.items()
        ]
    
    def get_default_config(self, widget_type: WidgetType) -> Dict[str, Any]:
        """获取组件默认配置"""
        info = self.widgets.get(widget_type)
        if not info:
            return {}
        
        config = {}
        for key, schema in info.get('config_schema', {}).items():
            config[key] = schema.get('default')
        return config


class DashboardTemplateLibrary:
    """仪表盘模板库"""
    
    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = {
            'trading_desk': {
                'name': '交易台',
                'description': '专业交易监控仪表盘',
                'theme': DashboardTheme.DARK,
                'layout_type': 'grid',
                'widgets': [
                    {'type': WidgetType.MARKET_OVERVIEW, 'position': {'x': 0, 'y': 0, 'width': 12, 'height': 2}},
                    {'type': WidgetType.CHART_CANDLESTICK, 'position': {'x': 0, 'y': 2, 'width': 8, 'height': 5}},
                    {'type': WidgetType.WATCHLIST, 'position': {'x': 8, 'y': 2, 'width': 4, 'height': 5}},
                    {'type': WidgetType.PORTFOLIO_SUMMARY, 'position': {'x': 0, 'y': 7, 'width': 6, 'height': 4}},
                    {'type': WidgetType.ALERT_LIST, 'position': {'x': 6, 'y': 7, 'width': 3, 'height': 4}},
                    {'type': WidgetType.NEWS_FEED, 'position': {'x': 9, 'y': 7, 'width': 3, 'height': 4}},
                ]
            },
            'portfolio_manager': {
                'name': '组合管理',
                'description': '投资组合分析仪表盘',
                'theme': DashboardTheme.LIGHT,
                'layout_type': 'grid',
                'widgets': [
                    {'type': WidgetType.METRIC_CARD, 'position': {'x': 0, 'y': 0, 'width': 3, 'height': 2}},
                    {'type': WidgetType.METRIC_CARD, 'position': {'x': 3, 'y': 0, 'width': 3, 'height': 2}},
                    {'type': WidgetType.METRIC_CARD, 'position': {'x': 6, 'y': 0, 'width': 3, 'height': 2}},
                    {'type': WidgetType.METRIC_CARD, 'position': {'x': 9, 'y': 0, 'width': 3, 'height': 2}},
                    {'type': WidgetType.CHART_PIE, 'position': {'x': 0, 'y': 2, 'width': 4, 'height': 4}},
                    {'type': WidgetType.CHART_LINE, 'position': {'x': 4, 'y': 2, 'width': 8, 'height': 4}},
                    {'type': WidgetType.DATA_TABLE, 'position': {'x': 0, 'y': 6, 'width': 12, 'height': 6}},
                ]
            },
            'market_analyst': {
                'name': '市场分析',
                'description': '市场研究和分析仪表盘',
                'theme': DashboardTheme.BLUE,
                'layout_type': 'grid',
                'widgets': [
                    {'type': WidgetType.MARKET_OVERVIEW, 'position': {'x': 0, 'y': 0, 'width': 12, 'height': 2}},
                    {'type': WidgetType.CHART_LINE, 'position': {'x': 0, 'y': 2, 'width': 6, 'height': 4}},
                    {'type': WidgetType.CHART_BAR, 'position': {'x': 6, 'y': 2, 'width': 6, 'height': 4}},
                    {'type': WidgetType.DATA_TABLE, 'position': {'x': 0, 'y': 6, 'width': 6, 'height': 5}},
                    {'type': WidgetType.NEWS_FEED, 'position': {'x': 6, 'y': 6, 'width': 6, 'height': 5}},
                ]
            },
            'minimal': {
                'name': '极简',
                'description': '简洁的监控仪表盘',
                'theme': DashboardTheme.LIGHT,
                'layout_type': 'grid',
                'widgets': [
                    {'type': WidgetType.METRIC_CARD, 'position': {'x': 0, 'y': 0, 'width': 4, 'height': 2}},
                    {'type': WidgetType.METRIC_CARD, 'position': {'x': 4, 'y': 0, 'width': 4, 'height': 2}},
                    {'type': WidgetType.METRIC_CARD, 'position': {'x': 8, 'y': 0, 'width': 4, 'height': 2}},
                    {'type': WidgetType.CHART_LINE, 'position': {'x': 0, 'y': 2, 'width': 12, 'height': 6}},
                ]
            }
        }
    
    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """获取模板"""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """列出所有模板"""
        return [
            {'id': tid, **info}
            for tid, info in self.templates.items()
        ]
    
    def create_dashboard_from_template(self, template_id: str, user_id: str,
                                      name: str) -> Optional[DashboardConfig]:
        """从模板创建仪表盘"""
        template = self.templates.get(template_id)
        if not template:
            return None
        
        widgets = []
        for i, widget_def in enumerate(template['widgets']):
            widget = WidgetConfig(
                widget_id=str(uuid.uuid4()),
                widget_type=widget_def['type'],
                title=widget_def['type'].value,
                position=WidgetPosition(**widget_def['position']),
                data_source='default',
                refresh_interval=RefreshInterval.MINUTE_1,
                settings={}
            )
            widgets.append(widget)
        
        return DashboardConfig(
            dashboard_id=str(uuid.uuid4()),
            user_id=user_id,
            name=name,
            description=template['description'],
            theme=template['theme'],
            layout_type=template['layout_type'],
            widgets=widgets
        )


class PersonalizedDashboardManager:
    """个性化仪表盘管理器"""
    
    def __init__(self, storage_path: str = "dashboards"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.widget_library = WidgetLibrary()
        self.template_library = DashboardTemplateLibrary()
        self.dashboards: Dict[str, DashboardConfig] = {}
        self._load_dashboards()
    
    def _load_dashboards(self) -> None:
        """加载保存的仪表盘"""
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    dashboard = self._deserialize_dashboard(data)
                    self.dashboards[dashboard.dashboard_id] = dashboard
            except Exception as e:
                logger.error(f"加载仪表盘失败 {file_path}: {e}")
    
    def _deserialize_dashboard(self, data: Dict) -> DashboardConfig:
        """反序列化仪表盘配置"""
        widgets = []
        for w_data in data.get('widgets', []):
            widget = WidgetConfig(
                widget_id=w_data['widget_id'],
                widget_type=WidgetType(w_data['widget_type']),
                title=w_data['title'],
                position=WidgetPosition(**w_data['position']),
                data_source=w_data['data_source'],
                refresh_interval=RefreshInterval(w_data['refresh_interval']),
                settings=w_data.get('settings', {}),
                theme=DashboardTheme(w_data.get('theme', 'light')),
                is_visible=w_data.get('is_visible', True),
                created_at=datetime.fromisoformat(w_data['created_at']),
                updated_at=datetime.fromisoformat(w_data['updated_at'])
            )
            widgets.append(widget)
        
        return DashboardConfig(
            dashboard_id=data['dashboard_id'],
            user_id=data['user_id'],
            name=data['name'],
            description=data.get('description', ''),
            theme=DashboardTheme(data.get('theme', 'light')),
            layout_type=data.get('layout_type', 'grid'),
            widgets=widgets,
            is_default=data.get('is_default', False),
            is_shared=data.get('is_shared', False),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at'])
        )
    
    def _save_dashboard(self, dashboard: DashboardConfig) -> bool:
        """保存仪表盘"""
        try:
            file_path = self.storage_path / f"{dashboard.dashboard_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dashboard.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"保存仪表盘失败: {e}")
            return False
    
    def create_dashboard(self, user_id: str, name: str,
                        description: str = "",
                        template_id: Optional[str] = None) -> DashboardConfig:
        """创建新仪表盘"""
        if template_id:
            dashboard = self.template_library.create_dashboard_from_template(
                template_id, user_id, name
            )
            if dashboard:
                dashboard.description = description
                self.dashboards[dashboard.dashboard_id] = dashboard
                self._save_dashboard(dashboard)
                return dashboard
        
        # 创建空白仪表盘
        dashboard = DashboardConfig(
            dashboard_id=str(uuid.uuid4()),
            user_id=user_id,
            name=name,
            description=description,
            theme=DashboardTheme.LIGHT,
            layout_type='grid',
            widgets=[]
        )
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        self._save_dashboard(dashboard)
        logger.info(f"创建仪表盘: {name} ({dashboard.dashboard_id})")
        return dashboard
    
    def get_dashboard(self, dashboard_id: str) -> Optional[DashboardConfig]:
        """获取仪表盘"""
        return self.dashboards.get(dashboard_id)
    
    def get_user_dashboards(self, user_id: str) -> List[DashboardConfig]:
        """获取用户的所有仪表盘"""
        return [
            d for d in self.dashboards.values()
            if d.user_id == user_id
        ]
    
    def update_dashboard(self, dashboard_id: str,
                        updates: Dict[str, Any]) -> bool:
        """更新仪表盘"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return False
        
        if 'name' in updates:
            dashboard.name = updates['name']
        if 'description' in updates:
            dashboard.description = updates['description']
        if 'theme' in updates:
            dashboard.theme = DashboardTheme(updates['theme'])
        if 'layout_type' in updates:
            dashboard.layout_type = updates['layout_type']
        
        dashboard.updated_at = datetime.now()
        return self._save_dashboard(dashboard)
    
    def delete_dashboard(self, dashboard_id: str) -> bool:
        """删除仪表盘"""
        if dashboard_id in self.dashboards:
            del self.dashboards[dashboard_id]
            file_path = self.storage_path / f"{dashboard_id}.json"
            if file_path.exists():
                file_path.unlink()
            return True
        return False
    
    def add_widget(self, dashboard_id: str, widget_type: WidgetType,
                  title: str, position: WidgetPosition,
                  data_source: str = "",
                  settings: Optional[Dict] = None) -> Optional[WidgetConfig]:
        """添加组件"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return None
        
        # 获取默认配置
        default_settings = self.widget_library.get_default_config(widget_type)
        if settings:
            default_settings.update(settings)
        
        widget = WidgetConfig(
            widget_id=str(uuid.uuid4()),
            widget_type=widget_type,
            title=title,
            position=position,
            data_source=data_source,
            refresh_interval=RefreshInterval.MINUTE_1,
            settings=default_settings
        )
        
        dashboard.widgets.append(widget)
        dashboard.updated_at = datetime.now()
        self._save_dashboard(dashboard)
        
        logger.info(f"添加组件到仪表盘 {dashboard_id}: {title}")
        return widget
    
    def remove_widget(self, dashboard_id: str, widget_id: str) -> bool:
        """移除组件"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return False
        
        dashboard.widgets = [w for w in dashboard.widgets if w.widget_id != widget_id]
        dashboard.updated_at = datetime.now()
        return self._save_dashboard(dashboard)
    
    def update_widget(self, dashboard_id: str, widget_id: str,
                     updates: Dict[str, Any]) -> bool:
        """更新组件"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return False
        
        for widget in dashboard.widgets:
            if widget.widget_id == widget_id:
                if 'title' in updates:
                    widget.title = updates['title']
                if 'position' in updates:
                    widget.position = WidgetPosition(**updates['position'])
                if 'settings' in updates:
                    widget.settings.update(updates['settings'])
                if 'refresh_interval' in updates:
                    widget.refresh_interval = RefreshInterval(updates['refresh_interval'])
                if 'is_visible' in updates:
                    widget.is_visible = updates['is_visible']
                
                widget.updated_at = datetime.now()
                dashboard.updated_at = datetime.now()
                return self._save_dashboard(dashboard)
        
        return False
    
    def reorder_widgets(self, dashboard_id: str,
                       widget_order: List[str]) -> bool:
        """重新排序组件"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return False
        
        # 创建widget_id到widget的映射
        widget_map = {w.widget_id: w for w in dashboard.widgets}
        
        # 按新顺序重组
        new_widgets = []
        for wid in widget_order:
            if wid in widget_map:
                new_widgets.append(widget_map[wid])
        
        # 添加未排序的组件
        for w in dashboard.widgets:
            if w.widget_id not in widget_order:
                new_widgets.append(w)
        
        dashboard.widgets = new_widgets
        dashboard.updated_at = datetime.now()
        return self._save_dashboard(dashboard)
    
    def set_default_dashboard(self, dashboard_id: str) -> bool:
        """设置默认仪表盘"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return False
        
        # 取消其他默认仪表盘
        for d in self.dashboards.values():
            if d.user_id == dashboard.user_id and d.is_default:
                d.is_default = False
                self._save_dashboard(d)
        
        dashboard.is_default = True
        return self._save_dashboard(dashboard)
    
    def export_dashboard(self, dashboard_id: str) -> Optional[str]:
        """导出仪表盘配置为JSON字符串"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return None
        
        return json.dumps(dashboard.to_dict(), ensure_ascii=False, indent=2)
    
    def import_dashboard(self, user_id: str, json_str: str) -> Optional[DashboardConfig]:
        """导入仪表盘配置"""
        try:
            data = json.loads(json_str)
            data['dashboard_id'] = str(uuid.uuid4())
            data['user_id'] = user_id
            data['created_at'] = datetime.now().isoformat()
            data['updated_at'] = datetime.now().isoformat()
            data['is_default'] = False
            
            dashboard = self._deserialize_dashboard(data)
            self.dashboards[dashboard.dashboard_id] = dashboard
            self._save_dashboard(dashboard)
            
            logger.info(f"导入仪表盘: {dashboard.name}")
            return dashboard
        except Exception as e:
            logger.error(f"导入仪表盘失败: {e}")
            return None
    
    def get_widget_library(self) -> WidgetLibrary:
        """获取组件库"""
        return self.widget_library
    
    def get_template_library(self) -> DashboardTemplateLibrary:
        """获取模板库"""
        return self.template_library


# 便捷函数
def get_dashboard_manager(storage_path: str = "dashboards") -> PersonalizedDashboardManager:
    """获取仪表盘管理器实例"""
    return PersonalizedDashboardManager(storage_path)


# 单例实例
_dashboard_manager_instance: Optional[PersonalizedDashboardManager] = None


def get_dashboard_manager_singleton() -> PersonalizedDashboardManager:
    """获取仪表盘管理器单例"""
    global _dashboard_manager_instance
    if _dashboard_manager_instance is None:
        _dashboard_manager_instance = PersonalizedDashboardManager()
    return _dashboard_manager_instance

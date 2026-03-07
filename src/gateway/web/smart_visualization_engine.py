"""
智能可视化推荐引擎模块

功能：
- 数据特征分析
- 可视化类型推荐
- 图表配置优化
- 交互式图表生成
- 多维度数据展示
- 自适应布局

技术栈：
- plotly: 交互式图表
- pandas: 数据分析

作者: Claude
创建日期: 2026-02-21
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChartType(Enum):
    """图表类型"""
    LINE = "line"                    # 折线图
    BAR = "bar"                      # 柱状图
    PIE = "pie"                      # 饼图
    SCATTER = "scatter"              # 散点图
    CANDLESTICK = "candlestick"      # K线图
    HEATMAP = "heatmap"              # 热力图
    AREA = "area"                    # 面积图
    HISTOGRAM = "histogram"          # 直方图
    BOX = "box"                      # 箱线图
    RADAR = "radar"                  # 雷达图
    TREEMAP = "treemap"              # 树图
    SANKEY = "sankey"                # 桑基图
    GAUGE = "gauge"                  # 仪表盘
    TABLE = "table"                  # 表格


class DataType(Enum):
    """数据类型"""
    TIME_SERIES = "time_series"      # 时间序列
    CATEGORICAL = "categorical"      # 分类数据
    NUMERICAL = "numerical"          # 数值数据
    GEOGRAPHICAL = "geographical"    # 地理数据
    HIERARCHICAL = "hierarchical"    # 层级数据
    CORRELATION = "correlation"      # 相关性数据


@dataclass
class DataProfile:
    """数据特征分析"""
    data_type: DataType
    row_count: int
    column_count: int
    has_datetime: bool
    has_numeric: bool
    has_categorical: bool
    time_span_days: Optional[int] = None
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)


@dataclass
class ChartRecommendation:
    """图表推荐"""
    chart_type: ChartType
    confidence: float
    reason: str
    config: Dict[str, Any]
    suitability_score: float


@dataclass
class VisualizationConfig:
    """可视化配置"""
    title: str
    chart_type: ChartType
    x_axis: Optional[str]
    y_axis: Optional[str]
    color_by: Optional[str]
    size_by: Optional[str]
    aggregation: Optional[str]
    theme: str = "default"
    interactive: bool = True
    height: int = 600
    width: int = 1000


class DataProfiler:
    """数据特征分析器"""
    
    def analyze(self, data: Union[pd.DataFrame, List[Dict]]) -> DataProfile:
        """分析数据特征"""
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        profile = DataProfile(
            data_type=self._determine_data_type(data),
            row_count=len(data),
            column_count=len(data.columns),
            has_datetime=False,
            has_numeric=False,
            has_categorical=False,
            numeric_columns=[],
            categorical_columns=[],
            datetime_columns=[]
        )
        
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                profile.has_datetime = True
                profile.datetime_columns.append(col)
            elif pd.api.types.is_numeric_dtype(data[col]):
                profile.has_numeric = True
                profile.numeric_columns.append(col)
            else:
                profile.has_categorical = True
                profile.categorical_columns.append(col)
        
        # 计算时间跨度
        if profile.datetime_columns:
            date_col = data[profile.datetime_columns[0]]
            profile.time_span_days = (date_col.max() - date_col.min()).days
        
        return profile
    
    def _determine_data_type(self, data: pd.DataFrame) -> DataType:
        """确定数据类型"""
        has_datetime = any(pd.api.types.is_datetime64_any_dtype(data[col]) for col in data.columns)
        has_numeric = any(pd.api.types.is_numeric_dtype(data[col]) for col in data.columns)
        
        if has_datetime and has_numeric:
            return DataType.TIME_SERIES
        elif has_numeric and len(data.columns) >= 2:
            return DataType.CORRELATION
        else:
            return DataType.CATEGORICAL


class ChartRecommender:
    """图表推荐器"""
    
    def __init__(self):
        self.recommendation_rules = self._init_rules()
    
    def _init_rules(self) -> List[Dict]:
        """初始化推荐规则"""
        return [
            {
                'chart_type': ChartType.CANDLESTICK,
                'condition': lambda p: p.data_type == DataType.TIME_SERIES and 
                                       'open' in p.numeric_columns and 
                                       'close' in p.numeric_columns,
                'weight': 1.0,
                'reason': '时间序列包含OHLC数据，适合K线图'
            },
            {
                'chart_type': ChartType.LINE,
                'condition': lambda p: p.data_type == DataType.TIME_SERIES,
                'weight': 0.9,
                'reason': '时间序列数据适合折线图展示趋势'
            },
            {
                'chart_type': ChartType.BAR,
                'condition': lambda p: p.has_categorical and p.has_numeric,
                'weight': 0.8,
                'reason': '分类数据对比适合柱状图'
            },
            {
                'chart_type': ChartType.PIE,
                'condition': lambda p: p.has_categorical and len(p.categorical_columns) == 1,
                'weight': 0.6,
                'reason': '单维度分类占比适合饼图'
            },
            {
                'chart_type': ChartType.SCATTER,
                'condition': lambda p: len(p.numeric_columns) >= 2,
                'weight': 0.7,
                'reason': '多数值变量关系适合散点图'
            },
            {
                'chart_type': ChartType.HEATMAP,
                'condition': lambda p: p.data_type == DataType.CORRELATION,
                'weight': 0.8,
                'reason': '相关性数据适合热力图'
            },
            {
                'chart_type': ChartType.HISTOGRAM,
                'condition': lambda p: len(p.numeric_columns) == 1,
                'weight': 0.5,
                'reason': '单数值分布适合直方图'
            },
            {
                'chart_type': ChartType.BOX,
                'condition': lambda p: p.has_numeric and p.has_categorical,
                'weight': 0.6,
                'reason': '数值分布对比适合箱线图'
            },
        ]
    
    def recommend(self, profile: DataProfile, 
                  top_n: int = 3) -> List[ChartRecommendation]:
        """推荐图表类型"""
        recommendations = []
        
        for rule in self.recommendation_rules:
            if rule['condition'](profile):
                confidence = rule['weight']
                config = self._generate_config(rule['chart_type'], profile)
                
                recommendations.append(ChartRecommendation(
                    chart_type=rule['chart_type'],
                    confidence=confidence,
                    reason=rule['reason'],
                    config=config,
                    suitability_score=confidence * 100
                ))
        
        # 按置信度排序
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations[:top_n]
    
    def _generate_config(self, chart_type: ChartType, 
                        profile: DataProfile) -> Dict[str, Any]:
        """生成图表配置"""
        config = {
            'chart_type': chart_type.value,
            'data_mapping': {}
        }
        
        if profile.datetime_columns:
            config['data_mapping']['x'] = profile.datetime_columns[0]
        elif profile.categorical_columns:
            config['data_mapping']['x'] = profile.categorical_columns[0]
        
        if profile.numeric_columns:
            config['data_mapping']['y'] = profile.numeric_columns[0]
        
        if len(profile.numeric_columns) > 1:
            config['data_mapping']['y2'] = profile.numeric_columns[1]
        
        return config


class SmartVisualizationEngine:
    """智能可视化引擎"""
    
    def __init__(self):
        self.profiler = DataProfiler()
        self.recommender = ChartRecommender()
    
    def analyze_and_recommend(self, data: Union[pd.DataFrame, List[Dict]], 
                              title: str = "") -> Dict[str, Any]:
        """分析数据并推荐可视化方案"""
        # 分析数据特征
        profile = self.profiler.analyze(data)
        
        # 获取推荐
        recommendations = self.recommender.recommend(profile)
        
        return {
            'data_profile': {
                'type': profile.data_type.value,
                'rows': profile.row_count,
                'columns': profile.column_count,
                'has_datetime': profile.has_datetime,
                'has_numeric': profile.has_numeric,
                'has_categorical': profile.has_categorical,
                'time_span_days': profile.time_span_days,
                'numeric_columns': profile.numeric_columns,
                'categorical_columns': profile.categorical_columns,
                'datetime_columns': profile.datetime_columns
            },
            'recommendations': [
                {
                    'chart_type': r.chart_type.value,
                    'confidence': r.confidence,
                    'reason': r.reason,
                    'config': r.config,
                    'suitability_score': r.suitability_score
                }
                for r in recommendations
            ],
            'title': title
        }
    
    def create_visualization(self, data: Union[pd.DataFrame, List[Dict]],
                           config: VisualizationConfig) -> Dict[str, Any]:
        """创建可视化"""
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        # 根据图表类型生成配置
        plot_config = self._generate_plotly_config(data, config)
        
        return {
            'config': config.__dict__,
            'plotly_config': plot_config,
            'data_summary': {
                'rows': len(data),
                'columns': list(data.columns)
            }
        }
    
    def _generate_plotly_config(self, data: pd.DataFrame, 
                               config: VisualizationConfig) -> Dict[str, Any]:
        """生成Plotly配置"""
        plot_config = {
            'data': [],
            'layout': {
                'title': config.title,
                'height': config.height,
                'width': config.width,
                'template': config.theme
            }
        }
        
        if config.chart_type == ChartType.LINE:
            trace = {
                'type': 'scatter',
                'mode': 'lines',
                'x': data[config.x_axis].tolist() if config.x_axis else list(range(len(data))),
                'y': data[config.y_axis].tolist() if config.y_axis else data.iloc[:, 0].tolist(),
                'name': config.y_axis or 'Value'
            }
            plot_config['data'].append(trace)
        
        elif config.chart_type == ChartType.BAR:
            trace = {
                'type': 'bar',
                'x': data[config.x_axis].tolist() if config.x_axis else list(range(len(data))),
                'y': data[config.y_axis].tolist() if config.y_axis else data.iloc[:, 0].tolist(),
                'name': config.y_axis or 'Value'
            }
            plot_config['data'].append(trace)
        
        elif config.chart_type == ChartType.PIE:
            trace = {
                'type': 'pie',
                'labels': data[config.x_axis].tolist() if config.x_axis else data.iloc[:, 0].tolist(),
                'values': data[config.y_axis].tolist() if config.y_axis else data.iloc[:, 1].tolist()
            }
            plot_config['data'].append(trace)
        
        elif config.chart_type == ChartType.SCATTER:
            trace = {
                'type': 'scatter',
                'mode': 'markers',
                'x': data[config.x_axis].tolist() if config.x_axis else data.iloc[:, 0].tolist(),
                'y': data[config.y_axis].tolist() if config.y_axis else data.iloc[:, 1].tolist(),
                'marker': {'size': 10}
            }
            plot_config['data'].append(trace)
        
        return plot_config


# 便捷函数
def get_visualization_engine() -> SmartVisualizationEngine:
    """获取可视化引擎实例"""
    return SmartVisualizationEngine()

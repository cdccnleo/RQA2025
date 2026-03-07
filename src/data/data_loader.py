#!/usr/bin/env python3
"""
数据加载器模块

提供统一的数据加载接口，支持多种数据源和格式。
"""

import sys
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    数据加载器

    统一的数据加载接口，支持：
    - 文件加载 (CSV, JSON, Excel, Parquet等)
    - 数据库加载
    - API数据加载
    - 流数据加载
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.loaders = {}
        self._register_default_loaders()

    def _register_default_loaders(self):
        """注册默认加载器"""
        self.loaders.update({
            'csv': self._load_csv,
            'json': self._load_json,
            'excel': self._load_excel,
            'parquet': self._load_parquet,
            'database': self._load_database,
            'api': self._load_api
        })

    def load(self, source: str, **kwargs) -> Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
        """
        加载数据

        Args:
            source: 数据源标识符或路径
            **kwargs: 加载参数

        Returns:
            加载的数据
        """
        try:
            # 解析数据源类型
            source_type = self._detect_source_type(source)

            # 获取对应的加载器
            loader = self.loaders.get(source_type)
            if not loader:
                raise ValueError(f"不支持的数据源类型: {source_type}")

            # 执行加载
            return loader(source, **kwargs)

        except Exception as e:
            logger.error(f"数据加载失败: {source}, 错误: {e}")
            raise

    def _detect_source_type(self, source: str) -> str:
        """检测数据源类型"""
        if source.startswith(('http://', 'https://')):
            return 'api'
        elif source.endswith('.csv'):
            return 'csv'
        elif source.endswith(('.json', '.jsonl')):
            return 'json'
        elif source.endswith(('.xlsx', '.xls')):
            return 'excel'
        elif source.endswith('.parquet'):
            return 'parquet'
        elif source.startswith(('postgresql://', 'mysql://', 'sqlite://')):
            return 'database'
        else:
            # 默认为文件路径，尝试自动检测
            path = Path(source)
            if path.exists():
                if path.is_file():
                    suffix = path.suffix.lower()
                    if suffix == '.csv':
                        return 'csv'
                    elif suffix in ['.json', '.jsonl']:
                        return 'json'
                    elif suffix in ['.xlsx', '.xls']:
                        return 'excel'
                    elif suffix == '.parquet':
                        return 'parquet'
            return 'csv'  # 默认

    def _load_csv(self, source: str, **kwargs) -> pd.DataFrame:
        """加载CSV文件"""
        return pd.read_csv(source, **kwargs)

    def _load_json(self, source: str, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """加载JSON文件"""
        import json
        with open(source, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_excel(self, source: str, **kwargs) -> pd.DataFrame:
        """加载Excel文件"""
        return pd.read_excel(source, **kwargs)

    def _load_parquet(self, source: str, **kwargs) -> pd.DataFrame:
        """加载Parquet文件"""
        try:
            import pyarrow.parquet as pq
            return pq.read_table(source).to_pandas()
        except ImportError:
            raise ImportError("需要安装pyarrow来加载Parquet文件")

    def _load_database(self, source: str, **kwargs) -> pd.DataFrame:
        """从数据库加载数据"""
        try:
            import sqlalchemy as sa
            engine = sa.create_engine(source)
            query = kwargs.get('query', 'SELECT * FROM table')
            return pd.read_sql(query, engine, **kwargs)
        except ImportError:
            raise ImportError("需要安装sqlalchemy来从数据库加载数据")

    def _load_api(self, source: str, **kwargs) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """从API加载数据"""
        try:
            import requests
            response = requests.get(source, **kwargs)
            response.raise_for_status()
            return response.json()
        except ImportError:
            raise ImportError("需要安装requests来从API加载数据")

    def register_loader(self, source_type: str, loader_func):
        """
        注册自定义加载器

        Args:
            source_type: 数据源类型
            loader_func: 加载函数
        """
        self.loaders[source_type] = loader_func


# 创建默认实例
default_data_loader = DataLoader()

def load_data(source: str, **kwargs) -> Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    """
    便捷函数：使用默认数据加载器加载数据

    Args:
        source: 数据源
        **kwargs: 加载参数

    Returns:
        加载的数据
    """
    return default_data_loader.load(source, **kwargs)

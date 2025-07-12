# RQA2025 数据层功能增强完整报告（续2）

## 2. 功能分析（续）

### 2.2 功能扩展（续）

#### 2.2.1 数据质量监控（续）

**核心代码示例**（续）：
```python
    def _check_outliers(
        self,
        data: pd.DataFrame,
        numeric_columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, Dict[str, Any]]:
        """
        检查异常值（续）
        """
        outliers = {}
        
        for column in numeric_columns:
            if column not in data.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(data[column]):
                continue
            
            if data[column].isna().all():
                continue
            
            values = data[column].dropna()
            
            if method == 'iqr':
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outlier_mask = (values < lower_bound) | (values > upper_bound)
            elif method == 'zscore':
                mean = values.mean()
                std = values.std()
                z_scores = (values - mean) / std
                outlier_mask = z_scores.abs() > threshold
            else:
                raise ValueError(f"Unsupported outlier detection method: {method}")
            
            outlier_count = outlier_mask.sum()
            outlier_ratio = outlier_count / len(values)
            
            outliers[column] = {
                'method': method,
                'threshold': threshold,
                'outlier_count': int(outlier_count),
                'outlier_ratio': float(outlier_ratio),
                'bounds': {
                    'lower': float(lower_bound) if method == 'iqr' else None,
                    'upper': float(upper_bound) if method == 'iqr' else None,
                    'zscore': float(threshold) if method == 'zscore' else None
                }
            }
        
        return outliers
    
    def _check_date_range(
        self,
        data: pd.DataFrame,
        date_column: str
    ) -> Dict[str, Any]:
        """
        检查日期范围
        
        Args:
            data: 数据
            date_column: 日期列名
            
        Returns:
            Dict[str, Any]: 日期范围信息
        """
        if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
            try:
                dates = pd.to_datetime(data[date_column])
            except Exception as e:
                return {
                    'error': f"Failed to parse dates: {str(e)}"
                }
        else:
            dates = data[date_column]
        
        dates = dates.dropna()
        
        if dates.empty:
            return {
                'error': "No valid dates found"
            }
        
        start_date = dates.min()
        end_date = dates.max()
        date_range = pd.date_range(start_date, end_date)
        
        missing_dates = set(date_range) - set(dates)
        
        return {
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'total_days': len(date_range),
            'available_days': len(dates.unique()),
            'missing_days': len(missing_dates),
            'missing_dates': sorted(d.isoformat() for d in missing_dates)[:10]  # 只返回前10个缺失日期
        }
    
    def _check_symbol_coverage(
        self,
        data: pd.DataFrame,
        symbol_column: str,
        date_column: Optional[str] = None,
        expected_symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        检查股票代码覆盖率
        
        Args:
            data: 数据
            symbol_column: 股票代码列名
            date_column: 日期列名（可选）
            expected_symbols: 预期的股票代码列表（可选）
            
        Returns:
            Dict[str, Any]: 股票代码覆盖率信息
        """
        actual_symbols = set(data[symbol_column].unique())
        
        coverage_info = {
            'total_symbols': len(actual_symbols),
            'symbols': sorted(actual_symbols)
        }
        
        if expected_symbols:
            expected_set = set(expected_symbols)
            missing_symbols = expected_set - actual_symbols
            extra_symbols = actual_symbols - expected_set
            
            coverage_info.update({
                'expected_symbols': len(expected_symbols),
                'missing_symbols': sorted(missing_symbols),
                'extra_symbols': sorted(extra_symbols),
                'coverage_ratio': len(actual_symbols) / len(expected_symbols)
            })
        
        if date_column:
            # 分析每个交易日的股票覆盖情况
            daily_coverage = data.groupby(date_column)[symbol_column].nunique()
            
            coverage_info.update({
                'daily_coverage': {
                    'min': int(daily_coverage.min()),
                    'max': int(daily_coverage.max()),
                    'mean': float(daily_coverage.mean()),
                    'std': float(daily_coverage.std())
                }
            })
        
        return coverage_info

```

#### 2.2.2 数据导出功能

**现状分析**：
缺乏将数据导出为不同格式的功能，不便于与其他系统进行数据交换和分析。

**实现建议**：
实现一个 `DataExporter` 类，用于将数据导出为不同格式。该类将提供以下功能：

- CSV导出：导出为CSV格式
- Excel导出：导出为Excel格式，支持多个工作表
- JSON导出：导出为JSON格式
- Parquet导出：导出为Parquet格式
- Feather导出：导出为Feather格式
- HDF5导出：导出为HDF5格式
- SQL导出：导出到SQL数据库

**核心代码示例**：
```python
import os
import json
import pandas as pd
import sqlite3
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataExporter:
    """数据导出器"""
    
    def __init__(self, export_dir: str = './exports'):
        """
        初始化数据导出器
        
        Args:
            export_dir: 导出目录
        """
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
    
    def export(
        self,
        data: pd.DataFrame,
        format: str,
        filename: str,
        **kwargs
    ) -> str:
        """
        导出数据
        
        Args:
            data: 要导出的数据
            format: 导出格式
            filename: 文件名（不包含扩展名）
            **kwargs: 其他参数
            
        Returns:
            str: 导出文件路径
        """
        if data is None or data.empty:
            raise ValueError("No data to export")
        
        # 根据格式选择导出方法
        export_methods = {
            'csv': self._export_csv,
            'excel': self._export_excel,
            'json': self._export_json,
            'parquet': self._export_parquet,
            'feather': self._export_feather,
            'hdf': self._export_hdf,
            'sql': self._export_sql
        }
        
        if format not in export_methods:
            raise ValueError(f"Unsupported export format: {format}")
        
        # 调用相应的导出方法
        return export_methods[format](data, filename, **kwargs)
    
    def _export_csv(
        self,
        data: pd.DataFrame,
        filename: str,
        **kwargs
    ) -> str:
        """导出为CSV格式"""
        filepath = os.path.join(self.export_dir, f"{filename}.csv")
        data.to_csv(filepath, **kwargs)
        return filepath
    
    def _export_excel(
        self,
        data: pd.DataFrame,
        filename: str,
        sheet_name: str = 'Sheet1',
        **kwargs
    ) -> str:
        """导出为Excel格式"""
        filepath = os.path.join(self.export_dir, f"{filename}.xlsx")
        data.to_excel(filepath, sheet_name=sheet_name, **kwargs)
        return filepath
    
    def _export_json(
        self,
        data: pd.DataFrame,
        filename: str,
        orient: str = 'records',
        **kwargs
    ) -> str:
        """导出为JSON格式"""
        filepath = os.path.join(self.export_dir, f"{filename}.json")
        data.to_json(filepath, orient=orient, **kwargs)
        return filepath
    
    def _export_parquet(
        self,
        data: pd.DataFrame,
        filename: str,
        **kwargs
    ) -> str:
        """导出为Parquet格式"""
        filepath = os.path.join(self.export_dir, f"{filename}.parquet")
        data.to_parquet(filepath, **kwargs)
        return filepath
    
    def _export_feather(
        self,
        data: pd.DataFrame,
        filename: str,
        **kwargs
    ) -> str:
        """导出为Feather格式"""
        filepath = os.path.join(self.export_dir, f"{filename}.feather")
        data.to_feather(filepath, **kwargs)
        return filepath
    
    def _export_hdf(
        self,
        data: pd.DataFrame,
        filename: str,
        key: str = 'data',
        **kwargs
    ) -> str:
        """导出为HDF5格式"""
        filepath = os.path.join(self.export_dir, f"{filename}.h5")
        data.to_hdf(filepath, key=key, **kwargs)
        return filepath
    
    def _export_sql(
        self,
        data: pd.DataFrame,
        filename: str,
        table_name: str,
        if_exists: str = 'fail',
        **kwargs
    ) -> str:
        """导出到SQLite数据库"""
        filepath = os.path.join(self.export_dir, f"{filename}.db")
        
        # 创建SQLite连接
        with sqlite3.connect(filepath) as conn:
            data.to_sql(
                name=table_name,
                con=conn,
                if_exists=if_exists,
                **kwargs
            )
        
        return filepath
```

### 2.3 监控告警

#### 2.3.1 异常告警

**现状分析**：
有基本的日志记录，但缺乏系统化的告警机制，无法及时发现和处理异常情况。

**实现建议**：
实现一个 `AlertManager` 类，用于管理异常告警。该类将提供以下功能：

- 阈值检查：检查指标是否超过阈值
- 告警级别：支持不同级别的告警（info、warning、error、critical）
- 告警通知：支持多种通知方式（邮件、webhook、日志）
- 告警历史：记录和查询告警历史

**核心代码示例**：
```python
import smtplib
import requests
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class AlertManager:
    """告警管理器"""
    
    def __init__(
        self,
        config: Dict[str, Any]
    ):
        """
        初始化告警管理器
        
        Args:
            config: 配置信息，包括：
                - email_config: 邮件配置
                - webhook_config: Webhook配置
                - alert_levels: 告警级别配置
        """
        self.config = config
        self.alert_history = []
    
    def check_threshold(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        operator: str = '>',
        level: str = 'warning'
    ) -> bool:
        """
        检查阈值
        
        Args:
            metric_name: 指标名称
            value: 当前值
            threshold: 阈值
            operator: 比较运算符（'>'、'<'、'>='、'<='、'=='、'!='）
            level: 告警级别
            
        Returns:
            bool: 是否触发告警
        """
        operators = {
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '==': lambda x, y: x == y,
            '!=': lambda x, y: x != y
        }
        
        if operator not in operators:
            raise ValueError(f"Unsupported operator: {operator}")
        
        if operators[operator](value, threshold):
            self.alert(
                level=level,
                message=f"Metric {metric_name} {operator} {threshold}: {value}"
            )
            return True
        
        return False
    
    def alert(
        self,
        level: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        发送告警
        
        Args:
            level: 告警级别
            message: 告警消息
            details: 详细信息
        """
        if level not in self.config.get('alert_levels', {}):
            raise ValueError(f"Invalid alert level: {level}")
        
        alert_info = {
            'level': level,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        # 记录告警历史
        self.alert_history.append(alert_info)
        
        # 根据级别配置发送通知
        level_config = self.config['alert_
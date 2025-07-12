# 数据层功能增强分析报告（续）

## 功能实现建议（续）

### 2. 功能扩展

#### 2.1 数据质量监控

建议实现一个 `DataQualityMonitor` 类，用于监控数据质量：

```python
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

class DataQualityMonitor:
    """数据质量监控器"""
    
    def __init__(self):
        """初始化数据质量监控器"""
        pass
    
    def check_missing_values(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        检查缺失值
        
        Args:
            data: 数据框
            
        Returns:
            Dict[str, float]: 每列的缺失值比例
        """
        missing_ratio = data.isnull().mean().to_dict()
        return missing_ratio
    
    def check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        检查重复值
        
        Args:
            data: 数据框
            
        Returns:
            Dict[str, Any]: 重复值信息
        """
        duplicate_count = data.duplicated().sum()
        duplicate_ratio = duplicate_count / len(data) if len(data) > 0 else 0
        return {
            'duplicate_count': duplicate_count,
            'duplicate_ratio': duplicate_ratio
        }
    
    def check_outliers(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, Dict[str, Any]]:
        """
        检查异常值
        
        Args:
            data: 数据框
            columns: 要检查的列，默认为所有数值列
            method: 检测方法，'iqr'或'zscore'
            threshold: 阈值，IQR方法的倍数或Z-Score方法的标准差倍数
            
        Returns:
            Dict[str, Dict[str, Any]]: 每列的异常值信息
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        result = {}
        for col in columns:
            if col not in data.columns or not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            elif method == 'zscore':
                mean = data[col].mean()
                std = data[col].std()
                z_scores = np.abs((data[col] - mean) / std)
                outliers = data[col][z_scores > threshold]
            else:
                raise ValueError(f"Unknown method: {method}")
            
            result[col] = {
                'outlier_count': len(outliers),
                'outlier_ratio': len(outliers) / len(data) if len(data) > 0 else 0,
                'min': data[col].min(),
                'max': data[col].max(),
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std()
            }
        
        return result
    
    def check_data_types(self, data: pd.DataFrame) -> Dict[str, str]:
        """
        检查数据类型
        
        Args:
            data: 数据框
            
        Returns:
            Dict[str, str]: 每列的数据类型
        """
        return {col: str(dtype) for col, dtype in data.dtypes.items()}
    
    def check_date_range(
        self,
        data: pd.DataFrame,
        date_column: str = 'date'
    ) -> Dict[str, Any]:
        """
        检查日期范围
        
        Args:
            data: 数据框
            date_column: 日期列名
            
        Returns:
            Dict[str, Any]: 日期范围信息
        """
        if date_column not in data.columns:
            return {'error': f"Date column '{date_column}' not found"}
        
        try:
            min_date = data[date_column].min()
            max_date = data[date_column].max()
            date_range = (max_date - min_date).days
            return {
                'min_date': min_date,
                'max_date': max_date,
                'date_range_days': date_range,
                'date_count': data[date_column].nunique()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def check_symbol_coverage(
        self,
        data: pd.DataFrame,
        symbol_column: str = 'symbol',
        date_column: str = 'date',
        expected_symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        检查股票代码覆盖率
        
        Args:
            data: 数据框
            symbol_column: 股票代码列名
            date_column: 日期列名
            expected_symbols: 预期的股票代码列表
            
        Returns:
            Dict[str, Any]: 股票代码覆盖率信息
        """
        if symbol_column not in data.columns:
            return {'error': f"Symbol column '{symbol_column}' not found"}
        
        actual_symbols = data[symbol_column].unique().tolist()
        symbol_count = len(actual_symbols)
        
        result = {
            'symbol_count': symbol_count,
            'actual_symbols': actual_symbols
        }
        
        if expected_symbols is not None:
            missing_symbols = [s for s in expected_symbols if s not in actual_symbols]
            extra_symbols = [s for s in actual_symbols if s not in expected_symbols]
            coverage_ratio = len([s for s in expected_symbols if s in actual_symbols]) / len(expected_symbols) if expected_symbols else 0
            
            result.update({
                'expected_symbol_count': len(expected_symbols),
                'missing_symbol_count': len(missing_symbols),
                'missing_symbols': missing_symbols,
                'extra_symbol_count': len(extra_symbols),
                'extra_symbols': extra_symbols,
                'coverage_ratio': coverage_ratio
            })
        
        if date_column in data.columns:
            # 检查每个股票的日期覆盖率
            date_coverage = {}
            all_dates = data[date_column].unique()
            for symbol in actual_symbols:
                symbol_dates = data[data[symbol_column] == symbol][date_column].unique()
                coverage = len(symbol_dates) / len(all_dates) if len(all_dates) > 0 else 0
                date_coverage[symbol] = coverage
            
            result['date_coverage'] = date_coverage
            result['avg_date_coverage'] = np.mean(list(date_coverage.values())) if date_coverage else 0
        
        return result
    
    def run_all_checks(
        self,
        data: pd.DataFrame,
        date_column: str = 'date',
        symbol_column: str = 'symbol',
        expected_symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        运行所有检查
        
        Args:
            data: 数据框
            date_column: 日期列名
            symbol_column: 股票代码列名
            expected_symbols: 预期的股票代码列表
            
        Returns:
            Dict[str, Any]: 所有检查结果
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': data.shape,
            'missing_values': self.check_missing_values(data),
            'duplicates': self.check_duplicates(data),
            'data_types': self.check_data_types(data),
        }
        
        # 检查数值列的异常值
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            result['outliers'] = self.check_outliers(data, numeric_columns)
        
        # 检查日期范围
        if date_column in data.columns:
            result['date_range'] = self.check_date_range(data, date_column)
        
        # 检查股票代码覆盖率
        if symbol_column in data.columns:
            result['symbol_coverage'] = self.check_symbol_coverage(
                data, symbol_column, date_column, expected_symbols
            )
        
        return result
```

在 `DataManager` 中集成数据质量监控功能：

```python
def __init__(self, config: Dict[str, Any]):
    # ... 其他初始化代码 ...
    
    # 初始化数据质量监控器
    self.quality_monitor = DataQualityMonitor()

def check_data_quality(
    self,
    data_model: Optional[DataModel] = None,
    date_column: str = 'date',
    symbol_column: str = 'symbol',
    expected_symbols: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    检查数据质量
    
    Args:
        data_model: 数据模型，默认为当前模型
        date_column: 日期列名
        symbol_column: 股票代码列名
        expected_symbols: 预期的股票代码列表
        
    Returns:
        Dict[str, Any]: 数据质量检查结果
    """
    if data_model is None:
        data_model = self.current_model
    
    if data_model is None:
        raise ValueError("No data model available")
    
    # 运行所有质量检查
    quality_report = self.quality_monitor.run_all_checks(
        data_model.data,
        date_column,
        symbol_column,
        expected_symbols
    )
    
    # 添加元数据
    quality_report['metadata'] = data_model.get_metadata()
    
    return quality_report
```

#### 2.2 数据导出功能

建议实现一个 `DataExporter` 类，用于将数据导出为不同格式：

```python
from typing import Dict, Any, Optional, Union
import pandas as pd
import json
import csv
import os
from pathlib import Path

class DataExporter:
    """数据导出器"""
    
    def __init__(self, export_dir: str = './exports'):
        """
        初始化数据导出器
        
        Args:
            export_dir: 导出目录
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
    
    def export_csv(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        filename: str,
        index: bool = False,
        **kwargs
    ) -> str:
        """
        导出为CSV格式
        
        Args:
            data: 数据框或字典
            filename: 文件名
            index: 是否包含索引
            **kwargs: 其他参数传递给to_csv
            
        Returns:
            str: 导出文件路径
        """
        filepath = self.export_dir / filename
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=index, **kwargs)
        elif isinstance(data, dict):
            # 如果是字典，尝试转换为DataFrame
            try:
                pd.DataFrame(data).to_csv(filepath, index=index, **kwargs)
            except Exception:
                # 如果转换失败，使用csv模块直接写入
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for key, value in data.items():
                        writer.writerow([key, value])
        else:
            raise TypeError("Data must be DataFrame or dict")
        
        return str(filepath)
    
    def export_excel(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        filename: str,
        index: bool = False,
        **kwargs
    ) -> str:
        """
        导出为Excel格式
        
        Args:
            data: 单个数据框或多个数据框的字典
            filename: 文件名
            index: 是否包含索引
            **kwargs: 其他参数传递给to_excel
            
        Returns:
            str: 导出文件路径
        """
        filepath = self.export_dir / filename
        
        if isinstance(data, pd.DataFrame):
            data.to_excel(filepath, index=index, **kwargs)
        elif isinstance(data, dict):
            with pd.ExcelWriter(filepath) as writer:
                for sheet_name, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_excel(writer, sheet_name=sheet_name, index=index, **kwargs)
                    else:
                        pd.DataFrame(df).to_excel(writer, sheet_name=sheet_name, index=index, **kwargs)
        else:
            raise TypeError("Data must be DataFrame or dict of DataFrames")
        
        return str(filepath)
    
    def export_json(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]],
        filename: str,
        orient: str = 'records',
        **kwargs
    ) -> str:
        """
        导出为JSON格式
        
        Args:
            data: 数据框或字典
            filename: 文件名
            orient: DataFrame转换为JSON的方向
            **kwargs: 其他参数传递给to_json
            
        Returns:
            str: 导出文件路径
        """
        filepath = self.export_dir / filename
        
        if isinstance(data, pd.DataFrame):
            data.to_json(filepath, orient=orient, **kwargs)
        elif isinstance(data, dict):
            with open(filepath, 'w') as f:
                json.dump(data, f, **kwargs)
        else:
            raise TypeError("Data must be DataFrame or dict")
        
        return str(filepath)
    
    def export_parquet(
        self,
        data: pd.DataFrame,
        filename: str,
        **kwargs
    ) -> str:
        """
        导出为Parquet格式
        
        Args:
            data: 
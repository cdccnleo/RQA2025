"""
数据处理模块，提供数据对齐、缺失值填充等通用功能
"""
from typing import Dict, List, Any, Optional, Union, Callable
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum

from src.infrastructure.utils.exceptions import DataProcessingError

logger = logging.getLogger(__name__)


class FillMethod(Enum):
    """缺失值填充方法枚举"""
    FORWARD = 'ffill'  # 前向填充
    BACKWARD = 'bfill'  # 后向填充
    MEAN = 'mean'      # 均值填充
    MEDIAN = 'median'  # 中位数填充
    ZERO = 'zero'      # 零值填充
    INTERPOLATE = 'interpolate'  # 插值填充
    CUSTOM = 'custom'  # 自定义填充


class DataProcessor:
    """
    数据处理器，提供数据对齐、缺失值填充等通用功能
    """
    def __init__(self):
        """初始化数据处理器"""
        logger.info("DataProcessor initialized")

    def align_data(
        self,
        data_frames: Dict[str, pd.DataFrame],
        freq: str = 'D',
        method: str = 'outer',
        fill_method: Optional[Union[str, Dict[str, str]]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        对齐多个数据框的时间索引

        Args:
            data_frames: 数据框字典，键为数据源名称，值为数据框
            freq: 重采样频率，如'D'表示日频，'H'表示小时频
            method: 对齐方法，'outer'表示并集，'inner'表示交集
            fill_method: 缺失值填充方法，可以是单一方法或按数据源指定的方法字典

        Returns:
            Dict[str, pd.DataFrame]: 对齐后的数据框字典

        Raises:
            DataProcessingError: 如果对齐失败
        """
        if not data_frames:
            return {}

        try:
            # 检查所有数据框是否都有DatetimeIndex
            for name, df in data_frames.items():
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        # 尝试将索引转换为DatetimeIndex
                        df.index = pd.to_datetime(df.index)
                        data_frames[name] = df
                    except Exception as e:
                        raise DataProcessingError(f"数据框 '{name}' 的索引无法转换为DatetimeIndex: {e}")

            # 确定对齐的时间范围
            if method == 'outer':
                # 并集：使用最早的开始日期和最晚的结束日期
                start_date = min(df.index.min() for df in data_frames.values())
                end_date = max(df.index.max() for df in data_frames.values())
            elif method == 'inner':
                # 交集：使用最晚的开始日期和最早的结束日期
                start_date = max(df.index.min() for df in data_frames.values())
                end_date = min(df.index.max() for df in data_frames.values())
            else:
                raise DataProcessingError(f"不支持的对齐方法: {method}")

            # 创建完整的时间索引
            full_index = pd.date_range(start=start_date, end=end_date, freq=freq)

            # 对齐数据框
            aligned_frames = {}
            for name, df in data_frames.items():
                # 重新索引
                aligned_df = df.reindex(full_index)

                # 如果指定了填充方法，填充缺失值
                if fill_method:
                    # 确定该数据源的填充方法
                    if isinstance(fill_method, dict):
                        source_fill_method = fill_method.get(name, None)
                    else:
                        source_fill_method = fill_method

                    if source_fill_method:
                        aligned_df = self.fill_missing(aligned_df, method=source_fill_method)

                aligned_frames[name] = aligned_df

            return aligned_frames

        except Exception as e:
            if not isinstance(e, DataProcessingError):
                e = DataProcessingError(f"数据对齐失败: {e}")
            logger.error(str(e))
            raise e

    def fill_missing(
        self,
        df: pd.DataFrame,
        method: Union[str, FillMethod, Dict[str, Union[str, FillMethod]]],
        custom_func: Optional[Callable] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        填充数据框中的缺失值

        Args:
            df: 数据框
            method: 填充方法，可以是单一方法或按列指定的方法字典
            custom_func: 自定义填充函数，当method为'custom'时使用
            limit: 填充的最大连续缺失值数量

        Returns:
            pd.DataFrame: 填充后的数据框

        Raises:
            DataProcessingError: 如果填充失败
        """
        if df.empty:
            return df

        try:
            # 创建数据框的副本
            filled_df = df.copy()

            # 如果method是字典，按列分别填充
            if isinstance(method, dict):
                for col, col_method in method.items():
                    if col in filled_df.columns:
                        filled_df[col] = self._fill_series(
                            filled_df[col],
                            col_method,
                            custom_func,
                            limit
                        )
            else:
                # 对所有列使用相同的填充方法
                for col in filled_df.columns:
                    filled_df[col] = self._fill_series(
                        filled_df[col],
                        method,
                        custom_func,
                        limit
                    )

            return filled_df

        except Exception as e:
            if not isinstance(e, DataProcessingError):
                e = DataProcessingError(f"缺失值填充失败: {e}")
            logger.error(str(e))
            raise e

    def _fill_series(
        self,
        series: pd.Series,
        method: Union[str, FillMethod],
        custom_func: Optional[Callable],
        limit: Optional[int]
    ) -> pd.Series:
        """
        填充Series中的缺失值

        Args:
            series: 数据序列
            method: 填充方法
            custom_func: 自定义填充函数
            limit: 填充的最大连续缺失值数量

        Returns:
            pd.Series: 填充后的数据序列
        """
        # 如果没有缺失值，直接返回
        if not series.isnull().any():
            return series

        # 转换方法为FillMethod枚举
        if isinstance(method, str):
            try:
                method = FillMethod(method)
            except ValueError:
                # 尝试匹配枚举值
                for fill_method in FillMethod:
                    if fill_method.value == method:
                        method = fill_method
                        break
                else:
                    raise DataProcessingError(f"不支持的填充方法: {method}")

        # 根据方法填充
        if method == FillMethod.FORWARD:
            return series.ffill(limit=limit)
        elif method == FillMethod.BACKWARD:
            return series.bfill(limit=limit)
        elif method == FillMethod.MEAN:
            return series.fillna(series.mean())
        elif method == FillMethod.MEDIAN:
            return series.fillna(series.median())
        elif method == FillMethod.ZERO:
            return series.fillna(0)
        elif method == FillMethod.INTERPOLATE:
            return series.interpolate(method='linear', limit=limit)
        elif method == FillMethod.CUSTOM:
            if custom_func is None:
                raise DataProcessingError("使用CUSTOM方法时必须提供custom_func")
            return custom_func(series)
        else:
            raise DataProcessingError(f"不支持的填充方法: {method}")

    def resample_data(
        self,
        df: pd.DataFrame,
        freq: str,
        method: str = 'mean',
        fill_method: Optional[str] = None
    ) -> pd.DataFrame:
        """
        重采样数据框

        Args:
            df: 数据框
            freq: 重采样频率，如'D'表示日频，'H'表示小时频
            method: 重采样方法，如'mean'、'sum'、'last'等
            fill_method: 重采样后的缺失值填充方法

        Returns:
            pd.DataFrame: 重采样后的数据框

        Raises:
            DataProcessingError: 如果重采样失败
        """
        if df.empty:
            return df

        try:
            # 确保索引是DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception as e:
                    raise DataProcessingError(f"数据框的索引无法转换为DatetimeIndex: {e}")

            # 根据数据类型选择合适的重采样方法
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

            # 对数值列应用指定的重采样方法
            if not numeric_cols.empty:
                numeric_df = df[numeric_cols].resample(freq)
                if method == 'mean':
                    numeric_df = numeric_df.mean()
                elif method == 'sum':
                    numeric_df = numeric_df.sum()
                elif method == 'last':
                    numeric_df = numeric_df.last()
                elif method == 'first':
                    numeric_df = numeric_df.first()
                elif method == 'max':
                    numeric_df = numeric_df.max()
                elif method == 'min':
                    numeric_df = numeric_df.min()
                else:
                    raise DataProcessingError(f"不支持的重采样方法: {method}")
            else:
                numeric_df = pd.DataFrame(index=pd.date_range(df.index.min(), df.index.max(), freq=freq))

            # 对非数值列使用最后一个值
            if not non_numeric_cols.empty:
                non_numeric_df = df[non_numeric_cols].resample(freq).last()
                # 合并数值和非数值列
                resampled_df = pd.concat([numeric_df, non_numeric_df], axis=1)
            else:
                resampled_df = numeric_df

            # 如果指定了填充方法，填充缺失值
            if fill_method:
                resampled_df = self.fill_missing(resampled_df, method=fill_method)

            return resampled_df

        except Exception as e:
            if not isinstance(e, DataProcessingError):
                e = DataProcessingError(f"数据重采样失败: {e}")
            logger.error(str(e))
            raise e

    def detect_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'zscore',
        threshold: float = 3.0,
        handle_method: str = 'mark'
    ) -> Union[pd.DataFrame, tuple]:
        """
        检测并处理异常值

        Args:
            df: 数据框
            method: 异常值检测方法，'zscore'、'iqr'或'mad'
            threshold: 异常值阈值
            handle_method: 处理方法，'mark'表示标记，'remove'表示移除，'fill'表示填充

        Returns:
            Union[pd.DataFrame, tuple]: 处理后的数据框，或者(处理后的数据框, 异常值掩码)

        Raises:
            DataProcessingError: 如果异常值检测失败
        """
        if df.empty:
            return df if handle_method != 'mark' else (df, pd.DataFrame(False, index=df.index, columns=df.columns))

        try:
            # 只对数值列进行异常值检测
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if numeric_cols.empty:
                return df if handle_method != 'mark' else (df, pd.DataFrame(False, index=df.index, columns=df.columns))

            # 创建异常值掩码
            outlier_mask = pd.DataFrame(False, index=df.index, columns=df.columns)

            # 对每个数值列进行异常值检测
            for col in numeric_cols:
                series = df[col].dropna()
                if len(series) == 0:
                    continue

                # 根据方法检测异常值
                if method == 'zscore':
                    # Z-score方法
                    z_scores = np.abs((series - series.mean()) / series.std())
                    outliers = z_scores > threshold
                elif method == 'iqr':
                    # IQR方法
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = (series < Q1 - threshold * IQR) | (series > Q3 + threshold * IQR)
                elif method == 'mad':
                    # MAD方法（中位数绝对偏差）
                    median = series.median()
                    mad = np.median(np.abs(series - median))
                    outliers = np.abs(series - median) / mad > threshold
                else:
                    raise DataProcessingError(f"不支持的异常值检测方法: {method}")
                
                # 更新异常值掩码
                outlier_mask.loc[outliers.index, col] = outliers
            
            # 根据处理方法处理异常值
            if handle_method == 'mark':
                # 仅标记异常值，返回原始数据框和异常值掩码
                return df, outlier_mask
            elif handle_method == 'remove':
                # 将异常值设为NaN
                result_df = df.copy()
                for col in numeric_cols:
                    result_df.loc[outlier_mask[col], col] = np.nan
                return result_df
            elif handle_method == 'fill':
                # 将异常值设为NaN，然后填充
                result_df = df.copy()
                for col in numeric_cols:
                    result_df.loc[outlier_mask[col], col] = np.nan
                # 使用前向填充方法填充
                result_df = self.fill_missing(result_df, method='ffill')
                return result_df
            else:
                raise DataProcessingError(f"不支持的异常值处理方法: {handle_method}")
            
        except Exception as e:
            if not isinstance(e, DataProcessingError):
                e = DataProcessingError(f"异常值检测失败: {e}")
            logger.error(str(e))
            raise e

    def normalize_data(
        self,
        df: pd.DataFrame,
        method: str = 'zscore',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        标准化数据
        
        Args:
            df: 数据框
            method: 标准化方法，'zscore'、'minmax'或'robust'
            columns: 要标准化的列，如果为None则标准化所有数值列
            
        Returns:
            pd.DataFrame: 标准化后的数据框
            
        Raises:
            DataProcessingError: 如果标准化失败
        """
        if df.empty:
            return df
        
        try:
            # 创建数据框的副本
            normalized_df = df.copy()
            
            # 确定要标准化的列
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns
            else:
                # 确保所有指定的列都存在
                missing_cols = [col for col in columns if col not in df.columns]
                if missing_cols:
                    raise DataProcessingError(f"以下列不存在: {missing_cols}")
                
                # 确保所有列都是数值类型
                non_numeric_cols = [col for col in columns if col not in df.select_dtypes(include=[np.number]).columns]
                if non_numeric_cols:
                    raise DataProcessingError(f"以下列不是数值类型: {non_numeric_cols}")
            
            # 对每列进行标准化
            for col in columns:
                series = df[col].dropna()
                if len(series) == 0:
                    continue
                
                # 根据方法进行标准化
                if method == 'zscore':
                    # Z-score标准化: (x - mean) / std
                    mean = series.mean()
                    std = series.std()
                    if std == 0:
                        # 避免除以零
                        normalized_df[col] = 0
                    else:
                        normalized_df[col] = (df[col] - mean) / std
                elif method == 'minmax':
                    # Min-Max标准化: (x - min) / (max - min)
                    min_val = series.min()
                    max_val = series.max()
                    if max_val == min_val:
                        # 避免除以零
                        normalized_df[col] = 0
                    else:
                        normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
                elif method == 'robust':
                    # 稳健标准化: (x - median) / IQR
                    median = series.median()
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    if iqr == 0:
                        # 避免除以零
                        normalized_df[col] = 0
                    else:
                        normalized_df[col] = (df[col] - median) / iqr
                else:
                    raise DataProcessingError(f"不支持的标准化方法: {method}")
            
            return normalized_df
            
        except Exception as e:
            if not isinstance(e, DataProcessingError):
                e = DataProcessingError(f"数据标准化失败: {e}")
            logger.error(str(e))
            raise e

    def merge_data(
        self,
        data_frames: Dict[str, pd.DataFrame],
        merge_on: str = 'index',
        how: str = 'outer',
        suffixes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        合并多个数据框
        
        Args:
            data_frames: 数据框字典，键为数据源名称，值为数据框
            merge_on: 合并依据，'index'表示按索引合并，或者指定列名
            how: 合并方式，'outer'、'inner'、'left'或'right'
            suffixes: 后缀列表，用于解决列名冲突
            
        Returns:
            pd.DataFrame: 合并后的数据框
            
        Raises:
            DataProcessingError: 如果合并失败
        """
        if not data_frames:
            return pd.DataFrame()
        
        try:
            # 生成后缀
            if suffixes is None:
                suffixes = [f"_{name}" for name in data_frames.keys()]
            
            # 确保后缀数量与数据框数量一致
            if len(suffixes) < len(data_frames) - 1:
                # 生成额外的后缀
                additional_suffixes = [f"_{i}" for i in range(len(suffixes), len(data_frames) - 1)]
                suffixes.extend(additional_suffixes)
            
            # 获取第一个数据框作为基础
            frames = list(data_frames.values())
            result = frames[0].copy()
            
            # 逐个合并其他数据框
            for i, df in enumerate(frames[1:]):
                suffix = suffixes[i]
                
                if merge_on == 'index':
                    # 按索引合并
                    if how == 'outer':
                        result = result.join(df, how='outer', lsuffix='', rsuffix=suffix)
                    elif how == 'inner':
                        result = result.join(df, how='inner', lsuffix='', rsuffix=suffix)
                    elif how == 'left':
                        result = result.join(df, how='left', lsuffix='', rsuffix=suffix)
                    elif how == 'right':
                        # 右连接需要反转操作
                        result = df.join(result, how='left', lsuffix=suffix, rsuffix='')
                    else:
                        raise DataProcessingError(f"不支持的合并方式: {how}")
                else:
                    # 按指定列合并
                    if merge_on not in result.columns:
                        raise DataProcessingError(f"列 '{merge_on}' 不在第一个数据框中")
                    if merge_on not in df.columns:
                        raise DataProcessingError(f"列 '{merge_on}' 不在第 {i+2} 个数据框中")
                    
                    result = pd.merge(
                        result, df,
                        on=merge_on,
                        how=how,
                        suffixes=('', suffix)
                    )
            
            return result
            
        except Exception as e:
            if not isinstance(e, DataProcessingError):
                e = DataProcessingError(f"数据合并失败: {e}")
            logger.error(str(e))
            raise e

    def filter_data(
        self,
        df: pd.DataFrame,
        conditions: Dict[str, Any],
        operator: str = 'and'
    ) -> pd.DataFrame:
        """
        根据条件筛选数据
        
        Args:
            df: 数据框
            conditions: 筛选条件字典，键为列名，值为条件值或(操作符, 值)元组
            operator: 条件组合操作符，'and'或'or'
            
        Returns:
            pd.DataFrame: 筛选后的数据框
            
        Raises:
            DataProcessingError: 如果筛选失败
        """
        if df.empty or not conditions:
            return df
        
        try:
            # 创建筛选掩码
            masks = []
            
            for col, condition in conditions.items():
                if col not in df.columns:
                    raise DataProcessingError(f"列 '{col}' 不存在")
                
                # 解析条件
                if isinstance(condition, tuple) and len(condition) == 2:
                    op, value = condition
                    
                    if op == '==':
                        mask = df[col] == value
                    elif op == '!=':
                        mask = df[col] != value
                    elif op == '>':
                        mask = df[col] > value
                    elif op == '>=':
                        mask = df[col] >= value
                    elif op == '<':
                        mask = df[col] < value
                    elif op == '<=':
                        mask = df[col] <= value
                    elif op == 'in':
                        if not isinstance(value, (list, tuple, set)):
                            raise DataProcessingError(f"'in'操作符需要可迭代的值")
                        mask = df[col].isin(value)
                    elif op == 'not in':
                        if not isinstance(value, (list, tuple, set)):
                            raise DataProcessingError(f"'not in'操作符需要可迭代的值")
                        mask = ~df[col].isin(value)
                    elif op == 'contains':
                        if not isinstance(value, str):
                            raise DataProcessingError(f"'contains'操作符需要字符串值")
                        mask = df[col].astype(str).str.contains(value)
                    elif op == 'between':
                        if not isinstance(value, (list, tuple)) or len(value) != 2:
                            raise DataProcessingError(f"'between'操作符需要包含两个元素的列表或元组")
                        mask = (df[col] >= value[0]) & (df[col] <= value[1])
                    else:
                        raise DataProcessingError(f"不支持的操作符: {op}")
                else:
                    # 默认使用等于操作符
                    mask = df[col] == condition
                
                masks.append(mask)
            
            # 组合所有掩码
            if operator.lower() == 'and':
                final_mask = masks[0]
                for mask in masks[1:]:
                    final_mask = final_mask & mask
            elif operator.lower() == 'or':
                final_mask = masks[0]
                for mask in masks[1:]:
                    final_mask = final_mask | mask
            else:
                raise DataProcessingError(f"不支持的操作符: {operator}")
            
            # 应用筛选
            return df[final_mask]
            
        except Exception as e:
            if not isinstance(e, DataProcessingError):
                e = DataProcessingError(f"数据筛选失败: {e}")
            logger.error(str(e))
            raise e
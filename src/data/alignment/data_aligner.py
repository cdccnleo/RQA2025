"""
数据对齐模块，专门处理多源数据的时间对齐问题
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json

from src.infrastructure.utils.exceptions import DataProcessingError
from src.data.processing.data_processor import DataProcessor, FillMethod

logger = logging.getLogger(__name__)


class AlignmentMethod(Enum):
    """对齐方法枚举"""
    OUTER = 'outer'  # 外部对齐（并集）
    INNER = 'inner'  # 内部对齐（交集）
    LEFT = 'left'    # 左对齐
    RIGHT = 'right'  # 右对齐


class FrequencyType(Enum):
    """频率类型枚举"""
    DAILY = 'D'      # 日频
    HOURLY = 'H'     # 小时频
    MINUTE = 'T'     # 分钟频
    SECOND = 'S'     # 秒频
    WEEKLY = 'W'     # 周频
    MONTHLY = 'M'    # 月频
    QUARTERLY = 'Q'  # 季频
    YEARLY = 'Y'     # 年频


class DataAligner:
    """
    数据对齐器，专门处理多源数据的时间对齐问题
    """
    def __init__(self):
        """初始化数据对齐器"""
        self.processor = DataProcessor()
        self.alignment_history = []
        logger.info("DataAligner initialized")

    def align_time_series(
        self,
        data_frames: Dict[str, pd.DataFrame],
        freq: Union[str, FrequencyType] = FrequencyType.DAILY,
        method: Union[str, AlignmentMethod] = AlignmentMethod.OUTER,
        fill_method: Optional[Union[str, Dict[str, str]]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        对齐多个时间序列数据
        
        Args:
            data_frames: 数据框字典，键为数据源名称，值为数据框
            freq: 重采样频率，可以是FrequencyType枚举或字符串
            method: 对齐方法，可以是AlignmentMethod枚举或字符串
            fill_method: 缺失值填充方法，可以是单一方法或按数据源指定的方法字典
            start_date: 起始日期，如果指定则覆盖method确定的起始日期
            end_date: 结束日期，如果指定则覆盖method确定的结束日期
            
        Returns:
            Dict[str, pd.DataFrame]: 对齐后的数据框字典
            
        Raises:
            DataProcessingError: 如果对齐失败
        """
        if not data_frames:
            return {}
        
        try:
            # 转换枚举为字符串
            if isinstance(freq, FrequencyType):
                freq = freq.value
            
            if isinstance(method, AlignmentMethod):
                method = method.value
            
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
            if start_date is not None:
                if isinstance(start_date, str):
                    start_date = pd.to_datetime(start_date)
            else:
                if method == 'outer':
                    # 并集：使用最早的开始日期
                    start_date = min(df.index.min() for df in data_frames.values())
                elif method == 'inner':
                    # 交集：使用最晚的开始日期
                    start_date = max(df.index.min() for df in data_frames.values())
                elif method == 'left':
                    # 左对齐：使用第一个数据框的开始日期
                    start_date = list(data_frames.values())[0].index.min()
                elif method == 'right':
                    # 右对齐：使用最后一个数据框的开始日期
                    start_date = list(data_frames.values())[-1].index.min()
                else:
                    raise DataProcessingError(f"不支持的对齐方法: {method}")
            
            if end_date is not None:
                if isinstance(end_date, str):
                    end_date = pd.to_datetime(end_date)
            else:
                if method == 'outer':
                    # 并集：使用最晚的结束日期
                    end_date = max(df.index.max() for df in data_frames.values())
                elif method == 'inner':
                    # 交集：使用最早的结束日期
                    end_date = min(df.index.max() for df in data_frames.values())
                elif method == 'left':
                    # 左对齐：使用第一个数据框的结束日期
                    end_date = list(data_frames.values())[0].index.max()
                elif method == 'right':
                    # 右对齐：使用最后一个数据框的结束日期
                    end_date = list(data_frames.values())[-1].index.max()
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
                        aligned_df = self.processor.fill_missing(aligned_df, method=source_fill_method)
                
                aligned_frames[name] = aligned_df
            
            # 记录对齐历史
            self._record_alignment(data_frames, aligned_frames, freq, method, fill_method, start_date, end_date)
            
            return aligned_frames
            
        except Exception as e:
            if not isinstance(e, DataProcessingError):
                e = DataProcessingError(f"时间序列对齐失败: {e}")
            logger.error(str(e))
            raise e

    def align_and_merge(
        self,
        data_frames: Dict[str, pd.DataFrame],
        freq: Union[str, FrequencyType] = FrequencyType.DAILY,
        method: Union[str, AlignmentMethod] = AlignmentMethod.OUTER,
        fill_method: Optional[Union[str, Dict[str, str]]] = None,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        suffixes: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        对齐并合并多个时间序列数据
        
        Args:
            data_frames: 数据框字典，键为数据源名称，值为数据框
            freq: 重采样频率，可以是FrequencyType枚举或字符串
            method: 对齐方法，可以是AlignmentMethod枚举或字符串
            fill_method: 缺失值填充方法，可以是单一方法或按数据源指定的方法字典
            start_date: 起始日期，如果指定则覆盖method确定的起始日期
            end_date: 结束日期，如果指定则覆盖method确定的结束日期
            suffixes: 后缀列表，用于解决列名冲突
            
        Returns:
            pd.DataFrame: 对齐并合并后的数据框
            
        Raises:
            DataProcessingError: 如果对齐或合并失败
        """
        try:
            # 先对齐
            aligned_frames = self.align_time_series(
                data_frames,
                freq=freq,
                method=method,
                fill_method=fill_method,
                start_date=start_date,
                end_date=end_date
            )
            
            # 再合并
            merged_df = self.processor.merge_data(
                aligned_frames,
                merge_on='index',
                how='outer' if method == 'outer' or method == AlignmentMethod.OUTER else 'inner',
                suffixes=suffixes
            )
            
            return merged_df
            
        except Exception as e:
            if not isinstance(e, DataProcessingError):
                e = DataProcessingError(f"对齐并合并失败: {e}")
            logger.error(str(e))
            raise e

    def align_to_reference(
        self,
        reference_df: pd.DataFrame,
        target_dfs: Dict[str, pd.DataFrame],
        freq: Optional[Union[str, FrequencyType]] = None,
        fill_method: Optional[Union[str, Dict[str, str]]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        将目标数据框对齐到参考数据框
        
        Args:
            reference_df: 参考数据框
            target_dfs: 目标数据框字典
            freq: 重采样频率，如果为None则使用参考数据框的频率
            fill_method: 缺失值填充方法
            
        Returns:
            Dict[str, pd.DataFrame]: 对齐后的数据框字典
            
        Raises:
            DataProcessingError: 如果对齐失败
        """
        try:
            # 确保参考数据框有DatetimeIndex
            if not isinstance(reference_df.index, pd.DatetimeIndex):
                try:
                    reference_df.index = pd.to_datetime(reference_df.index)
                except Exception as e:
                    raise DataProcessingError(f"参考数据框的索引无法转换为DatetimeIndex: {e}")
            
            # 确定频率
            if freq is None:
                # 尝试推断参考数据框的频率
                inferred_freq = pd.infer_freq(reference_df.index)
                if inferred_freq is None:
                    # 如果无法推断，默认使用日频
                    freq = 'D'
                    logger.warning("无法推断参考数据框的频率，使用默认频率'D'")
                else:
                    freq = inferred_freq
            elif isinstance(freq, FrequencyType):
                freq = freq.value
            
            # 对齐目标数据框到参考数据框的索引
            aligned_frames = {}
            for name, df in target_dfs.items():
                # 确保目标数据框有DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index)
                    except Exception as e:
                        raise DataProcessingError(f"数据框 '{name}' 的索引无法转换为DatetimeIndex: {e}")
                
                # 重新索引
                aligned_df = df.reindex(reference_df.index)
                
                # 如果指定了填充方法，填充缺失值
                if fill_method:
                    # 确定该数据源的填充方法
                    if isinstance(fill_method, dict):
                        source_fill_method = fill_method.get(name, None)
                    else:
                        source_fill_method = fill_method
                    
                    if source_fill_method:
                        aligned_df = self.processor.fill_missing(aligned_df, method=source_fill_method)
                
                aligned_frames[name] = aligned_df
            
            # 添加参考数据框
            aligned_frames['reference'] = reference_df
            
            return aligned_frames
            
        except Exception as e:
            if not isinstance(e, DataProcessingError):
                e = DataProcessingError(f"对齐到参考数据框失败: {e}")
            logger.error(str(e))
            raise e

    def align_multi_frequency(
        self,
        data_frames: Dict[str, pd.DataFrame],
        target_freq: Union[str, FrequencyType],
        resample_methods: Optional[Dict[str, str]] = None,
        fill_method: Optional[Union[str, Dict[str, str]]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        对齐多个不同频率的数据框到目标频率
        
        Args:
            data_frames: 数据框字典，键为数据源名称，值为数据框
            target_freq: 目标频率
            resample_methods: 重采样方法字典，键为数据源名称，值为重采样方法
            fill_method: 缺失值填充方法
            
        Returns:
            Dict[str, pd.DataFrame]: 对齐后的数据框字典
            
        Raises:
            DataProcessingError: 如果对齐失败
        """
        if not data_frames:
            return {}
        
        try:
            # 转换枚举为字符串
            if isinstance(target_freq, FrequencyType):
                target_freq = target_freq.value
            
            # 默认重采样方法
            if resample_methods is None:
                resample_methods = {}
            
            # 对齐数据框
            aligned_frames = {}
            for name, df in data_frames.items():
                # 确保数据框有DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index)
                    except Exception as e:
                        raise DataProcessingError(f"数据框 '{name}' 的索引无法转换为DatetimeIndex: {e}")
                
                # 获取重采样方法
                resample_method = resample_methods.get(name, 'mean')
                
                # 重采样到目标频率
                resampled_df = self.processor.resample_data(
                    df,
                    freq=target_freq,
                    method=resample_method,
                    fill_method=fill_method[name] if isinstance(fill_method, dict) else fill_method
                )
                
                aligned_frames[name] = resampled_df
            
            # 确保所有数据框有相同的索引
            # 找出最大的时间范围
            min_date = min(df.index.min() for df in aligned_frames.values())
            max_date = max(df.index.max() for df in aligned_frames.values())
            full_index = pd.date_range(start=min_date, end=max_date, freq=target_freq)
            
            # 重新索引所有数据框
            for name, df in aligned_frames.items():
                aligned_frames[name] = df.reindex(full_index)
                
                # 如果指定了填充方法，填充缺失值
                if fill_method:
                    # 确定该数据源的填充方法
                    if isinstance(fill_method, dict):
                        source_fill_method = fill_method.get(name, None)
                    else:
                        source_fill_method = fill_method
                    
                    if source_fill_method:
                        aligned_frames[name] = self.processor.fill_missing(aligned_frames[name], method=source_fill_method)
            
            return aligned_frames
            
        except Exception as e:
            if not isinstance(e, DataProcessingError):
                e = DataProcessingError(f"多频率数据对齐失败: {e}")
            logger.error(str(e))
            raise e

    def get_alignment_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        获取对齐历史记录
        
        Args:
            limit: 返回的记录数量限制
            
        Returns:
            List[Dict[str, Any]]: 对齐历史记录列表
        """
        if limit is not None:
            return self.alignment_history[-limit:]
        return self.alignment_history

    def _record_alignment(
        self,
        input_frames: Dict[str, pd.DataFrame],
        output_frames: Dict[str, pd.DataFrame],
        freq: Union[str, FrequencyType],
        method: Union[str, AlignmentMethod],
        fill_method: Optional[Union[str, Dict[str, str]]],
        start_date: datetime,
        end_date: datetime
    ) -> None:
        """
        记录对齐操作
        
        Args:
            input_frames: 输入数据框字典
            output_frames: 输出数据框字典
            freq: 频率
            method: 对齐方法
            fill_method: 填充方法
            start_date: 起始日期
            end_date: 结束日期
        """
        # 创建记录
        record = {
            'timestamp': datetime.now().isoformat(),
            'input_sources': list(input_frames.keys()),
            'output_sources': list(output_frames.keys()),
            'freq': freq.value if isinstance(freq, FrequencyType) else freq,
            'method': method.value if isinstance(method, AlignmentMethod) else method,
            'fill_method': fill_method,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'input_shapes': {name: df.shape for name, df in input_frames.items()},
            'output_shapes': {name: df.shape for name, df in output_frames.items()}
        }
        
        # 添加到历史记录
        self.alignment_history.append(record)
        
        # 限制历史记录长度
        max_history = 100
        if len(self.alignment_history) > max_history:
            self.alignment_history = self.alignment_history[-max_history:]

    def save_alignment_history(self, file_path: Union[str, Path]) -> None:
        """
        保存对齐历史记录到文件
        
        Args:
            file_path: 文件路径
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump(self.alignment_history, f, indent=2)
                
            logger.info(f"对齐历史记录已保存到 {file_path}")
        except Exception as e:
            logger.error(f"保存对齐历史记录失败: {e}")

    def load_alignment_history(self, file_path: Union[str, Path]) -> None:
        """
        从文件加载对齐历史记录
        
        Args:
            file_path: 文件路径
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.warning(f"文件 {file_path} 不存在")
                return
            
            with open(file_path, 'r') as f:
                self.alignment_history = json.load(f)
                
            logger.info(f"从 {file_path} 加载了 {len(self.alignment_history)} 条对齐历史记录")
        except Exception as e:
            logger.error(f"加载对齐历史记录失败: {e}")
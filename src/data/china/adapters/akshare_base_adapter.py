"""
AKShare统一适配器基类
提供AKShare各类数据采集的统一接口和通用功能
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class AKShareBaseAdapter(ABC):
    """AKShare适配器基类"""
    
    def __init__(self, source_config: Dict[str, Any]):
        """
        初始化适配器
        
        Args:
            source_config: 数据源配置字典
        """
        self.source_config = source_config
        self.source_id = source_config.get("id", "unknown")
        self.config = source_config.get("config", {})
        self._ak = None
    
    def _get_akshare(self):
        """
        延迟导入akshare
        
        Returns:
            akshare模块
            
        Raises:
            ImportError: 如果akshare未安装
        """
        if self._ak is None:
            try:
                import akshare as ak
                self._ak = ak
            except ImportError:
                logger.error("akshare未安装，无法采集数据")
                raise ImportError("akshare未安装，请运行: pip install akshare")
        return self._ak
    
    @abstractmethod
    async def collect(self, request_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        采集数据（子类必须实现）
        
        Args:
            request_data: 请求参数字典
            
        Returns:
            采集的数据列表
        """
        pass
    
    def _parse_date_range(self, request_data: Optional[Dict[str, Any]] = None) -> tuple:
        """
        解析日期范围
        
        Args:
            request_data: 请求参数字典
            
        Returns:
            (start_date, end_date) 元组
        """
        if request_data:
            start_date = request_data.get("start_date")
            end_date = request_data.get("end_date")
        else:
            # 默认最近30天
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
        
        if isinstance(start_date, str):
            if "-" in start_date:
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            else:
                start_date = datetime.strptime(start_date, "%Y%m%d")
        elif start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        
        if isinstance(end_date, str):
            if "-" in end_date:
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            else:
                end_date = datetime.strptime(end_date, "%Y%m%d")
        elif end_date is None:
            end_date = datetime.now()
        
        return start_date, end_date
    
    def _date_to_str(self, date_obj: datetime, format_type: str = "akshare") -> str:
        """
        将日期对象转换为字符串
        
        Args:
            date_obj: 日期对象
            format_type: 格式类型 ("akshare" 或 "standard")
            
        Returns:
            日期字符串
        """
        if format_type == "akshare":
            return date_obj.strftime("%Y%m%d")
        else:
            return date_obj.strftime("%Y-%m-%d")
    
    def _normalize_dataframe(self, df: pd.DataFrame, symbol: str = None, data_type: str = "stock") -> List[Dict[str, Any]]:
        """
        标准化DataFrame为字典列表
        
        Args:
            df: DataFrame对象
            symbol: 标的代码
            data_type: 数据类型
            
        Returns:
            标准化的数据字典列表
        """
        if df is None or df.empty:
            return []
        
        records = []
        for idx, row in df.iterrows():
            record = self._row_to_dict(row, df.columns, symbol, data_type)
            if record:
                records.append(record)
        return records
    
    def _row_to_dict(self, row: pd.Series, columns: pd.Index, symbol: str = None, data_type: str = "stock") -> Optional[Dict[str, Any]]:
        """
        将DataFrame行转换为字典（支持中英文列名）
        
        Args:
            row: DataFrame行
            columns: DataFrame列名
            symbol: 标的代码
            data_type: 数据类型
            
        Returns:
            数据字典，如果转换失败则返回None
        """
        # 子类可以重写此方法以自定义转换逻辑
        return {}
    
    def _get_value(self, row: pd.Series, columns: pd.Index, *keys) -> Any:
        """
        从row中获取值，支持多个可能的列名（中英文兼容）
        
        Args:
            row: DataFrame行
            columns: DataFrame列名
            *keys: 可能的列名列表
            
        Returns:
            找到的值，如果未找到则返回None
        """
        for key in keys:
            if key in columns:
                val = row[key]
                # 检查是否为NaN
                if pd.isna(val):
                    return None
                if val is None or (isinstance(val, float) and val != val):  # NaN检查
                    return None
                return val
        return None


__all__ = ['AKShareBaseAdapter']


"""数据模型模块

提供数据模型类，用于数据导出和元数据管理
"""

import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime


class DataModel:
    """数据模型类"""
    
    def __init__(self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None):
        """初始化数据模型
        
        Args:
            data: 数据DataFrame
            metadata: 元数据字典
        """
        self.data = data
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """设置元数据
        
        Args:
            metadata: 元数据字典
        """
        self.metadata.update(metadata)
        self.updated_at = datetime.now()
    
    def get_metadata(self) -> Dict[str, Any]:
        """获取元数据
        
        Returns:
            元数据字典
        """
        return self.metadata.copy()
    
    def get_data(self) -> pd.DataFrame:
        """获取数据
        
        Returns:
            数据DataFrame
        """
        return self.data.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            包含数据和元数据的字典
        """
        return {
            'data': self.data.to_dict('records'),
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    def __len__(self) -> int:
        """返回数据长度"""
        return len(self.data)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return f"DataModel(data_shape={self.data.shape}, metadata_keys={list(self.metadata.keys())})" 
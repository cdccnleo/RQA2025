#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
技术指标处理器
负责计算和管理技术指标
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from ..feature_manager import FeatureManager


class TechnicalProcessor:
    """技术指标处理器"""
    
    def __init__(self, feature_manager: FeatureManager):
        self.feature_manager = feature_manager
        self.indicators = {}
        
    def calculate_ma(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """计算移动平均线
        
        Args:
            data: 价格数据
            window: 窗口大小
            
        Returns:
            移动平均线序列
        """
        return data['close'].rolling(window=window).mean()
    
    def calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """计算相对强弱指数
        
        Args:
            data: 价格数据
            window: 窗口大小
            
        Returns:
            RSI序列
        """
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """计算MACD指标
        
        Args:
            data: 价格数据
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            
        Returns:
            MACD指标字典
        """
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """计算布林带
        
        Args:
            data: 价格数据
            window: 窗口大小
            num_std: 标准差倍数
            
        Returns:
            布林带指标字典
        """
        ma = data['close'].rolling(window=window).mean()
        std = data['close'].rolling(window=window).std()
        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)
        
        return {
            'upper': upper_band,
            'middle': ma,
            'lower': lower_band
        }
    
    def register_indicators(self):
        """向特征管理器注册技术指标"""
        self.feature_manager.register(
            name='ma_20',
            calculator=lambda data: self.calculate_ma(data, 20),
            description='20日移动平均线'
        )
        
        self.feature_manager.register(
            name='rsi_14',
            calculator=lambda data: self.calculate_rsi(data, 14),
            description='14日相对强弱指数'
        )
        
        self.feature_manager.register(
            name='macd',
            calculator=lambda data: self.calculate_macd(data)['macd'],
            description='MACD指标'
        )
        
        self.feature_manager.register(
            name='bollinger_upper',
            calculator=lambda data: self.calculate_bollinger_bands(data)['upper'],
            description='布林带上轨'
        ) 
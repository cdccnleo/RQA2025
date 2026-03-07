import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标处理器

from src.infrastructure.logging.core.unified_logger import get_unified_logger
负责计算各种技术指标，包括移动平均、RSI、MACD等。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

# from ...feature_config import TechnicalIndicatorType, TechnicalParams
from ..base_processor import BaseFeatureProcessor, ProcessorConfig


logger = logging.getLogger(__name__)


class BaseTechnicalProcessor(ABC):

    """技术指标处理器基类"""

    def __init__(self):

        pass

    @abstractmethod
    def calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算技术指标"""

    @abstractmethod
    def get_name(self) -> str:
        """获取指标名称"""


class SMAProcessor(BaseTechnicalProcessor):

    """简单移动平均处理器"""

    def calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算SMA"""
        period = params.get('period', 20)
        column = params.get('column', 'close')

        if column not in data.columns:
            raise ValueError(f"列 {column} 不存在")

        return data[column].rolling(window=period).mean()

    def get_name(self) -> str:

        return "sma"


class EMAProcessor(BaseTechnicalProcessor):

    """指数移动平均处理器"""

    def calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算EMA"""
        period = params.get('period', 20)
        column = params.get('column', 'close')

        if column not in data.columns:
            raise ValueError(f"列 {column} 不存在")

        return data[column].ewm(span=period).mean()

    def get_name(self) -> str:

        return "ema"


class RSIProcessor(BaseTechnicalProcessor):

    """相对强弱指数处理器"""

    def calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算RSI"""
        period = params.get('period', 14)
        column = params.get('column', 'close')

        if column not in data.columns:
            raise ValueError(f"列 {column} 不存在")

        delta = data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def get_name(self) -> str:

        return "rsi"


class MACDProcessor(BaseTechnicalProcessor):

    """MACD处理器"""

    def calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """计算MACD"""
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)
        column = params.get('column', 'close')

        if column not in data.columns:
            raise ValueError(f"列 {column} 不存在")

        ema_fast = data[column].ewm(span=fast_period).mean()
        ema_slow = data[column].ewm(span=slow_period).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def get_name(self) -> str:

        return "macd"


class BollingerBandsProcessor(BaseTechnicalProcessor):

    """布林带处理器"""

    def calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """计算布林带"""
        period = params.get('period', 20)
        std_dev = params.get('std_dev', 2.0)
        column = params.get('column', 'close')

        if column not in data.columns:
            raise ValueError(f"列 {column} 不存在")

        sma = data[column].rolling(window=period).mean()
        std = data[column].rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }

    def get_name(self) -> str:

        return "bbands"


class ATRProcessor(BaseTechnicalProcessor):

    """平均真实波幅处理器"""

    def calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
        """计算ATR"""
        period = params.get('period', 14)

        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺失必要列: {missing_columns}")

        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def get_name(self) -> str:

        return "atr"


class StochProcessor(BaseTechnicalProcessor):

    """随机指标(KDJ)处理器"""

    def calculate(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """计算KDJ指标"""
        k_period = params.get('k_period', 9)
        d_period = params.get('d_period', 3)
        j_period = params.get('j_period', 3)

        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺失必要列: {missing_columns}")

        # 计算 RSV (Raw Stochastic Value)
        lowest_low = data['low'].rolling(window=k_period).min()
        highest_high = data['high'].rolling(window=k_period).max()
        rsv = (data['close'] - lowest_low) / (highest_high - lowest_low) * 100

        # 计算 K 值
        k = rsv.ewm(com=d_period-1, adjust=False).mean()

        # 计算 D 值
        d = k.ewm(com=j_period-1, adjust=False).mean()

        # 计算 J 值
        j = 3 * k - 2 * d

        return {
            'k': k,
            'd': d,
            'j': j
        }

    def get_name(self) -> str:
        return "stoch"


class TechnicalProcessor(BaseFeatureProcessor):

    """技术指标处理器主类"""

    def __init__(self, config: Optional[ProcessorConfig] = None):

        if config is None:
            config = ProcessorConfig(
                processor_type="technical",
                feature_params={
                    "sma_periods": [5, 10, 20, 50],
                    "ema_periods": [12, 26],
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "bb_period": 20,
                    "bb_std": 2,
                    "atr_period": 14
                }
            )

        super().__init__(config)
        self.processors = {
            'sma': SMAProcessor(),
            'ema': EMAProcessor(),
            'rsi': RSIProcessor(),
            'macd': MACDProcessor(),
            'bbands': BollingerBandsProcessor(),
            'atr': ATRProcessor(),
            'stoch': StochProcessor()  # 添加 KDJ 处理器
        }

        # 添加indicators字典用于存储计算结果
        self.indicators = {}

        # 添加logger属性
        self.logger = logger

    def process(self, request) -> pd.DataFrame:
        """
        处理技术指标特征
        
        从 request.config 中获取 technical_indicators 配置
        """
        import pandas as pd
        
        # 验证输入数据
        if request.data.empty:
            raise ValueError("输入数据为空")
        
        # 获取配置中的 technical_indicators
        config = request.config if hasattr(request, 'config') else {}
        if isinstance(config, dict):
            technical_indicators = config.get('technical_indicators', [])
        else:
            technical_indicators = getattr(config, 'technical_indicators', [])
        
        # 如果没有指定指标，使用所有可用指标
        if not technical_indicators:
            technical_indicators = self.list_features()
        
        self.logger.info(f"🚀 TechnicalProcessor 处理指标: {technical_indicators}")
        
        # 🚀 关键修复：指标名称映射
        indicator_mapping = {
            'sma': 'sma',
            'ema': 'ema',
            'rsi': 'rsi',
            'macd': 'macd',
            'boll': 'bbands',  # BOLL 映射到 bbands
            'bbands': 'bbands',
            'kdj': 'stoch',    # KDJ 映射到 stoch（随机指标）
            'stoch': 'stoch',
            'atr': 'atr'
        }
        
        # 处理特征
        result_data = request.data.copy()
        for indicator_name in technical_indicators:
            # 映射指标名称
            mapped_name = indicator_mapping.get(indicator_name, indicator_name)
            
            if mapped_name in self.processors:
                try:
                    processor = self.processors[mapped_name]
                    result = processor.calculate(result_data, {})
                    
                    if isinstance(result, pd.Series):
                        # 单个特征
                        result_data[f'{indicator_name}'] = result
                    elif isinstance(result, dict):
                        # 多个特征（如 MACD）
                        for key, value in result.items():
                            if isinstance(value, pd.Series):
                                result_data[f'{indicator_name}_{key}'] = value
                    
                    self.logger.info(f"✅ 计算指标 {indicator_name} (映射: {mapped_name}) 成功")
                except Exception as e:
                    self.logger.error(f"❌ 计算指标 {indicator_name} (映射: {mapped_name}) 失败: {e}")
            else:
                self.logger.warning(f"⚠️ 未知指标: {indicator_name} (映射: {mapped_name})")
        
        return result_data

    def calc_ma(self, data: pd.DataFrame = None, window: int = 20, price_col: str = 'close') -> pd.Series:
        """
        计算移动平均（calc_ma别名方法）

        Args:
            data: 输入数据，如果为None则使用默认数据
            window: 窗口大小
            price_col: 价格列名

        Returns:
            移动平均序列
        """
        return self.calculate_ma(data, window, price_col)

    def calculate_ma(self, data: pd.DataFrame = None, window: int = 20, price_col: str = 'close') -> pd.Series:
        """
        计算移动平均线

        Args:
            data: 输入数据，如果为None则使用默认数据
            window: 窗口大小
            price_col: 价格列名

        Returns:
            移动平均线序列
        """
        if data is None:
            # 创建示例数据用于测试
            data = pd.DataFrame({
                'close': np.secrets.randn(100).cumsum() + 100,
                'high': np.secrets.randn(100).cumsum() + 102,
                'low': np.secrets.randn(100).cumsum() + 98
            })

        if price_col not in data.columns:
            raise ValueError(f"列 {price_col} 不存在")

        if window <= 0:
            raise ValueError("窗口大小必须大于0")

        # 检查数据是否全为NaN
        if data[price_col].isna().all():
            raise ValueError("数据全为NaN，无法计算移动平均")

        return data[price_col].rolling(window=window).mean()

    def calculate_rsi(self, data: pd.DataFrame = None, window: int = 14, price_col: str = 'close') -> pd.Series:
        """
        计算RSI指标

        Args:
            data: 输入数据，如果为None则使用默认数据
            window: 窗口大小
            price_col: 价格列名

        Returns:
            RSI指标序列
        """
        if data is None:
            # 创建示例数据用于测试
            data = pd.DataFrame({
                'close': np.secrets.randn(100).cumsum() + 100,
                'high': np.secrets.randn(100).cumsum() + 102,
                'low': np.secrets.randn(100).cumsum() + 98
            })

        # 处理numpy数组输入
        if isinstance(data, np.ndarray):
            data = pd.DataFrame({'close': data})
            price_col = 'close'

        if price_col not in data.columns:
            raise ValueError(f"列 {price_col} 不存在")

        if window <= 0:
            raise ValueError("窗口大小必须大于0")

        # 检查数据是否全为NaN
        if data[price_col].isna().all():
            raise ValueError("数据全为NaN，无法计算RSI")

        prices = data[price_col]
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_macd(self, data: pd.DataFrame = None, fast: int = 12, slow: int = 26, signal: int = 9, price_col: str = 'close') -> Dict[str, pd.Series]:
        """
        计算MACD指标

        Args:
            data: 输入数据，如果为None则使用默认数据
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期
            price_col: 价格列名

        Returns:
            包含MACD、信号线和柱状图的字典
        """
        if data is None:
            # 创建示例数据用于测试
            data = pd.DataFrame({
                'close': np.secrets.randn(100).cumsum() + 100,
                'high': np.secrets.randn(100).cumsum() + 102,
                'low': np.secrets.randn(100).cumsum() + 98
            })

        # 处理numpy数组输入
        if isinstance(data, np.ndarray):
            data = pd.DataFrame({'close': data})
            price_col = 'close'

        if price_col not in data.columns:
            raise ValueError(f"列 {price_col} 不存在")

        # 检查数据是否全为NaN
        if data[price_col].isna().all():
            raise ValueError("数据全为NaN，无法计算MACD")

        prices = data[price_col]

        # 计算EMA
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()

        # 计算MACD线
        macd_line = ema_fast - ema_slow

        # 计算信号线
        signal_line = macd_line.ewm(span=signal).mean()

        # 计算柱状图
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_bollinger_bands(self, data: pd.DataFrame = None, window: int = 20, std: float = 2.0, price_col: str = 'close') -> Dict[str, pd.Series]:
        """
        计算布林带

        Args:
            data: 输入数据，如果为None则使用默认数据
            window: 窗口大小
            std: 标准差倍数
            price_col: 价格列名

        Returns:
            包含上轨、中轨、下轨的字典
        """
        if data is None:
            # 创建示例数据用于测试
            data = pd.DataFrame({
                'close': np.secrets.randn(100).cumsum() + 100,
                'high': np.secrets.randn(100).cumsum() + 102,
                'low': np.secrets.randn(100).cumsum() + 98
            })

        if price_col not in data.columns:
            raise ValueError(f"列 {price_col} 不存在")

        # 检查数据是否全为NaN
        if data[price_col].isna().all():
            raise ValueError("数据全为NaN，无法计算布林带")

        prices = data[price_col]

        # 计算中轨（移动平均）
        middle_band = prices.rolling(window=window).mean()

        # 计算标准差
        std_dev = prices.rolling(window=window).std()

        # 计算上轨和下轨
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)

        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }

    def calculate_indicator(self, data: pd.DataFrame, indicator: str, params: Dict[str, Any]) -> Any:
        """
        计算指定技术指标

        Args:
            data: 输入数据
            indicator: 指标名称
            params: 参数

        Returns:
            计算结果
        """
        if indicator not in self.processors:
            raise ValueError(f"不支持的指标: {indicator}")

        try:
            processor = self.processors[indicator]
            result = processor.calculate(data, params)
            return result
        except Exception as e:
            self.logger.error(f"计算指标 {indicator} 失败: {e}")
            raise

    def calculate_multiple_indicators(self, data: pd.DataFrame, indicators: List[str],


                                      params: Dict[str, Any]) -> pd.DataFrame:
        """
        计算多个技术指标

        Args:
            data: 输入数据
            indicators: 指标列表
            params: 参数

        Returns:
            包含所有指标结果的DataFrame
        """
        results = pd.DataFrame(index=data.index)

        for indicator in indicators:
            try:
                result = self.calculate_indicator(data, indicator, params)

                if isinstance(result, dict):
                    # 处理返回多个值的指标（如MACD、布林带）
                    for key, value in result.items():
                        results[f"{indicator}_{key}"] = value
                else:
                    # 处理返回单个值的指标
                    results[indicator] = result

            except Exception as e:
                self.logger.warning(f"跳过指标 {indicator}: {e}")
                continue

        return results

    def get_supported_indicators(self) -> List[str]:
        """获取支持的指标列表"""
        return list(self.processors.keys())

    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证输入数据"""
        if data.empty:
            return False

        required_columns = ['close', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            return False

        return True

    def _compute_feature(self, data: pd.DataFrame, feature_name: str,


                         params: Dict[str, Any]) -> pd.Series:
        """计算单个特征"""
        if feature_name not in self.processors:
            return pd.Series(index=data.index, dtype=float)

        try:
            processor = self.processors[feature_name]
            result = processor.calculate(data, params)
            return result if isinstance(result, pd.Series) else pd.Series(index=data.index, dtype=float)
        except Exception as e:
            self.logger.error(f"计算特征 {feature_name} 失败: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """获取特征元数据"""
        if feature_name not in self.processors:
            return {}

        processor = self.processors[feature_name]
        return {
            "name": processor.get_name(),
            "type": "technical_indicator",
            "description": f"技术指标: {feature_name}",
            "parameters": self.config.feature_params.get(feature_name, {})
        }

    def _get_available_features(self) -> List[str]:
        """获取可用特征列表"""
        return list(self.processors.keys())


# 导出主要类
__all__ = [
    'BaseTechnicalProcessor',
    'SMAProcessor',
    'EMAProcessor',
    'RSIProcessor',
    'MACDProcessor',
    'BollingerBandsProcessor',
    'ATRProcessor',
    'TechnicalProcessor'
]

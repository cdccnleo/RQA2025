#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标插件示例

演示如何创建特征处理插件。
"""

import pandas as pd
from typing import Any, Dict
from src.features.plugins import BaseFeaturePlugin, PluginMetadata


class TechnicalIndicatorPlugin(BaseFeaturePlugin):
    """技术指标插件"""

    def _get_metadata(self) -> PluginMetadata:
        """获取插件元数据"""
        from src.features.plugins import PluginMetadata, PluginType

        return PluginMetadata(
            name="technical_indicator_plugin",
            version="1.0.0",
            description="技术指标计算插件，提供SMA、EMA、RSI等技术指标",
            author="RQA Team",
            plugin_type=PluginType.PROCESSOR,
            dependencies=["pandas", "numpy"],
            tags=["technical", "indicators", "finance"],
            config_schema={
                "sma_periods": {"type": list, "default": [5, 10, 20]},
                "ema_periods": {"type": list, "default": [12, 26]},
                "rsi_period": {"type": int, "default": 14},
                "macd_fast": {"type": int, "default": 12},
                "macd_slow": {"type": int, "default": 26},
                "macd_signal": {"type": int, "default": 9}
            },
            min_api_version="1.0.0",
            max_api_version="2.0.0"
        )

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        处理数据，计算技术指标

        Args:
            data: 输入数据（包含OHLCV）
            **kwargs: 额外参数

        Returns:
            包含技术指标的数据框
        """
        if data.empty:
            self.logger.warning("输入数据为空")
            return data

        # 获取配置
        sma_periods = self.config.get("sma_periods", [5, 10, 20])
        ema_periods = self.config.get("ema_periods", [12, 26])
        rsi_period = self.config.get("rsi_period", 14)
        macd_fast = self.config.get("macd_fast", 12)
        macd_slow = self.config.get("macd_slow", 26)
        macd_signal = self.config.get("macd_signal", 9)

        result = data.copy()

        # 计算SMA
        for period in sma_periods:
            if 'close' in data.columns:
                result[f'sma_{period}'] = data['close'].rolling(window=period).mean()

        # 计算EMA
        for period in ema_periods:
            if 'close' in data.columns:
                result[f'ema_{period}'] = data['close'].ewm(span=period).mean()

        # 计算RSI
        if 'close' in data.columns:
            result['rsi'] = self._calculate_rsi(data['close'], rsi_period)

        # 计算MACD
        if 'close' in data.columns:
            macd_data = self._calculate_macd(data['close'], macd_fast, macd_slow, macd_signal)
            result['macd'] = macd_data['macd']
            result['macd_signal'] = macd_data['signal']
            result['macd_histogram'] = macd_data['histogram']

        # 计算布林带
        if 'close' in data.columns:
            bb_data = self._calculate_bollinger_bands(data['close'])
            result['bb_upper'] = bb_data['upper']
            result['bb_middle'] = bb_data['middle']
            result['bb_lower'] = bb_data['lower']

        self.logger.info(f"技术指标计算完成，新增 {len(result.columns) - len(data.columns)} 个指标")
        return result

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12,
                        slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line

        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }

    def _calculate_bollinger_bands(self, prices: pd.Series,
                                   period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }

    def _get_capabilities(self) -> Dict[str, Any]:
        """获取插件能力"""
        return {
            "indicators": ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands"],
            "input_columns": ["close", "high", "low", "volume"],
            "output_columns": ["sma_*", "ema_*", "rsi", "macd", "macd_signal", "macd_histogram",
                               "bb_upper", "bb_middle", "bb_lower"],
            "configurable": True
        }

    def _validate_input(self, data: Any) -> bool:
        """验证输入数据"""
        if not isinstance(data, pd.DataFrame):
            return False

        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            self.logger.error(f"缺少必需列: {missing_columns}")
            return False

        return True

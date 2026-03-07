import logging
"""
Feature Processor Module

This module provides a concrete implementation of the base feature processor.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from .base_processor import BaseFeatureProcessor, ProcessorConfig

logger = logging.getLogger(__name__)


class FeatureProcessor(BaseFeatureProcessor):

    """Concrete feature processor implementation"""

    def __init__(self, config: Optional[ProcessorConfig] = None):
        """Initialize feature processor"""
        if config is None:
            config = ProcessorConfig(
                processor_type="general",
                feature_params={
                    "moving_averages": [5, 10, 20, 50],
                    "rsi_period": 14,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9,
                    "bollinger_period": 20,
                    "bollinger_std": 2
                }
            )

        super().__init__(config)
        self.logger = logger
        self._available_features = self._get_available_features()
        logger.info(f"FeatureProcessor initialized with {len(self._available_features)} features")

    def process(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """Process DataFrame input directly"""
        if data.empty:
            raise ValueError("输入数据为空")

        if features is None:
            features = self._available_features

        # 验证特征是否存在
        invalid_features = [f for f in features if f not in self._available_features]
        if invalid_features:
            raise ValueError(f"不支持的特征: {invalid_features}")

        # 处理特征
        result_data = data.copy()
        for feature_name in features:
            feature_values = self._compute_feature(
                result_data, feature_name, self.config.feature_params)
            result_data[f'feature_{feature_name}'] = feature_values

        return result_data

    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """Update configuration"""
        if hasattr(self.config, 'feature_params') and self.config.feature_params is not None:
            self.config.feature_params.update(config_updates)
        else:
            # 如果feature_params不存在或为None，创建一个新的
            self.config.feature_params = config_updates

    def _compute_feature(self, data: pd.DataFrame, feature_name: str,


                         params: Dict[str, Any]) -> pd.Series:
        """Compute a single feature"""
        if data.empty:
            return pd.Series(dtype=float)

        try:
            if feature_name == "sma":
                period = params.get("period", 20)
                return self._compute_sma(data, period)
            elif feature_name == "ema":
                period = params.get("period", 12)
                return self._compute_ema(data, period)
            elif feature_name == "rsi":
                period = params.get("period", 14)
                return self._compute_rsi(data, period)
            elif feature_name == "macd":
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                signal = params.get("signal", 9)
                return self._compute_macd(data, fast, slow, signal)
            elif feature_name == "bollinger_bands":
                period = params.get("period", 20)
                std_dev = params.get("std_dev", 2)
                return self._compute_bollinger_bands(data, period, std_dev)
            elif feature_name == "price_change":
                period = params.get("period", 1)
                return self._compute_price_change(data, period)
            elif feature_name == "volume_ratio":
                period = params.get("period", 10)
                return self._compute_volume_ratio(data, period)
            elif feature_name == "volatility":
                period = params.get("period", 20)
                return self._compute_volatility(data, period)
            else:
                logger.warning(f"Unknown feature: {feature_name}")
                return pd.Series(index=data.index, dtype=float)

        except Exception as e:
            logger.error(f"Error computing feature {feature_name}: {e}")
            return pd.Series(index=data.index, dtype=float)

    def _compute_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Compute Simple Moving Average"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        return data['close'].rolling(window=period).mean()

    def _compute_ema(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Compute Exponential Moving Average"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        return data['close'].ewm(span=period).mean()

    def _compute_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Compute Relative Strength Index"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index, dtype=float)

        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_macd(self, data: pd.DataFrame, fast: int, slow: int, signal: int) -> pd.Series:
        """Compute MACD"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index, dtype=float)

        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()

        return macd_line - signal_line

    def _compute_bollinger_bands(self, data: pd.DataFrame, period: int, std_dev: float) -> pd.Series:
        """Compute Bollinger Bands"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index, dtype=float)

        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        # Return the bandwidth
        return (upper_band - lower_band) / sma

    def _compute_price_change(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Compute price change percentage"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        return data['close'].pct_change(periods=period)

    def _compute_volume_ratio(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Compute volume ratio"""
        if 'volume' not in data.columns:
            return pd.Series(index=data.index, dtype=float)

        volume_ma = data['volume'].rolling(window=period).mean()
        return data['volume'] / volume_ma

    def _compute_volatility(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Compute price volatility"""
        if 'close' not in data.columns:
            return pd.Series(index=data.index, dtype=float)

        returns = data['close'].pct_change()
        return returns.rolling(window=period).std()

    def _get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """Get feature metadata"""
        metadata = {
            "name": feature_name,
            "type": "technical",
            "description": f"Technical indicator: {feature_name}",
            "parameters": self.config.feature_params.get(feature_name, {}),
            "created_at": datetime.now().isoformat(),
            "version": "1.0"
        }

        # Add specific metadata for each feature type
        if feature_name == "sma":
            metadata["description"] = "Simple Moving Average"
            metadata["parameters"] = {
                "period": self.config.feature_params.get("moving_averages", [20])}
        elif feature_name == "ema":
            metadata["description"] = "Exponential Moving Average"
            metadata["parameters"] = {"period": 12}
        elif feature_name == "rsi":
            metadata["description"] = "Relative Strength Index"
            metadata["parameters"] = {"period": self.config.feature_params.get("rsi_period", 14)}
        elif feature_name == "macd":
            metadata["description"] = "MACD (Moving Average Convergence Divergence)"
            metadata["parameters"] = {
                "fast": self.config.feature_params.get("macd_fast", 12),
                "slow": self.config.feature_params.get("macd_slow", 26),
                "signal": self.config.feature_params.get("macd_signal", 9)
            }
        elif feature_name == "bollinger_bands":
            metadata["description"] = "Bollinger Bands"
            metadata["parameters"] = {
                "period": self.config.feature_params.get("bollinger_period", 20),
                "std_dev": self.config.feature_params.get("bollinger_std", 2)
            }

        return metadata

    def _calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages"""
        if data.empty:
            return data

        result = data.copy()
        periods = self.config.feature_params.get("moving_averages", [5, 10, 20, 50])

        # 确保有收盘价数据
        if 'close' not in result.columns:
            logger.warning("No 'close' column found for moving averages calculation")
            return result

        # 计算SMA
        for period in periods:
            if len(result) >= period:
                sma_col = f"SMA_{period}"
                result[sma_col] = result['close'].rolling(window=period).mean()

        # 计算EMA
        for period in periods:
            if len(result) >= period:
                ema_col = f"EMA_{period}"
                result[ema_col] = result['close'].ewm(span=period, adjust=False).mean()

        logger.info(f"Calculated moving averages for periods: {periods}")
        return result

    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI indicator"""
        if data.empty or len(data) < period + 1:
            return data

        result = data.copy()

        # 计算价格变化
        delta = result['close'].diff()

        # 分离上涨和下跌
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # 计算RS和RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        result['RSI'] = rsi

        logger.info(f"Calculated RSI with period {period}")
        return result

    def _calculate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator"""
        if data.empty:
            return data

        result = data.copy()
        fast_period = self.config.feature_params.get("macd_fast", 12)
        slow_period = self.config.feature_params.get("macd_slow", 26)
        signal_period = self.config.feature_params.get("macd_signal", 9)

        if len(result) < slow_period:
            logger.warning(
                f"Insufficient data for MACD calculation. Need at least {slow_period} periods")
            return result

        # 计算快速和慢速EMA
        fast_ema = result['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = result['close'].ewm(span=slow_period, adjust=False).mean()

        # 计算MACD线
        result['MACD'] = fast_ema - slow_ema

        # 计算信号线
        result['MACD_Signal'] = result['MACD'].ewm(span=signal_period, adjust=False).mean()

        # 计算MACD柱状图
        result['MACD_Histogram'] = result['MACD'] - result['MACD_Signal']

        logger.info(
            f"Calculated MACD with fast={fast_period}, slow={slow_period}, signal={signal_period}")
        return result

    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        if data.empty:
            return data

        result = data.copy()
        period = self.config.feature_params.get("bollinger_period", 20)
        std_dev = self.config.feature_params.get("bollinger_std", 2)

        if len(result) < period:
            logger.warning(
                f"Insufficient data for Bollinger Bands calculation. Need at least {period} periods")
            return result

        # 计算移动平均线
        result['BB_Middle'] = result['close'].rolling(window=period).mean()

        # 计算标准差
        result['BB_Std'] = result['close'].rolling(window=period).std()

        # 计算布林带上轨和下轨
        result['BB_Upper'] = result['BB_Middle'] + (result['BB_Std'] * std_dev)
        result['BB_Lower'] = result['BB_Middle'] - (result['BB_Std'] * std_dev)

        # 计算布林带宽度
        result['BB_Width'] = (result['BB_Upper'] - result['BB_Lower']) / result['BB_Middle']

        logger.info(f"Calculated Bollinger Bands with period={period}, std_dev={std_dev}")
        return result

    def _get_available_features(self) -> List[str]:
        """Get available features list"""
        return [
            "sma",
            "ema",
            "rsi",
            "macd",
            "bollinger_bands",
            "price_change",
            "volume_ratio",
            "volatility"
        ]

    def process_data(self, data: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """Process data with specified features"""
        if data.empty:
            return pd.DataFrame()

        if features is None:
            features = self._available_features

        result = data.copy()

        for feature_name in features:
            if feature_name in self._available_features:
                feature_values = self._compute_feature(result, feature_name, {})
                result[f"{feature_name}"] = feature_values

        logger.info(f"Processed {len(features)} features for {len(data)} records")
        return result

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get feature processing summary"""
        return {
            "total_features": len(self._available_features),
            "available_features": self._available_features,
            "processor_type": self.processor_type,
            "config": self.config.feature_params
        }

    def list_features(self) -> List[Dict[str, Any]]:
        """列出可用特征"""
        available_features = self._get_available_features()

        feature_list = []
        for feature_name in available_features:
            feature_info = {
                "name": feature_name,
                "type": "technical_indicator",
                "description": self._get_feature_description(feature_name),
                "parameters": self._get_feature_parameters(feature_name)
            }
            feature_list.append(feature_info)

        return feature_list

    @property
    def feature_cache(self) -> Dict[str, Any]:
        """兼容旧版接口，暴露内部缓存"""
        return self._features_cache

    def validate_features(self, features: List[str]) -> tuple[bool, List[str]]:
        """验证特征列表"""
        available_features = self._get_available_features()
        errors = []

        for feature in features:
            if feature not in available_features:
                errors.append(f"特征 '{feature}' 不可用")

        return len(errors) == 0, errors

    def _get_feature_description(self, feature_name: str) -> str:
        """获取特征描述"""
        descriptions = {
            "sma": "简单移动平均线",
            "ema": "指数移动平均线",
            "rsi": "相对强弱指数",
            "macd": "MACD指标",
            "bollinger_bands": "布林带",
            "price_change": "价格变化率",
            "volume_ratio": "成交量比率",
            "volatility": "波动率指标"
        }
        return descriptions.get(feature_name, f"{feature_name}指标")

    def _get_feature_parameters(self, feature_name: str) -> Dict[str, Any]:
        """获取特征参数"""
        if feature_name == "sma":
            return {"periods": self.config.feature_params.get("moving_averages", [5, 10, 20, 50])}
        elif feature_name == "ema":
            return {"periods": self.config.feature_params.get("moving_averages", [5, 10, 20, 50])}
        elif feature_name == "rsi":
            return {"period": self.config.feature_params.get("rsi_period", 14)}
        elif feature_name == "macd":
            return {
                "fast_period": self.config.feature_params.get("macd_fast", 12),
                "slow_period": self.config.feature_params.get("macd_slow", 26),
                "signal_period": self.config.feature_params.get("macd_signal", 9)
            }
        elif feature_name == "bollinger_bands":
            return {
                "period": self.config.feature_params.get("bollinger_period", 20),
                "std_dev": self.config.feature_params.get("bollinger_std", 2)
            }
        else:
            return {}

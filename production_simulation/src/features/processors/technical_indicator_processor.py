"""
技术指标处理器
计算各种技术分析指标
"""

from ..indicators.volatility_calculator import VolatilityCalculator
from ..indicators.momentum_calculator import MomentumCalculator
from ..indicators.fibonacci_calculator import FibonacciCalculator
from ..indicators.ichimoku_calculator import IchimokuCalculator
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Iterable
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class IndicatorType(Enum):

    """技术指标类型枚举"""
    TREND = "trend"  # 趋势指标
    MOMENTUM = "momentum"  # 动量指标
    VOLATILITY = "volatility"  # 波动率指标
    VOLUME = "volume"  # 成交量指标
    OSCILLATOR = "oscillator"  # 震荡指标
    STATISTICAL = "statistical"  # 统计指标


@dataclass
class IndicatorConfig:

    """指标配置"""
    name: str
    type: IndicatorType
    parameters: Dict[str, Any]
    description: str = ""
    enabled: bool = True


class TechnicalIndicatorProcessor:

    """技术指标处理器"""

    # 预定义指标配置
    DEFAULT_INDICATORS = {
        'sma': IndicatorConfig(
            name='SMA',
            type=IndicatorType.TREND,
            parameters={'period': 20},
            description='简单移动平均线'
        ),
        'ema': IndicatorConfig(
            name='EMA',
            type=IndicatorType.TREND,
            parameters={'period': 12},
            description='指数移动平均线'
        ),
        'rsi': IndicatorConfig(
            name='RSI',
            type=IndicatorType.OSCILLATOR,
            parameters={'period': 14},
            description='相对强弱指数'
        ),
        'macd': IndicatorConfig(
            name='MACD',
            type=IndicatorType.MOMENTUM,
            parameters={'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            description='MACD指标'
        ),
        'bollinger_bands': IndicatorConfig(
            name='BBANDS',
            type=IndicatorType.VOLATILITY,
            parameters={'period': 20, 'std_dev': 2},
            description='布林带'
        ),
        'stochastic': IndicatorConfig(
            name='STOCH',
            type=IndicatorType.OSCILLATOR,
            parameters={'k_period': 14, 'd_period': 3},
            description='随机指标'
        ),
        'williams_r': IndicatorConfig(
            name='WILLR',
            type=IndicatorType.OSCILLATOR,
            parameters={'period': 14},
            description='威廉指标'
        ),
        'cci': IndicatorConfig(
            name='CCI',
            type=IndicatorType.OSCILLATOR,
            parameters={'period': 20},
            description='顺势指标'
        ),
        'atr': IndicatorConfig(
            name='ATR',
            type=IndicatorType.VOLATILITY,
            parameters={'period': 14},
            description='平均真实波幅'
        ),
        'obv': IndicatorConfig(
            name='OBV',
            type=IndicatorType.VOLUME,
            parameters={},
            description='能量潮指标'
        ),
        'ichimoku': IndicatorConfig(
            name='Ichimoku',
            type=IndicatorType.TREND,
            parameters={
                'tenkan_period': 9,
                'kijun_period': 26,
                'senkou_span_b_period': 52,
                'displacement': 26
            },
            description='一目均衡表'
        ),
        'fibonacci': IndicatorConfig(
            name='Fibonacci',
            type=IndicatorType.STATISTICAL,
            parameters={
                'lookback_period': 50,
                'min_swing_length': 5
            },
            description='斐波那契回撤水平'
        ),
        'momentum': IndicatorConfig(
            name='Momentum',
            type=IndicatorType.MOMENTUM,
            parameters={
                'momentum_period': 10,
                'roc_period': 12,
                'trix_period': 15
            },
            description='动量指标组合'
        ),
        'volatility': IndicatorConfig(
            name='Volatility',
            type=IndicatorType.VOLATILITY,
            parameters={
                'bb_period': 20,
                'kc_period': 20,
                'kc_multiplier': 2
            },
            description='波动率指标组合'
        )
    }

    def _normalize_config_input(
        self,
        config: Optional[Union[Dict[str, Any], List[IndicatorConfig]]]
    ) -> Dict[str, Any]:
        """标准化配置输入，支持list和dict两种形式"""
        if config is None:
            return {}

        if isinstance(config, list):
            custom = {}
            for indicator in config:
                if not isinstance(indicator, IndicatorConfig):
                    continue
                custom[indicator.name] = {
                    'name': indicator.name,
                    'type': indicator.type,
                    'parameters': indicator.parameters,
                    'description': indicator.description,
                    'enabled': indicator.enabled,
                }
            return {
                'custom_indicators': custom,
                'use_defaults': False
            }

        # dict 输入
        normalized = dict(config)
        normalized.setdefault('custom_indicators', {})
        normalized.setdefault('disabled_indicators', [])
        normalized.setdefault('use_defaults', True)
        return normalized

    def _update_enabled_count(self) -> None:
        """更新启用指标数量"""
        self.enabled_indicators_count = sum(
            1 for cfg in self.indicators.values() if cfg.enabled
        )

    def __init__(self, config: Optional[Union[Dict[str, Any], List[IndicatorConfig]]] = None):

        self.config = self._normalize_config_input(config)
        self.indicators: Dict[str, IndicatorConfig] = {}
        self.enabled_indicators_count: int = 0
        self._dependency_map: Dict[str, List[str]] = {
            'macd': ['ema'],
            'bollinger_bands': ['sma'],
            'atr': ['high', 'low', 'close'],
            'ichimoku': ['sma'],
        }
        self._initialize_indicators()
        self._update_enabled_count()

    def _initialize_indicators(self):
        """初始化指标配置"""
        use_defaults = self.config.get('use_defaults', True)
        if use_defaults:
            for indicator_name, indicator_config in self.DEFAULT_INDICATORS.items():
                if indicator_name not in self.config.get('disabled_indicators', []):
                    self.indicators[indicator_name] = indicator_config

        # 应用自定义配置
        custom_indicators = self.config.get('custom_indicators', {})
        for name, config in custom_indicators.items():
            if isinstance(config, IndicatorConfig):
                self.indicators[name] = config
            elif isinstance(config, dict):
                self.indicators[name] = IndicatorConfig(**config)
            else:
                logger.warning(f"无法解析的自定义指标配置: {name}")

        logger.info(f"初始化了 {len(self.indicators)} 个技术指标")
        self._update_enabled_count()

    @staticmethod
    def _validate_input_data(data: pd.DataFrame) -> None:
        """确保输入数据满足最基本的计算要求。"""
        if data is None or data.empty:
            raise ValueError("输入数据为空，无法计算技术指标")

        required_columns = {'close', 'high', 'low'}
        missing = required_columns.difference(data.columns)
        if missing:
            raise ValueError(f"数据缺少必要列: {sorted(missing)}")

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """对外暴露的统一处理接口"""
        return self.calculate_indicators(data)

    def calculate_trend_indicators(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """计算趋势类指标"""
        targets = indicators or self._filter_indicators_by_type(IndicatorType.TREND)
        return self.calculate_indicators(data, targets)

    def calculate_momentum_indicators(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """计算动量/震荡类指标"""
        targets = indicators or self._filter_indicators_by_type(IndicatorType.MOMENTUM)
        if indicators is None:
            targets += self._filter_indicators_by_type(IndicatorType.OSCILLATOR)
        return self.calculate_indicators(data, targets)

    def calculate_volatility_indicators(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """计算波动率指标"""
        targets = indicators or self._filter_indicators_by_type(IndicatorType.VOLATILITY)
        return self.calculate_indicators(data, targets)

    def calculate_volume_indicators(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """计算成交量相关指标"""
        targets = indicators or self._filter_indicators_by_type(IndicatorType.VOLUME)
        return self.calculate_indicators(data, targets)

    def calculate_indicators_batch(
        self,
        data: pd.DataFrame,
        indicator_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """批量计算指定指标"""
        return self.calculate_indicators(data, indicator_list)

    def calculate_all_indicators(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        计算所有启用的指标或指定指标集合。

        Args:
            data: 输入的OHLCV数据
            indicators: 需要计算的指标名称列表
            parallel: 是否启用并行计算
            max_workers: 并行计算的最大工作线程数
        """
        self._validate_input_data(data)

        targets = self._resolve_indicator_list(indicators)
        if not targets:
            return data.copy()

        # 当指标数量较多时自动切换并行模式
        if parallel or len(targets) > 10:
            return self.calculate_indicators_parallel(
                data,
                targets,
                max_workers=max_workers,
            )
        return self.calculate_indicators(data, targets)

    def calculate_indicators_by_type(
        self,
        data: pd.DataFrame,
        indicator_type: IndicatorType,
        indicators: Optional[List[str]] = None,
        parallel: bool = False,
        max_workers: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        根据指标类型计算指标。

        Args:
            data: 输入数据
            indicator_type: 指标类型
            indicators: 可选的精确指标列表（会在类型过滤基础上再次过滤）
        """
        self._validate_input_data(data)
        type_indicators = self._filter_indicators_by_type(indicator_type)
        if indicators:
            type_indicators = [name for name in type_indicators if name in indicators]

        return self.calculate_all_indicators(
            data,
            indicators=type_indicators,
            parallel=parallel,
            max_workers=max_workers,
        )

    def calculate_indicators_parallel(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None,
        max_workers: Optional[int] = None
    ) -> pd.DataFrame:
        """并行计算指标"""
        targets = self._resolve_indicator_list(indicators)
        if not targets:
            return data.copy()

        base_columns = list(data.columns)
        result_df = data.copy()
        worker_count = max_workers or min(len(targets), os.cpu_count() or 1) or 1

        def _calculate(name: str):
            cfg = self.indicators.get(name)
            if not cfg or not cfg.enabled:
                return pd.DataFrame(index=data.index)
            working_df = data.copy()
            updated = self._calculate_single_indicator(working_df, name, cfg)
            new_columns = [col for col in updated.columns if col not in base_columns]
            return updated[new_columns] if new_columns else pd.DataFrame(index=data.index)

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(_calculate, indicator): indicator for indicator in targets}
            for future in as_completed(futures):
                try:
                    new_cols = future.result()
                    for col in new_cols.columns:
                        result_df[col] = new_cols[col]
                except Exception as exc:
                    logger.error(f"并行计算指标 {futures[future]} 失败: {exc}")

        return result_df

    def calculate_indicators(
        self,
        data: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        计算技术指标

        Args:
            data: 输入数据，必须包含 'close', 'high', 'low', 'volume' 列
            indicators: 要计算的指标列表，None表示计算所有启用的指标

        Returns:
            包含原始数据和计算指标的DataFrame
        """
        if data is None or data.empty:
            logger.warning("输入数据为空")
            return pd.DataFrame()

        # 复制数据避免修改原始数据
        result_df = data.copy()

        # 确保必要列存在
        required_columns = ['close', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        if missing_columns:
            logger.error(f"数据缺少必要列: {missing_columns}")
            return result_df

        # 确定要计算的指标
        target_indicators = self._resolve_indicator_list(indicators)

        # 计算每个指标
        for indicator_name in target_indicators:
            indicator_config = self.indicators.get(indicator_name)

            try:
                if indicator_config is None or indicator_config.enabled:
                    result_df = self._calculate_single_indicator(
                        result_df, indicator_name, indicator_config
                    )
                    logger.debug(f"计算指标: {indicator_name}")
            except Exception as e:
                logger.error(f"计算指标 {indicator_name} 失败: {e}")

        return result_df

    def _calculate_single_indicator(self, data: pd.DataFrame,


                                    indicator_name: str,
                                    config: IndicatorConfig) -> pd.DataFrame:
        """计算单个指标"""
        config = self._resolve_indicator_config(indicator_name, config)
        if not config:
            return data

        if indicator_name not in self.indicators:
            self.indicators[indicator_name] = config
            self._update_enabled_count()

        if indicator_name == 'sma' or indicator_name.upper().startswith('SMA_'):
            return self._calculate_sma(data, config.parameters['period'])
        elif indicator_name == 'ema' or indicator_name.upper().startswith('EMA_'):
            return self._calculate_ema(data, config.parameters['period'])
        elif indicator_name == 'rsi' or indicator_name.upper().startswith('RSI_'):
            return self._calculate_rsi(data, config.parameters['period'])
        elif indicator_name == 'macd':
            return self._calculate_macd(data, config.parameters)
        elif indicator_name == 'bollinger_bands' or indicator_name.upper().startswith('BB_'):
            params = config.parameters
            return self._calculate_bollinger_bands(data, params)
        elif indicator_name == 'stochastic':
            return self._calculate_stochastic(data, config.parameters)
        elif indicator_name == 'williams_r':
            return self._calculate_williams_r(data, config.parameters['period'])
        elif indicator_name == 'cci':
            return self._calculate_cci(data, config.parameters['period'])
        elif indicator_name == 'atr' or indicator_name.upper().startswith('ATR_'):
            return self._calculate_atr(data, config.parameters['period'])
        elif indicator_name == 'obv':
            return self._calculate_obv(data)
        elif indicator_name == 'ichimoku':
            return self._calculate_ichimoku(data, config.parameters)
        elif indicator_name == 'fibonacci':
            return self._calculate_fibonacci(data, config.parameters)
        elif indicator_name == 'momentum':
            return self._calculate_momentum(data, config.parameters)
        elif indicator_name == 'volatility':
            return self._calculate_volatility(data, config.parameters)
        else:
            logger.warning(f"未实现的指标: {indicator_name}")
            return data

    def _resolve_indicator_list(self, indicators: Optional[List[str]]) -> List[str]:
        if indicators:
            return indicators
        return [name for name, cfg in self.indicators.items() if cfg.enabled]

    def _filter_indicators_by_type(self, indicator_type: IndicatorType) -> List[str]:
        return [
            name for name, cfg in self.indicators.items()
            if cfg.type == indicator_type and cfg.enabled
        ]

    def _resolve_indicator_config(
        self,
        indicator_name: str,
        config: Optional[IndicatorConfig]
    ) -> Optional[IndicatorConfig]:
        """根据名称动态解析指标配置"""
        if config:
            return config

        stored = self.indicators.get(indicator_name)
        if stored:
            return stored

        upper_name = indicator_name.upper()
        try:
            if upper_name.startswith('SMA_'):
                period = int(upper_name.split('_')[1])
                return IndicatorConfig(indicator_name, IndicatorType.TREND, {'period': period})
            if upper_name.startswith('EMA_'):
                period = int(upper_name.split('_')[1])
                return IndicatorConfig(indicator_name, IndicatorType.TREND, {'period': period})
            if upper_name.startswith('RSI_'):
                period = int(upper_name.split('_')[1])
                return IndicatorConfig(indicator_name, IndicatorType.OSCILLATOR, {'period': period})
            if upper_name.startswith('ATR_'):
                period = int(upper_name.split('_')[1])
                return IndicatorConfig(indicator_name, IndicatorType.VOLATILITY, {'period': period})
            if upper_name.startswith('BB_'):
                period = int(upper_name.split('_')[1])
                return IndicatorConfig(
                    indicator_name,
                    IndicatorType.VOLATILITY,
                    {'period': period, 'std_dev': 2}
                )
        except (IndexError, ValueError):
            logger.warning(f"无法解析动态指标参数: {indicator_name}")
            return None

        # 回退到默认配置
        default = self.DEFAULT_INDICATORS.get(indicator_name.lower())
        if default:
            return default

        logger.warning(f"未知指标: {indicator_name}")
        return None

    def _calculate_sma(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算简单移动平均线"""
        data[f'SMA_{period}'] = data['close'].rolling(window=period).mean()
        return data

    # ------------------------------------------------------------------
    # 公开简化接口（供测试与外部脚本复用）
    # ------------------------------------------------------------------
    def calculate_sma(self, close_series: pd.Series, window: int = 20) -> pd.Series:
        data = pd.DataFrame({'close': close_series})
        result = self._calculate_sma(data, window)
        column = f'SMA_{window}'
        return result[column] if column in result else pd.Series(index=close_series.index, dtype=float)

    def calculate_rsi(self, close_series: pd.Series, window: int = 14) -> pd.Series:
        data = pd.DataFrame({'close': close_series})
        result = self._calculate_rsi(data, window)
        column = f'RSI_{window}'
        series = result[column] if column in result else pd.Series(index=close_series.index, dtype=float)
        return series.fillna(50.0)

    def calculate_macd(
        self,
        close_series: pd.Series,
        fast_period: Optional[int] = None,
        slow_period: Optional[int] = None,
        signal_period: Optional[int] = None,
    ) -> Dict[str, pd.Series]:
        params = {
            'fast_period': fast_period or self.DEFAULT_INDICATORS['macd'].parameters['fast_period'],
            'slow_period': slow_period or self.DEFAULT_INDICATORS['macd'].parameters['slow_period'],
            'signal_period': signal_period or self.DEFAULT_INDICATORS['macd'].parameters['signal_period'],
        }
        data = pd.DataFrame({'close': close_series})
        result = self._calculate_macd(data, params)
        return {
            'macd': result['MACD'],
            'signal': result['MACD_Signal'],
            'histogram': result['MACD_Histogram'],
        }

    def _calculate_ema(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算指数移动平均线"""
        data[f'EMA_{period}'] = data['close'].ewm(span=period).mean()
        return data

    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算RSI指标"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        return data

    def _calculate_macd(self, data: pd.DataFrame, params: Optional[Dict[str, int]] = None) -> pd.DataFrame:
        """计算MACD指标"""
        defaults = {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
        if params:
            defaults.update(params)

        fast_period = defaults['fast_period']
        slow_period = defaults['slow_period']
        signal_period = defaults['signal_period']

        ema_fast = data['close'].ewm(span=fast_period).mean()
        ema_slow = data['close'].ewm(span=slow_period).mean()

        data['MACD'] = ema_fast - ema_slow
        data['MACD_Signal'] = data['MACD'].ewm(span=signal_period).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']

        return data

    def _calculate_bollinger_bands(
        self,
        data: pd.DataFrame,
        params: Union[Dict[str, Any], int],
        std_dev: Optional[int] = None
    ) -> pd.DataFrame:
        """计算布林带"""
        if isinstance(params, dict):
            period = params.get('period', 20)
            std = params.get('std_dev', std_dev if std_dev is not None else 2)
        else:
            period = params
            std = std_dev if std_dev is not None else 2

        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()

        data[f'BB_Upper_{period}'] = sma + (std * std)
        data[f'BB_Middle_{period}'] = sma
        data[f'BB_Lower_{period}'] = sma - (std * std)

        return data

    def _calculate_stochastic(self, data: pd.DataFrame, params: Dict[str, int]) -> pd.DataFrame:
        """计算随机指标"""
        k_period = params['k_period']
        d_period = params['d_period']

        # 计算 % K
        lowest_low = data['low'].rolling(window=k_period).min()
        highest_high = data['high'].rolling(window=k_period).max()
        data[f'STOCH_K_{k_period}'] = 100 * \
            ((data['close'] - lowest_low) / (highest_high - lowest_low))

        # 计算 % D
        data[f'STOCH_D_{k_period}_{d_period}'] = data[f'STOCH_K_{k_period}'].rolling(
            window=d_period).mean()

        return data

    def _calculate_williams_r(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算威廉指标"""
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()

        data[f'WILLR_{period}'] = -100 * \
            ((highest_high - data['close']) / (highest_high - lowest_low))

        return data

    def _calculate_cci(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算顺势指标"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = abs(typical_price - sma).rolling(window=period).mean()

        data[f'CCI_{period}'] = (typical_price - sma) / (0.015 * mean_deviation)

        return data

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算平均真实波幅"""
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data[f'ATR_{period}'] = true_range.rolling(window=period).mean()

        return data

    def _calculate_obv(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算能量潮指标"""
        if 'volume' not in data.columns:
            logger.warning("数据缺少volume列，跳过OBV计算")
            return data

        obv = [0]  # 第一个OBV值为0

        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i - 1]:
                obv.append(obv[-1] + data['volume'].iloc[i])
            elif data['close'].iloc[i] < data['close'].iloc[i - 1]:
                obv.append(obv[-1] - data['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        data['OBV'] = obv
        return data

    def get_available_indicators(self) -> List[str]:
        """获取可用的指标列表"""
        return list(self.indicators.keys())

    def get_indicator_info(self, indicator_name: str) -> Optional[IndicatorConfig]:
        """获取指标信息"""
        return self.indicators.get(indicator_name)

    def enable_indicator(self, indicator_name: str):
        """启用指标"""
        if indicator_name in self.indicators:
            self.indicators[indicator_name].enabled = True
            logger.info(f"启用指标: {indicator_name}")
            self._update_enabled_count()

    def disable_indicator(self, indicator_name: str):
        """禁用指标"""
        if indicator_name in self.indicators:
            self.indicators[indicator_name].enabled = False
            logger.info(f"禁用指标: {indicator_name}")
            self._update_enabled_count()

    def add_custom_indicator(self, name: str, config: Dict[str, Any]):
        """添加自定义指标"""
        self.indicators[name] = IndicatorConfig(**config)
        logger.info(f"添加自定义指标: {name}")
        self._update_enabled_count()

    def get_indicator_types(self) -> Dict[IndicatorType, List[str]]:
        """按类型分组获取指标"""
        result = {}
        for name, config in self.indicators.items():
            if config.enabled:
                if config.type not in result:
                    result[config.type] = []
                result[config.type].append(name)
        return result

    def get_indicator_config(self, indicator_name: str) -> Optional[IndicatorConfig]:
        """获取单个指标配置"""
        return self.indicators.get(indicator_name)

    def update_indicator_config(self, indicator_name: str, new_config: IndicatorConfig) -> bool:
        """更新指标配置"""
        if indicator_name not in self.indicators:
            return False
        self.indicators[indicator_name] = new_config
        self._update_enabled_count()
        return True

    def is_indicator_enabled(self, indicator_name: str) -> bool:
        """判断指标是否启用"""
        config = self.indicators.get(indicator_name)
        return bool(config and config.enabled)

    def list_available_indicators(self) -> List[Dict[str, Any]]:
        """列出所有可用指标的信息"""
        indicators = []
        for name, config in self.indicators.items():
            indicators.append({
                'name': name,
                'type': config.type.value,
                'description': config.description,
                'parameters': config.parameters,
                'enabled': config.enabled,
            })
        return indicators

    def _check_indicator_dependencies(self, indicator_name: str) -> List[str]:
        """返回指标依赖"""
        return self._dependency_map.get(indicator_name.lower(), [])

    def _calculate_ichimoku(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """计算一目均衡表"""
        calculator = IchimokuCalculator(params)
        return calculator.calculate(data)

    def _calculate_fibonacci(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """计算斐波那契水平"""
        calculator = FibonacciCalculator(params)
        return calculator.calculate(data)

    def _calculate_momentum(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """计算动量指标"""
        calculator = MomentumCalculator(params)
        return calculator.calculate(data)

    def _calculate_volatility(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """计算波动率指标"""
        calculator = VolatilityCalculator(params)
        return calculator.calculate(data)

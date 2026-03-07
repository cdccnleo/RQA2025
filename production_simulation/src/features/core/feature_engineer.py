from __future__ import annotations
import pandas as pd
from typing import List, Dict, Optional, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

from ..processors.feature_metadata import FeatureMetadata
from .config_integration import get_config_integration_manager, ConfigScope
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:

    """特征工程处理器，用于生成和管理特征"""

    def __init__(self, technical_processor=None, sentiment_analyzer=None, cache_dir: str = "./feature_cache", max_retries: int = 3, fallback_enabled: bool = True):

        # 初始化配置集成管理器
        self.config_manager = get_config_integration_manager()

        # 从配置管理器获取配置
        self.cache_dir = Path(cache_dir or self.config_manager.get_config(
            ConfigScope.GLOBAL, "cache_dir"))
        self.max_retries = max_retries or self.config_manager.get_config(
            ConfigScope.PROCESSING, "max_retries") or 3
        self.fallback_enabled = fallback_enabled or self.config_manager.get_config(
            ConfigScope.GLOBAL, "fallback_enabled") or True

        # 获取处理配置
        processing_config = self.config_manager.get_config(ConfigScope.PROCESSING)
        self.max_workers = processing_config.get("max_workers", 4) if processing_config else 4
        self.batch_size = processing_config.get("batch_size", 1000) if processing_config else 1000
        self.timeout = processing_config.get("timeout", 300) if processing_config else 300

        # 获取监控配置
        monitoring_config = self.config_manager.get_config(ConfigScope.MONITORING)
        self.enable_monitoring = monitoring_config.get(
            "enable_monitoring", True) if monitoring_config else True
        self.monitoring_level = monitoring_config.get(
            "monitoring_level", "standard") if monitoring_config else "standard"

        # 初始化组件
        self.technical_processor = technical_processor
        self.sentiment_analyzer = sentiment_analyzer
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.cache_metadata: Dict[str, Dict] = {}
        self._load_cache_metadata()
        self.logger = logger

        # 初始化特征元数据
        self.feature_metadata = FeatureMetadata()

        # 注册配置变更监听器
        self.config_manager.register_config_watcher(ConfigScope.PROCESSING, self._on_config_change)
        self.config_manager.register_config_watcher(ConfigScope.MONITORING, self._on_config_change)

    def _on_config_change(self, scope: ConfigScope, key: str, old_value: Any, new_value: Any):
        """配置变更处理"""
        self.logger.info(f"配置变更: {scope.value}.{key} = {old_value} -> {new_value}")

        if scope == ConfigScope.PROCESSING:
            if key == "max_workers":
                # 更新实例属性
                self.max_workers = new_value
                # 重新创建线程池
                self.executor.shutdown(wait=True)
                self.executor = ThreadPoolExecutor(max_workers=new_value)
            elif key == "batch_size":
                self.batch_size = new_value
            elif key == "timeout":
                self.timeout = new_value
        elif scope == ConfigScope.MONITORING:
            if key == "enable_monitoring":
                self.enable_monitoring = new_value
            elif key == "monitoring_level":
                self.monitoring_level = new_value

    def _load_cache_metadata(self):
        """加载缓存元数据"""
        metadata_file = self.cache_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf - 8') as f:
                    self.cache_metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.cache_metadata = {}
        else:
            self.cache_metadata = {}

    def register_feature(self, config):
        """注册特征配置"""
        # 简单的特征注册实现，将配置存储到缓存元数据中
        if hasattr(config, 'name') and hasattr(config, 'feature_type'):
            self.cache_metadata[config.name] = {
                'feature_type': config.feature_type.value,
                'params': getattr(config, 'params', {}),
                'dependencies': getattr(config, 'dependencies', []),
                'enabled': getattr(config, 'enabled', True),
                'version': getattr(config, 'version', '1.0')
            }
            # 保存到文件
            metadata_file = self.cache_dir / "metadata.json"
            try:
                with open(metadata_file, 'w', encoding='utf - 8') as f:
                    json.dump(self.cache_metadata, f, indent=2, ensure_ascii=False)
            except IOError:
                pass  # 忽略文件写入错误

    def _validate_stock_data(self, data: pd.DataFrame) -> None:
        """
        验证股票数据的有效性

        Args:
            data: 股票数据DataFrame

        Raises:
            ValueError: 数据验证失败时抛出
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        fallback_enabled = getattr(self, "fallback_enabled", True)

        # 检查必需列
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"缺少必需列: {missing_columns}")

        # 检查数据是否为空
        if data.empty:
            raise ValueError("数据为空")

        # 检查数据有效性 - 增加容错机制
        try:
            # 检查负值价格 - 允许配置容错
            if hasattr(self, 'config') and getattr(self.config, 'allow_negative_prices', False):
                pass  # 跳过负值检查
            else:
                if (data[['close', 'high', 'low']] < 0).any().any():
                    if not fallback_enabled:
                        raise ValueError("检测到负值价格")
                    for col in ['close', 'high', 'low']:
                        if col in data.columns:
                            data[col] = data[col].abs()
                    self.logger.warning("检测到负值价格，已自动修复为绝对值")

            # 检查负值交易量 - 允许配置容错
            if hasattr(self, 'config') and getattr(self.config, 'allow_negative_volume', False):
                pass  # 跳过负值检查
            else:
                if (data['volume'] < 0).any():
                    if not fallback_enabled:
                        raise ValueError("检测到负值交易量")
                    data['volume'] = data['volume'].abs()
                    self.logger.warning("检测到负值交易量，已自动修复为绝对值")

            # 检查价格逻辑 - 增加容错机制
            if hasattr(self, 'config') and getattr(self.config, 'strict_price_logic', True):
                # 严格模式：检查价格逻辑
                if (data['high'] < data['low']).any():
                    if not fallback_enabled:
                        raise ValueError("检测到价格高低值逻辑错误")
                    temp_high = data['high'].copy()
                    temp_low = data['low'].copy()
                    mask = data['high'] < data['low']
                    data.loc[mask, 'high'] = temp_low[mask]
                    data.loc[mask, 'low'] = temp_high[mask]
                    self.logger.warning("检测到价格高低值逻辑错误，已自动修复")

                if ((data['close'] > data['high']) | (data['close'] < data['low'])).any():
                    if not fallback_enabled:
                        raise ValueError("检测到收盘价超出高低价范围")
                    mask = data['close'] > data['high']
                    data.loc[mask, 'close'] = data.loc[mask, 'high']
                    mask = data['close'] < data['low']
                    data.loc[mask, 'close'] = data.loc[mask, 'low']
                    self.logger.warning("检测到收盘价超出高低价范围，已自动修复")

            # 检查NaN值 - 增加容错机制
            if hasattr(self, 'config') and getattr(self.config, 'allow_nan_values', False):
                pass  # 跳过NaN检查
            else:
                nan_columns = data[required_columns].columns[data[required_columns].isna().any()
                                                             ].tolist()
                if nan_columns:
                    if not fallback_enabled:
                        raise ValueError(f"检测到NaN值: {nan_columns}")
                    for col in nan_columns:
                        if col in data.columns:
                            data[col] = data[col].ffill().bfill()
                    self.logger.warning(f"检测到NaN值，已自动填充: {nan_columns}")

            # 检查索引 - 增加容错机制
            if not isinstance(data.index, pd.DatetimeIndex):
                try:
                    data.index = pd.to_datetime(data.index)
                    self.logger.warning("索引已转换为时间戳类型")
                except BaseException:
                    if not fallback_enabled:
                        raise ValueError("无法将索引转换为时间戳类型")
                    raise

            # 检查重复日期 - 增加容错机制
            if not data.index.is_unique:
                if not fallback_enabled:
                    raise ValueError("检测到重复日期")
                data = data[~data.index.duplicated(keep='last')]
                self.logger.warning("检测到重复日期，已保留最后一个值")

            # 检查未来日期 - 增加容错机制
            if hasattr(self, 'config') and getattr(self.config, 'allow_future_dates', False):
                pass  # 跳过未来日期检查
            else:
                current_time = pd.Timestamp.now()
                future_dates = data.index[data.index > current_time]
                if not future_dates.empty:
                    if not fallback_enabled:
                        raise ValueError("检测到未来日期数据")
                    data = data[data.index <= current_time]
                    self.logger.warning(f"检测到未来日期数据，已移除 {len(future_dates)} 条记录")

            # 检查索引排序 - 增加容错机制
            if not data.index.is_monotonic_increasing:
                if not fallback_enabled:
                    raise ValueError("索引未按时间排序")
                data = data.sort_index()
                self.logger.warning("索引已自动排序")

        except Exception as e:
            # 如果修复失败，记录错误但不抛出异常
            self.logger.error(f"数据验证和修复过程中出现错误: {str(e)}")
            fallback_enabled = getattr(self, "fallback_enabled", True)
            strict_mode = hasattr(self, 'config') and getattr(self.config, 'strict_validation', False)
            if strict_mode and not fallback_enabled:
                raise ValueError(f"数据验证失败: {str(e)}")
            if fallback_enabled:
                self.logger.warning("数据验证失败，但继续处理")
            else:
                raise ValueError(f"数据验证失败: {str(e)}")

    def generate_technical_features(


            self,
            stock_data: pd.DataFrame,
            indicators: List[str] = None,
            params: Dict = None
    ) -> pd.DataFrame:
        """
        生成技术指标特征

        Args:
            stock_data: 股票数据DataFrame
            indicators: 要计算的技术指标列表
            params: 技术指标参数

        Returns:
            技术指标特征DataFrame
        """
        # 验证数据
        self._validate_stock_data(stock_data)

        # 默认指标和参数
        if indicators is None:
            indicators = ["ma", "rsi", "macd", "bollinger"]
        if params is None:
            params = {
                "ma": {"windows": [5, 10, 20, 30, 60]},
                "rsi": {"window": 14},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger": {"window": 20, "num_std": 2}
            }

        # 更新元数据中的特征参数
        self.feature_metadata.update_feature_params({
            "technical_indicators": indicators,
            "technical_params": params
        })

        # 生成技术指标
        try:
            if self.technical_processor is None:
                raise ValueError("技术处理器未初始化")

            # 使用calculate_multiple_indicators方法
            features = self.technical_processor.calculate_multiple_indicators(
                data=stock_data,
                indicators=indicators or ['sma', 'rsi', 'macd'],
                params=params or {}
            )
            self.logger.info(f"技术指标特征生成完成，特征列名: {features.columns.tolist()}")
            return features

        except Exception as e:
            self.logger.error(f"生成技术指标特征失败: {str(e)}")
            raise

    def generate_sentiment_features(


            self,
            news_data: pd.DataFrame,
            text_col: str = "content",
            date_col: str = "date",
            output_cols: List[str] = None
    ) -> pd.DataFrame:
        """
        生成情感分析特征

        Args:
            news_data: 新闻数据DataFrame
            text_col: 文本列名
            date_col: 日期列名
            output_cols: 输出列名列表

        Returns:
            情感分析特征DataFrame
        """
        try:
            # 更新元数据中的特征参数
            self.feature_metadata.update_feature_params({
                "sentiment_text_col": text_col,
                "sentiment_date_col": date_col,
                "sentiment_output_cols": output_cols
            })

            features = self.sentiment_analyzer.generate_features(
                news_data=news_data,
                text_col=text_col,
                date_col=date_col,
                output_cols=output_cols
            )
            self.logger.info(f"情感分析特征生成完成，特征列名: {features.columns.tolist()}")
            return features

        except Exception as e:
            self.logger.error(f"生成情感分析特征失败: {str(e)}")
            raise

    def merge_features(


            self,
            stock_data: pd.DataFrame,
            technical_features: pd.DataFrame,
            sentiment_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        合并所有特征

        Args:
            stock_data: 原始股票数据
            technical_features: 技术指标特征
            sentiment_features: 情感分析特征（可选）

        Returns:
            合并后的特征DataFrame
        """
        try:
            # 验证索引
            if not technical_features.index.equals(stock_data.index):
                raise ValueError("技术指标特征索引不匹配")
            if sentiment_features is not None and not sentiment_features.index.equals(stock_data.index):
                raise ValueError("情感分析特征索引不匹配")

            # 合并特征
            result = pd.concat([
                stock_data,
                technical_features,
                sentiment_features if sentiment_features is not None else pd.DataFrame(
                    index=stock_data.index)
            ], axis=1)

            # 更新元数据
            self.feature_metadata.update_feature_columns(result.columns.tolist())

            self.logger.info(f"特征合并完成，最终特征数: {len(result.columns)}")
            return result

        except Exception as e:
            self.logger.error(f"特征合并失败: {str(e)}")
            raise

    def save_metadata(self, path: str) -> None:
        """
        保存特征元数据到文件

        Args:
            path: 元数据文件路径
        """
        try:
            self.feature_metadata.save_metadata(path)
            self.logger.info(f"特征元数据保存成功: {path}")
        except Exception as e:
            self.logger.error(f"保存特征元数据失败: {str(e)}")
            raise

    def load_metadata(self, path: str) -> None:
        """
        从文件加载特征元数据

        Args:
            path: 元数据文件路径
        """
        try:
            self.feature_metadata = FeatureMetadata(metadata_path=path)
            self.logger.info(f"特征元数据加载成功: {path}")
        except Exception as e:
            self.logger.error(f"加载特征元数据失败: {str(e)}")
            raise

# 保留A股特有特征混合类


class ASharesFeatureMixin:

    """A股特有特征混合类"""

    @staticmethod
    def calculate_limit_status(symbol: str, realtime_engine) -> int:
        """计算涨跌停状态特征"""
        status = realtime_engine.get_limit_status(symbol)
        return 1 if status == 'up' else -1 if status == 'down' else 0

    @staticmethod
    def calculate_margin_ratio(margin_data: pd.DataFrame) -> float:
        """计算融资融券余额比"""
        return (margin_data['margin_balance'] / margin_data['total_market_cap']).iloc[-1]

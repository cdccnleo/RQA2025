# src/data/loader/news_loader.py
from __future__ import annotations

import configparser
import re
import akshare as ak
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from src.infrastructure.utils.datetime_parser import DateTimeParser
from src.infrastructure.utils.exceptions import DataLoaderError
import os
import time
from numpy import dtype
from requests import RequestException
from bs4 import BeautifulSoup
from typing import Optional, Tuple, Any, Union
from src.features.processors.sentiment import SentimentAnalyzer
from src.infrastructure.utils.logger import get_logger
from src.data.base_loader import BaseDataLoader

logger = get_logger(__name__)


class FinancialNewsLoader(BaseDataLoader):
    """多源财经新闻数据加载器，支持缓存和自动清洗

    属性：
        source (str): 数据源标识 (cls/sina/em)
        save_path (Path): 数据存储路径
        cache_days (int): 本地缓存有效期天数
    """

    SOURCE_MAP = {
        "cls": ak.stock_info_global_cls,
        "sina": ak.stock_info_global_sina,
        "em": ak.stock_info_global_em
    }

    def __init__(self, source: str = "cls", save_path: Union[str, Path] = "data/news",
                 max_retries=3,
                 cache_days: int = 7):
        """
        初始化新闻加载器

        Args:
            source: 数据源标识 (cls/sina/em)
            save_path: 数据存储路径，默认data/news
            cache_days: 本地缓存有效期天数，默认7天
        """
        if source not in self.SOURCE_MAP:
            raise ValueError(f"Unsupported news source: {source}")

        self.source = source
        self.save_path = Path(save_path)
        self.max_retries = max_retries
        self.cache_days = cache_days
        self._setup()

    def _setup(self) -> None:
        """初始化工作目录和日志配置"""
        self.save_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_from_config(cls, config: Union[configparser.ConfigParser, dict], thread_pool=None):
        """从配置文件创建实例"""
        # 从 config.ini 获取默认值
        default_config = configparser.ConfigParser()
        default_config.read('config.ini', encoding='utf-8')

        if isinstance(config, configparser.ConfigParser):
            loader_config = config['News'] if 'Financial' in config else {}
        elif isinstance(config, dict):
            loader_config = config.get('News', {})
        elif hasattr(config, 'items'):  # 处理 Section 对象
            loader_config = dict(config)
        else:
            raise ValueError("不支持的配置类型")

        default_loader_config = default_config['News'] if 'News' in default_config else {}

        # 安全获取配置值
        save_path = loader_config.get('save_path', default_loader_config.get('save_path', 'data/news'))  # 修改默认路径

        # 处理整数值
        def safe_getint(source, key, default):
            value = source.get(key, None)
            if value is None:
                return default

            try:
                return int(value)
            except ValueError as e:
                raise DataLoaderError(f"配置项 {key} 的值无效（必须为整数）: {value}") from e

        max_retries = safe_getint(loader_config, 'max_retries',
                                  default_loader_config.getint('max_retries', 3) if hasattr(default_loader_config,
                                                                                           'getint') else 3)
        cache_days = safe_getint(loader_config, 'cache_days',
                                 default_loader_config.getint('cache_days', 30) if hasattr(default_loader_config,
                                                                                          'getint') else 30)

        return cls(
            source=loader_config.get('source', default_loader_config.get('source', 'cls')),
            save_path=save_path,
            max_retries=max_retries,
            cache_days=cache_days,
            thread_pool=thread_pool
        )

    def load_data(self, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        """
        加载指定时间范围的新闻数据
        """
        # 日期范围验证
        if pd.Timestamp(start_date) > pd.Timestamp(end_date):
            raise ValueError("开始日期不能大于结束日期")

        # 清理日期字符串中的特殊字符
        def sanitize_date_string(date_str):
            return re.sub(r'[:\s]', '_', str(date_str))

        sanitized_start = sanitize_date_string(start_date)
        sanitized_end = sanitize_date_string(end_date)

        file_path = self.save_path / f"news_{sanitized_start}_{sanitized_end}.csv"
        for attempt in range(self.max_retries + 1):
            try:
                # 尝试加载缓存数据
                if self._is_cache_valid(file_path):
                    logger.info(f"使用缓存数据: {file_path}")
                    return pd.read_csv(file_path, parse_dates=["publish_time"], encoding='utf-8')

                # 获取原始新闻数据
                raw_df = self._fetch_raw_data(start_date, end_date)
                if raw_df.empty:
                    logger.warning("原始数据为空，返回空 DataFrame")
                    return pd.DataFrame()  # 返回空的 DataFrame

                logger.debug(f"原始数据列名: {raw_df.columns.tolist()}")
                logger.debug(f"原始数据前几行:\n{raw_df.head()}")

                processed_df = self._process_raw_data(raw_df)
                if processed_df.empty:
                    logger.warning("处理后的数据为空，返回空 DataFrame")
                    return pd.DataFrame()  # 如果处理后的数据为空，直接返回

                logger.debug(f"处理后的数据列名: {processed_df.columns.tolist()}")
                logger.debug(f"处理后的数据前几行:\n{processed_df.head()}")

                # 按日期聚合，确保与价格数据时间粒度一致
                required_aggregation_columns = {'content', 'title'}
                if not required_aggregation_columns.issubset(processed_df.columns):
                    missing = required_aggregation_columns - set(processed_df.columns)
                    logger.error(f"聚合所需字段缺失: {missing}")
                    raise DataLoaderError(f"聚合所需字段缺失: {missing}")

                start_dt = pd.Timestamp(start_date).tz_localize(None)
                end_dt = pd.Timestamp(end_date).tz_localize(None)

                # 新增：确保publish_time列无时区信息
                if processed_df['publish_time'].dt.tz is not None:
                    processed_df['publish_time'] = processed_df['publish_time'].dt.tz_convert(None)

                # 新增时间戳对齐逻辑
                mask = (processed_df['publish_time'] >= start_dt) & \
                       (processed_df['publish_time'] <= end_dt)
                processed_df = processed_df[mask]

                daily_news = processed_df.groupby(processed_df['publish_time'].dt.date).agg({
                    'content': ' '.join,
                    'title': ' '.join
                }).reset_index()
                daily_news['publish_time'] = pd.to_datetime(daily_news['publish_time'])

                # 保存并返回数据
                daily_news.to_csv(file_path, index=False, encoding='utf-8')
                logger.info(f"数据已保存至: {file_path}")
                return daily_news
            except DataLoaderError as e:
                logger.error(f"加载新闻数据失败: {str(e)}")
                raise e
            except Exception as e:
                if attempt >= self.max_retries:
                    logger.error(f"加载新闻数据失败: {str(e)}")
                    return pd.DataFrame()  # 返回空的 DataFrame
                logger.warning(f"加载数据失败，第{attempt + 1}次重试...")
                time.sleep(2 ** attempt)
        return pd.DataFrame()  # 返回空的 DataFrame

    def _is_cache_valid(self, file_path: Path) -> bool:
        """检查缓存是否有效"""
        if not file_path.exists():
            logger.debug(f"缓存文件不存在: {file_path}")
            return False

        try:
            # 确保解析为本地时间
            cached_df = pd.read_csv(
                file_path,
                parse_dates=["publish_time"],
                date_parser=lambda x: pd.to_datetime(x, utc=False).tz_localize(None)
            )
            if cached_df.empty:
                logger.debug("缓存文件为空")
                return False

            # 检查必要列是否存在
            required_columns = {'publish_time', 'content', 'title'}
            if not required_columns.issubset(cached_df.columns):
                logger.debug("缓存文件缺少必要列")
                return False

            # 使用本地时间验证缓存有效期
            now = pd.Timestamp.now(tz=None)
            if now - cached_df["publish_time"].max() > pd.Timedelta(days=self.cache_days):
                return False

            return True
        except Exception as e:
            logger.warning(f"缓存验证失败: {str(e)}")
            return False

    def _fetch_raw_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取原始新闻数据"""
        logger.info(f"Fetching {self.source} news from {start_date} to {end_date}")
        try:
            # 根据不同数据源调整参数
            if self.source == "cls":
                raw_data = self._retry_api_call(
                    ak.stock_info_global_cls,
                    symbol="全部"  # 示例参数，实际使用时根据需要调整
                )
            elif self.source == "sina":
                raw_data = self._retry_api_call(
                    ak.stock_info_global_sina
                )
            elif self.source == "em":
                raw_data = self._retry_api_call(
                    ak.stock_info_global_em
                )
            else:
                raise DataLoaderError(f"Unsupported news source: {self.source}")

            # 增加空数据检查
            if raw_data is None or raw_data.empty:
                logger.warning(f"数据源 {self.source} 返回空数据")
                return pd.DataFrame(columns=["标题", "内容", "发布日期", "发布时间"])

            # 根据不同数据源调整和整合数据
            if self.source == "cls":
                # 财联社：直接映射已有字段
                raw_data.rename(columns={
                    "标题": "标题",
                    "内容": "内容",
                    "发布日期": "发布日期",
                    "发布时间": "发布时间"
                }, inplace=True)
            elif self.source == "sina":
                # 新浪财经：生成标题并标准化时间
                if "content" in raw_data.columns:
                    raw_data["标题"] = raw_data["content"].str.slice(0, 20) + "..."  # 截取内容前20字为标题
                if "time" in raw_data.columns:
                    # 确保 time 列是字符串类型
                    raw_data["time"] = raw_data["time"].astype(str)
                    raw_data[["发布日期", "发布时间"]] = raw_data["time"].str.split(" ", expand=True)
                    raw_data.drop(columns=["time"], inplace=True)  # 删除原始 time 列
                raw_data.rename(columns={
                    "content": "内容"
                }, inplace=True)
            elif self.source == "em":
                # 东方财富：拆分发布时间为日期和时间
                if "em_content" in raw_data.columns:
                    raw_data["内容"] = raw_data["em_content"]
                if "em_time" in raw_data.columns:
                    raw_data["em_time"] = raw_data["em_time"].astype(str)
                    raw_data[["发布日期", "发布时间"]] = raw_data["em_time"].str.split(" ", expand=True)
                    raw_data.drop(columns=["em_time"], inplace=True)  # 删除原始 em_time 列
                if "em_title" in raw_data.columns:
                    raw_data["标题"] = raw_data["em_title"]
                else:
                    raw_data["标题"] = "默认标题"

            # 统一必要字段检查
            required_columns = {"标题", "内容", "发布日期", "发布时间"}
            if not required_columns.issubset(raw_data.columns):
                missing = required_columns - set(raw_data.columns)
                logger.error(f"数据源 {self.source} 缺少字段: {missing}")
                raise DataLoaderError(f"数据源 {self.source} 缺少字段: {missing}")

            logger.debug(f"获取到的原始数据行数: {len(raw_data)}")
            logger.debug(f"原始数据列名: {raw_data.columns.tolist()}")
            logger.debug(f"原始数据前几行:\n{raw_data.head()}")
            return raw_data
        except DataLoaderError as e:
            logger.error(f"加载新闻数据失败: {str(e)}")
            return pd.DataFrame()

    def _process_raw_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """执行数据清洗流水线"""
        if raw_df.empty:
            logger.warning("原始数据为空，返回空 DataFrame")
            return pd.DataFrame()

        # 列名映射（严格匹配中文列名）
        column_mapping = {
            '标题': 'title',
            '内容': 'content',
            '发布日期': 'publish_date',
            '发布时间': 'publish_time'
        }
        raw_df = raw_df.rename(columns=column_mapping)
        logger.debug(f"列名映射后列名: {raw_df.columns.tolist()}")

        # 填充缺失字段
        if "source" not in raw_df.columns:
            raw_df["source"] = self.source  # 添加数据源标识

        # 检查必要列是否存在
        required_columns = {'publish_date', 'publish_time', 'content', 'title'}
        if not required_columns.issubset(raw_df.columns):
            missing = required_columns - set(raw_df.columns)
            logger.error(f"缺失必要列: {missing}")
            raise DataLoaderError(f"缺失必要列: {missing}")

        # 解析日期时间
        try:
            logger.debug(f"原始数据样例:\n{raw_df.head().to_dict()}")
            after_parse = self._parse_datetime(raw_df)
            logger.debug(f"解析后的数据样例:\n{after_parse.head().to_dict() if not after_parse.empty else '无数据'}")
            if after_parse.empty:
                logger.warning("日期解析后数据为空，请检查原始数据格式")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"日期解析失败: {str(e)}")
            return pd.DataFrame()

        # 清洗HTML内容
        try:
            after_clean = self._clean_html_content(after_parse)
            logger.debug(f"HTML清洗后行数: {len(after_clean)}")
            if after_clean.empty:
                logger.warning("HTML清洗后数据为空")
                return pd.DataFrame()
        except KeyError as e:
            logger.error(f"HTML清洗失败，缺少必要列: {str(e)}")
            return pd.DataFrame()

        # 去重
        after_remove_duplicates = self._remove_duplicates(after_clean)
        logger.debug(f"去重后行数: {len(after_remove_duplicates)}")

        return after_remove_duplicates

    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """解析和统一日期时间格式（增强多格式支持）"""
        if 'publish_date' not in df.columns or 'publish_time' not in df.columns:
            logger.error("Missing datetime columns for parsing")
            raise DataLoaderError("Missing datetime columns for parsing")

        # 尝试使用 DateTimeParser 解析日期时间
        try:
            parsed_df = DateTimeParser.parse_datetime(df, "publish_date", "publish_time")

            # 记录解析失败的行
            invalid_mask = parsed_df["publish_time"].isna()
            if invalid_mask.any():
                logger.warning(f"无法解析日期时间的行数: {invalid_mask.sum()}")
                logger.debug(f"示例数据: {parsed_df[invalid_mask][['publish_date', 'publish_time']].iloc[0]}")

            # 仅保留有效数据
            valid_df = parsed_df[~invalid_mask].copy()
            valid_df.drop(columns=["publish_date"], errors='ignore', inplace=True)
            return valid_df
        except Exception as e:
            logger.error(f"DateTimeParser 解析失败，使用备选方案: {str(e)}")
            # 备选方案：手动标准化日期和时间
            df = df.copy()
            df['publish_date'] = df['publish_date'].apply(DateTimeParser._normalize_date_format)
            df['publish_time'] = df['publish_time'].apply(DateTimeParser._normalize_time_format)

            # 合并日期和时间
            datetime_str = df['publish_date'] + ' ' + df['publish_time']

            # 解析为 datetime 对象
            df['publish_time'] = pd.to_datetime(
                datetime_str,
                errors='coerce',
                format='mixed',
                utc=False
            )

            df['publish_time'] = df['publish_time'].apply(
                lambda dt:
                dt.tz_convert('Asia/Shanghai').tz_localize(None)
                if dt.tzinfo is not None
                else dt.replace(tzinfo=None)
            )

            valid_df = df[df['publish_time'].notna()].copy()
            valid_df.drop(columns=["publish_date"], errors='ignore', inplace=True)
            return valid_df

    def _clean_html_content(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗HTML内容"""
        if 'content' not in df.columns:
            raise KeyError("DataFrame missing 'content' column")
        df["content"] = df["content"].apply(self._clean_single_html)
        return df

    def _clean_single_html(self, html: str) -> str:
        """清洗单个HTML文档（增加广告过滤）"""
        if not isinstance(html, str) or not html.strip():
            return ""

        try:
            soup = BeautifulSoup(html, "html.parser")

            # 移除广告元素
            for ad in soup.find_all(class_="ad"):
                ad.decompose()

            # 移除脚本和样式
            for script in soup(["script", "style", "noscript", "meta", "link"]):
                script.decompose()

            text = soup.get_text()
            return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            logger.error(f"HTML 清洗失败: {str(e)}")
            return ""  # 任何异常情况下都返回空字符串

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于标题和内容去重"""
        return df.drop_duplicates(subset=["title", "content"], keep="last")

    def _retry_api_call(self, func: callable, *args, **kwargs) -> Any:
        """带重试机制的API调用"""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (RequestException, ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise
                logger.warning(f"请求失败，第{attempt + 1}次重试...")
                time.sleep(2 ** attempt)
        return None

    def _handle_exception(self, e: Exception, stage: str):
        error_msg = f"{stage}阶段出错: {str(e)}"
        logger.error(error_msg)
        raise DataLoaderError(str(e)) from e

    def _save_data(self, df: pd.DataFrame, file_path: Path) -> bool:
        try:
            df.to_csv(file_path, index=False, encoding='utf-8')
            return True
        except Exception as e:
            self._handle_exception(e, "Data saving")

    def get_metadata(self) -> Dict[str, Any]:
        """获取金融新闻加载器的元数据
        
        返回:
            包含加载器元数据的字典
        """
        return {
            "loader_type": "FinancialNewsLoader",
            "data_source": self.source,
            "max_retries": self.max_retries,
            "cache_days": self.cache_days,
            "supports_sentiment": False
        }

    def load(self, *args, **kwargs) -> Any:
        """实现BaseDataLoader的抽象方法，包装load_data"""
        return self.load_data(*args, **kwargs)

    def validate(self, data: Any) -> bool:
        """验证加载的新闻数据是否符合预期
        
        参数:
            data: 要验证的数据
            
        返回:
            bool: 数据是否有效
        """
        if not isinstance(data, pd.DataFrame):
            return False
        required_columns = {'publish_time', 'content', 'title'}
        return not data.empty and required_columns.issubset(data.columns)


class SentimentNewsLoader(FinancialNewsLoader, BaseDataLoader):
    """情感分析的新闻加载器"""

    def __init__(self, source: str = "cls",
                 save_path: Union[str, Path] = "data/news",
                 max_retries: int = 3,
                 cache_days: int = 7,
                 debug_mode: bool = False,
                 thread_pool=None):  # 添加 thread_pool 参数
        super().__init__(source, save_path, max_retries, cache_days)
        self.save_path = Path(save_path)
        self.max_retries = max_retries
        self.cache_days = cache_days
        self.debug_mode = debug_mode
        self.thread_pool = thread_pool  # 存储线程池引用

    @classmethod
    def create_from_config(cls, config: Union[configparser.ConfigParser, dict], thread_pool=None):
        """从配置文件创建实例"""
        # 从 config.ini 获取默认值
        default_config = configparser.ConfigParser()
        default_config.read('config.ini', encoding='utf-8')

        if isinstance(config, configparser.ConfigParser):
            loader_config = config['News'] if 'Financial' in config else {}
        elif isinstance(config, dict):
            loader_config = config.get('News', {})
        elif hasattr(config, 'items'):  # 处理 Section 对象
            loader_config = dict(config)
        else:
            raise ValueError("不支持的配置类型")

        default_loader_config = default_config['News'] if 'News' in default_config else {}

        # 安全获取配置值
        save_path = loader_config.get('save_path', default_loader_config.get('save_path', 'data/news'))  # 修改默认路径

        # 处理整数值
        def safe_getint(source, key, default):
            value = source.get(key, None)
            if value is None:
                return default

            try:
                return int(value)
            except ValueError as e:
                raise DataLoaderError(f"配置项 {key} 的值无效（必须为整数）: {value}") from e

        max_retries = safe_getint(loader_config, 'max_retries',
                                  default_loader_config.getint('max_retries', 3) if hasattr(default_loader_config,
                                                                                           'getint') else 3)
        cache_days = safe_getint(loader_config, 'cache_days',
                                 default_loader_config.getint('cache_days', 30) if hasattr(default_loader_config,
                                                                                          'getint') else 30)

        return cls(
            source=loader_config.get('source', default_loader_config.get('source', 'cls')),
            save_path=save_path,
            max_retries=max_retries,
            cache_days=cache_days,
            thread_pool=thread_pool
        )

    def load_data(self, start_date: Union[str, datetime], end_date: Union[str, datetime]) -> pd.DataFrame:
        for attempt in range(self.max_retries + 1):
            try:
                df = super().load_data(start_date, end_date)
                if df.empty:
                    return df
                # 添加情感分析
                return self._add_sentiment_scores(df)
            except Exception as e:
                if attempt >= self.max_retries:
                    # 避免嵌套异常消息
                    if isinstance(e, DataLoaderError):
                        # 如果是已包装的异常，直接重新抛出
                        raise
                    else:
                        # 包装原始异常
                        raise DataLoaderError(f"情感分析失败: {str(e)}") from e
                logger.warning(f"第{attempt + 1}次重试...")
                time.sleep(2 ** attempt)
        raise DataLoaderError("无法加载数据")

    def _add_sentiment_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            if df.empty or 'content' not in df.columns:
                logger.warning("数据为空或缺少content列")
                return df

            # 创建情感分析器实例
            analyzer = SentimentAnalyzer(skip_config=True)

            # 应用情感分析
            df["sentiment"] = df["content"].apply(
                lambda x: self._safe_sentiment_score(x, analyzer) if pd.notna(x) and x.strip() else np.nan
            )

            # 检查是否所有情感分析都失败
            if df["sentiment"].isna().all():
                if self.debug_mode:
                    raise DataLoaderError("情感分析失败:所有情感分析均失败")
                else:
                    logger.warning("情感分析失败:所有情感分析均失败")
                    df["sentiment"] = np.nan

            return df
        except Exception as e:
            logger.error(f"情感分析失败: {str(e)}")
            if self.debug_mode:
                raise DataLoaderError(f"情感分析失败: {str(e)}") from e
            return df

    def _check_cache(self, file_path: Path) -> Tuple[bool, Optional[pd.DataFrame]]:
        if not file_path.exists():
            return False, None

        try:
            file_mtime = os.path.getmtime(file_path)
            if (time.time() - file_mtime) > (self.cache_days * 86400):
                return False, None

            df = pd.read_csv(file_path, encoding='utf-8', dtype=dtype)  # 添加dtype参数
            if df.empty:
                logger.warning(f"空缓存文件: {file_path}")
                return False, None
            return True, df

        except pd.errors.EmptyDataError as e:
            logger.error(f"空缓存文件错误: {str(e)}")
            return False, None
        except Exception as e:
            logger.error(f"加载缓存失败: {file_path} - {str(e)}")
            return False, None

    def _safe_sentiment_score(self, text: str, analyzer) -> float:
        """安全情感评分（处理单条异常）"""
        try:
            # 检查文本是否为空或空白
            if pd.isnull(text) or not text.strip():
                return np.nan
            return analyzer.snownlp_sentiment(text)
        except Exception as e:
            logger.warning(f"单条情感分析失败: {str(e)}")
            if self.debug_mode:
                raise  # 在调试模式下重新抛出异常
            return np.nan  # 在生产模式下返回NaN

    def get_metadata(self) -> Dict[str, Any]:
        """获取情感分析新闻加载器的元数据
        
        返回:
            包含加载器元数据的字典
        """
        base_meta = super().get_metadata()
        base_meta.update({
            "loader_type": "SentimentNewsLoader",
            "supports_sentiment": True,
            "debug_mode": self.debug_mode
        })
        return base_meta

    def load(self, *args, **kwargs) -> Any:
        """实现BaseDataLoader的抽象方法，包装load_data"""
        return self.load_data(*args, **kwargs)

    def validate(self, data: Any) -> bool:
        """验证加载的情感分析数据是否符合预期
        
        参数:
            data: 要验证的数据
            
        返回:
            bool: 数据是否有效
        """
        if not super().validate(data):
            return False
        return 'sentiment' in data.columns and pd.api.types.is_numeric_dtype(data['sentiment'])

    def get_required_config_fields(self) -> list:
        """获取必需的配置字段列表"""
        return ['save_path', 'max_retries', 'cache_days']

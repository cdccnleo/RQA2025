"""
convert 模块

提供 convert 相关功能和接口。
"""

import logging
import time

# 价格列（需要乘以因子）
import numpy as np

# 数据转换常量
import pandas as pd

from decimal import Decimal, getcontext
from typing import Dict, List

"""
数据转换工具
"""


logger = logging.getLogger(__name__)


def _safe_logger_log(level: int, message: str) -> None:
    """在单测环境中安全输出日志，兼容被 mock 的 handler.level。"""
    seen_handlers = set()
    visited_loggers = set()
    current_logger = logger
    depth = 0

    while current_logger and id(current_logger) not in visited_loggers and depth < 10:
        visited_loggers.add(id(current_logger))
        depth += 1

        handlers_attr = getattr(current_logger, "handlers", None)
        if isinstance(handlers_attr, (list, tuple, set)):
            handlers = list(handlers_attr)
        elif isinstance(handlers_attr, logging.Handler):
            handlers = [handlers_attr]
        else:
            handlers = []

        for handler in handlers:
            if id(handler) in seen_handlers:
                continue
            seen_handlers.add(id(handler))
            level_value = getattr(handler, "level", logging.NOTSET)
            if not isinstance(level_value, int):
                try:
                    handler.setLevel(logging.NOTSET)  # type: ignore[attr-defined]
                except Exception:
                    try:
                        handler.level = logging.NOTSET  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            handler.__dict__["level"] = logging.NOTSET  # type: ignore[attr-defined]
                        except Exception:
                            pass
            if not isinstance(getattr(handler, "level", None), int):
                try:
                    object.__setattr__(handler, "level", logging.NOTSET)  # type: ignore[attr-defined]
                except Exception:
                    pass

        if not getattr(current_logger, "propagate", True):
            break
        parent_logger = getattr(current_logger, "parent", None)
        if parent_logger is None or parent_logger is current_logger:
            break
        if not isinstance(parent_logger, logging.Logger):
            break
        current_logger = parent_logger

    try:
        logger.log(level, message)
    except TypeError:
        logging.getLogger(logger.name).log(level, message)


class DataConvertConstants:
    """数据转换相关常量"""

    # Decimal精度配置
    DECIMAL_PRECISION = 8

    # A股涨跌停价格计算
    NORMAL_STOCK_MULTIPLIER = 1.1  # 普通股票涨跌幅10%
    ST_STOCK_MULTIPLIER = 1.05  # ST股票涨跌幅5%
    PRICE_CALCULATION_BASE = 2  # 计算基准值

    # A股价格最小变动单位
    PRICE_MIN_CHANGE = 0.01  # 0.01元

    # 复权因子初始值
    INITIAL_CUM_FACTOR = 1.0

    # 价格精度
    PRICE_ROUNDING_DECIMALS = 2


def _apply_adjustment_factors_vectorized(
    data: pd.DataFrame, factor_dates: list, factor_values: list
) -> pd.DataFrame:
    """向量化应用复权因子 - 性能优化版"""
    price_columns = ["open", "high", "low", "close"]

    # 成交量列（需要除以因子）
    volume_columns = ["volume"]

    # 创建因子应用矩阵
    data_index = data.index
    factor_matrix = np.ones(len(data_index))

    # 为每个因子计算累积应用范围
    cumulative_factor = 1.0
    prev_date_idx = 0

    for i, (date, factor) in enumerate(zip(factor_dates, factor_values)):
        if date in data_index:
            # 找到日期在数据中的位置
            date_idx = data_index.get_loc(date)

            # 在前一个因子日期到当前因子日期之间的数据应用累积因子
            if date_idx > prev_date_idx:
                factor_matrix[prev_date_idx:date_idx] *= cumulative_factor

            # 更新累积因子
            cumulative_factor *= factor
            prev_date_idx = date_idx

    # 最后一个因子应用到剩余的所有数据
    if prev_date_idx < len(factor_matrix):
        factor_matrix[prev_date_idx:] *= cumulative_factor

    # 向量化应用到价格列
    for col in price_columns:
        if col in data.columns:
            data[col] = data[col].values * factor_matrix

    # 向量化应用到成交量列（除法）
    for col in volume_columns:
        if col in data.columns:
            data[col] = data[col].values / factor_matrix

    return data


class DataConverter:
    """
    convert - 工具组件

    数据转换工具类

    职责说明：
    提供通用工具函数、辅助类和基础组件

    核心职责:
    - 通用工具函数
    - 数据格式转换
    - 文件操作工具
    - 网络工具函数
    - 日期时间处理
    - 数学计算工具

    相关接口：
    - IUtilityComponent
    - IConverter
    - IHelper
    """

    # 设置Decimal精度
    getcontext().prec = DataConvertConstants.DECIMAL_PRECISION

    @staticmethod
    def calculate_limit_prices(
        prev_close: float, is_st: bool = False
    ) -> Dict[str, float]:
        """
        计算A股涨跌停价格
        Args:
            prev_close: 前收盘价
            is_st: 是否为ST股票
        Returns:
            {"upper_limit": 涨停价, "lower_limit": 跌停价}
        """
        if not isinstance(prev_close, (float, int, Decimal)):
            raise ValueError("prev_close必须是数值类型")

        multiplier = DataConvertConstants.NORMAL_STOCK_MULTIPLIER  # 普通股票
        if is_st:
            multiplier = DataConvertConstants.ST_STOCK_MULTIPLIER  # ST股票

        upper = float(Decimal(str(prev_close)) * Decimal(str(multiplier)))
        lower = float(
            Decimal(str(prev_close))
            * Decimal(str(DataConvertConstants.PRICE_CALCULATION_BASE - multiplier))
        )

        # A股价格最小变动单位为0.01元
        return {
            "upper_limit": round(upper, DataConvertConstants.PRICE_ROUNDING_DECIMALS),
            "lower_limit": round(lower, DataConvertConstants.PRICE_ROUNDING_DECIMALS),
        }

    @staticmethod
    def apply_adjustment_factor(
        data: pd.DataFrame, factors: Dict[str, float], inplace: bool = False
    ) -> pd.DataFrame:
        """应用复权因子调整历史价格数据 (性能优化版)"""
        start_time = time.time()

        if not inplace:
            data = data.copy()

        # 按日期升序排列
        data.sort_index(ascending=True, inplace=True)

        # 性能优化：向量化复权因子应用
        if factors:
            # 转换因子为时间序列
            factor_dates = sorted(factors.keys())
            factor_values = [factors[date] for date in factor_dates]

            # 使用向量化操作计算累积因子
            data = _apply_adjustment_factors_vectorized(
                data, factor_dates, factor_values
            )

        processing_time = time.time() - start_time
        _safe_logger_log(
            logging.DEBUG,
            f"复权因子应用完成，数据行数: {len(data)}, 处理时间: {processing_time:.4f}s",
        )

        return data

    @staticmethod
    def parse_margin_data(raw_data: Dict) -> pd.DataFrame:
        """
        解析融资融券数据
        Args:
            raw_data: 原始API返回的融资融券数据
        Returns:
            标准化后的DataFrame
        """
        required_fields = [
            "symbol",
            "name",
            "margin_balance",
            "short_balance",
            "margin_buy",
            "short_sell",
            "repayment",
        ]

        # 验证字段完整性
        for field in required_fields:
            if field not in raw_data:
                raise ValueError(f"缺少必要字段: {field}")

        df = pd.DataFrame([raw_data])

        # 转换数据类型
        numeric_cols = [
            "margin_balance",
            "short_balance",
            "margin_buy",
            "short_sell",
            "repayment",
        ]

        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        # 计算净融资数据
        df["net_margin"] = df["margin_buy"] - df["repayment"]

        return df

    @staticmethod
    def normalize_dragon_board(raw_data: List[Dict]) -> pd.DataFrame:
        """
        标准化龙虎榜数据
        Args:
            raw_data: 原始龙虎榜数据列表
        Returns:
            标准化DataFrame
        """
        df = pd.DataFrame(raw_data)

        # 统一营业部名称格式
        if "branch_name" in df.columns:
            df["branch_name"] = df["branch_name"].str.replace(r"\s+", "", regex=True)

        # 解析买卖方向
        if "direction" in df.columns:
            df["is_buy"] = df["direction"].str.contains("买")

        # 转换金额单位(万->元)
        amount_cols = [col for col in df.columns if "amount" in col.lower()]
        for col in amount_cols:
            if df[col].dtype == "object":
                df[col] = pd.to_numeric(
                    df[col].str.replace(",", "").str.replace("万", ""), errors="coerce"
                )

        return df

    @staticmethod
    def convert_frequency(
        data: pd.DataFrame, freq: str, agg_rules: Dict[str, str] = None
    ):
        """
        转换数据频率
        Args:
            data: 原始数据(必须包含datetime索引)
            freq: 目标频率('1min','5min','1H','1D','1W','1M')
            agg_rules: 各列的聚合规则
        Returns:
            转换频率后的DataFrame
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据必须包含datetime索引")

        default_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }

        agg_rules = agg_rules or default_rules

        # 保留原始数据中存在的列
        valid_cols = [col for col in agg_rules if col in data.columns]
        agg_rules = {col: agg_rules[col] for col in valid_cols}

        if not valid_cols:
            raise ValueError("没有有效的列可以进行频率转换")

        # 使用pandas的resample进行频率转换
        try:
            result = data.resample(freq).agg(agg_rules)
            return result
        except Exception as e:
            _safe_logger_log(logging.ERROR, f"频率转换失败: {e}")
            raise ValueError(f"不支持的频率或数据格式: {freq}")


# 向后兼容的别名
Convert = DataConverter
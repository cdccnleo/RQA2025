"""
数据转换工具
"""
import pandas as pd
import numpy as np
from typing import Union, Dict, List
from decimal import Decimal, getcontext
from .date import DateUtils

class DataConverter:
    """数据转换工具类"""

    # 设置Decimal精度
    getcontext().prec = 8

    @staticmethod
    def calculate_limit_prices(prev_close: float, is_st: bool = False) -> Dict[str, float]:
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

        multiplier = 1.1  # 普通股票
        if is_st:
            multiplier = 1.05  # ST股票

        upper = float(Decimal(str(prev_close)) * Decimal(str(multiplier)))
        lower = float(Decimal(str(prev_close)) * Decimal(str(2 - multiplier)))

        # A股价格最小变动单位为0.01元
        return {
            "upper_limit": round(upper, 2),
            "lower_limit": round(lower, 2)
        }

    @staticmethod
    def apply_adjustment_factor(data: pd.DataFrame,
                              factors: Dict[str, float],
                              inplace: bool = False) -> Union[None, pd.DataFrame]:
        """
        应用复权因子调整历史价格数据
        Args:
            data: 包含['open','high','low','close','volume']的DataFrame
            factors: {'date': 复权因子}
            inplace: 是否原地修改
        Returns:
            复权后的DataFrame (当inplace=False时)
        """
        if not inplace:
            data = data.copy()

        # 按日期升序排列
        data.sort_index(ascending=True, inplace=True)

        # 累积复权因子
        cum_factor = 1.0
        for date, factor in sorted(factors.items(), key=lambda x: x[0]):
            if date in data.index:
                mask = data.index >= date
                data.loc[mask, ['open','high','low','close']] *= factor
                data.loc[mask, 'volume'] /= factor
                cum_factor *= factor

        if not inplace:
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
            'symbol', 'name', 'margin_balance', 'short_balance',
            'margin_buy', 'short_sell', 'repayment'
        ]

        # 验证字段完整性
        for field in required_fields:
            if field not in raw_data:
                raise ValueError(f"缺少必要字段: {field}")

        df = pd.DataFrame(raw_data)

        # 转换数据类型
        numeric_cols = [
            'margin_balance', 'short_balance',
            'margin_buy', 'short_sell', 'repayment'
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        # 计算净融资数据
        df['net_margin'] = df['margin_buy'] - df['repayment']

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
        if 'branch_name' in df.columns:
            df['branch_name'] = df['branch_name'].str.replace(r'\s+', '', regex=True)

        # 解析买卖方向
        if 'direction' in df.columns:
            df['is_buy'] = df['direction'].str.contains('买')

        # 转换金额单位(万->元)
        amount_cols = [col for col in df.columns if 'amount' in col.lower()]
        for col in amount_cols:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(
                    df[col].str.replace(',', '').str.replace('万', ''),
                    errors='coerce'
                ) * 10000

        return df

    @staticmethod
    def convert_frequency(data: pd.DataFrame,
                        freq: str,
                        agg_rules: Dict[str, str] = None) -> pd.DataFrame:
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
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'amount': 'sum'
        }

        agg_rules = agg_rules or default_rules

        # 保留原始数据中存在的列
        valid_cols = [col for col in agg_rules if col in data.columns]
        agg_rules = {col: agg_rules[col] for col in valid_cols}

        return data.resample(freq).agg(agg_rules)

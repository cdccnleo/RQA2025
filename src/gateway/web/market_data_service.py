"""
市场数据查询服务
从数据管理层(PostgreSQL)获取历史市场数据
"""

import logging
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketDataService:
    """市场数据查询服务"""

    def __init__(self):
        self._cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self._cache_ttl = 300  # 5分钟缓存
        self._available_symbols: Optional[List[str]] = None
        self._symbols_cache_time: Optional[float] = None
        self._symbols_cache_ttl = 600  # 10分钟缓存

    def get_available_symbols(self) -> List[str]:
        """
        获取数据库中可用的股票代码列表
        
        Returns:
            股票代码列表
        """
        try:
            # 检查缓存
            if (self._available_symbols is not None and 
                self._symbols_cache_time is not None and
                datetime.now().timestamp() - self._symbols_cache_time < self._symbols_cache_ttl):
                logger.debug(f"从缓存获取可用股票代码: {len(self._available_symbols)} 个")
                return self._available_symbols

            # 从数据库查询
            from .postgresql_persistence import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()

            query = """
                SELECT DISTINCT symbol
                FROM akshare_stock_data
                ORDER BY symbol
            """

            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()

            symbols = [row[0] for row in rows if row[0]]
            
            # 更新缓存
            self._available_symbols = symbols
            self._symbols_cache_time = datetime.now().timestamp()

            logger.info(f"从数据库获取可用股票代码: {len(symbols)} 个")
            return symbols

        except Exception as e:
            logger.error(f"获取可用股票代码失败: {e}")
            return []

    def get_default_symbol(self) -> Optional[str]:
        """
        获取默认股票代码（数据库中数据最多的股票）
        
        Returns:
            默认股票代码
        """
        try:
            from .postgresql_persistence import get_db_connection
            conn = get_db_connection()
            cursor = conn.cursor()

            query = """
                SELECT symbol, COUNT(*) as count
                FROM akshare_stock_data
                GROUP BY symbol
                ORDER BY count DESC
                LIMIT 1
            """

            cursor.execute(query)
            row = cursor.fetchone()
            cursor.close()

            if row:
                symbol = row[0]
                count = row[1]
                logger.info(f"获取默认股票代码: {symbol} (记录数: {count})")
                return symbol
            else:
                logger.warning("数据库中没有股票数据")
                return None

        except Exception as e:
            logger.error(f"获取默认股票代码失败: {e}")
            return None

    def get_stock_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        获取股票历史数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            limit: 限制条数

        Returns:
            pandas DataFrame with columns: open, high, low, close, volume
        """
        try:
            # 1. 检查缓存
            cache_key = f"{symbol}_{start_date}_{end_date}_{limit}"
            if cache_key in self._cache:
                cached_data, timestamp = self._cache[cache_key]
                if datetime.now().timestamp() - timestamp < self._cache_ttl:
                    logger.debug(f"从缓存获取数据: {symbol}")
                    return cached_data

            # 2. 从数据库查询
            data = self._query_from_database(symbol, start_date, end_date, limit)

            # 3. 转换为标准格式
            df = self._convert_to_dataframe(data)

            # 4. 更新缓存
            self._cache[cache_key] = (df, datetime.now().timestamp())

            logger.info(f"成功获取股票数据: {symbol}, 记录数: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"获取股票数据失败 {symbol}: {e}")
            # 返回空DataFrame
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def _query_from_database(
        self,
        symbol: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> List[Dict[str, Any]]:
        """从PostgreSQL查询数据"""
        try:
            from .postgresql_persistence import get_db_connection

            conn = get_db_connection()
            cursor = conn.cursor()

            query = """
                SELECT date, open_price as open, high_price as high, low_price as low, close_price as close, volume
                FROM akshare_stock_data
                WHERE symbol = %s
            """
            params = [symbol]

            if start_date:
                query += " AND date >= %s"
                params.append(start_date.date() if isinstance(start_date, datetime) else start_date)
            if end_date:
                query += " AND date <= %s"
                params.append(end_date.date() if isinstance(end_date, datetime) else end_date)

            query += " ORDER BY date DESC LIMIT %s"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()

            logger.debug(f"数据库查询返回 {len(rows)} 条记录")

            return [
                {
                    'date': row[0],
                    'open': float(row[1]) if row[1] is not None else 0.0,
                    'high': float(row[2]) if row[2] is not None else 0.0,
                    'low': float(row[3]) if row[3] is not None else 0.0,
                    'close': float(row[4]) if row[4] is not None else 0.0,
                    'volume': float(row[5]) if row[5] is not None else 0.0
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"数据库查询失败: {e}")
            return []

    def _convert_to_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """转换为pandas DataFrame"""
        if not data:
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

        try:
            df = pd.DataFrame(data)

            # 设置日期索引
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            # 确保列名和数据类型正确
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0.0

            df = df[required_columns].astype(float)

            return df

        except Exception as e:
            logger.error(f"数据转换失败: {e}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        logger.info("市场数据缓存已清除")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return {
            'cache_size': len(self._cache),
            'cache_ttl': self._cache_ttl
        }


# 单例实例
_market_data_service: Optional[MarketDataService] = None


def get_market_data_service() -> MarketDataService:
    """获取市场数据服务实例"""
    global _market_data_service
    if _market_data_service is None:
        _market_data_service = MarketDataService()
    return _market_data_service

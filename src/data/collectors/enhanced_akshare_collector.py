#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版 AKShare 数据采集器

功能：
- 支持增量采集
- 支持多股票批量采集
- 数据质量检查
- 自动补全缺失数据

作者: AI Assistant
创建日期: 2026-02-21
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self):
        """初始化数据质量检查器"""
        self.quality_rules = {
            'price_positive': self._check_price_positive,
            'volume_positive': self._check_volume_positive,
            'price_range': self._check_price_range,
            'no_duplicates': self._check_no_duplicates,
            'date_continuity': self._check_date_continuity
        }
    
    def check_quality(self, data: List[Dict[str, Any]], symbol: str) -> Tuple[bool, List[str]]:
        """
        检查数据质量
        
        Args:
            data: 股票数据列表
            symbol: 股票代码
            
        Returns:
            (是否通过检查, 错误信息列表)
        """
        errors = []
        
        if not data:
            errors.append(f"{symbol}: 数据为空")
            return False, errors
        
        for rule_name, rule_func in self.quality_rules.items():
            try:
                passed, error_msg = rule_func(data, symbol)
                if not passed:
                    errors.append(error_msg)
            except Exception as e:
                errors.append(f"{symbol}: 规则 {rule_name} 检查失败: {e}")
        
        return len(errors) == 0, errors
    
    def _check_price_positive(self, data: List[Dict[str, Any]], symbol: str) -> Tuple[bool, str]:
        """检查价格是否为正数"""
        for record in data:
            for price_field in ['open', 'high', 'low', 'close']:
                if record.get(price_field, 0) <= 0:
                    return False, f"{symbol}: {price_field} 价格必须为正数"
        return True, ""
    
    def _check_volume_positive(self, data: List[Dict[str, Any]], symbol: str) -> Tuple[bool, str]:
        """检查成交量是否为正数"""
        for record in data:
            if record.get('volume', 0) < 0:
                return False, f"{symbol}: 成交量不能为负数"
        return True, ""
    
    def _check_price_range(self, data: List[Dict[str, Any]], symbol: str) -> Tuple[bool, str]:
        """检查价格范围是否合理"""
        for record in data:
            high = record.get('high', 0)
            low = record.get('low', 0)
            if high < low:
                return False, f"{symbol}: 最高价不能低于最低价"
        return True, ""
    
    def _check_no_duplicates(self, data: List[Dict[str, Any]], symbol: str) -> Tuple[bool, str]:
        """检查是否有重复日期"""
        dates = [record.get('date') for record in data]
        if len(dates) != len(set(dates)):
            return False, f"{symbol}: 存在重复日期"
        return True, ""
    
    def _check_date_continuity(self, data: List[Dict[str, Any]], symbol: str) -> Tuple[bool, str]:
        """检查日期连续性（简化版）"""
        if len(data) < 2:
            return True, ""
        
        # 检查数据是否按日期排序
        dates = [record.get('date') for record in data]
        if dates != sorted(dates):
            return False, f"{symbol}: 数据未按日期排序"
        
        return True, ""


class EnhancedAKShareCollector:
    """
    增强版 AKShare 数据采集器
    
    职责：
    1. 支持增量采集
    2. 支持多股票批量采集
    3. 数据质量检查
    4. 自动补全缺失数据
    """
    
    def __init__(self, max_workers: int = 4):
        """
        初始化增强版 AKShare 采集器
        
        Args:
            max_workers: 最大工作线程数
        """
        self._akshare_available = self._check_akshare()
        self.max_workers = max_workers
        self.quality_checker = DataQualityChecker()
        
        if self._akshare_available:
            logger.info(f"增强版 AKShare 采集器初始化成功，最大工作线程数: {max_workers}")
        else:
            logger.warning("AKShare 库不可用，采集器将无法工作")
    
    def _check_akshare(self) -> bool:
        """检查 AKShare 库是否可用"""
        try:
            import akshare as ak
            return True
        except ImportError:
            logger.error("AKShare 库未安装，请运行: pip install akshare")
            return False
    
    def get_last_collection_date(self, symbol: str) -> Optional[str]:
        """
        获取上次采集日期（用于增量采集）
        
        Args:
            symbol: 股票代码
            
        Returns:
            上次采集日期 (YYYY-MM-DD)，如果没有则返回 None
        """
        try:
            from src.gateway.web.postgresql_persistence import get_db_connection
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 查询该股票的最大日期
            cursor.execute(
                "SELECT MAX(date) FROM akshare_stock_data WHERE symbol = %s",
                (symbol,)
            )
            result = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if result and result[0]:
                # 返回下一天作为开始日期
                last_date = result[0]
                if isinstance(last_date, str):
                    last_date = datetime.strptime(last_date, "%Y-%m-%d")
                elif hasattr(last_date, 'strftime'):
                    # 已经是date或datetime对象
                    pass
                else:
                    # 尝试转换
                    last_date = datetime.combine(last_date, datetime.min.time())
                
                next_date = last_date + timedelta(days=1)
                return next_date.strftime("%Y-%m-%d")
            
            return None
        except Exception as e:
            logger.error(f"获取上次采集日期失败 {symbol}: {e}")
            return None
    
    def collect_stock_data_incremental(
        self,
        symbol: str,
        days_back: int = 30,
        adjust: str = "qfq"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        增量采集股票历史数据
        
        Args:
            symbol: 股票代码
            days_back: 如果数据库中没有数据，默认采集多少天的历史数据
            adjust: 复权类型
            
        Returns:
            股票数据列表
        """
        # 获取上次采集日期
        start_date = self.get_last_collection_date(symbol)
        
        if start_date is None:
            # 数据库中没有数据，采集默认天数的历史数据
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            logger.info(f"{symbol}: 数据库中没有数据，将采集最近 {days_back} 天的历史数据")
        else:
            logger.info(f"{symbol}: 上次采集到 {start_date}，将从该日期开始增量采集")
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        # 如果开始日期已经超过结束日期，说明数据已经是最新的
        if start_date > end_date:
            logger.info(f"{symbol}: 数据已经是最新的，无需采集")
            return []
        
        return self.collect_stock_data(symbol, start_date, end_date, adjust)
    
    def collect_stock_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq"
    ) -> Optional[List[Dict[str, Any]]]:
        """
        采集股票历史数据（基础方法）
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型
            
        Returns:
            股票数据列表
        """
        if not self._akshare_available:
            logger.error("AKShare 库不可用，无法采集数据")
            return None
        
        try:
            import akshare as ak
            
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            logger.info(f"开始采集股票数据: {symbol}, 时间范围: {start_date} 到 {end_date}")
            
            # 转换日期格式
            start_date_fmt = start_date.replace("-", "")
            end_date_fmt = end_date.replace("-", "")
            
            # 使用 AKShare 获取数据
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date_fmt,
                end_date=end_date_fmt,
                adjust=adjust
            )
            
            if df.empty:
                logger.warning(f"未获取到数据: {symbol}")
                return None
            
            logger.info(f"成功获取原始数据: {symbol}, 记录数: {len(df)}")
            
            # 转换数据格式
            data = self._convert_data(df, symbol)
            
            # 数据质量检查
            quality_passed, errors = self.quality_checker.check_quality(data, symbol)
            if not quality_passed:
                logger.warning(f"{symbol}: 数据质量检查未通过:")
                for error in errors:
                    logger.warning(f"  - {error}")
            
            logger.info(f"数据转换完成: {symbol}, 记录数: {len(data)}")
            return data
            
        except Exception as e:
            logger.error(f"采集股票数据失败 {symbol}: {e}")
            return None
    
    def collect_multiple_stocks(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
        use_incremental: bool = True
    ) -> Dict[str, Optional[List[Dict[str, Any]]]]:
        """
        批量采集多股票数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型
            use_incremental: 是否使用增量采集
            
        Returns:
            股票代码到数据的映射
        """
        results = {}
        
        logger.info(f"开始批量采集 {len(symbols)} 只股票的数据")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_symbol = {}
            
            for symbol in symbols:
                if use_incremental:
                    future = executor.submit(
                        self.collect_stock_data_incremental,
                        symbol,
                        30,  # days_back
                        adjust
                    )
                else:
                    future = executor.submit(
                        self.collect_stock_data,
                        symbol,
                        start_date,
                        end_date,
                        adjust
                    )
                future_to_symbol[future] = symbol
            
            # 收集结果
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                    if data:
                        logger.info(f"{symbol}: 成功采集 {len(data)} 条记录")
                    else:
                        logger.warning(f"{symbol}: 未获取到数据")
                except Exception as e:
                    logger.error(f"{symbol}: 采集失败: {e}")
                    results[symbol] = None
        
        # 统计结果
        success_count = sum(1 for data in results.values() if data is not None)
        logger.info(f"批量采集完成: {success_count}/{len(symbols)} 只股票成功")
        
        return results
    
    def _convert_data(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> List[Dict[str, Any]]:
        """
        转换数据格式
        
        Args:
            df: AKShare 返回的 DataFrame
            symbol: 股票代码
            
        Returns:
            标准格式的数据列表
        """
        data = []
        
        for _, row in df.iterrows():
            try:
                record = {
                    "symbol": symbol,
                    "date": row.get("日期") if "日期" in row else row.get("date"),
                    "open": float(row.get("开盘", 0)) if "开盘" in row else float(row.get("open", 0)),
                    "high": float(row.get("最高", 0)) if "最高" in row else float(row.get("high", 0)),
                    "low": float(row.get("最低", 0)) if "最低" in row else float(row.get("low", 0)),
                    "close": float(row.get("收盘", 0)) if "收盘" in row else float(row.get("close", 0)),
                    "volume": int(float(row.get("成交量", 0))) if "成交量" in row else int(float(row.get("volume", 0))),
                    "amount": float(row.get("成交额", 0)) if "成交额" in row else float(row.get("amount", 0)),
                    "amplitude": float(row.get("振幅", 0)) if "振幅" in row else float(row.get("amplitude", 0)),
                    "pct_change": float(row.get("涨跌幅", 0)) if "涨跌幅" in row else float(row.get("pct_change", 0)),
                    "change_amount": float(row.get("涨跌额", 0)) if "涨跌额" in row else float(row.get("change_amount", 0)),
                    "turnover": float(row.get("换手率", 0)) if "换手率" in row else float(row.get("turnover", 0))
                }
                data.append(record)
            except Exception as e:
                logger.warning(f"转换数据行失败: {e}, row={row.to_dict()}")
                continue
        
        return data
    
    def save_to_database(self, data: List[Dict[str, Any]], symbol: str) -> bool:
        """
        将数据保存到数据库
        
        Args:
            data: 股票数据列表
            symbol: 股票代码
            
        Returns:
            是否保存成功
        """
        if not data:
            logger.warning(f"没有数据需要保存: {symbol}")
            return False
        
        try:
            from src.gateway.web.postgresql_persistence import get_db_connection
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 先检查表是否存在，如果不存在则创建
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS akshare_stock_data (
                    id BIGSERIAL PRIMARY KEY,
                    source_id VARCHAR(50) NOT NULL,
                    symbol VARCHAR(20) NOT NULL,
                    date DATE NOT NULL,
                    open_price DECIMAL(15, 6),
                    high_price DECIMAL(15, 6),
                    low_price DECIMAL(15, 6),
                    close_price DECIMAL(15, 6),
                    volume BIGINT,
                    amount DECIMAL(20, 2),
                    pct_change DECIMAL(10, 4),
                    change DECIMAL(15, 6),
                    turnover_rate DECIMAL(10, 4),
                    amplitude DECIMAL(10, 4),
                    data_source VARCHAR(50) DEFAULT 'akshare',
                    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    persistence_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            
            # 检查约束是否存在，如果不存在则添加
            cursor.execute("""
                SELECT COUNT(*) FROM pg_constraint 
                WHERE conname = 'unique_akshare_record'
            """)
            constraint_exists = cursor.fetchone()[0] > 0
            
            if not constraint_exists:
                try:
                    cursor.execute("""
                        ALTER TABLE akshare_stock_data 
                        ADD CONSTRAINT unique_akshare_record 
                        UNIQUE (source_id, symbol, date)
                    """)
                    conn.commit()
                    logger.info("已添加唯一约束: unique_akshare_record")
                except Exception as e:
                    logger.warning(f"添加唯一约束失败（可能已存在）: {e}")
                    conn.rollback()
            
            # 准备插入语句（不使用ON CONFLICT，先删除后插入）
            source_id = "enhanced_akshare"
            
            # 获取要插入的所有日期
            dates = [record["date"] for record in data]
            
            # 先删除已存在的记录
            delete_query = """
                DELETE FROM akshare_stock_data 
                WHERE source_id = %s AND symbol = %s AND date = ANY(%s)
            """
            cursor.execute(delete_query, (source_id, symbol, dates))
            
            # 准备插入语句
            insert_query = """
                INSERT INTO akshare_stock_data (
                    source_id, symbol, date, open_price, high_price, low_price, close_price, 
                    volume, amount, amplitude, pct_change, change, turnover_rate
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # 批量插入数据
            records = []
            for record in data:
                records.append((
                    source_id,
                    record["symbol"],
                    record["date"],
                    record["open"],      # open_price
                    record["high"],      # high_price
                    record["low"],       # low_price
                    record["close"],     # close_price
                    record["volume"],
                    record["amount"],
                    record["amplitude"],
                    record["pct_change"],
                    record["change_amount"],  # change
                    record["turnover"]   # turnover_rate
                ))
            
            cursor.executemany(insert_query, records)
            conn.commit()
            
            cursor.close()
            conn.close()
            
            logger.info(f"成功保存数据到数据库: {symbol}, 记录数: {len(data)}")
            return True
            
        except Exception as e:
            logger.error(f"保存数据到数据库失败 {symbol}: {e}")
            return False
    
    def collect_and_save(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust: str = "qfq",
        use_incremental: bool = True
    ) -> bool:
        """
        采集并保存股票数据（一站式方法）
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            adjust: 复权类型
            use_incremental: 是否使用增量采集
            
        Returns:
            是否成功
        """
        if use_incremental:
            data = self.collect_stock_data_incremental(symbol, 30, adjust)
        else:
            data = self.collect_stock_data(symbol, start_date, end_date, adjust)
        
        if data:
            return self.save_to_database(data, symbol)
        
        return False
    
    def fill_missing_data(
        self,
        symbol: str,
        expected_dates: List[str]
    ) -> bool:
        """
        补全缺失数据
        
        Args:
            symbol: 股票代码
            expected_dates: 期望的日期列表
            
        Returns:
            是否补全成功
        """
        try:
            # 获取数据库中已有的日期
            from src.gateway.web.postgresql_persistence import get_db_connection
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT date FROM akshare_stock_data WHERE symbol = %s",
                (symbol,)
            )
            existing_dates = {str(row[0]) for row in cursor.fetchall()}
            
            cursor.close()
            conn.close()
            
            # 找出缺失的日期
            missing_dates = [d for d in expected_dates if d not in existing_dates]
            
            if not missing_dates:
                logger.info(f"{symbol}: 没有缺失数据需要补全")
                return True
            
            logger.info(f"{symbol}: 发现 {len(missing_dates)} 个缺失日期，开始补全")
            
            # 按日期范围分组，减少API调用
            missing_dates.sort()
            date_ranges = self._group_dates_to_ranges(missing_dates)
            
            success_count = 0
            for start, end in date_ranges:
                data = self.collect_stock_data(symbol, start, end)
                if data and self.save_to_database(data, symbol):
                    success_count += 1
            
            logger.info(f"{symbol}: 补全完成，成功 {success_count}/{len(date_ranges)} 个时间段")
            return success_count == len(date_ranges)
            
        except Exception as e:
            logger.error(f"补全缺失数据失败 {symbol}: {e}")
            return False
    
    def _group_dates_to_ranges(self, dates: List[str]) -> List[Tuple[str, str]]:
        """
        将日期列表分组为连续的范围
        
        Args:
            dates: 日期列表
            
        Returns:
            日期范围列表 [(start, end), ...]
        """
        if not dates:
            return []
        
        ranges = []
        start = dates[0]
        end = dates[0]
        
        for i in range(1, len(dates)):
            current = datetime.strptime(dates[i], "%Y-%m-%d")
            previous = datetime.strptime(dates[i-1], "%Y-%m-%d")
            
            # 如果日期不连续（跳过周末）
            if (current - previous).days > 3:  # 超过3天认为是不连续
                ranges.append((start, end))
                start = dates[i]
            
            end = dates[i]
        
        ranges.append((start, end))
        return ranges


# 单例实例
_collector: Optional[EnhancedAKShareCollector] = None


def get_enhanced_akshare_collector(max_workers: int = 4) -> EnhancedAKShareCollector:
    """
    获取增强版 AKShare 采集器单例
    
    Args:
        max_workers: 最大工作线程数
        
    Returns:
        EnhancedAKShareCollector实例
    """
    global _collector
    if _collector is None:
        _collector = EnhancedAKShareCollector(max_workers=max_workers)
    return _collector

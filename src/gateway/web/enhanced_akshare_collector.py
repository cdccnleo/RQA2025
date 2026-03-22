#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版 AKShare 数据采集器

解决连接中断问题，增加重试机制和超时配置。
"""

import time
import logging
import random
from typing import List, Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    """
    带指数退避的重试装饰器
    
    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        exponential_base: 指数基数
        jitter: 是否添加随机抖动
        exceptions: 需要重试的异常类型
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"❌ {func.__name__} 重试 {max_retries} 次后仍然失败: {e}")
                        raise
                    
                    # 计算延迟时间（指数退避）
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # 添加随机抖动
                    if jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        f"⚠️ {func.__name__} 第 {attempt + 1} 次失败: {e}，"
                        f"{delay:.2f} 秒后重试..."
                    )
                    
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    return decorator


class EnhancedAKShareCollector:
    """
    增强版 AKShare 数据采集器
    
    特性：
    - 指数退避重试机制
    - 请求超时配置
    - 连接池管理
    - 错误日志记录
    """
    
    def __init__(
        self,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        request_delay: float = 0.5
    ):
        """
        初始化采集器
        
        Args:
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            request_delay: 请求间隔（秒），避免触发限流
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_delay = request_delay
        self._last_request_time = 0
        
        self._init_akshare()
    
    def _init_akshare(self):
        """
        初始化 AKShare 配置
        """
        try:
            import akshare as ak
            self.ak = ak
            
            # 配置请求超时（如果 AKShare 支持）
            if hasattr(ak, 'set_timeout'):
                ak.set_timeout(self.timeout)
            
            logger.info(f"✅ AKShare 初始化成功，版本: {ak.__version__}")
            
        except ImportError:
            logger.error("❌ AKShare 未安装，请执行: pip install akshare")
            raise
    
    def _wait_for_rate_limit(self):
        """
        等待以遵守速率限制
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self._last_request_time = time.time()
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def get_stock_list(self) -> List[Dict[str, Any]]:
        """
        获取 A股股票列表
        
        Returns:
            股票列表
        """
        self._wait_for_rate_limit()
        
        logger.info("📊 正在获取 A股股票列表...")
        
        try:
            df = self.ak.stock_zh_a_spot_em()
            
            stocks = []
            for _, row in df.iterrows():
                stocks.append({
                    'symbol': row.get('代码', ''),
                    'name': row.get('名称', ''),
                    'price': row.get('最新价', 0),
                    'change_pct': row.get('涨跌幅', 0),
                    'volume': row.get('成交量', 0),
                    'amount': row.get('成交额', 0),
                    'market': 'A股'
                })
            
            logger.info(f"✅ 获取到 {len(stocks)} 只 A股股票")
            return stocks
            
        except Exception as e:
            logger.error(f"❌ 获取股票列表失败: {e}")
            raise
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def get_stock_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        period: str = 'daily',
        adjust: str = 'qfq'
    ) -> List[Dict[str, Any]]:
        """
        获取股票历史数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期 (YYYYMMDD)
            end_date: 结束日期 (YYYYMMDD)
            period: 周期 (daily/weekly/monthly)
            adjust: 复权类型 (qfq/hfq/None)
        
        Returns:
            历史数据列表
        """
        self._wait_for_rate_limit()
        
        logger.debug(f"📊 获取 {symbol} 历史数据: {start_date} ~ {end_date}")
        
        try:
            df = self.ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust
            )
            
            records = []
            for _, row in df.iterrows():
                records.append({
                    'symbol': symbol,
                    'date': row.get('日期', '').strftime('%Y-%m-%d') if hasattr(row.get('日期', ''), 'strftime') else str(row.get('日期', '')),
                    'open': float(row.get('开盘', 0) or 0),
                    'high': float(row.get('最高', 0) or 0),
                    'low': float(row.get('最低', 0) or 0),
                    'close': float(row.get('收盘', 0) or 0),
                    'volume': int(row.get('成交量', 0) or 0),
                    'amount': float(row.get('成交额', 0) or 0),
                    'pct_change': float(row.get('涨跌幅', 0) or 0),
                    'change': float(row.get('涨跌额', 0) or 0),
                    'turnover_rate': float(row.get('换手率', 0) or 0),
                    'amplitude': float(row.get('振幅', 0) or 0),
                    'data_type': period,
                    'data_source': 'akshare'
                })
            
            logger.debug(f"✅ 获取到 {len(records)} 条历史记录")
            return records
            
        except Exception as e:
            logger.error(f"❌ 获取 {symbol} 历史数据失败: {e}")
            raise
    
    def collect_batch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        批量采集股票数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            progress_callback: 进度回调函数
        
        Returns:
            股票代码到数据的映射
        """
        results = {}
        total = len(symbols)
        
        logger.info(f"📊 开始批量采集 {total} 只股票数据...")
        
        for i, symbol in enumerate(symbols):
            try:
                data = self.get_stock_history(symbol, start_date, end_date)
                results[symbol] = data
                
                if progress_callback:
                    progress_callback(i + 1, total, symbol, True, None)
                
            except Exception as e:
                logger.warning(f"⚠️ 采集 {symbol} 失败: {e}")
                results[symbol] = []
                
                if progress_callback:
                    progress_callback(i + 1, total, symbol, False, str(e))
        
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"✅ 批量采集完成: {success_count}/{total} 成功")
        
        return results
    
    def test_connection(self) -> Dict[str, Any]:
        """
        测试 AKShare API 连接
        
        Returns:
            测试结果
        """
        result = {
            'akshare_installed': False,
            'version': None,
            'api_accessible': False,
            'sample_data_count': 0,
            'error': None
        }
        
        try:
            import akshare as ak
            result['akshare_installed'] = True
            result['version'] = ak.__version__
            
            # 测试 API 连接
            self._wait_for_rate_limit()
            df = ak.stock_zh_a_spot_em()
            
            result['api_accessible'] = True
            result['sample_data_count'] = len(df)
            
            logger.info(f"✅ AKShare 连接测试成功: {len(df)} 只股票")
            
        except ImportError as e:
            result['error'] = f"AKShare 未安装: {e}"
            logger.error(f"❌ {result['error']}")
            
        except Exception as e:
            result['error'] = f"API 连接失败: {e}"
            logger.error(f"❌ {result['error']}")
        
        return result


# 单例实例
_collector_instance: Optional[EnhancedAKShareCollector] = None


def get_enhanced_collector(
    timeout: int = 30,
    max_retries: int = 3
) -> EnhancedAKShareCollector:
    """
    获取增强版采集器单例
    
    Args:
        timeout: 请求超时时间
        max_retries: 最大重试次数
    
    Returns:
        EnhancedAKShareCollector 实例
    """
    global _collector_instance
    
    if _collector_instance is None:
        _collector_instance = EnhancedAKShareCollector(
            timeout=timeout,
            max_retries=max_retries
        )
    
    return _collector_instance


if __name__ == "__main__":
    # 测试连接
    collector = get_enhanced_collector()
    result = collector.test_connection()
    print(f"\n连接测试结果:")
    for key, value in result.items():
        print(f"  {key}: {value}")

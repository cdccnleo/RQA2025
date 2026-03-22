"""
Tushare 数据源采集器

提供 Tushare API 的数据采集功能，作为 AKShare 的备用数据源。
支持股票行情、财务数据、指数数据等。
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from functools import wraps
import json

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
        
    Returns:
        装饰器函数
    """
    import random
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (exponential_base ** attempt), max_delay)
                        if jitter:
                            delay = delay * (0.5 + random.random())
                        logger.warning(
                            f"⚠️ {func.__name__} 第 {attempt + 1} 次失败: {e}，"
                            f"{delay:.2f} 秒后重试..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"❌ {func.__name__} 重试 {max_retries} 次后仍然失败: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


class TushareCollector:
    """
    Tushare 数据源采集器
    
    作为 AKShare 的备用数据源，提供稳定的股票数据采集能力。
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        token: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        request_interval: float = 0.5
    ):
        """
        初始化 Tushare 采集器
        
        Args:
            token: Tushare API Token
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            request_interval: 请求间隔时间（秒）
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.token = token or os.getenv('TUSHARE_TOKEN', '')
        self.timeout = timeout
        self.max_retries = max_retries
        self.request_interval = request_interval
        self._last_request_time = 0
        self._request_lock = threading.Lock()
        
        self._api = None
        self._initialize_api()
        
        self._initialized = True
        logger.info("Tushare 采集器初始化完成")
    
    def _initialize_api(self):
        """初始化 Tushare API"""
        try:
            import tushare as ts
            if self.token:
                ts.set_token(self.token)
                self._api = ts.pro_api()
                logger.info("✅ Tushare API 初始化成功")
            else:
                logger.warning("⚠️ Tushare Token 未配置，部分功能受限")
                self._api = ts
        except ImportError:
            logger.error("❌ Tushare 模块未安装，请运行: pip install tushare")
            self._api = None
        except Exception as e:
            logger.error(f"❌ Tushare API 初始化失败: {e}")
            self._api = None
    
    def _rate_limit(self):
        """请求限流控制"""
        with self._request_lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.request_interval:
                time.sleep(self.request_interval - elapsed)
            self._last_request_time = time.time()
    
    @property
    def is_available(self) -> bool:
        """检查 API 是否可用"""
        return self._api is not None
    
    def test_connection(self) -> Dict[str, Any]:
        """
        测试 API 连接
        
        Returns:
            连接测试结果
        """
        result = {
            'tushare_installed': False,
            'token_configured': bool(self.token),
            'api_accessible': False,
            'sample_data_count': 0,
            'error': None
        }
        
        try:
            import tushare as ts
            result['tushare_installed'] = True
            
            if not self._api:
                result['error'] = 'API 未初始化'
                return result
            
            self._rate_limit()
            
            if hasattr(self._api, 'trade_cal'):
                df = self._api.trade_cal(
                    exchange='SSE',
                    start_date=datetime.now().strftime('%Y%m%d'),
                    end_date=datetime.now().strftime('%Y%m%d')
                )
                result['api_accessible'] = True
                result['sample_data_count'] = len(df) if df is not None else 0
            else:
                result['api_accessible'] = True
                
        except ImportError as e:
            result['error'] = f'Tushare 模块未安装: {e}'
        except Exception as e:
            result['error'] = f'API 连接失败: {e}'
        
        return result
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def get_stock_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = 'qfq'
    ) -> List[Dict[str, Any]]:
        """
        获取股票日线行情数据
        
        Args:
            symbol: 股票代码（如 000001）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            adjust: 复权类型（qfq-前复权, hfq-后复权, None-不复权）
            
        Returns:
            行情数据列表
        """
        if not self.is_available:
            raise RuntimeError("Tushare API 不可用")
        
        self._rate_limit()
        
        try:
            if hasattr(self._api, 'daily'):
                df = self._api.daily(
                    ts_code=f"{symbol}.SZ" if symbol.startswith('0') or symbol.startswith('3') else f"{symbol}.SH",
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is None or df.empty:
                    return []
                
                df = df.sort_values('trade_date')
                
                records = df.to_dict('records')
                
                result = []
                for record in records:
                    result.append({
                        'symbol': record.get('ts_code', '').split('.')[0],
                        'trade_date': record.get('trade_date', ''),
                        'open': float(record.get('open', 0)),
                        'high': float(record.get('high', 0)),
                        'low': float(record.get('low', 0)),
                        'close': float(record.get('close', 0)),
                        'volume': float(record.get('vol', 0)),
                        'amount': float(record.get('amount', 0)),
                        'turnover_rate': float(record.get('turnover_rate', 0)) if 'turnover_rate' in record else None,
                        'source': 'tushare'
                    })
                
                logger.info(f"✅ Tushare 获取 {symbol} 数据成功: {len(result)} 条记录")
                return result
            else:
                logger.warning("Tushare API 不支持 daily 方法")
                return []
                
        except Exception as e:
            logger.error(f"❌ Tushare 获取 {symbol} 日线数据失败: {e}")
            raise
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def get_stock_list(self) -> List[Dict[str, Any]]:
        """
        获取股票列表
        
        Returns:
            股票列表
        """
        if not self.is_available:
            raise RuntimeError("Tushare API 不可用")
        
        self._rate_limit()
        
        try:
            if hasattr(self._api, 'stock_basic'):
                df = self._api.stock_basic(exchange='', list_status='L')
                
                if df is None or df.empty:
                    return []
                
                records = df.to_dict('records')
                
                result = []
                for record in records:
                    result.append({
                        'symbol': record.get('ts_code', '').split('.')[0],
                        'name': record.get('name', ''),
                        'exchange': record.get('exchange', ''),
                        'list_date': record.get('list_date', ''),
                        'source': 'tushare'
                    })
                
                logger.info(f"✅ Tushare 获取股票列表成功: {len(result)} 只股票")
                return result
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ Tushare 获取股票列表失败: {e}")
            raise
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def get_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """
        获取指数日线数据
        
        Args:
            index_code: 指数代码（如 000001.SH 为上证指数）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            
        Returns:
            指数数据列表
        """
        if not self.is_available:
            raise RuntimeError("Tushare API 不可用")
        
        self._rate_limit()
        
        try:
            if hasattr(self._api, 'index_daily'):
                df = self._api.index_daily(
                    ts_code=index_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is None or df.empty:
                    return []
                
                df = df.sort_values('trade_date')
                
                records = df.to_dict('records')
                
                result = []
                for record in records:
                    result.append({
                        'index_code': record.get('ts_code', ''),
                        'trade_date': record.get('trade_date', ''),
                        'open': float(record.get('open', 0)),
                        'high': float(record.get('high', 0)),
                        'low': float(record.get('low', 0)),
                        'close': float(record.get('close', 0)),
                        'volume': float(record.get('vol', 0)),
                        'amount': float(record.get('amount', 0)),
                        'source': 'tushare'
                    })
                
                return result
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ Tushare 获取指数数据失败: {e}")
            raise
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, exceptions=(Exception,))
    def get_financial_indicator(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """
        获取财务指标数据
        
        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            财务指标数据列表
        """
        if not self.is_available:
            raise RuntimeError("Tushare API 不可用")
        
        self._rate_limit()
        
        try:
            if hasattr(self._api, 'fina_indicator'):
                ts_code = f"{symbol}.SZ" if symbol.startswith('0') or symbol.startswith('3') else f"{symbol}.SH"
                df = self._api.fina_indicator(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is None or df.empty:
                    return []
                
                records = df.to_dict('records')
                
                result = []
                for record in records:
                    result.append({
                        'symbol': symbol,
                        'ann_date': record.get('ann_date', ''),
                        'end_date': record.get('end_date', ''),
                        'roe': float(record.get('roe', 0)) if record.get('roe') else None,
                        'roa': float(record.get('roa', 0)) if record.get('roa') else None,
                        'netprofit_margin': float(record.get('netprofit_margin', 0)) if record.get('netprofit_margin') else None,
                        'grossprofit_margin': float(record.get('grossprofit_margin', 0)) if record.get('grossprofit_margin') else None,
                        'debt_to_assets': float(record.get('debt_to_assets', 0)) if record.get('debt_to_assets') else None,
                        'current_ratio': float(record.get('current_ratio', 0)) if record.get('current_ratio') else None,
                        'source': 'tushare'
                    })
                
                return result
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ Tushare 获取财务指标失败: {e}")
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
            股票数据字典
        """
        results = {}
        total = len(symbols)
        
        logger.info(f"📊 开始批量采集 {total} 只股票数据 (Tushare)...")
        
        for i, symbol in enumerate(symbols, 1):
            try:
                data = self.get_stock_daily(symbol, start_date, end_date)
                results[symbol] = data
                if progress_callback:
                    progress_callback(i, total, symbol, True, None)
            except Exception as e:
                results[symbol] = []
                if progress_callback:
                    progress_callback(i, total, symbol, False, str(e))
                logger.warning(f"⚠️ 采集 {symbol} 失败: {e}")
        
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"✅ 批量采集完成: {success_count}/{total} 成功")
        
        return results


_collector_instance = None
_collector_lock = threading.RLock()


def get_tushare_collector(
    token: Optional[str] = None,
    timeout: int = 30,
    max_retries: int = 3
) -> TushareCollector:
    """
    获取 Tushare 采集器实例（单例模式）
    
    Args:
        token: Tushare API Token
        timeout: 请求超时时间
        max_retries: 最大重试次数
        
    Returns:
        TushareCollector 实例
    """
    global _collector_instance
    if _collector_instance is None:
        with _collector_lock:
            if _collector_instance is None:
                _collector_instance = TushareCollector(
                    token=token,
                    timeout=timeout,
                    max_retries=max_retries
                )
    return _collector_instance

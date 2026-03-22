"""
统一数据源管理器

提供多数据源的统一管理，支持 AKShare 和 Tushare 的自动切换。
当主数据源失败时自动切换到备用数据源。
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """数据源类型枚举"""
    AKSHARE = 'akshare'
    TUSHARE = 'tushare'


@dataclass
class DataSourceStatus:
    """数据源状态"""
    name: str
    available: bool
    last_success: Optional[float]
    last_failure: Optional[float]
    success_count: int
    failure_count: int
    avg_response_time: float
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


class UnifiedDataSourceManager:
    """
    统一数据源管理器
    
    管理多个数据源，支持自动切换和负载均衡。
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
        primary_source: DataSourceType = DataSourceType.AKSHARE,
        fallback_enabled: bool = True,
        health_check_interval: int = 300
    ):
        """
        初始化统一数据源管理器
        
        Args:
            primary_source: 主数据源
            fallback_enabled: 是否启用备用数据源
            health_check_interval: 健康检查间隔（秒）
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.primary_source = primary_source
        self.fallback_enabled = fallback_enabled
        self.health_check_interval = health_check_interval
        
        self._status_lock = threading.RLock()
        self._source_status: Dict[str, DataSourceStatus] = {}
        
        self._akshare_collector = None
        self._tushare_collector = None
        
        self._initialize_collectors()
        self._initialized = True
        
        logger.info(f"统一数据源管理器初始化完成，主数据源: {primary_source.value}")
    
    def _initialize_collectors(self):
        """初始化数据源采集器"""
        try:
            from src.gateway.web.enhanced_akshare_collector import get_enhanced_collector
            self._akshare_collector = get_enhanced_collector()
            self._source_status[DataSourceType.AKSHARE.value] = DataSourceStatus(
                name=DataSourceType.AKSHARE.value,
                available=True,
                last_success=None,
                last_failure=None,
                success_count=0,
                failure_count=0,
                avg_response_time=0.0
            )
            logger.info("✅ AKShare 采集器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ AKShare 采集器初始化失败: {e}")
        
        try:
            from src.gateway.web.tushare_collector import get_tushare_collector
            self._tushare_collector = get_tushare_collector()
            self._source_status[DataSourceType.TUSHARE.value] = DataSourceStatus(
                name=DataSourceType.TUSHARE.value,
                available=self._tushare_collector.is_available,
                last_success=None,
                last_failure=None,
                success_count=0,
                failure_count=0,
                avg_response_time=0.0
            )
            logger.info("✅ Tushare 采集器初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ Tushare 采集器初始化失败: {e}")
    
    def _update_status(
        self,
        source: DataSourceType,
        success: bool,
        response_time: float = 0.0
    ):
        """更新数据源状态"""
        with self._status_lock:
            status = self._source_status.get(source.value)
            if status:
                current_time = time.time()
                if success:
                    status.success_count += 1
                    status.last_success = current_time
                else:
                    status.failure_count += 1
                    status.last_failure = current_time
                
                if response_time > 0:
                    total = status.success_count + status.failure_count
                    status.avg_response_time = (
                        (status.avg_response_time * (total - 1) + response_time) / total
                    )
    
    def _get_available_source(self) -> Optional[DataSourceType]:
        """
        获取可用的数据源
        
        Returns:
            可用的数据源类型
        """
        primary = self.primary_source
        primary_status = self._source_status.get(primary.value)
        
        if primary_status and primary_status.available:
            if primary_status.success_rate >= 0.5 or primary_status.success_count == 0:
                return primary
        
        if self.fallback_enabled:
            for source_type in DataSourceType:
                if source_type != primary:
                    status = self._source_status.get(source_type.value)
                    if status and status.available:
                        return source_type
        
        return self.primary_source
    
    def get_stock_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = 'qfq'
    ) -> List[Dict[str, Any]]:
        """
        获取股票日线数据（自动选择数据源）
        
        Args:
            symbol: 股票代码
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            adjust: 复权类型
            
        Returns:
            行情数据列表
        """
        source = self._get_available_source()
        
        if source == DataSourceType.AKSHARE and self._akshare_collector:
            try:
                start_time = time.time()
                data = self._akshare_collector.get_stock_history(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    period='daily',
                    adjust=adjust
                )
                response_time = time.time() - start_time
                self._update_status(source, True, response_time)
                return data
            except Exception as e:
                self._update_status(source, False)
                logger.warning(f"⚠️ AKShare 获取数据失败: {e}，尝试切换数据源...")
                
                if self.fallback_enabled and self._tushare_collector:
                    try:
                        start_time = time.time()
                        data = self._tushare_collector.get_stock_daily(
                            symbol=symbol,
                            start_date=start_date,
                            end_date=end_date,
                            adjust=adjust
                        )
                        response_time = time.time() - start_time
                        self._update_status(DataSourceType.TUSHARE, True, response_time)
                        logger.info(f"✅ 切换到 Tushare 成功获取数据")
                        return data
                    except Exception as e2:
                        self._update_status(DataSourceType.TUSHARE, False)
                        logger.error(f"❌ Tushare 也失败: {e2}")
                        raise
        
        elif source == DataSourceType.TUSHARE and self._tushare_collector:
            try:
                start_time = time.time()
                data = self._tushare_collector.get_stock_daily(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )
                response_time = time.time() - start_time
                self._update_status(source, True, response_time)
                return data
            except Exception as e:
                self._update_status(source, False)
                logger.warning(f"⚠️ Tushare 获取数据失败: {e}")
                raise
        
        raise RuntimeError("没有可用的数据源")
    
    def get_stock_list(self) -> List[Dict[str, Any]]:
        """
        获取股票列表
        
        Returns:
            股票列表
        """
        source = self._get_available_source()
        
        if source == DataSourceType.AKSHARE and self._akshare_collector:
            try:
                start_time = time.time()
                data = self._akshare_collector.get_stock_list()
                response_time = time.time() - start_time
                self._update_status(source, True, response_time)
                return data
            except Exception as e:
                self._update_status(source, False)
                logger.warning(f"⚠️ AKShare 获取股票列表失败: {e}")
        
        if self._tushare_collector:
            try:
                start_time = time.time()
                data = self._tushare_collector.get_stock_list()
                response_time = time.time() - start_time
                self._update_status(DataSourceType.TUSHARE, True, response_time)
                return data
            except Exception as e:
                self._update_status(DataSourceType.TUSHARE, False)
                logger.warning(f"⚠️ Tushare 获取股票列表失败: {e}")
        
        raise RuntimeError("无法获取股票列表")
    
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
            progress_callback: 进度回调
            
        Returns:
            股票数据字典
        """
        results = {}
        total = len(symbols)
        
        logger.info(f"📊 开始批量采集 {total} 只股票数据...")
        
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
        
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"✅ 批量采集完成: {success_count}/{total} 成功")
        
        return results
    
    def get_source_status(self) -> Dict[str, Any]:
        """
        获取所有数据源状态
        
        Returns:
            数据源状态字典
        """
        with self._status_lock:
            result = {}
            for name, status in self._source_status.items():
                result[name] = {
                    'available': status.available,
                    'success_rate': round(status.success_rate * 100, 2),
                    'success_count': status.success_count,
                    'failure_count': status.failure_count,
                    'avg_response_time': round(status.avg_response_time, 2),
                    'last_success': datetime.fromtimestamp(status.last_success).isoformat() if status.last_success else None,
                    'last_failure': datetime.fromtimestamp(status.last_failure).isoformat() if status.last_failure else None
                }
            return result
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态
        
        Returns:
            健康状态字典
        """
        status = self.get_source_status()
        
        primary_status = status.get(self.primary_source.value, {})
        primary_success_rate = primary_status.get('success_rate', 0)
        
        health = 'HEALTHY'
        if primary_success_rate < 50:
            health = 'CRITICAL'
        elif primary_success_rate < 80:
            health = 'DEGRADED'
        
        return {
            'status': health,
            'primary_source': self.primary_source.value,
            'fallback_enabled': self.fallback_enabled,
            'sources': status
        }
    
    def switch_primary_source(self, source: DataSourceType):
        """
        切换主数据源
        
        Args:
            source: 新的主数据源
        """
        self.primary_source = source
        logger.info(f"主数据源已切换为: {source.value}")


_manager_instance = None
_manager_lock = threading.RLock()


def get_unified_data_source_manager() -> UnifiedDataSourceManager:
    """
    获取统一数据源管理器实例（单例模式）
    
    Returns:
        UnifiedDataSourceManager 实例
    """
    global _manager_instance
    if _manager_instance is None:
        with _manager_lock:
            if _manager_instance is None:
                _manager_instance = UnifiedDataSourceManager()
    return _manager_instance

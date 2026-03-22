"""
股票数据采集成功率监控模块

提供股票数据采集的成功率统计、性能监控和告警功能。
支持PostgreSQL持久化存储监控数据。
"""

import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class CollectionRecord:
    """采集记录数据类"""
    symbol: str
    source: str
    start_time: float
    end_time: float
    success: bool
    records_count: int
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class CollectionStats:
    """采集统计数据类"""
    total_attempts: int = 0
    successful_attempts: int = 0
    failed_attempts: int = 0
    total_records: int = 0
    total_time: float = 0.0
    error_distribution: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts
    
    @property
    def avg_time(self) -> float:
        """计算平均耗时"""
        if self.total_attempts == 0:
            return 0.0
        return self.total_time / self.total_attempts


class StockCollectionMonitor:
    """
    股票数据采集成功率监控器
    
    监控股票数据采集的成功率、性能指标，并支持持久化存储。
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
        persistence_enabled: bool = True,
        alert_threshold: float = 0.90,
        alert_consecutive_failures: int = 3
    ):
        """
        初始化监控器
        
        Args:
            persistence_enabled: 是否启用持久化
            alert_threshold: 成功率告警阈值
            alert_consecutive_failures: 连续失败次数告警阈值
        """
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.persistence_enabled = persistence_enabled
        self.alert_threshold = alert_threshold
        self.alert_consecutive_failures = alert_consecutive_failures
        
        self._stats_lock = threading.RLock()
        self._stats: Dict[str, CollectionStats] = defaultdict(CollectionStats)
        self._records: List[CollectionRecord] = []
        self._consecutive_failures: Dict[str, int] = defaultdict(int)
        self._alerts: List[Dict[str, Any]] = []
        
        self._db_available = False
        self._check_database_connection()
        
        self._initialized = True
        logger.info("股票数据采集监控器初始化完成")
    
    def _check_database_connection(self):
        """检查数据库连接"""
        if not self.persistence_enabled:
            return
            
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', '5432')),
                database=os.getenv('POSTGRES_DB', 'rqa2025_prod'),
                user=os.getenv('POSTGRES_USER', 'rqa2025_admin'),
                password=os.getenv('POSTGRES_PASSWORD', ''),
                connect_timeout=5
            )
            conn.close()
            self._db_available = True
            self._ensure_monitoring_table()
            logger.info("数据库连接检查成功，持久化已启用")
        except Exception as e:
            self._db_available = False
            logger.warning(f"数据库连接失败，使用内存存储: {e}")
    
    def _ensure_monitoring_table(self):
        """确保监控表存在"""
        if not self._db_available:
            return
            
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', '5432')),
                database=os.getenv('POSTGRES_DB', 'rqa2025_prod'),
                user=os.getenv('POSTGRES_USER', 'rqa2025_admin'),
                password=os.getenv('POSTGRES_PASSWORD', '')
            )
            cur = conn.cursor()
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stock_collection_monitoring (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    source VARCHAR(50) NOT NULL,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP NOT NULL,
                    success BOOLEAN NOT NULL,
                    records_count INTEGER DEFAULT 0,
                    error_type VARCHAR(100),
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    duration_ms FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_collection_symbol 
                ON stock_collection_monitoring(symbol);
            """)
            
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_collection_time 
                ON stock_collection_monitoring(start_time);
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"创建监控表失败: {e}")
    
    def record_collection(
        self,
        symbol: str,
        source: str,
        start_time: float,
        end_time: float,
        success: bool,
        records_count: int = 0,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        retry_count: int = 0
    ) -> None:
        """
        记录一次采集结果
        
        Args:
            symbol: 股票代码
            source: 数据源
            start_time: 开始时间戳
            end_time: 结束时间戳
            success: 是否成功
            records_count: 采集记录数
            error_type: 错误类型
            error_message: 错误消息
            retry_count: 重试次数
        """
        record = CollectionRecord(
            symbol=symbol,
            source=source,
            start_time=start_time,
            end_time=end_time,
            success=success,
            records_count=records_count,
            error_type=error_type,
            error_message=error_message,
            retry_count=retry_count
        )
        
        with self._stats_lock:
            stats = self._stats[source]
            stats.total_attempts += 1
            stats.total_time += (end_time - start_time)
            
            if success:
                stats.successful_attempts += 1
                stats.total_records += records_count
                self._consecutive_failures[source] = 0
            else:
                stats.failed_attempts += 1
                if error_type:
                    stats.error_distribution[error_type] = \
                        stats.error_distribution.get(error_type, 0) + 1
                
                self._consecutive_failures[source] += 1
                
                if self._consecutive_failures[source] >= self.alert_consecutive_failures:
                    self._trigger_alert(
                        alert_type='consecutive_failures',
                        source=source,
                        message=f"数据源 {source} 连续失败 {self._consecutive_failures[source]} 次",
                        severity='HIGH'
                    )
            
            self._records.append(record)
            
            if len(self._records) > 10000:
                self._records = self._records[-5000:]
        
        if self._db_available:
            self._persist_record(record)
        
        if success:
            logger.debug(f"采集成功: {symbol} @ {source}, {records_count} 条记录")
        else:
            logger.warning(f"采集失败: {symbol} @ {source}, 错误: {error_type}")
    
    def _persist_record(self, record: CollectionRecord) -> None:
        """持久化记录到数据库"""
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', '5432')),
                database=os.getenv('POSTGRES_DB', 'rqa2025_prod'),
                user=os.getenv('POSTGRES_USER', 'rqa2025_admin'),
                password=os.getenv('POSTGRES_PASSWORD', '')
            )
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO stock_collection_monitoring 
                (symbol, source, start_time, end_time, success, records_count, 
                 error_type, error_message, retry_count, duration_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                record.symbol,
                record.source,
                datetime.fromtimestamp(record.start_time),
                datetime.fromtimestamp(record.end_time),
                record.success,
                record.records_count,
                record.error_type,
                record.error_message,
                record.retry_count,
                (record.end_time - record.start_time) * 1000
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"持久化监控记录失败: {e}")
    
    def _trigger_alert(
        self,
        alert_type: str,
        source: str,
        message: str,
        severity: str
    ) -> None:
        """触发告警"""
        alert = {
            'alert_type': alert_type,
            'source': source,
            'message': message,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        
        self._alerts.append(alert)
        logger.warning(f"告警 [{severity}]: {message}")
    
    def get_stats(self, source: Optional[str] = None) -> Dict[str, Any]:
        """
        获取统计数据
        
        Args:
            source: 数据源，为None时返回所有数据源统计
            
        Returns:
            统计数据字典
        """
        with self._stats_lock:
            if source:
                stats = self._stats.get(source, CollectionStats())
                return {
                    'source': source,
                    'total_attempts': stats.total_attempts,
                    'successful_attempts': stats.successful_attempts,
                    'failed_attempts': stats.failed_attempts,
                    'success_rate': round(stats.success_rate * 100, 2),
                    'total_records': stats.total_records,
                    'avg_time': round(stats.avg_time, 2),
                    'error_distribution': dict(stats.error_distribution)
                }
            else:
                result = {}
                for src, stats in self._stats.items():
                    result[src] = {
                        'total_attempts': stats.total_attempts,
                        'successful_attempts': stats.successful_attempts,
                        'failed_attempts': stats.failed_attempts,
                        'success_rate': round(stats.success_rate * 100, 2),
                        'total_records': stats.total_records,
                        'avg_time': round(stats.avg_time, 2),
                        'error_distribution': dict(stats.error_distribution)
                    }
                return result
    
    def get_recent_records(
        self,
        source: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取最近的采集记录
        
        Args:
            source: 数据源
            limit: 返回记录数量限制
            
        Returns:
            记录列表
        """
        with self._stats_lock:
            records = self._records[-limit:] if source is None else [
                r for r in self._records if r.source == source
            ][-limit:]
            
            return [asdict(r) for r in records]
    
    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取告警列表
        
        Args:
            limit: 返回告警数量限制
            
        Returns:
            告警列表
        """
        return self._alerts[-limit:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        获取健康状态
        
        Returns:
            健康状态字典
        """
        with self._stats_lock:
            overall_stats = CollectionStats()
            for stats in self._stats.values():
                overall_stats.total_attempts += stats.total_attempts
                overall_stats.successful_attempts += stats.successful_attempts
                overall_stats.failed_attempts += stats.failed_attempts
                overall_stats.total_records += stats.total_records
                overall_stats.total_time += stats.total_time
            
            success_rate = overall_stats.success_rate
            health_status = 'HEALTHY'
            
            if success_rate < self.alert_threshold:
                health_status = 'DEGRADED'
            if success_rate < 0.50:
                health_status = 'CRITICAL'
            
            return {
                'status': health_status,
                'success_rate': round(success_rate * 100, 2),
                'total_attempts': overall_stats.total_attempts,
                'successful_attempts': overall_stats.successful_attempts,
                'failed_attempts': overall_stats.failed_attempts,
                'total_records': overall_stats.total_records,
                'avg_time': round(overall_stats.avg_time, 2),
                'active_alerts': len([a for a in self._alerts if a['severity'] == 'HIGH']),
                'persistence_enabled': self.persistence_enabled,
                'db_available': self._db_available
            }
    
    def get_hourly_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取按小时统计的数据
        
        Args:
            hours: 统计小时数
            
        Returns:
            按小时统计的数据
        """
        if not self._db_available:
            return {'error': '数据库不可用'}
        
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('POSTGRES_PORT', '5432')),
                database=os.getenv('POSTGRES_DB', 'rqa2025_prod'),
                user=os.getenv('POSTGRES_USER', 'rqa2025_admin'),
                password=os.getenv('POSTGRES_PASSWORD', '')
            )
            cur = conn.cursor()
            
            cur.execute("""
                SELECT 
                    DATE_TRUNC('hour', start_time) as hour,
                    COUNT(*) as total,
                    SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count,
                    AVG(duration_ms) as avg_duration
                FROM stock_collection_monitoring
                WHERE start_time >= NOW() - INTERVAL '%s hours'
                GROUP BY DATE_TRUNC('hour', start_time)
                ORDER BY hour DESC
            """, (hours,))
            
            hourly_data = []
            for row in cur.fetchall():
                hourly_data.append({
                    'hour': row[0].isoformat() if row[0] else None,
                    'total': row[1],
                    'success_count': row[2],
                    'success_rate': round(row[2] / row[1] * 100, 2) if row[1] > 0 else 0,
                    'avg_duration_ms': round(row[3], 2) if row[3] else 0
                })
            
            cur.close()
            conn.close()
            
            return {'hours': hours, 'data': hourly_data}
            
        except Exception as e:
            logger.error(f"获取小时统计失败: {e}")
            return {'error': str(e)}
    
    def reset_stats(self, source: Optional[str] = None) -> None:
        """
        重置统计数据
        
        Args:
            source: 数据源，为None时重置所有
        """
        with self._stats_lock:
            if source:
                self._stats[source] = CollectionStats()
                self._consecutive_failures[source] = 0
            else:
                self._stats.clear()
                self._consecutive_failures.clear()
                self._alerts.clear()
                self._records.clear()
        
        logger.info(f"统计数据已重置: {source or '全部'}")


def get_stock_collection_monitor() -> StockCollectionMonitor:
    """获取股票数据采集监控器实例"""
    return StockCollectionMonitor()

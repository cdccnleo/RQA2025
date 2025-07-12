"""存储监控模块"""
import time
from typing import Dict, Any, Optional


class StorageMonitor:
    """存储监控器"""
    
    def __init__(self):
        self._write_count = 0
        self._error_count = 0
        self._total_size = 0
        self._start_time = time.time()
    
    def record_write(self, symbol: Optional[str] = None, size: int = 0, status: bool = True):
        """记录写入操作"""
        self._write_count += 1
        if status:
            self._total_size += size
    
    def record_error(self, symbol: Optional[str] = None):
        """记录错误"""
        self._error_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """获取监控统计"""
        uptime = time.time() - self._start_time
        return {
            'write_count': self._write_count,
            'error_count': self._error_count,
            'total_size': self._total_size,
            'uptime': uptime,
            'write_rate': self._write_count / uptime if uptime > 0 else 0,
            'error_rate': self._error_count / uptime if uptime > 0 else 0
        }
    
    def reset(self):
        """重置统计"""
        self._write_count = 0
        self._error_count = 0
        self._total_size = 0
        self._start_time = time.time() 
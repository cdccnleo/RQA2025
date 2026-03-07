
import time

from typing import Dict, Any, Optional
"""存储监控模块"""


class StorageMonitorPlugin:
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
        try:
            uptime = max(time.time() - self._start_time, 0.0)
            if uptime < 1e-2:
                uptime = 0.0

            return {
                "write_count": self._write_count,
                "error_count": self._error_count,
                "total_size": self._total_size,
                "uptime": uptime,
                "write_rate": self._write_count / uptime if uptime > 0 else 0,
                "error_rate": self._error_count / uptime if uptime > 0 else 0,
            }
        except Exception as e:
            # 如果计算过程中出现异常，返回安全的值
            return {
                "write_count": self._write_count,
                "error_count": self._error_count,
                "total_size": self._total_size,
                "uptime": 0,
                "write_rate": 0,
                "error_rate": 0,
                "error": str(e),
            }

    def reset(self):
        """重置统计"""
        self._write_count = 0
        self._error_count = 0
        self._total_size = 0
        self._start_time = time.time()

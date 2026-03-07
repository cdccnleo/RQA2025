"""
健康检查监控管理器

负责健康检查的持续监控、状态跟踪、监控循环等功能。
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)

@dataclass
class MonitorStatus:
    """监控状态"""
    is_active: bool = False
    start_time: Optional[datetime] = None
    last_check_time: Optional[datetime] = None
    check_count: int = 0
    error_count: int = 0


class HealthCheckMonitor:
    """
    健康检查监控管理器
    
    职责：
    - 管理监控循环
    - 跟踪监控状态
    - 处理监控异常
    - 提供监控统计信息
    """
    
    def __init__(self, monitoring_interval: float = 60.0):
        """
        初始化监控管理器
        
        Args:
            monitoring_interval: 监控间隔(秒)
        """
        self._monitoring_interval = monitoring_interval
        self._monitor_task: Optional[asyncio.Task] = None
        self._status = MonitorStatus()
        self._check_callback: Optional[Callable] = None
        try:
            self._stop_event = asyncio.Event()
        except RuntimeError:
            # 如果没有事件循环，使用Mock
            self._stop_event = None
        
    async def start_monitoring(self, check_callback: Callable) -> bool:
        """
        启动监控
        
        Args:
            check_callback: 健康检查回调函数
            
        Returns:
            是否成功启动
        """
        if self._status.is_active:
            logger.warning("监控已在运行中")
            return False
            
        try:
            self._check_callback = check_callback
            self._status.is_active = True
            self._status.start_time = datetime.now()
            self._stop_event.clear()
            
            self._monitor_task = asyncio.create_task(self._monitoring_loop())
            logger.info(f"健康监控已启动，间隔: {self._monitoring_interval}s")
            return True
            
        except Exception as e:
            logger.error(f"启动监控失败: {e}")
            self._status.is_active = False
            return False
    
    async def stop_monitoring(self) -> bool:
        """
        停止监控
        
        Returns:
            是否成功停止
        """
        if not self._status.is_active:
            logger.warning("监控未在运行")
            return False
            
        try:
            self._status.is_active = False
            self._stop_event.set()
            
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
                    
            logger.info("健康监控已停止")
            return True
            
        except Exception as e:
            logger.error(f"停止监控失败: {e}")
            return False
    
    async def restart_monitoring(self, check_callback: Optional[Callable] = None) -> bool:
        """
        重启监控
        
        Args:
            check_callback: 新的检查回调函数，None则使用现有的
            
        Returns:
            是否成功重启
        """
        logger.info("正在重启健康监控...")
        
        # 先停止现有监控
        await self.stop_monitoring()
        
        # 等待一小段时间确保清理完成
        await asyncio.sleep(0.1)
        
        # 启动新监控
        callback = check_callback or self._check_callback
        if callback is None:
            logger.error("没有可用的检查回调函数")
            return False
            
        return await self.start_monitoring(callback)
    
    async def _monitoring_loop(self) -> None:
        """监控循环"""
        logger.info("监控循环已启动")
        
        while self._status.is_active and not self._stop_event.is_set():
            try:
                # 执行健康检查回调
                if self._check_callback:
                    await self._check_callback()
                    
                self._status.check_count += 1
                self._status.last_check_time = datetime.now()
                
                logger.debug(f"监控检查完成，第 {self._status.check_count} 次")
                
            except Exception as e:
                self._status.error_count += 1
                logger.error(f"监控检查失败: {e}")
                
            # 等待下次检查
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), 
                    timeout=self._monitoring_interval
                )
                # 如果stop_event被设置，退出循环
                break
            except asyncio.TimeoutError:
                # 超时是正常的，继续下一次检查
                continue
                
        logger.info("监控循环已结束")
    
    def get_monitor_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        uptime = None
        if self._status.start_time:
            uptime = (datetime.now() - self._status.start_time).total_seconds()
            
        return {
            "is_active": self._status.is_active,
            "start_time": self._status.start_time.isoformat() if self._status.start_time else None,
            "last_check_time": self._status.last_check_time.isoformat() if self._status.last_check_time else None,
            "check_count": self._status.check_count,
            "error_count": self._status.error_count,
            "uptime_seconds": uptime,
            "monitoring_interval": self._monitoring_interval,
            "success_rate": self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """计算成功率"""
        if self._status.check_count == 0:
            return 0.0
            
        success_count = self._status.check_count - self._status.error_count
        return round((success_count / self._status.check_count) * 100, 2)
    
    def is_monitoring(self) -> bool:
        """检查是否正在监控"""
        return self._status.is_active
    
    def get_monitoring_interval(self) -> float:
        """获取监控间隔"""
        return self._monitoring_interval
    
    def set_monitoring_interval(self, interval: float) -> None:
        """
        设置监控间隔
        
        Args:
            interval: 新的监控间隔(秒)
        """
        if interval <= 0:
            raise ValueError("监控间隔必须大于0")
            
        self._monitoring_interval = interval
        logger.info(f"监控间隔已更新为: {interval}s")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定时清理任务调度器

负责调度和执行数据清理、维护和优化任务
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from .data_cleanup import get_data_cleanup_service

logger = logging.getLogger(__name__)


class MaintenanceScheduler:
    """
    维护任务调度器
    
    负责定时执行数据清理和维护任务
    """
    
    def __init__(self):
        """
        初始化维护任务调度器
        """
        self.cleanup_service = get_data_cleanup_service()
        self.scheduler_task = None
        self.is_running = False
        self.task_history = []
        
        # 任务配置
        self.tasks = {
            "daily_validation": {
                "interval": 24 * 60 * 60,  # 24小时
                "next_run": self._calculate_next_run(24 * 60 * 60),
                "enabled": True,
                "description": "每日数据验证"
            },
            "weekly_cleanup": {
                "interval": 7 * 24 * 60 * 60,  # 7天
                "next_run": self._calculate_next_run(7 * 24 * 60 * 60),
                "enabled": True,
                "description": "每周数据清理"
            },
            "monthly_optimization": {
                "interval": 30 * 24 * 60 * 60,  # 30天
                "next_run": self._calculate_next_run(30 * 24 * 60 * 60),
                "enabled": True,
                "description": "每月表空间优化"
            }
        }
        
        logger.info("✅ 维护任务调度器初始化完成")
    
    def _calculate_next_run(self, interval: int) -> float:
        """
        计算下次运行时间
        
        Args:
            interval: 时间间隔（秒）
            
        Returns:
            下次运行时间戳
        """
        return time.time() + interval
    
    async def start(self):
        """
        启动调度器
        """
        if self.is_running:
            logger.warning("⚠️  调度器已在运行")
            return
        
        self.is_running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("🚀 维护任务调度器已启动")
    
    async def stop(self):
        """
        停止调度器
        """
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 维护任务调度器已停止")
    
    async def _scheduler_loop(self):
        """
        调度器主循环
        """
        logger.info("🔄 调度器主循环启动")
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 检查并执行任务
                for task_name, task_config in self.tasks.items():
                    if task_config["enabled"] and current_time >= task_config["next_run"]:
                        await self._execute_task(task_name)
                        task_config["next_run"] = self._calculate_next_run(task_config["interval"])
                
                # 休眠一段时间
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ 调度器循环异常: {e}")
                await asyncio.sleep(60)
        
        logger.info("🔄 调度器主循环结束")
    
    async def _execute_task(self, task_name: str):
        """
        执行维护任务
        
        Args:
            task_name: 任务名称
        """
        start_time = time.time()
        task_info = {
            "task_name": task_name,
            "start_time": start_time,
            "end_time": None,
            "success": False,
            "result": None,
            "error": None
        }
        
        logger.info(f"🚀 开始执行任务: {task_name}")
        
        try:
            if task_name == "daily_validation":
                result = await self._run_daily_validation()
            elif task_name == "weekly_cleanup":
                result = await self._run_weekly_cleanup()
            elif task_name == "monthly_optimization":
                result = await self._run_monthly_optimization()
            else:
                logger.warning(f"⚠️  未知任务: {task_name}")
                return
            
            task_info["success"] = True
            task_info["result"] = result
            logger.info(f"✅ 任务执行成功: {task_name}")
            
        except Exception as e:
            task_info["error"] = str(e)
            logger.error(f"❌ 任务执行失败: {task_name}, 错误: {e}")
        finally:
            task_info["end_time"] = time.time()
            self.task_history.append(task_info)
            
            # 限制历史记录长度
            if len(self.task_history) > 100:
                self.task_history = self.task_history[-100:]
            
            duration = task_info["end_time"] - task_info["start_time"]
            logger.info(f"⏱️  任务执行完成: {task_name}, 耗时: {duration:.2f}秒")
    
    async def _run_daily_validation(self) -> Dict[str, Any]:
        """
        执行每日数据验证
        
        Returns:
            验证结果
        """
        logger.info("🔍 执行每日数据验证")
        
        # 验证数据完整性
        validation_result = self.cleanup_service.validate_data_integrity()
        
        # 生成验证报告
        report = self.cleanup_service.generate_cleanup_report()
        
        return {
            "validation": validation_result,
            "report": report
        }
    
    async def _run_weekly_cleanup(self) -> Dict[str, Any]:
        """
        执行每周数据清理
        
        Returns:
            清理结果
        """
        logger.info("🧹 执行每周数据清理")
        
        # 清理无效数据
        cleanup_result = self.cleanup_service.clean_invalid_data(
            source_id="historical_collection_688702"
        )
        
        # 清理过期数据（保留365天）
        historical_cleanup = self.cleanup_service.clean_historical_data(days=365)
        
        return {
            "invalid_data_cleanup": cleanup_result,
            "historical_cleanup": historical_cleanup
        }
    
    async def _run_monthly_optimization(self) -> Dict[str, Any]:
        """
        执行每月表空间优化
        
        Returns:
            优化结果
        """
        logger.info("⚡ 执行每月表空间优化")
        
        # 优化表空间
        optimization_result = self.cleanup_service.optimize_table_space()
        
        # 生成优化报告
        report = self.cleanup_service.generate_cleanup_report()
        
        return {
            "optimization": optimization_result,
            "report": report
        }
    
    async def run_task_manually(self, task_name: str) -> Dict[str, Any]:
        """
        手动运行任务
        
        Args:
            task_name: 任务名称
            
        Returns:
            任务执行结果
        """
        if task_name not in self.tasks:
            return {
                "success": False,
                "error": f"未知任务: {task_name}"
            }
        
        await self._execute_task(task_name)
        
        # 返回最新的任务执行结果
        if self.task_history:
            for task_info in reversed(self.task_history):
                if task_info["task_name"] == task_name:
                    return {
                        "success": task_info["success"],
                        "result": task_info["result"],
                        "error": task_info["error"],
                        "duration": task_info["end_time"] - task_info["start_time"] if task_info["end_time"] else 0
                    }
        
        return {
            "success": False,
            "error": "任务执行结果未找到"
        }
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """
        获取调度器状态
        
        Returns:
            调度器状态
        """
        current_time = time.time()
        
        # 计算下次运行时间
        next_runs = {}
        for task_name, task_config in self.tasks.items():
            time_until_next = max(0, task_config["next_run"] - current_time)
            next_runs[task_name] = {
                "next_run": task_config["next_run"],
                "time_until_next": time_until_next,
                "time_until_next_human": self._format_time(time_until_next),
                "enabled": task_config["enabled"],
                "description": task_config["description"]
            }
        
        # 获取最近的任务历史
        recent_history = self.task_history[-10:] if self.task_history else []
        
        return {
            "is_running": self.is_running,
            "tasks": next_runs,
            "recent_history": recent_history,
            "total_tasks_executed": len(self.task_history)
        }
    
    def _format_time(self, seconds: float) -> str:
        """
        格式化时间
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时间字符串
        """
        if seconds < 60:
            return f"{seconds:.0f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.0f}分钟"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}小时"
        else:
            days = seconds / 86400
            return f"{days:.1f}天"
    
    def update_task_config(self, task_name: str, **config) -> bool:
        """
        更新任务配置
        
        Args:
            task_name: 任务名称
            **config: 配置参数
            
        Returns:
            是否更新成功
        """
        if task_name not in self.tasks:
            return False
        
        task_config = self.tasks[task_name]
        
        # 更新配置
        if "enabled" in config:
            task_config["enabled"] = config["enabled"]
        
        if "interval" in config:
            task_config["interval"] = config["interval"]
            task_config["next_run"] = self._calculate_next_run(config["interval"])
        
        logger.info(f"🔄 更新任务配置: {task_name}, 配置: {config}")
        return True


# 全局调度器实例
_maintenance_scheduler_instance = None


def get_maintenance_scheduler() -> MaintenanceScheduler:
    """
    获取维护任务调度器实例
    
    Returns:
        MaintenanceScheduler实例
    """
    global _maintenance_scheduler_instance
    if _maintenance_scheduler_instance is None:
        _maintenance_scheduler_instance = MaintenanceScheduler()
    return _maintenance_scheduler_instance

def reset_maintenance_scheduler():
    """
    重置维护任务调度器实例
    """
    global _maintenance_scheduler_instance
    _maintenance_scheduler_instance = None
    logger.info("🔄 维护任务调度器实例已重置")

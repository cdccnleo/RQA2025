#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定时清理任务模块

提供各层级数据的定时清理功能：
- 过期告警清理
- 历史风险数据归档
- 临时文件清理
- 缓存清理
"""

import os
import sys
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


class ScheduledCleanupTask:
    """
    定时清理任务
    
    支持多种清理任务的定时执行，包括：
    - 过期告警清理
    - 历史数据归档
    - 临时文件清理
    - 缓存清理
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化定时清理任务
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._stop_event = threading.Event()
        self._cleanup_thread = None
        self._tasks: Dict[str, Callable] = {}
        self._last_run: Dict[str, datetime] = {}
        
        # 默认配置
        self._alert_cleanup_hours = self.config.get('alert_cleanup_hours', 72)
        self._data_archive_days = self.config.get('data_archive_days', 90)
        self._temp_file_hours = self.config.get('temp_file_hours', 24)
        self._cleanup_interval = self.config.get('cleanup_interval', 3600)  # 1小时
        
        # 注册默认清理任务
        self._register_default_tasks()
        
        logger.info("定时清理任务模块初始化完成")
    
    def _register_default_tasks(self):
        """注册默认清理任务"""
        self.register_task('alert_cleanup', self._cleanup_expired_alerts)
        self.register_task('risk_data_archive', self._archive_historical_risk_data)
        self.register_task('temp_file_cleanup', self._cleanup_temp_files)
        self.register_task('cache_cleanup', self._cleanup_expired_cache)
    
    def register_task(self, task_name: str, task_func: Callable):
        """
        注册清理任务
        
        Args:
            task_name: 任务名称
            task_func: 任务函数
        """
        self._tasks[task_name] = task_func
        logger.info(f"注册清理任务: {task_name}")
    
    def start(self):
        """启动定时清理任务"""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            logger.warning("定时清理任务已在运行")
            return
        
        self._stop_event.clear()
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        logger.info("定时清理任务已启动")
    
    def stop(self):
        """停止定时清理任务"""
        logger.info("正在停止定时清理任务...")
        self._stop_event.set()
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=10)
        
        logger.info("定时清理任务已停止")
    
    def _cleanup_worker(self):
        """清理工作线程"""
        while not self._stop_event.is_set():
            try:
                self._run_all_tasks()
            except Exception as e:
                logger.error(f"清理任务执行异常: {e}")
            
            # 等待下一次执行
            if self._stop_event.wait(timeout=self._cleanup_interval):
                break
    
    def _run_all_tasks(self):
        """执行所有清理任务"""
        logger.info("开始执行定时清理任务...")
        
        for task_name, task_func in self._tasks.items():
            try:
                logger.info(f"执行清理任务: {task_name}")
                start_time = time.time()
                
                result = task_func()
                
                elapsed = time.time() - start_time
                logger.info(f"清理任务 {task_name} 完成，耗时 {elapsed:.2f}秒，结果: {result}")
                
                self._last_run[task_name] = datetime.now()
                
            except Exception as e:
                logger.error(f"清理任务 {task_name} 执行失败: {e}")
        
        logger.info("定时清理任务执行完成")
    
    def _cleanup_expired_alerts(self) -> Dict[str, int]:
        """
        清理过期告警
        
        Returns:
            清理统计
        """
        stats = {'cleaned': 0, 'failed': 0}
        
        try:
            from src.risk.persistence.risk_persistence import AlertPersistence
            
            persistence = AlertPersistence()
            
            # 获取活跃告警
            alerts = persistence.get_active_alerts(limit=10000)
            
            cutoff_time = datetime.now() - timedelta(hours=self._alert_cleanup_hours)
            
            for alert in alerts:
                if alert.created_at < cutoff_time:
                    try:
                        persistence.update_alert(alert.alert_id, {'status': 'expired'})
                        stats['cleaned'] += 1
                    except Exception as e:
                        logger.error(f"更新告警状态失败 {alert.alert_id}: {e}")
                        stats['failed'] += 1
            
            logger.info(f"清理过期告警: 清理={stats['cleaned']}, 失败={stats['failed']}")
            
        except Exception as e:
            logger.error(f"清理过期告警失败: {e}")
            stats['failed'] += 1
        
        return stats
    
    def _archive_historical_risk_data(self) -> Dict[str, int]:
        """
        归档历史风险数据
        
        Returns:
            归档统计
        """
        stats = {'archived_checks': 0, 'archived_metrics': 0, 'failed': 0}
        
        try:
            cutoff_date = datetime.now() - timedelta(days=self._data_archive_days)
            
            # 归档风险检查记录
            data_dir = Path("data/risk/checks")
            if data_dir.exists():
                for file_path in data_dir.glob("*.json"):
                    try:
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        created_at = datetime.fromisoformat(data.get('created_at', ''))
                        if created_at < cutoff_date and data.get('passed', False):
                            # 移动到归档目录
                            archive_dir = data_dir / "archive"
                            archive_dir.mkdir(exist_ok=True)
                            file_path.rename(archive_dir / file_path.name)
                            stats['archived_checks'] += 1
                    except Exception as e:
                        logger.error(f"归档风险检查失败 {file_path.name}: {e}")
                        stats['failed'] += 1
            
            # 归档风险指标
            metrics_dir = Path("data/risk/metrics")
            if metrics_dir.exists():
                for file_path in metrics_dir.glob("*.json"):
                    try:
                        import json
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        created_at = datetime.fromisoformat(data.get('created_at', ''))
                        if created_at < cutoff_date:
                            archive_dir = metrics_dir / "archive"
                            archive_dir.mkdir(exist_ok=True)
                            file_path.rename(archive_dir / file_path.name)
                            stats['archived_metrics'] += 1
                    except Exception as e:
                        logger.error(f"归档风险指标失败 {file_path.name}: {e}")
                        stats['failed'] += 1
            
            logger.info(f"归档历史风险数据: 检查={stats['archived_checks']}, 指标={stats['archived_metrics']}")
            
        except Exception as e:
            logger.error(f"归档历史风险数据失败: {e}")
            stats['failed'] += 1
        
        return stats
    
    def _cleanup_temp_files(self) -> Dict[str, int]:
        """
        清理临时文件
        
        Returns:
            清理统计
        """
        stats = {'cleaned': 0, 'failed': 0}
        
        temp_dirs = [
            Path("data/temp"),
            Path("data/cache/temp"),
            Path("logs/temp")
        ]
        
        cutoff_time = datetime.now() - timedelta(hours=self._temp_file_hours)
        
        for temp_dir in temp_dirs:
            if not temp_dir.exists():
                continue
            
            for file_path in temp_dir.glob("*"):
                try:
                    if file_path.is_file():
                        mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mtime < cutoff_time:
                            file_path.unlink()
                            stats['cleaned'] += 1
                except Exception as e:
                    logger.error(f"清理临时文件失败 {file_path}: {e}")
                    stats['failed'] += 1
        
        logger.info(f"清理临时文件: 清理={stats['cleaned']}, 失败={stats['failed']}")
        
        return stats
    
    def _cleanup_expired_cache(self) -> Dict[str, int]:
        """
        清理过期缓存
        
        Returns:
            清理统计
        """
        stats = {'cleaned': 0, 'failed': 0}
        
        cache_dirs = [
            Path("data/features/cache"),
            Path("data/ml/cache"),
            Path("data/trading/cache")
        ]
        
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for cache_dir in cache_dirs:
            if not cache_dir.exists():
                continue
            
            for file_path in cache_dir.glob("*.cache"):
                try:
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime < cutoff_time:
                        file_path.unlink()
                        stats['cleaned'] += 1
                except Exception as e:
                    logger.error(f"清理缓存文件失败 {file_path}: {e}")
                    stats['failed'] += 1
        
        logger.info(f"清理过期缓存: 清理={stats['cleaned']}, 失败={stats['failed']}")
        
        return stats
    
    def run_task(self, task_name: str) -> Optional[Dict[str, Any]]:
        """
        手动执行指定任务
        
        Args:
            task_name: 任务名称
        
        Returns:
            任务执行结果
        """
        if task_name not in self._tasks:
            logger.error(f"任务不存在: {task_name}")
            return None
        
        try:
            logger.info(f"手动执行清理任务: {task_name}")
            result = self._tasks[task_name]()
            self._last_run[task_name] = datetime.now()
            return result
        except Exception as e:
            logger.error(f"手动执行清理任务失败 {task_name}: {e}")
            return None
    
    def get_task_status(self) -> Dict[str, Any]:
        """
        获取任务状态
        
        Returns:
            任务状态信息
        """
        return {
            'running': self._cleanup_thread.is_alive() if self._cleanup_thread else False,
            'tasks': list(self._tasks.keys()),
            'last_run': {
                name: dt.isoformat() if dt else None
                for name, dt in self._last_run.items()
            },
            'config': {
                'alert_cleanup_hours': self._alert_cleanup_hours,
                'data_archive_days': self._data_archive_days,
                'temp_file_hours': self._temp_file_hours,
                'cleanup_interval': self._cleanup_interval
            }
        }


class CleanupScheduler:
    """
    清理任务调度器
    
    管理多个定时清理任务的调度执行。
    """
    
    def __init__(self):
        """初始化清理任务调度器"""
        self._tasks: Dict[str, ScheduledCleanupTask] = {}
        self._lock = threading.Lock()
        
        logger.info("清理任务调度器初始化完成")
    
    def add_task(self, task_name: str, task: ScheduledCleanupTask):
        """
        添加清理任务
        
        Args:
            task_name: 任务名称
            task: 清理任务实例
        """
        with self._lock:
            self._tasks[task_name] = task
            logger.info(f"添加清理任务到调度器: {task_name}")
    
    def remove_task(self, task_name: str):
        """
        移除清理任务
        
        Args:
            task_name: 任务名称
        """
        with self._lock:
            if task_name in self._tasks:
                self._tasks[task_name].stop()
                del self._tasks[task_name]
                logger.info(f"从调度器移除清理任务: {task_name}")
    
    def start_all(self):
        """启动所有清理任务"""
        with self._lock:
            for task_name, task in self._tasks.items():
                task.start()
                logger.info(f"启动清理任务: {task_name}")
    
    def stop_all(self):
        """停止所有清理任务"""
        with self._lock:
            for task_name, task in self._tasks.items():
                task.stop()
                logger.info(f"停止清理任务: {task_name}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取调度器状态
        
        Returns:
            调度器状态信息
        """
        with self._lock:
            return {
                'tasks': {
                    name: task.get_task_status()
                    for name, task in self._tasks.items()
                }
            }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="定时清理任务")
    parser.add_argument("--task", choices=["alert", "archive", "temp", "cache", "all"],
                       default="all", help="要执行的清理任务")
    parser.add_argument("--daemon", action="store_true", help="以守护进程方式运行")
    parser.add_argument("--interval", type=int, default=3600, help="清理间隔（秒）")
    
    args = parser.parse_args()
    
    config = {
        'cleanup_interval': args.interval
    }
    
    cleanup_task = ScheduledCleanupTask(config)
    
    if args.daemon:
        logger.info("以守护进程方式启动定时清理任务...")
        cleanup_task.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("接收到中断信号，停止清理任务...")
            cleanup_task.stop()
    else:
        if args.task == "all":
            cleanup_task._run_all_tasks()
        elif args.task == "alert":
            cleanup_task.run_task('alert_cleanup')
        elif args.task == "archive":
            cleanup_task.run_task('risk_data_archive')
        elif args.task == "temp":
            cleanup_task.run_task('temp_file_cleanup')
        elif args.task == "cache":
            cleanup_task.run_task('cache_cleanup')


if __name__ == "__main__":
    main()

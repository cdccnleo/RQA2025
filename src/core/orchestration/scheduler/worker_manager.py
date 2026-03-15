"""
工作进程管理器

管理工作进程的创建、监控和任务分配
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from queue import Queue, Empty

from .base import WorkerInfo, generate_worker_id


class WorkerManager:
    """
    工作进程管理器
    
    负责管理工作进程池，包括创建、监控、任务分配和心跳检测
    """
    
    def __init__(self, max_workers: int = 4, heartbeat_interval: int = 10):
        """
        初始化工作进程管理器
        
        Args:
            max_workers: 最大工作进程数
            heartbeat_interval: 心跳检测间隔（秒）
        """
        self._max_workers = max_workers
        self._heartbeat_interval = heartbeat_interval
        self._workers: Dict[str, WorkerInfo] = {}
        self._task_queue = Queue()
        self._running = False
        self._lock = threading.Lock()
        self._worker_threads: Dict[str, threading.Thread] = {}
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._task_handlers: Dict[str, Callable] = {}
        self._task_callbacks: Dict[str, Callable] = {}  # 任务完成/失败回调
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """
        注册任务处理器
        
        Args:
            task_type: 任务类型
            handler: 处理函数
        """
        self._task_handlers[task_type] = handler

    def register_task_callback(self, task_id: str, callback: Callable):
        """
        注册任务完成/失败回调

        当任务完成或失败时，会调用此回调函数

        Args:
            task_id: 任务ID
            callback: 回调函数，接收参数 (task_id, status, result, error)
        """
        self._task_callbacks[task_id] = callback

    def unregister_task_callback(self, task_id: str):
        """
        注销任务回调

        Args:
            task_id: 任务ID
        """
        if task_id in self._task_callbacks:
            del self._task_callbacks[task_id]
    
    async def start(self):
        """启动工作进程池"""
        if self._running:
            return
        
        self._running = True
        
        # 创建工作进程
        for i in range(self._max_workers):
            worker_id = generate_worker_id(i)
            worker = WorkerInfo(
                id=worker_id,
                status="idle",
                started_at=datetime.now(),
                last_heartbeat=datetime.now(),
                task_count=0
            )
            self._workers[worker_id] = worker
            
            # 启动工作线程
            thread = threading.Thread(
                target=self._worker_loop,
                args=(worker_id,),
                daemon=True,
                name=f"Worker-{worker_id}"
            )
            self._worker_threads[worker_id] = thread
            thread.start()
        
        # 启动心跳检测线程
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="WorkerHeartbeat"
        )
        self._heartbeat_thread.start()
    
    async def stop(self):
        """停止工作进程池"""
        self._running = False
        
        # 等待所有工作线程结束
        for worker_id, thread in self._worker_threads.items():
            if thread.is_alive():
                thread.join(timeout=5)
            self._workers[worker_id].status = "stopped"
        
        self._worker_threads.clear()
    
    def _worker_loop(self, worker_id: str):
        """
        工作线程主循环
        
        Args:
            worker_id: 工作进程ID
        """
        worker = self._workers[worker_id]
        
        while self._running:
            try:
                # 更新心跳
                worker.last_heartbeat = datetime.now()
                
                # 获取任务（阻塞，超时1秒）
                try:
                    task = self._task_queue.get(timeout=1)
                except Empty:
                    continue
                
                # 执行任务
                worker.status = "busy"
                worker.current_task = task.get("id")
                task_id = task.get("id")

                try:
                    # 获取任务处理器
                    task_type = task.get("type")
                    handler = self._task_handlers.get(task_type)
                    
                    print(f"[Worker {worker_id}] 获取任务: {task_id}, 类型: {task_type}, handler: {handler is not None}")

                    if handler:
                        # 执行处理函数
                        print(f"[Worker {worker_id}] 开始执行任务: {task_id}")
                        
                        # 将task_id添加到payload中，以便处理器获取
                        payload = task.get("payload", {})
                        if isinstance(payload, dict):
                            payload = payload.copy()
                            payload["_task_id"] = task_id
                        
                        if asyncio.iscoroutinefunction(handler):
                            # 异步函数
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(handler(payload))
                            loop.close()
                        else:
                            # 同步函数
                            result = handler(payload)
                        print(f"[Worker {worker_id}] 任务完成: {task_id}, 结果: {result}")

                        task["result"] = result
                        task["status"] = "completed"

                        # 调用回调
                        if task_id in self._task_callbacks:
                            try:
                                callback = self._task_callbacks[task_id]
                                callback(task_id, "completed", result, None)
                            except Exception as cb_error:
                                print(f"Task completion callback error: {cb_error}")
                            finally:
                                del self._task_callbacks[task_id]
                    else:
                        error_msg = f"No handler for task type: {task_type}"
                        print(f"[Worker {worker_id}] 错误: {error_msg}, 可用handlers: {list(self._task_handlers.keys())}")
                        task["error"] = error_msg
                        task["status"] = "failed"

                        # 调用回调
                        if task_id in self._task_callbacks:
                            try:
                                callback = self._task_callbacks[task_id]
                                callback(task_id, "failed", None, error_msg)
                            except Exception as cb_error:
                                print(f"Task failure callback error: {cb_error}")
                            finally:
                                del self._task_callbacks[task_id]

                except Exception as e:
                    error_msg = str(e)
                    task["error"] = error_msg
                    task["status"] = "failed"

                    # 调用回调
                    if task_id in self._task_callbacks:
                        try:
                            callback = self._task_callbacks[task_id]
                            callback(task_id, "failed", None, error_msg)
                        except Exception as cb_error:
                            print(f"Task failure callback error: {cb_error}")
                        finally:
                            del self._task_callbacks[task_id]

                finally:
                    worker.status = "idle"
                    worker.current_task = None
                    worker.task_count += 1
                    worker.last_heartbeat = datetime.now()
            
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                worker.status = "idle"
                worker.current_task = None
    
    def _heartbeat_loop(self):
        """心跳检测循环"""
        while self._running:
            time.sleep(self._heartbeat_interval)
            
            try:
                now = datetime.now()
                timeout = timedelta(seconds=self._heartbeat_interval * 3)
                
                for worker_id, worker in self._workers.items():
                    if worker.status == "stopped":
                        continue
                    
                    # 检查心跳超时
                    if now - worker.last_heartbeat > timeout:
                        print(f"Worker {worker_id} heartbeat timeout, restarting...")
                        worker.status = "idle"
                        worker.current_task = None
            
            except Exception as e:
                print(f"Heartbeat check error: {e}")
    
    def submit_task(self, task: Dict[str, Any]) -> bool:
        """
        提交任务到队列
        
        Args:
            task: 任务数据
        
        Returns:
            bool: 提交是否成功
        """
        try:
            self._task_queue.put(task, block=False)
            return True
        except:
            return False
    
    def get_workers(self) -> List[WorkerInfo]:
        """
        获取所有工作进程
        
        Returns:
            List[WorkerInfo]: 工作进程列表
        """
        return list(self._workers.values())
    
    def get_workers_dict(self) -> List[Dict[str, Any]]:
        """
        获取所有工作进程字典
        
        Returns:
            List[Dict]: 工作进程字典列表
        """
        return [worker.to_dict() for worker in self._workers.values()]
    
    def get_active_workers(self) -> List[WorkerInfo]:
        """
        获取活跃工作进程（运行中）
        
        Returns:
            List[WorkerInfo]: 活跃工作进程列表
        """
        return [
            w for w in self._workers.values()
            if w.status == "busy"
        ]
    
    def get_idle_workers(self) -> List[WorkerInfo]:
        """
        获取空闲工作进程
        
        Returns:
            List[WorkerInfo]: 空闲工作进程列表
        """
        return [
            w for w in self._workers.values()
            if w.status == "idle"
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        total = len(self._workers)
        active = len(self.get_active_workers())
        idle = len(self.get_idle_workers())
        stopped = len([w for w in self._workers.values() if w.status == "stopped"])
        
        total_tasks = sum(w.task_count for w in self._workers.values())
        
        return {
            "total": total,
            "active": active,
            "idle": idle,
            "stopped": stopped,
            "total_tasks_executed": total_tasks,
            "queue_size": self._task_queue.qsize()
        }
    
    def get_queue_size(self) -> int:
        """
        获取队列大小
        
        Returns:
            int: 队列中的任务数
        """
        return self._task_queue.qsize()
    
    def is_running(self) -> bool:
        """
        检查是否运行中
        
        Returns:
            bool: 是否运行中
        """
        return self._running

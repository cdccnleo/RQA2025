#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志聚合模块
负责收集、处理和存储系统各模块的日志数据
"""

import time
import threading
import queue
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.infrastructure.config import ConfigManager
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)

class StorageFailover:
    def __init__(self, primary, secondaries=None):
        self.primary = primary
        self.secondaries = secondaries or []
        self.current = primary
        
    def write(self, logs):
        try:
            return self.current.write(logs)
        except Exception as e:
            if self.secondaries:
                self.current = self.secondaries.pop(0)
                logger.warning(f"存储故障切换至备用: {type(self.current).__name__}")
                return self.write(logs)
            raise

class LogAggregator:
    def __init__(self, primary_storage=None, secondary_storage=None):
        """
        初始化日志聚合器
        :param primary_storage: 主存储后端
        :param secondary_storage: 备用存储后端
        """
        self.primary_storage = primary_storage
        self.secondary_storage = secondary_storage
        self.log_queue = queue.Queue(maxsize=10000)
        self.running = False
        self.processor_threads = []

        # 默认配置
        self.log_levels = ['INFO', 'WARNING', 'ERROR', 'CRITICAL']
        self.log_sources = ['trading', 'risk', 'data', 'feature']

    def start(self) -> None:
        """
        启动日志聚合器
        """
        self.running = True

        # 启动处理线程
        thread = threading.Thread(
            target=self._process_logs,
            daemon=True
        )
        thread.start()
        self.processor_threads.append(thread)
        logger.info("启动日志聚合器")

    def stop(self) -> None:
        """
        停止日志聚合器
        """
        self.running = False
        for thread in self.processor_threads:
            thread.join(timeout=5)
        logger.info("日志聚合器已停止")

    def add_log(self, log_data: Dict[str, Any]) -> bool:
        """
        添加日志到处理队列
        :param log_data: 日志数据字典
        :return: 是否添加成功
        """
        try:
            # 基本日志验证
            if not self._validate_log(log_data):
                return False

            # 添加时间戳
            log_data['@timestamp'] = datetime.utcnow().isoformat()

            # 放入队列
            self.log_queue.put(log_data, block=True, timeout=0.1)
            return True
        except queue.Full:
            logger.warning("日志队列已满，丢弃日志")
            return False
        except Exception as e:
            logger.error(f"添加日志失败: {str(e)}")
            return False

    def _validate_log(self, log_data: Dict[str, Any]) -> bool:
        """
        验证日志数据有效性
        :param log_data: 日志数据
        :return: 是否有效
        """
        required_fields = ['level', 'message', 'source']
        for field in required_fields:
            if field not in log_data:
                logger.warning(f"日志缺少必要字段: {field}")
                return False

        # 检查日志级别
        if log_data['level'] not in self.log_levels:
            return False

        # 检查日志来源
        if log_data.get('source') not in self.log_sources:
            return False

        return True

    def _process_logs(self) -> None:
        """
        日志处理线程
        """
        batch = []
        last_flush = time.time()

        while self.running or not self.log_queue.empty():
            try:
                # 从队列获取日志
                log = self.log_queue.get(block=True, timeout=1)
                batch.append(log)

                # 批量处理条件: 达到批量大小或超时
                if (len(batch) >= 100 or
                    time.time() - last_flush > 5):
                    try:
                        self._write_logs(batch)
                        batch.clear()
                        last_flush = time.time()
                    except Exception as e:
                        logger.error(f"日志处理失败: {str(e)}")
                        # 重试失败的批次
                        time.sleep(1)

            except queue.Empty:
                # 处理剩余日志
                if batch:
                    try:
                        self._write_logs(batch)
                        batch.clear()
                        last_flush = time.time()
                    except Exception as e:
                        logger.error(f"日志处理失败: {str(e)}")
                        time.sleep(1)
                continue

            except Exception as e:
                logger.error(f"日志处理线程异常: {str(e)}")
                time.sleep(1)

    def _write_logs(self, logs: List[Dict[str, Any]]) -> None:
        """
        写入日志到存储后端
        :param logs: 日志列表
        """
        try:
            # 尝试写入主存储
            if self.primary_storage:
                self.primary_storage.write(logs)
            else:
                # 默认文件存储
                self._write_to_file(logs)
        except Exception as e:
            logger.error(f"主存储写入失败: {str(e)}")
            # 尝试备用存储
            if self.secondary_storage:
                try:
                    self.secondary_storage.write(logs)
                except Exception as e2:
                    logger.error(f"备用存储写入也失败: {str(e2)}")
            else:
                # 最后的备用方案：写入文件
                self._write_to_file(logs)

    def _write_to_file(self, logs: List[Dict[str, Any]]) -> None:
        """
        写入日志到文件
        :param logs: 日志列表
        """
        import os
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = os.path.join(log_dir, f"aggregated_{timestamp}.log")
        
        with open(filename, 'a', encoding='utf-8') as f:
            for log in logs:
                f.write(f"{log.get('@timestamp', '')} [{log.get('level', 'INFO')}] {log.get('message', '')}\n")

    def search_logs(self, query: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """
        搜索日志
        :param query: 搜索条件
        :param limit: 返回结果数量限制
        :return: 日志列表
        """
        # 简化实现，实际项目中应使用搜索引擎
        results = []
        try:
            # 这里应该实现真实的搜索逻辑
            # 暂时返回空列表
            pass
        except Exception as e:
            logger.error(f"日志搜索失败: {str(e)}")
        return results

    def get_log_statistics(self, time_range: str = '1h') -> Dict[str, Any]:
        """
        获取日志统计信息
        :param time_range: 时间范围
        :return: 统计信息
        """
        stats = {
            'total_logs': 0,
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0,
            'sources': {}
        }
        
        try:
            # 这里应该实现真实的统计逻辑
            pass
        except Exception as e:
            logger.error(f"获取日志统计失败: {str(e)}")
        
        return stats

    def create_log_alert(self, condition: str, action: str) -> bool:
        """
        创建日志告警
        :param condition: 告警条件
        :param action: 告警动作
        :return: 是否创建成功
        """
        try:
            # 这里应该实现真实的告警创建逻辑
            logger.info(f"创建日志告警: {condition} -> {action}")
            return True
        except Exception as e:
            logger.error(f"创建日志告警失败: {str(e)}")
            return False

    def tail_logs(self, source: str = None, level: str = None, lines: int = 50) -> List[str]:
        """
        获取最新的日志
        :param source: 日志来源
        :param level: 日志级别
        :param lines: 行数
        :return: 日志行列表
        """
        try:
            # 这里应该实现真实的日志尾部获取逻辑
            return []
        except Exception as e:
            logger.error(f"获取日志尾部失败: {str(e)}")
            return []

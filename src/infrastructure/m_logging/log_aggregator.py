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
    def __init__(self, config: Dict[str, Any]):
        """
        初始化日志聚合器
        :param config: 系统配置
        """
        self.config = config
        self.config_manager = ConfigManager(config)
        self.log_queue = queue.Queue(maxsize=10000)
        
        # 初始化带故障转移的存储处理器
        storage_config = config.get('logging', {}).get('storage', {})
        primary = self._create_storage(storage_config.get('primary', 'file'))
        secondaries = [
            self._create_storage(s) 
            for s in storage_config.get('secondaries', [])
        ]
        
        self.log_processors = {
            'elasticsearch': StorageFailover(
                self._process_to_elasticsearch,
                [self._process_to_file, self._process_to_kafka]
            ),
            'file': StorageFailover(
                self._process_to_file,
                [self._process_to_kafka]
            ),
            'kafka': StorageFailover(
                self._process_to_kafka,
                [self._process_to_file]
            )
        }
        self.running = False
        self.processor_threads = []

        # 加载日志配置
        self.log_config = self.config_manager.get('logging', {})
        self.log_levels = self.log_config.get('levels', ['INFO', 'WARNING', 'ERROR', 'CRITICAL'])
        self.log_sources = self.log_config.get('sources', ['trading', 'risk', 'data', 'feature'])
        self.storage_backends = self.log_config.get('storage', ['file'])

    def start(self) -> None:
        """
        启动日志聚合器
        """
        self.running = True

        # 启动处理线程
        for backend in self.storage_backends:
            if backend in self.log_processors:
                thread = threading.Thread(
                    target=self._process_logs,
                    args=(backend,),
                    daemon=True
                )
                thread.start()
                self.processor_threads.append(thread)
                logger.info(f"启动日志处理器: {backend}")
            else:
                logger.warning(f"未知的日志存储后端: {backend}")

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

    def _process_logs(self, backend: str) -> None:
        """
        日志处理线程
        :param backend: 存储后端名称
        """
        processor = self.log_processors[backend]
        batch = []
        last_flush = time.time()

        while self.running or not self.log_queue.empty():
            try:
                # 从队列获取日志
                log = self.log_queue.get(block=True, timeout=1)
                batch.append(log)

                # 批量处理条件: 达到批量大小或超时
                if (len(batch) >= self.log_config.get('batch_size', 100) or
                    time.time() - last_flush > self.log_config.get('flush_interval', 5)):
                    try:
                        processor(batch)
                        batch.clear()
                        last_flush = time.time()
                    except Exception as e:
                        logger.error(f"{backend} 日志处理失败: {str(e)}")
                        # 重试失败的批次
                        time.sleep(1)

            except queue.Empty:
                # 处理剩余日志
                if batch:
                    try:
                        processor(batch)
                        batch.clear()
                        last_flush = time.time()
                    except Exception as e:
                        logger.error(f"{backend} 日志处理失败: {str(e)}")
                        time.sleep(1)
                continue

            except Exception as e:
                logger.error(f"日志处理线程异常: {str(e)}")
                time.sleep(1)

    def _process_to_elasticsearch(self, logs: List[Dict[str, Any]]) -> None:
        """
        处理日志到Elasticsearch
        :param logs: 日志列表
        """
        # 实际项目中应使用Elasticsearch客户端
        es_config = self.log_config.get('elasticsearch', {})
        logger.info(f"模拟将 {len(logs)} 条日志写入ES: {es_config.get('host')}")

    def _process_to_file(self, logs: List[Dict[str, Any]]) -> None:
        """
        处理日志到文件
        :param logs: 日志列表
        """
        log_file = self.log_config.get('file', {}).get('path', 'logs/rqa2025.log')
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                for log in logs:
                    f.write(f"{log['@timestamp']} [{log['level']}] {log['source']}: {log['message']}\n")
                    if 'exception' in log:
                        f.write(f"Exception: {log['exception']}\n")
            logger.debug(f"写入 {len(logs)} 条日志到文件: {log_file}")
        except Exception as e:
            logger.error(f"写入日志文件失败: {str(e)}")

    def _process_to_kafka(self, logs: List[Dict[str, Any]]) -> None:
        """
        处理日志到Kafka
        :param logs: 日志列表
        """
        kafka_config = self.log_config.get('kafka', {})
        # 实际项目中应使用Kafka生产者
        logger.info(f"模拟将 {len(logs)} 条日志发送到Kafka: {kafka_config.get('topic')}")

    def search_logs(self, query: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """
        搜索日志
        :param query: 查询条件
        :param limit: 返回条数限制
        :return: 匹配的日志列表
        """
        # 实际项目中应从存储后端查询
        return [{
            'timestamp': datetime.utcnow().isoformat(),
            'level': 'INFO',
            'source': 'log_aggregator',
            'message': 'Sample log entry for search result'
        }]

    def get_log_statistics(self, time_range: str = '1h') -> Dict[str, Any]:
        """
        获取日志统计信息
        :param time_range: 时间范围
        :return: 统计信息字典
        """
        # 实际项目中应从存储后端获取
        return {
            'total': 1000,
            'levels': {
                'INFO': 700,
                'WARNING': 200,
                'ERROR': 80,
                'CRITICAL': 20
            },
            'sources': {
                'trading': 400,
                'risk': 300,
                'data': 200,
                'feature': 100
            }
        }

    def create_log_alert(self, condition: str, action: str) -> bool:
        """
        创建日志告警规则
        :param condition: 告警条件
        :param action: 触发动作
        :return: 是否创建成功
        """
        # 实际项目中应保存到配置
        logger.info(f"创建日志告警规则: {condition} => {action}")
        return True

    def tail_logs(self, source: str = None, level: str = None, lines: int = 50) -> List[str]:
        """
        实时跟踪日志
        :param source: 日志来源
        :param level: 日志级别
        :param lines: 返回行数
        :return: 日志行列表
        """
        # 实际项目中应从存储后端获取
        sample_logs = []
        for i in range(lines):
            sample_logs.append(
                f"{datetime.utcnow().isoformat()} [INFO] {source or 'system'}: Sample log line {i}"
            )
        return sample_logs

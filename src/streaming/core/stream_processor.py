"""
Stream Processor Module
流处理器模块

This module provides unified stream processing capabilities
此模块提供统一的流处理能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import threading
import queue

try:
    from .constants import *
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from src.streaming.core.constants import *
    except ImportError:
        # 定义默认常量
        EVENT_QUEUE_SIZE = 10000
        DEFAULT_BATCH_SIZE = 1000

try:
    from .exceptions import *
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    try:
        from src.streaming.core.exceptions import *
    except ImportError:
        # 定义默认异常类
        class StreamingException(Exception):
            pass

# 使用标准logging
logger = logging.getLogger(__name__)


class StreamProcessor:

    """
    Unified Stream Processor
    统一流处理器

    Provides a unified interface for processing various types of data streams
    提供统一的流数据处理接口
    """

    def __init__(self, processor_id: str = None):
        """
        Initialize the stream processor
        初始化流处理器

        Args:
            processor_id: Unique identifier for this processor
                         此处理器的唯一标识符
        """
        self.processor_id = processor_id or f"stream_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.is_running = False
        self.processed_count = 0
        self.error_count = 0
        self.input_queue = queue.Queue(maxsize=EVENT_QUEUE_SIZE)
        self.output_queue = queue.Queue(maxsize=EVENT_QUEUE_SIZE)
        self.processing_thread = None
        self.middlewares: List[Callable] = []

        logger.info(f"Stream processor {self.processor_id} initialized")

    def add_middleware(self, middleware: Callable) -> None:
        """
        Add processing middleware
        添加处理中间件

        Args:
            middleware: Middleware function to process data
                       处理数据的中间件函数
        """
        self.middlewares.append(middleware)
        logger.info(f"Added middleware to processor {self.processor_id}")

    def start(self) -> bool:
        """
        Start the stream processor
        启动流处理器

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning(f"Processor {self.processor_id} is already running")
            return False

        try:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            logger.info(f"Stream processor {self.processor_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start processor {self.processor_id}: {str(e)}")
            self.is_running = False
            return False

    def stop(self) -> bool:
        """
        Stop the stream processor
        停止流处理器

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning(f"Processor {self.processor_id} is not running")
            return False

        try:
            self.is_running = False
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)
            logger.info(f"Stream processor {self.processor_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop processor {self.processor_id}: {str(e)}")
            return False

    def process_data(self, data: Any) -> bool:
        """
        Process input data
        处理输入数据

        Args:
            data: Data to be processed
                 要处理的数据

        Returns:
            bool: True if data was queued successfully, False otherwise
                  数据成功排队返回True，否则返回False
        """
        if not self.is_running:
            logger.warning(f"Processor {self.processor_id} is not running")
            return False

        try:
            self.input_queue.put(data, timeout=1.0)
            return True
        except queue.Full:
            logger.warning(f"Input queue full for processor {self.processor_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to queue data for processor {self.processor_id}: {str(e)}")
            return False

    def get_processed_data(self) -> Optional[Any]:
        """
        Get processed data from output queue
        从输出队列获取已处理的数据

        Returns:
            Processed data or None if no data available
            已处理的数据，如果没有数据则返回None
        """
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(
                f"Failed to get processed data from processor {self.processor_id}: {str(e)}")
            return None

    def _processing_loop(self) -> None:
        """
        Main processing loop
        主要的处理循环
        """
        logger.info(f"Processing loop started for processor {self.processor_id}")

        while self.is_running:
            try:
                # Get data from input queue
                data = self.input_queue.get(timeout=0.1)

                # Apply middlewares
                processed_data = data
                for middleware in self.middlewares:
                    try:
                        processed_data = middleware(processed_data)
                    except Exception as e:
                        logger.error(f"Middleware error in processor {self.processor_id}: {str(e)}")
                        self.error_count += 1
                        continue

                # Put processed data to output queue
                self.output_queue.put(processed_data, timeout=1.0)
                self.processed_count += 1

                # Log progress periodically
                if self.processed_count % DEFAULT_BATCH_SIZE == 0:
                    logger.info(
                        f"Processor {self.processor_id} processed {self.processed_count} items")

            except queue.Empty:
                continue
            except queue.Full:
                logger.warning(f"Output queue full for processor {self.processor_id}")
            except Exception as e:
                logger.error(f"Processing error in processor {self.processor_id}: {str(e)}")
                self.error_count += 1

        logger.info(f"Processing loop stopped for processor {self.processor_id}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics
        获取处理器统计信息

        Returns:
            dict: Processor statistics
                  处理器统计信息
        """
        return {
            'processor_id': self.processor_id,
            'is_running': self.is_running,
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'middleware_count': len(self.middlewares)
        }

    def reset_stats(self) -> None:
        """
        Reset processor statistics
        重置处理器统计信息
        """
        self.processed_count = 0
        self.error_count = 0
        logger.info(f"Statistics reset for processor {self.processor_id}")


# Global default processor instance
# 全局默认处理器实例

default_stream_processor = StreamProcessor("default_stream_processor")

__all__ = ['StreamProcessor', 'default_stream_processor']

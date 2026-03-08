import threading
import time
from typing import Any, Callable, List
from abc import ABC, abstractmethod

# 定义接口（如果不存在）
class IDataStream(ABC):
    """数据流接口"""
    @abstractmethod
    def subscribe(self, callback: Callable[[Any], None]) -> None:
        """订阅数据流"""
        pass
    
    @abstractmethod
    def unsubscribe(self, callback: Callable[[Any], None]) -> None:
        """取消订阅"""
        pass
    
    @abstractmethod
    def push(self, data: Any) -> None:
        """推送数据"""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """启动流"""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """停止流"""
        pass


class IDataStreamProcessor(ABC):
    """数据流处理器接口"""
    @abstractmethod
    def process(self, data: Any) -> None:
        """处理数据"""
        pass


class InMemoryStream(IDataStream):

    """
    内存队列实时数据流，支持多订阅者、异步推送
    """

    def __init__(self):

        self._subscribers: List[Callable[[Any], None]] = []
        self._queue: List[Any] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def subscribe(self, callback: Callable[[Any], None]) -> None:

        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Any], None]) -> None:

        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def push(self, data: Any) -> None:

        with self._lock:
            self._queue.append(data)

    def start(self) -> None:

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:

        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):

        while self._running:
            item = None
            with self._lock:
                if self._queue:
                    item = self._queue.pop(0)
            if item is not None:
                with self._lock:
                    for cb in self._subscribers:
                        try:
                            cb(item)
                        except Exception as e:
                            pass  # 可加日志
            else:
                time.sleep(0.01)


class SimpleStreamProcessor(IDataStreamProcessor):

    """
    简单流处理器，处理并打印数据
    """

    def process(self, data: Any) -> None:

        print(f"[StreamProcessor] Received: {data}")

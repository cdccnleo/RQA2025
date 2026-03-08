"""
ML模型缓存和批推理系统

功能:
- ML模型缓存和预热
- 批推理优化
- 模型版本管理
- 推理结果缓存
- GPU资源管理

性能目标:
- 模型加载时间减少 80%
- 推理吞吐量提升 200%
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Generic, List, Optional,
    TypeVar, Union, Tuple, AsyncIterator
)
from collections import OrderedDict
from contextlib import asynccontextmanager
from enum import Enum
import pickle

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class ModelFramework(Enum):
    """支持的ML框架"""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    SKLEARN = "sklearn"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """模型配置"""
    # 基础配置
    name: str
    version: str = "1.0.0"
    framework: ModelFramework = ModelFramework.CUSTOM
    model_path: Optional[str] = None

    # 缓存配置
    enable_cache: bool = True
    cache_ttl_seconds: float = 3600.0
    max_cache_size: int = 10000
    cache_key_generator: Optional[Callable[[Any], str]] = None

    # 批推理配置
    enable_batching: bool = True
    batch_size: int = 32
    max_wait_time: float = 0.05
    max_queue_size: int = 1000

    # 预热配置
    warmup_iterations: int = 10
    warmup_data: Optional[List[Any]] = None

    # GPU配置
    device: str = "cpu"
    gpu_memory_fraction: float = 0.8


@dataclass
class InferenceResult:
    """推理结果"""
    output: Any
    latency_ms: float
    batch_size: int = 1
    cached: bool = False
    model_version: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelMetrics:
    """模型指标"""
    total_requests: int = 0
    total_batches: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    throughput_per_sec: float = 0.0
    avg_batch_size: float = 0.0

    def record_request(self, latency_ms: float, cached: bool = False) -> None:
        """记录推理请求"""
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.total_requests

        if cached:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_batch(self, batch_size: int) -> None:
        """记录批处理"""
        self.total_batches += 1
        self.avg_batch_size = (
            (self.avg_batch_size * (self.total_batches - 1) + batch_size)
            / self.total_batches
        )

    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class LRUCache(Generic[T]):
    """
    LRU缓存实现

    用于缓存推理结果
    """

    def __init__(self, maxsize: int = 1000, ttl: float = 3600.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache: OrderedDict[str, Tuple[T, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[T]:
        """获取缓存值"""
        async with self._lock:
            if key not in self._cache:
                return None

            value, timestamp = self._cache[key]

            # 检查TTL
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                return None

            # 移动到末尾（最近使用）
            self._cache.move_to_end(key)
            return value

    async def set(self, key: str, value: T) -> None:
        """设置缓存值"""
        async with self._lock:
            # 如果已存在，先删除
            if key in self._cache:
                del self._cache[key]

            # 如果超过容量，删除最旧的
            while len(self._cache) >= self.maxsize:
                self._cache.popitem(last=False)

            self._cache[key] = (value, time.time())

    async def clear(self) -> None:
        """清空缓存"""
        async with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class ModelCache:
    """
    模型缓存管理器

    管理多个ML模型的加载、缓存和生命周期
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._configs: Dict[str, ModelConfig] = {}
        self._load_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def load_model(
        self,
        config: ModelConfig,
        loader: Callable[[ModelConfig], Any]
    ) -> Any:
        """
        加载模型

        Args:
            config: 模型配置
            loader: 模型加载函数

        Returns:
            加载的模型实例
        """
        model_key = f"{config.name}:{config.version}"

        async with self._lock:
            if model_key in self._models:
                logger.info(f"模型 '{model_key}' 已从缓存加载")
                return self._models[model_key]

            logger.info(f"正在加载模型 '{model_key}'...")
            start_time = time.time()

            try:
                model = await asyncio.get_event_loop().run_in_executor(
                    None, loader, config
                )

                load_time = time.time() - start_time
                self._models[model_key] = model
                self._configs[model_key] = config
                self._load_times[model_key] = load_time

                logger.info(f"模型 '{model_key}' 加载完成，耗时 {load_time:.2f}s")

                # 预热模型
                if config.warmup_iterations > 0:
                    await self._warmup_model(model_key, model, config)

                return model

            except Exception as e:
                logger.exception(f"模型 '{model_key}' 加载失败: {e}")
                raise

    async def _warmup_model(
        self,
        model_key: str,
        model: Any,
        config: ModelConfig
    ) -> None:
        """预热模型"""
        logger.info(f"正在预热模型 '{model_key}'...")

        warmup_data = config.warmup_data
        if warmup_data is None:
            # 生成随机预热数据
            warmup_data = [None] * config.warmup_iterations

        for i in range(config.warmup_iterations):
            try:
                # 这里假设模型有predict或__call__方法
                if hasattr(model, 'predict'):
                    await asyncio.get_event_loop().run_in_executor(
                        None, model.predict, warmup_data[i % len(warmup_data)]
                    )
                elif callable(model):
                    await asyncio.get_event_loop().run_in_executor(
                        None, model, warmup_data[i % len(warmup_data)]
                    )
            except Exception as e:
                logger.warning(f"预热迭代 {i} 失败: {e}")

        logger.info(f"模型 '{model_key}' 预热完成")

    async def get_model(self, name: str, version: str = "1.0.0") -> Optional[Any]:
        """获取已加载的模型"""
        model_key = f"{name}:{version}"
        async with self._lock:
            return self._models.get(model_key)

    async def unload_model(self, name: str, version: str = "1.0.0") -> bool:
        """卸载模型"""
        model_key = f"{name}:{version}"

        async with self._lock:
            if model_key in self._models:
                del self._models[model_key]
                del self._configs[model_key]
                del self._load_times[model_key]
                logger.info(f"模型 '{model_key}' 已卸载")
                return True
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """列出所有已加载的模型"""
        async with self._lock:
            return [
                {
                    "name": config.name,
                    "version": config.version,
                    "framework": config.framework.value,
                    "load_time": self._load_times.get(key, 0),
                    "device": config.device
                }
                for key, config in self._configs.items()
            ]

    async def clear(self) -> None:
        """清空所有模型"""
        async with self._lock:
            self._models.clear()
            self._configs.clear()
            self._load_times.clear()
            logger.info("所有模型已清空")


class BatchInferenceEngine(Generic[T, R]):
    """
    批推理引擎

    支持动态批处理和结果缓存
    """

    def __init__(
        self,
        model: Any,
        config: ModelConfig,
        inference_fn: Callable[[Any, List[T]], List[R]]
    ):
        self.model = model
        self.config = config
        self.inference_fn = inference_fn

        # 结果缓存
        self._cache = LRUCache[InferenceResult](
            maxsize=config.max_cache_size,
            ttl=config.cache_ttl_seconds
        )

        # 批处理队列
        self._queue: asyncio.Queue[Tuple[T, asyncio.Future[R]]] = asyncio.Queue(
            maxsize=config.max_queue_size
        )

        # 状态
        self._shutdown = False
        self._processing_task: Optional[asyncio.Task] = None

        # 指标
        self._metrics = ModelMetrics()
        self._start_time = time.time()

    async def start(self) -> None:
        """启动批推理引擎"""
        if self.config.enable_batching and self._processing_task is None:
            self._processing_task = asyncio.create_task(
                self._batch_processing_loop()
            )
            logger.info(f"批推理引擎 '{self.config.name}' 已启动")

    async def stop(self) -> None:
        """停止批推理引擎"""
        self._shutdown = True

        if self._processing_task:
            await self._queue.join()
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

        logger.info(f"批推理引擎 '{self.config.name}' 已停止")

    def _generate_cache_key(self, input_data: T) -> str:
        """生成缓存键"""
        if self.config.cache_key_generator:
            return self.config.cache_key_generator(input_data)

        # 默认使用输入数据的哈希
        try:
            data_str = json.dumps(input_data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            data_str = pickle.dumps(input_data)

        return hashlib.md5(data_str.encode()).hexdigest()

    async def predict(self, input_data: T) -> InferenceResult:
        """
        执行推理

        Args:
            input_data: 输入数据

        Returns:
            推理结果
        """
        # 检查缓存
        if self.config.enable_cache:
            cache_key = self._generate_cache_key(input_data)
            cached_result = await self._cache.get(cache_key)

            if cached_result is not None:
                cached_result.cached = True
                self._metrics.record_request(0, cached=True)
                return cached_result

        # 执行推理
        start_time = time.time()

        if self.config.enable_batching:
            # 使用批处理
            result = await self._batch_predict(input_data)
        else:
            # 直接推理
            result_data = await asyncio.get_event_loop().run_in_executor(
                None, self.inference_fn, self.model, [input_data]
            )
            latency_ms = (time.time() - start_time) * 1000
            result = InferenceResult(
                output=result_data[0],
                latency_ms=latency_ms,
                batch_size=1,
                model_version=self.config.version
            )

        # 缓存结果
        if self.config.enable_cache:
            await self._cache.set(cache_key, result)

        self._metrics.record_request(result.latency_ms, cached=False)
        return result

    async def predict_batch(self, inputs: List[T]) -> List[InferenceResult]:
        """
        批量推理

        Args:
            inputs: 输入数据列表

        Returns:
            推理结果列表
        """
        results = await asyncio.gather(*[self.predict(inp) for inp in inputs])
        return list(results)

    async def _batch_predict(self, input_data: T) -> InferenceResult:
        """使用批处理队列进行推理"""
        future: asyncio.Future[R] = asyncio.get_event_loop().create_future()

        try:
            await asyncio.wait_for(
                self._queue.put((input_data, future)),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError("推理队列已满")

        output = await future

        return InferenceResult(
            output=output,
            latency_ms=0,  # 将在外层计算
            batch_size=1,
            model_version=self.config.version
        )

    async def _batch_processing_loop(self) -> None:
        """批处理主循环"""
        while not self._shutdown:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
            except Exception as e:
                logger.exception(f"批处理循环错误: {e}")
                await asyncio.sleep(0.01)

    async def _collect_batch(self) -> List[Tuple[T, asyncio.Future[R]]]:
        """收集一批待处理项目"""
        batch: List[Tuple[T, asyncio.Future[R]]] = []

        # 等待第一个项目
        try:
            item = await asyncio.wait_for(
                self._queue.get(),
                timeout=0.5
            )
            batch.append(item)
        except asyncio.TimeoutError:
            return batch

        # 收集更多项目
        start_time = time.time()

        while len(batch) < self.config.batch_size:
            elapsed = time.time() - start_time
            remaining = self.config.max_wait_time - elapsed

            if remaining <= 0:
                break

            try:
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=remaining
                )
                batch.append(item)
            except asyncio.TimeoutError:
                break

        return batch

    async def _process_batch(self, batch: List[Tuple[T, asyncio.Future[R]]]) -> None:
        """处理一批项目"""
        start_time = time.time()
        inputs = [inp for inp, _ in batch]
        futures = [fut for _, fut in batch]

        try:
            # 执行批推理
            results = await asyncio.get_event_loop().run_in_executor(
                None, self.inference_fn, self.model, inputs
            )

            latency_ms = (time.time() - start_time) * 1000

            # 设置结果
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)

            # 更新指标
            self._metrics.record_batch(len(batch))

        except Exception as e:
            # 所有项目标记为失败
            for future in futures:
                if not future.done():
                    future.set_exception(e)
            logger.exception(f"批推理失败: {e}")

        finally:
            # 标记队列任务完成
            for _ in batch:
                self._queue.task_done()

    def get_metrics(self) -> ModelMetrics:
        """获取指标"""
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            self._metrics.throughput_per_sec = self._metrics.total_requests / elapsed
        return self._metrics


class ModelServingService:
    """
    模型服务总线

    统一管理多个模型的加载、缓存和推理
    """

    def __init__(self):
        self._model_cache = ModelCache()
        self._inference_engines: Dict[str, BatchInferenceEngine] = {}
        self._lock = asyncio.Lock()

    async def register_model(
        self,
        config: ModelConfig,
        loader: Callable[[ModelConfig], Any],
        inference_fn: Callable[[Any, List[T]], List[R]]
    ) -> BatchInferenceEngine:
        """
        注册模型

        Args:
            config: 模型配置
            loader: 模型加载函数
            inference_fn: 推理函数

        Returns:
            批推理引擎实例
        """
        model_key = f"{config.name}:{config.version}"

        async with self._lock:
            if model_key in self._inference_engines:
                logger.warning(f"模型 '{model_key}' 已注册")
                return self._inference_engines[model_key]

            # 加载模型
            model = await self._model_cache.load_model(config, loader)

            # 创建推理引擎
            engine = BatchInferenceEngine(model, config, inference_fn)
            await engine.start()

            self._inference_engines[model_key] = engine
            logger.info(f"模型 '{model_key}' 已注册并启动")

            return engine

    async def predict(
        self,
        model_name: str,
        input_data: T,
        version: str = "1.0.0"
    ) -> InferenceResult:
        """
        执行推理

        Args:
            model_name: 模型名称
            input_data: 输入数据
            version: 模型版本

        Returns:
            推理结果
        """
        model_key = f"{model_name}:{version}"

        async with self._lock:
            if model_key not in self._inference_engines:
                raise ValueError(f"模型 '{model_key}' 未注册")

            engine = self._inference_engines[model_key]

        return await engine.predict(input_data)

    async def predict_batch(
        self,
        model_name: str,
        inputs: List[T],
        version: str = "1.0.0"
    ) -> List[InferenceResult]:
        """批量推理"""
        model_key = f"{model_name}:{version}"

        async with self._lock:
            if model_key not in self._inference_engines:
                raise ValueError(f"模型 '{model_key}' 未注册")

            engine = self._inference_engines[model_key]

        return await engine.predict_batch(inputs)

    async def unregister_model(
        self,
        model_name: str,
        version: str = "1.0.0"
    ) -> bool:
        """注销模型"""
        model_key = f"{model_name}:{version}"

        async with self._lock:
            if model_key in self._inference_engines:
                engine = self._inference_engines[model_key]
                await engine.stop()
                del self._inference_engines[model_key]

            return await self._model_cache.unload_model(model_name, version)

    async def get_model_metrics(
        self,
        model_name: str,
        version: str = "1.0.0"
    ) -> Optional[ModelMetrics]:
        """获取模型指标"""
        model_key = f"{model_name}:{version}"

        async with self._lock:
            if model_key in self._inference_engines:
                return self._inference_engines[model_key].get_metrics()
            return None

    async def get_all_metrics(self) -> Dict[str, ModelMetrics]:
        """获取所有模型指标"""
        async with self._lock:
            return {
                key: engine.get_metrics()
                for key, engine in self._inference_engines.items()
            }

    async def shutdown(self) -> None:
        """关闭服务"""
        async with self._lock:
            for engine in self._inference_engines.values():
                await engine.stop()
            self._inference_engines.clear()
            await self._model_cache.clear()
            logger.info("模型服务已关闭")


# 便捷函数

async def create_inference_engine(
    model: Any,
    config: ModelConfig,
    inference_fn: Callable[[Any, List[T]], List[R]]
) -> BatchInferenceEngine:
    """
    创建并启动推理引擎的上下文管理器

    Args:
        model: 模型实例
        config: 模型配置
        inference_fn: 推理函数

    Yields:
        BatchInferenceEngine 实例
    """
    engine = BatchInferenceEngine(model, config, inference_fn)
    await engine.start()
    return engine

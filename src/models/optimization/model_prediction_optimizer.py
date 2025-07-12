import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
import time
from datetime import datetime
import torch
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import joblib
import os
import hashlib

logger = logging.getLogger(__name__)

class ModelPredictionOptimizer:
    """模型预测优化器"""
    
    def __init__(
        self,
        use_gpu: bool = True,
        batch_size: Optional[int] = None,
        n_jobs: int = -1,
        cache_dir: Optional[str] = None,
        cache_ttl: int = 3600  # 缓存有效期，单位秒
    ):
        """
        初始化模型预测优化器
        
        Args:
            use_gpu: 是否使用GPU
            batch_size: 批处理大小，None表示自动确定
            n_jobs: 并行作业数，-1表示使用所有可用CPU
            cache_dir: 缓存目录，None表示不使用缓存
            cache_ttl: 缓存有效期，单位秒
        """
        self.use_gpu = use_gpu and (torch.cuda.is_available() or tf.config.list_physical_devices('GPU'))
        self.batch_size = batch_size
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, os.cpu_count() - 1)
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl
        
        # 初始化设备
        self._init_devices()
        
        # 初始化缓存
        self._init_cache()
    
    def _init_devices(self) -> None:
        """初始化设备"""
        self.devices = []
        
        if self.use_gpu:
            # PyTorch设备
            if torch.cuda.is_available():
                self.torch_device = torch.device("cuda")
                for i in range(torch.cuda.device_count()):
                    self.devices.append(f"cuda:{i}")
            else:
                self.torch_device = torch.device("cpu")
                self.devices.append("cpu")
            
            # TensorFlow设备
            self.tf_gpus = tf.config.list_physical_devices('GPU')
            if self.tf_gpus:
                # 配置TensorFlow使用所有GPU
                try:
                    for gpu in self.tf_gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as e:
                    logger.warning(f"Error configuring TensorFlow GPU: {e}")
        else:
            self.torch_device = torch.device("cpu")
            self.devices.append("cpu")
            self.tf_gpus = []
    
    def _init_cache(self) -> None:
        """初始化缓存"""
        self.cache = {}
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, model_id: str, X: np.ndarray) -> str:
        """
        获取缓存键
        
        Args:
            model_id: 模型ID
            X: 输入数据
            
        Returns:
            str: 缓存键
        """
        # 计算输入数据的哈希值
        data_hash = hashlib.md5(X.tobytes()).hexdigest()
        
        return f"{model_id}_{data_hash}"
    
    def _get_from_cache(self, key: str) -> Optional[np.ndarray]:
        """
        从缓存获取预测结果
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[np.ndarray]: 预测结果
        """
        # 检查内存缓存
        if key in self.cache:
            cache_entry = self.cache[key]
            cache_time = cache_entry['time']
            
            # 检查缓存是否过期
            if time.time() - cache_time <= self.cache_ttl:
                return cache_entry['result']
            else:
                # 删除过期缓存
                del self.cache[key]
        
        # 检查磁盘缓存
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                try:
                    cache_entry = joblib.load(cache_file)
                    cache_time = cache_entry['time']
                    
                    # 检查缓存是否过期
                    if time.time() - cache_time <= self.cache_ttl:
                        # 更新内存缓存
                        self.cache[key] = cache_entry
                        return cache_entry['result']
                    else:
                        # 删除过期缓存
                        os.remove(cache_file)
                except Exception as e:
                    logger.warning(f"Error loading cache: {e}")
        
        return None
    
    def _save_to_cache(self, key: str, result: np.ndarray) -> None:
        """
        保存预测结果到缓存
        
        Args:
            key: 缓存键
            result: 预测结果
        """
        # 创建缓存条目
        cache_entry = {
            'result': result,
            'time': time.time()
        }
        
        # 保存到内存缓存
        self.cache[key] = cache_entry
        
        # 保存到磁盘缓存
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            try:
                joblib.dump(cache_entry, cache_file)
            except Exception as e:
                logger.warning(f"Error saving cache: {e}")
    
    def optimize_pytorch_prediction(
        self,
        model: torch.nn.Module,
        X: np.ndarray,
        batch_size: Optional[int] = None,
        model_id: Optional[str] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        优化PyTorch模型预测

        Args:
            model: PyTorch模型
            X: 输入数据
            batch_size: 批处理大小
            model_id: 模型ID，用于缓存
            use_cache: 是否使用缓存

        Returns:
            np.ndarray: 预测结果
        """
        # 检查缓存
        if use_cache and model_id:
            cache_key = self._get_cache_key(model_id, X)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

        # 确定批处理大小
        if batch_size is None:
            batch_size = self.batch_size or 128

        # 将模型移动到设备并设置为评估模式
        model = model.to(self.torch_device)
        model.eval()

        # 准备结果数组
        results = []

        # 批量预测
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                # 获取批次数据
                batch_X = X[i:i+batch_size]

                # 转换为PyTorch张量
                batch_tensor = torch.tensor(batch_X, dtype=torch.float32, device=self.torch_device)

                # 预测
                batch_output = model(batch_tensor)

                # 转换为NumPy数组并添加到结果
                batch_result = batch_output.cpu().numpy()
                results.append(batch_result)

        # 合并结果
        result = np.vstack(results)

        # 保存到缓存
        if use_cache and model_id:
            self._save_to_cache(cache_key, result)

        return result

    def optimize_tensorflow_prediction(
        self,
        model: tf.keras.Model,
        X: np.ndarray,
        batch_size: Optional[int] = None,
        model_id: Optional[str] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        优化TensorFlow模型预测

        Args:
            model: TensorFlow模型
            X: 输入数据
            batch_size: 批处理大小
            model_id: 模型ID，用于缓存
            use_cache: 是否使用缓存

        Returns:
            np.ndarray: 预测结果
        """
        # 检查缓存
        if use_cache and model_id:
            cache_key = self._get_cache_key(model_id, X)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

        # 确定批处理大小
        if batch_size is None:
            batch_size = self.batch_size or 128

        # 预测
        result = model.predict(X, batch_size=batch_size, verbose=0)

        # 保存到缓存
        if use_cache and model_id:
            self._save_to_cache(cache_key, result)

        return result

    def optimize_sklearn_prediction(
        self,
        model: Any,
        X: np.ndarray,
        batch_size: Optional[int] = None,
        model_id: Optional[str] = None,
        use_cache: bool = True,
        method: str = 'predict'
    ) -> np.ndarray:
        """
        优化scikit-learn模型预测

        Args:
            model: scikit-learn模型
            X: 输入数据
            batch_size: 批处理大小
            model_id: 模型ID，用于缓存
            use_cache: 是否使用缓存
            method: 预测方法，'predict'或'predict_proba'

        Returns:
            np.ndarray: 预测结果
        """
        # 检查缓存
        if use_cache and model_id:
            cache_key = self._get_cache_key(f"{model_id}_{method}", X)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                return cached_result

        # 确定批处理大小
        if batch_size is None or batch_size >= len(X):
            # 如果批处理大小未指定或大于数据量，直接预测
            predict_func = getattr(model, method)
            result = predict_func(X)
        else:
            # 批量预测
            results = []
            predict_func = getattr(model, method)

            for i in range(0, len(X), batch_size):
                # 获取批次数据
                batch_X = X[i:i+batch_size]

                # 预测
                batch_result = predict_func(batch_X)
                results.append(batch_result)

            # 合并结果
            result = np.vstack(results) if len(results[0].shape) > 1 else np.hstack(results)

        # 保存到缓存
        if use_cache and model_id:
            self._save_to_cache(cache_key, result)

        return result

    def parallel_batch_prediction(
        self,
        model: Any,
        X: np.ndarray,
        framework: str = 'sklearn',
        batch_size: Optional[int] = None,
        n_jobs: Optional[int] = None,
        method: str = 'predict'
    ) -> np.ndarray:
        """
        并行批量预测

        Args:
            model: 模型
            X: 输入数据
            framework: 框架，'pytorch'或'tensorflow'或'sklearn'
            batch_size: 批处理大小
            n_jobs: 并行作业数
            method: 预测方法，'predict'或'predict_proba'

        Returns:
            np.ndarray: 预测结果
        """
        # 确定并行作业数
        n_jobs = n_jobs or self.n_jobs

        # 确定批处理大小
        if batch_size is None:
            batch_size = self.batch_size or max(1, len(X) // (n_jobs * 4))

        # 划分数据
        batches = []
        for i in range(0, len(X), batch_size):
            batches.append(X[i:i+batch_size])

        # 定义预测函数
        def predict_batch(batch):
            if framework == 'pytorch':
                return self.optimize_pytorch_prediction(model, batch, batch_size=None)
            elif framework == 'tensorflow':
                return self.optimize_tensorflow_prediction(model, batch, batch_size=None)
            elif framework == 'sklearn':
                predict_func = getattr(model, method)
                return predict_func(batch)

        # 并行预测
        results = []

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            for result in executor.map(predict_batch, batches):
                results.append(result)

        # 合并结果
        result = np.vstack(results) if len(results[0].shape) > 1 else np.hstack(results)

        return result

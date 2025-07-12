# RQA2025 模型层功能增强分析报告（续）

## 2. 功能分析（续）

### 2.1 模型性能优化（续）

#### 2.1.1 模型训练优化（续）

**实现建议**（续）：

```python
    def optimize_tensorflow_training(
        self,
        model: tf.keras.Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        batch_size: Optional[int] = None,
        epochs: int = 100,
        callbacks: List[tf.keras.callbacks.Callback] = None
    ) -> tf.keras.callbacks.History:
        """
        优化TensorFlow模型训练
        
        Args:
            model: TensorFlow模型
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            batch_size: 批处理大小
            epochs: 训练轮数
            callbacks: 回调函数列表
            
        Returns:
            tf.keras.callbacks.History: 训练历史
        """
        # 确定批处理大小
        if batch_size is None:
            batch_size = self.batch_size or self.optimize_batch_size(
                model, X_train, y_train, framework='tensorflow'
            )
        
        # 准备验证数据
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # 准备回调函数
        all_callbacks = []
        
        # 添加早停回调
        if self.early_stopping:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=self.early_stopping_patience,
                min_delta=self.early_stopping_delta,
                restore_best_weights=True
            )
            all_callbacks.append(early_stopping_callback)
        
        # 添加用户提供的回调
        if callbacks:
            all_callbacks.extend(callbacks)
        
        # 训练模型
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=all_callbacks,
            verbose=1
        )
        
        return history
    
    def optimize_sklearn_training(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        callbacks: List[Callable] = None
    ) -> Dict:
        """
        优化scikit-learn模型训练
        
        Args:
            model: scikit-learn模型
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            callbacks: 回调函数列表
            
        Returns:
            Dict: 训练历史
        """
        # 初始化训练历史
        history = {
            'train_time': 0,
            'val_score': None,
            'val_time': 0
        }
        
        # 训练模型
        train_start_time = time.time()
        model.fit(X_train, y_train)
        history['train_time'] = time.time() - train_start_time
        
        # 验证模型
        if X_val is not None and y_val is not None:
            val_start_time = time.time()
            val_score = model.score(X_val, y_val)
            history['val_score'] = val_score
            history['val_time'] = time.time() - val_start_time
        
        # 执行回调
        if callbacks:
            for callback in callbacks:
                callback(model, history)
        
        return history
    
    def parallel_cross_validation(
        self,
        model_class: Any,
        model_params: Dict,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        framework: str = 'sklearn',
        fit_params: Optional[Dict] = None
    ) -> Dict:
        """
        并行交叉验证
        
        Args:
            model_class: 模型类
            model_params: 模型参数
            X: 特征数据
            y: 目标变量
            cv: 交叉验证折数
            framework: 框架，'pytorch'或'tensorflow'或'sklearn'
            fit_params: 拟合参数
            
        Returns:
            Dict: 交叉验证结果
        """
        from sklearn.model_selection import KFold
        
        # 初始化K折交叉验证
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        # 准备任务
        tasks = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            tasks.append((fold, X_train, y_train, X_val, y_val))
        
        # 定义训练函数
        def train_fold(args):
            fold, X_train, y_train, X_val, y_val = args
            
            # 创建模型
            if framework == 'sklearn':
                model = model_class(**model_params)
                result = self.optimize_sklearn_training(
                    model, X_train, y_train, X_val, y_val
                )
            elif framework == 'tensorflow':
                model = model_class(**model_params)
                result = self.optimize_tensorflow_training(
                    model, X_train, y_train, X_val, y_val,
                    **(fit_params or {})
                )
            elif framework == 'pytorch':
                # PyTorch需要更复杂的处理，这里简化处理
                model = model_class(**model_params)
                # 假设fit_params包含必要的训练参数
                result = {}
            
            return fold, model, result
        
        # 并行执行任务
        results = {}
        models = {}
        
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            for fold, model, result in executor.map(train_fold, tasks):
                results[fold] = result
                models[fold] = model
        
        # 计算平均结果
        avg_result = {}
        for key in results[0].keys():
            if isinstance(results[0][key], (int, float)):
                avg_result[key] = sum(results[fold][key] for fold in results) / len(results)
        
        return {
            'fold_results': results,
            'avg_result': avg_result,
            'models': models
        }
```

#### 2.1.2 模型预测优化

**现状分析**：
模型预测过程效率不高，尤其是在处理大量数据或需要实时预测时。

**实现建议**：
实现一个 `ModelPredictionOptimizer` 类，提供模型预测优化功能：

```python
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
        import hashlib
        
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
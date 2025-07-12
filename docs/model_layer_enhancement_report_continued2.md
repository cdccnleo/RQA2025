# RQA2025 模型层功能增强分析报告（续2）

## 2. 功能分析（续）

### 2.1 模型性能优化（续）

#### 2.1.2 模型预测优化（续）

**实现建议**（续）：

```python
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
```

### 2.2 模型评估

#### 2.2.1 全面评估指标

**现状分析**：
当前模型评估指标较为单一，难以全面评价模型性能。

**实现建议**：
实现一个 `ModelEvaluator` 类，提供全面的模型评估功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        初始化模型评估器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
        average: str = 'weighted'
    ) -> Dict:
        """
        评估分类模型
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_prob: 预测概率
            labels: 标签名称
            threshold: 分类阈值
            average: 平均方法
            
        Returns:
            Dict: 评估结果
        """
        # 确保y_pred是标签而不是概率
        if y_prob is not None and (y_pred is None or len(y_pred) == 0):
            y_pred = (y_prob >= threshold).astype(int)
        
        # 计算基本指标
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # 计算分类报告
        cr = classification_report(y_true, y_pred, output_dict=True)
        metrics['classification_report'] = cr
        
        # 计算ROC AUC（如果提供了预测概率）
        if y_prob is not None:
            try:
                if y_prob.shape[1] > 2:  # 多分类
                    metrics['roc_auc'] = roc_auc_score(
                        pd.get_dummies(y_true), y_prob, average=average, multi_class='ovr'
                    )
                else:  # 二分类
                    prob_col = 1 if y_prob.shape[1] == 2 else 0
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, prob_col])
            except Exception as e:
                logger.warning(f"Error calculating ROC AUC: {e}")
        
        return metrics
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        评估回归模型
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            Dict: 评估结果
        """
        # 计算基本指标
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # 计算平均绝对百分比误差
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Error calculating MAPE: {e}")
        
        # 计算中位数绝对误差
        metrics['median_ae'] = np.median(np.abs(y_true - y_pred))
        
        # 计算解释方差分数
        metrics['explained_variance'] = 1 - np.var(y_true - y_pred) / np.var(y_true)
        
        return metrics
    
    def evaluate_time_series(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon: int = 1
    ) -> Dict:
        """
        评估时间序列模型
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            horizon: 预测步长
            
        Returns:
            Dict: 评估结果
        """
        # 首先计算基本回归指标
        metrics = self.evaluate_regression(y_true, y_pred)
        
        # 计算方向准确率
        direction_true = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        metrics['direction_accuracy'] = np.mean(direction_true == direction_pred)
        
        # 计算自相关系数
        residuals = y_true - y_pred
        if len(residuals) > 1:
            metrics['residual_autocorr'] = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        
        # 计算不同步长的指标
        if horizon > 1:
            horizon_metrics = {}
            for h in range(1, min(horizon + 1, len(y_true))):
                # 计算h步预测的指标
                h_true = y_true[h:]
                h_pred = y_pred[:-h] if len(y_pred) > h else y_pred
                h_pred = h_pred[:len(h_true)]  # 确保长度匹配
                
                if len(h_true) > 0 and len(h_pred) > 0:
                    h_metrics = {
                        'mse': mean_squared_error(h_true, h_pred),
                        'rmse': np.sqrt(mean_squared_error(h_true, h_pred)),
                        'mae': mean_absolute_error(h_true, h_pred)
                    }
                    horizon_metrics[f'h{h}'] = h_metrics
            
            metrics['horizon_metrics'] = horizon_metrics
        
        return metrics
    
    def evaluate_ranking(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        k: int = 10
    ) -> Dict:
        """
        评估排序模型
        
        Args:
            y_true: 真实相关性
            y_score: 预测分数
            k: 前K个结果
            
        Returns:
            Dict: 评估结果
        """
        from sklearn.metrics import ndcg_score, average_precision_score
        
        # 计算NDCG@k
        try:
            ndcg = ndcg_score([y_true], [y_score], k=k)
        except Exception as e:
            logger.warning(f"Error calculating NDCG: {e}")
            ndcg = None
        
        # 计算MAP
        try:
            ap = average_precision_score(y_true, y_score)
        except Exception as e:
            logger.warning
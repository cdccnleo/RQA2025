# RQA2025 模型层功能增强分析报告（续6）

## 2. 功能分析（续）

### 2.3 模型集成（续）

#### 2.3.1 高级堆叠集成（续）

**实现建议**（续）：

```python
            'stratified': self.stratified,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'base_model_names': self.base_model_names,
            'is_classification': self.is_classification
        }
        joblib.dump(config, os.path.join(path, "config.pkl"))
    
    @classmethod
    def load(cls, path: str) -> 'AdvancedStackingEnsemble':
        """
        加载模型
        
        Args:
            path: 模型路径
            
        Returns:
            AdvancedStackingEnsemble: 加载的模型
        """
        # 加载配置
        config = joblib.load(os.path.join(path, "config.pkl"))
        
        # 加载基础模型
        base_models = []
        i = 0
        while os.path.exists(os.path.join(path, f"base_model_{i}.pkl")):
            model = joblib.load(os.path.join(path, f"base_model_{i}.pkl"))
            base_models.append(model)
            i += 1
        
        # 加载元模型
        meta_model = joblib.load(os.path.join(path, "meta_model.pkl"))
        
        # 创建实例
        instance = cls(
            base_models=[],  # 将在后面替换
            meta_model=None,  # 将在后面替换
            n_folds=config['n_folds'],
            use_features=config['use_features'],
            use_proba=config['use_proba'],
            stratified=config['stratified'],
            shuffle=config['shuffle'],
            random_state=config['random_state']
        )
        
        # 设置属性
        instance.base_models = base_models
        instance.meta_model = meta_model
        instance.base_models_fitted = base_models
        instance.base_model_names = config['base_model_names']
        instance.is_classification = config['is_classification']
        
        return instance
```

#### 2.3.2 动态集成选择

**现状分析**：
当前集成方法缺乏动态选择机制，无法根据具体样本选择最优的模型组合。

**实现建议**：
实现一个 `DynamicEnsembleSelector` 类，提供动态集成选择功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from sklearn.neighbors import NearestNeighbors
import joblib
import os

logger = logging.getLogger(__name__)

class DynamicEnsembleSelector:
    """动态集成选择器"""
    
    def __init__(
        self,
        base_models: List[BaseEstimator],
        k_neighbors: int = 5,
        selection_method: str = 'weighted',
        metric: str = 'accuracy',
        threshold: float = 0.0,
        use_proba: bool = False,
        random_state: int = 42
    ):
        """
        初始化动态集成选择器
        
        Args:
            base_models: 基础模型列表
            k_neighbors: 近邻数量
            selection_method: 选择方法，'weighted'或'threshold'或'top_k'
            metric: 评估指标
            threshold: 性能阈值
            use_proba: 是否使用预测概率
            random_state: 随机种子
        """
        self.base_models = base_models
        self.k_neighbors = k_neighbors
        self.selection_method = selection_method
        self.metric = metric
        self.threshold = threshold
        self.use_proba = use_proba
        self.random_state = random_state
        
        # 初始化评分器
        self.scorer = get_scorer(metric)
        
        # 初始化近邻模型
        self.knn = NearestNeighbors(
            n_neighbors=k_neighbors,
            metric='euclidean',
            n_jobs=-1
        )
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_size: float = 0.2
    ) -> 'DynamicEnsembleSelector':
        """
        训练动态集成选择器
        
        Args:
            X: 特征数据
            y: 目标变量
            validation_size: 验证集比例
            
        Returns:
            DynamicEnsembleSelector: 自身
        """
        # 分割验证集
        n_samples = len(X)
        n_val = int(n_samples * validation_size)
        
        # 随机打乱索引
        indices = np.random.RandomState(self.random_state).permutation(n_samples)
        val_indices = indices[:n_val]
        
        X_val = X[val_indices]
        y_val = y[val_indices]
        
        # 存储验证集特征
        self.X_val = X_val
        self.y_val = y_val
        
        # 训练近邻模型
        self.knn.fit(X_val)
        
        # 计算每个基础模型在每个验证样本上的性能
        self.model_performances = np.zeros((len(self.base_models), len(X_val)))
        
        for i, model in enumerate(self.base_models):
            if self.use_proba and hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_val)
            else:
                y_pred = model.predict(X_val)
            
            # 计算每个样本的性能
            for j in range(len(X_val)):
                if self.use_proba:
                    sample_pred = y_pred[j:j+1]
                else:
                    sample_pred = np.array([y_pred[j]])
                sample_true = np.array([y_val[j]])
                
                try:
                    score = self.scorer._score_func(sample_true, sample_pred)
                except:
                    # 如果单样本评分失败，使用局部窗口
                    window_size = min(10, len(X_val))
                    start_idx = max(0, j - window_size // 2)
                    end_idx = min(len(X_val), j + window_size // 2)
                    
                    window_pred = y_pred[start_idx:end_idx]
                    window_true = y_val[start_idx:end_idx]
                    
                    score = self.scorer._score_func(window_true, window_pred)
                
                self.model_performances[i, j] = score
        
        return self
    
    def select_models(
        self,
        X: np.ndarray,
        return_weights: bool = False
    ) -> Union[List[int], Tuple[List[int], np.ndarray]]:
        """
        为给定样本选择模型
        
        Args:
            X: 特征数据
            return_weights: 是否返回权重
            
        Returns:
            Union[List[int], Tuple[List[int], np.ndarray]]: 选择的模型索引和权重
        """
        # 找到最近邻
        distances, indices = self.knn.kneighbors(X)
        
        # 计算权重
        weights = 1 / (distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # 计算每个模型的加权性能
        model_scores = np.zeros((len(X), len(self.base_models)))
        
        for i in range(len(X)):
            # 获取近邻的性能
            neighbor_performances = self.model_performances[:, indices[i]]
            # 计算加权性能
            model_scores[i] = np.average(neighbor_performances, axis=1, weights=weights[i])
        
        # 根据选择方法选择模型
        selected_models = []
        model_weights = []
        
        for i in range(len(X)):
            if self.selection_method == 'weighted':
                # 使用性能作为权重
                weights = model_scores[i]
                weights = np.maximum(weights, 0)  # 确保权重非负
                weights = weights / (weights.sum() + 1e-10)  # 归一化
                
                selected_idx = np.where(weights > 1e-10)[0]
                selected_models.append(selected_idx)
                model_weights.append(weights[selected_idx])
            
            elif self.selection_method == 'threshold':
                # 选择性能超过阈值的模型
                selected_idx = np.where(model_scores[i] >= self.threshold)[0]
                if len(selected_idx) == 0:
                    # 如果没有模型超过阈值，选择最好的模型
                    selected_idx = [np.argmax(model_scores[i])]
                
                weights = model_scores[i][selected_idx]
                weights = weights / weights.sum()
                
                selected_models.append(selected_idx)
                model_weights.append(weights)
            
            elif self.selection_method == 'top_k':
                # 选择性能最好的k个模型
                k = min(self.k_neighbors, len(self.base_models))
                selected_idx = np.argsort(model_scores[i])[-k:]
                
                weights = model_scores[i][selected_idx]
                weights = weights / weights.sum()
                
                selected_models.append(selected_idx)
                model_weights.append(weights)
        
        if return_weights:
            return selected_models, np.array(model_weights)
        else:
            return selected_models
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            np.ndarray: 预测结果
        """
        # 选择模型
        selected_models, weights = self.select_models(X, return_weights=True)
        
        # 初始化预测结果
        predictions = np.zeros(len(X))
        
        # 对每个样本进行预测
        for i in range(len(X)):
            sample = X[i:i+1]
            sample_predictions = []
            
            # 获取选择的模型的预测
            for model_idx in selected_models[i]:
                if self.use_proba and hasattr(self.base_models[model_idx], 'predict_proba'):
                    pred = self.base_models[model_idx].predict_proba(sample)[0]
                    if len(pred) == 2:  # 二分类
                        sample_predictions.append(pred[1])
                    else:  # 多分类
                        sample_predictions.append(pred)
                else:
                    pred = self.base_models[model_idx].predict(sample)[0]
                    sample_predictions.append(pred)
            
            # 加权平均
            sample_predictions = np.array(sample_predictions)
            predictions[i] = np.average(sample_predictions, weights=weights[i])
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征数据
            
        Returns:
            np.ndarray: 预测概率
        """
        if not self.use_proba:
            raise ValueError("predict_proba is only available when use_proba=True")
        
        # 选择模型
        selected_models, weights = self.select_models(X, return_weights=True)
        
        # 获取类别数
        n_classes = None
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                test_pred = model.predict_proba(X[:1])
                n_classes = test_pred.shape[1]
                break
        
        if n_classes is None:
            raise ValueError("No model has predict_proba method")
        
        # 初始化预测结果
        predictions = np.zeros((len(X), n_classes))
        
        # 对每个样本进行预测
        for i in range(len(X)):
            sample = X[i:i+1]
            sample_predictions = []
            
            # 获取选择的模型的预测
            for model_idx in selected_models[i]:
                if hasattr(self.base_models[model_idx], 'predict_proba'):
                    pred = self.base_models[model_idx].predict_proba(sample)[0]
                    sample_predictions.append(pred)
            
            # 加权平均
            sample_predictions = np.array(sample_predictions)
            predictions[i] = np.average(sample_predictions, axis=0, weights=weights[i])
        
        return predictions
```

### 2.4 模型可解释性

#### 2.4.1 SHAP值分析

**现状分析**：
当前模型的可解释性不足，难以理解模型的决策过程。

**实现建议**：
实现一个 `ShapAnalyzer` 类，提供基于SHAP值的模型解释功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class ShapAnalyzer:
    """SHAP值分析器"""
    
    def __init__(
        self,
        model: Any,
        model_type: str = 'tree',
        output_dir: Optional[str] = None,
        random_state: int = 42
    ):
        """
        初始化SHAP值分析器
        
        Args:
            model: 模型
            model_type: 模型类型，'tree'或'kernel'或'deep'
            output_dir: 输出目录
            random_state: 随机种子
        """
        self.model = model
        self.model_type = model_type
        self.output_dir = output_dir
        self.random_state = random_state
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 初始化SHAP解释器
        self._init
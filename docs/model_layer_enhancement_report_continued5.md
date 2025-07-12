# RQA2025 模型层功能增强分析报告（续5）

## 2. 功能分析（续）

### 2.2 模型评估（续）

#### 2.2.2 交叉验证（续）

**实现建议**（续）：

```python
            # 创建网格搜索
            grid_search = GridSearchCV(
                model_class(),
                param_grid,
                cv=inner_cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                verbose=0 if not verbose else 1
            )
            
            # 训练网格搜索
            grid_search.fit(X_train, y_train, **(fit_params or {}))
            
            # 获取最佳参数和估计器
            best_params = grid_search.best_params_
            best_estimator = grid_search.best_estimator_
            
            # 在测试集上评估
            test_score = grid_search.score(X_test, y_test)
            
            # 保存结果
            results["test_score"][fold] = test_score
            results["best_params"].append(best_params)
            results["best_estimators"].append(best_estimator)
            
            if verbose:
                logger.info(f"Best parameters: {best_params}")
                logger.info(f"Test score: {test_score:.4f}")
        
        # 计算平均分数和标准差
        results["mean_test_score"] = np.mean(results["test_score"])
        results["std_test_score"] = np.std(results["test_score"])
        
        # 统计最佳参数
        param_counts = {}
        for params in results["best_params"]:
            param_str = str(params)
            param_counts[param_str] = param_counts.get(param_str, 0) + 1
        
        results["param_counts"] = param_counts
        results["most_common_params"] = max(param_counts.items(), key=lambda x: x[1])[0]
        
        # 显示结果
        if verbose:
            logger.info(f"Mean test score: {results['mean_test_score']:.4f} ± {results['std_test_score']:.4f}")
            logger.info(f"Most common parameters: {results['most_common_params']}")
        
        return results
```

### 2.3 模型集成

#### 2.3.1 高级堆叠集成

**现状分析**：
当前模型堆叠方法较为简单，未充分发挥集成学习的优势。

**实现建议**：
实现一个 `AdvancedStackingEnsemble` 类，提供高级堆叠集成功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold, StratifiedKFold
import joblib
import os

logger = logging.getLogger(__name__)

class AdvancedStackingEnsemble(BaseEstimator):
    """高级堆叠集成"""
    
    def __init__(
        self,
        base_models: List[BaseEstimator],
        meta_model: BaseEstimator,
        n_folds: int = 5,
        use_features: bool = False,
        use_proba: bool = False,
        stratified: bool = True,
        shuffle: bool = True,
        random_state: int = 42,
        verbose: bool = False
    ):
        """
        初始化高级堆叠集成
        
        Args:
            base_models: 基础模型列表
            meta_model: 元模型
            n_folds: 交叉验证折数
            use_features: 是否在元模型中使用原始特征
            use_proba: 是否使用预测概率（分类问题）
            stratified: 是否使用分层抽样
            shuffle: 是否打乱数据
            random_state: 随机种子
            verbose: 是否显示进度
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_features = use_features
        self.use_proba = use_proba
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        
        # 初始化模型名称
        self.base_model_names = [
            f"model_{i}" for i in range(len(base_models))
        ]
        
        # 初始化交叉验证模型
        self.cv_models = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdvancedStackingEnsemble':
        """
        训练堆叠集成
        
        Args:
            X: 特征数据
            y: 目标变量
            
        Returns:
            AdvancedStackingEnsemble: 自身
        """
        # 确定任务类型
        self.is_classification = isinstance(self.base_models[0], ClassifierMixin)
        
        # 创建交叉验证分割器
        if self.stratified and self.is_classification:
            kf = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        else:
            kf = KFold(
                n_splits=self.n_folds,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        
        # 准备元特征
        meta_features = np.zeros((X.shape[0], len(self.base_models) * (2 if self.use_proba and self.is_classification else 1)))
        
        # 生成元特征
        for i, model in enumerate(self.base_models):
            if self.verbose:
                logger.info(f"Training base model {i+1}/{len(self.base_models)}")
            
            # 初始化当前模型的交叉验证模型
            self.cv_models[i] = []
            
            # 交叉验证生成元特征
            for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                if self.verbose:
                    logger.info(f"Fold {fold+1}/{self.n_folds}")
                
                # 分割数据
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                # 创建并训练模型
                clone_model = joblib.clone(model)
                clone_model.fit(X_train, y_train)
                
                # 保存模型
                self.cv_models[i].append(clone_model)
                
                # 生成元特征
                if self.is_classification and self.use_proba and hasattr(clone_model, 'predict_proba'):
                    val_preds = clone_model.predict_proba(X_val)
                    if val_preds.shape[1] == 2:  # 二分类
                        meta_features[val_idx, i] = val_preds[:, 1]
                    else:  # 多分类
                        col_idx_start = i * val_preds.shape[1]
                        col_idx_end = (i + 1) * val_preds.shape[1]
                        meta_features[val_idx, col_idx_start:col_idx_end] = val_preds
                else:
                    meta_features[val_idx, i] = clone_model.predict(X_val)
        
        # 训练完整的基础模型
        self.base_models_fitted = []
        for i, model in enumerate(self.base_models):
            if self.verbose:
                logger.info(f"Training full base model {i+1}/{len(self.base_models)}")
            
            # 创建并训练模型
            clone_model = joblib.clone(model)
            clone_model.fit(X, y)
            
            # 保存模型
            self.base_models_fitted.append(clone_model)
        
        # 准备元模型的输入
        if self.use_features:
            meta_input = np.hstack([meta_features, X])
        else:
            meta_input = meta_features
        
        # 训练元模型
        if self.verbose:
            logger.info("Training meta model")
        
        self.meta_model.fit(meta_input, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            np.ndarray: 预测结果
        """
        # 生成元特征
        meta_features = self._get_meta_features(X)
        
        # 准备元模型的输入
        if self.use_features:
            meta_input = np.hstack([meta_features, X])
        else:
            meta_input = meta_features
        
        # 预测
        return self.meta_model.predict(meta_input)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征数据
            
        Returns:
            np.ndarray: 预测概率
        """
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification")
        
        # 生成元特征
        meta_features = self._get_meta_features(X)
        
        # 准备元模型的输入
        if self.use_features:
            meta_input = np.hstack([meta_features, X])
        else:
            meta_input = meta_features
        
        # 预测概率
        return self.meta_model.predict_proba(meta_input)
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        获取元特征
        
        Args:
            X: 特征数据
            
        Returns:
            np.ndarray: 元特征
        """
        # 准备元特征
        meta_features = np.zeros((X.shape[0], len(self.base_models) * (2 if self.use_proba and self.is_classification else 1)))
        
        # 生成元特征
        for i, model in enumerate(self.base_models_fitted):
            if self.is_classification and self.use_proba and hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)
                if preds.shape[1] == 2:  # 二分类
                    meta_features[:, i] = preds[:, 1]
                else:  # 多分类
                    col_idx_start = i * preds.shape[1]
                    col_idx_end = (i + 1) * preds.shape[1]
                    meta_features[:, col_idx_start:col_idx_end] = preds
            else:
                meta_features[:, i] = model.predict(X)
        
        return meta_features
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性
        
        Returns:
            pd.DataFrame: 特征重要性
        """
        # 检查元模型是否有特征重要性
        if not hasattr(self.meta_model, 'feature_importances_') and not hasattr(self.meta_model, 'coef_'):
            raise ValueError("Meta model does not have feature_importances_ or coef_ attribute")
        
        # 获取特征重要性
        if hasattr(self.meta_model, 'feature_importances_'):
            importances = self.meta_model.feature_importances_
        else:
            importances = np.abs(self.meta_model.coef_).mean(axis=0) if self.meta_model.coef_.ndim > 1 else np.abs(self.meta_model.coef_)
        
        # 创建特征名称
        feature_names = []
        for i, name in enumerate(self.base_model_names):
            if self.is_classification and self.use_proba:
                # 检查模型是否有predict_proba方法
                if hasattr(self.base_models_fitted[i], 'predict_proba'):
                    # 获取类别数
                    n_classes = self.base_models_fitted[i].predict_proba(np.zeros((1, self.base_models_fitted[i].n_features_in_))).shape[1]
                    if n_classes == 2:  # 二分类
                        feature_names.append(f"{name}_proba")
                    else:  # 多分类
                        for c in range(n_classes):
                            feature_names.append(f"{name}_proba_class{c}")
                else:
                    feature_names.append(name)
            else:
                feature_names.append(name)
        
        # 如果使用原始特征，添加原始特征名称
        if self.use_features:
            if hasattr(self.base_models_fitted[0], 'feature_names_in_'):
                original_feature_names = self.base_models_fitted[0].feature_names_in_
            else:
                original_feature_names = [f"feature_{i}" for i in range(self.base_models_fitted[0].n_features_in_)]
            
            feature_names.extend(original_feature_names)
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        })
        
        # 排序
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        # 创建目录
        os.makedirs(path, exist_ok=True)
        
        # 保存基础模型
        for i, model in enumerate(self.base_models_fitted):
            joblib.dump(model, os.path.join(path, f"base_model_{i}.pkl"))
        
        # 保存元模型
        joblib.dump(self.meta_model, os.path.join(path, "meta_model.pkl"))
        
        # 保存配置
        config = {
            'n_folds': self.n_folds,
            'use_features': self.use_features,
            'use_proba': self.use_proba,
            'stratified': self.strat
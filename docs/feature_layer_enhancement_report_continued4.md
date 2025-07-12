# RQA2025 特征层功能增强分析报告（续4）

## 2. 功能分析（续）

### 2.4 特征工程自动化（续）

#### 2.4.2 特征选择自动化（续）

**实现建议**（续）：

```python
    def select_with_boruta(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = 'classification',
        max_iter: int = 100,
        perc: int = 90
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        使用Boruta算法进行特征选择
        
        Args:
            X: 特征数据
            y: 目标变量
            task_type: 任务类型，'classification'或'regression'
            max_iter: 最大迭代次数
            perc: 百分位数阈值
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: 选择的特征数据和特征名称列表
        """
        # 选择基础模型
        if task_type == 'classification':
            base_model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state
            )
        else:
            base_model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state
            )
        
        # 创建Boruta选择器
        selector = BorutaPy(
            estimator=base_model,
            n_estimators='auto',
            max_iter=max_iter,
            perc=perc,
            random_state=self.random_state,
            verbose=0
        )
        
        # 拟合选择器
        selector.fit(X.values, y.values)
        
        # 获取选择的特征索引
        selected_indices = np.where(selector.support_)[0]
        
        # 获取选择的特征名称
        selected_features = [X.columns[i] for i in selected_indices]
        
        # 转换数据
        X_selected = X.iloc[:, selected_indices]
        
        return X_selected, selected_features
    
    def select_with_pca(
        self,
        X: pd.DataFrame,
        n_components: Optional[Union[int, float]] = None,
        variance_threshold: float = 0.95
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        使用PCA进行特征选择
        
        Args:
            X: 特征数据
            n_components: 主成分数量，如果为None则使用方差阈值
            variance_threshold: 方差阈值
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: 选择的特征数据和特征名称列表
        """
        # 确定主成分数量
        if n_components is None:
            # 使用方差阈值
            pca = PCA(random_state=self.random_state)
            pca.fit(X)
            
            # 计算累积方差比
            cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
            
            # 确定满足方差阈值的主成分数量
            n_components = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
        
        # 创建PCA
        pca = PCA(n_components=n_components, random_state=self.random_state)
        
        # 拟合并转换数据
        X_pca = pca.fit_transform(X)
        
        # 创建特征名称
        feature_names = [f'PC{i+1}' for i in range(X_pca.shape[1])]
        
        # 创建DataFrame
        X_pca_df = pd.DataFrame(
            X_pca,
            columns=feature_names,
            index=X.index
        )
        
        return X_pca_df, feature_names
    
    def auto_select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: List[str],
        task_type: str = 'classification',
        params: Optional[Dict] = None
    ) -> Dict[str, Tuple[pd.DataFrame, List[str]]]:
        """
        自动特征选择
        
        Args:
            X: 特征数据
            y: 目标变量
            methods: 特征选择方法列表
            task_type: 任务类型，'classification'或'regression'
            params: 方法参数
            
        Returns:
            Dict[str, Tuple[pd.DataFrame, List[str]]]: 特征选择结果
        """
        params = params or {}
        results = {}
        
        for method in methods:
            try:
                if method == 'k_best':
                    k = params.get('k_best', {}).get('k', 10)
                    results[method] = self.select_k_best(X, y, k, task_type)
                
                elif method == 'model_based':
                    model_type = params.get('model_based', {}).get('model_type', 'random_forest')
                    threshold = params.get('model_based', {}).get('threshold', 'mean')
                    max_features = params.get('model_based', {}).get('max_features')
                    
                    results[method] = self.select_from_model(
                        X, y, model_type, task_type, threshold, max_features
                    )
                
                elif method == 'boruta':
                    max_iter = params.get('boruta', {}).get('max_iter', 100)
                    perc = params.get('boruta', {}).get('perc', 90)
                    
                    results[method] = self.select_with_boruta(
                        X, y, task_type, max_iter, perc
                    )
                
                elif method == 'pca':
                    n_components = params.get('pca', {}).get('n_components')
                    variance_threshold = params.get('pca', {}).get('variance_threshold', 0.95)
                    
                    results[method] = self.select_with_pca(
                        X, n_components, variance_threshold
                    )
                
                else:
                    logger.warning(f"Unknown feature selection method: {method}")
            
            except Exception as e:
                logger.error(f"Error in feature selection method {method}: {e}")
        
        return results
```

### 2.5 特征可解释性

#### 2.5.1 特征重要性可视化

**现状分析**：
特征的可解释性不足，难以理解特征对模型的影响。

**实现建议**：
实现一个 `FeatureExplainer` 类，提供特征可解释性功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
import logging

logger = logging.getLogger(__name__)

class FeatureExplainer:
    """特征解释器"""
    
    def __init__(self, random_state: int = 42):
        """
        初始化特征解释器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
    
    def plot_feature_importance(
        self,
        model,
        X: pd.DataFrame,
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        绘制特征重要性图
        
        Args:
            model: 模型
            X: 特征数据
            top_n: 显示前N个特征
            figsize: 图形大小
        """
        # 获取特征重要性
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            raise ValueError("Model does not have feature_importances_ or coef_ attribute")
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importances
        })
        
        # 排序并获取前N个特征
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # 创建图形
        plt.figure(figsize=figsize)
        sns.barplot(
            x='importance',
            y='feature',
            data=importance_df
        )
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_shap_values(
        self,
        model,
        X: pd.DataFrame,
        sample_size: Optional[int] = None,
        plot_type: str = 'summary',
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        绘制SHAP值图
        
        Args:
            model: 模型
            X: 特征数据
            sample_size: 样本大小
            plot_type: 图形类型，'summary'或'bar'或'waterfall'
            figsize: 图形大小
        """
        # 如果指定了样本大小，则采样数据
        if sample_size is not None and sample_size < len(X):
            X_sample = X.sample(sample_size, random_state=self.random_state)
        else:
            X_sample = X
        
        # 创建SHAP解释器
        try:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
        except Exception as e:
            logger.error(f"Error creating SHAP explainer: {e}")
            # 尝试使用TreeExplainer
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
            except Exception as e:
                logger.error(f"Error creating TreeExplainer: {e}")
                return
        
        # 设置图形大小
        plt.figure(figsize=figsize)
        
        # 绘制SHAP值图
        if plot_type == 'summary':
            shap.summary_plot(shap_values, X_sample)
        elif plot_type == 'bar':
            shap.summary_plot(shap_values, X_sample, plot_type='bar')
        elif plot_type == 'waterfall':
            # 选择第一个样本进行瀑布图绘制
            shap.waterfall_plot(shap_values[0])
        else:
            logger.warning(f"Unknown plot type: {plot_type}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_partial_dependence(
        self,
        model,
        X: pd.DataFrame,
        features: List[Union[str, Tuple[str, str]]],
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        绘制部分依赖图
        
        Args:
            model: 模型
            X: 特征数据
            features: 特征列表
            figsize: 图形大小
        """
        # 将特征名称转换为索引
        feature_indices = []
        for feature in features:
            if isinstance(feature, str):
                if feature in X.columns:
                    feature_indices.append(list(X.columns).index(feature))
                else:
                    logger.warning(f"Feature {feature} not found in X")
            elif isinstance(feature, tuple) and len(feature) == 2:
                if feature[0] in X.columns and feature[1] in X.columns:
                    feature_indices.append(
                        (list(X.columns).index(feature[0]), list(X.columns).index(feature[1]))
                    )
                else:
                    logger.warning(f"Feature pair {feature} not found in X")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制部分依赖图
        try:
            PartialDependenceDisplay.from_estimator(
                model,
                X,
                feature_indices,
                ax=ax
            )
        except Exception as e:
            logger.error(f"Error plotting partial dependence: {e}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_interactions(
        self,
        model,
        X: pd.DataFrame,
        feature_pairs: List[Tuple[str, str]],
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        绘制特征交互图
        
        Args:
            model: 模型
            X: 特征数据
            feature_pairs: 特征对列表
            figsize: 图形大小
        """
        # 将特征对转换为索引
        feature_indices = []
        for pair in feature_pairs:
            if pair[0] in X.columns and pair[1] in X.columns:
                feature_indices.append(
                    (list(X.columns).index(pair[0]), list(X.columns).index(pair[1]))
                )
            else:
                logger.warning(f"Feature pair {pair} not found in X")
        
        # 创建图形
        fig, axes = plt.subplots(
            nrows=1,
            ncols=len(feature_indices),
            figsize=figsize
        )
        
        # 如果只有一个特征对，则将axes转换为列表
        if len(feature_indices) == 1:
            axes = [axes]
        
        # 绘制特征交互图
        for i, feature_idx in enumerate(feature_indices):
            try:
                PartialDependenceDisplay.from_estimator(
                    model,
                    X,
                    [feature_idx],
                    ax=axes[i],
                    kind='both'
                )
                
                # 设置标题
                axes[i].set_title(f"{X.columns[feature_idx[0]]} vs {X.columns[feature_idx[1]]}")
            
            except Exception as e:
                logger.error(f"Error plotting feature interaction: {e}")
        
        plt.tight_layout()
        plt.show()
```

#### 2.5.2 特征解释报告

**现状分析**：
缺乏系统化的特征解释报告，难以全面理解
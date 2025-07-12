# RQA2025 模型层功能增强分析报告（续7）

## 2. 功能分析（续）

### 2.4 模型可解释性（续）

#### 2.4.1 SHAP值分析（续）

**实现建议**（续）：

```python
        # 初始化SHAP解释器
        self._init_explainer()
    
    def _init_explainer(self) -> None:
        """初始化SHAP解释器"""
        self.explainer = None
    
    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        background_samples: int = 100
    ) -> 'ShapAnalyzer':
        """
        训练SHAP解释器
        
        Args:
            X: 特征数据
            feature_names: 特征名称
            background_samples: 背景样本数量
            
        Returns:
            ShapAnalyzer: 自身
        """
        # 保存特征名称
        if feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names
        
        # 选择背景数据
        if background_samples < len(X):
            np.random.seed(self.random_state)
            background_indices = np.random.choice(len(X), background_samples, replace=False)
            background_data = X[background_indices]
        else:
            background_data = X
        
        # 创建SHAP解释器
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model, background_data)
        elif self.model_type == 'kernel':
            self.explainer = shap.KernelExplainer(self.model.predict, background_data)
        elif self.model_type == 'deep':
            self.explainer = shap.DeepExplainer(self.model, background_data)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self
    
    def explain(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        plot_type: str = 'summary',
        save_plots: bool = True,
        max_display: int = 20
    ) -> Dict:
        """
        解释模型预测
        
        Args:
            X: 特征数据
            y: 目标变量（可选）
            plot_type: 图表类型，'summary'或'dependence'或'force'或'interaction'
            save_plots: 是否保存图表
            max_display: 最大显示特征数量
            
        Returns:
            Dict: 解释结果
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        # 计算SHAP值
        shap_values = self.explainer.shap_values(X)
        
        # 准备结果
        results = {
            'shap_values': shap_values,
            'feature_names': self.feature_names,
            'plots': {}
        }
        
        # 生成图表
        if plot_type == 'summary':
            results['plots']['summary'] = self._plot_summary(
                shap_values, X, max_display, save_plots
            )
        elif plot_type == 'dependence':
            results['plots']['dependence'] = self._plot_dependence(
                shap_values, X, max_display, save_plots
            )
        elif plot_type == 'force':
            results['plots']['force'] = self._plot_force(
                shap_values, X, y, save_plots
            )
        elif plot_type == 'interaction':
            results['plots']['interaction'] = self._plot_interaction(
                shap_values, X, max_display, save_plots
            )
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        return results
    
    def _plot_summary(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        max_display: int,
        save_plots: bool
    ) -> Dict[str, plt.Figure]:
        """
        绘制SHAP值总结图
        
        Args:
            shap_values: SHAP值
            X: 特征数据
            max_display: 最大显示特征数量
            save_plots: 是否保存图表
            
        Returns:
            Dict[str, plt.Figure]: 图表字典
        """
        plots = {}
        
        # 条形图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            plot_type='bar',
            show=False
        )
        plots['bar'] = plt.gcf()
        
        if save_plots and self.output_dir:
            plots['bar'].savefig(
                os.path.join(self.output_dir, 'shap_summary_bar.png'),
                dpi=300,
                bbox_inches='tight'
            )
        
        # 蜂群图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plots['beeswarm'] = plt.gcf()
        
        if save_plots and self.output_dir:
            plots['beeswarm'].savefig(
                os.path.join(self.output_dir, 'shap_summary_beeswarm.png'),
                dpi=300,
                bbox_inches='tight'
            )
        
        return plots
    
    def _plot_dependence(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        max_display: int,
        save_plots: bool
    ) -> Dict[str, plt.Figure]:
        """
        绘制SHAP依赖图
        
        Args:
            shap_values: SHAP值
            X: 特征数据
            max_display: 最大显示特征数量
            save_plots: 是否保存图表
            
        Returns:
            Dict[str, plt.Figure]: 图表字典
        """
        plots = {}
        
        # 获取特征重要性排序
        feature_importance = np.abs(shap_values).mean(0)
        feature_order = np.argsort(-feature_importance)
        
        # 为最重要的特征绘制依赖图
        for i in range(min(max_display, len(self.feature_names))):
            feature_idx = feature_order[i]
            feature_name = self.feature_names[feature_idx]
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_idx,
                shap_values,
                X,
                feature_names=self.feature_names,
                show=False
            )
            plots[feature_name] = plt.gcf()
            
            if save_plots and self.output_dir:
                plots[feature_name].savefig(
                    os.path.join(self.output_dir, f'shap_dependence_{feature_name}.png'),
                    dpi=300,
                    bbox_inches='tight'
                )
        
        return plots
    
    def _plot_force(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        y: Optional[np.ndarray],
        save_plots: bool
    ) -> Dict[str, plt.Figure]:
        """
        绘制SHAP力图
        
        Args:
            shap_values: SHAP值
            X: 特征数据
            y: 目标变量
            save_plots: 是否保存图表
            
        Returns:
            Dict[str, plt.Figure]: 图表字典
        """
        plots = {}
        
        # 选择一些代表性样本
        n_samples = min(5, len(X))
        sample_indices = np.linspace(0, len(X)-1, n_samples, dtype=int)
        
        for i, idx in enumerate(sample_indices):
            plt.figure(figsize=(15, 3))
            shap.force_plot(
                self.explainer.expected_value,
                shap_values[idx],
                X[idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            
            title = f"Sample {idx}"
            if y is not None:
                title += f" (y={y[idx]})"
            plt.title(title)
            
            plots[f'sample_{idx}'] = plt.gcf()
            
            if save_plots and self.output_dir:
                plots[f'sample_{idx}'].savefig(
                    os.path.join(self.output_dir, f'shap_force_{idx}.png'),
                    dpi=300,
                    bbox_inches='tight'
                )
        
        return plots
    
    def _plot_interaction(
        self,
        shap_values: np.ndarray,
        X: np.ndarray,
        max_display: int,
        save_plots: bool
    ) -> Dict[str, plt.Figure]:
        """
        绘制SHAP交互图
        
        Args:
            shap_values: SHAP值
            X: 特征数据
            max_display: 最大显示特征数量
            save_plots: 是否保存图表
            
        Returns:
            Dict[str, plt.Figure]: 图表字典
        """
        plots = {}
        
        # 计算特征交互
        interaction_values = shap.TreeExplainer(self.model).shap_interaction_values(X)
        
        # 获取特征重要性排序
        feature_importance = np.abs(shap_values).mean(0)
        feature_order = np.argsort(-feature_importance)
        
        # 为最重要的特征对绘制交互图
        for i in range(min(max_display, len(self.feature_names))):
            feature_idx = feature_order[i]
            feature_name = self.feature_names[feature_idx]
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                (feature_idx, feature_idx),
                interaction_values,
                X,
                feature_names=self.feature_names,
                show=False
            )
            plots[f'{feature_name}_self'] = plt.gcf()
            
            if save_plots and self.output_dir:
                plots[f'{feature_name}_self'].savefig(
                    os.path.join(self.output_dir, f'shap_interaction_{feature_name}_self.png'),
                    dpi=300,
                    bbox_inches='tight'
                )
            
            # 与其他重要特征的交互
            for j in range(i+1, min(max_display, len(self.feature_names))):
                other_idx = feature_order[j]
                other_name = self.feature_names[other_idx]
                
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    (feature_idx, other_idx),
                    interaction_values,
                    X,
                    feature_names=self.feature_names,
                    show=False
                )
                plots[f'{feature_name}_{other_name}'] = plt.gcf()
                
                if save_plots and self.output_dir:
                    plots[f'{feature_name}_{other_name}'].savefig(
                        os.path.join(self.output_dir, f'shap_interaction_{feature_name}_{other_name}.png'),
                        dpi=300,
                        bbox_inches='tight'
                    )
        
        return plots
    
    def get_feature_importance(
        self,
        shap_values: np.ndarray,
        aggregate_method: str = 'mean_abs'
    ) -> pd.DataFrame:
        """
        获取特征重要性
        
        Args:
            shap_values: SHAP值
            aggregate_method: 聚合方法，'mean_abs'或'mean'或'sum'或'sum_abs'
            
        Returns:
            pd.DataFrame: 特征重要性
        """
        if aggregate_method == 'mean_abs':
            importance = np.abs(shap_values).mean(0)
        elif aggregate_method == 'mean':
            importance = shap_values.mean(0)
        elif aggregate_method == 'sum':
            importance = shap_values.sum(0)
        elif aggregate_method == 'sum_abs':
            importance = np.abs(shap_values).sum(0)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate_method}")
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })
        
        # 排序
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
```

### 2.5 模型监控

#### 2.5.1 模型漂移检测

**现状分析**：
缺乏对模型在生产环境中的监控机制，难以及时发现模型漂移问题。

**实现建议**：
实现一个 `ModelDriftDetector` 类，提供模型漂移检测功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from scipy import stats
from sklearn.metrics import wasserstein_distance
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class ModelDriftDetector:
    """模型漂移检测器"""
    
    def __init__(
        self,
        drift_threshold: float = 0.1,
        window_size: int = 1000,
        output_dir: Optional[str] = None
    ):
        """
        初始化模型漂移检测器
        
        Args:
            drift_threshold: 漂移阈值
            window_size: 滑动窗口大小
            output_dir: 输出目录
        """
        self.drift_threshold = drift_threshold
        self.window_size = window_size
        self.output_dir = output_dir
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # 初始化基准数据
        self.baseline_data = None
        self.baseline_predictions = None
        self.feature_distributions = None
    
    def set_baseline(
        self,
        X: np.ndarray,
        predictions:
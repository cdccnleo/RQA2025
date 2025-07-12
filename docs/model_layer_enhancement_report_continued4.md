# RQA2025 模型层功能增强分析报告（续4）

## 2. 功能分析（续）

### 2.2 模型评估（续）

#### 2.2.1 全面评估指标（续）

**实现建议**（续）：

```python
                # 添加准确率
                if 'accuracy' in cr:
                    report += f"\n{'accuracy':15s} {' ':10s} {' ':10s} {cr['accuracy']:10.4f} {cr['macro avg']['support']:10d}\n"
                
                report += "```\n"
        
        elif task_type == 'regression':
            report += "### Regression Metrics\n\n"
            report += f"- Mean Squared Error (MSE): {metrics.get('mse', 'N/A'):.4f}\n"
            report += f"- Root Mean Squared Error (RMSE): {metrics.get('rmse', 'N/A'):.4f}\n"
            report += f"- Mean Absolute Error (MAE): {metrics.get('mae', 'N/A'):.4f}\n"
            report += f"- R² Score: {metrics.get('r2', 'N/A'):.4f}\n"
            
            if 'mape' in metrics:
                report += f"- Mean Absolute Percentage Error (MAPE): {metrics['mape']:.4f}\n"
            
            if 'median_ae' in metrics:
                report += f"- Median Absolute Error: {metrics['median_ae']:.4f}\n"
            
            if 'explained_variance' in metrics:
                report += f"- Explained Variance: {metrics['explained_variance']:.4f}\n"
        
        elif task_type == 'time_series':
            report += "### Time Series Metrics\n\n"
            report += f"- Mean Squared Error (MSE): {metrics.get('mse', 'N/A'):.4f}\n"
            report += f"- Root Mean Squared Error (RMSE): {metrics.get('rmse', 'N/A'):.4f}\n"
            report += f"- Mean Absolute Error (MAE): {metrics.get('mae', 'N/A'):.4f}\n"
            report += f"- Direction Accuracy: {metrics.get('direction_accuracy', 'N/A'):.4f}\n"
            
            if 'residual_autocorr' in metrics:
                report += f"- Residual Autocorrelation: {metrics['residual_autocorr']:.4f}\n"
            
            # 添加不同步长的指标
            if 'horizon_metrics' in metrics:
                report += "\n### Horizon Metrics\n\n"
                report += "| Horizon | MSE | RMSE | MAE |\n"
                report += "| ------- | --- | ---- | --- |\n"
                
                for h, h_metrics in metrics['horizon_metrics'].items():
                    report += f"| {h} | {h_metrics['mse']:.4f} | {h_metrics['rmse']:.4f} | {h_metrics['mae']:.4f} |\n"
        
        elif task_type == 'ranking':
            report += "### Ranking Metrics\n\n"
            
            if 'ndcg@k' in metrics:
                report += f"- NDCG@k: {metrics['ndcg@k']:.4f}\n"
            
            if 'map' in metrics:
                report += f"- Mean Average Precision (MAP): {metrics['map']:.4f}\n"
            
            report += f"- Precision@k: {metrics.get('precision@k', 'N/A'):.4f}\n"
            report += f"- Recall@k: {metrics.get('recall@k', 'N/A'):.4f}\n"
        
        # 添加图表
        if plots:
            report += "\n## Visualization\n\n"
            
            for plot_name, fig in plots.items():
                if save_path:
                    plot_dir = os.path.join(os.path.dirname(save_path), 'plots')
                    os.makedirs(plot_dir, exist_ok=True)
                    plot_path = os.path.join(plot_dir, f"{plot_name}.png")
                    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                    report += f"### {plot_name.replace('_', ' ').title()}\n\n"
                    report += f"![{plot_name}](plots/{plot_name}.png)\n\n"
        
        # 保存报告
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report
```

#### 2.2.2 交叉验证

**现状分析**：
当前模型评估缺乏系统化的交叉验证机制，难以评估模型的稳定性和泛化能力。

**实现建议**：
实现一个 `CrossValidator` 类，提供交叉验证功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from concurrent.futures import ProcessPoolExecutor
import time
import os
import copy

logger = logging.getLogger(__name__)

class CrossValidator:
    """交叉验证器"""
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        初始化交叉验证器
        
        Args:
            n_splits: 分割数
            shuffle: 是否打乱数据
            random_state: 随机种子
            n_jobs: 并行作业数，-1表示使用所有可用CPU
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs > 0 else max(1, os.cpu_count() - 1)
    
    def _get_splitter(
        self,
        cv_type: str,
        y: Optional[np.ndarray] = None
    ) -> Union[KFold, StratifiedKFold, TimeSeriesSplit]:
        """
        获取分割器
        
        Args:
            cv_type: 交叉验证类型，'kfold'或'stratified'或'timeseries'
            y: 目标变量，用于分层抽样
            
        Returns:
            Union[KFold, StratifiedKFold, TimeSeriesSplit]: 分割器
        """
        if cv_type == 'stratified' and y is not None:
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        elif cv_type == 'timeseries':
            return TimeSeriesSplit(n_splits=self.n_splits)
        else:
            return KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
    
    def cross_validate(
        self,
        model_class: Any,
        model_params: Dict,
        X: np.ndarray,
        y: np.ndarray,
        cv_type: str = 'kfold',
        scoring: Union[str, List[str], Dict[str, Callable]] = 'accuracy',
        fit_params: Optional[Dict] = None,
        return_train_score: bool = False,
        return_estimator: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        交叉验证
        
        Args:
            model_class: 模型类
            model_params: 模型参数
            X: 特征数据
            y: 目标变量
            cv_type: 交叉验证类型，'kfold'或'stratified'或'timeseries'
            scoring: 评分方法
            fit_params: 拟合参数
            return_train_score: 是否返回训练集分数
            return_estimator: 是否返回估计器
            verbose: 是否显示进度
            
        Returns:
            Dict: 交叉验证结果
        """
        from sklearn.metrics import get_scorer
        
        # 获取分割器
        splitter = self._get_splitter(cv_type, y)
        
        # 准备评分器
        if isinstance(scoring, str):
            scorers = {scoring: get_scorer(scoring)}
        elif isinstance(scoring, list):
            scorers = {s: get_scorer(s) for s in scoring}
        elif isinstance(scoring, dict):
            scorers = scoring
        else:
            raise ValueError("scoring should be a string, a list or a dict")
        
        # 准备结果
        results = {
            f"test_{name}": np.zeros(self.n_splits) for name in scorers
        }
        
        if return_train_score:
            results.update({
                f"train_{name}": np.zeros(self.n_splits) for name in scorers
            })
        
        if return_estimator:
            results["estimator"] = []
        
        # 准备任务
        tasks = []
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            tasks.append((fold, X_train, y_train, X_test, y_test))
        
        # 定义训练函数
        def train_and_evaluate(args):
            fold, X_train, y_train, X_test, y_test = args
            
            # 创建模型
            model = model_class(**model_params)
            
            # 训练模型
            start_time = time.time()
            model.fit(X_train, y_train, **(fit_params or {}))
            train_time = time.time() - start_time
            
            # 评估模型
            fold_results = {
                "fold": fold,
                "train_time": train_time,
                "test_scores": {},
                "train_scores": {} if return_train_score else None,
                "estimator": model if return_estimator else None
            }
            
            # 计算测试集分数
            for name, scorer in scorers.items():
                score = scorer(model, X_test, y_test)
                fold_results["test_scores"][name] = score
            
            # 计算训练集分数
            if return_train_score:
                for name, scorer in scorers.items():
                    score = scorer(model, X_train, y_train)
                    fold_results["train_scores"][name] = score
            
            return fold_results
        
        # 并行执行任务
        all_fold_results = []
        
        if self.n_jobs > 1:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                for fold_result in executor.map(train_and_evaluate, tasks):
                    all_fold_results.append(fold_result)
        else:
            for task in tasks:
                fold_result = train_and_evaluate(task)
                all_fold_results.append(fold_result)
        
        # 整理结果
        for fold_result in sorted(all_fold_results, key=lambda x: x["fold"]):
            fold = fold_result["fold"]
            
            # 添加测试集分数
            for name, score in fold_result["test_scores"].items():
                results[f"test_{name}"][fold] = score
            
            # 添加训练集分数
            if return_train_score:
                for name, score in fold_result["train_scores"].items():
                    results[f"train_{name}"][fold] = score
            
            # 添加估计器
            if return_estimator:
                results["estimator"].append(fold_result["estimator"])
        
        # 计算平均分数和标准差
        for key in results:
            if key != "estimator":
                results[f"mean_{key}"] = np.mean(results[key])
                results[f"std_{key}"] = np.std(results[key])
        
        # 显示结果
        if verbose:
            for name in scorers:
                mean_test_score = results[f"mean_test_{name}"]
                std_test_score = results[f"std_test_{name}"]
                logger.info(f"{name}: {mean_test_score:.4f} ± {std_test_score:.4f}")
        
        return results
    
    def nested_cross_validate(
        self,
        model_class: Any,
        param_grid: Dict[str, List],
        X: np.ndarray,
        y: np.ndarray,
        outer_cv_type: str = 'kfold',
        inner_cv_type: str = 'kfold',
        inner_n_splits: int = 3,
        scoring: str = 'accuracy',
        fit_params: Optional[Dict] = None,
        verbose: bool = True
    ) -> Dict:
        """
        嵌套交叉验证
        
        Args:
            model_class: 模型类
            param_grid: 参数网格
            X: 特征数据
            y: 目标变量
            outer_cv_type: 外层交叉验证类型
            inner_cv_type: 内层交叉验证类型
            inner_n_splits: 内层分割数
            scoring: 评分方法
            fit_params: 拟合参数
            verbose: 是否显示进度
            
        Returns:
            Dict: 嵌套交叉验证结果
        """
        from sklearn.model_selection import GridSearchCV
        
        # 获取外层分割器
        outer_splitter = self._get_splitter(outer_cv_type, y)
        
        # 准备结果
        results = {
            "test_score": np.zeros(self.n_splits),
            "best_params": [],
            "best_estimators": []
        }
        
        # 遍历外层折
        for fold, (train_idx, test_idx) in enumerate(outer_splitter.split(X, y)):
            if verbose:
                logger.info(f"Outer fold {fold+1}/{self.n_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 创建内层交叉验证器
            inner_cv = self._get_splitter(inner_cv_type, y_train)
            
            # 创建网格搜索
            grid_search = GridSearch
# RQA2025 特征层功能增强分析报告（续）

## 2. 功能分析（续）

### 2.2 特征质量评估（续）

#### 2.2.1 特征重要性评估（续）

**实现建议**（续）：

```python
    def evaluate_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = 'classification',
        n_neighbors: int = 3
    ) -> pd.DataFrame:
        """
        基于互信息的特征重要性评估
        
        Args:
            X: 特征数据
            y: 目标变量
            task_type: 任务类型，'classification'或'regression'
            n_neighbors: 最近邻数量
            
        Returns:
            pd.DataFrame: 特征重要性
        """
        # 选择互信息函数
        if task_type == 'classification':
            mi_func = mutual_info_classif
        else:
            mi_func = mutual_info_regression
        
        # 计算互信息
        importance = mi_func(
            X, y,
            n_neighbors=n_neighbors,
            random_state=self.random_state
        )
        
        # 创建结果DataFrame
        result = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        })
        
        return result.sort_values('importance', ascending=False)
    
    def evaluate_permutation_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model,
        n_repeats: int = 10,
        scoring: Optional[str] = None
    ) -> pd.DataFrame:
        """
        基于排列重要性的特征重要性评估
        
        Args:
            X: 特征数据
            y: 目标变量
            model: 已训练的模型
            n_repeats: 重复次数
            scoring: 评分方法
            
        Returns:
            pd.DataFrame: 特征重要性
        """
        # 计算排列重要性
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring=scoring
        )
        
        # 创建结果DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        })
        
        return importance_df.sort_values('importance_mean', ascending=False)
    
    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        importance_col: str = 'importance',
        top_n: int = 20,
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """
        绘制特征重要性图
        
        Args:
            importance_df: 特征重要性DataFrame
            importance_col: 重要性列名
            top_n: 显示前N个特征
            figsize: 图形大小
        """
        # 获取前N个特征
        top_features = importance_df.head(top_n)
        
        # 创建图形
        plt.figure(figsize=figsize)
        sns.barplot(
            x=importance_col,
            y='feature',
            data=top_features
        )
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        plt.show()
```

#### 2.2.2 特征相关性分析

**现状分析**：
缺乏对特征间相关性的系统分析，可能导致特征冗余。

**实现建议**：
实现一个 `FeatureCorrelationAnalyzer` 类，提供特征相关性分析功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)

class FeatureCorrelationAnalyzer:
    """特征相关性分析器"""
    
    def __init__(self, threshold: float = 0.7):
        """
        初始化特征相关性分析器
        
        Args:
            threshold: 相关性阈值
        """
        self.threshold = threshold
    
    def compute_correlation_matrix(
        self,
        X: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        计算相关性矩阵
        
        Args:
            X: 特征数据
            method: 相关性计算方法，'pearson'或'spearman'
            
        Returns:
            pd.DataFrame: 相关性矩阵
        """
        return X.corr(method=method)
    
    def find_correlated_features(
        self,
        X: pd.DataFrame,
        method: str = 'pearson'
    ) -> List[Tuple[str, str, float]]:
        """
        查找高相关性特征对
        
        Args:
            X: 特征数据
            method: 相关性计算方法，'pearson'或'spearman'
            
        Returns:
            List[Tuple[str, str, float]]: 高相关性特征对列表
        """
        corr_matrix = self.compute_correlation_matrix(X, method)
        
        # 查找高相关性特征对
        correlated_features = []
        
        # 获取上三角矩阵的索引
        upper_tri = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        
        # 提取上三角矩阵的值
        upper_corr = corr_matrix.where(upper_tri)
        
        # 查找高相关性特征对
        for col in upper_corr.columns:
            for idx, value in upper_corr[col].items():
                if abs(value) > self.threshold:
                    correlated_features.append((idx, col, value))
        
        # 按相关性绝对值排序
        correlated_features.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return correlated_features
    
    def plot_correlation_matrix(
        self,
        X: pd.DataFrame,
        method: str = 'pearson',
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = 'coolwarm',
        annot: bool = True
    ) -> None:
        """
        绘制相关性矩阵热图
        
        Args:
            X: 特征数据
            method: 相关性计算方法，'pearson'或'spearman'
            figsize: 图形大小
            cmap: 颜色映射
            annot: 是否显示注释
        """
        corr_matrix = self.compute_correlation_matrix(X, method)
        
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            annot=annot,
            fmt='.2f',
            square=True,
            linewidths=.5,
            cbar_kws={'shrink': .5}
        )
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        plt.show()
    
    def get_feature_clusters(
        self,
        X: pd.DataFrame,
        method: str = 'pearson',
        linkage_method: str = 'average'
    ) -> Dict[int, List[str]]:
        """
        获取特征聚类
        
        Args:
            X: 特征数据
            method: 相关性计算方法，'pearson'或'spearman'
            linkage_method: 层次聚类连接方法
            
        Returns:
            Dict[int, List[str]]: 特征聚类
        """
        from scipy.cluster import hierarchy
        
        corr_matrix = self.compute_correlation_matrix(X, method)
        
        # 计算距离矩阵
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # 执行层次聚类
        Z = hierarchy.linkage(
            hierarchy.distance.squareform(distance_matrix),
            method=linkage_method
        )
        
        # 根据相关性阈值确定聚类数
        clusters = hierarchy.fcluster(
            Z,
            t=1 - self.threshold,
            criterion='distance'
        )
        
        # 将特征分组到聚类中
        feature_clusters = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in feature_clusters:
                feature_clusters[cluster_id] = []
            feature_clusters[cluster_id].append(X.columns[i])
        
        return feature_clusters
    
    def suggest_feature_removal(
        self,
        X: pd.DataFrame,
        importance_df: Optional[pd.DataFrame] = None,
        method: str = 'pearson'
    ) -> List[str]:
        """
        建议移除的特征
        
        Args:
            X: 特征数据
            importance_df: 特征重要性DataFrame
            method: 相关性计算方法，'pearson'或'spearman'
            
        Returns:
            List[str]: 建议移除的特征列表
        """
        correlated_features = self.find_correlated_features(X, method)
        
        # 如果提供了特征重要性，则使用它来决定保留哪些特征
        if importance_df is not None:
            importance_dict = dict(zip(
                importance_df['feature'],
                importance_df['importance']
            ))
            
            # 对于每对高相关性特征，保留重要性更高的特征
            features_to_remove = set()
            for feat1, feat2, _ in correlated_features:
                imp1 = importance_dict.get(feat1, 0)
                imp2 = importance_dict.get(feat2, 0)
                
                if imp1 >= imp2:
                    features_to_remove.add(feat2)
                else:
                    features_to_remove.add(feat1)
        else:
            # 如果没有提供特征重要性，则保留第一个特征
            features_to_remove = set()
            for _, feat2, _ in correlated_features:
                features_to_remove.add(feat2)
        
        return list(features_to_remove)
```

### 2.3 特征存储和复用

#### 2.3.1 特征存储

**现状分析**：
特征计算结果未有效存储，导致重复计算，浪费计算资源。

**实现建议**：
实现一个 `FeatureStore` 类，提供特征存储和检索功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import os
import json
import hashlib
import logging
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

class FeatureStore:
    """特征存储"""
    
    def __init__(
        self,
        store_dir: str = './feature_store',
        metadata_file: str = 'metadata.json'
    ):
        """
        初始化特征存储
        
        Args:
            store_dir: 存储目录
            metadata_file: 元数据文件名
        """
        self.store_dir = store_dir
        self.metadata_file = os.path.join(store_dir, metadata_file)
        
        # 创建存储目录
        os.makedirs(store_dir, exist_ok=True)
        
        # 加载元数据
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """
        加载元数据
        
        Returns:
            Dict: 元数据
        """
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                return {'features': {}}
        else:
            return {'features': {}}
    
    def _save_metadata(self) -> None:
        """保存元数据"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _generate_feature_id(
        self,
        name: str,
        params: Dict,
        data_hash: str
    ) -> str:
        """
        生成特征ID
        
        Args:
            name: 特征名称
            params: 特征参数
            data_hash: 数据哈希
            
        Returns:
            str: 特征ID
        """
        # 创建特征签名
        signature = {
            'name': name,
            'params': params,
            'data_hash': data_hash
        }
        
        # 计算哈希
        signature_str = json.dumps(signature, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()
    
    def _generate_data_hash(self, data: pd.DataFrame) -> str:
        """
        生成数据哈希
        
        Args:
            data: 输入数据
            
        Returns:
            str: 数据哈希
        """
        # 使用数据的形状和列名生成哈希
        data_info = {
            'shape': data.shape,
            'columns': list(data.columns),
            'index_start': str(data.index[0]),
            'index_end': str(data.index[-1])
        }
        
        # 计算哈希
        data_info_str = json.dumps(data_info, sort_keys=True)
        return hashlib.md5(data_info_str.encode()).hexdigest()
    
    def store_feature(
        self,
        name: str,
        feature_data: pd.DataFrame,
        source_data: pd.DataFrame,
        params: Dict,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        存储特征
        
        Args:
            name: 特征名称
            feature_data: 特征数据
            source_data: 源数据
            params
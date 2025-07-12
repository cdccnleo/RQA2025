# RQA2025 特征层功能增强分析报告（续3）

## 2. 功能分析（续）

### 2.4 特征工程自动化（续）

#### 2.4.1 自动特征生成（续）

**实现建议**（续）：

```python
        # 添加季节性特征
        date_df['sin_month'] = np.sin(2 * np.pi * df[date_column].dt.month / 12)
        date_df['cos_month'] = np.cos(2 * np.pi * df[date_column].dt.month / 12)
        date_df['sin_day'] = np.sin(2 * np.pi * df[date_column].dt.day / 30)
        date_df['cos_day'] = np.cos(2 * np.pi * df[date_column].dt.day / 30)
        date_df['sin_dayofweek'] = np.sin(2 * np.pi * df[date_column].dt.dayofweek / 7)
        date_df['cos_dayofweek'] = np.cos(2 * np.pi * df[date_column].dt.dayofweek / 7)
        
        return date_df
    
    def generate_lag_features(
        self,
        data: pd.DataFrame,
        target_columns: List[str],
        lag_periods: List[int],
        group_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        生成滞后特征
        
        Args:
            data: 输入数据
            target_columns: 目标列名列表
            lag_periods: 滞后周期列表
            group_column: 分组列名
            
        Returns:
            pd.DataFrame: 特征数据
        """
        df = data.copy()
        lag_df = pd.DataFrame(index=df.index)
        
        for col in target_columns:
            for lag in lag_periods:
                if group_column:
                    # 按组生成滞后特征
                    lag_values = df.groupby(group_column)[col].shift(lag)
                    lag_df[f'{col}_lag_{lag}'] = lag_values
                else:
                    # 生成全局滞后特征
                    lag_df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return lag_df
    
    def generate_rolling_features(
        self,
        data: pd.DataFrame,
        target_columns: List[str],
        window_sizes: List[int],
        functions: List[str] = ['mean', 'std', 'min', 'max'],
        group_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        生成滚动特征
        
        Args:
            data: 输入数据
            target_columns: 目标列名列表
            window_sizes: 窗口大小列表
            functions: 聚合函数列表
            group_column: 分组列名
            
        Returns:
            pd.DataFrame: 特征数据
        """
        df = data.copy()
        rolling_df = pd.DataFrame(index=df.index)
        
        for col in target_columns:
            for window in window_sizes:
                for func in functions:
                    if group_column:
                        # 按组生成滚动特征
                        grouped = df.groupby(group_column)[col]
                        rolling_values = getattr(grouped.rolling(window), func)()
                        rolling_df[f'{col}_rolling_{window}_{func}'] = rolling_values
                    else:
                        # 生成全局滚动特征
                        rolling_values = getattr(df[col].rolling(window), func)()
                        rolling_df[f'{col}_rolling_{window}_{func}'] = rolling_values
        
        return rolling_df
    
    def generate_interaction_features(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        interaction_types: List[str] = ['multiply', 'divide', 'add', 'subtract']
    ) -> pd.DataFrame:
        """
        生成交互特征
        
        Args:
            data: 输入数据
            feature_columns: 特征列名列表
            interaction_types: 交互类型列表
            
        Returns:
            pd.DataFrame: 特征数据
        """
        df = data.copy()
        interaction_df = pd.DataFrame(index=df.index)
        
        # 生成所有可能的列对
        from itertools import combinations
        column_pairs = list(combinations(feature_columns, 2))
        
        for col1, col2 in column_pairs:
            if 'multiply' in interaction_types:
                interaction_df[f'{col1}_mul_{col2}'] = df[col1] * df[col2]
            
            if 'divide' in interaction_types:
                # 避免除以零
                interaction_df[f'{col1}_div_{col2}'] = df[col1] / df[col2].replace(0, np.nan)
                interaction_df[f'{col2}_div_{col1}'] = df[col2] / df[col1].replace(0, np.nan)
            
            if 'add' in interaction_types:
                interaction_df[f'{col1}_add_{col2}'] = df[col1] + df[col2]
            
            if 'subtract' in interaction_types:
                interaction_df[f'{col1}_sub_{col2}'] = df[col1] - df[col2]
                interaction_df[f'{col2}_sub_{col1}'] = df[col2] - df[col1]
        
        return interaction_df
    
    def generate_polynomial_features(
        self,
        data: pd.DataFrame,
        feature_columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """
        生成多项式特征
        
        Args:
            data: 输入数据
            feature_columns: 特征列名列表
            degree: 多项式阶数
            
        Returns:
            pd.DataFrame: 特征数据
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        df = data.copy()
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        
        # 提取特征列
        X = df[feature_columns].values
        
        # 生成多项式特征
        poly_features = poly.fit_transform(X)
        
        # 创建特征名称
        feature_names = poly.get_feature_names_out(feature_columns)
        
        # 创建DataFrame
        poly_df = pd.DataFrame(
            poly_features,
            columns=feature_names,
            index=df.index
        )
        
        # 移除原始特征
        for col in feature_columns:
            if col in poly_df.columns:
                poly_df = poly_df.drop(col, axis=1)
        
        return poly_df
    
    def auto_generate_features(
        self,
        data: pd.DataFrame,
        config: Dict
    ) -> pd.DataFrame:
        """
        自动生成特征
        
        Args:
            data: 输入数据
            config: 配置字典
            
        Returns:
            pd.DataFrame: 特征数据
        """
        df = data.copy()
        result = pd.DataFrame(index=df.index)
        
        # 生成时间特征
        if 'time_features' in config:
            for time_config in config['time_features']:
                date_column = time_config['date_column']
                time_features = self.generate_time_features(df, date_column)
                result = pd.concat([result, time_features], axis=1)
        
        # 生成滞后特征
        if 'lag_features' in config:
            for lag_config in config['lag_features']:
                target_columns = lag_config['target_columns']
                lag_periods = lag_config['lag_periods']
                group_column = lag_config.get('group_column')
                
                lag_features = self.generate_lag_features(
                    df, target_columns, lag_periods, group_column
                )
                result = pd.concat([result, lag_features], axis=1)
        
        # 生成滚动特征
        if 'rolling_features' in config:
            for rolling_config in config['rolling_features']:
                target_columns = rolling_config['target_columns']
                window_sizes = rolling_config['window_sizes']
                functions = rolling_config.get('functions', ['mean', 'std', 'min', 'max'])
                group_column = rolling_config.get('group_column')
                
                rolling_features = self.generate_rolling_features(
                    df, target_columns, window_sizes, functions, group_column
                )
                result = pd.concat([result, rolling_features], axis=1)
        
        # 生成交互特征
        if 'interaction_features' in config:
            for interaction_config in config['interaction_features']:
                feature_columns = interaction_config['feature_columns']
                interaction_types = interaction_config.get('interaction_types', ['multiply', 'divide', 'add', 'subtract'])
                
                interaction_features = self.generate_interaction_features(
                    df, feature_columns, interaction_types
                )
                result = pd.concat([result, interaction_features], axis=1)
        
        # 生成多项式特征
        if 'polynomial_features' in config:
            for poly_config in config['polynomial_features']:
                feature_columns = poly_config['feature_columns']
                degree = poly_config.get('degree', 2)
                
                poly_features = self.generate_polynomial_features(
                    df, feature_columns, degree
                )
                result = pd.concat([result, poly_features], axis=1)
        
        return result
```

#### 2.4.2 特征选择自动化

**现状分析**：
特征选择过程较为手动，缺乏自动化工具。

**实现建议**：
实现一个 `AutoFeatureSelector` 类，提供自动特征选择功能：

```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.decomposition import PCA
from boruta import BorutaPy

logger = logging.getLogger(__name__)

class AutoFeatureSelector:
    """自动特征选择器"""
    
    def __init__(self, random_state: int = 42):
        """
        初始化自动特征选择器
        
        Args:
            random_state: 随机种子
        """
        self.random_state = random_state
    
    def select_k_best(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        k: int = 10,
        task_type: str = 'classification'
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        选择K个最佳特征
        
        Args:
            X: 特征数据
            y: 目标变量
            k: 选择的特征数量
            task_type: 任务类型，'classification'或'regression'
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: 选择的特征数据和特征名称列表
        """
        # 选择评分函数
        score_func = f_classif if task_type == 'classification' else f_regression
        
        # 创建选择器
        selector = SelectKBest(score_func=score_func, k=k)
        
        # 拟合选择器
        selector.fit(X, y)
        
        # 获取选择的特征索引
        selected_indices = selector.get_support(indices=True)
        
        # 获取选择的特征名称
        selected_features = [X.columns[i] for i in selected_indices]
        
        # 转换数据
        X_selected = selector.transform(X)
        
        # 创建DataFrame
        X_selected_df = pd.DataFrame(
            X_selected,
            columns=selected_features,
            index=X.index
        )
        
        return X_selected_df, selected_features
    
    def select_from_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = 'random_forest',
        task_type: str = 'classification',
        threshold: str = 'mean',
        max_features: Optional[int] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        基于模型的特征选择
        
        Args:
            X: 特征数据
            y: 目标变量
            model_type: 模型类型，'random_forest'或'lasso'或'logistic_regression'
            task_type: 任务类型，'classification'或'regression'
            threshold: 特征重要性阈值
            max_features: 最大特征数量
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: 选择的特征数据和特征名称列表
        """
        # 选择模型
        if model_type == 'random_forest':
            if task_type == 'classification':
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state
                )
        elif model_type == 'lasso':
            model = Lasso(alpha=0.01, random_state=self.random_state)
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 创建选择器
        selector = SelectFromModel(
            estimator=model,
            threshold=threshold,
            max_features=max_features
        )
        
        # 拟合选择器
        selector.fit(X, y)
        
        # 获取选择的特征索引
        selected_indices = selector.get_support(indices=True)
        
        # 获取选择的特征名称
        selected_features = [X.columns[i] for i in selected_indices]
        
        # 转换数据
        X_selected = selector.transform(X)
        
        # 创建DataFrame
        X_selected_df = pd.DataFrame(
            X_selected,
            columns=selected_features,
            index=X.index
        )
        
        return X_selected_df, selected_features
    
    def select_with_boruta(
        self,
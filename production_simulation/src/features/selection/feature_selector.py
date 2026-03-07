"""
特征选择器实现

提供多种特征选择方法：
- 方差阈值：移除低方差特征
- 相关性：移除高相关性特征
- 重要性：基于模型特征重要性
- 递归消除：递归特征消除
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """特征选择方法"""
    VARIANCE = "variance"          # 方差阈值
    CORRELATION = "correlation"    # 相关性过滤
    IMPORTANCE = "importance"      # 特征重要性
    RFE = "rfe"                    # 递归特征消除
    MUTUAL_INFO = "mutual_info"    # 互信息
    NONE = "none"                  # 不选择


class FeatureSelector:
    """特征选择器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化特征选择器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        
    def select_features(
        self,
        data: pd.DataFrame,
        method: SelectionMethod = SelectionMethod.VARIANCE,
        target: Optional[pd.Series] = None,
        max_features: Optional[int] = None,
        min_importance: float = 0.01,
        **kwargs
    ) -> pd.DataFrame:
        """
        选择特征
        
        Args:
            data: 特征数据
            method: 选择方法
            target: 目标变量（某些方法需要）
            max_features: 最大特征数
            min_importance: 最小重要性阈值
            **kwargs: 其他参数
            
        Returns:
            选择后的特征数据
        """
        if data.empty:
            logger.warning("输入数据为空，返回空数据")
            return data
            
        # 排除非数值列
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            logger.warning("没有数值型特征，返回原始数据")
            return data
            
        logger.info(f"开始特征选择，方法: {method.value}, 原始特征数: {len(numeric_data.columns)}")
        
        if method == SelectionMethod.VARIANCE:
            selected_data = self._variance_threshold(numeric_data, **kwargs)
        elif method == SelectionMethod.CORRELATION:
            selected_data = self._correlation_filter(numeric_data, **kwargs)
        elif method == SelectionMethod.IMPORTANCE:
            if target is None:
                logger.warning("特征重要性方法需要目标变量，跳过选择")
                selected_data = numeric_data
            else:
                selected_data = self._importance_selection(numeric_data, target, min_importance, **kwargs)
        elif method == SelectionMethod.MUTUAL_INFO:
            if target is None:
                logger.warning("互信息方法需要目标变量，跳过选择")
                selected_data = numeric_data
            else:
                selected_data = self._mutual_info_selection(numeric_data, target, **kwargs)
        elif method == SelectionMethod.NONE:
            selected_data = numeric_data
        else:
            logger.warning(f"未知的选择方法: {method.value}，返回原始数据")
            selected_data = numeric_data
            
        # 限制最大特征数
        if max_features and len(selected_data.columns) > max_features:
            logger.info(f"限制特征数为: {max_features}")
            selected_data = selected_data.iloc[:, :max_features]
            
        self.selected_features = list(selected_data.columns)
        logger.info(f"特征选择完成，保留特征数: {len(self.selected_features)}")
        
        # 保留原始数据中的非数值列
        non_numeric_cols = [col for col in data.columns if col not in numeric_data.columns]
        if non_numeric_cols:
            selected_data = pd.concat([selected_data, data[non_numeric_cols]], axis=1)
            
        return selected_data
        
    def _variance_threshold(
        self,
        data: pd.DataFrame,
        threshold: float = 0.01,
        **kwargs
    ) -> pd.DataFrame:
        """
        方差阈值选择
        
        移除方差低于阈值的特征
        
        Args:
            data: 特征数据
            threshold: 方差阈值
            
        Returns:
            选择后的数据
        """
        # 计算每列的方差
        variances = data.var()
        
        # 选择方差大于阈值的特征
        selected_cols = variances[variances > threshold].index.tolist()
        
        if not selected_cols:
            logger.warning(f"方差阈值 {threshold} 过滤后没有剩余特征，返回原始数据")
            return data
            
        logger.info(f"方差阈值选择: 保留 {len(selected_cols)}/{len(data.columns)} 个特征")
        
        # 记录特征得分
        for col in data.columns:
            self.feature_scores[col] = variances[col]
            
        return data[selected_cols]
        
    def _correlation_filter(
        self,
        data: pd.DataFrame,
        threshold: float = 0.95,
        **kwargs
    ) -> pd.DataFrame:
        """
        相关性过滤
        
        移除高度相关的特征（保留其中一个）
        
        Args:
            data: 特征数据
            threshold: 相关性阈值
            
        Returns:
            选择后的数据
        """
        # 计算相关性矩阵
        corr_matrix = data.corr().abs()
        
        # 获取上三角矩阵
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # 找出高度相关的特征对
        to_drop = []
        for column in upper.columns:
            if any(upper[column] > threshold):
                to_drop.append(column)
                
        # 保留的特征
        selected_cols = [col for col in data.columns if col not in to_drop]
        
        if not selected_cols:
            logger.warning(f"相关性过滤后没有剩余特征，返回原始数据")
            return data
            
        logger.info(f"相关性过滤: 移除 {len(to_drop)} 个高度相关特征，保留 {len(selected_cols)} 个")
        
        return data[selected_cols]
        
    def _importance_selection(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        min_importance: float = 0.01,
        **kwargs
    ) -> pd.DataFrame:
        """
        基于特征重要性的选择
        
        使用随机森林计算特征重要性
        
        Args:
            data: 特征数据
            target: 目标变量
            min_importance: 最小重要性阈值
            
        Returns:
            选择后的数据
        """
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            # 判断是分类还是回归问题
            if target.nunique() <= 10 or target.dtype == 'object':
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                
            # 训练模型
            model.fit(data, target)
            
            # 获取特征重要性
            importances = model.feature_importances_
            
            # 创建特征重要性字典
            self.feature_scores = dict(zip(data.columns, importances))
            
            # 选择重要性大于阈值的特征
            selected_cols = [col for col, imp in zip(data.columns, importances) 
                           if imp > min_importance]
            
            if not selected_cols:
                logger.warning(f"重要性阈值 {min_importance} 过滤后没有剩余特征，返回原始数据")
                return data
                
            logger.info(f"特征重要性选择: 保留 {len(selected_cols)}/{len(data.columns)} 个特征")
            
            return data[selected_cols]
            
        except ImportError:
            logger.warning("sklearn 未安装，跳过特征重要性选择")
            return data
        except Exception as e:
            logger.error(f"特征重要性选择失败: {e}")
            return data
            
    def _mutual_info_selection(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        k: int = 10,
        **kwargs
    ) -> pd.DataFrame:
        """
        基于互信息的特征选择
        
        选择与目标变量互信息最高的k个特征
        
        Args:
            data: 特征数据
            target: 目标变量
            k: 选择的特征数
            
        Returns:
            选择后的数据
        """
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            
            # 判断是分类还是回归问题
            if target.nunique() <= 10 or target.dtype == 'object':
                mi_scores = mutual_info_classif(data, target, random_state=42)
            else:
                mi_scores = mutual_info_regression(data, target, random_state=42)
                
            # 创建互信息字典
            self.feature_scores = dict(zip(data.columns, mi_scores))
            
            # 选择互信息最高的k个特征
            k = min(k, len(data.columns))
            top_k_indices = np.argsort(mi_scores)[-k:]
            selected_cols = [data.columns[i] for i in top_k_indices]
            
            logger.info(f"互信息选择: 保留 {len(selected_cols)}/{len(data.columns)} 个特征")
            
            return data[selected_cols]
            
        except ImportError:
            logger.warning("sklearn 未安装，跳过互信息选择")
            return data
        except Exception as e:
            logger.error(f"互信息选择失败: {e}")
            return data
            
    def get_feature_scores(self) -> Dict[str, float]:
        """
        获取特征得分
        
        Returns:
            特征得分字典
        """
        return self.feature_scores
        
    def get_selected_features(self) -> List[str]:
        """
        获取已选择的特征列表
        
        Returns:
            特征列表
        """
        return self.selected_features

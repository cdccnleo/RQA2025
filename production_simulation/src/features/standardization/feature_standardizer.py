"""
特征标准化器实现

提供多种特征标准化方法：
- Z-score标准化：均值为0，标准差为1
- Min-Max归一化：缩放到[0, 1]范围
- Robust标准化：基于中位数和四分位数，对异常值稳健
- Log变换：对数变换，处理偏态分布
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class StandardizationMethod(Enum):
    """特征标准化方法"""
    ZSCORE = "zscore"          # Z-score标准化
    MINMAX = "minmax"          # Min-Max归一化
    ROBUST = "robust"          # Robust标准化
    LOG = "log"                # Log变换
    NONE = "none"              # 不标准化


class FeatureStandardizer:
    """特征标准化器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化特征标准化器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.scaler_params: Dict[str, Dict[str, float]] = {}
        self.fitted = False
        
    def standardize_features(
        self,
        data: pd.DataFrame,
        method: StandardizationMethod = StandardizationMethod.ZSCORE,
        robust: bool = False,
        exclude_cols: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        标准化特征
        
        Args:
            data: 特征数据
            method: 标准化方法
            robust: 是否使用稳健方法（对异常值不敏感）
            exclude_cols: 排除的列（如目标变量、时间戳等）
            **kwargs: 其他参数
            
        Returns:
            标准化后的特征数据
        """
        if data.empty:
            logger.warning("输入数据为空，返回空数据")
            return data
            
        # 排除非数值列和指定列
        exclude_cols = exclude_cols or []
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
        
        if not cols_to_scale:
            logger.warning("没有需要标准化的数值型特征，返回原始数据")
            return data
            
        logger.info(f"开始特征标准化，方法: {method.value}, 特征数: {len(cols_to_scale)}")
        
        # 复制数据，避免修改原始数据
        result = data.copy()
        
        if method == StandardizationMethod.ZSCORE:
            result = self._zscore_standardize(result, cols_to_scale, robust)
        elif method == StandardizationMethod.MINMAX:
            result = self._minmax_scale(result, cols_to_scale, robust)
        elif method == StandardizationMethod.ROBUST:
            result = self._robust_scale(result, cols_to_scale)
        elif method == StandardizationMethod.LOG:
            result = self._log_transform(result, cols_to_scale)
        elif method == StandardizationMethod.NONE:
            logger.info("跳过特征标准化")
        else:
            logger.warning(f"未知的标准化方法: {method.value}，返回原始数据")
            
        self.fitted = True
        logger.info(f"特征标准化完成")
        
        return result
        
    def _zscore_standardize(
        self,
        data: pd.DataFrame,
        cols: List[str],
        robust: bool = False
    ) -> pd.DataFrame:
        """
        Z-score标准化
        
        (x - mean) / std
        
        Args:
            data: 数据
            cols: 需要标准化的列
            robust: 是否使用中位数代替均值
            
        Returns:
            标准化后的数据
        """
        result = data.copy()
        
        for col in cols:
            if robust:
                # 使用中位数和中位数绝对偏差（MAD）
                median = result[col].median()
                mad = np.median(np.abs(result[col] - median))
                if mad != 0:
                    result[col] = (result[col] - median) / mad
                    self.scaler_params[col] = {'median': median, 'mad': mad}
                else:
                    logger.warning(f"列 {col} 的MAD为0，跳过标准化")
            else:
                # 使用均值和标准差
                mean = result[col].mean()
                std = result[col].std()
                if std != 0:
                    result[col] = (result[col] - mean) / std
                    self.scaler_params[col] = {'mean': mean, 'std': std}
                else:
                    logger.warning(f"列 {col} 的标准差为0，跳过标准化")
                    
        logger.info(f"Z-score标准化完成（robust={robust}）")
        return result
        
    def _minmax_scale(
        self,
        data: pd.DataFrame,
        cols: List[str],
        robust: bool = False
    ) -> pd.DataFrame:
        """
        Min-Max归一化
        
        (x - min) / (max - min)
        
        Args:
            data: 数据
            cols: 需要归一化的列
            robust: 是否使用百分位数代替min/max
            
        Returns:
            归一化后的数据
        """
        result = data.copy()
        
        for col in cols:
            if robust:
                # 使用1%和99%百分位数，减少异常值影响
                min_val = result[col].quantile(0.01)
                max_val = result[col].quantile(0.99)
            else:
                min_val = result[col].min()
                max_val = result[col].max()
                
            range_val = max_val - min_val
            if range_val != 0:
                result[col] = (result[col] - min_val) / range_val
                self.scaler_params[col] = {'min': min_val, 'max': max_val}
            else:
                logger.warning(f"列 {col} 的取值范围为0，跳过归一化")
                
        logger.info(f"Min-Max归一化完成（robust={robust}）")
        return result
        
    def _robust_scale(
        self,
        data: pd.DataFrame,
        cols: List[str]
    ) -> pd.DataFrame:
        """
        Robust标准化
        
        (x - median) / IQR
        
        Args:
            data: 数据
            cols: 需要标准化的列
            
        Returns:
            标准化后的数据
        """
        result = data.copy()
        
        for col in cols:
            median = result[col].median()
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            
            if iqr != 0:
                result[col] = (result[col] - median) / iqr
                self.scaler_params[col] = {'median': median, 'q1': q1, 'q3': q3, 'iqr': iqr}
            else:
                logger.warning(f"列 {col} 的IQR为0，跳过标准化")
                
        logger.info(f"Robust标准化完成")
        return result
        
    def _log_transform(
        self,
        data: pd.DataFrame,
        cols: List[str]
    ) -> pd.DataFrame:
        """
        Log变换
        
        log(1 + x)，处理偏态分布
        
        Args:
            data: 数据
            cols: 需要变换的列
            
        Returns:
            变换后的数据
        """
        result = data.copy()
        
        for col in cols:
            # 检查是否有负值
            min_val = result[col].min()
            if min_val < 0:
                logger.warning(f"列 {col} 包含负值，使用log1p变换前平移数据")
                shift = abs(min_val) + 1
                result[col] = np.log1p(result[col] + shift)
                self.scaler_params[col] = {'shift': shift}
            else:
                result[col] = np.log1p(result[col])
                
        logger.info(f"Log变换完成")
        return result
        
    def inverse_transform(
        self,
        data: pd.DataFrame,
        cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        逆变换，将标准化后的数据还原
        
        Args:
            data: 标准化后的数据
            cols: 需要逆变换的列，None表示所有已标准化的列
            
        Returns:
            还原后的数据
        """
        if not self.fitted:
            logger.warning("标准化器尚未拟合，无法逆变换")
            return data
            
        result = data.copy()
        cols = cols or list(self.scaler_params.keys())
        
        for col in cols:
            if col not in self.scaler_params:
                logger.warning(f"列 {col} 没有标准化参数，跳过逆变换")
                continue
                
            params = self.scaler_params[col]
            
            if 'mean' in params and 'std' in params:
                # Z-score逆变换
                result[col] = result[col] * params['std'] + params['mean']
            elif 'median' in params and 'mad' in params:
                # Robust Z-score逆变换
                result[col] = result[col] * params['mad'] + params['median']
            elif 'min' in params and 'max' in params:
                # Min-Max逆变换
                result[col] = result[col] * (params['max'] - params['min']) + params['min']
            elif 'median' in params and 'iqr' in params:
                # Robust标准化逆变换
                result[col] = result[col] * params['iqr'] + params['median']
            elif 'shift' in params:
                # Log变换逆变换
                result[col] = np.expm1(result[col]) - params['shift']
                
        logger.info(f"逆变换完成")
        return result
        
    def get_scaler_params(self) -> Dict[str, Dict[str, float]]:
        """
        获取标准化参数
        
        Returns:
            标准化参数字典
        """
        return self.scaler_params

"""通用工具函数集合"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Union
import numpy as np
import pandas as pd
import warnings
import joblib
from pathlib import Path
from .exception_utils import DataLoaderError

def validate_dates(start_date: str, end_date: str) -> tuple:
    """验证日期格式和范围

    Args:
        start_date: 开始日期字符串 (YYYY-MM-DD)
        end_date: 结束日期字符串 (YYYY-MM-DD)

    Returns:
        元组 (datetime, datetime): 验证后的日期对象

    Raises:
        ValueError: 如果日期格式无效或范围不合理
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"日期格式无效，请使用YYYY-MM-DD格式: {e}")

    if start > end:
        raise ValueError("开始日期不能晚于结束日期")

    return start, end

def fill_missing_values(data: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """填充数据中的缺失值

    Args:
        data: 包含缺失值的DataFrame
        method: 填充方法 ('ffill', 'bfill', 'mean', 'median')

    Returns:
        填充后的DataFrame
    """
    if method == 'ffill':
        return data.ffill()
    elif method == 'bfill':
        return data.bfill()
    elif method == 'mean':
        return data.fillna(data.mean())
    elif method == 'median':
        return data.fillna(data.median())
    else:
        raise ValueError(f"不支持的填充方法: {method}")

def convert_to_ordered_dict(input_dict: dict) -> dict:
    """将字典转换为按键排序的有序字典
    
    Args:
        input_dict: 要排序的字典
        
    Returns:
        按键排序后的新字典
    """
    from collections import OrderedDict
    
    def sort_dict(d):
        if not isinstance(d, dict):
            return d
        return OrderedDict(sorted((k, sort_dict(v)) for k, v in d.items()))
    
    return sort_dict(input_dict)

def get_dynamic_dates(days_back: int = 30, date_format: str = "%Y-%m-%d") -> tuple:
    """获取动态日期范围
    
    Args:
        days_back: 从当前日期回溯的天数，默认30天
        date_format: 返回日期的格式字符串，默认YYYY-MM-DD
        
    Returns:
        元组 (start_date, end_date): 格式化后的日期字符串
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    if date_format:
        return start_date.strftime(date_format), end_date.strftime(date_format)
    return start_date, end_date

def safe_divide(numerator: Union[np.ndarray, pd.Series, pd.DataFrame],
               denominator: Union[np.ndarray, pd.Series, pd.DataFrame],
               default: float = 0.0) -> Union[float, pd.Series, pd.DataFrame]:
    """安全除法运算，避免除以零错误
    
    Args:
        numerator: 分子(numpy数组或pandas Series/DataFrame)
        denominator: 分母(numpy数组或pandas Series/DataFrame)
        default: 当分母为零时的默认返回值，默认为0.0
        
    Returns:
        除法结果，类型与输入相同
        
    Raises:
        ValueError: 如果输入形状不匹配
    """
    if isinstance(numerator, (pd.Series, pd.DataFrame)) or isinstance(denominator, (pd.Series, pd.DataFrame)):
        # 确保输入都是pandas类型
        if not isinstance(numerator, (pd.Series, pd.DataFrame)):
            numerator = pd.Series(numerator) if isinstance(numerator, np.ndarray) else pd.Series([numerator])
        if not isinstance(denominator, (pd.Series, pd.DataFrame)):
            denominator = pd.Series(denominator) if isinstance(denominator, np.ndarray) else pd.Series([denominator])
        
        # 检查形状是否匹配
        if len(numerator) != len(denominator):
            raise ValueError("分子和分母的长度必须相同")
            
        # 使用pandas的除法，自动处理零除
        result = numerator / denominator
        result[denominator == 0] = default
        return result
    else:
        # numpy数组或标量处理
        numerator = np.asarray(numerator)
        denominator = np.asarray(denominator)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(numerator, denominator)
            result[~np.isfinite(result)] = default
        return result[0] if result.size == 1 else result

def calculate_volatility(prices: Union[np.ndarray, pd.Series, pd.DataFrame],
                        window: int = 20,
                        annualize: bool = True,
                        periods: int = 252) -> pd.Series:
    """计算价格序列的滚动波动率(标准差)
    
    Args:
        prices: 价格序列(numpy数组或pandas Series/DataFrame)
        window: 滚动窗口大小，默认为20
        annualize: 是否年化波动率，默认为True
        periods: 年化周期数(默认252个交易日)
        
    Returns:
        滚动波动率序列(pandas Series)
        
    Raises:
        ValueError: 如果输入数据无效
    """
    if isinstance(prices, pd.DataFrame):
        if len(prices.columns) != 1:
            raise ValueError("DataFrame应只包含一列价格数据")
        prices = prices.iloc[:, 0]
    elif isinstance(prices, np.ndarray):
        prices = pd.Series(prices)
    
    if len(prices) < 2:
        raise ValueError("价格序列至少需要2个数据点")
    
    if window > len(prices):
        window = len(prices)
    
    # 计算对数收益率
    returns = np.log(prices / prices.shift(1))
    
    # 计算滚动标准差
    volatility = returns.rolling(window=window).std()
    
    # 年化处理
    if annualize:
        volatility *= np.sqrt(periods)
        
    return volatility

def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """验证DataFrame是否有效
    
    Args:
        df: 要验证的DataFrame
        required_columns: 必需的列名列表
        
    Returns:
        bool: 如果DataFrame有效且包含所有必需列则返回True，否则返回False
    """
    if not isinstance(df, pd.DataFrame):
        return False
    if df.empty:
        return False
    if not all(col in df.columns for col in required_columns):
        return False
    return True

def validate_feature_compatibility(stock_code: str, 
                                 features: list, 
                                 model_path: Path) -> bool:
    """验证特征是否与模型兼容
    
    Args:
        stock_code: 股票代码
        features: 要验证的特征列表
        model_path: 包含特征元数据的模型目录路径
        
    Returns:
        bool: 如果所有特征都兼容则返回True，否则返回False
        
    Raises:
        DataLoaderError: 如果元数据文件损坏或格式无效
    """
    metadata_path = model_path / f"{stock_code}_feature_metadata.pkl"
    
    if not metadata_path.exists():
        warnings.warn(f"特征元数据文件不存在: {metadata_path}")
        return False
        
    try:
        metadata = joblib.load(metadata_path)
        if not isinstance(metadata, dict) or 'features' not in metadata:
            raise DataLoaderError(f"特征元数据文件损坏或格式无效: {metadata_path}")
            
        return all(feat in metadata['features'] for feat in features)
        
    except Exception as e:
        raise DataLoaderError(f"加载特征元数据失败: {str(e)}")

def normalize_data(data: dict, schema: dict) -> dict:
    """规范化数据字典，确保字段类型和格式符合预期
    
    Args:
        data: 原始数据字典
        schema: 字段规范字典，格式为{'field_name': {'type': type, 'default': value}}
        
    Returns:
        规范化后的数据字典
    """
    normalized = {}
    for field, config in schema.items():
        if field in data:
            try:
                normalized[field] = config['type'](data[field])
            except (ValueError, TypeError):
                normalized[field] = config['default']
        else:
            normalized[field] = config['default']
    return normalized

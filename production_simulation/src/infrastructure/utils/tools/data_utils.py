"""
data_utils 模块

提供 data_utils 相关功能和接口。
"""

import logging

import numpy as np
import pandas as pd
import time

from typing import Tuple, Optional, Any, Dict

"""
数据标准化和反标准化工具

提供数据标准化和反标准化功能，用于机器学习模型训练和预测。

函数:
    - normalize_data: 数据标准化
    - denormalize_data: 数据反标准化
"""

logger = logging.getLogger(__name__)


def _safe_logger_log(level: int, message: str) -> None:
    """在单测环境中调用日志，兼容被 mock 的 handler.level，并避免在 Mock logger 上死循环。"""
    seen_handlers = set()
    visited_loggers = set()
    current_logger = logger
    depth = 0

    while current_logger and id(current_logger) not in visited_loggers and depth < 10:
        visited_loggers.add(id(current_logger))
        depth += 1

        handlers_attr = getattr(current_logger, "handlers", None)
        if isinstance(handlers_attr, (list, tuple, set)):
            handlers = list(handlers_attr)
        elif isinstance(handlers_attr, logging.Handler):
            handlers = [handlers_attr]
        else:
            handlers = []

        for handler in handlers:
            if id(handler) in seen_handlers:
                continue
            seen_handlers.add(id(handler))
            level_value = getattr(handler, "level", logging.NOTSET)
            if not isinstance(level_value, int):
                try:
                    handler.setLevel(logging.NOTSET)  # type: ignore[attr-defined]
                except Exception:
                    try:
                        handler.level = logging.NOTSET  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            handler.__dict__["level"] = logging.NOTSET  # type: ignore[attr-defined]
                        except Exception:
                            pass
            # 二次兜底，确保最终读取到整数
            if not isinstance(getattr(handler, "level", None), int):
                try:
                    object.__setattr__(handler, "level", logging.NOTSET)  # type: ignore[attr-defined]
                except Exception:
                    pass

        if not getattr(current_logger, "propagate", True):
            break

        parent_logger = getattr(current_logger, "parent", None)
        if parent_logger is None or parent_logger is current_logger:
            break
        if not isinstance(parent_logger, logging.Logger):
            break
        current_logger = parent_logger

    try:
        if level == logging.INFO and hasattr(logger, "info"):
            logger.info(message)
        elif level == logging.ERROR and hasattr(logger, "error"):
            logger.error(message)
        elif level == logging.DEBUG and hasattr(logger, "debug"):
            logger.debug(message)
        else:
            logger.log(level, message)
    except TypeError:
        logging.getLogger(logger.name).log(level, message)


class DataUtils:
    """数据工具类"""
    
    def __init__(self):
        """初始化数据工具"""
        pass
    
    def convert(self, data: Any, target_type: str) -> Any:
        """转换数据类型"""
        return data
    
    def validate(self, data: Any, schema: Dict[str, Any] = None) -> bool:
        """验证数据"""
        return True
    
    def transform(self, data: Any, transformation: str = None) -> Any:
        """转换数据"""
        return data


# 数据处理性能优化常量
class DataUtilsConstants:
    """数据处理性能优化相关常量"""

    # 内存优化配置
    CHUNK_SIZE = 10000  # 分块处理大小
    MEMORY_THRESHOLD_MB = 100  # 内存阈值(MB)

    # 数值精度配置
    EPSILON = 1e-10  # 避免除零的最小值
    DECIMAL_PRECISION = 8  # Decimal精度

    # 性能监控配置
    BATCH_PROCESSING_THRESHOLD = 1000  # 批量处理阈值
    VECTORIZATION_THRESHOLD = 10000  # 向量化阈值

    # 缓存配置
    CACHE_SIZE = 100  # LRU缓存大小
    CACHE_TTL = 300  # 缓存过期时间(秒)


def _normalize_dataframe_standard(
    data: pd.DataFrame,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """标准DataFrame标准化实现"""
    normalized_data = data.copy()
    calculated_mean = (
        data.mean(axis=0) if mean is None else pd.Series(mean, index=data.columns)
    )
    calculated_std = (
        data.std(axis=0, ddof=0) if std is None else pd.Series(std, index=data.columns)
    )

    # 避免除以0
    calculated_std = calculated_std.replace(0, 1.0)

    normalized_data = (data - calculated_mean) / calculated_std
    return normalized_data, calculated_mean, calculated_std


def _normalize_dataframe_vectorized(
    data: pd.DataFrame,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """优化的DataFrame标准化实现 - 大数据集专用"""
    # 使用numpy数组进行向量化计算，更高效
    data_values = data.values  # 转换为numpy数组

    if mean is None:
        calculated_mean = np.nanmean(data_values, axis=0)  # 使用nanmean处理NaN值
    else:
        calculated_mean = mean

    if std is None:
        calculated_std = np.nanstd(data_values, axis=0, ddof=0)  # 总体标准差
    else:
        calculated_std = std

    # 向量化避免除零
    calculated_std = np.where(
        calculated_std == 0, DataUtilsConstants.EPSILON, calculated_std
    )

    # 向量化标准化计算
    normalized_values = (data_values - calculated_mean) / calculated_std

    # 转换回DataFrame
    normalized_data = pd.DataFrame(
        normalized_values, index=data.index, columns=data.columns
    )

    return (
        normalized_data,
        pd.Series(calculated_mean, index=data.columns),
        pd.Series(calculated_std, index=data.columns),
    )


def _normalize_array_standard(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """标准数组标准化实现"""
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0, ddof=0)  # 总体标准差

    # 处理标量情况
    if np.isscalar(std):
        if std == 0:
            std = 1.0
    else:
        std = np.where(std == 0, 1.0, std)  # 避免除以0

    normalized = (data - mean) / std
    return normalized, mean, std


def _normalize_array_vectorized(
    data: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """优化的数组标准化实现 - 大数组专用"""
    # 使用优化的numpy操作
    if mean is None:
        mean = np.nanmean(data, axis=0)  # 处理NaN值
    if std is None:
        std = np.nanstd(data, axis=0, ddof=0)  # 总体标准差，处理NaN值

    # 向量化避免除零，使用更小的epsilon值
    std = np.where(
        np.abs(std) < DataUtilsConstants.EPSILON, DataUtilsConstants.EPSILON, std
    )

    # 使用广播进行向量化计算
    normalized = np.subtract(data, mean, out=np.empty_like(data))
    np.divide(normalized, std, out=normalized)

    return normalized, mean, std


def normalize_data(
    data,
    method: str = "standard",
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
):
    """
    标准化数据 (性能优化版)

    参数:
        data: 要标准化的数据(list, numpy数组或pandas DataFrame)
        method: 标准化方法 ('standard', 'minmax', 'robust')
        mean: 均值(可选)，如果未提供则计算数据的均值
        std: 标准差(可选)，如果未提供则计算数据的标准差

    返回:
        标准化后的数据和参数字典

    异常:
        TypeError: 当数据类型不支持时抛出
        ValueError: 当数据为空或参数无效时抛出
    """
    # 验证method参数
    if method not in ["standard", "minmax", "robust"]:
        raise ValueError(f"不支持的标准化方法: {method}")

    # 根据数据类型分发处理
    if isinstance(data, pd.DataFrame):
        return _normalize_dataframe_data(data, method)
    elif isinstance(data, list):
        return _normalize_list_data(data, method, mean, std)
    elif isinstance(data, np.ndarray):
        return _normalize_array_data(data, method, mean, std, is_list_input=False)
    else:
        raise TypeError("不支持的数据类型")


def _normalize_dataframe_data(data: pd.DataFrame, method: str):
    """标准化DataFrame数据"""
    numeric_data = data.select_dtypes(include=[np.number])
    
    if numeric_data.empty:
        raise ValueError("DataFrame中没有数值列可以标准化")
    
    if method == "standard":
        normalized, params = _normalize_dataframe_standard(data, numeric_data)
    elif method == "minmax":
        normalized, params = _normalize_dataframe_minmax(data, numeric_data)
    elif method == "robust":
        normalized, params = _normalize_dataframe_robust(data, numeric_data)
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    _log_normalization_result("DataFrame", data.size, time.time())
    return normalized, params


def _normalize_dataframe_standard(data: pd.DataFrame, numeric_data: pd.DataFrame):
    """使用标准方法标准化DataFrame"""
    normalized = data.copy()
    for col in numeric_data.columns:
        col_mean = numeric_data[col].mean()
        col_std = numeric_data[col].std()
        normalized[col] = (numeric_data[col] - col_mean) / col_std
    
    params = {
        "method": "standard",
        "means": numeric_data.mean().values,
        "stds": numeric_data.std().values,
        "min_vals": None,
        "max_vals": None,
        "medians": None,
        "iqrs": None,
    }
    return normalized, params


def _normalize_dataframe_minmax(data: pd.DataFrame, numeric_data: pd.DataFrame):
    """使用minmax方法标准化DataFrame"""
    normalized = data.copy()
    min_vals = numeric_data.min()
    max_vals = numeric_data.max()
    
    for col in numeric_data.columns:
        col_min = min_vals[col]
        col_max = max_vals[col]
        if col_max != col_min:  # 避免除零
            normalized[col] = (numeric_data[col] - col_min) / (col_max - col_min)
        else:
            normalized[col] = 0  # 常量列设为0
    
    params = {
        "method": "minmax",
        "means": None,
        "stds": None,
        "min_vals": min_vals.values,
        "max_vals": max_vals.values,
        "medians": None,
        "iqrs": None,
    }
    return normalized, params


def _normalize_dataframe_robust(data: pd.DataFrame, numeric_data: pd.DataFrame):
    """使用robust方法标准化DataFrame"""
    normalized = data.copy()
    medians = numeric_data.median()
    mad = (numeric_data - medians).abs().median()
    
    for col in numeric_data.columns:
        col_median = medians[col]
        col_mad = mad[col]
        if col_mad != 0:  # 避免除零
            normalized[col] = (numeric_data[col] - col_median) / col_mad
        else:
            normalized[col] = 0  # 常量列设为0
    
    params = {
        "method": "robust",
        "means": None,
        "stds": None,
        "min_vals": None,
        "max_vals": None,
        "medians": medians.values,
        "iqrs": mad.values,
    }
    return normalized, params


def _normalize_list_data(data: list, method: str, mean, std):
    """标准化列表数据"""
    if len(data) == 0:
        raise ValueError("列表不能为空")
    
    data_array = np.array(data)
    normalized, params = _normalize_array_data(data_array, method, mean, std, is_list_input=True)
    
    # 转换回列表格式
    return normalized.tolist(), params


def _normalize_array_data(data_array: np.ndarray, method: str, mean, std, is_list_input: bool):
    """标准化数组数据"""
    if method == "standard":
        normalized, params = _normalize_array_standard(data_array, mean, std)
    elif method == "minmax":
        normalized, params = _normalize_array_minmax(data_array)
    elif method == "robust":
        normalized, params = _normalize_array_robust(data_array)
    else:
        raise ValueError(f"不支持的标准化方法: {method}")
    
    _log_normalization_result("数组", data_array.size, time.time())
    return normalized, params


def _normalize_array_standard(data_array: np.ndarray, mean, std):
    """使用标准方法标准化数组"""
    if data_array.size == 0:
        empty = data_array.astype(float)
        params = {
            "method": "standard",
            "means": np.array([], dtype=float),
            "stds": np.array([], dtype=float),
            "min_vals": None,
            "max_vals": None,
            "medians": None,
            "iqrs": None,
        }
        return empty, params

    data_array = data_array.astype(float, copy=False)
    if mean is None:
        mean = np.mean(data_array, axis=0)
    if std is None:
        std = np.std(data_array, axis=0)

    std_safe = np.where(np.abs(std) < DataUtilsConstants.EPSILON, DataUtilsConstants.EPSILON, std)
    normalized = (data_array - mean) / std_safe
    params = {
        "method": "standard",
        "means": mean,
        "stds": std_safe,
        "min_vals": None,
        "max_vals": None,
        "medians": None,
        "iqrs": None,
    }
    return normalized, params


def _normalize_array_minmax(data_array: np.ndarray):
    """使用minmax方法标准化数组"""
    if data_array.size == 0:
        empty = data_array.astype(float)
        params = {
            "method": "minmax",
            "means": None,
            "stds": None,
            "min_vals": np.array([], dtype=float),
            "max_vals": np.array([], dtype=float),
            "medians": None,
            "iqrs": None,
        }
        return empty, params

    data_array = data_array.astype(float, copy=False)
    min_vals = np.min(data_array, axis=0)
    max_vals = np.max(data_array, axis=0)
    denom = max_vals - min_vals
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(
            data_array - min_vals,
            denom,
            out=np.full_like(data_array, np.nan, dtype=float),
            where=denom != 0,
        )
    normalized = np.where(denom == 0, np.nan, normalized)
    
    params = {
        "method": "minmax",
        "means": None,
        "stds": None,
        "min_vals": min_vals,
        "max_vals": max_vals,
        "medians": None,
        "iqrs": None,
    }
    return normalized, params


def _normalize_array_robust(data_array: np.ndarray):
    """使用robust方法标准化数组"""
    if data_array.size == 0:
        empty = data_array.astype(float)
        params = {
            "method": "robust",
            "means": None,
            "stds": None,
            "min_vals": None,
            "max_vals": None,
            "medians": np.array([], dtype=float),
            "iqrs": np.array([], dtype=float),
        }
        return empty, params

    data_array = data_array.astype(float, copy=False)
    medians = np.median(data_array, axis=0)
    mad = np.median(np.abs(data_array - medians), axis=0)
    mad_safe = np.where(np.abs(mad) < DataUtilsConstants.EPSILON, DataUtilsConstants.EPSILON, mad)
    normalized = (data_array - medians) / mad_safe
    
    params = {
        "method": "robust",
        "means": None,
        "stds": None,
        "min_vals": None,
        "max_vals": None,
        "medians": medians,
        "iqrs": mad,
    }
    return normalized, params


def _validate_normalize_input(data):
    """验证标准化输入参数"""
    if data is None:
        raise ValueError("输入数据不能为None")

    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("DataFrame不能为空")
    elif isinstance(data, np.ndarray):
        if data.size == 0:
            raise ValueError("数组不能为空")
    elif isinstance(data, list):
        if len(data) == 0:
            raise ValueError("列表不能为空")
    else:
        raise TypeError("不支持的数据类型")


def _normalize_dataframe(
    data: pd.DataFrame,
    method: str = "standard",
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, dict]:
    """标准化DataFrame数据"""
    data_size = len(data)

    # 只处理数值列
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        # 如果没有数值列，返回原数据
        params = {col: {"means": 0, "stds": 1} for col in data.columns}
        return data.copy(), params

    # 根据method选择标准化策略
    if method == "standard":
        if mean is None or std is None:
            calculated_mean = data[numeric_cols].mean()
            calculated_std = data[numeric_cols].std()
        else:
            calculated_mean = pd.Series(mean, index=numeric_cols)
            calculated_std = pd.Series(std, index=numeric_cols)

        # 创建副本并只标准化数值列
        normalized_data = data.copy()
        normalized_data[numeric_cols] = (data[numeric_cols] - calculated_mean) / calculated_std

        # 返回以列名为键的字典
        params = {}
        for col in data.columns:
            if col in numeric_cols:
                params[col] = {"means": calculated_mean[col], "stds": calculated_std[col]}
            else:
                params[col] = {"means": 0, "stds": 1}

    elif method == "minmax":
        min_vals = data[numeric_cols].min()
        max_vals = data[numeric_cols].max()
        normalized_data = data.copy()
        for col in numeric_cols:
            col_min = min_vals[col]
            col_max = max_vals[col]
            if col_max != col_min:  # 避免除零
                normalized_data[col] = (data[col] - col_min) / (col_max - col_min)
            else:
                normalized_data[col] = 0  # 常量列设为0
        # 返回以列名为键的字典
        params = {}
        for col in data.columns:
            if col in numeric_cols:
                params[col] = {"min_vals": min_vals[col], "max_vals": max_vals[col]}
            else:
                params[col] = {"min_vals": 0, "max_vals": 1}

    elif method == "robust":
        medians = data[numeric_cols].median()
        mad = (data[numeric_cols] - medians).abs().median()
        normalized_data = data.copy()
        for col in numeric_cols:
            col_median = medians[col]
            col_mad = mad[col]
            if col_mad != 0:  # 避免除零
                normalized_data[col] = (data[col] - col_median) / col_mad
            else:
                normalized_data[col] = 0  # 常量列设为0
        # 返回以列名为键的字典
        params = {}
        for col in data.columns:
            if col in numeric_cols:
                params[col] = {"medians": medians[col], "iqrs": mad[col]}
            else:
                params[col] = {"medians": 0, "iqrs": 1}

    else:
        raise ValueError(f"不支持的标准化方法: {method}")

    # 记录处理结果
    _log_normalization_result("DataFrame", data_size, time.time())

    return normalized_data, params


def _normalize_array(
    data: np.ndarray,
    method: str = "standard",
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """标准化数组数据"""
    data_size = data.size

    # 根据method选择标准化策略
    if method == "standard":
        if data_size == 0:
            empty = data.astype(float)
            params = {
                "means": np.array([], dtype=float),
                "stds": np.array([], dtype=float),
            }
            _log_normalization_result("数组", data_size, time.time())
            return empty, params
        if mean is None:
            mean = np.nanmean(data, axis=0)
        if std is None:
            std = np.nanstd(data, axis=0, ddof=0)
        # 对于标准差为0的情况（常量数组），使用epsilon作为分母
        std_safe = np.where(
            np.abs(std) < DataUtilsConstants.EPSILON, DataUtilsConstants.EPSILON, std
        )
        normalized = (data - mean) / std_safe
        params = {"means": mean, "stds": std_safe}

    elif method == "minmax":
        if data_size == 0:
            empty = data.astype(float)
            params = {
                "min_vals": np.array([], dtype=float),
                "max_vals": np.array([], dtype=float),
            }
            _log_normalization_result("数组", data_size, time.time())
            return empty, params
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        normalized = (data - min_vals) / np.where(
            (max_vals - min_vals) == 0, 1, max_vals - min_vals
        )
        params = {"min_vals": min_vals, "max_vals": max_vals}

    elif method == "robust":
        if data_size == 0:
            empty = data.astype(float)
            params = {
                "medians": np.array([], dtype=float),
                "iqrs": np.array([], dtype=float),
            }
            _log_normalization_result("数组", data_size, time.time())
            return empty, params
        medians = np.median(data, axis=0)
        mad = np.median(np.abs(data - medians), axis=0)
        normalized = (data - medians) / np.where(
            np.abs(mad) < DataUtilsConstants.EPSILON, DataUtilsConstants.EPSILON, mad
        )
        params = {"medians": medians, "iqrs": mad}

    else:
        raise ValueError(f"不支持的标准化方法: {method}")

    # 记录处理结果
    _log_normalization_result("数组", data_size, time.time())

    return normalized, params


def _log_normalization_result(data_type: str, data_size: int, start_time: float):
    """记录标准化处理结果"""
    processing_time = time.time() - start_time
    message = f"{data_type}标准化完成，数据点: {data_size}, 处理时间: {processing_time:.4f}s"
    _safe_logger_log(logging.INFO, message)


def denormalize_data(normalized_data, params: dict, method: str = "standard"):
    """
    反标准化数据

    参数:
        normalized_data: 标准化后的数据
        params: 标准化时使用的参数字典
        method: 标准化方法 ('standard', 'minmax', 'robust')

    返回:
        原始数据

    异常:
        ValueError: 当参数无效时抛出
    """
    try:
        # 参数验证
        _validate_denormalize_params(normalized_data, params, method)

        # 根据方法执行反标准化
        if method == "standard":
            result = _denormalize_standard(normalized_data, params)
        elif method == "minmax":
            result = _denormalize_minmax(normalized_data, params)
        elif method == "robust":
            result = _denormalize_robust(normalized_data, params)
        elif method == "mixed":
            result = _denormalize_mixed(normalized_data, params)
        else:
            raise ValueError(f"不支持的反标准化方法: {method}")
        
        _safe_logger_log(logging.DEBUG, "成功执行数据反标准化")
        return result

    except Exception as e:
        _safe_logger_log(logging.ERROR, f"数据反标准化过程中发生错误: {e}")
        raise


def _validate_denormalize_params(normalized_data, params, method):
    """验证反标准化参数"""
    if normalized_data is None:
        raise ValueError("标准化数据不能为None")
    if params is None:
        raise ValueError("参数字典不能为None")
    if method not in ["standard", "minmax", "robust", "mixed"]:
        raise ValueError(f"不支持的反标准化方法: {method}")


def _denormalize_standard(normalized_data, params):
    """使用标准方法反标准化"""
    if "means" not in params or "stds" not in params:
        raise ValueError("标准方法需要means和stds参数")

    mean = params["means"]
    std = params["stds"]

    try:
        # 尝试广播操作
        result = normalized_data * std + mean
        return result
    except ValueError as e:
        if "operands could not be broadcast together" in str(e):
            # 使用广播处理维度不匹配的情况
            mean_broadcast = np.broadcast_to(mean, normalized_data.shape)
            std_broadcast = np.broadcast_to(std, normalized_data.shape)
            return normalized_data * std_broadcast + mean_broadcast
        raise


def _denormalize_minmax(normalized_data, params):
    """使用minmax方法反标准化"""
    if "min_vals" not in params or "max_vals" not in params:
        raise ValueError("minmax方法需要min_vals和max_vals参数")
    
    min_vals = params["min_vals"]
    max_vals = params["max_vals"]
    
    try:
        return normalized_data * (max_vals - min_vals) + min_vals
    except ValueError as e:
        if "operands could not be broadcast together" in str(e):
            raise ValueError(
                f"参数维度不匹配: 数据形状 {normalized_data.shape}, "
                f"参数形状 min_vals={np.array(min_vals).shape}, max_vals={np.array(max_vals).shape}"
            )
        raise


def _denormalize_robust(normalized_data, params):
    """使用robust方法反标准化"""
    if "medians" not in params or "iqrs" not in params:
        raise ValueError("robust方法需要medians和iqrs参数")
    
    medians = params["medians"]
    iqrs = params["iqrs"]
    
    try:
        return normalized_data * iqrs + medians
    except ValueError as e:
        if "operands could not be broadcast together" in str(e):
            raise ValueError(
                f"参数维度不匹配: 数据形状 {normalized_data.shape}, "
                f"参数形状 medians={np.array(medians).shape}, iqrs={np.array(iqrs).shape}"
            )
        raise


def _denormalize_mixed(normalized_data, params):
    """使用混合方法反标准化"""
    if not isinstance(normalized_data, pd.DataFrame):
        raise ValueError("mixed方法只支持DataFrame")
    
    result = normalized_data.copy()
    for col in normalized_data.columns:
        if col in params and isinstance(params[col], dict):
            col_params = params[col]
            result[col] = _denormalize_column(normalized_data[col], col_params)
    
    return result


def _denormalize_column(column_data, col_params):
    """反标准化单个列"""
    if "means" in col_params and "stds" in col_params:
        # 标准方法
        means = _extract_scalar_value(col_params["means"])
        stds = _extract_scalar_value(col_params["stds"])
        return column_data * stds + means
    elif "min_vals" in col_params and "max_vals" in col_params:
        # minmax方法
        min_vals = _extract_scalar_value(col_params["min_vals"])
        max_vals = _extract_scalar_value(col_params["max_vals"])
        return column_data * (max_vals - min_vals) + min_vals
    return column_data


def _extract_scalar_value(value):
    """从可能是数组的值中提取标量"""
    if hasattr(value, "__len__") and len(value) > 0:
        return value[0]
    return value


__all__ = ["normalize_data", "denormalize_data"]

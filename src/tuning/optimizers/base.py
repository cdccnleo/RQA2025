from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Optional
import pandas as pd

class SearchMethod(Enum):
    """搜索方法枚举"""
    GRID = auto()        # 网格搜索
    RANDOM = auto()      # 随机搜索
    BAYESIAN = auto()    # 贝叶斯优化
    TPE = auto()         # TPE优化
    CMAES = auto()       # CMA-ES优化

class ObjectiveDirection(Enum):
    """优化方向枚举"""
    MAXIMIZE = auto()    # 最大化
    MINIMIZE = auto()    # 最小化

@dataclass
class TuningResult:
    """调参结果数据结构"""
    best_params: Dict
    best_value: float
    trials: pd.DataFrame
    importance: Optional[Dict] = None

class BaseTuner(ABC):
    """调参器基类"""

    @abstractmethod
    def tune(self, objective_func, param_space,
            n_trials=100, direction=ObjectiveDirection.MAXIMIZE) -> TuningResult:
        """执行参数优化"""
        pass

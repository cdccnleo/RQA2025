
"""
导入别名定义
简化复杂的导入路径
"""

# 加速层别名 - FPGA和GPU加速组件
from .acceleration.fpga import FpgaManager as FPGA
from acceleration.gpu import GPUManager as GPU, CUDAComputeEngine as CUDA

# 基础设施层别名 - 配置管理、缓存、监控等基础设施组件
from infrastructure.config.unified_config import get_config, set_config, ConfigScope
from infrastructure.cache.thread_safe_cache import ThreadSafeCache, CachePolicy

# 数据层别名 - 数据加载和处理组件
from data.loader.stock_loader import StockDataLoader as StockLoader
from data.loader.news_loader import FinancialNewsLoader as NewsLoader

# 特征层别名 - 特征工程和信号处理组件
from features.signal_generator import SignalGenerator as SignalGen
from features.config import feature_config_manager as FeatureConfig

# 模型层别名 - 模型管理和推理组件
from models.model_manager import ModelManager as ModelMgr
from models.base_model import BaseModel as Model

# 交易层别名 - 交易引擎和风险控制组件
from trading.trading_engine import TradingEngine as Trading
from trading.risk.risk_controller import RiskController as Risk

# 导出所有别名
__all__ = [
    'FPGA', 'GPU', 'CUDA',
    'get_config', 'set_config', 'ConfigScope',
    'ThreadSafeCache', 'CachePolicy',
    'StockLoader', 'NewsLoader',
    'SignalGen', 'FeatureConfig',
    'ModelMgr', 'Model',
    'Trading', 'Risk'
]

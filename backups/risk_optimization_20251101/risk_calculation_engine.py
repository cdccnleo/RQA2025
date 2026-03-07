    #!/usr/bin/env python
    # -*- coding: utf-8 -*-

"""
    风险计算引擎

提供VaR计算、风险指标计算、风险模型集成等核心功能
"""

import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import json
import warnings
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# 导入AI风险预测模型
AI_PREDICTION_AVAILABLE = False
try:
    from .ai_risk_prediction_model import (
        AIRiskPredictionModel, PredictionModelConfig,
        PredictionModelType, RiskPredictionType
    )
    # 延迟导入PredictionResult以避免循环导入
    AI_PREDICTION_AVAILABLE = True
except ImportError:
    logger.warning("AI风险预测模型不可用，将使用传统方法")

# 单独导入PredictionResult和RiskPredictionType
try:
    from .ai_risk_prediction_model import PredictionResult, RiskPredictionType
except ImportError:
    # 如果导入失败，创建本地定义
    from typing import Tuple
    from dataclasses import dataclass, field
    from datetime import datetime
    from enum import Enum

    class RiskPredictionType(Enum):
        """风险预测类型枚举"""
        VOLATILITY = "volatility"
        DRAWDOWN = "drawdown"
        CORRELATION = "correlation"
        LIQUIDITY = "liquidity"

    @dataclass
    class PredictionResult:
        """预测结果"""
        prediction_type: str
        predicted_value: float
        confidence_interval: Tuple[float, float]
        prediction_timestamp: datetime
        model_accuracy: float
        feature_importance: Dict[str, float] = field(default_factory=dict)
        metadata: Dict[str, Any] = field(default_factory=dict)

# 导入多资产风险管理器
try:
    from .multi_asset_risk_manager import (
            MultiAssetRiskManager, AssetType, AssetClass,
            AssetConfig, AssetPosition, MultiAssetRiskMetrics
            )
    MULTI_ASSET_AVAILABLE = True
except ImportError:
    MULTI_ASSET_AVAILABLE = False
    logger.warning("多资产风险管理器不可用，将使用单资产模式")

# 导入GPU加速风险计算器
try:
            from .gpu_accelerated_risk_calculator import (
            GPUAcceleratedRiskCalculator, GPUConfig, GPUBackend,
            ComputationType, ComputationResult
            )
            GPU_CALCULATOR_AVAILABLE = True
except ImportError:
    GPU_CALCULATOR_AVAILABLE = False
    logger.warning("GPU加速风险计算器不可用，将使用CPU模式")

# 导入分布式缓存管理器
try:
            from .distributed_cache_manager import (
            DistributedCacheManager, CacheLevel, CacheConfig,
            CacheStrategy, CacheBackend
            )
            CACHE_MANAGER_AVAILABLE = True
except ImportError:
    CACHE_MANAGER_AVAILABLE = False
    logger.warning("分布式缓存管理器不可用")

# 导入异步任务管理器
try:
            from .async_task_manager import (
            AsyncTaskManager, TaskPriority, TaskStatus, TaskType,
            TaskResult, submit_async_task, get_task_status, get_task_result
            )
            ASYNC_TASK_AVAILABLE = True
except ImportError:
    ASYNC_TASK_AVAILABLE = False
    logger.warning("异步任务管理器不可用")

# 导入内存优化器
try:
            from .memory_optimizer import (
            MemoryOptimizer, MemoryPoolType, MemoryWarningLevel,
            MemoryStats, MemoryLeakReport
            )
            MEMORY_OPTIMIZER_AVAILABLE = True
except ImportError:
    MEMORY_OPTIMIZER_AVAILABLE = False
    logger.warning("内存优化器不可用")


class RiskMetricType(Enum):
    """风险指标类型"""
    VAR = "var"                           # 风险价值
    CVAR = "cvar"                         # 条件风险价值
    VOLATILITY = "volatility"             # 波动率
    BETA = "beta"                         # 贝塔系数
    SHARPE_RATIO = "sharpe_ratio"        # 夏普比率
    MAX_DRAWDOWN = "max_drawdown"        # 最大回撤
    CORRELATION = "correlation"           # 相关性
    CONCENTRATION = "concentration"       # 集中度
    LIQUIDITY = "liquidity"              # 流动性
    STRESS_TEST = "stress_test"          # 压力测试


class ConfidenceLevel(Enum):
    """置信水平"""
    P90 = 0.90
    P95 = 0.95
    P99 = 0.99
    P99_5 = 0.995


@dataclass
class RiskCalculationConfig:

    """风险计算配置"""
    confidence_level: ConfidenceLevel = ConfidenceLevel.P95
    time_horizon_days: int = 1
    historical_window_days: int = 252
    monte_carlo_simulations: int = 10000
    enable_stress_testing: bool = True
    enable_monte_carlo: bool = True
    enable_historical_simulation: bool = True
    risk_free_rate: float = 0.02
    max_iterations: int = 1000
    tolerance: float = 1e-6


@dataclass
class RiskCalculationResult:

    """风险计算结果"""
    metric_type: RiskMetricType
    value: float
    confidence_level: float
    time_horizon: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_interval: Optional[Tuple[float, float]] = None
    p_value: Optional[float] = None


@dataclass
class PortfolioRiskProfile:

    """组合风险画像"""
    portfolio_id: str
    total_value: float
    total_risk: float
    risk_metrics: Dict[RiskMetricType, RiskCalculationResult]
    risk_decomposition: Dict[str, float]
    stress_test_results: Dict[str, float]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskCalculationEngine:

    """风险计算引擎"""

    def __init__(self, config: Optional[RiskCalculationConfig] = None):
        self.config = config or RiskCalculationConfig()
        self.lock = threading.RLock()

        # 历史数据缓存
        self.price_cache = defaultdict(lambda: deque(maxlen=1000))
        self.return_cache = defaultdict(lambda: deque(maxlen=1000))
        self.risk_cache = defaultdict(lambda: deque(maxlen=100))

        # 风险模型
        self.risk_models = {
        RiskMetricType.VAR: self._calculate_var,
        RiskMetricType.CVAR: self._calculate_cvar,
        RiskMetricType.VOLATILITY: self._calculate_volatility,
        RiskMetricType.BETA: self._calculate_beta,
        RiskMetricType.SHARPE_RATIO: self._calculate_sharpe_ratio,
        RiskMetricType.MAX_DRAWDOWN: self._calculate_max_drawdown,
        RiskMetricType.CONCENTRATION: self._calculate_concentration,
        RiskMetricType.LIQUIDITY: self._calculate_liquidity,
        RiskMetricType.STRESS_TEST: self._calculate_stress_test
        }

        # AI风险预测模型
        self.ai_prediction_model = None
        if AI_PREDICTION_AVAILABLE:
            try:
                prediction_config = PredictionModelConfig(
                    model_type=PredictionModelType.RANDOM_FOREST,
                    prediction_horizon=1,
                    training_window=self.config.historical_window_days,
                    enable_feature_selection=True,
                    enable_hyperparameter_tuning=False
                )
                self.ai_prediction_model = AIRiskPredictionModel(prediction_config)
                logger.info("AI风险预测模型初始化成功")
            except Exception as e:
                logger.warning(f"AI风险预测模型初始化失败: {e}")
                self.ai_prediction_model = None

        # 多资产风险管理器
        self.multi_asset_manager = None
        if MULTI_ASSET_AVAILABLE:
            try:
                self.multi_asset_manager = MultiAssetRiskManager()
                logger.info("多资产风险管理器初始化成功")
            except Exception as e:
                logger.warning(f"多资产风险管理器初始化失败: {e}")
                self.multi_asset_manager = None

        # GPU加速风险计算器
        self.gpu_calculator = None
        if GPU_CALCULATOR_AVAILABLE:
            try:
                # 尝试使用最佳可用的GPU后端
                gpu_config = GPUConfig()
                if self._is_cupy_available():
                    gpu_config.backend = GPUBackend.CUPY
                elif self._is_pytorch_cuda_available():
                    gpu_config.backend = GPUBackend.PYTORCH
                elif self._is_numba_cuda_available():
                    gpu_config.backend = GPUBackend.NUMBA_CUDA
                else:
                    gpu_config.backend = GPUBackend.CPU

                self.gpu_calculator = GPUAcceleratedRiskCalculator(gpu_config)
                logger.info(f"GPU加速风险计算器初始化成功，后端: {gpu_config.backend.value}")
            except Exception as e:
                logger.warning(f"GPU加速风险计算器初始化失败: {e}")
                self.gpu_calculator = None

        # 分布式缓存管理器
        self.cache_manager = None
        if CACHE_MANAGER_AVAILABLE:
            try:
                cache_config = {
                    'l1_max_size': 1000,          # L1缓存最大条目数
                    'l1_max_memory_mb': 256,      # L1缓存最大内存MB
                    'l1_ttl': 300,                # L1缓存默认TTL(5分钟)
                    'l2_max_size': 5000,          # L2缓存最大条目数
                    'l2_max_memory_mb': 1024,     # L2缓存最大内存MB
                    'l2_ttl': 1800,               # L2缓存默认TTL(30分钟)
                    'enable_redis': True,         # 启用Redis缓存
                    'redis_host': 'localhost',
                    'redis_port': 6379,
                    'redis_db': 1,                # 使用数据库存储风险计算缓存
                    'enable_compression': True,   # 启用压缩
                    'enable_prewarm': True        # 启用缓存预热
                }
                self.cache_manager = DistributedCacheManager(cache_config)
                logger.info("分布式缓存管理器初始化成功")
            except Exception as e:
                logger.warning(f"分布式缓存管理器初始化失败: {e}")
                self.cache_manager = None

        # 异步任务管理器
        self.async_task_manager = None
        if ASYNC_TASK_AVAILABLE:
            try:
                self.async_task_manager = AsyncTaskManager(
                    max_workers=4,      # 最大工作线程数
                    max_queue_size=100  # 最大队列大小
                )
                self.async_task_manager.start()
                logger.info("异步任务管理器初始化成功")
            except Exception as e:
                logger.warning(f"异步任务管理器初始化失败: {e}")
                self.async_task_manager = None

        # 内存优化器
        self.memory_optimizer = None
        if MEMORY_OPTIMIZER_AVAILABLE:
            try:
                memory_config = {
                    'monitor_interval': 30,      # 监控间隔(秒)
                    'enable_tracemalloc': True,  # 启用内存追踪
                    'batch_size': 1000,          # 批处理大小
                    'max_memory_mb': 1024,       # 最大内存使用MB
                    # 内存池配置
                    'numpy_pool_max_size': 50,
                    'numpy_pool_max_memory': 1024,
                    'pandas_pool_max_size': 20,
                    'pandas_pool_max_memory': 2048,
                    'results_pool_max_size': 100,
                    'results_pool_max_memory': 512,
                    'cache_pool_max_size': 200,
                    'cache_pool_max_memory': 1024,
                    # GC配置
                    'gc_threshold': 1000
                }
                self.memory_optimizer = MemoryOptimizer(memory_config)
                self.memory_optimizer.start()
                logger.info("内存优化器初始化成功")
            except Exception as e:
                logger.warning(f"内存优化器初始化失败: {e}")
                self.memory_optimizer = None

        logger.info("风险计算引擎初始化完成")

    def predict_future_risk(self, prediction_type: str, current_data: Dict[str, Any]) -> Optional[PredictionResult]:
        """预测未来风险

        Args:
            prediction_type: 预测类型('volatility', 'drawdown', 'correlation', 'liquidity')
            current_data: 当前市场数据

        Returns:
            预测结果
        """
        if not self.ai_prediction_model:
            logger.warning("AI预测模型不可用")
            return None

        try:
            # 映射预测类型
            risk_prediction_type = self._map_prediction_type(prediction_type)
            if not risk_prediction_type:
                return None

            # 获取AI预测
            return self.ai_prediction_model.predict_risk(risk_prediction_type, current_data)
        except Exception as e:
            logger.error(f"AI风险预测失败: {e}")
            return None

    def train_ai_models(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        训练AI模型

        Args:
            historical_data: 历史数据列表

        Returns:
            训练结果
        """
        if not self.ai_prediction_model:
            logger.warning("AI预测模型不可用")
            return {"error": "AI模型不可用"}
        try:
            training_results = {}

            # 准备训练数据
            training_data = self._prepare_training_data(historical_data)

            # 训练不同类型的预测模型
            for prediction_type in RiskPredictionType:
                try:
                    result = self.ai_prediction_model.train_model(
                        prediction_type,
                        training_data,
                        target_column=self._get_target_column(prediction_type)
                    )
                    training_results[prediction_type.value] = result
                except Exception as e:
                    logger.error(f"训练{prediction_type.value}模型失败: {e}")
                    training_results[prediction_type.value] = {"error": str(e)}

            return training_results
        except Exception as e:
            logger.error(f"AI模型训练失败: {e}")
            return {"error": str(e)}

    def _should_use_ai_prediction(self, metric_type: RiskMetricType) -> bool:
        """判断是否应该使用AI预测"""
        try:
            # 对于波动率、VaR等指标，使用AI预测增强
            ai_supported_metrics = [
                RiskMetricType.VOLATILITY,
                RiskMetricType.VAR,
                RiskMetricType.MAX_DRAWDOWN,
                RiskMetricType.CORRELATION
            ]
            return metric_type in ai_supported_metrics and self.ai_prediction_model is not None
        except Exception as e:
            logger.error(f"判断AI预测使用失败: {e}")
            return False

    def _get_ai_prediction(self, metric_type: RiskMetricType,
                           portfolio_data: Dict[str, Any],
                           current_value: float) -> Optional[PredictionResult]:
        """获取AI预测结果"""
        try:
            # 映射到AI预测类型
            prediction_type = self._map_metric_to_prediction_type(metric_type)
            if not prediction_type:
                return None

            # 准备预测数据
            prediction_data = self._prepare_prediction_data(portfolio_data, current_value)

            # 获取AI预测
            return self.ai_prediction_model.predict_risk(prediction_type, prediction_data)
        except Exception as e:
            logger.error(f"获取AI预测失败: {e}")
            return None

    def _map_prediction_type(self, prediction_type_str: str) -> Optional[RiskPredictionType]:
        """映射预测类型字符串到枚举"""
        mapping = {
            'volatility': RiskPredictionType.VOLATILITY_FORECAST,
            'drawdown': RiskPredictionType.DRAWDOWN_PREDICTION,
            'correlation': RiskPredictionType.CORRELATION_CHANGE,
            'liquidity': RiskPredictionType.LIQUIDITY_RISK,
            'market_impact': RiskPredictionType.MARKET_IMPACT,
            'extreme_event': RiskPredictionType.EXTREME_EVENT_RISK
        }
        return mapping.get(prediction_type_str)

    def _map_metric_to_prediction_type(self, metric_type: RiskMetricType) -> Optional[RiskPredictionType]:
        """映射风险指标类型到预测类型"""
        mapping = {
            RiskMetricType.VOLATILITY: RiskPredictionType.VOLATILITY_FORECAST,
            RiskMetricType.VAR: RiskPredictionType.EXTREME_EVENT_RISK,
            RiskMetricType.MAX_DRAWDOWN: RiskPredictionType.DRAWDOWN_PREDICTION,
            RiskMetricType.CORRELATION: RiskPredictionType.CORRELATION_CHANGE,
            RiskMetricType.LIQUIDITY: RiskPredictionType.LIQUIDITY_RISK
        }
        return mapping.get(metric_type)

    def _prepare_prediction_data(self, portfolio_data: Dict[str, Any], current_value: float) -> Dict[str, Any]:
        """准备预测数据"""
        try:
            prediction_data = {
                'timestamp': datetime.now().isoformat(),
                'current_risk_value': current_value,
                'portfolio_value': portfolio_data.get('portfolio_value', 0),
                'positions_count': len(portfolio_data.get('positions', {})),
                'market_data': {},
                'technical_indicators': {}
            }

            # 添加市场数据
            prices = portfolio_data.get('prices', {})
            if prices:
                prediction_data['market_data'] = {
                    'avg_price': np.mean(list(prices.values())),
                    'price_volatility': np.std(list(prices.values())),
                    'price_range': max(prices.values()) - min(prices.values())
                }
            return prediction_data
        except Exception as e:
            logger.error(f"准备预测数据失败: {e}")
            return {}

    def _prepare_training_data(self, historical_data: List[Dict[str, Any]]) -> List:
        """准备训练数据"""
        try:
            from .ai_risk_prediction_model import RiskPredictionData

            training_data = []
            for data in historical_data:
                prediction_data = RiskPredictionData(
                    timestamp=data.get('timestamp', datetime.now()),
                    features=data.get('features', {}),
                    actual_risk_value=data.get('risk_value'),
                    predicted_risk_value=data.get('predicted_value'),
                    prediction_error=data.get('prediction_error')
                )
                training_data.append(prediction_data)
            return training_data
        except Exception as e:
            logger.error(f"准备训练数据失败: {e}")
            return []

    def _get_target_column(self, prediction_type: RiskPredictionType) -> str:
        """获取目标列名"""
        mapping = {
            RiskPredictionType.VOLATILITY_FORECAST: 'volatility',
            RiskPredictionType.DRAWDOWN_PREDICTION: 'max_drawdown',
            RiskPredictionType.CORRELATION_CHANGE: 'correlation',
            RiskPredictionType.LIQUIDITY_RISK: 'liquidity',
            RiskPredictionType.MARKET_IMPACT: 'market_impact',
            RiskPredictionType.EXTREME_EVENT_RISK: 'extreme_event_risk'
        }
        return mapping.get(prediction_type, 'risk_value')

    def _is_cupy_available(self) -> bool:
        """检查CuPy是否可用"""
        try:
            import cupy as cp
            return cp.cuda.is_available()
        except ImportError:
            return False

    def _is_pytorch_cuda_available(self) -> bool:
        """检查PyTorch CUDA是否可用"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _is_numba_cuda_available(self) -> bool:
        """检查Numba CUDA是否可用"""
        try:
            from numba import cuda
            return cuda.is_available()
        except ImportError:
            return False

    def calculate_gpu_accelerated_var(self, portfolio_data: Dict[str, Any],
            confidence_level: float = 0.95,  n_simulations: int = 10000) -> Dict[str, Any]:
        """
        GPU加速VaR计算

        Args:
            portfolio_data: 组合数据
            confidence_level: 置信水平
            n_simulations: 模拟次数

        Returns:
            计算结果
        """
        if not self.gpu_calculator:
            logger.warning("GPU加速计算器不可用，使用CPU计算")
            return self._calculate_var_cpu(portfolio_data, confidence_level, n_simulations)
        try:
            # 生成缓存key
            cache_key = None
            if self.cache_manager:
                portfolio_id = portfolio_data.get('portfolio_id', 'default')
                positions = portfolio_data.get('positions', {})
                cache_key = self.cache_manager.create_cache_key(
                    'gpu_var',
                    portfolio_id,
                    str(sorted(positions.items())),
                    confidence_level,
                    n_simulations,
                    portfolio_data.get('data_hash', '')
                )

            # 尝试从缓存获取结果
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.debug(f"从缓存获取GPU VaR结果: {portfolio_id}")
                cached_result['cached'] = True
                return cached_result

            # 提取投资组合数据
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            if not positions or not prices:
                raise ValueError("投资组合数据不完整")

            # 获取历史数据（实际应用中应该从数据层获取）
            asset_symbols = list(positions.keys())

            # 模拟历史收益率数据
            n_periods = 252  # 一年的交易日
            n_assets = len(asset_symbols)
            np.random.seed(42)  # 确保可重复性
            returns_data = np.secrets.normal(0.0005, 0.02, (n_assets, n_periods))

            # 计算持仓权重
            position_values = np.array([positions[symbol] * prices.get(symbol, 0)
            for symbol in asset_symbols])
            total_value = np.sum(position_values)
            if total_value == 0:
                raise ValueError("组合总价值为0")

            weights = position_values / total_value

            # 使用GPU加速计算VaR
            result = self.gpu_calculator.monte_carlo_var_calculation(
                returns_data, weights, confidence_level, n_simulations
            )

            result_data = {
                'var': result.result_data['var'],
                'cvar': result.result_data['cvar'],
                'confidence_level': confidence_level,
                'n_simulations': n_simulations,
                'computation_time': result.computation_time,
                'gpu_memory_used': result.gpu_memory_used,
                'backend_used': result.backend_used.value,
                'accelerated': True,
                'cached': False
            }

            # 存储到分布式缓存
            if self.cache_manager and cache_key:
                try:
                    self.cache_manager.set(cache_key, result_data, ttl=600)  # 缓存10分钟
                    logger.debug("GPU VaR结果已缓存")
                except Exception as e:
                    logger.warning(f"GPU VaR结果缓存失败: {e}")
            return result_data
        except Exception as e:
            logger.error(f"GPU加速VaR计算失败: {e}")
            return self._calculate_var_cpu(portfolio_data, confidence_level, n_simulations)

    def _calculate_var_cpu(self, portfolio_data: Dict[str, Any],
                           confidence_level: float = 0.95,
                           n_simulations: int = 10000) -> Dict[str, Any]:
        """CPU备选VaR计算"""
        try:
            # 提取投资组合数据
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            if not positions or not prices:
                raise ValueError("投资组合数据不完整")

            # 模拟历史收益率数据
            asset_symbols = list(positions.keys())
            n_periods = 252
            n_assets = len(asset_symbols)
            np.random.seed(42)
            returns_data = np.secrets.normal(0.0005, 0.02, (n_assets, n_periods))

            # 计算权重
            position_values = np.array([positions[symbol] * prices.get(symbol, 0)
            for symbol in asset_symbols])
            total_value = np.sum(position_values)
            if total_value == 0:
                raise ValueError("组合总价值为0")

            weights = position_values / total_value

            # 计算协方差矩阵
            cov_matrix = np.cov(returns_data)

            # Cholesky分解
            try:
                L = np.linalg.cholesky(cov_matrix)
            except np.linalg.LinAlgError:
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                eigenvalues = np.maximum(eigenvalues, 0)
                L = eigenvectors @ np.sqrt(np.diag(eigenvalues))

            # 蒙特卡洛模拟
            random_matrix = np.secrets.normal(0, 1, (n_simulations, n_assets))
            correlated_returns = random_matrix @ L.T
            portfolio_returns = correlated_returns @ weights

            # 计算VaR
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(portfolio_returns, var_percentile)

            # 计算CVaR
            tail_returns = portfolio_returns[portfolio_returns <= var_value]
            cvar_value = np.mean(tail_returns) if len(tail_returns) > 0 else var_value
            return {
                'var': float(var_value),
                'cvar': float(cvar_value),
                'confidence_level': confidence_level,
                'n_simulations': n_simulations,
                'computation_time': 0.0,
                'gpu_memory_used': 0.0,
                'backend_used': 'cpu',
                'accelerated': False
            }
        except Exception as e:
            logger.error(f"CPU VaR计算失败: {e}")
            return {
                'var': 0.0,
                'cvar': 0.0,
                'error': str(e),
                'accelerated': False
            }

    def batch_gpu_risk_calculation(self, portfolio_list: List[Dict[str, Any]],
                                   risk_types: List[str] = None) -> List[Dict[str, Any]]:
        """
    批量GPU风险计算

    Args:
    portfolio_list: 组合列表
    risk_types: 风险类型列表

    Returns:
    计算结果列表
        """
        if not self.gpu_calculator:
            logger.warning("GPU加速计算器不可用，使用CPU计算")
            return self._batch_cpu_risk_calculation(portfolio_list, risk_types)
        try:
            if risk_types is None:
                risk_types = ['var', 'volatility']

            # 使用GPU计算器的批量计算功能
            results = self.gpu_calculator.batch_risk_calculation(portfolio_list, risk_types)

            # 转换结果格式
            formatted_results = []
            for result in results:
                formatted_result = {
                    'portfolio_id': result.result_data.get('portfolio_id', 'unknown'),
                    'risk_metrics': result.result_data.get('risk_metrics', {}),
                    'n_assets': result.result_data.get('n_assets', 0),
                    'total_value': result.result_data.get('total_value', 0),
                    'computation_time': result.computation_time,
                    'gpu_memory_used': result.gpu_memory_used,
                    'backend_used': result.backend_used.value,
                    'accelerated': True
                }
                formatted_results.append(formatted_result)
            return formatted_results
        except Exception as e:
            logger.error(f"批量GPU风险计算失败: {e}")
            return self._batch_cpu_risk_calculation(portfolio_list, risk_types)

    def _batch_cpu_risk_calculation(self, portfolio_list: List[Dict[str, Any]],
                                     risk_types: List[str] = None) -> List[Dict[str, Any]]:
        """CPU备选批量风险计算"""
        try:
            if risk_types is None:
                risk_types = ['var', 'volatility']

            results = []
            for portfolio_data in portfolio_list:
                try:
                    portfolio_result = {
                        'portfolio_id': portfolio_data.get('portfolio_id', 'unknown'),
                        'risk_metrics': {},
                        'accelerated': False
                    }

                    # 计算各项风险指标
                    for risk_type in risk_types:
                        if risk_type == 'var':
                            var_result = self._calculate_var_cpu(portfolio_data, 0.95, 1000)
                            portfolio_result['risk_metrics']['var'] = var_result['var']
                        elif risk_type == 'volatility':
                            # 简化波动率计算
                            positions = portfolio_data.get('positions', {})
                            if positions:
                                # 假设波动率为2%
                                portfolio_result['risk_metrics']['volatility'] = 0.02

                    results.append(portfolio_result)
                except Exception as e:
                    logger.error(f"计算组合风险失败: {e}")
                    results.append({
                        'portfolio_id': portfolio_data.get('portfolio_id', 'unknown'),
                        'risk_metrics': {},
                        'error': str(e),
                        'accelerated': False
                    })
            return results
        except Exception as e:
            logger.error(f"批量CPU风险计算失败: {e}")
            return []

    def get_gpu_performance_stats(self) -> Dict[str, Any]:
        """
    获取GPU性能统计

    Returns:
    性能统计信息
    """
        try:
            if not self.gpu_calculator:
                return {
                    'gpu_available': False,
                    'message': 'GPU加速计算器不可用'
                }
            return self.gpu_calculator.get_performance_stats()
        except Exception as e:
            logger.error(f"获取GPU性能统计失败: {e}")
            return {
                'gpu_available': False,
                'error': str(e)
            }

    def get_cache_stats(self) -> Dict[str, Any]:
        """
    获取缓存统计信息

    Returns:
    缓存统计信息
    """
        try:
            if not self.cache_manager:
                return {
                    'cache_available': False,
                    'message': '分布式缓存管理器不可用'
                }
            return self.cache_manager.get_stats()
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {
                'cache_available': False,
                'error': str(e)
            }

    def prewarm_risk_cache(self, historical_portfolios: List[Dict[str, Any]]) -> bool:
        """
    预热风险计算缓存

    Args:
    historical_portfolios: 历史组合数据列表

    Returns:
    预热是否成功
    """
        try:
            if not self.cache_manager:
                logger.warning("分布式缓存管理器不可用，无法预热缓存")
                return False
            logger.info(f"开始预热风险计算缓存，{len(historical_portfolios)} 个组合")

            def generate_prewarm_data():
                prewarm_data = {}
                for i, portfolio_data in enumerate(historical_portfolios):
                    try:
                        # 计算组合风险
                        risk_result = self.calculate_portfolio_risk(portfolio_data)

                        # 生成缓存key
                        portfolio_id = portfolio_data.get('portfolio_id', f'prewarm_{i}')
                        cache_key = self.cache_manager.create_cache_key(
                            'portfolio_risk',
                            portfolio_id,
                            str(sorted(portfolio_data.get('positions', {}).items())),
                            portfolio_data.get('prices', {}),
                            portfolio_data.get('historical_data_hash', '')
                        )

                        prewarm_data[cache_key] = risk_result
                        if (i + 1) % 10 == 0:
                            logger.info(f"预热进度: {i + 1}/{len(historical_portfolios)}")
                    except Exception as e:
                        logger.warning(f"预热组合 {i} 失败: {e}")
                return prewarm_data

            # 执行缓存预热
            success = self.cache_manager.prewarm_cache(generate_prewarm_data)
            if success:
                logger.info("风险计算缓存预热完成")
            else:
                logger.warning("风险计算缓存预热失败")
            return success
        except Exception as e:
            logger.error(f"缓存预热失败: {e}")
            return False

    def clear_risk_cache(self, cache_levels: Optional[List[str]]=None) -> bool:
        """
    清空风险计算缓存

    Args:
    cache_levels: 要清空的缓存级别

    Returns:
    清空是否成功
    """
        try:
            if not self.cache_manager:
                logger.warning("分布式缓存管理器不可用")
                return False
            # 转换字符串为枚举
            levels = None
            if cache_levels:
                levels = []
                for level_str in cache_levels:
                    if level_str.lower() == 'l1':
                        levels.append(CacheLevel.L1_MEMORY)
                    elif level_str.lower() == 'l2':
                        levels.append(CacheLevel.L2_DISTRIBUTED)
                    elif level_str.lower() == 'all':
                        levels = [CacheLevel.L1_MEMORY,
                            CacheLevel.L2_DISTRIBUTED, CacheLevel.L3_PERSISTENT]
                        break

            success = self.cache_manager.clear(levels)
            if success:
                logger.info("风险计算缓存已清空")
            else:
                logger.warning("风险计算缓存清空失败")
            return success
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
            return False

    def optimize_cache_settings(self) -> Dict[str, Any]:
        """
    优化缓存设置

    Returns:
    优化建议和结误
    """
        try:
            if not self.cache_manager:
                return {
                    'cache_available': False,
                    'message': '分布式缓存管理器不可用'
                }
            return self.cache_manager.optimize_cache()
        except Exception as e:
            logger.error(f"缓存优化失败: {e}")
            return {
                'cache_available': False,
                'error': str(e)
            }
    def get_cache_health_status(self) -> Dict[str, Any]:
        """
        获取缓存健康状态

        Returns:
        缓存健康状态
        """
        try:
            if not self.cache_manager:
                return {
                    'cache_available': False,
                    'message': '分布式缓存管理器不可用'
                }
            return self.cache_manager.get_cache_health()
        except Exception as e:
            logger.error(f"获取缓存健康状态失败 {e}")
            return {
                'overall_status': 'error',
                'issues': [f'健康检查失败 {e}']
            }
    def submit_async_risk_calculation(self, portfolio_data: Dict[str, Any],
            priority: TaskPriority=TaskPriority.NORMAL,
            callback: Optional[Callable]=None) -> Optional[str]:
        """
    提交异步风险计算任务

    Args:
    portfolio_data: 组合数据
    priority: 任务优先误
    callback: 完成回调函数

    Returns:
    任务ID，如果异步处理不可用则返回None
    """
        try:
            if not self.async_task_manager:
                logger.warning("异步任务管理器不可用")
                return None
            task_id=self.async_task_manager.submit_task(
            task_type=TaskType.RISK_CALCULATION,
            name=f"风险计算: {portfolio_data.get('portfolio_id', 'unknown')}",
            func=self.calculate_portfolio_risk,
            args=(portfolio_data,),
            priority=priority,
            callback=callback,
            timeout=30.0  # 30秒超误
            )

            logger.info(f"异步风险计算任务已提误 {task_id}")
            return task_id
        except Exception as e:
            logger.error(f"提交异步风险计算任务失败: {e}")
            return None

    def submit_async_gpu_var_calculation(self, portfolio_data: Dict[str, Any],
                                       confidence_level: float=0.95,
                                       n_simulations: int=10000,
                                       priority: TaskPriority=TaskPriority.NORMAL,
                                       callback: Optional[Callable]=None) -> Optional[str]:
        """
    提交异步GPU VaR计算任务

    Args:
    portfolio_data: 组合数据
    confidence_level: 置信水平
    n_simulations: 模拟次数
    priority: 任务优先误
    callback: 完成回调函数

    Returns:
    任务ID
    """
        try:
            if not self.async_task_manager:
                logger.warning("异步任务管理器不可用")
                return None
            task_id=self.async_task_manager.submit_task(
            task_type=TaskType.RISK_CALCULATION,
            name=f"GPU VaR计算: {portfolio_data.get('portfolio_id', 'unknown')}",
            func=self.calculate_gpu_accelerated_var,
            args=(portfolio_data, confidence_level, n_simulations),
            priority=priority,
            callback=callback,
            timeout=60.0  # 60秒超误
            )

            logger.info(f"异步GPU VaR计算任务已提误 {task_id}")
            return task_id
        except Exception as e:
            logger.error(f"提交异步GPU VaR计算任务失败: {e}")
            return None

    def submit_batch_async_calculations(self, portfolio_list: List[Dict[str, Any]],
                                       risk_types: List[str]=None,
                                       priority: TaskPriority=TaskPriority.NORMAL) -> List[str]:
        """
    提交批量异步计算任务

    Args:
    portfolio_list: 组合数据列表
    risk_types: 风险类型列表
    priority: 任务优先误

    Returns:
    任务ID列表
    """
        try:
            if not self.async_task_manager:
                logger.warning("异步任务管理器不可用")
                return []
            if risk_types is None:
                risk_types = ['var', 'volatility']

            task_ids = []
            for portfolio_data in portfolio_list:
                # 为每个组合提交批量计算任务
                task_id = self.async_task_manager.submit_task(
                    task_type=TaskType.BATCH_PROCESSING,
                    name=f"批量风险计算: {portfolio_data.get('portfolio_id', 'unknown')}",
                    func=self._batch_calculate_single_portfolio,
                    args=(portfolio_data, risk_types),
                    priority=priority,
                    timeout=120.0  # 2分钟超时
                )
                if task_id:
                    task_ids.append(task_id)

            logger.info(f"批量异步计算任务已提交 {len(task_ids)} 个任务")
            return task_ids
        except Exception as e:
            logger.error(f"提交批量异步计算任务失败: {e}")
            return []

    def _batch_calculate_single_portfolio(self, portfolio_data: Dict[str, Any],
                                         risk_types: List[str]) -> Dict[str, Any]:
        """批量计算单个组合的风险指标"""
        try:
            results = {}
            for risk_type in risk_types:
                if risk_type == 'var':
                    var_result = self.calculate_gpu_accelerated_var(
                        portfolio_data, 0.95, 5000
                    )
                    results['var'] = var_result
                elif risk_type == 'volatility':
                    # 简化波动率计算
                    positions = portfolio_data.get('positions', {})
                    if positions:
                        results['volatility'] = 0.02  # 默认波动率
                elif risk_type == 'sharpe_ratio':
                    # 简化夏普比率计算
                    results['sharpe_ratio'] = 1.5  # 默认夏普比率
            return {
                'portfolio_id': portfolio_data.get('portfolio_id', 'unknown'),
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"批量计算组合风险失败: {e}")
            return {
                'portfolio_id': portfolio_data.get('portfolio_id', 'unknown'),
                'error': str(e)
            }

    def get_async_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        获取异步任务状态

    Args:
    task_id: 任务ID

    Returns:
    任务状误
    """
        try:
            if not self.async_task_manager:
                return None
            return self.async_task_manager.get_task_status(task_id)
        except Exception as e:
            logger.error(f"获取异步任务状态失败: {e}")
            return None
    def get_async_task_result(self, task_id: str) -> Optional[TaskResult]:
        """
        获取异步任务结果

        Args:
        task_id: 任务ID

        Returns:
        任务结果
        """
        try:
            if not self.async_task_manager:
                return None
            return self.async_task_manager.get_task_result(task_id)
        except Exception as e:
            logger.error(f"获取异步任务结果失败: {e}")
            return None
    def cancel_async_task(self, task_id: str) -> bool:
        """
        取消异步任务

        Args:
        task_id: 任务ID

        Returns:
        是否成功取消
        """
        try:
            if not self.async_task_manager:
                return False
            return self.async_task_manager.cancel_task(task_id)
        except Exception as e:
            logger.error(f"取消异步任务失败: {e}")
            return False

    def wait_for_async_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
    等待异步任务完成

    Args:
    task_id: 任务ID
    timeout: 超时时间（秒误

    Returns:
    任务结果
    """
        try:
            if not self.async_task_manager:
                return None
            return self.async_task_manager.wait_for_task(task_id, timeout)
        except Exception as e:
            logger.error(f"等待异步任务完成失败: {e}")
            return None
    def get_async_task_queue_status(self) -> Dict[str, Any]:
        """
        获取异步任务队列状态

        Returns:
        队列状态信息
        """
        try:
            if not self.async_task_manager:
                return {
                    'available': False,
                    'message': '异步任务管理器不可用'
                }
            return self.async_task_manager.get_queue_status()
        except Exception as e:
            logger.error(f"获取异步任务队列状态失败 {e}")
            return {
                'available': False,
                'error': str(e)
            }
    def get_async_task_stats(self) -> Dict[str, Any]:
        """
        获取异步任务统计信息

        Returns:
        任务统计信息
        """
        try:
            if not self.async_task_manager:
                return {
                    'available': False,
                    'message': '异步任务管理器不可用'
                }
            queue_status = self.async_task_manager.get_queue_status()
            task_stats = self.async_task_manager.get_task_stats()
            running_tasks = self.async_task_manager.get_running_tasks()
            pending_tasks = self.async_task_manager.get_pending_tasks()
            return {
                'available': True,
                'queue_status': queue_status,
                'task_stats': task_stats,
                'running_tasks_count': len(running_tasks),
                'pending_tasks_count': len(pending_tasks),
                'running_task_ids': running_tasks[:10],  # 只返回前10个
                'pending_task_ids': pending_tasks[:10],   # 只返回前10个
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取异步任务统计失败: {e}")
            return {
                'available': False,
                'error': str(e)
            }
    def submit_async_prediction_task(self, prediction_type: str, current_data: Dict[str, Any],
            priority: TaskPriority = TaskPriority.NORMAL,
            callback: Optional[Callable] = None) -> Optional[str]:
        """
    提交异步预测任务

    Args:
    prediction_type: 预测类型
    current_data: 当前数据
    priority: 任务优先误
    callback: 完成回调函数

    Returns:
    任务ID
    """
        try:
            if not self.async_task_manager:
                logger.warning("异步任务管理器不可用")
                return None
            task_id = self.async_task_manager.submit_task(
            task_type=TaskType.PREDICTION,
            name=f"风险预测: {prediction_type}",
            func=self.predict_future_risk,
            args=(prediction_type, current_data),
            priority=priority,
            callback=callback,
            timeout=30.0  # 30秒超误
            )

            logger.info(f"异步预测任务已提误 {task_id}")
            return task_id
        except Exception as e:
            logger.error(f"提交异步预测任务失败: {e}")
            return None

    def submit_async_report_generation(self, report_type: str, parameters: Dict[str, Any],
                                       priority: TaskPriority = TaskPriority.NORMAL,
                                       callback: Optional[Callable] = None) -> Optional[str]:
        """
    提交异步报告生成任务

    Args:
    report_type: 报告类型
    parameters: 报告参数
    priority: 任务优先误
    callback: 完成回调函数

    Returns:
    任务ID
    """
        try:
            if not self.async_task_manager:
                logger.warning("异步任务管理器不可用")
                return None
            if report_type == 'risk_report':
                func = self.generate_multi_asset_risk_report
                args = ()
            else:
                logger.error(f"不支持的报告类型: {report_type}")
            return None

            task_id = self.async_task_manager.submit_task(
            task_type=TaskType.REPORT_GENERATION,
            name=f"报告生成: {report_type}",
            func=func,
            args=args,
            kwargs=parameters,
            priority=priority,
            callback=callback,
            timeout=300.0  # 5分钟超时
            )

            logger.info(f"异步报告生成任务已提交 {task_id}")
            return task_id
        except Exception as e:
            logger.error(f"提交异步报告生成任务失败: {e}")
            return None

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        获取内存统计信息

        Returns:
        内存统计信息
        """
        try:
            if not self.memory_optimizer:
                return {
                    'memory_optimizer_available': False,
                    'message': '内存优化器不可用'
                }
            return self.memory_optimizer.get_memory_report()
        except Exception as e:
            logger.error(f"获取内存统计失败: {e}")
            return {
                'memory_optimizer_available': False,
                'error': str(e)
            }

    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
    优化内存使用

    Returns:
    优化结果
    """
        try:
            if not self.memory_optimizer:
                return {
                    'memory_optimizer_available': False,
                    'message': '内存优化器不可用'
                }
            return self.memory_optimizer.optimize_memory_usage()
        except Exception as e:
            logger.error(f"内存优化失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def process_large_portfolio_dataset(self, portfolio_list: List[Dict[str, Any]],
                                        risk_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        处理大型组合数据集（内存优化版本）

    Args:
    portfolio_list: 组合数据列表
    risk_types: 风险类型列表

    Returns:
    处理结果列表
    """
        try:
            if not self.memory_optimizer:
                logger.warning("内存优化器不可用，使用标准处理")
                return self._process_portfolio_list_standard(portfolio_list, risk_types)
            if risk_types is None:
                risk_types = ['var', 'volatility', 'sharpe_ratio']
            # 定义批处理函数
            def batch_processor(batch_portfolios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
                """处理一批组合"""
                results = []
                for portfolio_data in batch_portfolios:
                    try:
                        # 检查缓存
                        cache_key = None
                        if self.cache_manager:
                            portfolio_id = portfolio_data.get('portfolio_id', 'unknown')
                            cache_key = f"portfolio_batch_{portfolio_id}"
                            cached_result = self.cache_manager.get(cache_key)
                        if cached_result:
                            results.append(cached_result)
                            continue

                        # 计算风险指标
                        risk_result = self.calculate_portfolio_risk(portfolio_data)
                        portfolio_result = {
                            'portfolio_id': portfolio_data.get('portfolio_id', 'unknown'),
                            'total_value': risk_result.total_value,
                            'total_risk': risk_result.total_risk,
                            'risk_metrics': {},
                            'timestamp': datetime.now().isoformat()
                        }

                        # 提取所需的风险指标
                        for risk_type in risk_types:
                            if risk_type == 'var':
                                portfolio_result['risk_metrics']['var'] = risk_result.risk_metrics.get(
                                    RiskMetricType.VAR, RiskCalculationResult(RiskMetricType.VAR, 0, 0, 0, datetime.now())).value
                            elif risk_type == 'volatility':
                                # 简化波动率计算
                                portfolio_result['risk_metrics']['volatility'] = 0.02
                            elif risk_type == 'sharpe_ratio':
                                # 简化夏普比率计算
                                portfolio_result['risk_metrics']['sharpe_ratio'] = 1.5

                        # 缓存结果
                        if self.cache_manager and cache_key:
                            self.cache_manager.set(cache_key, portfolio_result, ttl=600)

                        results.append(portfolio_result)
                    except Exception as e:
                        logger.error(f"处理组合失败: {e}")
                        results.append({
                            'portfolio_id': portfolio_data.get('portfolio_id', 'unknown'),
                            'error': str(e)
                        })
                return results

            # 使用内存优化器处理大数据集
            final_results = self.memory_optimizer.process_large_data(
            portfolio_list,
            lambda batch: batch_processor(batch),
            lambda all_results: [item for sublist in all_results for item in sublist]
            )

            logger.info(f"成功处理{len(final_results)} 个组合")
            return final_results
        except Exception as e:
            logger.error(f"批量处理组合失败: {e}")
            return self._process_portfolio_list_standard(portfolio_list, risk_types)

    def _process_portfolio_list_standard(self, portfolio_list: List[Dict[str, Any]],
                                         risk_types: List[str] = None) -> List[Dict[str, Any]]:
        """标准处理组合列表（备选方案）"""
        try:
            if risk_types is None:
                risk_types = ['var', 'volatility', 'sharpe_ratio']

            results = []
            for portfolio_data in portfolio_list:
                try:
                    risk_result = self.calculate_portfolio_risk(portfolio_data)
                    portfolio_result = {
                        'portfolio_id': portfolio_data.get('portfolio_id', 'unknown'),
                        'total_value': risk_result.total_value,
                        'total_risk': risk_result.total_risk,
                        'risk_metrics': {},
                        'timestamp': datetime.now().isoformat()
                    }
                    for risk_type in risk_types:
                        if risk_type == 'var':
                            portfolio_result['risk_metrics']['var'] = risk_result.risk_metrics.get(
                                RiskMetricType.VAR, RiskCalculationResult(RiskMetricType.VAR, 0, 0, 0, datetime.now())).value
                        elif risk_type == 'volatility':
                            portfolio_result['risk_metrics']['volatility'] = 0.02
                        elif risk_type == 'sharpe_ratio':
                            portfolio_result['risk_metrics']['sharpe_ratio'] = 1.5

                    results.append(portfolio_result)
                except Exception as e:
                    logger.error(f"处理组合失败: {e}")
                    results.append({
                        'portfolio_id': portfolio_data.get('portfolio_id', 'unknown'),
                        'error': str(e)
                    })
            return results
        except Exception as e:
            logger.error(f"标准处理组合列表失败: {e}")
            return []

    def cache_risk_calculation_result(self, key: str, result: Any,
                                      pool_type: MemoryPoolType = MemoryPoolType.COMPUTATION_RESULTS) -> bool:
        """
    缓存风险计算结果

    Args:
    key: 缓存误
    result: 计算结果
    pool_type: 内存池类误

    Returns:
    是否缓存成功
    """
        try:
            if not self.memory_optimizer:
                return False
            return self.memory_optimizer.cache_computation_result(key, result, pool_type)
        except Exception as e:
            logger.error(f"缓存计算结果失败: {e}")
            return False

    def get_cached_risk_result(self, key: str,
                               pool_type: MemoryPoolType = MemoryPoolType.COMPUTATION_RESULTS) -> Optional[Any]:
        """
        获取缓存的风险计算结果

    Args:
    key: 缓存误
    pool_type: 内存池类误

    Returns:
    缓存的结误
    """
        try:
            if not self.memory_optimizer:
                return None
            return self.memory_optimizer.get_cached_result(key, pool_type)
        except Exception as e:
            logger.error(f"获取缓存结果失败: {e}")
            return None

    def force_memory_cleanup(self) -> Dict[str, Any]:
        """
        强制内存清理

    Returns:
    清理结果
    """
        try:
            if not self.memory_optimizer:
                return {
                    'memory_optimizer_available': False,
                    'message': '内存优化器不可用'
                }
            return self.memory_optimizer.force_memory_cleanup()
        except Exception as e:
            logger.error(f"强制内存清理失败: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def set_memory_gc_strategy(self, strategy: str) -> bool:
        """
    设置内存垃圾回收策略

    Args:
    strategy: 策略类型('aggressive', 'conservative', 'disabled')

    Returns:
    是否设置成功
    """
        try:
            if not self.memory_optimizer:
                return False
            self.memory_optimizer.set_gc_strategy(strategy)
            logger.info(f"内存GC策略已设置为: {strategy}")
            return True
        except Exception as e:
            logger.error(f"设置GC策略失败: {e}")
            return False

    def get_memory_pool_stats(self) -> Dict[str, Any]:
        """
        获取内存池统计信息

        Returns:
        内存池统计
    """
        try:
            if not self.memory_optimizer:
                return {
                    'memory_optimizer_available': False,
                    'message': '内存优化器不可用'
                }
            return self.memory_optimizer.pool_manager.get_pool_stats()
        except Exception as e:
            logger.error(f"获取内存池统计失败: {e}")
            return {
                'error': str(e)
            }

    def register_asset(self, asset_config: Dict[str, Any]):
        """
        注册资产配置

        Args:
        asset_config: 资产配置字典
        """
        try:
            if not self.multi_asset_manager:
                logger.warning("多资产风险管理器不可用")
                return
            # 转换字典为AssetConfig对象
            config = AssetConfig(
                asset_type=AssetType(asset_config.get('asset_type', 'stock')),
                symbol=asset_config['symbol'],
                currency=asset_config.get('currency', 'CNY'),
                exchange=asset_config.get('exchange', ''),
                contract_size=asset_config.get('contract_size', 1.0),
            tick_size=asset_config.get('tick_size', 0.01),
            multiplier=asset_config.get('multiplier', 1.0),
            margin_requirement=asset_config.get('margin_requirement', 0.0),
            trading_hours=asset_config.get('trading_hours', {})
            )

            self.multi_asset_manager.register_asset(config)
            logger.info(f"资产注册成功: {config.symbol}")
        except Exception as e:
            logger.error(f"资产注册失败: {e}")

    def update_multi_asset_position(self, symbol: str, quantity: float, price: float,
                                   current_price: Optional[float] = None):
        """
        更新多资产持仓

    Args:
        symbol: 资产代码
        quantity: 持仓数量
        price: 平均成本价
        current_price: 当前价格
    """
        try:
            if not self.multi_asset_manager:
                logger.warning("多资产风险管理器不可用")
                return
            self.multi_asset_manager.update_position(symbol, quantity, price, current_price)
            logger.debug(f"多资产持仓更新成功: {symbol}")
        except Exception as e:
            logger.error(f"多资产持仓更新失败: {e}")

    def calculate_multi_asset_portfolio_risk(self, market_data: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        计算多资产组合风险

    Args:
    market_data: 市场数据字典

    Returns:
    多资产风险分析结果
    """
        try:
            if not self.multi_asset_manager:
                logger.warning("多资产风险管理器不可用")
                return {"error": "多资产风险管理器不可用"}
            # 计算多资产风险指标
            risk_metrics = self.multi_asset_manager.calculate_portfolio_risk(market_data)

            # 获取资产配置信息
            allocation = self.multi_asset_manager.get_asset_allocation()

            # 获取风险归因
            attribution = self.multi_asset_manager.get_risk_attribution()
            return {
            'risk_metrics': {
            'total_value': risk_metrics.total_value,
            'total_risk': risk_metrics.total_risk,
            'portfolio_volatility': risk_metrics.portfolio_volatility,
            'portfolio_var': risk_metrics.portfolio_var,
            'diversification_ratio': risk_metrics.diversification_ratio,
            'concentration_index': risk_metrics.concentration_index,
            'asset_class_risk': {k.value: v for k, v in risk_metrics.asset_class_risk.items()},
            'asset_type_risk': {k.value: v for k, v in risk_metrics.asset_type_risk.items()}
            },
            'allocation': allocation,
            'attribution': attribution,
            'correlation_matrix': risk_metrics.correlation_matrix
            }
        except Exception as e:
            logger.error(f"计算多资产组合风险失败: {e}")
            return {"error": str(e)}

    def generate_multi_asset_risk_report(self) -> Dict[str, Any]:
        """
        生成多资产风险报告

        Returns:
        完整风险报告
        """
        try:
            if not self.multi_asset_manager:
                logger.warning("多资产风险管理器不可用")
                return {"error": "多资产风险管理器不可用"}
            return self.multi_asset_manager.generate_risk_report()
        except Exception as e:
            logger.error(f"生成多资产风险报告失败: {e}")
            return {"error": str(e)}

    def get_supported_asset_types(self) -> List[str]:
        """
        获取支持的资产类型

        Returns:
        支持的资产类型列表
        """
        try:
            if MULTI_ASSET_AVAILABLE:
                return [asset_type.value for asset_type in AssetType]
            return []
        except Exception as e:
            logger.error(f"获取支持资产类型失败: {e}")
            return []

    def get_asset_class_info(self) -> Dict[str, List[str]]:
        """
        获取资产大类信息

    Returns:
        资产大类及其包含的资产类型
    """
        try:
            if not MULTI_ASSET_AVAILABLE:
                return {}

            asset_class_info = {}
            for asset_type in AssetType:
                asset_class = self.multi_asset_manager.asset_class_mapping.get(
                    asset_type, AssetClass.ALTERNATIVE)

                class_name = asset_class.value
                if class_name not in asset_class_info:
                    asset_class_info[class_name] = []
                asset_class_info[class_name].append(asset_type.value)
            return asset_class_info
        except Exception as e:
            logger.error(f"获取资产大类信息失败: {e}")
            return {}

    def calculate_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> PortfolioRiskProfile:
        """计算组合风险"""
        with self.lock:
            try:
                portfolio_id = portfolio_data.get('portfolio_id', 'default')
                positions = portfolio_data.get('positions', {})

                # 生成缓存key
                cache_key = None
                if self.cache_manager:
                    cache_key = self.cache_manager.create_cache_key(
                        'portfolio_risk',
                        portfolio_id,
                        str(sorted(positions.items())),
                        portfolio_data.get('prices', {}),
                        portfolio_data.get('historical_data_hash', '')
                    )

                    # 尝试从缓存获取结果
                    cached_result = self.cache_manager.get(cache_key)
                    if cached_result:
                        logger.debug(f"从缓存获取组合风险结果 {portfolio_id}")
                        return cached_result
                    prices = portfolio_data.get('prices', {})

                    # 计算总价值
                    total_value = sum(abs(pos['quantity']) * prices.get(symbol, 0)
                        for symbol, pos in positions.items())

                    # 计算各项风险指标
                    risk_metrics = {}
                    for metric_type in RiskMetricType:
                        if metric_type == RiskMetricType.STRESS_TEST:
                            continue
                        try:
                            result = self._calculate_risk_metric(metric_type, portfolio_data)

                            # 集成AI预测功能
                            if self.ai_prediction_model and self._should_use_ai_prediction(metric_type):
                                ai_prediction = self._get_ai_prediction(
                                    metric_type, portfolio_data, result.value)
                                if ai_prediction:
                                    result.metadata['ai_prediction'] = ai_prediction.predicted_value
                                    result.metadata['ai_confidence_interval'] = ai_prediction.confidence_interval
                                    result.metadata['ai_accuracy'] = ai_prediction.model_accuracy

                            risk_metrics[metric_type] = result
                        except Exception as e:
                            logger.warning(f"计算风险指标 {metric_type.value} 失败: {e}")
                            continue

                    # 计算压力测试
                    stress_test_results = {}
                    if self.config.enable_stress_testing:
                        stress_test_results = self._calculate_stress_test(portfolio_data)

                    # 风险分解
                    risk_decomposition = self._decompose_risk(portfolio_data, risk_metrics)

                    # 总风险（基于VaR）
                    total_risk = risk_metrics.get(RiskMetricType.VAR,
                        RiskCalculationResult(RiskMetricType.VAR, 0, 0, 0, datetime.now())).value

                    profile = PortfolioRiskProfile(
                        portfolio_id=portfolio_id,
                        total_value=total_value,
                        total_risk=total_risk,
                        risk_metrics=risk_metrics,
                        risk_decomposition=risk_decomposition,
            stress_test_results=stress_test_results,
            timestamp=datetime.now(),
            metadata={'calculation_method': 'comprehensive'}
                    )

                    # 存储到分布式缓存
                    if self.cache_manager and cache_key:
                        try:
                            self.cache_manager.set(cache_key, profile, ttl=300)  # 缓存5分钟
                            logger.debug(f"组合风险结果已缓存 {portfolio_id}")
                        except Exception as e:
                            logger.warning(f"缓存存储失败: {e}")

                    # 缓存结果到原有缓存
                    self.risk_cache[portfolio_id].append(profile)
                    return profile
            except Exception as e:
                logger.error(f"计算组合风险失败: {e}")
                raise

    def _calculate_risk_metric(self, metric_type: RiskMetricType,
            portfolio_data: Dict[str, Any]) -> RiskCalculationResult:
        """计算特定风险指标"""
        try:
            if metric_type not in self.risk_models:
                raise ValueError(f"不支持的风险指标类型: {metric_type}")

            calculator = self.risk_models[metric_type]
            value = calculator(portfolio_data)

            # 计算置信区间（如果适用）
            confidence_interval = None
            if metric_type in [RiskMetricType.VAR, RiskMetricType.CVAR]:
                confidence_interval = self._calculate_confidence_interval(
                    metric_type, portfolio_data, value)
            return RiskCalculationResult(
                metric_type=metric_type,
                value=value,
                confidence_level=self.config.confidence_level.value,
                time_horizon=self.config.time_horizon_days,
                timestamp=datetime.now(),
                confidence_interval=confidence_interval
            )
        except Exception as e:
            logger.error(f"计算风险指标失败: {e}")
            return RiskCalculationResult(
                metric_type=metric_type,
                value=0.0,
                confidence_level=self.config.confidence_level.value,
                time_horizon=self.config.time_horizon_days,
                timestamp=datetime.now()
            )

    def _calculate_var(self, portfolio_data: Dict[str, Any]) -> float:
        """计算VaR（风险价值）"""
        try:
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            returns = portfolio_data.get('returns', {})
            if not positions or not returns:
                return 0.0

            # 计算组合收益
            portfolio_returns = []
            weights = []
            for symbol, pos in positions.items():
                if symbol in returns and len(returns[symbol]) > 0:
                    symbol_returns = returns[symbol]
                    weight = abs(pos['quantity']) * prices.get(symbol, 0)
                    weights.append(weight)
                    portfolio_returns.append(symbol_returns)
            if not portfolio_returns:
                return 0.0

            # 计算组合收益
            portfolio_returns = np.array(portfolio_returns).T
            weights = np.array(weights)
            weights = weights / weights.sum()

            # 计算加权组合收益
            weighted_returns = np.dot(portfolio_returns, weights)

            # 计算VaR
            if self.config.enable_monte_carlo:
                var = self._calculate_monte_carlo_var(weighted_returns)
            else:
                var = self._calculate_historical_var(weighted_returns)
            return abs(var)
        except Exception as e:
            logger.error(f"计算VaR失败: {e}")
            return 0.0

    def _calculate_cvar(self, portfolio_data: Dict[str, Any]) -> float:
        """计算CVaR（条件风险价值）"""
        try:
            var = self._calculate_var(portfolio_data)

            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            returns = portfolio_data.get('returns', {})
            if not positions or not returns:
                return var

            # 计算组合收益
            portfolio_returns = []
            weights = []
            for symbol, pos in positions.items():
                if symbol in returns and len(returns[symbol]) > 0:
                    symbol_returns = returns[symbol]
                    weight = abs(pos['quantity']) * prices.get(symbol, 0)
                    weights.append(weight)
                    portfolio_returns.append(symbol_returns)
            if not portfolio_returns:
                return var

            portfolio_returns = np.array(portfolio_returns).T
            weights = np.array(weights)
            weights = weights / weights.sum()
            weighted_returns = np.dot(portfolio_returns, weights)

            # 计算CVaR
            threshold = np.percentile(weighted_returns,
                (1 - self.config.confidence_level.value) * 100)
            tail_returns = weighted_returns[weighted_returns <= threshold]
            if len(tail_returns) > 0:
                cvar = np.mean(tail_returns)
            else:
                cvar = threshold
            return abs(cvar)
        except Exception as e:
            logger.error(f"计算CVaR失败: {e}")
            return abs(var)

    def _calculate_volatility(self, portfolio_data: Dict[str, Any]) -> float:
        """计算波动率"""
        try:
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            returns = portfolio_data.get('returns', {})
            if not positions or not returns:
                return 0.0

            # 计算组合收益
            portfolio_returns = []
            weights = []
            for symbol, pos in positions.items():
                if symbol in returns and len(returns[symbol]) > 0:
                    symbol_returns = returns[symbol]
                    weight = abs(pos['quantity']) * prices.get(symbol, 0)
                    weights.append(weight)
                    portfolio_returns.append(symbol_returns)
            if not portfolio_returns:
                return 0.0

            portfolio_returns = np.array(portfolio_returns).T
            weights = np.array(weights)
            weights = weights / weights.sum()
            weighted_returns = np.dot(portfolio_returns, weights)

            # 年化波动率
            daily_volatility = np.std(weighted_returns, ddof=1)
            annual_volatility = daily_volatility * np.sqrt(252)
            return annual_volatility
        except Exception as e:
            logger.error(f"计算波动率失败: {e}")
            return 0.0

    def _calculate_beta(self, portfolio_data: Dict[str, Any]) -> float:
        """计算贝塔系数"""
        try:
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            returns = portfolio_data.get('returns', {})
            market_returns = portfolio_data.get('market_returns', [])
            if not positions or not returns or not market_returns:
                return 1.0

            # 计算组合收益
            portfolio_returns = []
            weights = []
            for symbol, pos in positions.items():
                if symbol in returns and len(returns[symbol]) > 0:
                    symbol_returns = returns[symbol]
                    weight = abs(pos['quantity']) * prices.get(symbol, 0)
                    weights.append(weight)
                    portfolio_returns.append(symbol_returns)
            if not portfolio_returns:
                return 1.0

            portfolio_returns = np.array(portfolio_returns).T
            weights = np.array(weights)
            weights = weights / weights.sum()
            weighted_returns = np.dot(portfolio_returns, weights)

            # 计算贝塔
            if len(weighted_returns) == len(market_returns):
                covariance = np.cov(weighted_returns, market_returns)[0, 1]
                market_variance = np.var(market_returns, ddof=1)
                beta = covariance / market_variance if market_variance > 0 else 1.0
            else:
                beta = 1.0
            return beta
        except Exception as e:
            logger.error(f"计算贝塔失败: {e}")
            return 1.0

    def _calculate_sharpe_ratio(self, portfolio_data: Dict[str, Any]) -> float:
        """计算夏普比率"""
        try:
            volatility = self._calculate_volatility(portfolio_data)
            if volatility == 0:
                return 0.0

            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            returns = portfolio_data.get('returns', {})
            if not positions or not returns:
                return 0.0

            # 计算组合收益
            portfolio_returns = []
            weights = []
            for symbol, pos in positions.items():
                if symbol in returns and len(returns[symbol]) > 0:
                    symbol_returns = returns[symbol]
                    weight = abs(pos['quantity']) * prices.get(symbol, 0)
                    weights.append(weight)
                    portfolio_returns.append(symbol_returns)
            if not portfolio_returns:
                return 0.0

            portfolio_returns = np.array(portfolio_returns).T
            weights = np.array(weights)
            weights = weights / weights.sum()
            weighted_returns = np.dot(portfolio_returns, weights)

            # 年化收益
            daily_return = np.mean(weighted_returns)
            annual_return = daily_return * 252

            # 夏普比率
            excess_return = annual_return - self.config.risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0.0
            return sharpe_ratio
        except Exception as e:
            logger.error(f"计算夏普比率失败: {e}")
            return 0.0

    def _calculate_max_drawdown(self, portfolio_data: Dict[str, Any]) -> float:
        """计算最大回撤"""
        try:
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            returns = portfolio_data.get('returns', {})
            if not positions or not returns:
                return 0.0

            # 计算组合收益
            portfolio_returns = []
            weights = []
            for symbol, pos in positions.items():
                if symbol in returns and len(returns[symbol]) > 0:
                    symbol_returns = returns[symbol]
                    weight = abs(pos['quantity']) * prices.get(symbol, 0)
                    weights.append(weight)
                    portfolio_returns.append(symbol_returns)
            if not portfolio_returns:
                return 0.0

            portfolio_returns = np.array(portfolio_returns).T
            weights = np.array(weights)
            weights = weights / weights.sum()
            weighted_returns = np.dot(portfolio_returns, weights)

            # 计算累积收益
            cumulative_returns = np.cumprod(1 + weighted_returns)

            # 计算最大回撤
            max_drawdown = 0.0
            peak = cumulative_returns[0]
            for value in cumulative_returns:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            return max_drawdown
        except Exception as e:
            logger.error(f"计算最大回撤失败: {e}")
            return 0.0

    def _calculate_correlation(self, portfolio_data: Dict[str, Any]) -> float:
        """计算相关系数"""
        try:
            positions = portfolio_data.get('positions', {})
            returns = portfolio_data.get('returns', {})
            if len(positions) < 2:
                return 0.0

            # 获取所有资产的收益
            asset_returns = []
            for symbol, pos in positions.items():
                if symbol in returns and len(returns[symbol]) > 0:
                    asset_returns.append(returns[symbol])
            if len(asset_returns) < 2:
                return 0.0

            # 计算平均相关系数
            correlations = []
            for i in range(len(asset_returns)):
                for j in range(i + 1, len(asset_returns)):
                    if len(asset_returns[i]) == len(asset_returns[j]):
                        corr = np.corrcoef(asset_returns[i], asset_returns[j])[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
            return np.mean(correlations) if correlations else 0.0
        except Exception as e:
            logger.error(f"计算相关系数失败: {e}")
            return 0.0

    def _calculate_concentration(self, portfolio_data: Dict[str, Any]) -> float:
        """计算集中度"""
        try:
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            if not positions:
                return 0.0

            # 计算各资产权重
            weights = []
            for symbol, pos in positions.items():
                weight = abs(pos['quantity']) * prices.get(symbol, 0)
                weights.append(weight)
            if not weights:
                return 0.0

            weights = np.array(weights)
            total_value = weights.sum()
            if total_value == 0:
                return 0.0

            # 计算赫芬达尔指数
            normalized_weights = weights / total_value
            hhi = np.sum(normalized_weights ** 2)
            return hhi
        except Exception as e:
            logger.error(f"计算集中度失败: {e}")
            return 0.0

    def _calculate_liquidity(self, portfolio_data: Dict[str, Any]) -> float:
        """计算流动性风险"""
        try:
            positions = portfolio_data.get('positions', {})
            prices = portfolio_data.get('prices', {})
            volumes = portfolio_data.get('volumes', {})
            if not positions:
                return 0.0

            # 计算流动性指标
            liquidity_scores = []
            for symbol, pos in positions.items():
                position_value = abs(pos['quantity']) * prices.get(symbol, 0)
                daily_volume = volumes.get(symbol, 0)
                if daily_volume > 0:
                    # 流动性比率：持仓价值/日均成交量
                    liquidity_ratio = position_value / daily_volume
                    liquidity_scores.append(liquidity_ratio)
            if not liquidity_scores:
                return 0.0

            # 返回平均流动性风险（值越大风险越高）
            return np.mean(liquidity_scores)
        except Exception as e:
            logger.error(f"计算流动性风险失败: {e}")
            return 0.0

    def _calculate_stress_test(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """计算压力测试"""
        try:
            if not self.config.enable_stress_testing:
                return {}

            stress_scenarios = {
                'market_crash': {'market_shock': -0.20, 'volatility_shock': 2.0},
                'interest_rate_spike': {'rate_shock': 0.05, 'duration_impact': -0.10},
                'liquidity_crisis': {'liquidity_shock': -0.50, 'spread_widening': 0.02},
                'correlation_breakdown': {'correlation_shock': 0.30},
                'extreme_volatility': {'volatility_shock': 3.0}
            }

            results = {}
            base_var = self._calculate_var(portfolio_data)
            for scenario_name, shocks in stress_scenarios.items():
                try:
                    # 应用压力情景
                    stressed_data = self._apply_stress_scenario(portfolio_data, shocks)
                    stressed_var = self._calculate_var(stressed_data)

                    # 计算压力测试结果
                    var_change = (stressed_var - base_var) / base_var if base_var > 0 else 0
                    results[scenario_name] = var_change
                except Exception as e:
                    logger.warning(f"压力测试 {scenario_name} 失败: {e}")
                    results[scenario_name] = 0.0
            return results
        except Exception as e:
            logger.error(f"计算压力测试失败: {e}")
            return {}

    def _apply_stress_scenario(self, portfolio_data: Dict[str, Any],
            shocks: Dict[str, float]) -> Dict[str, Any]:
        """应用压力情景"""
        try:
            stressed_data = portfolio_data.copy()

            # 应用市场冲击
            if 'market_shock' in shocks:
                returns = stressed_data.get('returns', {})
                for symbol in returns:
                    if returns[symbol]:
                        returns[symbol] = [r + shocks['market_shock'] for r in returns[symbol]]

            # 应用波动率冲击
            if 'volatility_shock' in shocks:
                returns = stressed_data.get('returns', {})
                for symbol in returns:
                    if returns[symbol]:
                        volatility_multiplier = shocks['volatility_shock']
                        returns[symbol] = [r * volatility_multiplier for r in returns[symbol]]
            return stressed_data
        except Exception as e:
            logger.error(f"应用压力情景失败: {e}")
            return portfolio_data

    def _calculate_monte_carlo_var(self, returns: np.ndarray) -> float:
        """使用蒙特卡洛方法计算VaR"""
        try:
            if len(returns) < 30:
                return self._calculate_historical_var(returns)

            # 拟合分布参数
            mu = np.mean(returns)
            sigma = np.std(returns, ddof=1)

            # 生成蒙特卡洛样本
            np.random.seed(42)  # 固定随机种子以确保可重复性
            samples = np.secrets.normal(mu, sigma, self.config.monte_carlo_simulations)

            # 计算VaR
            var_percentile = (1 - self.config.confidence_level.value) * 100
            var = np.percentile(samples, var_percentile)
            return var
        except Exception as e:
            logger.error(f"蒙特卡洛VaR计算失败: {e}")
            return 0.0

    def _calculate_historical_var(self, returns: np.ndarray) -> float:
        """使用历史模拟方法计算VaR"""
        try:
            if len(returns) == 0:
                return 0.0

            var_percentile = (1 - self.config.confidence_level.value) * 100
            var = np.percentile(returns, var_percentile)
            return var
        except Exception as e:
            logger.error(f"历史VaR计算失败: {e}")
            return 0.0

    def _calculate_confidence_interval(self, metric_type: RiskMetricType,
            portfolio_data: Dict[str, Any],
            point_estimate: float) -> Tuple[float, float]:
        """计算置信区间"""
        try:
            if metric_type not in [RiskMetricType.VAR, RiskMetricType.CVAR]:
                return None

            positions = portfolio_data.get('positions', {})
            returns = portfolio_data.get('returns', {})
            if not positions or not returns:
                return (point_estimate, point_estimate)

            # 使用Bootstrap方法计算置信区间
            portfolio_returns = []
            weights = []
            for symbol, pos in positions.items():
                if symbol in returns and len(returns[symbol]) > 0:
                    symbol_returns = returns[symbol]
                    weight = abs(pos['quantity']) * portfolio_data.get('prices', {}).get(symbol, 0)
                    weights.append(weight)
                    portfolio_returns.append(symbol_returns)
            if not portfolio_returns:
                return (point_estimate, point_estimate)

            portfolio_returns = np.array(portfolio_returns).T
            weights = np.array(weights)
            weights = weights / weights.sum()
            weighted_returns = np.dot(portfolio_returns, weights)

            # Bootstrap采样
            bootstrap_samples = []
            n_bootstrap = 1000
            for _ in range(n_bootstrap):
                indices = np.secrets.choice(len(weighted_returns),
                                            len(weighted_returns), replace=True)
                sample = weighted_returns[indices]
                if metric_type == RiskMetricType.VAR:
                    var_percentile = (1 - self.config.confidence_level.value) * 100
                    sample_var = np.percentile(sample, var_percentile)
                    bootstrap_samples.append(sample_var)
                else:  # CVaR
                    var_percentile = (1 - self.config.confidence_level.value) * 100
                    threshold = np.percentile(sample, var_percentile)
                    tail_returns = sample[sample <= threshold]
                    sample_cvar = np.mean(tail_returns) if len(tail_returns) > 0 else threshold
                    bootstrap_samples.append(sample_cvar)

            # 计算置信区间
            alpha = 0.05  # 95% 置信区间
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = np.percentile(bootstrap_samples, lower_percentile)
            upper_bound = np.percentile(bootstrap_samples, upper_percentile)
            return (lower_bound, upper_bound)
        except Exception as e:
            logger.error(f"计算置信区间失败: {e}")
            return (point_estimate, point_estimate)

    def _decompose_risk(self, portfolio_data: Dict[str, Any],
            risk_metrics: Dict[RiskMetricType, RiskCalculationResult]) -> Dict[str, float]:
        """风险分解"""
        try:
            positions = portfolio_data.get('positions', {})
            if not positions:
                return {}

            # 计算各资产的风险贡献
            risk_contributions = {}
            total_risk = risk_metrics.get(RiskMetricType.VAR,
                RiskCalculationResult(RiskMetricType.VAR, 0, 0, 0, datetime.now())).value
            if total_risk == 0:
                return {}
            for symbol, pos in positions.items():
                # 简化的风险贡献计算
                position_value = abs(pos['quantity']) * \
                                     portfolio_data.get('prices', {}).get(symbol, 0)
                portfolio_value = sum(abs(p['quantity']) * portfolio_data.get('prices', {}).get(s, 0)
                    for s, p in positions.items())
                if portfolio_value > 0:
                    weight = position_value / portfolio_value
                    risk_contributions[symbol] = weight * total_risk
            return risk_contributions
        except Exception as e:
            logger.error(f"风险分解失败: {e}")
            return {}

    def get_calculation_history(self, portfolio_id: str, limit: int = 100) -> List[PortfolioRiskProfile]:
        """获取计算历史"""
        with self.lock:
            if portfolio_id in self.risk_cache:
                return list(self.risk_cache[portfolio_id])[-limit:]
            return []
    def clear_cache(self, portfolio_id: Optional[str] = None):
        """清理缓存"""
        with self.lock:
            if portfolio_id:
                if portfolio_id in self.risk_cache:
                    del self.risk_cache[portfolio_id]
            else:
                self.risk_cache.clear()

    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        with self.lock:
            total_calculations = sum(len(cache) for cache in self.risk_cache.values())
            total_portfolios = len(self.risk_cache)
            return {
                'total_calculations': total_calculations,
                'total_portfolios': total_portfolios,
                'cache_size': sum(len(cache) for cache in self.risk_cache.values()),
                'last_calculation': datetime.now().isoformat() if total_calculations > 0 else None
            }

    def calculate_risk(self, metric_type: RiskMetricType, data: Any) -> RiskCalculationResult:
        """计算指定类型的风险指标"""
        try:
            # 如果data是DataFrame，转换为字典格式
            if hasattr(data, 'to_dict'):
                # 对于DataFrame，提取相关数据
                returns_data = data.get('returns', data.get('close', []))
                prices_data = data.get('prices', data.get('close', []))
                volumes_data = data.get('volumes', data.get('volume', []))

                portfolio_data = {
                    'returns': returns_data.tolist() if hasattr(returns_data, 'tolist') else returns_data,
                    'prices': prices_data.tolist() if hasattr(prices_data, 'tolist') else prices_data,
                    'volumes': volumes_data.tolist() if hasattr(volumes_data, 'tolist') else volumes_data,
                    'market_data': data.to_dict('records') if hasattr(data, 'to_dict') else {}
                }
            elif isinstance(data, dict):
                portfolio_data = data
            else:
                # 如果是其他类型，创建一个基本的portfolio_data
                portfolio_data = {
                    'returns': [0.01, 0.02, -0.01, 0.015, -0.005] if not isinstance(data, list) else data,
                    'prices': [100.0, 101.0, 100.0, 101.5, 101.0] if not isinstance(data, list) else [100 + i for i in range(len(data))],
                }

            # 计算风险指标
            if metric_type == RiskMetricType.VOLATILITY:
                value = self._calculate_volatility(portfolio_data)
            elif metric_type == RiskMetricType.VAR:
                value = self._calculate_var(portfolio_data)
            elif metric_type == RiskMetricType.CVAR:
                value = self._calculate_cvar(portfolio_data)
            else:
                # 对于其他指标，使用默认计算
                value = self._calculate_risk_metric(metric_type, portfolio_data)

            # 创建结果对象
            result = RiskCalculationResult(
                metric_type=metric_type,
                value=value,
                confidence_level=self.config.confidence_level if hasattr(
                    self.config, 'confidence_level') else 0.95,
                time_horizon=self.config.time_horizon_days if hasattr(
                    self.config, 'time_horizon_days') else 30,
                timestamp=datetime.now(),
                confidence_interval=(value * 0.9, value * 1.1),  # 简化的置信区间
                metadata={'calculation_method': 'historical',
                    'sample_size': len(portfolio_data.get('returns', []))}
            )

            return result

        except Exception as e:
            logger.error(f"计算风险指标失败 {metric_type}: {e}")
            # 返回默认结果
            return RiskCalculationResult(
                metric_type=metric_type,
                value=0.0,
                confidence_level=0.95,
                time_horizon=30,
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )

    def batch_calculate(self, metric_types: List[RiskMetricType], data: Any) -> List[RiskCalculationResult]:
        """批量计算多个风险指标"""
        results = []
        for metric_type in metric_types:
            try:
                result = self.calculate_risk(metric_type, data)
                results.append(result)
            except Exception as e:
                logger.error(f"批量计算失败 {metric_type}: {e}")
                # 返回错误结果
                error_result = RiskCalculationResult(
                    metric_type=metric_type,
                    value=0.0,
                    confidence_level=0.95,
                    time_horizon=30,
                    timestamp=datetime.now(),
                    metadata={'error': str(e)}
                )
                results.append(error_result)

        return results

    def shutdown(self):
        """关闭风险计算引擎"""
        logger.info("正在关闭风险计算引擎...")
        try:
            # 清理缓存
            self.clear_cache()

            # 停止实时风险监控
            if hasattr(self, 'real_time_monitor') and self.real_time_monitor:
                try:
                    if hasattr(self.real_time_monitor, 'stop_monitoring'):
                        self.real_time_monitor.stop_monitoring()
                except Exception as e:
                    logger.warning(f"停止实时监控失败: {e}")

            # 停止异步任务管理器
            if hasattr(self, 'task_manager') and self.task_manager:
                try:
                    if hasattr(self.task_manager, 'stop'):
                        self.task_manager.stop(timeout=2.0)
                except Exception as e:
                    logger.warning(f"停止任务管理器失败: {e}")

            # 清理线程池（如果有的话）
            if hasattr(self, 'executor') and self.executor:
                self.executor.shutdown(wait=True)

            # 清理其他资源
            self.risk_cache.clear()

            logger.info("风险计算引擎已关闭")
        except Exception as e:
            logger.error(f"关闭风险计算引擎时出错: {e}")

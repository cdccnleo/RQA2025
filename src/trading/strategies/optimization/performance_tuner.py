import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging
from functools import wraps
from memory_profiler import memory_usage
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标收集"""
    latency_ms: float
    throughput: float
    cpu_usage: float
    memory_mb: float
    error_rate: float

def profile_performance(func):
    """性能分析装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        mem_usage = memory_usage((func, args, kwargs), interval=0.1, timeout=1)
        end_time = time.perf_counter()

        metrics = PerformanceMetrics(
            latency_ms=(end_time - start_time) * 1000,
            throughput=1/(end_time - start_time) if (end_time - start_time) > 0 else float('inf'),
            cpu_usage=np.mean(mem_usage),
            memory_mb=max(mem_usage) if mem_usage else 0,
            error_rate=0.0
        )

        logger.info(f"Function {func.__name__} performance: {metrics}")
        return metrics
    return wrapper

class DataLayerOptimizer:
    """数据层性能优化"""

    def __init__(self, data_manager):
        self.data_manager = data_manager

    @profile_performance
    def optimize_data_loading(self, symbols: List[str], batch_size: int = 100) -> Dict:
        """优化数据批量加载"""
        results = {}
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i+batch_size]
                futures.append(executor.submit(self.data_manager.load_data, batch))

            for future in futures:
                results.update(future.result())
        return results

class FeatureLayerOptimizer:
    """特征层性能优化"""

    def __init__(self, feature_manager):
        self.feature_manager = feature_manager

    @profile_performance
    def parallel_feature_generation(self, data: Dict, n_workers: int = 4) -> pd.DataFrame:
        """并行特征生成"""
        chunks = np.array_split(data, n_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self.feature_manager.generate_features, chunk)
                      for chunk in chunks]
            results = pd.concat([f.result() for f in futures])
        return results

class ModelLayerOptimizer:
    """模型层性能优化"""

    def __init__(self, model_manager):
        self.model_manager = model_manager

    @profile_performance
    def optimize_inference(self, features: pd.DataFrame, batch_size: int = 32) -> np.ndarray:
        """批量推理优化"""
        predictions = []
        for i in range(0, len(features), batch_size):
            batch = features.iloc[i:i+batch_size]
            predictions.append(self.model_manager.predict(batch))
        return np.concatenate(predictions)

class TradingLayerOptimizer:
    """交易层性能优化"""

    def __init__(self, trading_strategy):
        self.trading_strategy = trading_strategy

    @profile_performance
    def optimize_signal_generation(self, predictions: np.ndarray) -> Dict:
        """信号生成优化"""
        return self.trading_strategy.generate_signals(predictions)

class PerformanceMonitor:
    """实时性能监控"""

    def __init__(self):
        self.metrics_history = []
        self.alert_rules = {
            'latency_ms': {'threshold': 100, 'window': '5m'},
            'cpu_usage': {'threshold': 90, 'window': '1m'},
            'memory_mb': {'threshold': 1024, 'window': '10m'}
        }

    def update_metrics(self, metrics: PerformanceMetrics):
        """更新性能指标"""
        self.metrics_history.append(metrics)
        self._check_alerts(metrics)

    def _check_alerts(self, metrics: PerformanceMetrics):
        """检查性能告警"""
        for field, rule in self.alert_rules.items():
            value = getattr(metrics, field)
            if value > rule['threshold']:
                logger.warning(f"Performance alert: {field}={value} exceeds {rule['threshold']}")

    def get_performance_report(self) -> Dict:
        """生成性能报告"""
        if not self.metrics_history:
            return {}

        latest = self.metrics_history[-1]
        return {
            'latency': f"{latest.latency_ms:.2f}ms",
            'throughput': f"{latest.throughput:.2f}/s",
            'cpu': f"{latest.cpu_usage:.1f}%",
            'memory': f"{latest.memory_mb:.1f}MB",
            'error_rate': f"{latest.error_rate:.2%}"
        }

def main():
    """性能优化主流程"""
    # 初始化各模块
    from src.data.data_manager import DataManager
    from src.features.feature_manager import FeatureManager
    from src.models.model_manager import ModelManager
    from src.trading.enhanced_trading_strategy import EnhancedTradingStrategy

    data_mgr = DataManager()
    feature_mgr = FeatureManager()
    model_mgr = ModelManager()
    strategy = EnhancedTradingStrategy()

    # 初始化优化器
    data_optimizer = DataLayerOptimizer(data_mgr)
    feature_optimizer = FeatureLayerOptimizer(feature_mgr)
    model_optimizer = ModelLayerOptimizer(model_mgr)
    trading_optimizer = TradingLayerOptimizer(strategy)
    monitor = PerformanceMonitor()

    # 测试数据
    symbols = ['600519.SH', '000858.SZ', '601318.SH'] * 100

    # 执行优化流程
    data_metrics = data_optimizer.optimize_data_loading(symbols)
    feature_metrics = feature_optimizer.parallel_feature_generation(data_metrics)
    model_metrics = model_optimizer.optimize_inference(feature_metrics)
    trading_metrics = trading_optimizer.optimize_signal_generation(model_metrics)

    # 监控报告
    for metrics in [data_metrics, feature_metrics, model_metrics, trading_metrics]:
        monitor.update_metrics(metrics)

    print("Performance optimization completed:")
    print(monitor.get_performance_report())

if __name__ == "__main__":
    main()

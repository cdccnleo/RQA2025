#!/usr/bin/env python3
"""
CPU性能分析和优化工具

用于分析CPU热点代码、识别性能瓶颈、实施优化策略。
支持策略计算并行化、GPU加速、缓存优化等。
"""

import time
import psutil
import concurrent.futures
import cProfile
import numpy as np
from typing import Dict, Any, List, Callable
from functools import wraps
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """性能分析器"""

    def __init__(self):
        self.profiles = {}
        self.baselines = {}

    def profile_function(self, func: Callable) -> Callable:
        """函数性能分析装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()

            start_time = time.time()
            start_cpu = psutil.cpu_percent(interval=None)
            start_memory = psutil.virtual_memory().percent

            result = func(*args, **kwargs)

            end_time = time.time()
            end_cpu = psutil.cpu_percent(interval=None)
            end_memory = psutil.virtual_memory().percent

            profiler.disable()

            execution_time = end_time - start_time
            cpu_usage = (start_cpu + end_cpu) / 2
            memory_delta = end_memory - start_memory

            func_name = f"{func.__module__}.{func.__name__}"

            self.profiles[func_name] = {
                "execution_time": execution_time,
                "cpu_usage": cpu_usage,
                "memory_delta": memory_delta,
                "call_count": 1,
                "timestamp": time.time()
            }

            logger.info(
                f"性能分析 - {func_name}: {execution_time:.4f}s, CPU: {cpu_usage:.1f}%, 内存变化: {memory_delta:.1f}%")

            return result

        return wrapper

    def analyze_hotspots(self) -> Dict[str, Any]:
        """分析热点代码"""
        if not self.profiles:
            return {"error": "没有性能分析数据"}

        # 按执行时间排序
        sorted_by_time = sorted(self.profiles.items(),
                                key=lambda x: x[1]["execution_time"], reverse=True)
        sorted_by_cpu = sorted(self.profiles.items(), key=lambda x: x[1]["cpu_usage"], reverse=True)

        hotspots = {
            "by_execution_time": sorted_by_time[:10],
            "by_cpu_usage": sorted_by_cpu[:10],
            "total_functions": len(self.profiles),
            "analysis_time": time.time()
        }

        return hotspots

    def get_optimization_suggestions(self) -> List[str]:
        """生成优化建议"""
        suggestions = []

        if not self.profiles:
            return ["需要先运行性能分析来获取优化建议"]

        hotspots = self.analyze_hotspots()

        # 基于执行时间的建议
        slow_functions = [name for name, data in hotspots["by_execution_time"][:3]]
        if slow_functions:
            suggestions.append(f"⚡ 考虑优化慢速函数: {', '.join(slow_functions[:3])}")

        # 基于CPU使用率的建议
        cpu_intensive = [name for name, data in hotspots["by_cpu_usage"][:3]]
        if cpu_intensive:
            suggestions.append(f"🖥️ CPU密集型函数: {', '.join(cpu_intensive[:3])} - 考虑并行化")

        # 通用优化建议
        suggestions.extend([
            "🔄 实施算法并行化处理",
            "🎯 启用GPU加速计算",
            "💾 优化缓存策略和命中率",
            "📊 实施内存池管理机制",
            "⚡ 使用NumPy向量化操作",
            "🔧 配置适当的线程/进程池大小"
        ])

        return suggestions


class CPUOptimizer:
    """CPU优化器"""

    def __init__(self):
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=psutil.cpu_count())
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=psutil.cpu_count())

    def parallelize_computation(self, func: Callable, data_list: List[Any], method: str = "thread") -> List[Any]:
        """并行化计算"""
        if method == "thread":
            pool = self.thread_pool
        elif method == "process":
            pool = self.process_pool
        else:
            raise ValueError("method必须是'thread'或'process'")

        # 提交任务
        futures = [pool.submit(func, data) for data in data_list]

        # 收集结果
        results = []
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"并行计算出错: {e}")
                results.append(None)

        return results

    def vectorize_operations(self, data: np.ndarray, operation: str) -> np.ndarray:
        """向量化操作"""
        if operation == "mean":
            return np.mean(data, axis=0)
        elif operation == "std":
            return np.std(data, axis=0)
        elif operation == "sum":
            return np.sum(data, axis=0)
        elif operation == "normalize":
            return (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        else:
            raise ValueError(f"不支持的操作: {operation}")

    def optimize_cache_strategy(self, cache_manager, strategy: str = "lru") -> Dict[str, Any]:
        """优化缓存策略"""
        if strategy == "lru":
            # LRU策略优化
            if hasattr(cache_manager, 'cleanup'):
                removed_items = cache_manager.cleanup()
                return {"strategy": "lru", "removed_items": removed_items}

        return {"strategy": strategy, "status": "optimized"}

    def monitor_cpu_usage(self, duration: int = 60) -> Dict[str, Any]:
        """监控CPU使用情况"""
        logger.info(f"开始监控CPU使用情况，持续{duration}秒...")

        cpu_usage = []
        memory_usage = []

        start_time = time.time()
        while time.time() - start_time < duration:
            cpu_usage.append(psutil.cpu_percent(interval=1))
            memory_usage.append(psutil.virtual_memory().percent)

        return {
            "duration": duration,
            "cpu_avg": np.mean(cpu_usage),
            "cpu_max": np.max(cpu_usage),
            "cpu_min": np.min(cpu_usage),
            "memory_avg": np.mean(memory_usage),
            "memory_max": np.max(memory_usage),
            "samples": len(cpu_usage)
        }


class GPUAccelerator:
    """GPU加速器"""

    def __init__(self):
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self) -> bool:
        """检查GPU可用性"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf
                return len(tf.config.list_physical_devices('GPU')) > 0
            except ImportError:
                return False

    def move_to_gpu(self, data):
        """将数据移动到GPU"""
        if not self.gpu_available:
            logger.warning("GPU不可用，使用CPU处理")
            return data

        try:
            import torch
            if isinstance(data, torch.Tensor):
                return data.cuda()
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).cuda()
        except ImportError:
            pass

        try:
            import tensorflow as tf
            if isinstance(data, tf.Tensor):
                return data
        except ImportError:
            pass

        return data

    def gpu_matrix_multiplication(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """GPU矩阵乘法"""
        if not self.gpu_available:
            return np.dot(a, b)

        try:
            import torch
            a_gpu = torch.from_numpy(a).cuda()
            b_gpu = torch.from_numpy(b).cuda()
            result_gpu = torch.mm(a_gpu, b_gpu)
            return result_gpu.cpu().numpy()
        except ImportError:
            return np.dot(a, b)


class StrategyComputationOptimizer:
    """策略计算优化器"""

    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.cpu_optimizer = CPUOptimizer()
        self.gpu_accelerator = GPUAccelerator()

    def optimize_portfolio_calculation(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化投资组合计算"""
        logger.info("开始优化投资组合计算...")

        # 并行化资产收益率计算
        assets = portfolio_data.get("assets", [])
        returns_data = portfolio_data.get("returns", [])

        if not assets or returns_data.size == 0:
            return {"error": "缺少资产或收益率数据"}

        # 使用并行计算计算各资产统计指标
        @self.profiler.profile_function
        def calculate_asset_stats(asset_returns):
            return {
                "mean": np.mean(asset_returns),
                "std": np.std(asset_returns),
                "sharpe": np.mean(asset_returns) / np.std(asset_returns) if np.std(asset_returns) > 0 else 0
            }

        asset_stats = self.cpu_optimizer.parallelize_computation(
            calculate_asset_stats,
            [returns_data[:, i] for i in range(len(assets))],
            method="thread"
        )

        # GPU加速协方差矩阵计算
        cov_matrix = self.gpu_accelerator.gpu_matrix_multiplication(
            returns_data.T, returns_data
        ) / len(returns_data)

        # 优化投资组合权重计算
        @self.profiler.profile_function
        def optimize_portfolio(weights, cov_matrix, returns):
            portfolio_return = np.dot(weights, np.mean(returns, axis=0))
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            return {
                "weights": weights,
                "return": portfolio_return,
                "risk": portfolio_risk,
                "sharpe": sharpe_ratio
            }

        # 简单的等权重优化作为示例
        n_assets = len(assets)
        equal_weights = np.ones(n_assets) / n_assets

        result = optimize_portfolio(equal_weights, cov_matrix, returns_data)

        return {
            "optimized_portfolio": result,
            "asset_statistics": asset_stats,
            "covariance_matrix": cov_matrix,
            "optimization_method": "equal_weight",
            "performance_profile": self.profiler.profiles
        }

    def optimize_signal_generation(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化信号生成"""
        logger.info("开始优化信号生成...")

        # 并行化技术指标计算
        @self.profiler.profile_function
        def calculate_technical_indicators(price_data):
            close_prices = price_data.get("close", [])

            # 计算多个技术指标
            sma_5 = self.cpu_optimizer.vectorize_operations(
                np.array([close_prices[max(0, i-5):i] for i in range(len(close_prices))]),
                "mean"
            )

            rsi = self._calculate_rsi_vectorized(np.array(close_prices))

            return {
                "sma_5": sma_5[-1] if len(sma_5) > 0 else 0,
                "rsi": rsi[-1] if len(rsi) > 0 else 50
            }

        # 并行处理多个资产
        assets_data = market_data.get("assets", [])
        signals = self.cpu_optimizer.parallelize_computation(
            calculate_technical_indicators,
            assets_data,
            method="thread"
        )

        # 生成交易信号
        @self.profiler.profile_function
        def generate_signals(technical_data):
            signals = []
            for data in technical_data:
                if data and "sma_5" in data and "rsi" in data:
                    sma = data["sma_5"]
                    rsi = data["rsi"]

                    # 简单的信号生成逻辑
                    if rsi < 30:  # 超卖
                        signal = "BUY"
                    elif rsi > 70:  # 超买
                        signal = "SELL"
                    else:
                        signal = "HOLD"

                    signals.append({
                        "signal": signal,
                        "strength": abs(50 - rsi) / 50,  # 信号强度
                        "indicators": data
                    })
                else:
                    signals.append({"signal": "HOLD", "strength": 0, "indicators": {}})

            return signals

        final_signals = generate_signals(signals)

        return {
            "signals": final_signals,
            "total_assets": len(assets_data),
            "signal_distribution": {
                "BUY": len([s for s in final_signals if s["signal"] == "BUY"]),
                "SELL": len([s for s in final_signals if s["signal"] == "SELL"]),
                "HOLD": len([s for s in final_signals if s["signal"] == "HOLD"])
            },
            "performance_profile": self.profiler.profiles
        }

    def _calculate_rsi_vectorized(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """向量化RSI计算"""
        if len(prices) < period:
            return np.array([50.0] * len(prices))

        # 计算价格变化
        delta = np.diff(prices)

        # 分离上涨和下跌
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        # 计算平均涨幅和跌幅
        avg_gain = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period)/period, mode='valid')

        # 计算RS和RSI
        rs = avg_gain / np.where(avg_loss == 0, 1, avg_loss)
        rsi = 100 - (100 / (1 + rs))

        # 填充初始值
        rsi_full = np.full(len(prices), 50.0)
        rsi_full[period:] = rsi

        return rsi_full


def main():
    """主函数"""
    print("🚀 CPU/内存性能优化专项 - CPU性能分析")
    print("=" * 60)

    # 创建优化器实例
    optimizer = StrategyComputationOptimizer()

    # 1. 分析当前CPU热点
    print("\n1. 执行性能分析...")

    # 创建示例数据进行分析
    np.random.seed(42)

    # 投资组合优化示例
    portfolio_data = {
        "assets": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        "returns": np.random.normal(0.001, 0.02, (1000, 5))  # 1000天，5个资产
    }

    print("   📊 分析投资组合计算性能...")
    portfolio_result = optimizer.optimize_portfolio_calculation(portfolio_data)

    # 信号生成示例
    market_data = {
        "assets": [
            {"close": np.random.uniform(100, 200, 100).tolist()} for _ in range(10)
        ]
    }

    print("   📊 分析信号生成性能...")
    signal_result = optimizer.optimize_signal_generation(market_data)

    # 2. 生成性能报告
    print("\n2. 生成性能分析报告...")

    # 分析热点代码
    hotspots = optimizer.profiler.analyze_hotspots()

    # 生成优化建议
    suggestions = optimizer.profiler.get_optimization_suggestions()

    # CPU监控
    cpu_monitor = optimizer.cpu_optimizer.monitor_cpu_usage(10)  # 10秒监控

    # 3. 输出报告
    print("\n📊 性能分析报告")
    print("-" * 50)

    if portfolio_result and "performance_profile" in portfolio_result:
        profile = portfolio_result["performance_profile"]
        print(f"投资组合计算函数数量: {len(profile)}")

        for func_name, data in list(profile.items())[:3]:
            print(
                f"      {func_name}: {data['execution_time']:.4f}s, CPU: {data['cpu_usage']:.1f}%")
    print("\n🔥 热点代码分析:")
    if "by_execution_time" in hotspots:
        print("   最耗时函数:")
        for i, (func_name, data) in enumerate(hotspots["by_execution_time"][:3], 1):
            print(f"      {i}. {func_name}: {data['execution_time']:.4f}s")
    print("\n🖥️ CPU监控结果:")
    print(f"   CPU平均使用: {cpu_monitor['cpu_avg']:.1f}%")
    print(f"   CPU峰值使用: {cpu_monitor['cpu_max']:.1f}%")
    print(f"   内存平均使用: {cpu_monitor['memory_avg']:.1f}%")

    print("\n💡 优化建议:")
    for i, suggestion in enumerate(suggestions[:5], 1):
        print(f"   {i}. {suggestion}")

    # 4. 保存详细报告
    report = {
        "portfolio_optimization": portfolio_result,
        "signal_generation": signal_result,
        "hotspots_analysis": hotspots,
        "optimization_suggestions": suggestions,
        "cpu_monitoring": cpu_monitor,
        "gpu_available": optimizer.gpu_accelerator.gpu_available,
        "report_generated": time.time()
    }

    with open("cpu_performance_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print("\n📄 详细报告已保存到: cpu_performance_report.json")
    print("\n✅ CPU性能分析完成！")


if __name__ == "__main__":
    main()

"""
性能测试模块
用于测试和比较各个优化模块的性能提升
"""

import logging
import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

# 使用统一日志系统
logger = logging.getLogger(__name__)

# 延迟导入模块
try:
    from .redis_cache import RedisCacheManager
except ImportError:
    logger.warning("Redis缓存模块不可用")
    RedisCacheManager = None

try:
    from .postgresql_persistence import get_db_connection, return_db_connection
except ImportError:
    logger.warning("PostgreSQL持久化模块不可用")
    get_db_connection = None
    return_db_connection = None

try:
    from .data_compression import DataCompressor
except ImportError:
    logger.warning("数据压缩模块不可用")
    DataCompressor = None

try:
    from .historical_trend_analysis import HistoricalTrendAnalyzer
except ImportError:
    logger.warning("历史趋势分析模块不可用")
    HistoricalTrendAnalyzer = None

try:
    from .performance_anomaly_detection import AnomalyDetector
except ImportError:
    logger.warning("性能异常检测模块不可用")
    AnomalyDetector = None


class PerformanceTester:
    """
    性能测试器类
    负责测试各个优化模块的性能
    """
    
    def __init__(self):
        """
        初始化性能测试器
        """
        self.test_results = []
        self.redis_cache = RedisCacheManager() if RedisCacheManager else None
        self.data_compressor = DataCompressor() if DataCompressor else None
    
    def test_redis_cache_performance(self, iterations: int = 1000) -> Dict[str, Any]:
        """
        测试Redis缓存性能
        
        Args:
            iterations: 测试迭代次数
        
        Returns:
            测试结果
        """
        if not self.redis_cache:
            return {
                "test": "redis_cache",
                "status": "skipped",
                "reason": "Redis缓存模块不可用"
            }
        
        start_time = time.time()
        cache_hits = 0
        cache_misses = 0
        
        # 测试数据
        test_key = "test:strategy:performance"
        test_data = {
            "strategy_id": "test_strategy",
            "returns": [random.uniform(-0.02, 0.02) for _ in range(100)],
            "metrics": {
                "sharpe": random.uniform(0.5, 3.0),
                "max_drawdown": random.uniform(-0.2, -0.05),
                "win_rate": random.uniform(0.4, 0.7)
            }
        }
        
        # 第一次写入缓存
        self.redis_cache.set(test_key, test_data, expire=3600)
        
        # 测试读写性能
        for i in range(iterations):
            # 随机选择读写操作
            if random.random() < 0.7:  # 70%读操作
                # 读取缓存
                data = self.redis_cache.get(test_key)
                if data:
                    cache_hits += 1
                else:
                    cache_misses += 1
            else:  # 30%写操作
                # 更新缓存
                test_data["metrics"]["sharpe"] = random.uniform(0.5, 3.0)
                self.redis_cache.set(test_key, test_data, expire=3600)
        
        end_time = time.time()
        duration = end_time - start_time
        operations_per_second = iterations / duration
        
        result = {
            "test": "redis_cache",
            "status": "completed",
            "iterations": iterations,
            "duration": duration,
            "operations_per_second": operations_per_second,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "hit_rate": cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        }
        
        self.test_results.append(result)
        return result
    
    def test_database_query_performance(self, iterations: int = 100) -> Dict[str, Any]:
        """
        测试数据库查询性能
        
        Args:
            iterations: 测试迭代次数
        
        Returns:
            测试结果
        """
        if not get_db_connection:
            return {
                "test": "database_query",
                "status": "skipped",
                "reason": "PostgreSQL持久化模块不可用"
            }
        
        start_time = time.time()
        successful_queries = 0
        failed_queries = 0
        
        for i in range(iterations):
            conn = None
            try:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    
                    # 测试查询：获取策略性能数据
                    strategy_id = f"test_strategy_{random.randint(1, 100)}"
                    cursor.execute("""
                        SELECT * FROM backtest_results 
                        WHERE strategy_id = %s 
                        ORDER BY created_at DESC 
                        LIMIT 10
                    """, (strategy_id,))
                    
                    # 获取结果
                    results = cursor.fetchall()
                    successful_queries += 1
                    cursor.close()
                else:
                    failed_queries += 1
            except Exception as e:
                logger.error(f"数据库查询测试失败: {e}")
                failed_queries += 1
            finally:
                if conn:
                    return_db_connection(conn)
        
        end_time = time.time()
        duration = end_time - start_time
        queries_per_second = iterations / duration
        
        result = {
            "test": "database_query",
            "status": "completed",
            "iterations": iterations,
            "duration": duration,
            "queries_per_second": queries_per_second,
            "successful_queries": successful_queries,
            "failed_queries": failed_queries,
            "success_rate": successful_queries / iterations
        }
        
        self.test_results.append(result)
        return result
    
    def test_data_compression_performance(self, iterations: int = 1000) -> Dict[str, Any]:
        """
        测试数据压缩性能
        
        Args:
            iterations: 测试迭代次数
        
        Returns:
            测试结果
        """
        if not self.data_compressor:
            return {
                "test": "data_compression",
                "status": "skipped",
                "reason": "数据压缩模块不可用"
            }
        
        # 生成测试数据
        test_data = {
            "strategy_id": "test_strategy",
            "returns": [random.uniform(-0.02, 0.02) for _ in range(1000)],
            "metrics": {
                "sharpe": random.uniform(0.5, 3.0),
                "max_drawdown": random.uniform(-0.2, -0.05),
                "win_rate": random.uniform(0.4, 0.7),
                "avg_win": random.uniform(0.01, 0.05),
                "avg_loss": random.uniform(-0.05, -0.01),
                "trades_per_day": random.uniform(1, 50)
            },
            "trades": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "price": random.uniform(100, 200),
                    "quantity": random.uniform(1, 10),
                    "direction": random.choice(["buy", "sell"])
                }
                for _ in range(100)
            ]
        }
        
        # 测试压缩性能
        compression_start = time.time()
        compressed_data = None
        
        for i in range(iterations):
            compressed_data = self.data_compressor.compress_json(test_data)
        
        compression_end = time.time()
        compression_duration = compression_end - compression_start
        
        # 测试解压缩性能
        decompression_start = time.time()
        decompressed_data = None
        
        for i in range(iterations):
            decompressed_data = self.data_compressor.decompress_json(compressed_data)
        
        decompression_end = time.time()
        decompression_duration = decompression_end - decompression_start
        
        # 计算压缩率
        original_size = len(json.dumps(test_data))
        compressed_size = len(compressed_data)
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        
        result = {
            "test": "data_compression",
            "status": "completed",
            "iterations": iterations,
            "compression_duration": compression_duration,
            "decompression_duration": decompression_duration,
            "total_duration": compression_duration + decompression_duration,
            "original_size": original_size,
            "compressed_size": compressed_size,
            "compression_ratio": compression_ratio,
            "compression_speed": iterations / compression_duration,
            "decompression_speed": iterations / decompression_duration
        }
        
        self.test_results.append(result)
        return result
    
    def test_trend_analysis_performance(self, iterations: int = 100) -> Dict[str, Any]:
        """
        测试趋势分析性能
        
        Args:
            iterations: 测试迭代次数
        
        Returns:
            测试结果
        """
        if not HistoricalTrendAnalyzer:
            return {
                "test": "trend_analysis",
                "status": "skipped",
                "reason": "历史趋势分析模块不可用"
            }
        
        start_time = time.time()
        successful_analyses = 0
        
        # 生成测试数据
        test_returns = [random.uniform(-0.02, 0.02) for _ in range(100)]
        test_timestamps = [datetime.now() - timedelta(days=i) for i in range(100)]
        
        for i in range(iterations):
            try:
                # 测试趋势分析
                analysis = HistoricalTrendAnalyzer.analyze_trend(
                    "test_strategy", "return", test_returns, test_timestamps
                )
                successful_analyses += 1
            except Exception as e:
                logger.error(f"趋势分析测试失败: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        analyses_per_second = iterations / duration
        
        result = {
            "test": "trend_analysis",
            "status": "completed",
            "iterations": iterations,
            "duration": duration,
            "analyses_per_second": analyses_per_second,
            "successful_analyses": successful_analyses,
            "success_rate": successful_analyses / iterations
        }
        
        self.test_results.append(result)
        return result
    
    def test_anomaly_detection_performance(self, iterations: int = 100) -> Dict[str, Any]:
        """
        测试异常检测性能
        
        Args:
            iterations: 测试迭代次数
        
        Returns:
            测试结果
        """
        if not AnomalyDetector:
            return {
                "test": "anomaly_detection",
                "status": "skipped",
                "reason": "性能异常检测模块不可用"
            }
        
        start_time = time.time()
        detected_anomalies = 0
        
        # 生成正常历史数据
        historical_returns = [random.uniform(-0.02, 0.02) for _ in range(50)]
        
        for i in range(iterations):
            # 随机生成正常值或异常值
            if random.random() < 0.2:  # 20%异常值
                test_value = random.uniform(0.05, 0.15)  # 异常高的收益
            else:  # 80%正常值
                test_value = random.uniform(-0.02, 0.02)
            
            # 测试异常检测
            result = AnomalyDetector.detect_z_score_anomaly(historical_returns, test_value)
            if result["is_anomaly"]:
                detected_anomalies += 1
        
        end_time = time.time()
        duration = end_time - start_time
        detections_per_second = iterations / duration
        
        result = {
            "test": "anomaly_detection",
            "status": "completed",
            "iterations": iterations,
            "duration": duration,
            "detections_per_second": detections_per_second,
            "detected_anomalies": detected_anomalies,
            "anomaly_rate": detected_anomalies / iterations
        }
        
        self.test_results.append(result)
        return result
    
    def run_all_tests(self, iterations: Dict[str, int] = None) -> List[Dict[str, Any]]:
        """
        运行所有性能测试
        
        Args:
            iterations: 各测试的迭代次数
        
        Returns:
            测试结果列表
        """
        # 默认迭代次数
        default_iterations = {
            "redis_cache": 1000,
            "database_query": 100,
            "data_compression": 1000,
            "trend_analysis": 100,
            "anomaly_detection": 100
        }
        
        if iterations:
            default_iterations.update(iterations)
        
        # 清空之前的测试结果
        self.test_results = []
        
        # 运行测试
        print("开始性能测试...")
        print("=" * 60)
        
        # 1. Redis缓存测试
        print("测试Redis缓存性能...")
        self.test_redis_cache_performance(default_iterations["redis_cache"])
        
        # 2. 数据库查询测试
        print("测试数据库查询性能...")
        self.test_database_query_performance(default_iterations["database_query"])
        
        # 3. 数据压缩测试
        print("测试数据压缩性能...")
        self.test_data_compression_performance(default_iterations["data_compression"])
        
        # 4. 趋势分析测试
        print("测试趋势分析性能...")
        self.test_trend_analysis_performance(default_iterations["trend_analysis"])
        
        # 5. 异常检测测试
        print("测试异常检测性能...")
        self.test_anomaly_detection_performance(default_iterations["anomaly_detection"])
        
        print("=" * 60)
        print("性能测试完成！")
        
        # 打印测试结果摘要
        print("\n测试结果摘要:")
        print("-" * 60)
        
        for result in self.test_results:
            if result["status"] == "completed":
                if result["test"] == "redis_cache":
                    print(f"Redis缓存: {result['operations_per_second']:.2f} ops/s")
                elif result["test"] == "database_query":
                    print(f"数据库查询: {result['queries_per_second']:.2f} qps")
                elif result["test"] == "data_compression":
                    print(f"数据压缩: {result['compression_speed']:.2f} comp/s, {result['decompression_speed']:.2f} decomp/s")
                    print(f"  压缩率: {result['compression_ratio']:.4f}")
                elif result["test"] == "trend_analysis":
                    print(f"趋势分析: {result['analyses_per_second']:.2f} analyses/s")
                elif result["test"] == "anomaly_detection":
                    print(f"异常检测: {result['detections_per_second']:.2f} detections/s")
            else:
                print(f"{result['test']}: {result['status']} - {result['reason']}")
        
        print("-" * 60)
        
        return self.test_results
    
    def generate_performance_report(self, filename: str = "performance_report.json") -> bool:
        """
        生成性能测试报告
        
        Args:
            filename: 报告文件名
        
        Returns:
            是否成功生成报告
        """
        if not self.test_results:
            logger.warning("没有测试结果可生成报告")
            return False
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "tests": self.test_results,
            "summary": {
                "total_tests": len(self.test_results),
                "completed_tests": sum(1 for r in self.test_results if r["status"] == "completed"),
                "skipped_tests": sum(1 for r in self.test_results if r["status"] == "skipped")
            }
        }
        
        try:
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"性能测试报告生成成功: {filename}")
            return True
        except Exception as e:
            logger.error(f"生成性能测试报告失败: {e}")
            return False


# 全局性能测试器实例
performance_tester = PerformanceTester()


# 工具函数
def run_performance_tests(iterations: Dict[str, int] = None, 
                         generate_report: bool = True) -> List[Dict[str, Any]]:
    """
    运行性能测试
    
    Args:
        iterations: 各测试的迭代次数
        generate_report: 是否生成测试报告
    
    Returns:
        测试结果列表
    """
    tester = PerformanceTester()
    results = tester.run_all_tests(iterations)
    
    if generate_report:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"performance_report_{timestamp}.json"
        tester.generate_performance_report(report_filename)
        print(f"性能测试报告已保存到: {report_filename}")
    
    return results

def compare_performance_baselines(baseline_results: List[Dict[str, Any]], 
                                current_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    比较性能基线和当前结果
    
    Args:
        baseline_results: 基线测试结果
        current_results: 当前测试结果
    
    Returns:
        比较结果
    """
    # 将测试结果转换为字典，便于比较
    baseline_dict = {r["test"]: r for r in baseline_results if r["status"] == "completed"}
    current_dict = {r["test"]: r for r in current_results if r["status"] == "completed"}
    
    comparisons = []
    
    # 比较各项测试
    for test_name in set(baseline_dict.keys()) & set(current_dict.keys()):
        baseline = baseline_dict[test_name]
        current = current_dict[test_name]
        
        # 计算性能变化
        if test_name == "redis_cache":
            baseline_perf = baseline["operations_per_second"]
            current_perf = current["operations_per_second"]
        elif test_name == "database_query":
            baseline_perf = baseline["queries_per_second"]
            current_perf = current["queries_per_second"]
        elif test_name == "data_compression":
            baseline_perf = baseline["compression_speed"] + baseline["decompression_speed"]
            current_perf = current["compression_speed"] + current["decompression_speed"]
        elif test_name == "trend_analysis":
            baseline_perf = baseline["analyses_per_second"]
            current_perf = current["analyses_per_second"]
        elif test_name == "anomaly_detection":
            baseline_perf = baseline["detections_per_second"]
            current_perf = current["detections_per_second"]
        else:
            continue
        
        # 计算性能变化百分比
        if baseline_perf > 0:
            performance_change = ((current_perf - baseline_perf) / baseline_perf) * 100
        else:
            performance_change = 0
        
        comparison = {
            "test": test_name,
            "baseline_performance": baseline_perf,
            "current_performance": current_perf,
            "performance_change": performance_change,
            "status": "improved" if performance_change > 0 else "regressed" if performance_change < 0 else "no_change"
        }
        
        comparisons.append(comparison)
    
    # 生成摘要
    summary = {
        "improved_tests": sum(1 for c in comparisons if c["status"] == "improved"),
        "regressed_tests": sum(1 for c in comparisons if c["status"] == "regressed"),
        "no_change_tests": sum(1 for c in comparisons if c["status"] == "no_change"),
        "overall_change": sum(c["performance_change"] for c in comparisons) / len(comparisons) if comparisons else 0
    }
    
    return {
        "comparisons": comparisons,
        "summary": summary,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # 运行性能测试
    results = run_performance_tests()
    
    # 打印详细结果
    print("\n详细测试结果:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

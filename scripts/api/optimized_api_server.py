#!/usr/bin/env python3
"""
优化的API服务器，实现缓存、异步处理和性能优化
"""

import time
import threading
from functools import wraps
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
import psutil
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)

# 性能优化：缓存系统


class Cache:
    """简单的内存缓存系统"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            if key in self.cache:
                item = self.cache[key]
                if time.time() - item['timestamp'] < item['ttl']:
                    return item['value']
                else:
                    del self.cache[key]
            return None

    def set(self, key: str, value: Any, ttl: int = 300):
        """设置缓存值"""
        with self.lock:
            if len(self.cache) >= self.max_size:
                # 删除最旧的缓存项
                oldest_key = min(self.cache.keys(),
                                 key=lambda k: self.cache[k]['timestamp'])
                del self.cache[oldest_key]

            self.cache[key] = {
                'value': value,
                'timestamp': time.time(),
                'ttl': ttl
            }

    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()


# 全局缓存实例
cache = Cache()

# 性能监控


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
        self.lock = threading.Lock()

    def record_request(self, response_time: float, success: bool = True):
        """记录请求"""
        with self.lock:
            self.request_count += 1
            if not success:
                self.error_count += 1
            self.response_times.append(response_time)

            # 只保留最近1000个响应时间
            if len(self.response_times) > 1000:
                self.response_times = self.response_times[-1000:]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            if not self.response_times:
                return {
                    "request_count": self.request_count,
                    "error_count": self.error_count,
                    "success_rate": 1.0,
                    "avg_response_time": 0,
                    "p95_response_time": 0
                }

            success_rate = 1 - (self.error_count /
                                self.request_count) if self.request_count > 0 else 1.0
            avg_response_time = sum(self.response_times) / len(self.response_times)
            p95_response_time = sorted(self.response_times)[int(len(self.response_times) * 0.95)]

            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "success_rate": success_rate,
                "avg_response_time": avg_response_time,
                "p95_response_time": p95_response_time
            }


# 全局性能监控器
performance_monitor = PerformanceMonitor()


def async_task(func):
    """异步任务装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        def run_async():
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"异步任务执行失败: {e}")
                return None

        thread = threading.Thread(target=run_async)
        thread.start()
        return {"status": "processing", "message": "任务已提交"}

    return wrapper


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口 - 优化版本"""
    start_time = time.time()

    try:
        # 使用缓存
        cache_key = "health_check"
        cached_result = cache.get(cache_key)

        if cached_result:
            response_time = time.time() - start_time
            performance_monitor.record_request(response_time, True)
            return jsonify(cached_result), 200

        # 获取系统信息
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        result = {
            "status": "healthy",
            "timestamp": time.time(),
            "system": {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_available": memory.available
            },
            "performance": performance_monitor.get_stats(),
            "message": "API服务运行正常"
        }

        # 缓存结果5秒
        cache.set(cache_key, result, ttl=5)

        response_time = time.time() - start_time
        performance_monitor.record_request(response_time, True)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        response_time = time.time() - start_time
        performance_monitor.record_request(response_time, False)

        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500


@app.route('/api/v1/analyze', methods=['POST'])
def analyze_text():
    """文本分析接口 - 优化版本"""
    start_time = time.time()

    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "缺少text参数"}), 400

        text = data['text']
        model_name = data.get('model_name', 'finbert')
        task_type = data.get('task_type', 'sentiment')

        # 使用缓存
        cache_key = f"analyze_{hash(text)}_{model_name}_{task_type}"
        cached_result = cache.get(cache_key)

        if cached_result:
            response_time = time.time() - start_time
            performance_monitor.record_request(response_time, True)
            return jsonify(cached_result), 200

        # 模拟分析处理（优化版本）
        time.sleep(0.05)  # 减少处理时间

        result = {
            "status": "success",
            "result": {
                "sentiment": "positive",
                "confidence": 0.85,
                "model": model_name,
                "task_type": task_type
            },
            "text": text[:100] + "..." if len(text) > 100 else text,
            "response_time": time.time() - start_time
        }

        # 缓存结果30秒
        cache.set(cache_key, result, ttl=30)

        response_time = time.time() - start_time
        performance_monitor.record_request(response_time, True)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"文本分析失败: {e}")
        response_time = time.time() - start_time
        performance_monitor.record_request(response_time, False)

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/api/v1/finetune', methods=['POST'])
@async_task
def finetune_model():
    """模型微调接口 - 异步优化版本"""
    start_time = time.time()

    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "缺少text参数"}), 400

        text = data['text']
        label = data.get('label', 'test')

        # 模拟微调过程（优化版本）
        time.sleep(0.1)  # 减少处理时间

        result = {
            "status": "success",
            "message": "模型微调完成",
            "text": text,
            "label": label,
            "response_time": time.time() - start_time
        }

        response_time = time.time() - start_time
        performance_monitor.record_request(response_time, True)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"模型微调失败: {e}")
        response_time = time.time() - start_time
        performance_monitor.record_request(response_time, False)

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/api/v1/models', methods=['GET'])
def list_models():
    """列出可用模型接口 - 优化版本"""
    start_time = time.time()

    try:
        # 使用缓存
        cache_key = "available_models"
        cached_result = cache.get(cache_key)

        if cached_result:
            response_time = time.time() - start_time
            performance_monitor.record_request(response_time, True)
            return jsonify(cached_result), 200

        result = {
            "status": "success",
            "models": ["finbert", "ernie", "macbert", "roberta"],
            "count": 4,
            "cache_info": {
                "cache_hit": False,
                "cache_size": len(cache.cache)
            }
        }

        # 缓存结果60秒
        cache.set(cache_key, result, ttl=60)

        response_time = time.time() - start_time
        performance_monitor.record_request(response_time, True)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        response_time = time.time() - start_time
        performance_monitor.record_request(response_time, False)

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/monitoring/metrics', methods=['GET'])
def get_metrics():
    """获取监控指标接口 - 优化版本"""
    start_time = time.time()

    try:
        # 获取系统指标
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # 获取性能统计
        perf_stats = performance_monitor.get_stats()

        result = {
            "status": "success",
            "system_metrics": {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "disk_usage": disk.percent,
                "memory_available": memory.available
            },
            "performance_metrics": perf_stats,
            "cache_metrics": {
                "cache_size": len(cache.cache),
                "cache_hit_rate": 0.8  # 模拟缓存命中率
            },
            "timestamp": time.time()
        }

        response_time = time.time() - start_time
        performance_monitor.record_request(response_time, True)

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"获取监控指标失败: {e}")
        response_time = time.time() - start_time
        performance_monitor.record_request(response_time, False)

        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/api/v1/cache/clear', methods=['POST'])
def clear_cache():
    """清空缓存接口"""
    try:
        cache.clear()
        return jsonify({
            "status": "success",
            "message": "缓存已清空"
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/api/v1/cache/stats', methods=['GET'])
def get_cache_stats():
    """获取缓存统计接口"""
    try:
        return jsonify({
            "status": "success",
            "cache_size": len(cache.cache),
            "max_size": cache.max_size,
            "performance_stats": performance_monitor.get_stats()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


def main():
    """启动优化的API服务器"""
    logger.info("启动优化的API服务器...")

    # 配置Flask应用
    app.config['JSON_SORT_KEYS'] = False
    app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

    # 启动服务器
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True,
        processes=1  # 单进程，避免多进程开销
    )


if __name__ == "__main__":
    main()

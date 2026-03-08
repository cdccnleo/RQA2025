#!/usr/bin/env python3
"""
简单的健康检查API服务
用于验证RQA2025生产环境部署

安全更新：使用环境变量替代硬编码密码
"""

import os
from flask import Flask, jsonify
import psycopg2
import redis
import requests
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 从环境变量读取配置
DB_CONFIG = {
    'host': os.environ.get('DB_HOST', 'postgres'),
    'port': int(os.environ.get('DB_PORT', '5432')),
    'database': os.environ.get('DB_NAME', 'rqa2025'),
    'user': os.environ.get('DB_USER', 'postgres'),
    'password': os.environ.get('DB_PASSWORD', '')
}

REDIS_CONFIG = {
    'host': os.environ.get('REDIS_HOST', 'redis'),
    'port': int(os.environ.get('REDIS_PORT', '6379')),
    'password': os.environ.get('REDIS_PASSWORD', ''),
    'db': int(os.environ.get('REDIS_DB', '0'))
}

# 验证必要的环境变量
def validate_config():
    """验证配置是否完整"""
    required_vars = ['DB_PASSWORD', 'REDIS_PASSWORD']
    missing = [var for var in required_vars if not os.environ.get(var)]
    if missing:
        logger.warning(f"缺少环境变量: {missing}")
        logger.warning("请设置必要的环境变量后再启动服务")


@app.route('/health')
def health_check():
    """健康检查端点"""
    try:
        # 检查数据库连接
        db_status = check_database()

        # 检查Redis连接
        redis_status = check_redis()

        # 检查监控系统
        monitoring_status = check_monitoring()

        # 总体状态
        overall_status = 'healthy' if all([
            db_status['status'] == 'healthy',
            redis_status['status'] == 'healthy',
            monitoring_status['status'] == 'healthy'
        ]) else 'unhealthy'

        return jsonify({
            'status': overall_status,
            'timestamp': time.time(),
            'services': {
                'database': db_status,
                'redis': redis_status,
                'monitoring': monitoring_status
            }
        }), 200

    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': time.time()
        }), 500


@app.route('/api/v1/status')
def api_status():
    """API状态端点"""
    return jsonify({
        'service': 'RQA2025 Health Check API',
        'version': '1.0.0',
        'status': 'running',
        'timestamp': time.time()
    })


@app.route('/metrics')
def metrics():
    """Prometheus指标端点"""
    try:
        # 简单的指标收集
        db_status = check_database()
        redis_status = check_redis()

        metrics_data = f"""
# HELP rqa2025_database_status Database connection status
# TYPE rqa2025_database_status gauge
rqa2025_database_status {1 if db_status['status'] == 'healthy' else 0}

# HELP rqa2025_redis_status Redis connection status
# TYPE rqa2025_redis_status gauge
rqa2025_redis_status {1 if redis_status['status'] == 'healthy' else 0}

# HELP rqa2025_health_check_total Total health checks
# TYPE rqa2025_health_check_total counter
rqa2025_health_check_total 1
"""
        return metrics_data, 200, {'Content-Type': 'text/plain'}

    except Exception as e:
        logger.error(f"指标收集失败: {e}")
        return f"# ERROR: {e}", 500, {'Content-Type': 'text/plain'}


def check_database():
    """检查数据库连接"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        return {
            'status': 'healthy',
            'version': version.split()[1] if version else 'unknown',
            'connection': 'established'
        }
    except Exception as e:
        logger.error(f"数据库检查失败: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'connection': 'failed'
        }


def check_redis():
    """检查Redis连接"""
    try:
        r = redis.Redis(**REDIS_CONFIG, decode_responses=True)
        r.ping()
        info = r.info()

        return {
            'status': 'healthy',
            'version': info.get('redis_version', 'unknown'),
            'memory_used': info.get('used_memory_human', 'unknown'),
            'connection': 'established'
        }
    except Exception as e:
        logger.error(f"Redis检查失败: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e),
            'connection': 'failed'
        }


def check_monitoring():
    """检查监控系统"""
    try:
        # 检查Prometheus
        prometheus_response = requests.get('http://prometheus:9090/-/healthy', timeout=5)
        prometheus_healthy = prometheus_response.status_code == 200

        # 检查Grafana
        grafana_response = requests.get('http://grafana:3000/api/health', timeout=5)
        grafana_healthy = grafana_response.status_code == 200

        if prometheus_healthy and grafana_healthy:
            return {
                'status': 'healthy',
                'prometheus': 'healthy',
                'grafana': 'healthy'
            }
        else:
            return {
                'status': 'unhealthy',
                'prometheus': 'healthy' if prometheus_healthy else 'unhealthy',
                'grafana': 'healthy' if grafana_healthy else 'unhealthy'
            }

    except Exception as e:
        logger.error(f"监控系统检查失败: {e}")
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


if __name__ == '__main__':
    logger.info("启动健康检查API服务...")
    validate_config()
    app.run(host='0.0.0.0', port=8000, debug=False)

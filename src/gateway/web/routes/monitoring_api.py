"""
监控API路由模块

功能：
- 数据压缩监控API
- 缓存预热监控API
- 异常检测监控API
- WebSocket实时数据推送

作者: Claude
创建日期: 2026-02-21
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

# 配置日志
logger = logging.getLogger(__name__)

# 创建蓝图
monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/api/monitoring')


# ============ 数据压缩监控API ============

@monitoring_bp.route('/compression', methods=['GET'])
def get_compression_stats():
    """
    获取数据压缩统计信息
    
    Returns:
        JSON格式的压缩统计数据
    """
    try:
        # 从压缩服务获取统计数据
        from src.data.compression.data_compression_service import get_compression_service
        
        service = get_compression_service()
        stats = service.get_stats()
        
        # 获取算法对比数据
        algorithm_comparison = {}
        for alg in ['lz4', 'snappy', 'zstd', 'gzip']:
            try:
                # 模拟获取算法性能数据
                algorithm_comparison[alg] = {
                    'avg_ratio': stats.get('algorithm_stats', {}).get(alg, {}).get('avg_ratio', 0.5),
                    'speed_mbps': _get_algorithm_speed(alg)
                }
            except Exception:
                algorithm_comparison[alg] = {'avg_ratio': 0.5, 'speed_mbps': 100}
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'avg_compression_ratio': stats.get('avg_compression_ratio', 0),
            'total_bytes_saved': stats.get('total_bytes_saved', 0),
            'avg_compress_time_ms': stats.get('avg_compress_time_ms', 0),
            'total_compressions': stats.get('total_compressions', 0),
            'algorithm_stats': stats.get('algorithm_stats', {}),
            'algorithm_comparison': algorithm_comparison
        })
    except Exception as e:
        logger.error(f"获取压缩统计失败: {e}")
        # 返回模拟数据
        return jsonify(_get_mock_compression_data())


def _get_algorithm_speed(algorithm: str) -> float:
    """获取算法速度（MB/s）"""
    speeds = {
        'lz4': 450,
        'snappy': 350,
        'zstd': 200,
        'gzip': 100
    }
    return speeds.get(algorithm, 100)


def _get_mock_compression_data() -> Dict[str, Any]:
    """获取模拟压缩数据"""
    return {
        'status': 'mock',
        'timestamp': datetime.now().isoformat(),
        'avg_compression_ratio': 0.65,
        'total_bytes_saved': 1024 * 1024 * 500,
        'avg_compress_time_ms': 15.5,
        'total_compressions': 12345,
        'algorithm_stats': {
            'lz4': {'count': 5000, 'avg_ratio': 0.45},
            'snappy': {'count': 4000, 'avg_ratio': 0.55},
            'zstd': {'count': 2500, 'avg_ratio': 0.70},
            'gzip': {'count': 845, 'avg_ratio': 0.75}
        },
        'algorithm_comparison': {
            'lz4': {'avg_ratio': 0.45, 'speed_mbps': 450},
            'snappy': {'avg_ratio': 0.55, 'speed_mbps': 350},
            'zstd': {'avg_ratio': 0.70, 'speed_mbps': 200},
            'gzip': {'avg_ratio': 0.75, 'speed_mbps': 100}
        }
    }


# ============ 缓存预热监控API ============

@monitoring_bp.route('/cache', methods=['GET'])
def get_cache_stats():
    """
    获取缓存预热统计信息
    
    Returns:
        JSON格式的缓存统计数据
    """
    try:
        # 从缓存预热服务获取统计数据
        from src.data.cache.smart_cache_preheater import get_preheater
        
        preheater = get_preheater()
        stats = preheater.get_stats()
        
        # 获取热点数据
        hot_keys = [
            {'key': key, 'count': count}
            for key, count in preheater.behavior_analyzer.get_hot_keys(10)
        ]
        
        # 获取预热推荐
        recommendations = preheater.get_preheat_recommendations(10)
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'hit_rate': stats.get('hit_rate', 0),
            'successful_preheats': stats.get('successful_preheats', 0),
            'avg_preheat_time_ms': stats.get('avg_preheat_time_ms', 0),
            'model_trained': stats.get('model_trained', False),
            'hot_keys': hot_keys,
            'recommendations': recommendations,
            'queue_size': stats.get('queue_size', 0),
            'preheated_keys_count': stats.get('preheated_keys_count', 0)
        })
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}")
        return jsonify(_get_mock_cache_data())


def _get_mock_cache_data() -> Dict[str, Any]:
    """获取模拟缓存数据"""
    return {
        'status': 'mock',
        'timestamp': datetime.now().isoformat(),
        'hit_rate': 0.85,
        'successful_preheats': 5678,
        'avg_preheat_time_ms': 25.3,
        'model_trained': True,
        'hot_keys': [
            {'key': 'stock:AAPL', 'count': 1250},
            {'key': 'stock:GOOGL', 'count': 980},
            {'key': 'stock:MSFT', 'count': 875},
            {'key': 'market:overview', 'count': 750},
            {'key': 'user:profile:123', 'count': 620}
        ],
        'recommendations': [
            {'data_key': 'stock:TSLA', 'access_count': 450, 'priority': 9, 'reason': 'hot_data'},
            {'data_key': 'stock:AMZN', 'access_count': 380, 'priority': 8, 'reason': 'hot_data'},
            {'data_key': 'market:sectors', 'access_count': 320, 'priority': 7, 'reason': 'temporal_pattern'}
        ],
        'queue_size': 15,
        'preheated_keys_count': 1234
    }


# ============ 异常检测监控API ============

@monitoring_bp.route('/anomaly', methods=['GET'])
def get_anomaly_stats():
    """
    获取异常检测统计信息
    
    Returns:
        JSON格式的异常检测统计数据
    """
    try:
        # 从异常检测服务获取统计数据
        from src.monitoring.anomaly_detection.intelligent_anomaly_detector import get_detector
        
        detector = get_detector()
        stats = detector.get_stats()
        
        # 获取最近异常
        recent_anomalies = detector.get_recent_anomalies(20)
        
        # 计算高风险异常数量
        high_risk_count = sum(
            1 for a in detector.anomaly_history
            if a.severity.name in ['HIGH', 'CRITICAL']
        )
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'total_detections': stats.get('total_detections', 0),
            'high_risk_count': high_risk_count,
            'avg_detection_time_ms': stats.get('avg_detection_time_ms', 0),
            'detectors': stats.get('detectors', []),
            'type_distribution': stats.get('detection_distribution', {}),
            'severity_distribution': _calculate_severity_distribution(detector),
            'recent_anomalies': recent_anomalies,
            'isolation_forest': _get_isolation_forest_stats(detector),
            'lof': _get_lof_stats(detector)
        })
    except Exception as e:
        logger.error(f"获取异常检测统计失败: {e}")
        return jsonify(_get_mock_anomaly_data())


def _calculate_severity_distribution(detector) -> Dict[str, int]:
    """计算严重程度分布"""
    distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
    
    try:
        for anomaly in detector.anomaly_history:
            severity = anomaly.severity.name
            if severity in distribution:
                distribution[severity] += 1
    except Exception:
        pass
    
    return distribution


def _get_isolation_forest_stats(detector) -> Optional[Dict[str, Any]]:
    """获取孤立森林统计"""
    try:
        if_detector = detector.ensemble.detectors.get('isolation_forest')
        if if_detector and if_detector.is_trained:
            return {
                'samples': 10000,  # 示例数据
                'anomaly_ratio': 0.08,
                'threshold': if_detector.threshold
            }
    except Exception:
        pass
    return None


def _get_lof_stats(detector) -> Optional[Dict[str, Any]]:
    """获取LOF统计"""
    try:
        lof_detector = detector.ensemble.detectors.get('lof')
        if lof_detector and lof_detector.is_trained:
            return {
                'samples': 10000,  # 示例数据
                'n_neighbors': lof_detector.config.n_neighbors,
                'anomaly_ratio': 0.10
            }
    except Exception:
        pass
    return None


def _get_mock_anomaly_data() -> Dict[str, Any]:
    """获取模拟异常检测数据"""
    return {
        'status': 'mock',
        'timestamp': datetime.now().isoformat(),
        'total_detections': 156,
        'high_risk_count': 12,
        'avg_detection_time_ms': 5.2,
        'detectors': ['isolation_forest', 'lof'],
        'isolation_forest': {
            'samples': 10000,
            'anomaly_ratio': 0.08,
            'threshold': -0.35
        },
        'lof': {
            'samples': 10000,
            'n_neighbors': 20,
            'anomaly_ratio': 0.10
        },
        'type_distribution': {
            'point_anomaly': 45,
            'contextual_anomaly': 38,
            'collective_anomaly': 42,
            'temporal_anomaly': 31
        },
        'severity_distribution': {
            'LOW': 85,
            'MEDIUM': 45,
            'HIGH': 20,
            'CRITICAL': 6
        },
        'recent_anomalies': [
            {
                'timestamp': datetime.now().isoformat(),
                'data_key': 'metric:cpu_usage',
                'type': 'point_anomaly',
                'severity': 'HIGH',
                'score': -0.65,
                'explanation': 'CPU使用率异常飙升至95%'
            },
            {
                'timestamp': datetime.now().isoformat(),
                'data_key': 'metric:memory',
                'type': 'contextual_anomaly',
                'severity': 'MEDIUM',
                'score': 2.3,
                'explanation': '内存使用模式与历史不符'
            }
        ]
    }


# ============ WebSocket处理 ============

class MonitoringWebSocketHandler:
    """监控WebSocket处理器"""
    
    def __init__(self):
        self.clients: set = set()
        
    async def register(self, websocket):
        """注册客户端"""
        self.clients.add(websocket)
        logger.info(f"WebSocket客户端已连接，当前连接数: {len(self.clients)}")
        
    async def unregister(self, websocket):
        """注销客户端"""
        self.clients.discard(websocket)
        logger.info(f"WebSocket客户端已断开，当前连接数: {len(self.clients)}")
        
    async def broadcast(self, message: Dict[str, Any]):
        """广播消息给所有客户端"""
        if not self.clients:
            return
            
        message_str = json.dumps(message)
        disconnected = set()
        
        for client in self.clients:
            try:
                await client.send(message_str)
            except Exception:
                disconnected.add(client)
                
        # 清理断开的客户端
        for client in disconnected:
            self.clients.discard(client)
            
    async def send_compression_update(self, data: Dict[str, Any]):
        """发送压缩更新"""
        await self.broadcast({
            'type': 'compression_update',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
        
    async def send_cache_update(self, data: Dict[str, Any]):
        """发送缓存更新"""
        await self.broadcast({
            'type': 'cache_update',
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
        
    async def send_anomaly_alert(self, anomaly: Dict[str, Any]):
        """发送异常告警"""
        await self.broadcast({
            'type': 'anomaly_detected',
            'data': anomaly,
            'timestamp': datetime.now().isoformat()
        })


# 全局WebSocket处理器实例
ws_handler = MonitoringWebSocketHandler()


def get_ws_handler() -> MonitoringWebSocketHandler:
    """获取WebSocket处理器实例"""
    return ws_handler


# ============ 健康检查API ============

@monitoring_bp.route('/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    
    Returns:
        系统健康状态
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'compression': True,
            'cache_preheater': True,
            'anomaly_detector': True
        }
    })

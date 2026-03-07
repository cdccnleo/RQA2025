"""
统一的缓存配置模块
定义各服务的缓存TTL常量，统一管理后端服务的缓存配置
"""

from typing import Dict, Optional
from functools import lru_cache


class CacheConfig:
    """缓存配置类"""
    
    # 架构状态相关缓存TTL（秒）
    ARCHITECTURE_STATUS_TTL = 10
    ARCHITECTURE_LAYER_STATUS_TTL = 10
    ARCHITECTURE_HEALTH_TTL = 10
    
    # 数据质量相关缓存TTL（秒）
    DATA_QUALITY_METRICS_TTL = 5
    DATA_QUALITY_ISSUES_TTL = 5
    DATA_QUALITY_HISTORY_TTL = 5
    
    # 数据缓存相关TTL（秒）
    DATA_CACHE_STATS_TTL = 5
    DATA_CACHE_HIT_RATE_TTL = 5
    
    # 数据湖管理相关TTL（秒）
    DATA_LAKE_STATS_TTL = 10
    DATA_LAKE_DATASETS_TTL = 15
    
    # 数据性能监控相关TTL（秒）
    DATA_PERFORMANCE_METRICS_TTL = 3
    DATA_PERFORMANCE_HISTORY_TTL = 3
    
    # 性能指标相关TTL（秒）
    SYSTEM_METRICS_TTL = 3
    DATA_SOURCES_METRICS_TTL = 3
    
    # 告警事件相关TTL（秒）
    RISK_ALERTS_TTL = 2
    SYSTEM_EVENTS_TTL = 2
    
    # 策略相关TTL（秒）
    STRATEGY_LIST_TTL = 30
    STRATEGY_DETAIL_TTL = 60
    
    # 交易执行相关TTL（秒）
    TRADING_OVERVIEW_TTL = 5
    TRADING_SIGNALS_TTL = 3
    
    # 风险控制相关TTL（秒）
    RISK_STATUS_TTL = 2
    RISK_METRICS_TTL = 5
    
    @classmethod
    def get_ttl_for_endpoint(cls, endpoint: str) -> int:
        """
        根据API端点获取对应的TTL
        
        Args:
            endpoint: API端点路径
            
        Returns:
            TTL（秒）
        """
        endpoint_lower = endpoint.lower()
        
        # 架构状态
        if '/architecture' in endpoint_lower or '/layer' in endpoint_lower:
            if 'status' in endpoint_lower or 'health' in endpoint_lower:
                return cls.ARCHITECTURE_STATUS_TTL
            return cls.ARCHITECTURE_STATUS_TTL
        
        # 数据质量
        if '/data/quality' in endpoint_lower:
            if 'metrics' in endpoint_lower:
                return cls.DATA_QUALITY_METRICS_TTL
            if 'issues' in endpoint_lower:
                return cls.DATA_QUALITY_ISSUES_TTL
            if 'history' in endpoint_lower:
                return cls.DATA_QUALITY_HISTORY_TTL
            return cls.DATA_QUALITY_METRICS_TTL
        
        # 数据缓存
        if '/data/cache' in endpoint_lower:
            if 'hit' in endpoint_lower or 'rate' in endpoint_lower:
                return cls.DATA_CACHE_HIT_RATE_TTL
            return cls.DATA_CACHE_STATS_TTL
        
        # 数据湖
        if '/data/lake' in endpoint_lower or '/data-lake' in endpoint_lower:
            if 'datasets' in endpoint_lower:
                return cls.DATA_LAKE_DATASETS_TTL
            return cls.DATA_LAKE_STATS_TTL
        
        # 数据性能
        if '/data/performance' in endpoint_lower or '/data-performance' in endpoint_lower:
            if 'history' in endpoint_lower:
                return cls.DATA_PERFORMANCE_HISTORY_TTL
            return cls.DATA_PERFORMANCE_METRICS_TTL
        
        # 系统指标
        if '/data-sources/metrics' in endpoint_lower or '/system/metrics' in endpoint_lower:
            return cls.SYSTEM_METRICS_TTL
        
        # 告警事件
        if '/risk/status' in endpoint_lower or '/risk/alerts' in endpoint_lower:
            return cls.RISK_ALERTS_TTL
        if '/system/events' in endpoint_lower:
            return cls.SYSTEM_EVENTS_TTL
        
        # 策略
        if '/strategy' in endpoint_lower:
            if 'list' in endpoint_lower or endpoint_lower.endswith('/strategies'):
                return cls.STRATEGY_LIST_TTL
            return cls.STRATEGY_DETAIL_TTL
        
        # 交易执行
        if '/trading' in endpoint_lower:
            if 'signals' in endpoint_lower:
                return cls.TRADING_SIGNALS_TTL
            return cls.TRADING_OVERVIEW_TTL
        
        # 风险控制
        if '/risk' in endpoint_lower:
            if 'status' in endpoint_lower:
                return cls.RISK_STATUS_TTL
            return cls.RISK_METRICS_TTL
        
        # 默认TTL
        return 10
    
    @classmethod
    def get_cache_config_dict(cls) -> Dict[str, int]:
        """
        获取所有缓存配置的字典
        
        Returns:
            配置字典
        """
        return {
            'architecture_status': cls.ARCHITECTURE_STATUS_TTL,
            'data_quality_metrics': cls.DATA_QUALITY_METRICS_TTL,
            'data_cache_stats': cls.DATA_CACHE_STATS_TTL,
            'data_lake_stats': cls.DATA_LAKE_STATS_TTL,
            'data_performance_metrics': cls.DATA_PERFORMANCE_METRICS_TTL,
            'system_metrics': cls.SYSTEM_METRICS_TTL,
            'risk_alerts': cls.RISK_ALERTS_TTL,
            'system_events': cls.SYSTEM_EVENTS_TTL,
            'strategy_list': cls.STRATEGY_LIST_TTL,
            'trading_overview': cls.TRADING_OVERVIEW_TTL,
            'risk_status': cls.RISK_STATUS_TTL,
        }


# 向后兼容：导出常用配置
ARCHITECTURE_STATUS_TTL = CacheConfig.ARCHITECTURE_STATUS_TTL
DATA_QUALITY_METRICS_TTL = CacheConfig.DATA_QUALITY_METRICS_TTL
DATA_CACHE_STATS_TTL = CacheConfig.DATA_CACHE_STATS_TTL
DATA_PERFORMANCE_METRICS_TTL = CacheConfig.DATA_PERFORMANCE_METRICS_TTL
SYSTEM_METRICS_TTL = CacheConfig.SYSTEM_METRICS_TTL
RISK_ALERTS_TTL = CacheConfig.RISK_ALERTS_TTL
SYSTEM_EVENTS_TTL = CacheConfig.SYSTEM_EVENTS_TTL


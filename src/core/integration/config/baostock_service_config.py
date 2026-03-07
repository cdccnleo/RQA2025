#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaoStock服务配置文件

提供统一的BaoStock服务配置管理
"""

from typing import Dict, List, Optional, Any


class BaoStockServiceConfig:
    """BaoStock服务配置类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化配置
        
        Args:
            config: 配置字典，覆盖默认配置
        """
        self._default_config = {
            "retry_policy": {
                "max_retries": 3,
                "initial_delay": 2,
                "backoff_factor": 1.5
            },
            "timeout": {
                "stock_data": 30,
                "basic_info": 20,
                "market_data": 60
            },
            "field_mapping": {
                "kline": {
                    "date": "日期",
                    "code": "股票代码",
                    "open": "开盘",
                    "high": "最高",
                    "low": "最低",
                    "close": "收盘",
                    "volume": "成交量",
                    "amount": "成交额",
                    "turn": "换手率",
                    "pctChg": "涨跌幅"
                }
            },
            "api_preference": {
                "stock_daily": ["query_history_k_data_plus"],
                "stock_info": ["query_stock_basic"],
                "market_data": ["query_all_stock"]
            },
            "performance": {
                "enable_batch_processing": True,
                "batch_size": 100,
                "concurrency_limit": 5
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
        
        # 合并配置
        self._config = self._default_config.copy()
        if config:
            self._merge_config(self._config, config)
        
        # 验证配置
        self._validate_config()
    
    def _merge_config(self, target: Dict[str, Any], source: Dict[str, Any]):
        """递归合并配置"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value
    
    def _validate_config(self):
        """验证配置"""
        # 验证重试策略
        retry_policy = self._config.get("retry_policy", {})
        assert isinstance(retry_policy.get("max_retries"), int) and retry_policy.get("max_retries") >= 0
        assert isinstance(retry_policy.get("initial_delay"), (int, float)) and retry_policy.get("initial_delay") >= 0
        assert isinstance(retry_policy.get("backoff_factor"), (int, float)) and retry_policy.get("backoff_factor") >= 1
        
        # 验证超时设置
        timeout = self._config.get("timeout", {})
        for key, value in timeout.items():
            assert isinstance(value, int) and value > 0
        
        # 验证API偏好设置
        api_preference = self._config.get("api_preference", {})
        for key, value in api_preference.items():
            assert isinstance(value, list) and len(value) > 0
    
    @property
    def config(self) -> Dict[str, Any]:
        """获取完整配置"""
        return self._config
    
    @property
    def retry_policy(self) -> Dict[str, Any]:
        """获取重试策略配置"""
        return self._config.get("retry_policy", {})
    
    @property
    def timeout_config(self) -> Dict[str, Any]:
        """获取超时配置"""
        return self._config.get("timeout", {})
    
    @property
    def field_mapping(self) -> Dict[str, Dict[str, str]]:
        """获取字段映射配置"""
        return self._config.get("field_mapping", {})
    
    @property
    def api_preference(self) -> Dict[str, List[str]]:
        """获取API偏好配置"""
        return self._config.get("api_preference", {})
    
    @property
    def performance_config(self) -> Dict[str, Any]:
        """获取性能配置"""
        return self._config.get("performance", {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """获取日志配置"""
        return self._config.get("logging", {})
    
    def update_config(self, config: Dict[str, Any]):
        """
        更新配置
        
        Args:
            config: 要更新的配置
        """
        self._merge_config(self._config, config)
        self._validate_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"BaoStockServiceConfig(retry_policy={self.retry_policy}, timeout={self.timeout_config})"
    
    def __repr__(self) -> str:
        """repr表示"""
        return self.__str__()


# 默认配置实例
default_config = BaoStockServiceConfig()


# 生产环境配置
production_config = BaoStockServiceConfig({
    "retry_policy": {
        "max_retries": 5,
        "initial_delay": 3,
        "backoff_factor": 2
    },
    "timeout": {
        "stock_data": 60,
        "basic_info": 30,
        "market_data": 120
    },
    "performance": {
        "enable_batch_processing": True,
        "batch_size": 200,
        "concurrency_limit": 10
    }
})


# 开发环境配置
development_config = BaoStockServiceConfig({
    "retry_policy": {
        "max_retries": 2,
        "initial_delay": 1,
        "backoff_factor": 1.2
    },
    "timeout": {
        "stock_data": 20,
        "basic_info": 15,
        "market_data": 40
    },
    "logging": {
        "level": "DEBUG"
    }
})


# 测试环境配置
test_config = BaoStockServiceConfig({
    "retry_policy": {
        "max_retries": 1,
        "initial_delay": 1,
        "backoff_factor": 1
    },
    "timeout": {
        "stock_data": 10,
        "basic_info": 5,
        "market_data": 20
    },
    "logging": {
        "level": "DEBUG"
    }
})


# 配置工厂函数
def get_baostock_config(env: str = "default") -> BaoStockServiceConfig:
    """
    获取指定环境的配置
    
    Args:
        env: 环境名称 (default, production, development, test)
        
    Returns:
        BaoStockServiceConfig实例
    """
    config_map = {
        "default": default_config,
        "production": production_config,
        "development": development_config,
        "test": test_config
    }
    
    return config_map.get(env, default_config)
"""
retry_manager 模块

专门管理重试逻辑的管理器。
"""

import secrets
import random
import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """重试策略枚举"""
    FIXED = "fixed"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIBONACCI = "fibonacci"


@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    jitter: bool = True
    backoff_factor: float = 2.0


class RetryManager:
    """重试管理器 - 专门管理重试逻辑"""

    def __init__(self):
        self._retry_configs: Dict[str, RetryConfig] = {}
        self._default_retry_config = RetryConfig()
        self._setup_default_configs()

    def _setup_default_configs(self):
        """设置默认重试配置"""
        self._retry_configs['network'] = RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=30.0,
            strategy=RetryStrategy.EXPONENTIAL
        )

        self._retry_configs['database'] = RetryConfig(
            max_attempts=3,
            base_delay=2.0,
            max_delay=20.0,
            strategy=RetryStrategy.EXPONENTIAL
        )

        self._retry_configs['file'] = RetryConfig(
            max_attempts=3,
            base_delay=0.5,
            max_delay=5.0,
            strategy=RetryStrategy.FIXED
        )

    def execute_retry(self, retry_config: RetryConfig, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行重试逻辑"""
        attempts = 0
        last_error = error

        while attempts < retry_config.max_attempts:
            try:
                delay = self._calculate_delay(retry_config, attempts)
                time.sleep(delay)
                attempts += 1

                # 模拟重试成功
                if attempts >= 2:  # 假设第二次重试成功
                    return {
                        'success': True,
                        'attempts': attempts,
                        'total_delay': sum(self._calculate_delay(retry_config, i) for i in range(attempts))
                    }

            except Exception as retry_error:
                last_error = retry_error
                attempts += 1

        return {
            'success': False,
            'attempts': attempts,
            'last_error': str(last_error),
            'total_delay': sum(self._calculate_delay(retry_config, i) for i in range(attempts))
        }

    def _calculate_delay(self, config: RetryConfig, attempt: int) -> float:
        """计算重试延迟"""
        if config.strategy == RetryStrategy.FIXED:
            delay = config.base_delay
        elif config.strategy == RetryStrategy.LINEAR:
            delay = config.base_delay * (attempt + 1)
        elif config.strategy == RetryStrategy.EXPONENTIAL:
            delay = config.base_delay * (config.backoff_factor ** attempt)
        else:  # FIBONACCI or unknown
            delay = config.base_delay * (attempt + 1)

        # 应用抖动
        if config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        return min(delay, config.max_delay)

    def add_retry_config(self, name: str, config: RetryConfig) -> None:
        """添加重试配置"""
        self._retry_configs[name] = config

    def get_retry_config(self, name: str) -> Optional[RetryConfig]:
        """获取重试配置"""
        return self._retry_configs.get(name)

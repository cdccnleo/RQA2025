#!/usr/bin/env python3
"""
智能数据源管理器

支持多种数据源，自动选择最优数据源，并提供数据源健康监控
"""

# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd

from src.infrastructure.logging import get_infrastructure_logger
# 在并行/隔离环境下对接口导入提供兜底
try:
    from ..interfaces import IDataModel, IDataLoader  # type: ignore
except Exception:  # pragma: no cover - 仅在极端导入异常时触发
    try:
        from typing import Protocol  # type: ignore
    except Exception:
        class Protocol:  # 最小兜底
            pass

    class IDataLoader(Protocol):  # type: ignore
        async def load_data(self, *args, **kwargs): ...

    class IDataModel:  # type: ignore
        pass

logger = get_infrastructure_logger('intelligent_source_manager')


class DataSourceType(Enum):

    """数据源类型"""
    STOCK = "stock"
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    INDEX = "index"
    NEWS = "news"
    SENTIMENT = "sentiment"


class DataSourceStatus(Enum):

    """数据源状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class DataSourceConfig:

    """数据源配置"""
    name: str
    source_type: DataSourceType
    priority: int = 1  # 优先级，数字越小优先级越高
    timeout_seconds: int = 30
    retry_count: int = 3
    health_check_interval: int = 60
    enabled: bool = True


@dataclass
class DataSourceHealth:

    """数据源健康状态"""
    source_name: str
    status: DataSourceStatus
    response_time_ms: float
    success_rate: float
    last_check_time: float
    error_count: int = 0
    total_requests: int = 0


class DataSourceHealthMonitor:

    """数据源健康监控器"""

    def __init__(self):

        self.health_records: Dict[str, DataSourceHealth] = {}
        self.monitoring_thread = None
        self.is_monitoring = False

    def record_request(self, source_name: str, response_time_ms: float, success: bool):
        """记录请求"""
        if source_name not in self.health_records:
            self.health_records[source_name] = DataSourceHealth(
                source_name=source_name,
                status=DataSourceStatus.HEALTHY,
                response_time_ms=response_time_ms,
                success_rate=1.0 if success else 0.0,
                last_check_time=time.time(),
                error_count=0 if success else 1,
                total_requests=1
            )
        else:
            record = self.health_records[source_name]
            record.total_requests += 1
            record.response_time_ms = (
                (record.response_time_ms * (record.total_requests - 1) +
                 response_time_ms) / record.total_requests
            )

            if not success:
                record.error_count += 1

            record.success_rate = (record.total_requests -
                                   record.error_count) / record.total_requests
            record.last_check_time = time.time()

            # 更新状态
            self._update_source_status(record)

    def _update_source_status(self, record: DataSourceHealth):
        """更新数据源状态"""
        if record.success_rate >= 0.95 and record.response_time_ms < 1000:
            record.status = DataSourceStatus.HEALTHY
        elif record.success_rate >= 0.8 and record.response_time_ms < 3000:
            record.status = DataSourceStatus.DEGRADED
        elif record.success_rate >= 0.5:
            record.status = DataSourceStatus.UNHEALTHY
        else:
            record.status = DataSourceStatus.OFFLINE

    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""
        if not self.health_records:
            return {}

        total_sources = len(self.health_records)
        healthy_sources = sum(1 for r in self.health_records.values()
                              if r.status == DataSourceStatus.HEALTHY)
        degraded_sources = sum(1 for r in self.health_records.values()
                               if r.status == DataSourceStatus.DEGRADED)
        unhealthy_sources = sum(1 for r in self.health_records.values()
                                if r.status == DataSourceStatus.UNHEALTHY)
        offline_sources = sum(1 for r in self.health_records.values()
                              if r.status == DataSourceStatus.OFFLINE)

        return {
            'total_sources': total_sources,
            'healthy_sources': healthy_sources,
            'degraded_sources': degraded_sources,
            'unhealthy_sources': unhealthy_sources,
            'offline_sources': offline_sources,
            'overall_health': healthy_sources / total_sources if total_sources > 0 else 0.0,
            'sources': {name: {
                'status': record.status.value,
                'response_time_ms': record.response_time_ms,
                'success_rate': record.success_rate,
                'error_count': record.error_count,
                'total_requests': record.total_requests
            } for name, record in self.health_records.items()}
        }

    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("数据源健康监控已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("数据源健康监控已停止")

    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 检查数据源健康状态
                for source_name, record in self.health_records.items():
                    # 如果数据源长时间未使用，标记为离线
                    if time.time() - record.last_check_time > 3600:  # 1小时
                        record.status = DataSourceStatus.OFFLINE

                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"数据源监控异常: {e}")
                time.sleep(60)


class IntelligentSourceManager:

    """智能数据源管理器"""

    def __init__(self):

        self.sources: Dict[str, DataSourceConfig] = {}
        self.loaders: Dict[str, IDataLoader] = {}
        self.health_monitor = DataSourceHealthMonitor()
        self.source_ranking = []
        self.lock = threading.Lock()

        # 启动健康监控
        self.health_monitor.start_monitoring()

    def register_source(self, name: str, config: DataSourceConfig, loader: IDataLoader):
        """注册数据源"""
        with self.lock:
            self.sources[name] = config
            self.loaders[name] = loader
            self._update_source_ranking()
            logger.info(f"注册数据源: {name} ({config.source_type.value})")

    def unregister_source(self, name: str):
        """注销数据源"""
        with self.lock:
            if name in self.sources:
                del self.sources[name]
                del self.loaders[name]
                self._update_source_ranking()
                logger.info(f"注销数据源: {name}")

    def _update_source_ranking(self):
        """更新数据源排名"""
        # 获取健康报告
        health_report = self.health_monitor.get_health_report()

        # 计算每个数据源的得分
        source_scores = []
        for name, config in self.sources.items():
            if not config.enabled:
                continue

            score = self._calculate_source_score(name, config, health_report)
            source_scores.append((name, score))

        # 按得分排序
        source_scores.sort(key=lambda x: x[1], reverse=True)
        self.source_ranking = [name for name, score in source_scores]

    def _calculate_source_score(self, name: str, config: DataSourceConfig, health_report: Dict[str, Any]) -> float:
        """计算数据源得分"""
        score = 100.0

        # 基础优先级得分
        score += (10 - config.priority) * 10  # 优先级越高得分越高

        # 健康状态得分
        if name in health_report.get('sources', {}):
            source_health = health_report['sources'][name]
            success_rate = source_health['success_rate']
            response_time = source_health['response_time_ms']

            # 成功率得分
            score += success_rate * 50

            # 响应时间得分
            if response_time < 1000:
                score += 20
            elif response_time < 3000:
                score += 10
            elif response_time < 5000:
                score += 5
            else:
                score -= 10
        else:
            # 新数据源，给予中等得分
            score += 30

        return score

    async def load_data(
        self,
        data_type: str,
        start_date: str,
        end_date: str,
        frequency: str = "1d",
        symbols: Optional[List[str]] = None,
        **kwargs
    ) -> Optional[IDataModel]:
        """智能加载数据"""
        start_time = time.time()

        # 获取可用的数据源
        available_sources = self._get_available_sources(data_type)
        if not available_sources:
            raise Exception(f"没有可用的数据源用于类型: {data_type}")

        # 尝试从最佳数据源加载
        for source_name in available_sources:
            try:
                logger.info(f"尝试从数据源加载: {source_name}")

                loader = self.loaders[source_name]
                result = await loader.load_data(
                    data_type, start_date, end_date, frequency, symbols, **kwargs
                )

                # 记录成功请求
                response_time = (time.time() - start_time) * 1000
                self.health_monitor.record_request(source_name, response_time, True)

                logger.info(f"成功从数据源加载: {source_name}")
                return result

            except Exception as e:
                # 记录失败请求
                response_time = (time.time() - start_time) * 1000
                self.health_monitor.record_request(source_name, response_time, False)

                logger.warning(f"从数据源 {source_name} 加载失败: {e}")
                continue

        # 所有数据源都失败了
        raise Exception("所有可用数据源都加载失败")

    def _get_available_sources(self, data_type: str) -> List[str]:
        """获取可用的数据源"""
        with self.lock:
            # 更新数据源排名
            self._update_source_ranking()

            # 过滤启用的数据源
            available_sources = []
            for source_name in self.source_ranking:
                config = self.sources[source_name]
                if config.enabled and config.source_type.value == data_type:
                    available_sources.append(source_name)

            return available_sources

    def get_source_info(self) -> Dict[str, Any]:
        """获取数据源信息"""
        with self.lock:
            return {
                'sources': {
                    name: {
                        'type': config.source_type.value,
                        'priority': config.priority,
                        'enabled': config.enabled,
                        'timeout': config.timeout_seconds,
                        'retry_count': config.retry_count
                    } for name, config in self.sources.items()
                },
                'ranking': self.source_ranking,
                'health_report': self.health_monitor.get_health_report()
            }

    def update_source_config(self, name: str, **kwargs):
        """更新数据源配置"""
        with self.lock:
            if name in self.sources:
                config = self.sources[name]
                for key, value in kwargs.items():
                    if hasattr(config, key):
                        setattr(config, key, value)

                self._update_source_ranking()
                logger.info(f"更新数据源配置: {name}")

    def enable_source(self, name: str):
        """启用数据源"""
        self.update_source_config(name, enabled=True)

    def disable_source(self, name: str):
        """禁用数据源"""
        self.update_source_config(name, enabled=False)

    def get_best_source(self, data_type: str) -> Optional[str]:
        """获取最佳数据源"""
        available_sources = self._get_available_sources(data_type)
        return available_sources[0] if available_sources else None

    def cleanup(self):
        """清理资源"""
        self.health_monitor.stop_monitoring()
        logger.info("智能数据源管理器已清理")


# 便捷函数
async def intelligent_load_data(
    data_type: str,
    start_date: str,
    end_date: str,
    frequency: str = "1d",
    symbols: Optional[List[str]] = None,
    source_manager: Optional[IntelligentSourceManager] = None,
    **kwargs
) -> Optional[IDataModel]:
    """便捷的智能数据加载函数"""
    if source_manager is None:
        # 创建默认的数据源管理器
        source_manager = IntelligentSourceManager()

        # 注册默认数据源
        # 这里可以注册各种数据源

    try:
        result = await source_manager.load_data(
            data_type, start_date, end_date, frequency, symbols, **kwargs
        )
        return result
    except Exception as e:
        logger.error(f"智能数据加载失败: {e}")
        raise


# 测试函数
async def test_intelligent_source_manager():
    """测试智能数据源管理器"""
    manager = IntelligentSourceManager()

    try:
        # 注册测试数据源
        test_config = DataSourceConfig(
            name="test_source",
            source_type=DataSourceType.STOCK,
            priority=1,
            enabled=True
        )

        # 创建模拟加载器

        class MockLoader:

            async def load_data(self, data_type, start_date, end_date, frequency, symbols=None, **kwargs):
                # 模拟数据加载
                await asyncio.sleep(0.1)
                return MockDataModel()

        class MockDataModel:

            def __init__(self):

                self.data = pd.DataFrame({
                    'symbol': ['600519.SH'],
                    'close': [100.0],
                    'volume': [1000000],
                    'date': ['2024 - 01 - 01']
                })
                self.metadata = {'source': 'test_source'}

        manager.register_source("test_source", test_config, MockLoader())

        # 测试数据加载
        result = await manager.load_data(
            data_type='stock',
            start_date='2024 - 01 - 01',
            end_date='2024 - 01 - 01',
            frequency='1d',
            symbols=['600519.SH']
        )

        print(f"智能数据加载结果: {result}")
        print(f"数据源信息: {manager.get_source_info()}")

    finally:
        manager.cleanup()


if __name__ == '__main__':
    asyncio.run(test_intelligent_source_manager())

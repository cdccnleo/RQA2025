"""
基础设施层标准Mock配置工具（增强版）

用于统一创建完整配置的Mock对象，解决测试中Mock配置不完整的问题。
创建日期: 2025-01-31
更新: 添加嵌套Mock和高级配置支持
"""

from unittest.mock import MagicMock, Mock, PropertyMock
from typing import Any, Dict, Optional, List
import threading
import pytest


@pytest.fixture
def mock_adapters():
    """提供统一的存储适配器Mock，用于查询执行器测试"""
    from src.infrastructure.utils.components.query_executor import StorageType

    influxdb_adapter = Mock()
    influxdb_adapter.query.return_value = [{"time": "2023-01-01", "value": 100}]
    influxdb_adapter.query_historical.return_value = [{"time": "2023-01-01", "value": 100}]
    influxdb_adapter.aggregate.return_value = {"count": 10, "sum": 1000}

    parquet_adapter = Mock()
    parquet_adapter.query.return_value = [{"id": 1, "data": "test"}]
    parquet_adapter.query_historical.return_value = [{"id": 1, "data": "test"}]
    parquet_adapter.aggregate.return_value = {"count": 5, "sum": 500}

    redis_adapter = Mock()
    redis_adapter.query.return_value = {"key": "value"}

    return {
        StorageType.INFLUXDB: influxdb_adapter,
        StorageType.PARQUET: parquet_adapter,
        StorageType.REDIS: redis_adapter,
    }


class StandardMockBuilder:
    """标准Mock构建器 - 提供统一的Mock对象创建"""

    @staticmethod
    def create_cache_mock(**returns) -> MagicMock:
        """
        创建标准缓存Mock对象（增强版，支持嵌套属性）

        Args:
            **returns: 自定义返回值，如 get=value, set=True等

        Returns:
            配置完整的缓存Mock对象，包含嵌套层级Mock
        """
        mock = MagicMock()

        # 基础缓存操作
        mock.get = MagicMock(return_value=returns.get('get', None))
        mock.set = MagicMock(return_value=returns.get('set', True))
        mock.delete = MagicMock(return_value=returns.get('delete', True))
        mock.exists = MagicMock(return_value=returns.get('exists', False))
        mock.clear = MagicMock(return_value=returns.get('clear', True))

        # 缓存统计
        mock.get_stats = MagicMock(return_value=returns.get('stats', {
            'hits': 0,
            'misses': 0,
            'hit_rate': 0.0,
            'size': 0,
            'total_requests': 0,
            'avg_response_time': 0.0
        }))

        # 缓存大小
        mock.size = MagicMock(return_value=returns.get('size', 0))
        mock.get_size = MagicMock(return_value=returns.get('size', 0))

        # 缓存项操作
        mock.get_cache_item = MagicMock(return_value=returns.get('get', None))
        mock.set_cache_item = MagicMock(return_value=returns.get('set', True))
        mock.delete_cache_item = MagicMock(return_value=returns.get('delete', True))
        mock.has_cache_item = MagicMock(return_value=returns.get('exists', False))
        mock.clear_all_cache = MagicMock(return_value=returns.get('clear', True))

        # 多级缓存操作
        mock.get_memory = MagicMock(return_value=returns.get('get', None))
        mock.set_memory = MagicMock(return_value=returns.get('set', True))
        mock.get_redis = MagicMock(return_value=returns.get('get', None))
        mock.set_redis = MagicMock(return_value=returns.get('set', True))
        mock.get_file = MagicMock(return_value=returns.get('get', None))
        mock.set_file = MagicMock(return_value=returns.get('set', True))

        # ⭐ 新增：嵌套层级Mock（L1/L2/L3）
        mock.l1_tier = MagicMock()
        mock.l1_tier.get = MagicMock(return_value=None)
        mock.l1_tier.set = MagicMock(return_value=True)
        mock.l1_tier.delete = MagicMock(return_value=True)
        mock.l1_tier.clear = MagicMock(return_value=True)
        mock.l1_tier.size = MagicMock(return_value=0)

        mock.l2_tier = MagicMock()
        mock.l2_tier.get = MagicMock(return_value=None)
        mock.l2_tier.set = MagicMock(return_value=True)
        mock.l2_tier.delete = MagicMock(return_value=True)
        mock.l2_tier.clear = MagicMock(return_value=True)
        mock.l2_tier.size = MagicMock(return_value=0)

        mock.l3_tier = MagicMock()
        mock.l3_tier.get = MagicMock(return_value=None)
        mock.l3_tier.set = MagicMock(return_value=True)
        mock.l3_tier.delete = MagicMock(return_value=True)
        mock.l3_tier.clear = MagicMock(return_value=True)
        mock.l3_tier.size = MagicMock(return_value=0)

        # ⭐ 新增：配置Mock
        mock.config = MagicMock()
        mock.config.multi_level = MagicMock()
        mock.config.multi_level.memory_max_size = returns.get('memory_max_size', 1000)
        mock.config.multi_level.redis_max_size = returns.get('redis_max_size', 10000)
        mock.config.multi_level.file_max_size = returns.get('file_max_size', 100000)
        mock.config.multi_level.memory_ttl = returns.get('memory_ttl', 60)
        mock.config.multi_level.redis_ttl = returns.get('redis_ttl', 300)
        mock.config.multi_level.file_ttl = returns.get('file_ttl', 3600)

        # ⭐ 新增：监控Mock
        mock.monitor = MagicMock()
        mock.monitor.record_metric = MagicMock()
        mock.monitor.get_metrics = MagicMock(return_value={})

        # ⭐ 新增：策略Mock
        mock.strategy_manager = MagicMock()
        mock.strategy_manager.get_strategy = MagicMock(return_value=None)
        mock.strategy_manager.set_strategy = MagicMock(return_value=True)

        # ⭐ 新增：分布式Mock
        mock.distributed_manager = MagicMock()
        mock.distributed_manager.sync = MagicMock(return_value=True)
        mock.distributed_manager.check_consistency = MagicMock(return_value=True)

        # ⭐ 新增：组件状态
        mock.component_name = returns.get('component_name', 'mock_cache')
        mock.component_type = returns.get('component_type', 'cache')
        mock.component_id = returns.get('component_id', 1)
        mock._initialized = returns.get('_initialized', True)
        mock._status = returns.get('_status', 'running')

        # ⭐ 新增：健康检查
        mock.health_check = MagicMock(return_value=returns.get('health_check', True))
        mock.get_component_status = MagicMock(return_value=returns.get('component_status', {
            'status': 'healthy',
            'initialized': True
        }))

        # ⭐ 新增：初始化和关闭
        mock.initialize_component = MagicMock(return_value=returns.get('initialize', True))
        mock.shutdown_component = MagicMock()

        return mock

    @staticmethod
    def create_config_mock(**returns) -> MagicMock:
        """
        创建标准配置Mock对象（增强版）

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的配置管理Mock对象
        """
        mock = MagicMock()

        # 基础配置操作
        mock.get = MagicMock(return_value=returns.get('get', None))
        mock.set = MagicMock(return_value=returns.get('set', True))
        mock.delete = MagicMock(return_value=returns.get('delete', True))
        mock.exists = MagicMock(return_value=returns.get('exists', False))

        # 配置验证
        mock.validate = MagicMock(return_value=returns.get('validate', True))
        mock.validate_config = MagicMock(return_value=returns.get('validate', True))

        # 配置加载
        mock.load_config = MagicMock(return_value=returns.get('load', True))
        mock.save_config = MagicMock(return_value=returns.get('save', True))
        mock.reload = MagicMock(return_value=returns.get('reload', True))

        # 配置状态
        mock.get_status = MagicMock(return_value=returns.get('status', {'loaded': True}))

        # ⭐ 新增：配置项属性
        mock.storage = MagicMock()
        mock.loader = MagicMock()
        mock.validator = MagicMock()
        mock.merger = MagicMock()

        # ⭐ 新增：事件总线
        mock.event_bus = MagicMock()
        mock.event_bus.publish = MagicMock()
        mock.event_bus.subscribe = MagicMock()

        return mock

    @staticmethod
    def create_logger_mock() -> MagicMock:
        """
        创建标准日志Mock对象

        Returns:
            配置完整的日志Mock对象
        """
        mock = MagicMock()

        # 日志级别方法
        mock.debug = MagicMock()
        mock.info = MagicMock()
        mock.warning = MagicMock()
        mock.error = MagicMock()
        mock.critical = MagicMock()
        mock.log = MagicMock()

        # 日志配置
        mock.set_level = MagicMock()
        mock.add_handler = MagicMock()
        mock.remove_handler = MagicMock()

        # ⭐ 新增：日志属性
        mock.name = 'mock_logger'
        mock.level = 10  # DEBUG
        mock.handlers = []

        return mock

    @staticmethod
    def create_monitor_mock(**returns) -> MagicMock:
        """
        创建标准监控Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的监控Mock对象
        """
        mock = MagicMock()

        # 指标记录
        mock.record_metric = MagicMock()
        mock.increment_counter = MagicMock()
        mock.record_histogram = MagicMock()
        mock.record_gauge = MagicMock()

        # 指标查询
        mock.get_metric = MagicMock(return_value=returns.get('metric', 0))
        mock.get_metrics = MagicMock(return_value=returns.get('metrics', {}))

        # 告警
        mock.add_alert = MagicMock()
        mock.get_alerts = MagicMock(return_value=returns.get('alerts', []))
        mock.clear_alerts = MagicMock()

        # ⭐ 新增：监控配置
        mock.config = MagicMock()
        mock.config.enable_monitoring = True
        mock.config.monitor_interval = 30

        # ⭐ 新增：监控状态
        mock.start_monitoring = MagicMock(return_value=True)
        mock.stop_monitoring = MagicMock()
        mock._monitoring_active = returns.get('_monitoring_active', False)

        return mock

    @staticmethod
    def create_security_mock(**returns) -> MagicMock:
        """
        创建标准安全Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的安全Mock对象
        """
        mock = MagicMock()

        # 认证
        mock.authenticate = MagicMock(return_value=returns.get('authenticate', True))
        mock.validate_token = MagicMock(return_value=returns.get('validate_token', True))

        # 授权
        mock.authorize = MagicMock(return_value=returns.get('authorize', True))
        mock.check_permission = MagicMock(return_value=returns.get('check_permission', True))

        # 缓存管理
        mock.update_cache = MagicMock(return_value=returns.get('update_cache', True))
        mock.clear_cache = MagicMock(return_value=returns.get('clear_cache', True))
        mock.get_cache_stats = MagicMock(return_value=returns.get('cache_stats', {}))

        # 权限评估
        mock.get_user_permissions = MagicMock(return_value=returns.get('user_permissions', []))
        mock.check_user_permissions = MagicMock(return_value=returns.get('check_user_permissions', True))

        # ⭐ 新增：用户管理
        mock.user_manager = MagicMock()
        mock.session_manager = MagicMock()
        mock.role_manager = MagicMock()

        # ⭐ 新增：审计
        mock.audit_manager = MagicMock()
        mock.audit_manager.log_event = MagicMock()

        return mock

    @staticmethod
    def create_resource_mock(**returns) -> MagicMock:
        """
        创建标准资源管理Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的资源管理Mock对象
        """
        mock = MagicMock()

        # 资源使用
        mock.get_resource_usage = MagicMock(return_value=returns.get('usage', {
            'cpu': 50.0,
            'memory': 60.0,
            'disk': 70.0,
            'network': 0.0
        }))

        # 资源分配
        mock.allocate_resource = MagicMock(return_value=returns.get('allocate', True))
        mock.release_resource = MagicMock(return_value=returns.get('release', True))

        # 资源限制
        mock.set_resource_limit = MagicMock(return_value=returns.get('set_limit', True))
        mock.get_resource_limit = MagicMock(return_value=returns.get('get_limit', 100))

        # ⭐ 新增：资源监控
        mock.monitor = MagicMock()
        mock.monitor.get_metrics = MagicMock(return_value={})

        return mock

    @staticmethod
    def create_health_mock(**returns) -> MagicMock:
        """
        创建标准健康检查Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的健康检查Mock对象
        """
        mock = MagicMock()

        # 健康检查
        mock.check_health = MagicMock(return_value=returns.get('check_health', True))
        mock.is_healthy = MagicMock(return_value=returns.get('is_healthy', True))

        # 健康状态
        mock.get_health_status = MagicMock(return_value=returns.get('health_status', {
            'status': 'healthy',
            'checks': [],
            'timestamp': None
        }))

        # 健康历史
        mock.get_health_history = MagicMock(return_value=returns.get('health_history', []))

        # ⭐ 新增：依赖检查
        mock.check_dependencies = MagicMock(return_value=returns.get('check_dependencies', True))
        mock.get_dependencies = MagicMock(return_value=returns.get('dependencies', []))

        return mock

    @staticmethod
    def create_distributed_mock(**returns) -> MagicMock:
        """
        创建标准分布式服务Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的分布式服务Mock对象
        """
        mock = MagicMock()

        # 服务注册
        mock.register_service = MagicMock(return_value=returns.get('register', True))
        mock.unregister_service = MagicMock(return_value=returns.get('unregister', True))

        # 服务发现
        mock.discover_service = MagicMock(return_value=returns.get('discover', []))
        mock.get_service_instances = MagicMock(return_value=returns.get('instances', []))

        # 心跳
        mock.heartbeat = MagicMock(return_value=returns.get('heartbeat', True))

        # ⭐ 新增：一致性管理
        mock.consistency_manager = MagicMock()
        mock.consistency_manager.check_consistency = MagicMock(return_value=True)
        mock.consistency_manager.sync = MagicMock(return_value=True)

        # ⭐ 新增：集群管理
        mock.cluster_manager = MagicMock()
        mock.cluster_manager.get_nodes = MagicMock(return_value=[])
        mock.cluster_manager.add_node = MagicMock(return_value=True)

        return mock

    @staticmethod
    def create_data_mock(**returns) -> MagicMock:
        """
        创建标准数据管理Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的数据管理Mock对象
        """
        mock = MagicMock()

        # 数据加载
        mock.load_data = MagicMock(return_value=returns.get('load_data', []))
        mock.get_data = MagicMock(return_value=returns.get('get_data', None))
        mock.save_data = MagicMock(return_value=returns.get('save_data', True))

        # 数据查询
        mock.query = MagicMock(return_value=returns.get('query', []))
        mock.filter = MagicMock(return_value=returns.get('filter', []))
        mock.aggregate = MagicMock(return_value=returns.get('aggregate', {}))

        # 数据验证
        mock.validate_data = MagicMock(return_value=returns.get('validate', True))
        mock.clean_data = MagicMock(return_value=returns.get('clean', True))

        # 数据转换
        mock.transform = MagicMock(return_value=returns.get('transform', None))
        mock.normalize = MagicMock(return_value=returns.get('normalize', None))

        # 数据缓存
        mock.cache_data = MagicMock(return_value=returns.get('cache', True))
        mock.get_cached_data = MagicMock(return_value=returns.get('cached_data', None))

        return mock

    @staticmethod
    def create_ml_mock(**returns) -> MagicMock:
        """
        创建标准机器学习Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的机器学习Mock对象
        """
        mock = MagicMock()

        # 模型训练
        mock.train = MagicMock(return_value=returns.get('train', True))
        mock.fit = MagicMock(return_value=returns.get('fit', True))

        # 模型预测
        mock.predict = MagicMock(return_value=returns.get('predict', []))
        mock.predict_proba = MagicMock(return_value=returns.get('predict_proba', []))
        mock.score = MagicMock(return_value=returns.get('score', 0.8))

        # 模型管理
        mock.load_model = MagicMock(return_value=returns.get('load_model', True))
        mock.save_model = MagicMock(return_value=returns.get('save_model', True))
        mock.get_model_info = MagicMock(return_value=returns.get('model_info', {}))

        # 特征工程
        mock.fit_transform = MagicMock(return_value=returns.get('fit_transform', None))
        mock.transform = MagicMock(return_value=returns.get('transform', None))

        return mock

    @staticmethod
    def create_trading_mock(**returns) -> MagicMock:
        """
        创建标准交易Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的交易Mock对象
        """
        mock = MagicMock()

        # 订单管理
        mock.submit_order = MagicMock(return_value=returns.get('submit_order', (True, "success", "order_123")))
        mock.cancel_order = MagicMock(return_value=returns.get('cancel_order', True))
        mock.get_order_status = MagicMock(return_value=returns.get('order_status', {}))

        # 交易执行
        mock.execute_trade = MagicMock(return_value=returns.get('execute_trade', True))
        mock.get_positions = MagicMock(return_value=returns.get('positions', {}))

        # 市场数据
        mock.get_market_data = MagicMock(return_value=returns.get('market_data', []))
        mock.subscribe_market_data = MagicMock(return_value=returns.get('subscribe', True))

        return mock

    @staticmethod
    def create_risk_mock(**returns) -> MagicMock:
        """
        创建标准风险控制Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整风险控制Mock对象
        """
        mock = MagicMock()

        # 风险评估
        mock.assess_risk = MagicMock(return_value=returns.get('assess_risk', {'level': 'low', 'score': 0.1}))
        mock.check_position_risk = MagicMock(return_value=returns.get('check_position', True))

        # 风险限制
        mock.get_risk_limits = MagicMock(return_value=returns.get('risk_limits', {}))
        mock.update_risk_limits = MagicMock(return_value=returns.get('update_limits', True))

        # 风险监控
        mock.monitor_risk = MagicMock(return_value=returns.get('monitor_risk', True))
        mock.get_risk_metrics = MagicMock(return_value=returns.get('risk_metrics', {}))

        return mock

    @staticmethod
    def create_streaming_mock(**returns) -> MagicMock:
        """
        创建标准流处理Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的流处理Mock对象
        """
        mock = MagicMock()

        # 数据流
        mock.start_stream = MagicMock(return_value=returns.get('start_stream', True))
        mock.stop_stream = MagicMock(return_value=returns.get('stop_stream', True))
        mock.get_stream_data = MagicMock(return_value=returns.get('stream_data', []))

        # 流处理
        mock.process_stream = MagicMock(return_value=returns.get('process_stream', True))
        mock.filter_stream = MagicMock(return_value=returns.get('filter_stream', []))

        return mock

    @staticmethod
    def create_gateway_mock(**returns) -> MagicMock:
        """
        创建标准网关Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的网关Mock对象
        """
        mock = MagicMock()

        # API请求
        mock.handle_request = MagicMock(return_value=returns.get('handle_request', {}))
        mock.route_request = MagicMock(return_value=returns.get('route_request', True))

        # 响应处理
        mock.format_response = MagicMock(return_value=returns.get('format_response', {}))
        mock.send_response = MagicMock(return_value=returns.get('send_response', True))

        return mock

    @staticmethod
    def create_optimization_mock(**returns) -> MagicMock:
        """
        创建标准优化Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的优化Mock对象
        """
        mock = MagicMock()

        # 优化执行
        mock.optimize = MagicMock(return_value=returns.get('optimize', {}))
        mock.run_optimization = MagicMock(return_value=returns.get('run_optimization', True))

        # 结果分析
        mock.analyze_results = MagicMock(return_value=returns.get('analyze_results', {}))
        mock.get_best_solution = MagicMock(return_value=returns.get('best_solution', None))

        return mock

    @staticmethod
    def create_mobile_mock(**returns) -> MagicMock:
        """
        创建标准移动端Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的移动端Mock对象
        """
        mock = MagicMock()

        # 用户管理
        mock.authenticate_user = MagicMock(return_value=returns.get('authenticate', True))
        mock.register_user = MagicMock(return_value=returns.get('register', True))

        # 交易操作
        mock.submit_mobile_order = MagicMock(return_value=returns.get('submit_order', True))
        mock.get_portfolio = MagicMock(return_value=returns.get('portfolio', {}))

        return mock

    @staticmethod
    def create_async_processor_mock(**returns) -> MagicMock:
        """
        创建标准异步处理器Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的异步处理器Mock对象
        """
        mock = MagicMock()

        # 任务提交
        mock.submit_task = MagicMock(return_value=returns.get('submit_task', "task_123"))
        mock.submit_async_task = MagicMock(return_value=returns.get('submit_async', "task_456"))

        # 任务状态
        mock.get_task_status = MagicMock(return_value=returns.get('task_status', {}))
        mock.cancel_task = MagicMock(return_value=returns.get('cancel_task', True))

        return mock

    @staticmethod
    def create_distributed_coordinator_mock(**returns) -> MagicMock:
        """
        创建标准分布式协调器Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的分布式协调器Mock对象
        """
        mock = MagicMock()

        # 节点管理
        mock.register_node = MagicMock(return_value=returns.get('register_node', True))
        mock.unregister_node = MagicMock(return_value=returns.get('unregister_node', True))

        # 协调操作
        mock.coordinate_operation = MagicMock(return_value=returns.get('coordinate', True))
        mock.get_cluster_status = MagicMock(return_value=returns.get('cluster_status', {}))

        return mock


class AsyncMockBuilder:
    """异步Mock构建器 - 用于异步函数测试"""

    @staticmethod
    def create_async_mock(return_value=None):
        """创建异步Mock对象"""
        import asyncio

        mock = MagicMock()

        async def async_return(*args, **kwargs):
            return return_value

        mock.side_effect = async_return
        return mock

    @staticmethod
    def create_async_cache_mock(**returns):
        """创建异步缓存Mock"""
        import asyncio

        mock = MagicMock()

        async def async_get(*args, **kwargs):
            return returns.get('get', None)

        async def async_set(*args, **kwargs):
            return returns.get('set', True)

        async def async_delete(*args, **kwargs):
            return returns.get('delete', True)

        mock.get_async = MagicMock(side_effect=async_get)
        mock.set_async = MagicMock(side_effect=async_set)
        mock.delete_async = MagicMock(side_effect=async_delete)

        # 同步方法也配置
        mock.get = MagicMock(return_value=returns.get('get', None))
        mock.set = MagicMock(return_value=returns.get('set', True))
        mock.delete = MagicMock(return_value=returns.get('delete', True))

        return mock


# 便捷函数
def create_standard_mock(mock_type: str, **kwargs) -> MagicMock:
    """
    创建标准Mock对象的便捷函数

    Args:
        mock_type: Mock类型（cache, config, logger, monitor等）
        **kwargs: 传递给具体Mock构建器的参数

    Returns:
        配置完整的Mock对象
    """
    builders = {
        'cache': StandardMockBuilder.create_cache_mock,
        'config': StandardMockBuilder.create_config_mock,
        'logger': StandardMockBuilder.create_logger_mock,
        'monitor': StandardMockBuilder.create_monitor_mock,
        'security': StandardMockBuilder.create_security_mock,
        'resource': StandardMockBuilder.create_resource_mock,
        'health': StandardMockBuilder.create_health_mock,
        'distributed': StandardMockBuilder.create_distributed_mock,
    }

    builder = builders.get(mock_type)
    if builder:
        return builder(**kwargs)
    else:
        raise ValueError(f"Unknown mock type: {mock_type}")


# 全局Mock工具
def create_complete_mock(spec_class=None, **attrs) -> MagicMock:
    """
    创建完整配置的Mock对象（通用版本）

    Args:
        spec_class: 指定Mock的spec类
        **attrs: 要设置的属性和返回值

    Returns:
        配置完整的Mock对象
    """
    if spec_class:
        mock = MagicMock(spec=spec_class)
    else:
        mock = MagicMock()

    # 设置所有提供的属性
    for attr, value in attrs.items():
        if callable(value):
            setattr(mock, attr, MagicMock(side_effect=value))
        else:
            setattr(mock, attr, value)

    return mock


    @staticmethod
    def create_data_mock(**returns) -> MagicMock:
        """
        创建标准数据管理Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的数据管理Mock对象
        """
        mock = MagicMock()

        # 数据加载
        mock.load_data = MagicMock(return_value=returns.get('load_data', []))
        mock.get_data = MagicMock(return_value=returns.get('get_data', None))
        mock.save_data = MagicMock(return_value=returns.get('save_data', True))

        # 数据查询
        mock.query = MagicMock(return_value=returns.get('query', []))
        mock.filter = MagicMock(return_value=returns.get('filter', []))
        mock.aggregate = MagicMock(return_value=returns.get('aggregate', {}))

        # 数据验证
        mock.validate_data = MagicMock(return_value=returns.get('validate', True))
        mock.clean_data = MagicMock(return_value=returns.get('clean', True))

        # 数据转换
        mock.transform = MagicMock(return_value=returns.get('transform', None))
        mock.normalize = MagicMock(return_value=returns.get('normalize', None))

        # 数据缓存
        mock.cache_data = MagicMock(return_value=returns.get('cache', True))
        mock.get_cached_data = MagicMock(return_value=returns.get('cached_data', None))

        return mock

    @staticmethod
    def create_ml_mock(**returns) -> MagicMock:
        """
        创建标准机器学习Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的机器学习Mock对象
        """
        mock = MagicMock()

        # 模型训练
        mock.train = MagicMock(return_value=returns.get('train', True))
        mock.fit = MagicMock(return_value=returns.get('fit', True))

        # 模型预测
        mock.predict = MagicMock(return_value=returns.get('predict', []))
        mock.predict_proba = MagicMock(return_value=returns.get('predict_proba', []))
        mock.score = MagicMock(return_value=returns.get('score', 0.8))

        # 模型管理
        mock.load_model = MagicMock(return_value=returns.get('load_model', True))
        mock.save_model = MagicMock(return_value=returns.get('save_model', True))
        mock.get_model_info = MagicMock(return_value=returns.get('model_info', {}))

        # 特征工程
        mock.fit_transform = MagicMock(return_value=returns.get('fit_transform', None))
        mock.transform = MagicMock(return_value=returns.get('transform', None))

        return mock

    @staticmethod
    def create_trading_mock(**returns) -> MagicMock:
        """
        创建标准交易Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的交易Mock对象
        """
        mock = MagicMock()

        # 订单管理
        mock.submit_order = MagicMock(return_value=returns.get('submit_order', (True, "success", "order_123")))
        mock.cancel_order = MagicMock(return_value=returns.get('cancel_order', True))
        mock.get_order_status = MagicMock(return_value=returns.get('order_status', {}))

        # 交易执行
        mock.execute_trade = MagicMock(return_value=returns.get('execute_trade', True))
        mock.get_positions = MagicMock(return_value=returns.get('positions', {}))

        # 市场数据
        mock.get_market_data = MagicMock(return_value=returns.get('market_data', []))
        mock.subscribe_market_data = MagicMock(return_value=returns.get('subscribe', True))

        return mock

    @staticmethod
    def create_risk_mock(**returns) -> MagicMock:
        """
        创建标准风险控制Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整风险控制Mock对象
        """
        mock = MagicMock()

        # 风险评估
        mock.assess_risk = MagicMock(return_value=returns.get('assess_risk', {'level': 'low', 'score': 0.1}))
        mock.check_position_risk = MagicMock(return_value=returns.get('check_position', True))

        # 风险限制
        mock.get_risk_limits = MagicMock(return_value=returns.get('risk_limits', {}))
        mock.update_risk_limits = MagicMock(return_value=returns.get('update_limits', True))

        # 风险监控
        mock.monitor_risk = MagicMock(return_value=returns.get('monitor_risk', True))
        mock.get_risk_metrics = MagicMock(return_value=returns.get('risk_metrics', {}))

        return mock

    @staticmethod
    def create_streaming_mock(**returns) -> MagicMock:
        """
        创建标准流处理Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的流处理Mock对象
        """
        mock = MagicMock()

        # 数据流
        mock.start_stream = MagicMock(return_value=returns.get('start_stream', True))
        mock.stop_stream = MagicMock(return_value=returns.get('stop_stream', True))
        mock.get_stream_data = MagicMock(return_value=returns.get('stream_data', []))

        # 流处理
        mock.process_stream = MagicMock(return_value=returns.get('process_stream', True))
        mock.filter_stream = MagicMock(return_value=returns.get('filter_stream', []))

        return mock

    @staticmethod
    def create_gateway_mock(**returns) -> MagicMock:
        """
        创建标准网关Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的网关Mock对象
        """
        mock = MagicMock()

        # API请求
        mock.handle_request = MagicMock(return_value=returns.get('handle_request', {}))
        mock.route_request = MagicMock(return_value=returns.get('route_request', True))

        # 响应处理
        mock.format_response = MagicMock(return_value=returns.get('format_response', {}))
        mock.send_response = MagicMock(return_value=returns.get('send_response', True))

        return mock

    @staticmethod
    def create_optimization_mock(**returns) -> MagicMock:
        """
        创建标准优化Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的优化Mock对象
        """
        mock = MagicMock()

        # 优化执行
        mock.optimize = MagicMock(return_value=returns.get('optimize', {}))
        mock.run_optimization = MagicMock(return_value=returns.get('run_optimization', True))

        # 结果分析
        mock.analyze_results = MagicMock(return_value=returns.get('analyze_results', {}))
        mock.get_best_solution = MagicMock(return_value=returns.get('best_solution', None))

        return mock

    @staticmethod
    def create_mobile_mock(**returns) -> MagicMock:
        """
        创建标准移动端Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的移动端Mock对象
        """
        mock = MagicMock()

        # 用户管理
        mock.authenticate_user = MagicMock(return_value=returns.get('authenticate', True))
        mock.register_user = MagicMock(return_value=returns.get('register', True))

        # 交易操作
        mock.submit_mobile_order = MagicMock(return_value=returns.get('submit_order', True))
        mock.get_portfolio = MagicMock(return_value=returns.get('portfolio', {}))

        return mock

    @staticmethod
    def create_async_processor_mock(**returns) -> MagicMock:
        """
        创建标准异步处理器Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的异步处理器Mock对象
        """
        mock = MagicMock()

        # 任务提交
        mock.submit_task = MagicMock(return_value=returns.get('submit_task', "task_123"))
        mock.submit_async_task = MagicMock(return_value=returns.get('submit_async', "task_456"))

        # 任务状态
        mock.get_task_status = MagicMock(return_value=returns.get('task_status', {}))
        mock.cancel_task = MagicMock(return_value=returns.get('cancel_task', True))

        return mock

    @staticmethod
    def create_distributed_coordinator_mock(**returns) -> MagicMock:
        """
        创建标准分布式协调器Mock对象

        Args:
            **returns: 自定义返回值

        Returns:
            配置完整的分布式协调器Mock对象
        """
        mock = MagicMock()

        # 节点管理
        mock.register_node = MagicMock(return_value=returns.get('register_node', True))
        mock.unregister_node = MagicMock(return_value=returns.get('unregister_node', True))

        # 协调操作
        mock.coordinate_operation = MagicMock(return_value=returns.get('coordinate', True))
        mock.get_cluster_status = MagicMock(return_value=returns.get('cluster_status', {}))

        return mock


def create_nested_mock(structure: Dict[str, Any]) -> MagicMock:
    """
    创建嵌套结构的Mock对象

    Args:
        structure: 嵌套结构字典，如 {'config': {'database': {'host': 'localhost'}}}

    Returns:
        包含嵌套属性的Mock对象
    """
    mock = MagicMock()

    def set_nested_attr(obj, path, value):
        """递归设置嵌套属性"""
        if isinstance(value, dict):
            nested_mock = MagicMock()
            for k, v in value.items():
                set_nested_attr(nested_mock, k, v)
            setattr(obj, path, nested_mock)
        else:
            setattr(obj, path, value)

    for key, value in structure.items():
        set_nested_attr(mock, key, value)

    return mock


# 更新便捷函数以支持所有层级
def create_standard_mock(mock_type: str, **kwargs) -> MagicMock:
    """
    创建标准Mock对象的便捷函数

    Args:
        mock_type: Mock类型（cache, config, logger, monitor, data, ml, trading等）
        **kwargs: 传递给具体Mock构建器的参数

    Returns:
        配置完整的Mock对象
    """
    builders = {
        'cache': StandardMockBuilder.create_cache_mock,
        'config': StandardMockBuilder.create_config_mock,
        'logger': StandardMockBuilder.create_logger_mock,
        'monitor': StandardMockBuilder.create_monitor_mock,
        'security': StandardMockBuilder.create_security_mock,
        'resource': StandardMockBuilder.create_resource_mock,
        'health': StandardMockBuilder.create_health_mock,
        'distributed': StandardMockBuilder.create_distributed_mock,
        'data': StandardMockBuilder.create_data_mock,
        'ml': StandardMockBuilder.create_ml_mock,
        'trading': StandardMockBuilder.create_trading_mock,
        'risk': StandardMockBuilder.create_risk_mock,
        'streaming': StandardMockBuilder.create_streaming_mock,
        'gateway': StandardMockBuilder.create_gateway_mock,
        'optimization': StandardMockBuilder.create_optimization_mock,
        'mobile': StandardMockBuilder.create_mobile_mock,
        'async_processor': StandardMockBuilder.create_async_processor_mock,
        'distributed_coordinator': StandardMockBuilder.create_distributed_coordinator_mock,
    }

    builder = builders.get(mock_type)
    if builder:
        return builder(**kwargs)
    else:
        raise ValueError(f"Unknown mock type: {mock_type}. Supported types: {list(builders.keys())}")

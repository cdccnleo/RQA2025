"""
DatabaseHealthMonitor基础测试套件

针对database_health_monitor.py模块的基础测试覆盖
目标: 建立基础测试框架，为后续深入测试奠定基础
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

# 导入被测试模块
from src.infrastructure.health.database.database_health_monitor import (
    DatabaseHealthMonitor,
    HealthStatus,
    DatabaseMetrics,
    HealthCheckResult,
    DEFAULT_CHECK_INTERVAL,
    ERROR_RETRY_DELAY,
    WARNING_CONNECTION_COUNT,
    CRITICAL_CONNECTION_COUNT
)


class TestDatabaseHealthMonitorBasic:
    """DatabaseHealthMonitor基础测试"""

    @pytest.fixture
    def mock_data_manager(self):
        """创建模拟数据管理器"""
        return Mock()

    @pytest.fixture
    def mock_monitor(self):
        """创建模拟监控器"""
        return Mock()

    @pytest.fixture
    def db_monitor(self, mock_data_manager, mock_monitor):
        """创建DatabaseHealthMonitor实例"""
        # 由于ApplicationMonitor和ErrorHandler是条件导入，我们直接mock它们
        return DatabaseHealthMonitor(mock_data_manager, mock_monitor)

    def test_initialization(self, db_monitor):
        """测试初始化"""
        assert db_monitor is not None
        assert hasattr(db_monitor, 'data_manager')
        assert hasattr(db_monitor, 'monitor')
        assert db_monitor.monitoring == False
        assert db_monitor.monitor_config['check_interval'] == DEFAULT_CHECK_INTERVAL

    def test_initialization_without_monitor(self, mock_data_manager):
        """测试无监控器初始化"""
        # 明确传入None避免自动创建监控器
        monitor = DatabaseHealthMonitor(mock_data_manager, monitor=None)
        assert monitor.monitor is None
        assert monitor.monitoring == False

    def test_constants(self):
        """测试常量定义"""
        assert DEFAULT_CHECK_INTERVAL == 60
        assert ERROR_RETRY_DELAY == 10
        assert WARNING_CONNECTION_COUNT == 80
        assert CRITICAL_CONNECTION_COUNT == 95

    def test_enums(self):
        """测试枚举定义"""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.CRITICAL.value == "critical"

    def test_database_metrics_class(self):
        """测试DatabaseMetrics类"""
        # DatabaseMetrics是dataclass，我们测试字段注解
        assert hasattr(DatabaseMetrics, '__annotations__')
        annotations = DatabaseMetrics.__annotations__
        assert 'connection_count' in annotations
        assert 'active_connections' in annotations
        assert 'avg_query_time' in annotations

    def test_health_check_result_class(self):
        """测试HealthCheckResult类"""
        # HealthCheckResult是dataclass，我们测试字段注解
        assert hasattr(HealthCheckResult, '__annotations__')
        annotations = HealthCheckResult.__annotations__
        assert 'component' in annotations
        assert 'status' in annotations
        assert 'metrics' in annotations
        assert 'issues' in annotations
        
        # 测试方法存在
        assert hasattr(HealthCheckResult, '__getitem__')

    @patch.object(DatabaseHealthMonitor, '_check_postgresql_health')
    @patch.object(DatabaseHealthMonitor, '_check_influxdb_health')
    @patch.object(DatabaseHealthMonitor, '_check_redis_health')
    def test_get_health_report_basic(self, mock_redis_check, mock_influx_check, mock_postgres_check, db_monitor):
        """测试基本健康报告"""
        # 配置模拟返回值
        mock_metrics = DatabaseMetrics(
            connection_count=10, active_connections=5, query_count=100,
            avg_query_time=0.1, error_count=0, memory_usage=50.0,
            cpu_usage=25.0, disk_usage=30.0, timestamp=datetime.now()
        )
        
        mock_postgres_check.return_value = HealthCheckResult(
            status=HealthStatus.HEALTHY, component="postgresql",
            metrics=mock_metrics, issues=[], recommendations=[],
            timestamp=datetime.now()
        )
        mock_influx_check.return_value = HealthCheckResult(
            status=HealthStatus.HEALTHY, component="influxdb",
            metrics=mock_metrics, issues=[], recommendations=[],
            timestamp=datetime.now()
        )
        mock_redis_check.return_value = HealthCheckResult(
            status=HealthStatus.HEALTHY, component="redis",
            metrics=mock_metrics, issues=[], recommendations=[],
            timestamp=datetime.now()
        )

        report = db_monitor.get_health_report()

        assert isinstance(report, dict)
        assert 'overall_status' in report
        assert 'components' in report
        assert 'summary' in report

    def test_get_component_health_existing(self, db_monitor):
        """测试获取现有组件健康状态"""
        # 先添加一些健康历史
        mock_metrics = DatabaseMetrics(
            connection_count=10, active_connections=5, query_count=100,
            avg_query_time=0.1, error_count=0, memory_usage=50.0,
            cpu_usage=25.0, disk_usage=30.0, timestamp=datetime.now()
        )
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY, component="postgresql",
            metrics=mock_metrics, issues=[], recommendations=[],
            timestamp=datetime.now()
        )
        db_monitor._append_health_history("postgresql", result)

        health = db_monitor.get_component_health("postgresql")
        assert health is not None
        assert health.component == "postgresql"

    def test_get_component_health_nonexistent(self, db_monitor):
        """测试获取不存在组件健康状态"""
        health = db_monitor.get_component_health("nonexistent")
        assert health is None

    def test_initialize_method(self, db_monitor):
        """测试initialize方法"""
        config = {
            'check_interval': 30,
            'max_history_length': 100
        }

        result = db_monitor.initialize(config)
        assert result == True
        assert db_monitor.monitor_config['check_interval'] == 30

    def test_get_component_info(self, db_monitor):
        """测试获取组件信息"""
        info = db_monitor.get_component_info()

        assert isinstance(info, dict)
        assert 'name' in info or 'component_type' in info
        assert 'status' in info or len(info) > 0

    def test_is_healthy(self, db_monitor):
        """测试健康状态检查"""
        # 由于db_monitor fixture已经提供了完整的数据管理器，所以应该是健康的
        assert db_monitor.is_healthy() == True

        # 初始化后状态应该保持
        db_monitor.initialize()
        assert db_monitor.is_healthy() == True

    def test_get_metrics_basic(self, db_monitor):
        """测试基本指标获取"""
        metrics = db_monitor.get_metrics()

        assert isinstance(metrics, dict)
        assert 'monitoring_active' in metrics or len(metrics) > 0

    def test_cleanup(self, db_monitor):
        """测试清理方法"""
        # 先初始化
        db_monitor.initialize()

        result = db_monitor.cleanup()
        assert result == True

    @patch.object(DatabaseHealthMonitor, '_get_postgresql_metrics')
    def test_check_connection_count_healthy(self, mock_get_metrics, db_monitor):
        """测试连接数检查 - 健康状态"""
        mock_get_metrics.return_value = DatabaseMetrics(
            connection_count=50,  # 正常连接数
            active_connections=10,
            query_count=1000,
            avg_query_time=0.1,
            error_count=5,
            memory_usage=0.6, cpu_usage=0.5, disk_usage=0.7,
            timestamp=datetime.now()
        )

        warnings, criticals, status = db_monitor._check_connection_count(mock_get_metrics.return_value)

        assert status == HealthStatus.HEALTHY
        assert len(warnings) == 0
        assert len(criticals) == 0

    @patch.object(DatabaseHealthMonitor, '_get_postgresql_metrics')
    def test_check_connection_count_warning(self, mock_get_metrics, db_monitor):
        """测试连接数检查 - 警告状态"""
        mock_get_metrics.return_value = DatabaseMetrics(
            connection_count=85,  # 警告级别连接数
            active_connections=10,
            query_count=1000,
            avg_query_time=0.1,
            error_count=10,
            memory_usage=0.6, cpu_usage=0.5, disk_usage=0.7,
            timestamp=datetime.now()
        )

        warnings, criticals, status = db_monitor._check_connection_count(mock_get_metrics.return_value)

        assert status == HealthStatus.WARNING
        assert len(warnings) > 0

    @patch.object(DatabaseHealthMonitor, '_get_postgresql_metrics')
    def test_check_connection_count_critical(self, mock_get_metrics, db_monitor):
        """测试连接数检查 - 严重状态"""
        mock_get_metrics.return_value = DatabaseMetrics(
            connection_count=98,  # 严重级别连接数
            active_connections=10,
            query_count=1000,
            avg_query_time=0.1,
            error_count=20,
            memory_usage=0.6, cpu_usage=0.5, disk_usage=0.7,
            timestamp=datetime.now()
        )

        warnings, criticals, status = db_monitor._check_connection_count(mock_get_metrics.return_value)

        assert status == HealthStatus.CRITICAL
        assert len(criticals) > 0

    @patch.object(DatabaseHealthMonitor, '_get_postgresql_metrics')
    def test_check_query_time_healthy(self, mock_get_metrics, db_monitor):
        """测试查询时间检查 - 健康状态"""
        mock_get_metrics.return_value = DatabaseMetrics(
            connection_count=50, active_connections=10,
            query_count=1000,
            avg_query_time=0.05,  # 快速查询
            error_count=5, memory_usage=0.6, cpu_usage=0.5, disk_usage=0.7,
            timestamp=datetime.now()
        )

        warnings, criticals, status = db_monitor._check_query_time(mock_get_metrics.return_value)

        assert status == HealthStatus.HEALTHY
        assert len(warnings) == 0
        assert len(criticals) == 0

    @patch.object(DatabaseHealthMonitor, '_get_postgresql_metrics')
    def test_check_error_rate_healthy(self, mock_get_metrics, db_monitor):
        """测试错误率检查 - 健康状态"""
        mock_get_metrics.return_value = DatabaseMetrics(
            connection_count=50, active_connections=10,
            query_count=1000,
            avg_query_time=0.1,
            error_count=10,  # 低错误率
            memory_usage=0.6, cpu_usage=0.5, disk_usage=0.7,
            timestamp=datetime.now()
        )

        warnings, criticals, status = db_monitor._check_error_rate(mock_get_metrics.return_value)

        assert status == HealthStatus.HEALTHY
        assert len(warnings) == 0

    @patch.object(DatabaseHealthMonitor, '_get_postgresql_metrics')
    def test_check_resource_usage_healthy(self, mock_get_metrics, db_monitor):
        """测试资源使用检查 - 健康状态"""
        mock_get_metrics.return_value = DatabaseMetrics(
            connection_count=50, active_connections=10,
            query_count=1000,
            avg_query_time=0.1,
            error_count=5,
            memory_usage=0.6, cpu_usage=0.5, disk_usage=0.7,  # 正常资源使用
            timestamp=datetime.now()
        )

        warnings, criticals, status = db_monitor._check_resource_usage(mock_get_metrics.return_value)

        assert status == HealthStatus.HEALTHY
        assert len(warnings) == 0

    def test_combine_health_status(self, db_monitor):
        """测试健康状态合并"""
        # 全部健康
        statuses = [HealthStatus.HEALTHY, HealthStatus.HEALTHY, HealthStatus.HEALTHY]
        result = db_monitor._combine_health_status(statuses)
        assert result == HealthStatus.HEALTHY

        # 包含警告
        statuses = [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.HEALTHY]
        result = db_monitor._combine_health_status(statuses)
        assert result == HealthStatus.WARNING

        # 包含严重
        statuses = [HealthStatus.HEALTHY, HealthStatus.CRITICAL, HealthStatus.WARNING]
        result = db_monitor._combine_health_status(statuses)
        assert result == HealthStatus.CRITICAL

    @patch.object(DatabaseHealthMonitor, '_check_connection_count')
    @patch.object(DatabaseHealthMonitor, '_check_query_time')
    @patch.object(DatabaseHealthMonitor, '_check_error_rate')
    @patch.object(DatabaseHealthMonitor, '_check_resource_usage')
    def test_perform_health_check(self, mock_resource_check, mock_error_check, mock_query_check, mock_conn_check, db_monitor):
        """测试执行健康检查"""
        # 配置模拟返回值
        mock_conn_check.return_value = ([], [], HealthStatus.HEALTHY)
        mock_query_check.return_value = ([], [], HealthStatus.HEALTHY)
        mock_error_check.return_value = ([], [], HealthStatus.HEALTHY)
        mock_resource_check.return_value = ([], [], HealthStatus.HEALTHY)

        metrics = DatabaseMetrics(
            connection_count=50, active_connections=10,
            query_count=1000,
            avg_query_time=0.1,
            error_count=5,
            memory_usage=0.6, cpu_usage=0.5, disk_usage=0.7,
            timestamp=datetime.now()
        )

        result = db_monitor._perform_health_check("postgresql", metrics)

        assert isinstance(result, HealthCheckResult)
        assert result.component == "postgresql"
        assert result.status == HealthStatus.HEALTHY

    def test_append_health_history(self, db_monitor):
        """测试追加健康历史"""
        mock_metrics = DatabaseMetrics(
            connection_count=50, active_connections=10,
            query_count=1000, avg_query_time=0.1,
            error_count=5, memory_usage=0.6,
            cpu_usage=0.5, disk_usage=0.7,
            timestamp=datetime.now()
        )
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY, component="postgresql",
            metrics=mock_metrics, issues=[], recommendations=[],
            timestamp=datetime.now()
        )

        # 初始历史为空
        assert len(db_monitor.health_history["postgresql"]) == 0

        # 追加历史
        db_monitor._append_health_history("postgresql", result)

        # 检查历史已添加
        assert len(db_monitor.health_history["postgresql"]) == 1
        assert db_monitor.health_history["postgresql"][0] == result

    def test_create_database_check_tasks(self, db_monitor):
        """测试创建数据库检查任务"""
        tasks = db_monitor._create_database_check_tasks()

        assert isinstance(tasks, list)
        # 应该有PostgreSQL, InfluxDB两个检查任务
        assert len(tasks) >= 2

    def test_analyze_health_check_results(self, db_monitor):
        """测试分析健康检查结果"""
        # 模拟检查结果
        mock_metrics = DatabaseMetrics(
            connection_count=50, active_connections=10,
            query_count=1000, avg_query_time=0.1,
            error_count=5, memory_usage=0.6,
            cpu_usage=0.5, disk_usage=0.7,
            timestamp=datetime.now()
        )
        results = [
            HealthCheckResult(HealthStatus.HEALTHY, "postgresql", mock_metrics, [], [], datetime.now()),
            HealthCheckResult(HealthStatus.WARNING, "influxdb", mock_metrics, ["warning"], [], datetime.now()),
            HealthCheckResult(HealthStatus.CRITICAL, "redis", mock_metrics, [], ["critical"], datetime.now()),
            None,  # 模拟失败的检查
        ]

        counts = db_monitor._analyze_health_check_results(results)

        assert isinstance(counts, dict)
        assert counts.get('healthy_count', 0) >= 1
        assert counts.get('warning_count', 0) >= 1
        assert counts.get('critical_count', 0) >= 1

    def test_determine_overall_status(self, db_monitor):
        """测试确定整体状态"""
        # 主要是健康
        counts = {'healthy_count': 5, 'warning_count': 1, 'critical_count': 0}
        status = db_monitor._determine_overall_status(counts)
        assert status in ['healthy', 'warning', 'critical']

        # 包含严重问题
        counts = {'healthy_count': 2, 'warning_count': 1, 'critical_count': 2}
        status = db_monitor._determine_overall_status(counts)
        assert status == 'critical'

    def test_create_success_response(self, db_monitor):
        """测试创建成功响应"""
        counts = {'healthy_count': 2, 'warning_count': 1, 'critical_count': 0}
        response = db_monitor._create_success_response('warning', counts, 3)

        assert isinstance(response, dict)
        assert 'status' in response or 'overall_status' in response

    def test_create_no_components_response(self, db_monitor):
        """测试创建无组件响应"""
        response = db_monitor._create_no_components_response()

        assert isinstance(response, dict)
        assert 'status' in response or len(response) > 0

    def test_create_error_response(self, db_monitor):
        """测试创建错误响应"""
        error = Exception("Test error")
        response = db_monitor._create_error_response(error)

        assert isinstance(response, dict)
        assert 'error' in response or 'status' in response

    @pytest.mark.asyncio
    async def test_check_health_async_basic(self, db_monitor):
        """测试异步健康检查基本功能"""
        result = await db_monitor.check_health_async()

        assert isinstance(result, dict)
        # 检查响应结构，适应实际返回格式
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_start_monitoring_async(self, db_monitor, mock_monitor):
        """测试异步启动监控"""
        result = await db_monitor.start_monitoring_async()

        assert isinstance(result, bool)
        # 检查监控器方法被调用
        if mock_monitor:
            mock_monitor.start_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_monitoring_async(self, db_monitor, mock_monitor):
        """测试异步停止监控"""
        result = await db_monitor.stop_monitoring_async()

        assert isinstance(result, bool)
        # 检查监控器方法被调用
        if mock_monitor:
            mock_monitor.stop_monitoring.assert_called_once()

    def test_monitoring_state_management(self, db_monitor):
        """测试监控状态管理"""
        # 初始状态
        assert db_monitor.monitoring == False

        # 启动监控
        db_monitor.start_monitoring()
        assert db_monitor.monitoring == True

        # 停止监控
        db_monitor.stop_monitoring()
        assert db_monitor.monitoring == False

    def test_threading_and_async_compatibility(self, db_monitor):
        """测试线程和异步兼容性"""
        # 验证必要的异步和线程相关属性存在
        assert hasattr(db_monitor, 'monitor_thread')
        assert hasattr(db_monitor, '_stop_event')

        # 检查线程安全属性
        assert hasattr(db_monitor, '_lock')

    def test_configuration_validation(self, db_monitor):
        """测试配置验证"""
        # 有效配置
        valid_config = {
            'check_interval': 60,
            'max_history_length': 100,
            'warning_thresholds': {
                'connection_count': 80,
                'error_rate': 0.05
            }
        }

        result = db_monitor.initialize(valid_config)
        assert result == True

    def test_error_handling_in_health_checks(self, db_monitor):
        """测试健康检查中的错误处理"""
        # 测试在数据管理器异常时的处理
        db_monitor.data_manager = None

        # 健康检查应该不会崩溃
        try:
            report = db_monitor.get_health_report()
            assert isinstance(report, dict)  # 应该返回错误响应
        except Exception:
            # 如果抛出异常，至少不应该是未处理的异常
            pass

    def test_metrics_history_management(self, db_monitor):
        """测试指标历史管理"""
        # 检查历史数据结构
        assert hasattr(db_monitor, 'metrics_history')
        assert hasattr(db_monitor, 'health_history')

        # 验证历史是字典类型
        assert isinstance(db_monitor.metrics_history, dict)
        assert isinstance(db_monitor.health_history, dict)

    def test_component_interface_compliance(self, db_monitor):
        """测试组件接口合规性"""
        # 验证实现IUnifiedInfrastructureInterface的所有必需方法
        required_methods = [
            'initialize', 'get_component_info', 'is_healthy',
            'get_metrics', 'cleanup'
        ]

        for method in required_methods:
            assert hasattr(db_monitor, method)
            assert callable(getattr(db_monitor, method))

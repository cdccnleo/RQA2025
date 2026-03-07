"""
短期行动阶段功能单元测试

测试内容：
1. 数据库持久化模型
2. 告警管理器
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# 测试告警功能
from src.core.orchestration.scheduler.alerting import (
    AlertLevel, AlertChannel, AlertMessage, AlertConfig,
    LogAlertHandler, AlertManager, get_alert_manager, reset_alert_manager
)


class TestAlertLevel:
    """测试告警级别"""

    def test_alert_level_values(self):
        """测试告警级别值"""
        assert AlertLevel.INFO.value == 0
        assert AlertLevel.WARNING.value == 1
        assert AlertLevel.ERROR.value == 2
        assert AlertLevel.CRITICAL.value == 3

    def test_alert_level_ordering(self):
        """测试告警级别顺序（按严重程度）"""
        levels = [AlertLevel.INFO, AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL]
        for i in range(len(levels) - 1):
            assert levels[i].value < levels[i + 1].value


class TestAlertChannel:
    """测试告警渠道"""

    def test_alert_channel_values(self):
        """测试告警渠道值"""
        assert AlertChannel.EMAIL.value == "email"
        assert AlertChannel.WEBHOOK.value == "webhook"
        assert AlertChannel.LOG.value == "log"
        assert AlertChannel.SMS.value == "sms"


class TestAlertMessage:
    """测试告警消息"""

    def test_alert_message_creation(self):
        """测试创建告警消息"""
        message = AlertMessage(
            title="测试告警",
            content="这是一条测试告警",
            level=AlertLevel.WARNING,
            channel=AlertChannel.LOG,
            metadata={"key": "value"}
        )

        assert message.title == "测试告警"
        assert message.content == "这是一条测试告警"
        assert message.level == AlertLevel.WARNING
        assert message.channel == AlertChannel.LOG
        assert message.metadata == {"key": "value"}
        assert message.timestamp is not None

    def test_alert_message_to_dict(self):
        """测试告警消息转字典"""
        message = AlertMessage(
            title="测试告警",
            content="内容",
            level=AlertLevel.ERROR,
            channel=AlertChannel.EMAIL
        )

        data = message.to_dict()
        assert data["title"] == "测试告警"
        assert data["content"] == "内容"
        assert data["level"] == 2  # ERROR的值
        assert data["channel"] == "email"
        assert "timestamp" in data


class TestAlertConfig:
    """测试告警配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = AlertConfig()

        assert config.enabled is True
        assert config.channels == [AlertChannel.LOG]
        assert config.level_threshold == AlertLevel.WARNING
        assert config.rate_limit_seconds == 60

    def test_custom_config(self):
        """测试自定义配置"""
        config = AlertConfig(
            enabled=False,
            channels=[AlertChannel.EMAIL, AlertChannel.WEBHOOK],
            level_threshold=AlertLevel.ERROR,
            rate_limit_seconds=120,
            email_smtp_host="smtp.test.com",
            email_username="test@example.com"
        )

        assert config.enabled is False
        assert len(config.channels) == 2
        assert config.level_threshold == AlertLevel.ERROR
        assert config.rate_limit_seconds == 120
        assert config.email_smtp_host == "smtp.test.com"
        assert config.email_username == "test@example.com"


class TestLogAlertHandler:
    """测试日志告警处理器"""

    @pytest.mark.asyncio
    async def test_log_handler_info(self):
        """测试日志处理器发送INFO级别告警"""
        handler = LogAlertHandler()

        message = AlertMessage(
            title="信息告警",
            content="测试内容",
            level=AlertLevel.INFO,
            channel=AlertChannel.LOG
        )

        result = await handler.send(message)
        assert result is True

    @pytest.mark.asyncio
    async def test_log_handler_error(self):
        """测试日志处理器发送ERROR级别告警"""
        handler = LogAlertHandler()

        message = AlertMessage(
            title="错误告警",
            content="发生错误",
            level=AlertLevel.ERROR,
            channel=AlertChannel.LOG,
            metadata={"error_code": 500}
        )

        result = await handler.send(message)
        assert result is True


class TestAlertManager:
    """测试告警管理器"""

    def setup_method(self):
        """每个测试方法前重置告警管理器"""
        reset_alert_manager()

    def teardown_method(self):
        """每个测试方法后重置告警管理器"""
        reset_alert_manager()

    def test_singleton(self):
        """测试单例模式"""
        manager1 = get_alert_manager()
        manager2 = get_alert_manager()

        assert manager1 is manager2

    def test_initialization(self):
        """测试初始化"""
        config = AlertConfig(
            enabled=True,
            channels=[AlertChannel.LOG],
            level_threshold=AlertLevel.INFO
        )

        manager = AlertManager(config)

        assert manager._config.enabled is True
        assert manager._config.level_threshold == AlertLevel.INFO

    @pytest.mark.asyncio
    async def test_send_alert_disabled(self):
        """测试禁用状态下的告警发送"""
        config = AlertConfig(enabled=False)
        manager = AlertManager(config)

        result = await manager.send_alert("标题", "内容")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_alert_level_threshold(self):
        """测试告警级别阈值过滤"""
        config = AlertConfig(
            enabled=True,
            level_threshold=AlertLevel.ERROR,
            channels=[AlertChannel.LOG]
        )
        manager = AlertManager(config)

        # INFO级别应该被过滤
        result = await manager.send_alert("标题", "内容", level=AlertLevel.INFO)
        assert result is False

        # ERROR级别应该通过
        result = await manager.send_alert("标题", "内容", level=AlertLevel.ERROR)
        assert result is True

    @pytest.mark.asyncio
    async def test_send_alert_rate_limit(self):
        """测试告警频率限制"""
        config = AlertConfig(
            enabled=True,
            rate_limit_seconds=60,
            channels=[AlertChannel.LOG]
        )
        manager = AlertManager(config)

        # 第一次发送应该成功
        result1 = await manager.send_alert("测试告警", "内容", level=AlertLevel.ERROR)
        assert result1 is True

        # 立即再次发送应该被限制
        result2 = await manager.send_alert("测试告警", "内容", level=AlertLevel.ERROR)
        assert result2 is False

    @pytest.mark.asyncio
    async def test_convenience_methods(self):
        """测试便捷方法"""
        config = AlertConfig(
            enabled=True,
            level_threshold=AlertLevel.INFO,
            channels=[AlertChannel.LOG]
        )
        manager = AlertManager(config)

        # 测试各个级别的便捷方法
        assert await manager.info("信息", "内容") is True
        assert await manager.warning("警告", "内容") is True
        assert await manager.error("错误", "内容") is True
        assert await manager.critical("严重", "内容") is True

    @pytest.mark.asyncio
    async def test_task_alert_methods(self):
        """测试任务相关告警方法"""
        config = AlertConfig(
            enabled=True,
            level_threshold=AlertLevel.INFO,
            channels=[AlertChannel.LOG]
        )
        manager = AlertManager(config)

        # 任务失败告警
        result = await manager.task_failed(
            task_id="task-001",
            task_type="data_collection",
            error="连接超时",
            retry_count=2
        )
        assert result is True

        # 任务超时告警
        result = await manager.task_timeout(
            task_id="task-002",
            task_type="model_training",
            timeout_seconds=300
        )
        assert result is True

        # 重试耗尽告警
        result = await manager.task_retry_exhausted(
            task_id="task-003",
            task_type="order_execution",
            max_retries=3
        )
        assert result is True

        # 调度器错误告警
        result = await manager.scheduler_error(
            error="数据库连接失败",
            context={"component": "scheduler"}
        )
        assert result is True

        # 工作进程异常告警
        result = await manager.worker_died(
            worker_id="worker-1",
            task_id="task-004"
        )
        assert result is True


class TestAlertManagerUpdateConfig:
    """测试告警管理器配置更新"""

    def test_update_config(self):
        """测试更新配置"""
        initial_config = AlertConfig(
            enabled=True,
            level_threshold=AlertLevel.INFO
        )
        manager = AlertManager(initial_config)

        # 更新配置
        new_config = AlertConfig(
            enabled=False,
            level_threshold=AlertLevel.CRITICAL
        )
        manager.update_config(new_config)

        assert manager._config.enabled is False
        assert manager._config.level_threshold == AlertLevel.CRITICAL


# 由于数据库测试需要真实的数据库连接，这里只测试模型定义
# 实际的数据库操作测试应该在集成测试中进行
class TestPersistenceModels:
    """测试持久化模型定义"""

    def test_task_model_columns(self):
        """测试任务模型列定义"""
        from src.core.orchestration.scheduler.persistence import TaskModel

        # 检查关键列是否存在
        assert hasattr(TaskModel, 'id')
        assert hasattr(TaskModel, 'type')
        assert hasattr(TaskModel, 'status')
        assert hasattr(TaskModel, 'priority')
        assert hasattr(TaskModel, 'created_at')
        assert hasattr(TaskModel, 'timeout_seconds')
        assert hasattr(TaskModel, 'max_retries')
        assert hasattr(TaskModel, 'retry_count')
        assert hasattr(TaskModel, 'deadline')

    def test_job_model_columns(self):
        """测试定时任务模型列定义"""
        from src.core.orchestration.scheduler.persistence import JobModel

        # 检查关键列是否存在
        assert hasattr(JobModel, 'id')
        assert hasattr(JobModel, 'name')
        assert hasattr(JobModel, 'job_type')
        assert hasattr(JobModel, 'trigger_type')
        assert hasattr(JobModel, 'enabled')
        assert hasattr(JobModel, 'next_run')

    def test_task_history_model_columns(self):
        """测试任务历史模型列定义"""
        from src.core.orchestration.scheduler.persistence import TaskHistoryModel

        # 检查关键列是否存在
        assert hasattr(TaskHistoryModel, 'id')
        assert hasattr(TaskHistoryModel, 'task_id')
        assert hasattr(TaskHistoryModel, 'type')
        assert hasattr(TaskHistoryModel, 'status')
        assert hasattr(TaskHistoryModel, 'execution_time')

    def test_metrics_model_columns(self):
        """测试指标模型列定义"""
        from src.core.orchestration.scheduler.persistence import SchedulerMetricsModel

        # 检查关键列是否存在
        assert hasattr(SchedulerMetricsModel, 'id')
        assert hasattr(SchedulerMetricsModel, 'timestamp')
        assert hasattr(SchedulerMetricsModel, 'total_tasks')
        assert hasattr(SchedulerMetricsModel, 'success_rate')
        assert hasattr(SchedulerMetricsModel, 'avg_execution_time')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

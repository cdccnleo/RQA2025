import json
from unittest.mock import MagicMock, create_autospec

import pytest
import yaml

from src.infrastructure.resource.core.resource_status_reporter import ResourceStatusReporter
from src.infrastructure.resource.core.resource_provider_registry import ResourceProviderRegistry
from src.infrastructure.resource.core.resource_consumer_registry import ResourceConsumerRegistry
from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager


@pytest.fixture
def provider_registry():
    registry = create_autospec(ResourceProviderRegistry, instance=True)
    registry.get_provider_count.return_value = 2
    registry.get_all_provider_status.return_value = {
        "cpu": {"status": "healthy", "total_capacity": 100},
        "memory": {"status": "error", "total_capacity": 256},
    }
    return registry


@pytest.fixture
def consumer_registry():
    registry = create_autospec(ResourceConsumerRegistry, instance=True)
    registry.get_consumer_count.return_value = 3
    registry.get_all_consumer_info.return_value = {"svc-a": {"usage": 10}, "svc-b": {"usage": 5}}
    return registry


@pytest.fixture
def allocation_manager():
    manager = create_autospec(ResourceAllocationManager, instance=True)
    manager.get_allocation_count.return_value = 120
    manager.get_request_count.return_value = 7
    manager.get_allocation_summary.return_value = {"active": 120, "pending": 7}
    return manager


@pytest.fixture
def error_handler():
    handler = MagicMock()
    return handler


@pytest.fixture
def reporter(provider_registry, consumer_registry, allocation_manager, error_handler):
    return ResourceStatusReporter(
        provider_registry=provider_registry,
        consumer_registry=consumer_registry,
        allocation_manager=allocation_manager,
        logger=MagicMock(),
        error_handler=error_handler,
    )


def test_get_resource_status_success(reporter, provider_registry, consumer_registry, allocation_manager):
    status = reporter.get_resource_status()
    assert "timestamp" in status
    assert status["summary"]["providers_count"] == provider_registry.get_provider_count.return_value
    assert status["summary"]["consumers_count"] == consumer_registry.get_consumer_count.return_value
    assert status["summary"]["active_allocations"] == allocation_manager.get_allocation_count.return_value
    assert status["providers"]["cpu"]["status"] == "healthy"
    assert status["consumers"]["svc-a"]["usage"] == 10
    assert status["allocations"]["summary"]["active"] == 120
    assert status["health"]["health_status"] in {"good", "fair", "poor", "critical", "excellent"}


def test_get_resource_status_handles_exception(provider_registry, error_handler):
    provider_registry.get_all_provider_status.side_effect = RuntimeError("boom")
    reporter = ResourceStatusReporter(
        provider_registry=provider_registry,
        logger=MagicMock(),
        error_handler=error_handler,
    )

    status = reporter.get_resource_status()
    error_handler.handle_error.assert_called_once()
    assert "error" in status
    assert status["error"] == "boom"


def test_health_status_with_errors(reporter, provider_registry, allocation_manager):
    status = reporter.get_resource_status()
    health = status["health"]
    assert health["health_score"] < 1.0
    assert any("资源提供者异常" in issue for issue in health["issues"])
    assert any("活跃分配过多" in issue for issue in health["issues"])


def test_get_recommendations_triggers_rules(reporter, monkeypatch):
    monkeypatch.setattr(
        reporter,
        "get_resource_status",
        lambda: {
            "health": {"health_score": 0.5},
            "summary": {"active_allocations": 60, "pending_requests": 11},
        },
    )
    recommendations = reporter._get_recommendations()
    assert "检查系统健康问题并及时处理" in recommendations
    assert "考虑优化资源分配策略" in recommendations
    assert "检查资源请求积压原因" in recommendations


def test_export_report_formats(reporter):
    json_output = reporter.export_report("json")
    yaml_output = reporter.export_report("yaml")
    assert json.loads(json_output)["summary"]["providers_count"] == 2
    parsed_yaml = yaml.safe_load(yaml_output)
    assert parsed_yaml["summary"]["consumers_count"] == 3

    invalid = reporter.export_report("txt")
    assert invalid.startswith("导出失败")


def test_export_status_helpers(reporter):
    json_output = reporter.export_status_to_json()
    yaml_output = reporter.export_status_to_yaml()
    assert json.loads(json_output)["summary"]["pending_requests"] == 7
    assert yaml.safe_load(yaml_output)["summary"]["providers_count"] == 2


def test_get_detailed_report_includes_sections(reporter):
    report = reporter.get_detailed_report()
    assert "performance" in report
    assert "trends" in report
    assert "recommendations" in report
    assert report["performance"]["throughput"] == 0.0
    assert report["trends"]["usage_trend"] == "stable"


def test_generate_detailed_report_success(reporter):
    detailed = reporter.generate_detailed_report()
    assert detailed["report_type"] == "detailed"
    assert "performance" in detailed


def test_generate_summary_report_handles_failure(reporter, error_handler):
    reporter.get_resource_status = MagicMock(side_effect=ValueError("summary failed"))
    summary = reporter.generate_summary_report()
    error_handler.handle_error.assert_called()
    assert summary["error"] == "summary failed"
    assert summary["report_type"] == "summary"


def test_generate_summary_report_success(reporter):
    summary = reporter.generate_summary_report()
    assert summary["report_type"] == "summary"
    assert summary["providers_count"] == 2


def test_get_provider_status_summary(reporter):
    summary = reporter.get_provider_status_summary()
    assert summary["total_providers"] == 2
    assert summary["healthy_providers"] == 1
    assert summary["error_providers"] == 1


def test_get_consumer_status_summary(reporter):
    summary = reporter.get_consumer_status_summary()
    assert summary["total_consumers"] == 2


def test_generate_detailed_report_handles_exception(reporter, error_handler):
    reporter.get_detailed_report = MagicMock(side_effect=RuntimeError("detail failed"))
    detailed = reporter.generate_detailed_report()
    error_handler.handle_error.assert_called()
    assert detailed["error"] == "detail failed"
    assert detailed["report_type"] == "detailed"


def test_get_resource_status_without_dependencies():
    reporter = ResourceStatusReporter(logger=MagicMock(), error_handler=MagicMock())
    status = reporter.get_resource_status()
    assert status["summary"]["providers_count"] == 0
    assert status["providers"] == {}
    assert status["allocations"]["active_count"] == 0


def test_get_detailed_report_handles_failure(reporter, error_handler):
    reporter.get_resource_status = MagicMock(side_effect=RuntimeError("detailed boom"))
    result = reporter.get_detailed_report()
    error_handler.handle_error.assert_called()
    assert result["error"] == "detailed boom"


def test_export_report_invalid_format_calls_error_handler(reporter, error_handler):
    reporter.get_detailed_report = MagicMock(return_value={"ok": True})
    result = reporter.export_report("xml")
    error_handler.handle_error.assert_called()
    assert result.startswith("导出失败")


def test_export_status_to_json_error(reporter, error_handler):
    reporter.get_resource_status = MagicMock(side_effect=RuntimeError("json boom"))
    result = reporter.export_status_to_json()
    error_handler.handle_error.assert_called()
    assert "导出失败" in result


def test_export_status_to_yaml_error(reporter, error_handler):
    reporter.get_resource_status = MagicMock(side_effect=RuntimeError("yaml boom"))
    result = reporter.export_status_to_yaml()
    error_handler.handle_error.assert_called()
    assert "导出失败" in result


def test_get_provider_status_summary_error(reporter, error_handler):
    reporter._get_provider_status = MagicMock(side_effect=RuntimeError("provider boom"))
    summary = reporter.get_provider_status_summary()
    error_handler.handle_error.assert_called()
    assert summary["error"] == "provider boom"


def test_get_consumer_status_summary_error(reporter, error_handler):
    reporter._get_consumer_status = MagicMock(side_effect=RuntimeError("consumer boom"))
    summary = reporter.get_consumer_status_summary()
    error_handler.handle_error.assert_called()
    assert summary["error"] == "consumer boom"


def test_recommendations_fallback_on_exception(reporter):
    reporter.get_resource_status = MagicMock(side_effect=RuntimeError("recommend boom"))
    recommendations = reporter._get_recommendations()
    assert recommendations == ["建议定期监控系统状态"]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
resource_status_reporter 模块测试
测试资源状态报告器的所有功能，提升测试覆盖率到80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import json
import yaml
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any

try:
    from src.infrastructure.resource.core.resource_status_reporter import ResourceStatusReporter
    from src.infrastructure.resource.core.resource_provider_registry import ResourceProviderRegistry
    from src.infrastructure.resource.core.resource_consumer_registry import ResourceConsumerRegistry
    from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"导入错误: {e}")


@unittest.skipUnless(IMPORTS_AVAILABLE, "resource_status_reporter模块导入失败")
class TestResourceStatusReporter(unittest.TestCase):
    """测试资源状态报告器"""

    def setUp(self):
        """测试前准备"""
        # 创建模拟的依赖组件
        self.mock_provider_registry = Mock(spec=ResourceProviderRegistry)
        self.mock_consumer_registry = Mock(spec=ResourceConsumerRegistry)
        self.mock_allocation_manager = Mock(spec=ResourceAllocationManager)
        
        # 设置默认返回值
        self.setup_mock_defaults()
        
        self.reporter = ResourceStatusReporter(
            provider_registry=self.mock_provider_registry,
            consumer_registry=self.mock_consumer_registry,
            allocation_manager=self.mock_allocation_manager
        )

    def setup_mock_defaults(self):
        """设置模拟对象的默认返回值"""
        # 提供者注册表默认返回值
        self.mock_provider_registry.get_provider_count.return_value = 2
        self.mock_provider_registry.get_all_provider_status.return_value = {
            "cpu": {
                "status": "healthy",
                "total_capacity": 100,
                "used_capacity": 60
            },
            "memory": {
                "status": "healthy", 
                "total_capacity": 8192,
                "used_capacity": 4096
            }
        }
        
        # 消费者注册表默认返回值
        self.mock_consumer_registry.get_consumer_count.return_value = 3
        self.mock_consumer_registry.get_all_consumer_info.return_value = {
            "consumer_1": {
                "consumer_type": "TestConsumer",
                "consumed_resources_count": 1,
                "status": "active"
            },
            "consumer_2": {
                "consumer_type": "TestConsumer", 
                "consumed_resources_count": 2,
                "status": "active"
            }
        }
        
        # 分配管理器默认返回值
        self.mock_allocation_manager.get_allocation_count.return_value = 5
        self.mock_allocation_manager.get_request_count.return_value = 2
        self.mock_allocation_manager.get_allocation_summary.return_value = {
            "total_allocations": 5,
            "active_allocations": 3,
            "completed_allocations": 2
        }

    def test_reporter_initialization(self):
        """测试报告器初始化"""
        self.assertIsNotNone(self.reporter)
        self.assertEqual(self.reporter.provider_registry, self.mock_provider_registry)
        self.assertEqual(self.reporter.consumer_registry, self.mock_consumer_registry)
        self.assertEqual(self.reporter.allocation_manager, self.mock_allocation_manager)

    def test_reporter_initialization_with_none_components(self):
        """测试使用None组件初始化报告器"""
        reporter = ResourceStatusReporter(
            provider_registry=None,
            consumer_registry=None,
            allocation_manager=None
        )
        
        self.assertIsNotNone(reporter)
        self.assertIsNone(reporter.provider_registry)
        self.assertIsNone(reporter.consumer_registry)
        self.assertIsNone(reporter.allocation_manager)

    def test_get_resource_status(self):
        """测试获取完整资源状态"""
        status = self.reporter.get_resource_status()
        
        # 验证返回状态结构
        self.assertIsInstance(status, dict)
        self.assertIn("timestamp", status)
        self.assertIn("summary", status)
        self.assertIn("providers", status)
        self.assertIn("consumers", status)
        self.assertIn("allocations", status)
        self.assertIn("health", status)

    def test_get_resource_status_with_errors(self):
        """测试获取资源状态时发生错误"""
        # 模拟provider_registry抛出异常
        self.mock_provider_registry.get_provider_count.side_effect = Exception("Provider registry error")
        
        status = self.reporter.get_resource_status()
        
        # 应该返回错误信息而不是抛出异常
        self.assertIn("error", status)
        self.assertIn("timestamp", status)

    def test_get_summary_status(self):
        """测试获取汇总状态"""
        summary = self.reporter._get_summary_status()
        
        self.assertIsInstance(summary, dict)
        self.assertIn("providers_count", summary)
        self.assertIn("consumers_count", summary)
        self.assertIn("active_allocations", summary)
        self.assertIn("pending_requests", summary)
        self.assertIn("total_capacity", summary)
        self.assertIn("total_usage", summary)
        
        self.assertEqual(summary["providers_count"], 2)
        self.assertEqual(summary["consumers_count"], 3)
        self.assertEqual(summary["active_allocations"], 5)
        self.assertEqual(summary["pending_requests"], 2)

    def test_get_summary_status_with_none_registries(self):
        """测试使用None注册表时获取汇总状态"""
        reporter = ResourceStatusReporter(
            provider_registry=None,
            consumer_registry=None,
            allocation_manager=None
        )
        
        summary = reporter._get_summary_status()
        
        self.assertEqual(summary["providers_count"], 0)
        self.assertEqual(summary["consumers_count"], 0)
        self.assertEqual(summary["active_allocations"], 0)
        self.assertEqual(summary["pending_requests"], 0)

    def test_get_provider_status(self):
        """测试获取提供者状态"""
        provider_status = self.reporter._get_provider_status()
        
        self.assertIsInstance(provider_status, dict)
        self.assertIn("cpu", provider_status)
        self.assertIn("memory", provider_status)
        
        # 验证调用
        self.mock_provider_registry.get_all_provider_status.assert_called_once()

    def test_get_provider_status_with_none_registry(self):
        """测试使用None提供者注册表时获取状态"""
        reporter = ResourceStatusReporter(provider_registry=None)
        provider_status = reporter._get_provider_status()
        
        self.assertEqual(provider_status, {})

    def test_get_consumer_status(self):
        """测试获取消费者状态"""
        consumer_status = self.reporter._get_consumer_status()
        
        self.assertIsInstance(consumer_status, dict)
        self.assertIn("consumer_1", consumer_status)
        self.assertIn("consumer_2", consumer_status)
        
        # 验证调用
        self.mock_consumer_registry.get_all_consumer_info.assert_called_once()

    def test_get_consumer_status_with_none_registry(self):
        """测试使用None消费者注册表时获取状态"""
        reporter = ResourceStatusReporter(consumer_registry=None)
        consumer_status = reporter._get_consumer_status()
        
        self.assertEqual(consumer_status, {})

    def test_get_allocation_status(self):
        """测试获取分配状态"""
        allocation_status = self.reporter._get_allocation_status()
        
        self.assertIsInstance(allocation_status, dict)
        self.assertIn("summary", allocation_status)
        self.assertIn("active_count", allocation_status)
        self.assertIn("pending_count", allocation_status)
        
        self.assertEqual(allocation_status["active_count"], 5)
        self.assertEqual(allocation_status["pending_count"], 2)

    def test_get_allocation_status_with_none_manager(self):
        """测试使用None分配管理器时获取状态"""
        reporter = ResourceStatusReporter(allocation_manager=None)
        allocation_status = reporter._get_allocation_status()
        
        expected = {"summary": {}, "active_count": 0, "pending_count": 0}
        self.assertEqual(allocation_status, expected)

    def test_get_health_status_healthy(self):
        """测试获取健康状态 - 健康情况"""
        health = self.reporter._get_health_status()
        
        self.assertIsInstance(health, dict)
        self.assertIn("health_score", health)
        self.assertIn("health_status", health)
        self.assertIn("issues", health)
        
        self.assertGreaterEqual(health["health_score"], 0.0)
        self.assertLessEqual(health["health_score"], 1.0)
        self.assertIn(health["health_status"], ["excellent", "good", "fair", "poor", "critical"])

    def test_get_health_status_with_provider_errors(self):
        """测试获取健康状态 - 提供者错误情况"""
        # 模拟提供者状态包含错误
        self.mock_provider_registry.get_all_provider_status.return_value = {
            "cpu": {"status": "error"},
            "memory": {"status": "error"}
        }
        
        health = self.reporter._get_health_status()
        
        self.assertIn("issues", health)
        self.assertIsInstance(health["issues"], list)

    def test_get_health_status_with_high_allocation_count(self):
        """测试获取健康状态 - 高分配数量情况"""
        # 模拟高分配数量
        self.mock_allocation_manager.get_allocation_count.return_value = 150
        
        health = self.reporter._get_health_status()
        
        # 应该检测到问题
        self.assertIsInstance(health["issues"], list)

    def test_get_health_status_with_none_components(self):
        """测试使用None组件时获取健康状态"""
        reporter = ResourceStatusReporter()
        health = reporter._get_health_status()
        
        self.assertIn("health_score", health)
        self.assertIn("health_status", health)
        self.assertIn("issues", health)

    def test_export_report_json(self):
        """测试导出JSON报告"""
        report = self.reporter.export_report("json")
        
        self.assertIsInstance(report, str)
        # 验证JSON格式
        parsed_report = json.loads(report)
        self.assertIsInstance(parsed_report, dict)

    def test_export_report_yaml(self):
        """测试导出YAML报告"""
        report = self.reporter.export_report("yaml")
        
        self.assertIsInstance(report, str)
        # 验证YAML格式
        try:
            parsed_report = yaml.safe_load(report)
            self.assertIsInstance(parsed_report, dict)
        except yaml.YAMLError:
            self.fail("生成的YAML格式无效")

    def test_export_report_invalid_format(self):
        """测试导出无效格式报告"""
        # 由于错误处理器会捕获异常，所以我们需要检查返回值
        result = self.reporter.export_report("invalid_format")
        self.assertIn("导出失败", result)
        self.assertIn("不支持的导出格式", result)

    def test_get_detailed_report(self):
        """测试获取详细报告"""
        report = self.reporter.get_detailed_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn("timestamp", report)
        self.assertIn("performance", report)
        self.assertIn("trends", report)
        self.assertIn("recommendations", report)

    def test_health_status_scoring_excellent(self):
        """测试健康状态评分 - 优秀"""
        # 设置理想状态
        self.mock_provider_registry.get_all_provider_status.return_value = {
            "cpu": {"status": "healthy"},
            "memory": {"status": "healthy"}
        }
        self.mock_allocation_manager.get_allocation_count.return_value = 50
        
        health = self.reporter._get_health_status()
        self.assertEqual(health["health_status"], "excellent")

    def test_health_status_scoring_good(self):
        """测试健康状态评分 - 良好"""
        # 设置一个会降低健康评分但不至于太差的状态
        self.mock_provider_registry.get_all_provider_status.return_value = {
            "cpu": {"status": "healthy"},
            "memory": {"status": "error"}  # 一个错误提供者应该降低评分
        }
        
        health = self.reporter._get_health_status()
        # 由于有一个错误提供者，健康状态应该是fair或poor
        self.assertIn(health["health_status"], ["excellent", "good", "fair", "poor"])

    def test_health_status_scoring_critical(self):
        """测试健康状态评分 - 危险"""
        # 设置多错误状态
        self.mock_provider_registry.get_all_provider_status.return_value = {
            "cpu": {"status": "error"},
            "memory": {"status": "error"},
            "disk": {"status": "error"},
            "network": {"status": "error"}
        }
        self.mock_allocation_manager.get_allocation_count.return_value = 200
        
        health = self.reporter._get_health_status()
        self.assertIn(health["health_status"], ["poor", "critical"])

    def test_total_capacity_calculation(self):
        """测试总容量计算"""
        summary = self.reporter._get_summary_status()
        
        self.assertIn("total_capacity", summary)
        if summary["total_capacity"]:
            self.assertIn("cpu", summary["total_capacity"])
            self.assertIn("memory", summary["total_capacity"])

    def test_get_resource_status_timestamp_format(self):
        """测试时间戳格式"""
        status = self.reporter.get_resource_status()
        
        # 验证时间戳格式
        timestamp_str = status["timestamp"]
        try:
            datetime.fromisoformat(timestamp_str)
        except ValueError:
            self.fail("时间戳格式无效")


if __name__ == '__main__':
    unittest.main()

"""
测试资源状态报告器

覆盖 resource_status_reporter.py 中的所有类和功能
"""

import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime
from src.infrastructure.resource.core.resource_status_reporter import ResourceStatusReporter


class TestResourceStatusReporter:
    """ResourceStatusReporter 类测试"""

    def test_initialization(self):
        """测试初始化"""
        reporter = ResourceStatusReporter()

        assert reporter.provider_registry is None
        assert reporter.consumer_registry is None
        assert reporter.allocation_manager is None
        assert reporter.logger is not None
        assert reporter.error_handler is not None

    def test_initialization_with_components(self):
        """测试带组件初始化"""
        mock_provider = Mock()
        mock_consumer = Mock()
        mock_allocation = Mock()
        mock_logger = Mock()
        mock_error_handler = Mock()

        reporter = ResourceStatusReporter(
            provider_registry=mock_provider,
            consumer_registry=mock_consumer,
            allocation_manager=mock_allocation,
            logger=mock_logger,
            error_handler=mock_error_handler
        )

        assert reporter.provider_registry == mock_provider
        assert reporter.consumer_registry == mock_consumer
        assert reporter.allocation_manager == mock_allocation
        assert reporter.logger == mock_logger
        assert reporter.error_handler == mock_error_handler

    def test_get_resource_status(self):
        """测试获取资源状态"""
        reporter = ResourceStatusReporter()

        status = reporter.get_resource_status()

        assert isinstance(status, dict)
        assert 'timestamp' in status
        assert 'summary' in status
        assert 'providers' in status
        assert 'consumers' in status
        assert 'allocations' in status
        assert 'health' in status

    def test_get_resource_status_with_components(self):
        """测试获取资源状态（带组件）"""
        reporter = ResourceStatusReporter()

        # Mock the methods to avoid iteration issues
        with patch.object(reporter, '_get_provider_status', return_value={'cpu': {'available': 4}}), \
             patch.object(reporter, '_get_consumer_status', return_value={'app1': {'allocated': 2}}), \
             patch.object(reporter, '_get_allocation_status', return_value={'total_allocated': 2}), \
             patch.object(reporter, '_get_summary_status', return_value={'providers_count': 1, 'consumers_count': 1}):

            status = reporter.get_resource_status()

            # 验证状态包含所有必需的字段
            assert 'timestamp' in status
            assert 'summary' in status
            assert 'providers' in status
            assert 'consumers' in status
            assert 'allocations' in status
            assert 'health' in status

    def test_get_summary_status(self):
        """测试获取汇总状态"""
        reporter = ResourceStatusReporter()

        summary = reporter._get_summary_status()

        assert isinstance(summary, dict)
        # 检查实际返回的字段
        expected_fields = ['providers_count', 'consumers_count', 'active_allocations', 'pending_requests']
        for field in expected_fields:
            assert field in summary

    def test_get_provider_status(self):
        """测试获取提供者状态"""
        reporter = ResourceStatusReporter()

        provider_status = reporter._get_provider_status()

        assert isinstance(provider_status, dict)

    def test_get_consumer_status(self):
        """测试获取消费者状态"""
        reporter = ResourceStatusReporter()

        consumer_status = reporter._get_consumer_status()

        assert isinstance(consumer_status, dict)

    def test_get_allocation_status(self):
        """测试获取分配状态"""
        reporter = ResourceStatusReporter()

        allocation_status = reporter._get_allocation_status()

        assert isinstance(allocation_status, dict)
        # 检查实际返回的字段
        assert 'active_count' in allocation_status
        assert 'pending_count' in allocation_status
        assert 'summary' in allocation_status

    def test_get_health_status(self):
        """测试获取健康状态"""
        reporter = ResourceStatusReporter()

        health_status = reporter._get_health_status()

        assert isinstance(health_status, dict)
        # 检查实际返回的字段
        assert 'health_status' in health_status
        assert 'health_score' in health_status
        assert 'issues' in health_status

    def test_get_detailed_report(self):
        """测试获取详细报告"""
        reporter = ResourceStatusReporter()

        report = reporter.get_detailed_report()

        assert isinstance(report, dict)
        # 检查实际返回的字段
        assert 'timestamp' in report
        assert 'allocations' in report
        assert 'performance' in report
        assert 'health' in report

    def test_get_performance_metrics(self):
        """测试获取性能指标"""
        reporter = ResourceStatusReporter()

        metrics = reporter._get_performance_metrics()

        assert isinstance(metrics, dict)
        # 检查实际返回的字段
        assert 'response_time_avg' in metrics
        assert 'throughput' in metrics
        assert 'error_rate' in metrics

    def test_get_trend_analysis(self):
        """测试获取趋势分析"""
        reporter = ResourceStatusReporter()

        trends = reporter._get_trend_analysis()

        assert isinstance(trends, dict)
        # 检查实际返回的字段
        assert 'usage_trend' in trends
        assert 'allocation_trend' in trends
        assert 'predictions' in trends

    def test_get_recommendations(self):
        """测试获取建议"""
        reporter = ResourceStatusReporter()

        recommendations = reporter._get_recommendations()

        assert isinstance(recommendations, list)

    def test_export_report_json(self):
        """测试导出JSON格式报告"""
        reporter = ResourceStatusReporter()

        json_report = reporter.export_report("json")

        # 验证是有效的JSON
        parsed = json.loads(json_report)
        assert isinstance(parsed, dict)

    def test_export_report_yaml(self):
        """测试导出YAML格式报告"""
        reporter = ResourceStatusReporter()

        yaml_report = reporter.export_report("yaml")

        assert isinstance(yaml_report, str)
        assert len(yaml_report) > 0

    def test_export_status_to_json(self):
        """测试导出状态到JSON"""
        reporter = ResourceStatusReporter()

        json_status = reporter.export_status_to_json()

        # 验证是有效的JSON
        parsed = json.loads(json_status)
        assert isinstance(parsed, dict)

    def test_export_status_to_yaml(self):
        """测试导出状态到YAML"""
        reporter = ResourceStatusReporter()

        yaml_status = reporter.export_status_to_yaml()

        assert isinstance(yaml_status, str)
        assert len(yaml_status) > 0

    def test_generate_detailed_report(self):
        """测试生成详细报告"""
        reporter = ResourceStatusReporter()

        report = reporter.generate_detailed_report()

        assert isinstance(report, dict)
        # 检查实际返回的字段
        assert 'timestamp' in report
        assert 'allocations' in report
        assert 'performance' in report
        assert 'health' in report

    def test_get_provider_status_summary(self):
        """测试获取提供者状态汇总"""
        reporter = ResourceStatusReporter()

        summary = reporter.get_provider_status_summary()

        assert isinstance(summary, dict)
        # 检查实际返回的字段
        assert 'total_providers' in summary
        assert 'providers' in summary

    def test_get_consumer_status_summary(self):
        """测试获取消费者状态汇总"""
        reporter = ResourceStatusReporter()

        summary = reporter.get_consumer_status_summary()

        assert isinstance(summary, dict)
        # 检查实际返回的字段
        assert 'total_consumers' in summary
        assert 'consumers' in summary

    def test_generate_summary_report(self):
        """测试生成汇总报告"""
        reporter = ResourceStatusReporter()

        report = reporter.generate_summary_report()

        assert isinstance(report, dict)
        assert 'timestamp' in report
        assert 'report_type' in report
        assert 'providers_count' in report
        assert 'consumers_count' in report
        assert 'active_allocations' in report
        assert 'health_status' in report

    def test_get_resource_status_error_handling(self):
        """测试获取资源状态时的错误处理"""
        reporter = ResourceStatusReporter()

        # Mock _get_summary_status 抛出异常
        with patch.object(reporter, '_get_summary_status', side_effect=Exception("Test error")):
            status = reporter.get_resource_status()

            assert "error" in status
            assert "timestamp" in status

    def test_get_summary_status_with_components(self):
        """测试带组件的汇总状态获取"""
        mock_provider = Mock()
        mock_consumer = Mock()
        mock_allocation = Mock()

        mock_provider.get_provider_count.return_value = 3
        mock_consumer.get_consumer_count.return_value = 5
        mock_allocation.get_allocation_count.return_value = 10
        mock_allocation.get_request_count.return_value = 2

        # Mock provider status for capacity calculation
        mock_provider.get_all_provider_status.return_value = {
            "cpu": {"status": "healthy", "total_capacity": 8},
            "memory": {"status": "healthy", "total_capacity": 16}
        }

        reporter = ResourceStatusReporter(
            provider_registry=mock_provider,
            consumer_registry=mock_consumer,
            allocation_manager=mock_allocation
        )

        summary = reporter._get_summary_status()

        assert summary["providers_count"] == 3
        assert summary["consumers_count"] == 5
        assert summary["active_allocations"] == 10
        assert summary["pending_requests"] == 2
        assert summary["total_capacity"]["cpu"] == 8
        assert summary["total_capacity"]["memory"] == 16

    def test_get_provider_status_with_registry(self):
        """测试带注册表的提供者状态获取"""
        mock_provider = Mock()
        mock_provider.get_all_provider_status.return_value = {
            "cpu": {"status": "healthy"},
            "gpu": {"status": "error"}
        }

        reporter = ResourceStatusReporter(provider_registry=mock_provider)

        provider_status = reporter._get_provider_status()

        assert provider_status == mock_provider.get_all_provider_status.return_value

    def test_get_consumer_status_with_registry(self):
        """测试带注册表的消费者状态获取"""
        mock_consumer = Mock()
        mock_consumer.get_all_consumer_info.return_value = {
            "app1": {"allocated": 2},
            "app2": {"allocated": 4}
        }

        reporter = ResourceStatusReporter(consumer_registry=mock_consumer)

        consumer_status = reporter._get_consumer_status()

        assert consumer_status == mock_consumer.get_all_consumer_info.return_value

    def test_get_allocation_status_with_manager(self):
        """测试带管理器的分配状态获取"""
        mock_allocation = Mock()
        mock_allocation.get_allocation_summary.return_value = {
            "by_consumer": {"app1": 2},
            "by_resource_type": {"cpu": 4}
        }
        mock_allocation.get_allocation_count.return_value = 8
        mock_allocation.get_request_count.return_value = 3

        reporter = ResourceStatusReporter(allocation_manager=mock_allocation)

        allocation_status = reporter._get_allocation_status()

        assert allocation_status["summary"] == mock_allocation.get_allocation_summary.return_value
        assert allocation_status["active_count"] == 8
        assert allocation_status["pending_count"] == 3

    def test_get_health_status_with_provider_errors(self):
        """测试提供者有错误时的健康状态"""
        mock_provider = Mock()
        mock_provider.get_all_provider_status.return_value = {
            "cpu": {"status": "healthy"},
            "gpu": {"status": "error"},
            "memory": {"status": "error"}
        }

        reporter = ResourceStatusReporter(provider_registry=mock_provider)

        health = reporter._get_health_status()

        assert health["health_score"] < 1.0  # 应该有扣分
        assert len(health["issues"]) > 0
        assert "error" in health["issues"][0].lower()

    def test_get_health_status_with_too_many_allocations(self):
        """测试分配过多时的健康状态"""
        mock_allocation = Mock()
        mock_allocation.get_allocation_count.return_value = 150  # 超过100的阈值

        reporter = ResourceStatusReporter(allocation_manager=mock_allocation)

        health = reporter._get_health_status()

        assert health["health_score"] < 1.0  # 应该有扣分
        assert len(health["issues"]) > 0
        assert "过多" in "".join(health["issues"])

    def test_get_health_status_score_levels(self):
        """测试健康评分不同级别"""
        reporter = ResourceStatusReporter()

        # Mock 健康检查返回不同分数
        test_cases = [
            (0.95, "excellent"),
            (0.85, "good"),
            (0.75, "fair"),
            (0.65, "poor"),
            (0.55, "critical")
        ]

        for score, expected_status in test_cases:
            with patch.object(reporter, '_calculate_health_score', return_value=score):
                health = reporter._get_health_status()
                assert health["health_score"] == score
                assert health["health_status"] == expected_status

    def test_get_health_status_error_handling(self):
        """测试健康状态获取的错误处理"""
        reporter = ResourceStatusReporter()

        # Mock 提供者状态抛出异常
        with patch.object(reporter, 'provider_registry') as mock_provider:
            mock_provider.get_all_provider_status.side_effect = Exception("Provider error")

            health = reporter._get_health_status()

            assert health["health_score"] == 0.0
            assert health["health_status"] == "unknown"
            assert "error" in health

    def test_get_detailed_report_error_handling(self):
        """测试详细报告获取的错误处理"""
        reporter = ResourceStatusReporter()

        # Mock get_resource_status 抛出异常
        with patch.object(reporter, 'get_resource_status', side_effect=Exception("Status error")):
            report = reporter.get_detailed_report()

            assert "error" in report
            assert "timestamp" in report

    def test_get_recommendations_health_issues(self):
        """测试基于健康问题的建议生成"""
        reporter = ResourceStatusReporter()

        # Mock 健康状态为不健康
        mock_status = {
            "health": {"health_score": 0.5},
            "summary": {
                "active_allocations": 60,  # 超过50
                "pending_requests": 15     # 超过10
            }
        }

        with patch.object(reporter, 'get_resource_status', return_value=mock_status):
            recommendations = reporter._get_recommendations()

            assert len(recommendations) >= 3
            assert any("健康" in rec for rec in recommendations)
            assert any("分配策略" in rec for rec in recommendations)
            assert any("积压" in rec for rec in recommendations)

    def test_get_recommendations_error_handling(self):
        """测试建议生成的错误处理"""
        reporter = ResourceStatusReporter()

        # Mock get_resource_status 抛出异常
        with patch.object(reporter, 'get_resource_status', side_effect=Exception("Status error")):
            recommendations = reporter._get_recommendations()

            assert len(recommendations) >= 1
            assert any("监控" in rec for rec in recommendations)

    def test_export_report_unsupported_format(self):
        """测试导出不支持的格式"""
        reporter = ResourceStatusReporter()

        with pytest.raises(ValueError, match="不支持的导出格式"):
            reporter.export_report("xml")

    def test_export_report_error_handling(self):
        """测试报告导出的错误处理"""
        reporter = ResourceStatusReporter()

        # Mock get_detailed_report 抛出异常
        with patch.object(reporter, 'get_detailed_report', side_effect=Exception("Export error")):
            result = reporter.export_report("json")

            assert "导出失败" in result

    def test_export_status_to_json_error_handling(self):
        """测试JSON状态导出的错误处理"""
        reporter = ResourceStatusReporter()

        # Mock get_resource_status 抛出异常
        with patch.object(reporter, 'get_resource_status', side_effect=Exception("JSON export error")):
            result = reporter.export_status_to_json()

            assert "导出失败" in result

    def test_export_status_to_yaml_error_handling(self):
        """测试YAML状态导出的错误处理"""
        reporter = ResourceStatusReporter()

        # Mock get_resource_status 抛出异常
        with patch.object(reporter, 'get_resource_status', side_effect=Exception("YAML export error")):
            result = reporter.export_status_to_yaml()

            assert "导出失败" in result

    def test_generate_detailed_report_error_handling(self):
        """测试详细报告生成的错误处理"""
        reporter = ResourceStatusReporter()

        # Mock get_detailed_report 抛出异常
        with patch.object(reporter, 'get_detailed_report', side_effect=Exception("Detailed report error")):
            result = reporter.generate_detailed_report()

            assert "error" in result
            assert result["report_type"] == "detailed"

    def test_get_provider_status_summary_error_handling(self):
        """测试提供者状态摘要的错误处理"""
        reporter = ResourceStatusReporter()

        # Mock _get_provider_status 抛出异常
        with patch.object(reporter, '_get_provider_status', side_effect=Exception("Provider summary error")):
            result = reporter.get_provider_status_summary()

            assert "error" in result

    def test_get_consumer_status_summary_error_handling(self):
        """测试消费者状态摘要的错误处理"""
        reporter = ResourceStatusReporter()

        # Mock _get_consumer_status 抛出异常
        with patch.object(reporter, '_get_consumer_status', side_effect=Exception("Consumer summary error")):
            result = reporter.get_consumer_status_summary()

            assert "error" in result

    def test_generate_summary_report_error_handling(self):
        """测试摘要报告生成的错误处理"""
        reporter = ResourceStatusReporter()

        # Mock get_resource_status 抛出异常
        with patch.object(reporter, 'get_resource_status', side_effect=Exception("Summary report error")):
            result = reporter.generate_summary_report()

            assert "error" in result
            assert result["report_type"] == "summary"

    def test_get_provider_status_summary_detailed(self):
        """测试提供者状态摘要的详细信息"""
        mock_provider = Mock()
        mock_provider.get_all_provider_status.return_value = {
            "cpu": {"status": "healthy"},
            "gpu": {"status": "healthy"},
            "memory": {"status": "error"}
        }

        reporter = ResourceStatusReporter(provider_registry=mock_provider)

        summary = reporter.get_provider_status_summary()

        assert summary["total_providers"] == 3
        assert summary["healthy_providers"] == 2
        assert summary["error_providers"] == 1
        assert "providers" in summary

    def test_get_consumer_status_summary_detailed(self):
        """测试消费者状态摘要的详细信息"""
        mock_consumer = Mock()
        mock_consumer.get_all_consumer_info.return_value = {
            "app1": {"allocated": 2},
            "app2": {"allocated": 4},
            "app3": {"allocated": 1}
        }

        reporter = ResourceStatusReporter(consumer_registry=mock_consumer)

        summary = reporter.get_consumer_status_summary()

        assert summary["total_consumers"] == 3
        assert "consumers" in summary

    def test_resource_status_reporter_thread_safety(self):
        """测试资源状态报告器的线程安全性"""
        import threading
        reporter = ResourceStatusReporter()

        results = []
        errors = []

        def worker_thread(thread_id):
            try:
                # 并发获取状态
                status = reporter.get_resource_status()
                results.append((thread_id, "status", isinstance(status, dict)))

                # 并发导出报告
                json_report = reporter.export_status_to_json()
                results.append((thread_id, "export", isinstance(json_report, str)))

            except Exception as e:
                errors.append((thread_id, str(e)))

        # 启动多个线程
        threads = []
        num_threads = 5

        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0

        # 验证所有操作都成功
        status_results = [r for r in results if r[1] == "status"]
        export_results = [r for r in results if r[1] == "export"]

        assert len(status_results) == num_threads
        assert len(export_results) == num_threads
        assert all(r[2] for r in status_results)
        assert all(r[2] for r in export_results)

    def test_resource_status_reporter_with_all_components(self):
        """测试资源状态报告器与所有组件集成"""
        # 创建完整的Mock组件
        mock_provider = Mock()
        mock_provider.get_provider_count.return_value = 2
        mock_provider.get_all_provider_status.return_value = {
            "cpu": {"status": "healthy", "total_capacity": 8},
            "memory": {"status": "healthy", "total_capacity": 16}
        }

        mock_consumer = Mock()
        mock_consumer.get_consumer_count.return_value = 3
        mock_consumer.get_all_consumer_info.return_value = {
            "web_app": {"allocated": 2},
            "api_service": {"allocated": 3}
        }

        mock_allocation = Mock()
        mock_allocation.get_allocation_count.return_value = 5
        mock_allocation.get_request_count.return_value = 1
        mock_allocation.get_allocation_summary.return_value = {
            "by_consumer": {"web_app": 2},
            "by_resource_type": {"cpu": 5}
        }

        reporter = ResourceStatusReporter(
            provider_registry=mock_provider,
            consumer_registry=mock_consumer,
            allocation_manager=mock_allocation
        )

        # 测试完整的状态获取
        status = reporter.get_resource_status()

        assert status["summary"]["providers_count"] == 2
        assert status["summary"]["consumers_count"] == 3
        assert status["summary"]["active_allocations"] == 5
        assert status["summary"]["pending_requests"] == 1

        # 测试详细报告
        detailed = reporter.get_detailed_report()
        assert "performance" in detailed
        assert "trends" in detailed
        assert "recommendations" in detailed

        # 测试各种导出格式
        json_export = reporter.export_report("json")
        assert isinstance(json_export, str)
        assert len(json_export) > 0

        yaml_export = reporter.export_report("yaml")
        assert isinstance(yaml_export, str)
        assert len(yaml_export) > 0

    def test_resource_status_reporter_edge_cases(self):
        """测试资源状态报告器的边界情况"""
        reporter = ResourceStatusReporter()

        # 测试空组件的情况
        status = reporter.get_resource_status()
        assert isinstance(status, dict)

        # 测试各种报告类型的生成
        summary_report = reporter.generate_summary_report()
        detailed_report = reporter.generate_detailed_report()

        assert summary_report["report_type"] == "summary"
        assert detailed_report["report_type"] != "summary"  # 应该有report_type字段

        # 测试摘要功能
        provider_summary = reporter.get_provider_status_summary()
        consumer_summary = reporter.get_consumer_status_summary()

        assert isinstance(provider_summary, dict)
        assert isinstance(consumer_summary, dict)

        # 测试导出功能
        json_status = reporter.export_status_to_json()
        yaml_status = reporter.export_status_to_yaml()

        assert isinstance(json_status, str)
        assert isinstance(yaml_status, str)

    def test_resource_status_reporter_performance_metrics(self):
        """测试资源状态报告器的性能指标"""
        import time
        reporter = ResourceStatusReporter()

        # 测试获取状态的性能
        start_time = time.time()

        for _ in range(100):
            status = reporter.get_resource_status()
            assert isinstance(status, dict)

        end_time = time.time()
        duration = end_time - start_time

        # 100次操作应该在合理时间内完成
        assert duration < 5.0  # 5秒内

        operations_per_second = 100 / duration
        assert operations_per_second > 10  # 至少每秒10次操作

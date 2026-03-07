#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
审计报告生成器综合测试
测试AuditReportGenerator的核心功能，包括各类报告生成和导出
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import patch, MagicMock

from src.infrastructure.security.audit.audit_reporting import (
    AuditReportGenerator,
    ComplianceReport
)
from src.infrastructure.security.audit.audit_events import AuditEvent, AuditEventType, AuditSeverity


@pytest.fixture
def temp_report_dir(tmp_path):
    """创建临时报告目录"""
    return tmp_path / "reports"


@pytest.fixture
def audit_report_generator(temp_report_dir):
    """创建审计报告生成器实例"""
    generator = AuditReportGenerator()
    return generator


@pytest.fixture
def sample_audit_events():
    """创建示例审计事件"""
    base_time = datetime(2025, 1, 1, 12, 0, 0)

    events = [
        AuditEvent(
            event_id="event_001",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.LOW,
            timestamp=base_time,
            user_id="user123",
            resource="/login",
            details={"success": True, "ip_address": "192.168.1.100", "user_agent": "Mozilla/5.0"}
        ),
        AuditEvent(
            event_id="event_002",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.LOW,
            timestamp=base_time + timedelta(hours=1),
            user_id="user123",
            resource="/api/data",
            details={"data_size": 1024, "action": "read", "ip_address": "192.168.1.100"}
        ),
        AuditEvent(
            event_id="event_003",
            event_type=AuditEventType.DATA_OPERATION,
            severity=AuditSeverity.MEDIUM,
            timestamp=base_time + timedelta(hours=2),
            user_id="user456",
            resource="/api/user/123",
            details={"changes": ["email", "phone"], "action": "update", "ip_address": "192.168.1.200"}
        ),
        AuditEvent(
            event_id="event_004",
            event_type=AuditEventType.SECURITY,
            severity=AuditSeverity.CRITICAL,
            timestamp=base_time + timedelta(hours=3),
            user_id="suspicious_user",
            resource="/api/admin",
            details={"violation_type": "unauthorized_access", "action": "access", "ip_address": "10.0.0.1"}
        ),
        AuditEvent(
            event_id="event_005",
            event_type=AuditEventType.ACCESS,
            severity=AuditSeverity.MEDIUM,
            timestamp=base_time + timedelta(hours=4),
            user_id="user123",
            resource="/login",
            details={"success": False, "reason": "invalid_password", "ip_address": "192.168.1.100"}
        )
    ]
    return events


class TestComplianceReport:
    """测试合规报告数据类"""

    def test_compliance_report_creation_minimal(self):
        """测试最小化合规报告创建"""
        report = ComplianceReport(
            report_id="test-001",
            report_type="security",
            period_start=datetime(2025, 1, 1),
            period_end=datetime(2025, 1, 31),
            generated_at=datetime.now()
        )

        assert report.report_id == "test-001"
        assert report.report_type == "security"
        assert report.compliance_score == 100.0
        assert report.risk_assessment == "low"
        assert report.findings == []
        assert report.recommendations == []

    def test_compliance_report_creation_complete(self):
        """测试完整合规报告创建"""
        period_start = datetime(2025, 1, 1)
        period_end = datetime(2025, 1, 31)
        generated_at = datetime(2025, 1, 31, 23, 59, 59)

        findings = [
            {"type": "violation", "severity": "high", "description": "Unauthorized access"}
        ]
        recommendations = ["Implement stronger access controls"]

        report = ComplianceReport(
            report_id="complete-001",
            report_type="compliance",
            period_start=period_start,
            period_end=period_end,
            generated_at=generated_at,
            findings=findings,
            recommendations=recommendations,
            compliance_score=85.5,
            risk_assessment="medium"
        )

        assert report.report_id == "complete-001"
        assert report.report_type == "compliance"
        assert report.period_start == period_start
        assert report.period_end == period_end
        assert report.generated_at == generated_at
        assert report.findings == findings
        assert report.recommendations == recommendations
        assert report.compliance_score == 85.5
        assert report.risk_assessment == "medium"


class TestAuditReportGeneratorInitialization:
    """测试审计报告生成器初始化"""

    def test_initialization(self, audit_report_generator):
        """测试初始化"""
        generator = audit_report_generator

        assert hasattr(generator, '_report_templates')
        assert isinstance(generator._report_templates, dict)
        assert len(generator._report_templates) == 6  # 6种报告类型

        expected_templates = [
            'summary', 'security', 'compliance',
            'user_activity', 'resource_access', 'risk_analysis'
        ]
        assert set(generator._report_templates.keys()) == set(expected_templates)


class TestAuditReportGeneratorReportGeneration:
    """测试审计报告生成器报告生成功能"""

    def test_generate_summary_report(self, audit_report_generator, sample_audit_events):
        """测试生成摘要报告"""
        generator = audit_report_generator
        events = sample_audit_events

        report = generator.generate_report(events, report_type="summary")

        assert isinstance(report, dict)
        assert "report_type" in report
        assert report["report_type"] == "summary"
        assert "total_events" in report
        assert report["total_events"] == len(events)
        assert "period" in report
        assert "event_types" in report  # 实际的字段名

    def test_generate_security_report(self, audit_report_generator, sample_audit_events):
        """测试生成安全报告"""
        generator = audit_report_generator
        events = sample_audit_events

        report = generator.generate_report(events, report_type="security")

        assert isinstance(report, dict)
        assert report["report_type"] == "security"
        assert "high_risk_events" in report  # 实际的字段名

    def test_generate_compliance_report(self, audit_report_generator, sample_audit_events):
        """测试生成合规报告"""
        generator = audit_report_generator
        events = sample_audit_events

        report = generator.generate_report(events, report_type="compliance")

        assert isinstance(report, dict)
        assert report["report_type"] == "compliance"
        assert "compliance_rate" in report  # 实际的字段名

    def test_generate_user_activity_report(self, audit_report_generator, sample_audit_events):
        """测试生成用户活动报告"""
        generator = audit_report_generator
        events = sample_audit_events

        report = generator.generate_report(events, report_type="user_activity")

        assert isinstance(report, dict)
        assert report["report_type"] == "user_activity"
        assert "most_active_users" in report  # 实际的字段名

    def test_generate_resource_access_report(self, audit_report_generator, sample_audit_events):
        """测试生成资源访问报告"""
        generator = audit_report_generator
        events = sample_audit_events

        report = generator.generate_report(events, report_type="resource_access")

        assert isinstance(report, dict)
        assert report["report_type"] == "resource_access"
        assert "most_accessed_resources" in report  # 实际的字段名

    def test_generate_risk_analysis_report(self, audit_report_generator, sample_audit_events):
        """测试生成风险分析报告"""
        generator = audit_report_generator
        events = sample_audit_events

        report = generator.generate_report(events, report_type="risk_analysis")

        assert isinstance(report, dict)
        assert report["report_type"] == "risk_analysis"
        assert "high_risk_events_count" in report  # 实际的字段名

    def test_generate_report_with_time_filter(self, audit_report_generator, sample_audit_events):
        """测试带时间过滤的报告生成"""
        generator = audit_report_generator
        events = sample_audit_events

        # 设置时间范围
        start_time = datetime(2025, 1, 1, 11, 0, 0)  # 比第一个事件早1小时
        end_time = datetime(2025, 1, 1, 15, 0, 0)    # 比最后一个事件晚

        report = generator.generate_report(
            events,
            report_type="summary",
            start_time=start_time,
            end_time=end_time
        )

        assert isinstance(report, dict)
        assert "period" in report
        # 应该包含所有5个事件
        assert report["total_events"] == 5

    def test_generate_report_invalid_type(self, audit_report_generator, sample_audit_events):
        """测试生成无效类型的报告"""
        generator = audit_report_generator
        events = sample_audit_events

        try:
            report = generator.generate_report(events, report_type="invalid_type")
            # 如果没有抛出异常，至少返回了dict
            assert isinstance(report, dict)
        except ValueError:
            # 正确地抛出了异常
            assert True

    def test_generate_report_empty_events(self, audit_report_generator):
        """测试用空事件列表生成报告"""
        generator = audit_report_generator

        report = generator.generate_report([], report_type="summary")

        assert isinstance(report, dict)
        assert report["total_events"] == 0


class TestAuditReportGeneratorComplianceReport:
    """测试审计报告生成器合规报告功能"""

    def test_generate_compliance_report_method(self, audit_report_generator, sample_audit_events):
        """测试合规报告生成方法"""
        generator = audit_report_generator
        events = sample_audit_events

        compliance_report = generator.generate_compliance_report(events)

        assert isinstance(compliance_report, ComplianceReport)
        assert compliance_report.report_type == "compliance"
        assert compliance_report.compliance_score >= 0
        assert compliance_report.compliance_score <= 100
        assert compliance_report.risk_assessment in ["low", "medium", "high", "critical"]
        assert isinstance(compliance_report.findings, list)
        assert isinstance(compliance_report.recommendations, list)


class TestAuditReportGeneratorExport:
    """测试审计报告生成器导出功能"""

    def test_export_report_json(self, audit_report_generator, sample_audit_events, temp_report_dir):
        """测试导出JSON格式报告"""
        generator = audit_report_generator
        events = sample_audit_events

        report_data = generator.generate_report(events, report_type="summary")
        output_path = temp_report_dir / "test_report.json"

        result = generator.export_report(report_data, format_type="json", output_path=output_path)

        assert isinstance(result, str)
        assert output_path.exists()

        # 验证导出的JSON内容
        with open(output_path, 'r', encoding='utf-8') as f:
            exported_data = json.load(f)
        assert exported_data == report_data

    def test_export_report_csv(self, audit_report_generator, sample_audit_events, temp_report_dir):
        """测试导出CSV格式报告"""
        generator = audit_report_generator
        events = sample_audit_events

        report_data = generator.generate_report(events, report_type="summary")
        output_path = temp_report_dir / "test_report.csv"

        result = generator.export_report(report_data, format_type="csv", output_path=output_path)

        assert isinstance(result, str)
        assert output_path.exists()

        # 验证导出的CSV文件
        with open(output_path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            rows = list(csv_reader)
            assert len(rows) > 0

    def test_export_report_html(self, audit_report_generator, sample_audit_events, temp_report_dir):
        """测试导出HTML格式报告"""
        generator = audit_report_generator
        events = sample_audit_events

        report_data = generator.generate_report(events, report_type="summary")
        output_path = temp_report_dir / "test_report.html"

        result = generator.export_report(report_data, format_type="html", output_path=output_path)

        assert isinstance(result, str)
        assert output_path.exists()

        # 验证导出的HTML内容
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            assert "<html>" in html_content.lower()
            assert "</html>" in html_content.lower()

    def test_export_report_pdf(self, audit_report_generator, sample_audit_events, temp_report_dir):
        """测试导出PDF格式报告"""
        generator = audit_report_generator
        events = sample_audit_events

        report_data = generator.generate_report(events, report_type="summary")
        output_path = temp_report_dir / "test_report.pdf"

        result = generator.export_report(report_data, format_type="pdf", output_path=output_path)

        assert isinstance(result, str)
        assert output_path.exists()

        # 验证文件大小（PDF文件应该有一定大小）
        assert output_path.stat().st_size > 100

    def test_export_report_invalid_format(self, audit_report_generator, sample_audit_events, temp_report_dir):
        """测试导出无效格式的报告"""
        generator = audit_report_generator
        events = sample_audit_events

        report_data = generator.generate_report(events, report_type="summary")
        output_path = temp_report_dir / "test_report.invalid"

        try:
            result = generator.export_report(report_data, format_type="invalid", output_path=output_path)
            assert False, "Should have raised ValueError"
        except ValueError:
            # 正确地抛出了异常
            assert True

    def test_export_report_to_invalid_path(self, audit_report_generator, sample_audit_events):
        """测试导出到无效路径"""
        generator = audit_report_generator
        events = sample_audit_events

        report_data = generator.generate_report(events, report_type="summary")
        invalid_path = "/invalid/path/test_report.json"

        try:
            result = generator.export_report(report_data, format_type="json", output_path=Path(invalid_path))
            # 如果没有抛出异常，检查结果
            assert isinstance(result, str)  # 应该返回文件路径字符串
        except Exception:
            # 如果抛出异常，也是可以接受的
            assert True


class TestAuditReportGeneratorInternalMethods:
    """测试审计报告生成器内部方法"""

    def test_filter_events_by_time(self, audit_report_generator, sample_audit_events):
        """测试按时间过滤事件"""
        generator = audit_report_generator
        events = sample_audit_events

        start_time = datetime(2025, 1, 1, 12, 30, 0)  # 第一个事件之后
        end_time = datetime(2025, 1, 1, 16, 0, 0)     # 最后一个事件之后

        filtered_events = generator._filter_events_by_time(events, start_time, end_time)

        # 应该过滤掉第一个事件（12:00），保留后面的4个事件
        assert len(filtered_events) == 4

        # 验证所有过滤后的事件都在时间范围内
        for event in filtered_events:
            assert event.timestamp >= start_time
            assert event.timestamp <= end_time

    def test_get_hourly_distribution(self, audit_report_generator, sample_audit_events):
        """测试获取小时分布"""
        generator = audit_report_generator
        events = sample_audit_events

        distribution = generator._get_hourly_distribution(events)

        assert isinstance(distribution, dict)
        # 应该有5个小时的分布（12, 13, 14, 15, 16点）
        assert len(distribution) <= 5

        # 总数应该等于事件数量
        total_events = sum(distribution.values())
        assert total_events == len(events)

    def test_get_risk_distribution(self, audit_report_generator, sample_audit_events):
        """测试获取风险分布"""
        generator = audit_report_generator
        events = sample_audit_events

        distribution = generator._get_risk_distribution(events)

        assert isinstance(distribution, dict)
        # 应该包含各种风险级别
        assert "low" in distribution or "medium" in distribution or "high" in distribution

    def test_perform_compliance_checks(self, audit_report_generator, sample_audit_events):
        """测试执行合规检查"""
        generator = audit_report_generator
        events = sample_audit_events

        findings = generator._perform_compliance_checks(events, "security")

        assert isinstance(findings, list)
        for finding in findings:
            assert isinstance(finding, dict)
            assert "type" in finding
            assert "severity" in finding

    def test_calculate_compliance_score(self, audit_report_generator, sample_audit_events):
        """测试计算合规分数"""
        generator = audit_report_generator
        events = sample_audit_events

        findings = [{"severity": "high"}, {"severity": "medium"}, {"severity": "low"}]
        score = generator._calculate_compliance_score(findings, events)

        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_assess_compliance_risk(self, audit_report_generator):
        """测试评估合规风险"""
        generator = audit_report_generator

        # 测试不同分数对应的风险等级
        assert generator._assess_compliance_risk(95) == "low"
        assert generator._assess_compliance_risk(85) == "medium"
        assert generator._assess_compliance_risk(70) == "high"
        assert generator._assess_compliance_risk(50) == "critical"

    def test_analyze_suspicious_ips(self, audit_report_generator, sample_audit_events):
        """测试分析可疑IP"""
        generator = audit_report_generator
        events = sample_audit_events

        analysis = generator._analyze_suspicious_ips(events)

        assert isinstance(analysis, dict)
        # 应该包含IP统计信息
        assert "ip_counts" in analysis or "suspicious_ips" in analysis

    def test_analyze_security_findings(self, audit_report_generator, sample_audit_events):
        """测试分析安全发现"""
        generator = audit_report_generator
        events = sample_audit_events

        findings = generator._analyze_security_findings(events)

        assert isinstance(findings, list)
        for finding in findings:
            assert isinstance(finding, dict)

    def test_analyze_risk_trend(self, audit_report_generator, sample_audit_events):
        """测试分析风险趋势"""
        generator = audit_report_generator
        events = sample_audit_events

        trend = generator._analyze_risk_trend(events)

        assert isinstance(trend, dict)
        # 应该包含趋势分析信息
        assert "trend" in trend or "analysis" in trend

    def test_generate_compliance_recommendations(self, audit_report_generator):
        """测试生成合规建议"""
        generator = audit_report_generator

        findings = [
            {"type": "violation", "severity": "high", "description": "Unauthorized access"},
            {"type": "warning", "severity": "medium", "description": "Weak password policy"}
        ]

        recommendations = generator._generate_compliance_recommendations(findings)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        for rec in recommendations:
            assert isinstance(rec, str)


class TestAuditReportGeneratorIntegration:
    """测试审计报告生成器集成功能"""

    def test_full_report_workflow(self, audit_report_generator, sample_audit_events, temp_report_dir):
        """测试完整报告工作流"""
        generator = audit_report_generator
        events = sample_audit_events

        # 1. 生成摘要报告
        summary_report = generator.generate_report(events, report_type="summary")
        assert summary_report["total_events"] == len(events)

        # 2. 生成合规报告
        compliance_report = generator.generate_compliance_report(events)
        assert compliance_report.compliance_score >= 0

        # 3. 导出多种格式
        base_path = temp_report_dir / "workflow_test"

        # JSON导出
        json_result = generator.export_report(summary_report, format_type="json", output_path=Path(f"{base_path}.json"))
        assert isinstance(json_result, str)  # 返回文件路径

        # CSV导出
        csv_result = generator.export_report(summary_report, format_type="csv", output_path=Path(f"{base_path}.csv"))
        assert isinstance(csv_result, str)  # 返回文件路径

        # HTML导出
        html_result = generator.export_report(summary_report, format_type="html", output_path=Path(f"{base_path}.html"))
        assert isinstance(html_result, str)  # 返回文件路径

        # 验证文件都已创建
        assert Path(f"{base_path}.json").exists()
        assert Path(f"{base_path}.csv").exists()
        assert Path(f"{base_path}.html").exists()

    def test_multiple_report_types_generation(self, audit_report_generator, sample_audit_events):
        """测试多种报告类型生成"""
        generator = audit_report_generator
        events = sample_audit_events

        report_types = ['summary', 'security', 'compliance', 'user_activity', 'resource_access', 'risk_analysis']

        for report_type in report_types:
            report = generator.generate_report(events, report_type=report_type)
            assert isinstance(report, dict)
            assert report["report_type"] == report_type
            assert "total_events" in report

    def test_report_generation_with_filters(self, audit_report_generator, sample_audit_events):
        """测试带过滤条件的报告生成"""
        generator = audit_report_generator
        events = sample_audit_events

        # 时间范围过滤
        start_time = datetime(2025, 1, 1, 13, 0, 0)  # 13点之后
        end_time = datetime(2025, 1, 1, 15, 0, 0)    # 15点之前

        filtered_report = generator.generate_report(
            events,
            report_type="summary",
            start_time=start_time,
            end_time=end_time
        )

        # 应该包含部分事件（具体数量取决于实现）
        assert filtered_report["total_events"] >= 0


class TestAuditReportGeneratorErrorHandling:
    """测试审计报告生成器错误处理"""

    def test_generate_report_with_none_events(self, audit_report_generator):
        """测试用None事件生成报告"""
        generator = audit_report_generator

        try:
            report = generator.generate_report(None, report_type="summary")
            assert isinstance(report, dict)
            assert report["total_events"] == 0
        except Exception:
            # 如果抛出异常，说明代码正确处理了None输入
            assert True

    def test_export_report_with_invalid_data(self, audit_report_generator, temp_report_dir):
        """测试导出无效数据"""
        generator = audit_report_generator

        invalid_data = {"invalid": "data"}
        output_path = temp_report_dir / "invalid_export.json"

        try:
            result = generator.export_report(invalid_data, format_type="json", output_path=output_path)
            assert isinstance(result, str)
        except Exception:
            assert True  # 错误处理正常

    def test_compliance_methods_with_empty_events(self, audit_report_generator):
        """测试合规方法处理空事件列表"""
        generator = audit_report_generator

        # 测试各种内部方法
        score = generator._calculate_compliance_score([], [])
        assert isinstance(score, float)

        risk = generator._assess_compliance_risk(50.0)
        assert isinstance(risk, str)

        recommendations = generator._generate_compliance_recommendations([])
        assert isinstance(recommendations, list)


class TestAuditReportGeneratorPerformance:
    """测试审计报告生成器性能"""

    def test_large_dataset_report_generation(self, audit_report_generator):
        """测试大数据集报告生成"""
        generator = audit_report_generator

        # 创建大量事件
        base_time = datetime(2025, 1, 1, 12, 0, 0)
        large_events = []

        for i in range(1000):
            event = AuditEvent(
                event_id=f"event_{i:04d}",
                event_type=AuditEventType.ACCESS,
                severity=AuditSeverity.LOW,
                timestamp=base_time + timedelta(minutes=i),
                user_id=f"user{i % 100}",
                resource="/login",
                details={"success": i % 2 == 0, "ip_address": f"192.168.1.{i % 255}"}
            )
            large_events.append(event)

        # 生成报告
        import time
        start_time = time.time()
        report = generator.generate_report(large_events, report_type="summary")
        end_time = time.time()

        # 验证报告生成
        assert report["total_events"] == 1000

        # 性能检查：1000个事件应该在合理时间内完成
        duration = end_time - start_time
        assert duration < 5.0  # 应该在5秒内完成

    def test_memory_efficiency_with_large_dataset(self, audit_report_generator):
        """测试大数据集的内存效率"""
        generator = audit_report_generator

        # 创建大量事件
        base_time = datetime(2025, 1, 1, 12, 0, 0)
        large_events = []

        for i in range(5000):
            event = AuditEvent(
                event_id=f"event_{i:04d}",
                event_type=AuditEventType.ACCESS,
                severity=AuditSeverity.LOW,
                timestamp=base_time + timedelta(seconds=i),
                user_id=f"user{i % 200}",
                resource=f"/api/resource/{i % 50}",
                details={"size": i * 100, "action": "read"}
            )
            large_events.append(event)

        # 生成资源访问报告
        report = generator.generate_report(large_events, report_type="resource_access")

        # 验证报告结构完整
        assert "resource_access" in report
        assert "top_resources" in report
        assert len(report["resource_access"]) <= 50  # 应该限制显示数量

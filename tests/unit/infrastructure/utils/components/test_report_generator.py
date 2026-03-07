#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层报告生成器组件测试

测试目标：提升utils/components/report_generator.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.report_generator模块
"""

import pytest
from unittest.mock import MagicMock, Mock


class TestReportGeneratorConstants:
    """测试报告生成器常量"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.components.report_generator import ReportGeneratorConstants
        
        assert ReportGeneratorConstants.DEFAULT_EXECUTED_ORDERS == 0
        assert ReportGeneratorConstants.DEFAULT_CANCELLED_ORDERS == 0
        assert ReportGeneratorConstants.DEFAULT_RISK_CHECKS == 0
        assert ReportGeneratorConstants.DEFAULT_VIOLATIONS == 0
        assert ReportGeneratorConstants.DEFAULT_AUTO_REJECTS == 0
        assert ReportGeneratorConstants.DAYS_IN_WEEK == 6
        assert ReportGeneratorConstants.REPORT_TYPE_DAILY == "daily"
        assert ReportGeneratorConstants.REPORT_TYPE_WEEKLY == "weekly"
        assert ReportGeneratorConstants.REPORT_TYPE_MONTHLY == "monthly"
        assert ReportGeneratorConstants.DEFAULT_WEEKLY_VOLUME == 0
        assert ReportGeneratorConstants.DEFAULT_WEEKLY_TRADES == 0
        assert ReportGeneratorConstants.DEFAULT_MONTHLY_VOLUME == 0
        assert ReportGeneratorConstants.DEFAULT_MONTHLY_TRADES == 0
        assert ReportGeneratorConstants.REPORT_FILE_EXTENSION == ".json"
        assert ReportGeneratorConstants.TIMESTAMP_FORMAT == "%Y%m%d_%H%M%S"
        assert ReportGeneratorConstants.DEFAULT_REPORT_PATH == "/tmp"


class TestComplianceReportGenerator:
    """测试合规报告生成器"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator
        
        generator = ComplianceReportGenerator()
        assert generator is not None
        assert generator.config is None
        assert generator.data_adapter is None
        assert generator.order_manager is None
        assert generator.risk_controller is None
        assert hasattr(generator, 'report_templates')
    
    def test_init_with_config(self):
        """测试使用配置初始化"""
        from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator
        
        config = {"key": "value"}
        generator = ComplianceReportGenerator(config)
        assert generator.config == config
    
    def test_call_or_default_none_component(self):
        """测试安全调用方法（组件为None）"""
        from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator
        
        generator = ComplianceReportGenerator()
        result = generator._call_or_default(None, "method", default="default_value")
        assert result == "default_value"
    
    def test_call_or_default_method_not_exists(self):
        """测试安全调用方法（方法不存在）"""
        from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator
        
        generator = ComplianceReportGenerator()
        component = MagicMock()
        del component.non_existent_method
        
        result = generator._call_or_default(component, "non_existent_method", default="default_value")
        assert result == "default_value"
    
    def test_call_or_default_method_exists(self):
        """测试安全调用方法（方法存在）"""
        from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator
        
        generator = ComplianceReportGenerator()
        component = MagicMock()
        component.test_method.return_value = "test_result"
        
        result = generator._call_or_default(component, "test_method", default="default_value")
        assert result == "test_result"
    
    def test_get_all_templates(self):
        """测试获取所有模板"""
        from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator
        
        generator = ComplianceReportGenerator()
        templates = generator._get_all_templates()
        
        assert isinstance(templates, dict)
        assert "daily" in templates
        assert "weekly" in templates
        assert "monthly" in templates
        assert "exception" in templates
    
    def test_get_daily_template(self):
        """测试获取每日报告模板"""
        from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator
        
        generator = ComplianceReportGenerator()
        template = generator._get_daily_template()
        
        assert isinstance(template, dict)
        assert "title" in template
        assert "sections" in template
    
    def test_get_weekly_template(self):
        """测试获取每周报告模板"""
        from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator
        
        generator = ComplianceReportGenerator()
        template = generator._get_weekly_template()
        
        assert isinstance(template, dict)
        assert "title" in template
        assert "sections" in template
    
    def test_get_monthly_template(self):
        """测试获取每月报告模板"""
        from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator
        
        generator = ComplianceReportGenerator()
        template = generator._get_monthly_template()
        
        assert isinstance(template, dict)
        assert "title" in template
        assert "sections" in template
    
    def test_get_exception_template(self):
        """测试获取异常报告模板"""
        from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator
        
        generator = ComplianceReportGenerator()
        template = generator._get_exception_template()
        
        assert isinstance(template, dict)
        assert "title" in template
        assert "sections" in template
    
    def test_get_template_by_type(self):
        """测试按类型获取模板"""
        from src.infrastructure.utils.components.report_generator import ComplianceReportGenerator
        
        generator = ComplianceReportGenerator()
        templates = generator._get_all_templates()
        
        daily_template = generator._get_template_by_type(templates, "daily")
        assert daily_template == templates["daily"]
        
        weekly_template = generator._get_template_by_type(templates, "weekly")
        assert weekly_template == templates["weekly"]


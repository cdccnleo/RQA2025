#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层代码质量工具组件测试

测试目标：提升utils/patterns/code_quality.py的真实覆盖率
实际导入和使用src.infrastructure.utils.patterns.code_quality模块
"""

import pytest
import tempfile
import os
from pathlib import Path


class TestInfrastructureCodeFormatter:
    """测试基础设施代码格式化工具类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.patterns.code_quality import InfrastructureCodeFormatter
        
        formatter = InfrastructureCodeFormatter()
        assert isinstance(formatter._config, dict)
    
    def test_format_imports(self):
        """测试格式化导入语句"""
        from src.infrastructure.utils.patterns.code_quality import InfrastructureCodeFormatter
        
        code = "import os\nimport sys"
        result = InfrastructureCodeFormatter.format_imports(code)
        assert result == code
    
    def test_fix_line_length(self):
        """测试修复行长度"""
        from src.infrastructure.utils.patterns.code_quality import InfrastructureCodeFormatter
        
        code = "test code"
        result = InfrastructureCodeFormatter.fix_line_length(code, max_length=100)
        assert result == code
    
    def test_standardize_docstrings(self):
        """测试标准化文档字符串"""
        from src.infrastructure.utils.patterns.code_quality import InfrastructureCodeFormatter
        
        code = "def test(): pass"
        result = InfrastructureCodeFormatter.standardize_docstrings(code)
        assert result == code
    
    def test_apply_all_formatting(self):
        """测试应用所有格式化"""
        from src.infrastructure.utils.patterns.code_quality import InfrastructureCodeFormatter
        
        code = "import os"
        result = InfrastructureCodeFormatter.apply_all_formatting(code)
        assert isinstance(result, str)
    
    def test_format_code(self):
        """测试格式化代码"""
        from src.infrastructure.utils.patterns.code_quality import InfrastructureCodeFormatter
        
        formatter = InfrastructureCodeFormatter()
        code = "test code"
        result = formatter.format_code(code)
        assert result == code
    
    def test_check_style(self):
        """测试检查代码风格"""
        from src.infrastructure.utils.patterns.code_quality import InfrastructureCodeFormatter
        
        formatter = InfrastructureCodeFormatter()
        code = "test code"
        result = formatter.check_style(code)
        
        assert isinstance(result, dict)
        assert "compliant" in result
        assert "issues" in result


class TestInfrastructureQualityMonitor:
    """测试基础设施质量监控器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.patterns.code_quality import InfrastructureQualityMonitor
        
        monitor = InfrastructureQualityMonitor()
        assert isinstance(monitor._metrics, dict)
        assert isinstance(monitor.metrics_history, list)
        assert isinstance(monitor.alerts, list)
    
    def test_collect_quality_metrics_code_string(self):
        """测试收集代码字符串的质量指标"""
        from src.infrastructure.utils.patterns.code_quality import InfrastructureQualityMonitor
        
        monitor = InfrastructureQualityMonitor()
        code = "def test():\n    pass\n"
        
        result = monitor.collect_quality_metrics(code)
        
        assert isinstance(result, dict)
        assert "quality_score" in result
        assert "files_analyzed" in result
        assert result["files_analyzed"] == 1
    
    def test_collect_quality_metrics_file_path(self):
        """测试收集文件路径的质量指标"""
        from src.infrastructure.utils.patterns.code_quality import InfrastructureQualityMonitor
        
        monitor = InfrastructureQualityMonitor()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test():\n    pass\n")
            temp_path = f.name
        
        try:
            result = monitor.collect_quality_metrics(temp_path)
            
            assert isinstance(result, dict)
            assert "files_analyzed" in result
            assert result["files_analyzed"] == 1
        finally:
            os.unlink(temp_path)
    
    def test_collect_quality_metrics_directory(self):
        """测试收集目录的质量指标"""
        from src.infrastructure.utils.patterns.code_quality import InfrastructureQualityMonitor
        
        monitor = InfrastructureQualityMonitor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试文件
            test_file = Path(temp_dir) / "test.py"
            test_file.write_text("def test():\n    pass\n")
            
            result = monitor.collect_quality_metrics(temp_dir)
            
            assert isinstance(result, dict)
            assert "files_analyzed" in result
            assert result["files_analyzed"] >= 1

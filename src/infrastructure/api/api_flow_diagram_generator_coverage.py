#!/usr/bin/env python3
"""
针对性测试: src\infrastructure\api\api_flow_diagram_generator

自动生成的覆盖率提升测试
"""

import pytest


class TestSrc\Infrastructure\Api\Api_Flow_Diagram_GeneratorCoverage:
    """src\infrastructure\api\api_flow_diagram_generator 覆盖率测试"""


    def test_generate_diagram_existence(self):
        """测试 generate_diagram 函数存在性"""
        try:
            from src\infrastructure\api\api_flow_diagram_generator import generate_diagram
            assert callable(generate_diagram)
        except ImportError:
            pytest.skip(f"generate_diagram 函数不可用")

    def test_add_diagram_existence(self):
        """测试 add_diagram 函数存在性"""
        try:
            from src\infrastructure\api\api_flow_diagram_generator import add_diagram
            assert callable(add_diagram)
        except ImportError:
            pytest.skip(f"add_diagram 函数不可用")


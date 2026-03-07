"""
ML接口定义单元测试
"""
import pytest
from unittest.mock import Mock

from src.ml.interfaces.interfaces import IMlComponent


class TestIMlComponent:
    """测试IMlComponent接口"""

    def test_iml_component_is_abstract(self):
        """测试IMlComponent是抽象类"""
        with pytest.raises(TypeError):
            IMlComponent()

    def test_iml_component_implementation(self):
        """测试IMlComponent具体实现"""
        class ConcreteComponent(IMlComponent):
            def get_status(self):
                return {"status": "running", "health": "good"}

            def health_check(self):
                return {"healthy": True, "message": "OK"}

        component = ConcreteComponent()
        status = component.get_status()
        assert status["status"] == "running"
        assert status["health"] == "good"

        health = component.health_check()
        assert health["healthy"] is True
        assert health["message"] == "OK"


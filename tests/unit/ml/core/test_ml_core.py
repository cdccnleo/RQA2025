"""
机器学习核心模块单元测试
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from src.ml.core.ml_core import MLCore


class TestMLCore:
    """测试机器学习核心"""

    def setup_method(self):
        """测试前准备"""
        self.ml_core = MLCore()

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_ml_core_initialization(self):
        """测试ML核心初始化"""
        assert self.ml_core is not None
        assert hasattr(self.ml_core, 'models')
        assert hasattr(self.ml_core, 'feature_processors')
        assert hasattr(self.ml_core, 'logger')

    def test_apply_default_services(self):
        """测试应用默认服务"""
        # 强制使用降级服务
        with patch.dict('os.environ', {'ML_CORE_FORCE_FALLBACK': '1'}):
            ml_core = MLCore()
            ml_core._apply_default_services()

            assert ml_core.logger is not None
            assert ml_core.cache_manager is None
            assert ml_core.config_manager is None

    def test_ml_core_repr(self):
        """测试MLCore字符串表示"""
        repr_str = repr(self.ml_core)
        assert "MLCore" in repr_str


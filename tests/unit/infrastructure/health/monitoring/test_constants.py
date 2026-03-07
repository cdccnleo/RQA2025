#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""健康监控常量测试"""

import pytest
from unittest.mock import Mock, patch
from src.infrastructure.health.monitoring import constants


class TestConstants:
    """测试常量定义"""

    def test_constants_exist(self):
        """测试常量存在"""
        assert hasattr(constants, 'SECONDS_TO_HOURS')
        assert hasattr(constants, 'QUALITY_SCORE_PENALTY_PER_ERROR')
        assert hasattr(constants, 'GPU_PERCENTAGE_MULTIPLIER')
        assert hasattr(constants, 'DEFAULT_HISTORY_SIZE')
        assert hasattr(constants, 'MAX_HISTORY_SIZE')
        assert hasattr(constants, 'MIN_HISTORY_SIZE')

    def test_constant_values(self):
        """测试常量值"""
        assert constants.SECONDS_TO_HOURS == 3600
        assert constants.QUALITY_SCORE_PENALTY_PER_ERROR == 0.1
        assert constants.GPU_PERCENTAGE_MULTIPLIER == 100.0
        assert constants.DEFAULT_HISTORY_SIZE == 1000
        assert constants.MAX_HISTORY_SIZE == 10000
        assert constants.MIN_HISTORY_SIZE == 100

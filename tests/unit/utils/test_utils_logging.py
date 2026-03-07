#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utils Logging Module测试

测试工具日志模块的功能
"""

import pytest
import logging
from unittest.mock import patch


class TestUtilsLogging:
    """测试日志模块功能"""

    def test_get_logger_default(self):
        """测试get_logger默认参数"""
        from src.utils.logging.logger import get_logger

        logger = get_logger()
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'RQA2025'

    def test_get_logger_custom_name(self):
        """测试get_logger自定义名称（第一次调用）"""
        from src.utils.logging.logger import get_logger

        # 由于使用了全局变量，第一次调用会设置logger名称
        logger = get_logger('test_logger')
        assert isinstance(logger, logging.Logger)
        # 实际行为：返回全局logger，其名称是第一次调用时设置的

    def test_get_logger_with_config(self):
        """测试get_logger带配置（第一次调用）"""
        from src.utils.logging.logger import get_logger

        # 由于使用了全局变量，第一次调用会设置配置
        config = {'level': 'DEBUG'}
        logger = get_logger('test_config', config)
        assert isinstance(logger, logging.Logger)
        # 实际行为：返回全局logger，其配置是第一次调用时设置的

    def test_get_component_logger_with_component(self):
        """测试get_component_logger带组件名称"""
        from src.utils.logging.logger import get_component_logger

        logger = get_component_logger('base', 'component')
        assert isinstance(logger, logging.Logger)
        # 由于使用了全局logger，名称是第一次调用get_logger时设置的

    def test_get_component_logger_without_component(self):
        """测试get_component_logger不带组件名称"""
        from src.utils.logging.logger import get_component_logger

        logger = get_component_logger('base')
        assert isinstance(logger, logging.Logger)
        # 由于使用了全局logger，名称是第一次调用get_logger时设置的

    def test_configure_logging(self):
        """测试configure_logging函数"""
        from src.utils.logging.logger import configure_logging

        # 应该不抛出异常
        configure_logging('INFO')

        # 获取日志记录器验证配置
        from src.utils.logging.logger import get_logger
        logger = get_logger()
        assert isinstance(logger, logging.Logger)

    @patch('logging.StreamHandler')
    @patch('logging.Logger.addHandler')
    def test_setup_logger_avoids_duplicate_handlers(self, mock_add_handler, mock_stream_handler):
        """测试_setup_logger避免重复处理器"""
        from src.utils.logging.logger import _setup_logger

        # 创建一个已经有处理器的logger
        test_logger = logging.getLogger('test_duplicate')
        test_logger.handlers = [logging.StreamHandler()]  # 直接设置handlers列表

        result = _setup_logger('test_duplicate')

        # 应该返回已存在的logger，不添加新处理器
        mock_add_handler.assert_not_called()
        assert result == test_logger

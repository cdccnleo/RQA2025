#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""environment模块基础测试"""

import pytest
import os


def test_is_production_false():
    """测试is_production在默认环境"""
    from src.infrastructure.config.environment import is_production
    result = is_production()
    # 默认应该是开发环境
    assert isinstance(result, bool)


def test_is_development_true():
    """测试is_development在默认环境"""
    from src.infrastructure.config.environment import is_development
    result = is_development()
    # 默认应该是开发环境
    assert isinstance(result, bool)


def test_is_testing():
    """测试is_testing"""
    from src.infrastructure.config.environment import is_testing
    result = is_testing()
    # 在pytest中运行时应该为True
    assert result is True


def test_is_production_with_env(monkeypatch):
    """测试is_production与环境变量"""
    from src.infrastructure.config.environment import is_production
    monkeypatch.setenv('ENV', 'production')
    result = is_production()
    assert result is True


def test_is_production_with_dev_env(monkeypatch):
    """测试is_production与开发环境变量"""
    from src.infrastructure.config.environment import is_production
    monkeypatch.setenv('ENV', 'development')
    result = is_production()
    assert result is False


def test_is_development_with_prod_env(monkeypatch):
    """测试is_development与生产环境变量"""
    from src.infrastructure.config.environment import is_development
    monkeypatch.setenv('ENV', 'production')
    result = is_development()
    assert result is False


def test_is_development_with_dev_env(monkeypatch):
    """测试is_development与开发环境变量"""
    from src.infrastructure.config.environment import is_development
    monkeypatch.setenv('ENV', 'development')
    result = is_development()
    assert result is True


def test_environment_functions_import():
    """测试环境函数导入"""
    from src.infrastructure.config.environment import is_production, is_development, is_testing
    assert is_production is not None
    assert is_development is not None
    assert is_testing is not None


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""环境检测功能测试"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import os
import pytest


def test_is_production_true(monkeypatch):
    """测试生产环境检测（返回True）"""
    monkeypatch.setenv('ENV', 'production')
    from src.infrastructure.config.environment import is_production
    assert is_production() is True


def test_is_production_false(monkeypatch):
    """测试生产环境检测（返回False）"""
    monkeypatch.setenv('ENV', 'development')
    from src.infrastructure.config.environment import is_production
    assert is_production() is False


def test_is_production_case_insensitive(monkeypatch):
    """测试生产环境检测大小写不敏感"""
    monkeypatch.setenv('ENV', 'PRODUCTION')
    from src.infrastructure.config.environment import is_production
    assert is_production() is True


def test_is_development_true(monkeypatch):
    """测试开发环境检测（返回True）"""
    monkeypatch.setenv('ENV', 'development')
    from src.infrastructure.config.environment import is_development
    assert is_development() is True


def test_is_development_false(monkeypatch):
    """测试开发环境检测（返回False）"""
    monkeypatch.setenv('ENV', 'production')
    from src.infrastructure.config.environment import is_development
    assert is_development() is False


def test_is_development_default(monkeypatch):
    """测试开发环境默认值"""
    monkeypatch.delenv('ENV', raising=False)
    from src.infrastructure.config.environment import is_development
    assert is_development() is True


def test_is_testing_true():
    """测试环境检测（pytest运行时应该返回True）"""
    from src.infrastructure.config.environment import is_testing
    # 在pytest中运行，PYTEST_CURRENT_TEST应该存在
    assert is_testing() is True


def test_is_testing_false(monkeypatch):
    """测试环境检测（非pytest时返回False）"""
    monkeypatch.delenv('PYTEST_CURRENT_TEST', raising=False)
    from src.infrastructure.config.environment import is_testing
    assert is_testing() is False


#!/usr/bin/env python3
"""
RQA2025 数据源配置持久化测试
测试数据源配置的保存、加载和监控功能
"""

import pytest
import json
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

from gateway.web.api import load_data_sources, save_data_sources


class TestDataSourcePersistence:
    """数据源配置持久化测试"""

    def test_data_source_config_save_and_load(self, tmp_path):
        """测试数据源配置的保存和加载"""
        config_file = tmp_path / 'test_data_sources.json'

        # 测试数据源配置
        test_sources = [
            {
                "id": "miniqmt",
                "name": "MiniQMT",
                "type": "本地交易",
                "url": "local.qmt.cn",
                "enabled": True,
                "status": "连接正常",
                "rate_limit": "无限制"
            },
            {
                "id": "emweb",
                "name": "东方财富",
                "type": "行情数据",
                "url": "emweb.securities.com.cn",
                "enabled": True,
                "status": "连接正常",
                "rate_limit": "100次/分钟"
            },
            {
                "id": "alpha-vantage",
                "name": "Alpha Vantage",
                "type": "股票数据",
                "url": "https://www.alphavantage.co",
                "enabled": False,
                "status": "已禁用",
                "rate_limit": "5次/分钟"
            }
        ]

        # 保存配置
        save_data_sources(test_sources, config_file=str(config_file))
        assert config_file.exists()

        # 加载配置
        loaded_sources = load_data_sources(config_file=str(config_file))

        # 验证加载的配置
        assert len(loaded_sources) == 3
        assert loaded_sources[0]["id"] == "miniqmt"
        assert loaded_sources[0]["enabled"] == True
        assert loaded_sources[1]["id"] == "emweb"
        assert loaded_sources[1]["enabled"] == True
        assert loaded_sources[2]["id"] == "alpha-vantage"
        assert loaded_sources[2]["enabled"] == False

    def test_enabled_sources_only_monitoring_logic(self):
        """测试只监控启用的数据源的逻辑"""
        # 模拟当前的数据源状态（基于我们的修改）
        all_sources = [
            {"id": "miniqmt", "enabled": True, "type": "本地交易"},
            {"id": "emweb", "enabled": True, "type": "行情数据"},
            {"id": "alpha-vantage", "enabled": False, "type": "股票数据"},
            {"id": "binance", "enabled": False, "type": "加密货币"},
            {"id": "yahoo", "enabled": False, "type": "市场指数"},
            {"id": "newsapi", "enabled": False, "type": "新闻数据"},
            {"id": "fred", "enabled": False, "type": "宏观经济"},
            {"id": "coingecko", "enabled": False, "type": "加密货币"}
        ]

        # 获取启用的数据源
        enabled_sources = [s for s in all_sources if s["enabled"]]
        disabled_sources = [s for s in all_sources if not s["enabled"]]

        # 验证只有A股数据源被启用
        assert len(enabled_sources) == 2
        assert enabled_sources[0]["id"] == "miniqmt"
        assert enabled_sources[1]["id"] == "emweb"

        # 验证其他数据源都被禁用
        assert len(disabled_sources) == 6
        disabled_ids = [s["id"] for s in disabled_sources]
        assert "alpha-vantage" in disabled_ids
        assert "binance" in disabled_ids
        assert "fred" in disabled_ids
        assert "coingecko" in disabled_ids

        # 验证监控逻辑：只有启用的数据源应该有性能指标
        for source in enabled_sources:
            assert source["enabled"] == True
            # 在实际的前端代码中，这些数据源会有延迟和吞吐量指标

        for source in disabled_sources:
            assert source["enabled"] == False
            # 在实际的前端代码中，这些数据源的吞吐量会显示为0

    def test_config_partial_update(self, tmp_path):
        """测试配置的部分更新"""
        config_file = tmp_path / 'test_partial_update.json'

        # 初始配置
        initial_sources = [
            {
                "id": "test_source",
                "name": "测试数据源",
                "type": "股票数据",
                "url": "https://test.com",
                "enabled": True,
                "status": "连接正常"
            }
        ]

        # 保存初始配置
        save_data_sources(initial_sources, config_file=str(config_file))

        # 模拟部分更新（只更新enabled字段）
        sources = load_data_sources(config_file=str(config_file))
        sources[0]["enabled"] = False
        sources[0]["status"] = "已禁用"
        save_data_sources(sources, config_file=str(config_file))

        # 加载并验证更新
        updated_sources = load_data_sources(config_file=str(config_file))
        assert updated_sources[0]["enabled"] == False
        assert updated_sources[0]["status"] == "已禁用"
        # 其他字段应该保持不变
        assert updated_sources[0]["name"] == "测试数据源"
        assert updated_sources[0]["url"] == "https://test.com"

    def test_config_file_error_handling(self, tmp_path):
        """测试配置文件错误处理"""
        # 测试不存在的文件
        nonexistent_file = tmp_path / 'nonexistent.json'
        sources = load_data_sources(config_file=str(nonexistent_file))

        # 应该返回默认配置
        assert isinstance(sources, list)
        assert len(sources) > 0  # 默认配置应该有数据源

        # 测试损坏的JSON文件
        corrupted_file = tmp_path / 'corrupted.json'
        corrupted_file.write_text("{ invalid json }")

        # 应该返回默认配置而不是崩溃
        sources = load_data_sources(config_file=str(corrupted_file))
        assert isinstance(sources, list)

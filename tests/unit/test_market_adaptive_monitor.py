#!/usr/bin/env python3
"""
市场状态自适应监控器单元测试
"""

import pytest
import sys
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.orchestration.market_adaptive_monitor import (
    MarketAdaptiveMonitor,
    MarketRegime,
    MarketMetrics,
    MarketRegimeAnalysis
)


class TestMarketAdaptiveMonitor:
    """市场状态自适应监控器测试类"""

    def setup_method(self):
        """测试前准备"""
        self.config = {
            'update_interval_seconds': 300,
            'history_window_days': 7,
            'volatility_thresholds': {
                'high': 0.05,
                'extreme': 0.10
            },
            'trend_thresholds': {
                'strong_bull': 0.03,
                'strong_bear': -0.03,
                'sideways_range': 0.02
            },
            'volume_thresholds': {
                'low_liquidity': 0.3
            },
            'analysis_window_days': 3,
            'min_data_points': 3
        }
        self.monitor = MarketAdaptiveMonitor(self.config)

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_initialization(self):
        """测试初始化"""
        assert self.monitor.config == self.config
        assert len(self.monitor._market_history) == 0
        assert len(self.monitor._regime_history) == 0
        assert self.monitor._last_update is None

    def test_default_config(self):
        """测试默认配置"""
        monitor = MarketAdaptiveMonitor()

        assert monitor.config['update_interval_seconds'] == 300
        assert monitor.config['volatility_thresholds']['high'] == 0.05
        assert monitor.config['trend_thresholds']['strong_bull'] == 0.03

    @patch('src.core.orchestration.market_adaptive_monitor.datetime')
    async def test_get_current_regime_first_time(self, mock_datetime):
        """测试首次获取市场状态"""
        mock_datetime.now.return_value = datetime(2023, 1, 1, 12, 0, 0)

        # Mock基础设施数据获取
        with patch.object(self.monitor, '_fetch_market_data_from_infrastructure', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = self._get_mock_market_data()

            regime_analysis = await self.monitor.get_current_regime()

            # 验证结果
            assert isinstance(regime_analysis, MarketRegimeAnalysis)
            assert regime_analysis.current_regime == MarketRegime.SIDEWAYS  # 默认状态
            assert regime_analysis.confidence == 0.7
            assert len(regime_analysis.recommended_actions) > 0

            # 验证历史记录
            assert len(self.monitor._regime_history) == 1
            assert len(self.monitor._market_history) == 1

    async def test_get_current_regime_with_history(self):
        """测试有历史数据时获取市场状态"""
        # 添加历史数据
        historical_metrics = MarketMetrics(
            timestamp=datetime(2023, 1, 1, 10, 0, 0),
            volatility=0.02,
            trend_strength=0.015,
            market_breadth=0.52
        )
        self.monitor._market_history.append(historical_metrics)

        # Mock基础设施数据获取
        with patch.object(self.monitor, '_fetch_market_data_from_infrastructure', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = self._get_mock_market_data()

            regime_analysis = await self.monitor.get_current_regime()

            assert isinstance(regime_analysis, MarketRegimeAnalysis)
            assert len(self.monitor._regime_history) == 1

    def test_regime_analysis_high_volatility(self):
        """测试高波动状态识别"""
        metrics = MarketMetrics(
            timestamp=datetime.now(),
            volatility=0.08,  # 高波动
            trend_strength=0.01,
            market_breadth=0.5,
            volatility_percentile=85.0  # 高百分位
        )

        regime_analysis = self.monitor._analyze_market_regime(metrics)

        assert regime_analysis.current_regime == MarketRegime.HIGH_VOLATILITY
        assert regime_analysis.confidence > 0.8
        assert len(regime_analysis.recommended_actions) > 0
        assert "减少采集频率" in regime_analysis.recommended_actions[0]

    def test_regime_analysis_bull_market(self):
        """测试牛市状态识别"""
        metrics = MarketMetrics(
            timestamp=datetime.now(),
            volatility=0.02,
            trend_strength=0.04,  # 强上涨趋势
            market_breadth=0.65,
            volatility_percentile=30.0
        )

        regime_analysis = self.monitor._analyze_market_regime(metrics)

        assert regime_analysis.current_regime == MarketRegime.BULL
        assert regime_analysis.confidence > 0.8
        assert "增加数据采集频率" in regime_analysis.recommended_actions[0]

    def test_regime_analysis_bear_market(self):
        """测试熊市状态识别"""
        metrics = MarketMetrics(
            timestamp=datetime.now(),
            volatility=0.025,
            trend_strength=-0.04,  # 强下跌趋势
            market_breadth=0.35,
            volatility_percentile=40.0
        )

        regime_analysis = self.monitor._analyze_market_regime(metrics)

        assert regime_analysis.current_regime == MarketRegime.BEAR
        assert regime_analysis.confidence > 0.8
        assert "保持正常采集频率" in regime_analysis.recommended_actions[0]

    def test_regime_analysis_sideways(self):
        """测试横盘状态识别"""
        metrics = MarketMetrics(
            timestamp=datetime.now(),
            volatility=0.015,
            trend_strength=0.008,  # 小幅波动
            market_breadth=0.48,
            volatility_percentile=25.0
        )

        regime_analysis = self.monitor._analyze_market_regime(metrics)

        assert regime_analysis.current_regime == MarketRegime.SIDEWAYS
        assert regime_analysis.confidence >= 0.7
        assert "均衡采集各数据源" in regime_analysis.recommended_actions[1]

    def test_regime_analysis_low_liquidity(self):
        """测试低流动性状态识别"""
        metrics = MarketMetrics(
            timestamp=datetime.now(),
            volatility=0.02,
            trend_strength=0.01,
            market_breadth=0.5,
            volume_percentile=15.0  # 低成交量百分位
        )

        regime_analysis = self.monitor._analyze_market_regime(metrics)

        assert regime_analysis.current_regime == MarketRegime.LOW_LIQUIDITY
        assert regime_analysis.confidence > 0.7
        assert "延长采集间隔" in regime_analysis.recommended_actions[0]

    def test_process_market_data(self):
        """测试市场数据处理"""
        raw_data = self._get_mock_market_data()

        metrics = self.monitor._process_market_data(raw_data)

        assert isinstance(metrics, MarketMetrics)
        assert metrics.volatility > 0
        assert metrics.trend_strength >= -1 and metrics.trend_strength <= 1
        assert metrics.market_breadth >= 0 and metrics.market_breadth <= 1

    def test_calculate_percentile_with_history(self):
        """测试百分位数计算（有历史数据）"""
        # 添加历史数据
        for i in range(10):
            metrics = MarketMetrics(
                timestamp=datetime.now() - timedelta(hours=i),
                volatility=0.01 + i * 0.005,  # 递增波动率
                market_breadth=0.4 + i * 0.05  # 递增市场宽度
            )
            self.monitor._market_history.append(metrics)

        # 测试波动率百分位数
        percentile = self.monitor._calculate_percentile(0.03, 'volatility')
        assert percentile >= 0 and percentile <= 100

        # 测试市场宽度百分位数
        percentile = self.monitor._calculate_percentile(0.55, 'breadth')
        assert percentile >= 0 and percentile <= 100

    def test_calculate_percentile_without_history(self):
        """测试百分位数计算（无历史数据）"""
        percentile = self.monitor._calculate_percentile(0.05, 'volatility')
        assert percentile == 50.0  # 默认中位数

    def test_calculate_volume_trend(self):
        """测试成交量趋势计算"""
        # 无历史数据
        trend = self.monitor._calculate_volume_trend(1000000)
        assert trend == 0.0

        # 添加历史数据
        current_time = datetime.now()
        for i in range(5):
            metrics = MarketMetrics(
                timestamp=current_time - timedelta(hours=i+1),
                volatility=0.02
            )
            self.monitor._market_history.append(metrics)

        trend = self.monitor._calculate_volume_trend(1200000)
        assert isinstance(trend, float)

    def test_get_regime_statistics(self):
        """测试市场状态统计"""
        # 无历史数据
        stats = self.monitor.get_regime_statistics()
        assert stats == {}

        # 添加历史分析
        for regime in [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS]:
            analysis = MarketRegimeAnalysis(
                current_regime=regime,
                confidence=0.8,
                metrics=MarketMetrics(timestamp=datetime.now()),
                indicators={},
                recommended_actions=["test"],
                analysis_timestamp=datetime.now()
            )
            self.monitor._regime_history.append(analysis)

        stats = self.monitor.get_regime_statistics()

        assert 'regime_distribution' in stats
        assert 'most_common_regime' in stats
        assert 'analysis_count' in stats
        assert stats['analysis_count'] == 3
        assert len(stats['regime_distribution']) == 3

    def test_get_regime_history(self):
        """测试获取市场状态历史"""
        # 无历史数据
        history = self.monitor.get_regime_history()
        assert history == []

        # 添加历史分析
        for i in range(5):
            analysis = MarketRegimeAnalysis(
                current_regime=MarketRegime.SIDEWAYS,
                confidence=0.8,
                metrics=MarketMetrics(timestamp=datetime.now()),
                indicators={},
                recommended_actions=["test"],
                analysis_timestamp=datetime.now()
            )
            self.monitor._regime_history.append(analysis)

        # 获取全部历史
        history = self.monitor.get_regime_history()
        assert len(history) == 5

        # 获取最近3个
        history = self.monitor.get_regime_history(limit=3)
        assert len(history) == 3

    def test_get_market_metrics_history(self):
        """测试获取市场指标历史"""
        # 无历史数据
        history = self.monitor.get_market_metrics_history()
        assert history == []

        # 添加历史指标
        for i in range(10):
            metrics = MarketMetrics(
                timestamp=datetime.now() - timedelta(hours=i),
                volatility=0.02
            )
            self.monitor._market_history.append(metrics)

        # 获取全部历史
        history = self.monitor.get_market_metrics_history()
        assert len(history) == 10

        # 获取最近5个
        history = self.monitor.get_market_metrics_history(limit=5)
        assert len(history) == 5

    def test_error_handling(self):
        """测试错误处理"""
        # 测试数据处理异常
        with patch.object(self.monitor, '_fetch_market_data_from_infrastructure', new_callable=AsyncMock) as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")

            # 应该返回默认状态而不是抛出异常
            regime_analysis = await self.monitor.get_current_regime()
            assert isinstance(regime_analysis, MarketRegimeAnalysis)
            assert regime_analysis.current_regime == MarketRegime.SIDEWAYS

    def test_mock_data_generation(self):
        """测试模拟数据生成"""
        mock_data = self.monitor._get_mock_market_data()

        assert 'timestamp' in mock_data
        assert 'indices' in mock_data
        assert 'market_breadth' in mock_data
        assert 'sentiment_score' in mock_data
        assert isinstance(mock_data['indices'], dict)
        assert len(mock_data['indices']) > 0

    def test_default_regime_analysis(self):
        """测试默认市场状态分析"""
        default_analysis = self.monitor._get_default_regime_analysis()

        assert isinstance(default_analysis, MarketRegimeAnalysis)
        assert default_analysis.current_regime == MarketRegime.SIDEWAYS
        assert default_analysis.confidence == 0.5
        assert len(default_analysis.recommended_actions) > 0

    def _get_mock_market_data(self):
        """获取模拟市场数据"""
        return {
            'timestamp': datetime.now(),
            'indices': {
                'sh000001': {
                    'price': 3200.0,
                    'change_pct': 0.015,
                    'volume': 180000000,
                    'volatility': 0.022
                },
                'sz399001': {
                    'price': 10500.0,
                    'change_pct': -0.008,
                    'volume': 150000000,
                    'volatility': 0.018
                }
            },
            'market_breadth': 0.52,
            'total_volume': 320000000,
            'sentiment_score': 0.55
        }

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """测试并发访问"""
        import asyncio

        async def worker(worker_id):
            for i in range(5):
                regime_analysis = await self.monitor.get_current_regime()
                assert isinstance(regime_analysis, MarketRegimeAnalysis)
                await asyncio.sleep(0.01)  # 避免过于频繁的调用
            return worker_id

        # 创建多个并发任务
        tasks = [worker(i) for i in range(3)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert set(results) == {0, 1, 2}

    def test_config_edge_cases(self):
        """测试配置边界情况"""
        # 空配置
        monitor = MarketAdaptiveMonitor({})
        assert monitor.config['update_interval_seconds'] == 300  # 应该使用默认值

        # 无效配置值
        invalid_config = {
            'update_interval_seconds': -1,  # 无效值
            'volatility_thresholds': {}
        }
        monitor = MarketAdaptiveMonitor(invalid_config)
        # 应该能正常初始化，不抛出异常
        assert monitor.config is not None


if __name__ == '__main__':
    pytest.main([__file__])
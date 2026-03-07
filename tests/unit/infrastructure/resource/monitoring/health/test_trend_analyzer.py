"""
测试目标：提升resource/monitoring/health/trend_analyzer.py的真实覆盖率
实际导入和使用src.infrastructure.resource.monitoring.health.trend_analyzer模块
"""

from unittest.mock import Mock
import pytest

from src.infrastructure.resource.monitoring.health.trend_analyzer import TrendAnalyzer


class TestTrendAnalyzer:
    """测试TrendAnalyzer类"""

    @pytest.fixture
    def mock_logger(self):
        """模拟logger"""
        return Mock()

    @pytest.fixture
    def analyzer(self, mock_logger):
        """创建分析器实例"""
        return TrendAnalyzer(mock_logger)

    def test_initialization(self, analyzer, mock_logger):
        """测试初始化"""
        assert analyzer.logger == mock_logger

    def test_initialization_without_logger(self):
        """测试不提供logger时的初始化"""
        analyzer = TrendAnalyzer()

        assert analyzer.logger is not None
        assert hasattr(analyzer.logger, 'log_info')

    def test_analyze_health_trends_insufficient_data(self, analyzer):
        """测试数据不足时的趋势分析"""
        health_history = [{"score": 80}]  # 只有一条数据

        result = analyzer.analyze_health_trends(health_history)

        assert result["trend"] == "insufficient_data"
        assert result["direction"] == "unknown"
        assert result["confidence"] == 0.0

    def test_analyze_health_trends_empty_data(self, analyzer):
        """测试空数据时的趋势分析"""
        result = analyzer.analyze_health_trends([])

        assert result["trend"] == "insufficient_data"
        assert result["direction"] == "unknown"
        assert result["confidence"] == 0.0

    def test_analyze_health_trends_improving(self, analyzer):
        """测试改善趋势"""
        health_history = [
            {"score": 60, "timestamp": 1000},
            {"score": 70, "timestamp": 1100},
            {"score": 80, "timestamp": 1200},
            {"score": 90, "timestamp": 1300}
        ]

        result = analyzer.analyze_health_trends(health_history)

        assert result["trend"] == "improving"
        assert result["direction"] == "up"
        assert result["confidence"] > 0.5
        assert "slope" in result
        assert "r_squared" in result
        assert "change_rate" in result

    def test_analyze_health_trends_declining(self, analyzer):
        """测试下降趋势"""
        health_history = [
            {"score": 90, "timestamp": 1000},
            {"score": 80, "timestamp": 1100},
            {"score": 70, "timestamp": 1200},
            {"score": 60, "timestamp": 1300}
        ]

        result = analyzer.analyze_health_trends(health_history)

        assert result["trend"] == "declining"
        assert result["direction"] == "down"
        assert result["confidence"] > 0.5

    def test_analyze_health_trends_stable(self, analyzer):
        """测试稳定趋势"""
        health_history = [
            {"score": 75, "timestamp": 1000},
            {"score": 78, "timestamp": 1100},
            {"score": 76, "timestamp": 1200},
            {"score": 77, "timestamp": 1300}
        ]

        result = analyzer.analyze_health_trends(health_history)

        assert result["trend"] == "stable"
        assert result["direction"] == "stable"
        assert result["confidence"] >= 0.0

    def test_get_time_range_empty_history(self, analyzer):
        """测试空历史的时间范围"""
        result = analyzer.get_time_range([])

        assert result["start"] is None
        assert result["end"] is None
        assert result["duration_hours"] == 0

    def test_get_time_range_no_timestamps(self, analyzer):
        """测试无时间戳的历史"""
        health_history = [{"score": 80}]  # 没有timestamp字段

        result = analyzer.get_time_range(health_history)

        assert result["start"] is None
        assert result["end"] is None
        assert result["duration_hours"] == 0

    def test_get_time_range_valid_history(self, analyzer):
        """测试有效历史的时间范围"""
        health_history = [
            {"score": 80, "timestamp": 1000},
            {"score": 85, "timestamp": 2000},
            {"score": 90, "timestamp": 1500}
        ]

        result = analyzer.get_time_range(health_history)

        assert result["start"] == 1000
        assert result["end"] == 2000
        assert result["duration_hours"] == 1.0  # (2000-1000)/3600 = 1.0

    def test_calculate_average_score_empty_history(self, analyzer):
        """测试空历史的平均评分"""
        result = analyzer.calculate_average_score([])

        assert result == 0.0

    def test_calculate_average_score_no_scores(self, analyzer):
        """测试无评分的平均评分"""
        health_history = [{"timestamp": 1000}]  # 没有score字段

        result = analyzer.calculate_average_score(health_history)

        assert result == 0.0

    def test_calculate_average_score_valid_history(self, analyzer):
        """测试有效历史的平均评分"""
        health_history = [
            {"score": 80, "timestamp": 1000},
            {"score": 85, "timestamp": 1100},
            {"score": 90, "timestamp": 1200}
        ]

        result = analyzer.calculate_average_score(health_history)

        assert result == 85.0  # (80 + 85 + 90) / 3

    def test_analyze_key_metrics_trends(self, analyzer):
        """测试关键指标趋势分析"""
        health_history = [
            {"score": 80, "timestamp": 1000}
        ]

        result = analyzer.analyze_key_metrics_trends(health_history)

        assert result["performance_trend"] == "stable"
        assert result["alert_trend"] == "stable"
        assert result["test_trend"] == "stable"
        assert isinstance(result["details"], dict)

    def test_extract_scores(self, analyzer):
        """测试评分提取"""
        health_history = [
            {"score": 80, "timestamp": 1000},
            {"score": 85, "timestamp": 1100},
            {"timestamp": 1200},  # 无score
            {"score": 90, "timestamp": 1300}
        ]

        scores = analyzer._extract_scores(health_history)

        assert scores == [80, 85, 90]

    def test_extract_scores_empty_history(self, analyzer):
        """测试空历史的评分提取"""
        scores = analyzer._extract_scores([])

        assert scores == []

    def test_extract_timestamps(self, analyzer):
        """测试时间戳提取"""
        health_history = [
            {"score": 80, "timestamp": 1000},
            {"score": 85, "timestamp": 1100},
            {"score": 90}  # 无timestamp
        ]

        timestamps = analyzer._extract_timestamps(health_history)

        assert timestamps == [1000, 1100]

    def test_extract_timestamps_empty_history(self, analyzer):
        """测试空历史的时间戳提取"""
        timestamps = analyzer._extract_timestamps([])

        assert timestamps == []

    def test_calculate_trend_info_improving(self, analyzer):
        """测试改善趋势信息计算"""
        scores = [60, 70, 80, 90]

        result = analyzer._calculate_trend_info(scores)

        assert result["slope"] > 0
        assert "r_squared" in result
        assert "direction" in result

    def test_calculate_trend_info_declining(self, analyzer):
        """测试下降趋势信息计算"""
        scores = [90, 80, 70, 60]

        result = analyzer._calculate_trend_info(scores)

        assert result["slope"] < 0
        assert "r_squared" in result
        assert "direction" in result

    def test_calculate_trend_info_stable(self, analyzer):
        """测试稳定趋势信息计算"""
        scores = [75, 76, 77, 78]

        result = analyzer._calculate_trend_info(scores)

        assert abs(result["slope"]) < 1  # 接近水平
        assert "r_squared" in result
        assert "direction" in result

    def test_build_trend_result(self, analyzer):
        """测试趋势结果构建"""
        trend_info = {
            "slope": 2.5,
            "r_squared": 0.85,
            "direction": "up"
        }
        scores = [60, 70, 80, 90]

        result = analyzer._build_trend_result(trend_info, scores)

        assert result["trend"] == "improving"
        assert result["direction"] == "up"
        assert result["confidence"] == 0.85
        assert result["slope"] == 2.5
        assert result["r_squared"] == 0.85
        assert result["data_points"] == 4
        assert "change_rate" in result

    def test_calculate_time_range(self, analyzer):
        """测试时间范围计算"""
        timestamps = [1000, 1500, 2000]

        result = analyzer._calculate_time_range(timestamps)

        assert result["start"] == 1000
        assert result["end"] == 2000
        assert result["duration_hours"] == 1.0  # (2000-1000)/3600

    def test_calculate_time_range_single_timestamp(self, analyzer):
        """测试单个时间戳的时间范围计算"""
        timestamps = [1000]

        result = analyzer._calculate_time_range(timestamps)

        assert result["start"] == 1000
        assert result["end"] == 1000
        assert result["duration_hours"] == 0

import pytest

class TestFPGAModules:
    @pytest.mark.hardware
    def test_sentiment_analysis(self, fpga_health_check):
        """FPGA情感分析测试"""
        # 跳过测试如果FPGA不健康
        if fpga_health_check["temperature"] > 50:
            pytest.skip("FPGA温度过高")

        from src.fpga.fpga_manager import FPGASentimentAnalyzer
        analyzer = FPGASentimentAnalyzer()
        result = analyzer.analyze("利好公告发布")
        assert 0 <= result["score"] <= 1

    @pytest.mark.performance
    def test_throughput(self):
        """FPGA吞吐量测试"""
        from src.fpga.fpga_manager import FPGASentimentAnalyzer
        analyzer = FPGASentimentAnalyzer()
        samples = ["测试文本"] * 1000
        stats = analyzer.benchmark(samples)
        assert stats["throughput"] > 8000  # 8000次/秒

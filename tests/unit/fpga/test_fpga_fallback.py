class TestFallbackScenarios:
    def test_high_load_fallback(self):
        """高负载下自动降级测试"""
        with simulate_load(cpu=90):  # 模拟高负载
            engine = RiskEngine()
            assert engine.current_mode == 'software'

    def test_switch_latency(self):
        """模式切换延迟测试"""
        t = measure_switch_time('fpga', 'software')
        assert t < 100  # 切换时间<100ms
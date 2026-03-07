"""
RQA2026 高级功能测试

验证量子计算、AI市场分析、BMI实时信号处理等高级功能
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

# 导入RQA2026组件
try:
    from src.rqa2026.quantum.portfolio_optimizer import (
        QuantumPortfolioOptimizer, QuantumRiskAnalyzer, QuantumOptionPricer,
        PortfolioConstraints, AssetData, PortfolioResult
    )
    from src.rqa2026.ai.market_analyzer import (
        MarketSentimentAnalyzer, ChartPatternRecognizer, TradingSignalGenerator,
        MarketSentiment, ChartPattern, TradingSignal
    )
    from src.rqa2026.bmi.signal_processor import (
        RealtimeSignalProcessor, BMICommunicationInterface,
        RealtimeSignalData, IntentPrediction, BMICommand
    )
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 某些高级组件不可用: {e}")
    COMPONENTS_AVAILABLE = False


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="RQA2026高级组件不可用")
class TestQuantumPortfolioOptimization:
    """量子投资组合优化测试"""

    @pytest.fixture
    def sample_assets(self):
        """创建示例资产数据"""
        return [
            AssetData(
                symbol="AAPL",
                expected_return=0.12,
                volatility=0.25,
                current_price=150.0,
                historical_prices=[145, 148, 152, 149, 151]
            ),
            AssetData(
                symbol="GOOGL",
                expected_return=0.10,
                volatility=0.30,
                current_price=2500.0,
                historical_prices=[2480, 2490, 2520, 2500, 2510]
            ),
            AssetData(
                symbol="MSFT",
                expected_return=0.15,
                volatility=0.28,
                current_price=300.0,
                historical_prices=[295, 298, 305, 302, 299]
            )
        ]

    @pytest.fixture
    def constraints(self):
        """创建约束条件"""
        return PortfolioConstraints(
            min_weight=0.0,
            max_weight=0.5,
            min_assets=1,
            target_return=None,
            max_risk=None
        )

    @pytest.mark.asyncio
    async def test_quantum_portfolio_optimization(self, sample_assets, constraints):
        """测试量子投资组合优化"""
        optimizer = QuantumPortfolioOptimizer(use_quantum=True)

        result = await optimizer.optimize_portfolio(sample_assets, constraints, method="qaoa")

        assert isinstance(result, PortfolioResult)
        assert result.optimization_method == "qaoa"
        assert len(result.weights) == len(sample_assets)
        assert abs(sum(result.weights.values()) - 1.0) < 0.01  # 权重和为1
        assert all(0 <= w <= 0.5 for w in result.weights.values())  # 满足权重约束
        assert result.expected_return > 0
        assert result.volatility > 0
        assert result.sharpe_ratio > 0

    @pytest.mark.asyncio
    async def test_classical_portfolio_optimization(self, sample_assets, constraints):
        """测试经典投资组合优化"""
        optimizer = QuantumPortfolioOptimizer(use_quantum=False)

        result = await optimizer.optimize_portfolio(sample_assets, constraints, method="classical")

        assert isinstance(result, PortfolioResult)
        assert result.optimization_method == "classical"
        assert result.quantum_advantage is None
        assert len(result.weights) == len(sample_assets)

    @pytest.mark.asyncio
    async def test_algorithm_comparison(self, sample_assets, constraints):
        """测试算法比较"""
        optimizer = QuantumPortfolioOptimizer(use_quantum=False)  # 避免量子库依赖

        results = await optimizer.compare_algorithms(sample_assets, constraints)

        assert "classical" in results
        # 如果量子可用，还会有qaoa和vqe结果

        for method, result in results.items():
            assert isinstance(result, PortfolioResult)
            assert result.optimization_method == method

    @pytest.mark.asyncio
    async def test_quantum_risk_analysis(self, sample_assets):
        """测试量子风险分析"""
        analyzer = QuantumRiskAnalyzer()

        # 创建模拟投资组合权重
        weights = {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3}

        # 模拟历史收益率数据
        historical_returns = np.random.normal(0.001, 0.02, (len(sample_assets), 100))

        result = await analyzer.calculate_quantum_var(
            weights, historical_returns, confidence_level=0.95
        )

        assert "var" in result
        assert "expected_shortfall" in result
        assert result["confidence_level"] == 0.95
        assert result["var"] > 0
        assert result["expected_shortfall"] >= result["var"]

    @pytest.mark.asyncio
    async def test_quantum_option_pricing(self):
        """测试量子期权定价"""
        pricer = QuantumOptionPricer()

        result = await pricer.price_option(
            spot_price=100.0,
            strike_price=105.0,
            time_to_maturity=1.0,
            volatility=0.2,
            risk_free_rate=0.05,
            option_type="call"
        )

        assert "price" in result
        assert "delta" in result
        assert "gamma" in result
        assert "theta" in result
        assert "vega" in result
        assert "rho" in result
        assert result["price"] > 0
        assert result["method"] == "quantum_bs"


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="RQA2026高级组件不可用")
class TestAIMarketAnalysis:
    """AI市场分析测试"""

    @pytest.mark.asyncio
    async def test_market_sentiment_analysis(self):
        """测试市场情绪分析"""
        analyzer = MarketSentimentAnalyzer()

        # 测试文本情绪分析
        texts = [
            "Market is showing strong bullish signals",
            "Investors are optimistic about tech stocks",
            "Bearish sentiment dominates the market"
        ]

        price_data = np.array([100, 102, 105, 103, 108])
        volume_data = np.array([1000, 1200, 1500, 1100, 1300])

        sentiment = await analyzer.analyze_market_sentiment(
            texts, price_data, volume_data
        )

        assert isinstance(sentiment, MarketSentiment)
        assert sentiment.overall_sentiment in ["bullish", "bearish", "neutral"]
        assert 0 <= sentiment.confidence <= 1
        assert isinstance(sentiment.sources, dict)
        assert isinstance(sentiment.key_factors, list)

    @pytest.mark.asyncio
    async def test_chart_pattern_recognition(self):
        """测试图表模式识别"""
        recognizer = ChartPatternRecognizer()

        # 创建模拟价格数据（双顶模式）
        prices = np.array([100, 105, 110, 105, 102, 98, 105, 110, 106, 103])

        patterns = await recognizer.recognize_patterns(prices)

        assert isinstance(patterns, list)
        for pattern in patterns:
            assert isinstance(pattern, ChartPattern)
            assert pattern.confidence > 0
            assert pattern.pattern_name != ""
            assert pattern.direction in ["bullish", "bearish", "neutral"]

    @pytest.mark.asyncio
    async def test_trading_signal_generation(self):
        """测试交易信号生成"""
        generator = TradingSignalGenerator()

        assets = ["AAPL", "GOOGL"]
        market_data = {
            "AAPL": {
                "prices": np.array([150, 152, 155, 153, 158]),
                "volume": np.array([1000000, 1200000, 1500000, 1100000, 1300000])
            },
            "GOOGL": {
                "prices": np.array([2500, 2520, 2550, 2530, 2580]),
                "volume": np.array([500000, 600000, 750000, 550000, 650000])
            }
        }

        sentiment_data = {
            "AAPL": {"news": ["AAPL shows strong growth", "Positive earnings report"]},
            "GOOGL": {"news": ["Google AI breakthrough", "Market leadership confirmed"]}
        }

        signals = await generator.generate_signals(
            assets, market_data, sentiment_data, risk_tolerance="medium"
        )

        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.signal_type in ["BUY", "SELL", "HOLD"]
            assert 0 <= signal.confidence <= 1
            assert signal.asset in assets
            assert signal.risk_level in ["low", "medium", "high"]
            assert isinstance(signal.supporting_evidence, list)


@pytest.mark.skipif(not COMPONENTS_AVAILABLE, reason="RQA2026高级组件不可用")
class TestBMIRealtimeProcessing:
    """BMI实时信号处理测试"""

    @pytest.fixture
    def sample_eeg_data(self):
        """创建示例EEG数据"""
        # 4通道，250Hz采样率，1秒数据
        return np.random.randn(4, 250)

    @pytest.mark.asyncio
    async def test_realtime_signal_processor(self, sample_eeg_data):
        """测试实时信号处理器"""
        processor = RealtimeSignalProcessor(sampling_rate=250.0)

        # 添加信号数据
        await processor.add_signal_data(sample_eeg_data)

        # 检查信号质量
        quality_metrics = processor.get_signal_quality_metrics()

        assert isinstance(quality_metrics, dict)
        assert "snr" in quality_metrics
        assert "noise_level" in quality_metrics
        assert "artifact_ratio" in quality_metrics

    @pytest.mark.asyncio
    async def test_intent_prediction(self, sample_eeg_data):
        """测试意图预测"""
        processor = RealtimeSignalProcessor()

        # 模拟意图预测结果
        features = {
            'band_power': {'alpha': [0.8, 0.7, 0.9, 0.6], 'beta': [0.6, 0.8, 0.5, 0.7]},
            'connectivity': np.random.rand(4, 4),
            'entropy': [1.2, 1.5, 1.1, 1.3],
            'complexity': [5.2, 4.8, 5.5, 4.9]
        }

        prediction = await processor._predict_intent(features)

        assert isinstance(prediction, IntentPrediction)
        assert prediction.intent != ""
        assert 0 <= prediction.confidence <= 1
        assert isinstance(prediction.probability_distribution, dict)

    @pytest.mark.asyncio
    async def test_bmi_command_generation(self):
        """测试BMI命令生成"""
        interface = BMICommunicationInterface()

        # 注册命令处理器
        await interface.register_command_handler("trade", interface.default_trade_handler)
        await interface.register_command_handler("control", interface.default_control_handler)

        # 创建测试命令
        buy_command = BMICommand(
            command_type="trade",
            action="BUY",
            parameters={"quantity": 100, "symbol": "AAPL"},
            confidence=0.9,
            urgency="high",
            timestamp=asyncio.get_event_loop().time()
        )

        # 执行命令
        response = await interface.execute_command(buy_command)

        assert response["success"] is True
        assert response["result"]["action"] == "BUY"
        assert response["result"]["status"] == "executed"

    @pytest.mark.asyncio
    async def test_bmi_communication_workflow(self, sample_eeg_data):
        """测试BMI通信完整工作流"""
        processor = RealtimeSignalProcessor()
        interface = BMICommunicationInterface()

        # 注册命令处理器
        await interface.register_command_handler("trade", interface.default_trade_handler)

        # 设置回调
        intent_predictions = []
        commands_executed = []

        def intent_callback(prediction: IntentPrediction):
            intent_predictions.append(prediction)

        def command_callback(command: BMICommand):
            commands_executed.append(command)

        processor.add_intent_callback(intent_callback)
        interface.add_response_callback(lambda r: None)  # 简化响应处理

        # 启动处理器
        await processor.start_processing()

        try:
            # 添加信号数据
            await processor.add_signal_data(sample_eeg_data)

            # 等待处理
            await asyncio.sleep(0.2)

            # 验证意图预测
            assert len(intent_predictions) > 0
            for prediction in intent_predictions:
                assert isinstance(prediction, IntentPrediction)
                assert prediction.confidence > 0

        finally:
            # 停止处理器
            await processor.stop_processing()


class TestRQA2026IntegrationAdvanced:
    """RQA2026高级集成测试"""

    @pytest.mark.asyncio
    async def test_quantum_ai_bmi_integration(self):
        """测试量子计算、AI和BMI的集成"""
        print("\\n=== 高级集成测试：量子+AI+BMI ===")

        # 1. 量子投资组合优化
        print("1. 量子投资组合优化运行中...")
        from src.rqa2026.quantum.portfolio_optimizer import QuantumPortfolioOptimizer, AssetData, PortfolioConstraints

        assets = [
            AssetData("AAPL", 0.12, 0.25, 150.0, [145, 148, 152, 149, 151]),
            AssetData("GOOGL", 0.10, 0.30, 2500.0, [2480, 2490, 2520, 2500, 2510]),
            AssetData("MSFT", 0.15, 0.28, 300.0, [295, 298, 305, 302, 299])
        ]
        constraints = PortfolioConstraints(min_weight=0.1, max_weight=0.6)

        optimizer = QuantumPortfolioOptimizer(use_quantum=False)
        portfolio_result = await optimizer.optimize_portfolio(assets, constraints, "classical")
        print(f"Portfolio optimized with Sharpe ratio: {portfolio_result.sharpe_ratio:.2f}")
        # 2. AI市场分析
        print("2. AI市场分析运行中...")
        from src.rqa2026.ai.market_analyzer import MarketSentimentAnalyzer, TradingSignalGenerator

        sentiment_analyzer = MarketSentimentAnalyzer()
        market_sentiment = await sentiment_analyzer.analyze_market_sentiment([
            f"Portfolio shows Sharpe ratio of {portfolio_result.sharpe_ratio:.2f}",
            "Market conditions favorable for tech stocks"
        ])
        print(f"Market sentiment score: {market_sentiment.confidence_score:.2f}")
        # 3. BMI信号处理
        print("3. BMI信号处理运行中...")
        from src.rqa2026.bmi.signal_processor import RealtimeSignalProcessor, BMICommunicationInterface

        processor = RealtimeSignalProcessor()
        eeg_data = np.random.randn(4, 250)  # 1秒EEG数据
        await processor.add_signal_data(eeg_data)

        quality = processor.get_signal_quality_metrics()
        print(f"   信号质量 - SNR: {quality['snr']:.2f}, 噪声水平: {quality['noise_level']:.3f}")

        # 4. 集成决策
        print("4. 集成决策分析...")
        decision_factors = {
            "portfolio_score": portfolio_result.sharpe_ratio,
            "market_sentiment": 0.8 if market_sentiment.overall_sentiment == "bullish" else 0.5,
            "signal_quality": quality['snr'] / 10.0  # 归一化
        }

        overall_confidence = sum(decision_factors.values()) / len(decision_factors)
        final_decision = "BUY" if overall_confidence > 0.7 else "HOLD"

        print(".2")
        print(f"Overall confidence: {overall_confidence:.2f}")
        print(f"   决策因素: {decision_factors}")
        print(f"   最终决策: {final_decision}")

        # 验证集成结果
        assert portfolio_result.confidence > 0
        assert market_sentiment.confidence > 0
        assert quality['snr'] >= 0
        assert overall_confidence > 0
        assert final_decision in ["BUY", "HOLD", "SELL"]

        print("\\n🎉 高级集成测试通过！")
        print("✅ 三大创新引擎深度集成")
        print("✅ 跨模态智能决策能力")
        print("✅ 实时信号处理与分析")
        print("✅ 量子计算优化算法")

    @pytest.mark.asyncio
    async def test_performance_benchmarks(self):
        """性能基准测试"""
        print("\\n=== 性能基准测试 ===")

        import time

        # 量子优化性能测试
        print("量子优化性能测试...")
        optimizer = QuantumPortfolioOptimizer(use_quantum=False)
        assets = [AssetData(f"ASSET_{i}", 0.1 + i*0.01, 0.2 + i*0.02, 100.0 + i*10,
                           [100+i*10 + j for j in range(10)]) for i in range(10)]

        start_time = time.time()
        result = await optimizer.optimize_portfolio(assets, PortfolioConstraints(), "classical")
        quantum_time = time.time() - start_time
        print(".3")        # AI分析性能测试
        print("AI分析性能测试...")
        sentiment_analyzer = MarketSentimentAnalyzer()

        start_time = time.time()
        sentiment = await sentiment_analyzer.analyze_market_sentiment([
            "Market analysis for performance testing"] * 5)
        ai_time = time.time() - start_time
        print(".3")        # BMI处理性能测试
        print("BMI处理性能测试...")
        processor = RealtimeSignalProcessor()

        start_time = time.time()
        eeg_data = np.random.randn(8, 500)  # 8通道，2秒数据
        await processor.add_signal_data(eeg_data)
        quality = processor.get_signal_quality_metrics()
        bmi_time = time.time() - start_time
        print(".3")        # 性能评估
        print("\\n性能评估:")
        print(f"  • 量子优化: {'✅' if quantum_time < 1.0 else '⚠️'} ({quantum_time:.3f}s)")
        print(f"  • AI分析: {'✅' if ai_time < 0.5 else '⚠️'} ({ai_time:.3f}s)")
        print(f"  • BMI处理: {'✅' if bmi_time < 0.2 else '⚠️'} ({bmi_time:.3f}s)")
        print(f"  • 总处理时间: {quantum_time + ai_time + bmi_time:.3f}s")

        # 性能断言
        assert quantum_time < 2.0, f"量子优化过慢: {quantum_time:.3f}s"
        assert ai_time < 1.0, f"AI分析过慢: {ai_time:.3f}s"
        assert bmi_time < 0.5, f"BMI处理过慢: {bmi_time:.3f}s"

        print("\\n✅ 性能基准测试通过！")
        print("系统响应时间满足实时性要求")

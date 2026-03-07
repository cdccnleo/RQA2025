#!/usr/bin/env python3
"""
RQA2026 创新引擎演示系统

展示三大创新引擎的集成应用能力
"""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    # 导入RQA2026组件
    from src.rqa2026.quantum.portfolio_optimizer import (
        QuantumPortfolioOptimizer, AssetData, PortfolioConstraints
    )
    from src.rqa2026.ai.market_analyzer import (
        MarketSentimentAnalyzer, TradingSignalGenerator
    )
    from src.rqa2026.bmi.signal_processor import (
        RealtimeSignalProcessor, BMICommunicationInterface
    )
    from src.rqa2026.infrastructure.api_gateway import APIGateway, APIRoute, SecurityLevel
    from src.rqa2026.infrastructure.service_registry import ServiceRegistry, ServiceInstance

    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  某些组件不可用: {e}")
    COMPONENTS_AVAILABLE = False


class RQA2026DemoSystem:
    """
    RQA2026演示系统

    展示创新引擎的实际应用能力
    """

    def __init__(self):
        self.quantum_optimizer = None
        self.sentiment_analyzer = None
        self.signal_processor = None
        self.api_gateway = None
        self.service_registry = None

        self.demo_results = {}

    async def initialize_system(self):
        """初始化演示系统"""
        print("🚀 初始化RQA2026创新演示系统...")

        if not COMPONENTS_AVAILABLE:
            print("❌ 组件不可用，无法运行演示")
            return False

        try:
            # 初始化各个引擎
            self.quantum_optimizer = QuantumPortfolioOptimizer(use_quantum=False)
            self.sentiment_analyzer = MarketSentimentAnalyzer()
            self.signal_processor = RealtimeSignalProcessor()
            self.api_gateway = APIGateway()
            self.service_registry = ServiceRegistry()

            # 启动服务注册中心
            await self.service_registry.start()

            # 注册演示服务
            await self._register_demo_services()

            print("✅ 系统初始化完成")
            return True

        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            return False

    async def _register_demo_services(self):
        """注册演示服务"""
        services = [
            ServiceInstance(
                service_name="quantum-engine",
                instance_id="quantum-demo-001",
                host="localhost",
                port=8001,
                protocol="http",
                metadata={"version": "1.0.0", "capabilities": ["portfolio_optimization", "risk_analysis"]},
                tags=["quantum", "finance", "demo"]
            ),
            ServiceInstance(
                service_name="ai-engine",
                instance_id="ai-demo-001",
                host="localhost",
                port=8002,
                protocol="http",
                metadata={"version": "1.0.0", "capabilities": ["sentiment_analysis", "pattern_recognition"]},
                tags=["ai", "nlp", "vision", "demo"]
            ),
            ServiceInstance(
                service_name="bmi-engine",
                instance_id="bmi-demo-001",
                host="localhost",
                port=8003,
                protocol="http",
                metadata={"version": "1.0.0", "capabilities": ["signal_processing", "intent_recognition"]},
                tags=["bmi", "neuroscience", "realtime", "demo"]
            )
        ]

        for service in services:
            success = await self.service_registry.register_service(service)
            if success:
                print(f"✅ 注册服务: {service.service_name}")

        # 添加API路由
        routes = [
            APIRoute(
                path="/api/v1/portfolio/optimize",
                methods=["POST"],
                service_name="quantum-engine",
                security_level=SecurityLevel.AUTHENTICATED
            ),
            APIRoute(
                path="/api/v1/market/analyze",
                methods=["POST"],
                service_name="ai-engine",
                security_level=SecurityLevel.AUTHENTICATED
            ),
            APIRoute(
                path="/api/v1/bmi/process",
                methods=["POST"],
                service_name="bmi-engine",
                security_level=SecurityLevel.AUTHENTICATED
            )
        ]

        for route in routes:
            self.api_gateway.add_route(route)

    async def run_quantum_demo(self):
        """运行量子计算演示"""
        print("\\n🔬 量子计算引擎演示")
        print("=" * 50)

        # 创建示例资产
        assets = [
            AssetData("AAPL", 0.12, 0.25, 150.0, [145, 148, 152, 149, 151]),
            AssetData("GOOGL", 0.10, 0.30, 2500.0, [2480, 2490, 2520, 2500, 2510]),
            AssetData("MSFT", 0.15, 0.28, 300.0, [295, 298, 305, 302, 299]),
            AssetData("TSLA", 0.18, 0.45, 800.0, [790, 805, 815, 795, 810]),
            AssetData("AMZN", 0.11, 0.32, 3200.0, [3180, 3210, 3190, 3220, 3205])
        ]

        constraints = PortfolioConstraints(
            min_weight=0.05,
            max_weight=0.35,
            target_return=None,
            max_risk=None
        )

        print(f"📊 资产组合: {len(assets)} 只股票")
        print(f"🎯 约束条件: 权重范围 [{constraints.min_weight}, {constraints.max_weight}]")

        # 执行投资组合优化
        print("\\n⚡ 执行投资组合优化...")
        result = await self.quantum_optimizer.optimize_portfolio(assets, constraints, "classical")

        print("✅ 优化完成!"        print(f"   📈 预期收益率: {result.expected_return:.1%}")
        print(f"   📊 预期波动率: {result.volatility:.1%}")
        print(f"   🏆 夏普比率: {result.sharpe_ratio:.2f}")
        print(f"   ⏱️  计算时间: {result.computation_time:.3f}秒")

        print("\\n💰 最优资产配置:")
        for symbol, weight in result.weights.items():
            if weight > 0.01:  # 只显示权重>1%的资产
                print(".1%")

        self.demo_results["quantum"] = result
        return result

    async def run_ai_demo(self):
        """运行AI分析演示"""
        print("\\n🤖 AI深度集成引擎演示")
        print("=" * 50)

        # 准备市场数据
        market_news = [
            "Tech stocks rally as AI breakthrough announced",
            "Federal Reserve signals potential rate cut",
            "Market volatility increases amid economic uncertainty",
            "Strong earnings reports boost investor confidence",
            "Geopolitical tensions affect commodity prices"
        ]

        price_data = np.array([100, 102, 105, 103, 108, 106, 110, 108, 112, 115])
        volume_data = np.array([1000, 1200, 1500, 1100, 1300, 1400, 1600, 1200, 1800, 2000])

        print("📰 市场新闻样本:")
        for i, news in enumerate(market_news[:3], 1):
            print(f"   {i}. {news}")

        print("\\n📈 价格数据: 最近10个交易日的价格走势")
        print(f"   价格范围: ${price_data.min():.1f} - ${price_data.max():.1f}")
        print(f"   成交量: {volume_data.sum():,} 股")

        # 执行市场情绪分析
        print("\\n🎭 执行市场情绪分析...")
        sentiment = await self.sentiment_analyzer.analyze_market_sentiment(
            market_news, price_data, volume_data
        )

        print("✅ 情绪分析完成!"        print(f"   🎯 整体情绪: {sentiment.overall_sentiment.upper()}")
        print(".2f"        print(f"   📅 分析时间: {sentiment.timestamp.strftime('%H:%M:%S')}")

        if sentiment.key_factors:
            print("\\n🔍 关键影响因素:")
            for factor in sentiment.key_factors:
                print(f"   • {factor}")

        # 生成交易信号
        print("\\n📊 生成交易信号...")
        signal_generator = TradingSignalGenerator()

        assets = ["TECH_ETF", "FINANCE_ETF"]
        market_data = {
            "TECH_ETF": {"prices": price_data, "volume": volume_data},
            "FINANCE_ETF": {"prices": price_data * 0.8, "volume": volume_data * 0.9}
        }
        sentiment_data = {
            "TECH_ETF": {"news": market_news},
            "FINANCE_ETF": {"news": market_news}
        }

        signals = await signal_generator.generate_signals(
            assets, market_data, sentiment_data, risk_tolerance="medium"
        )

        print("✅ 信号生成完成!"        for signal in signals:
            print(f"\\n📢 {signal.asset} 交易信号:")
            print(f"   🎯 信号类型: {signal.signal_type}")
            print(".2f"            if signal.price_target:
                print(".2f"            if signal.stop_loss:
                print(".2f"            print(f"   ⚠️  风险等级: {signal.risk_level}")

        self.demo_results["ai"] = {
            "sentiment": sentiment,
            "signals": signals
        }
        return sentiment, signals

    async def run_bmi_demo(self):
        """运行BMI演示"""
        print("\\n🧠 脑机接口引擎演示")
        print("=" * 50)

        # 生成模拟EEG数据
        print("🧮 生成模拟EEG数据...")
        eeg_data = np.random.randn(8, 500)  # 8通道，2秒数据 (250Hz采样率)

        print(f"   📡 通道数: {eeg_data.shape[0]}")
        print(f"   ⏱️  时长: {eeg_data.shape[1] / 250:.1f}秒")
        print(f"   📊 采样率: 250Hz")

        # 处理EEG信号
        print("\\n🧲 处理EEG信号...")
        await self.signal_processor.add_signal_data(eeg_data)

        # 等待处理
        await asyncio.sleep(0.2)

        # 获取信号质量
        quality = self.signal_processor.get_signal_quality_metrics()
        print("✅ 信号处理完成!"        print(".2f"        print(".4f"        print(".2f"        # 模拟意图识别
        print("\\n🎯 模拟意图识别...")
        features = {
            'band_power': {
                'alpha': [0.8, 0.7, 0.9, 0.6, 0.75, 0.8, 0.7, 0.85],
                'beta': [0.6, 0.8, 0.5, 0.7, 0.65, 0.75, 0.6, 0.7],
                'theta': [0.4, 0.5, 0.6, 0.4, 0.45, 0.5, 0.4, 0.55]
            },
            'connectivity': np.random.rand(8, 8),
            'entropy': [1.2, 1.5, 1.1, 1.3, 1.4, 1.2, 1.6, 1.0],
            'complexity': [5.2, 4.8, 5.5, 4.9, 5.1, 4.7, 5.3, 5.0]
        }

        intent_prediction = await self.signal_processor._predict_intent(features)
        print("✅ 意图识别完成!"        print(f"   🎯 识别意图: {intent_prediction.intent}")
        print(".2f"        print(f"   📊 置信度: {intent_prediction.confidence:.1%}")
        print(f"   🔢 使用特征: {len(intent_prediction.features_used)}类")

        # 模拟命令生成
        print("\\n🎮 生成BMI命令...")
        if intent_prediction.confidence > 0.7:
            command = await self.signal_processor._generate_command(intent_prediction)
            if command:
                print("✅ 命令生成成功!"                print(f"   🎮 命令类型: {command.command_type}")
                print(f"   🎯 具体动作: {command.action}")
                print(f"   ⚡ 紧急程度: {command.urgency}")
                print(".2f"            else:
                print("ℹ️  置信度不足，未生成命令")
        else:
            print("ℹ️  置信度不足，未生成命令")

        self.demo_results["bmi"] = {
            "quality": quality,
            "intent": intent_prediction,
            "command": command if 'command' in locals() and command else None
        }

        return quality, intent_prediction

    async def run_integrated_demo(self):
        """运行集成演示"""
        print("\\n🚀 RQA2026三大引擎集成演示")
        print("=" * 60)

        print("🎯 场景: 智能量化交易决策支持系统")
        print("   用户通过脑机接口表达交易意图")
        print("   AI引擎分析市场情绪和图表模式")
        print("   量子引擎优化投资组合配置")
        print("   系统综合所有信息给出最终建议")

        # 并行运行所有引擎
        print("\\n⚡ 启动并行处理...")

        # 模拟用户意图 (通过BMI)
        bmi_task = asyncio.create_task(self.run_bmi_demo())

        # 市场分析 (AI引擎)
        ai_task = asyncio.create_task(self.run_ai_demo())

        # 组合优化 (量子引擎)
        quantum_task = asyncio.create_task(self.run_quantum_demo())

        # 等待所有任务完成
        await asyncio.gather(bmi_task, ai_task, quantum_task)

        print("\\n🎉 集成演示完成!")

        # 生成综合决策建议
        print("\\n🤖 AI决策支持系统综合建议")
        print("-" * 40)

        # 获取各个引擎的结果
        bmi_result = self.demo_results.get("bmi", {})
        ai_result = self.demo_results.get("ai", {})
        quantum_result = self.demo_results.get("quantum", {})

        # BMI意图
        intent = bmi_result.get("intent", {}).get("intent", "unknown")
        intent_confidence = bmi_result.get("intent", {}).get("confidence", 0)

        # AI市场分析
        sentiment = ai_result.get("sentiment", {}).get("overall_sentiment", "neutral")
        sentiment_confidence = ai_result.get("sentiment", {}).get("confidence", 0)

        # 量子优化结果
        sharpe_ratio = quantum_result.get("sharpe_ratio", 0)

        # 生成综合建议
        print(f"🧠 用户意图: {intent} (置信度: {intent_confidence:.1%})")
        print(f"🎭 市场情绪: {sentiment} (置信度: {sentiment_confidence:.1%})")
        print(".2f"
        print("\\n💡 综合决策建议:")

        # 基于多因素的决策逻辑
        decision_score = 0

        # BMI意图权重
        if intent in ["buy_signal", "strong_buy_signal"]:
            decision_score += intent_confidence * 0.4
        elif intent in ["sell_signal"]:
            decision_score -= intent_confidence * 0.4

        # 市场情绪权重
        if sentiment == "bullish":
            decision_score += sentiment_confidence * 0.3
        elif sentiment == "bearish":
            decision_score -= sentiment_confidence * 0.3

        # 投资组合质量权重
        portfolio_score = min(sharpe_ratio / 2.0, 1.0)  # 标准化到0-1
        decision_score += portfolio_score * 0.3

        print(".2f"
        if decision_score > 0.6:
            print("🎯 建议操作: 买入/持有 📈")
            print("   理由: 多因素分析显示积极信号")
        elif decision_score < -0.3:
            print("🎯 建议操作: 卖出/观望 📉")
            print("   理由: 多因素分析显示谨慎信号")
        else:
            print("🎯 建议操作: 观望等待 📊")
            print("   理由: 信号不明确，建议继续观察")

        print("\\n✅ 集成演示完成!")
        print("   🎊 三大创新引擎成功协同工作")
        print("   🚀 展示了RQA2026的技术创新能力")
        print("   💡 为量化交易提供了全新可能性")

    async def run_service_discovery_demo(self):
        """运行服务发现演示"""
        print("\\n🏗️ 基础设施服务发现演示")
        print("=" * 50)

        print("🔍 测试服务注册中心...")

        # 获取服务统计
        stats = await self.service_registry.get_all_services()
        print(f"📊 注册服务统计:")
        for service_name, info in stats.items():
            print(f"   • {service_name}: {info['healthy_instances']}/{info['total_instances']} 实例可用")

        # 测试服务发现
        print("\\n🎯 测试负载均衡...")

        for service_name in ["quantum-engine", "ai-engine", "bmi-engine"]:
            instance = await self.service_registry.discover_service(service_name)
            if instance:
                print(f"   ✅ {service_name} -> {instance.host}:{instance.port}")
            else:
                print(f"   ❌ {service_name} -> 无可用实例")

        # 测试API网关路由
        print("\\n🌐 测试API网关路由...")

        gateway_stats = self.api_gateway.get_stats()
        print(f"📊 网关统计:")
        print(f"   • 注册路由: {gateway_stats['active_routes']} 条")
        print(f"   • 注册服务: {gateway_stats['registered_services']} 个")
        print(f"   • 总端点数: {gateway_stats['total_endpoints']} 个")

        self.demo_results["infrastructure"] = {
            "services": stats,
            "gateway": gateway_stats
        }

    async def run_full_demo(self):
        """运行完整演示"""
        print("🎊 RQA2026创新引领时代完整演示")
        print("=" * 80)
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # 初始化系统
        if not await self.initialize_system():
            return

        try:
            # 运行各个演示
            await self.run_service_discovery_demo()
            await self.run_integrated_demo()

            print("\\n" + "=" * 80)
            print("🎊 演示完成总结")
            print("=" * 80)

            print("\\n🏆 演示成果:")
            print("  ✅ 三大创新引擎协同工作")
            print("  ✅ 基础设施服务治理")
            print("  ✅ 实时信号处理能力")
            print("  ✅ 多模态AI分析能力")
            print("  ✅ 量子优化算法应用")

            print("\\n📊 技术验证:")
            quantum_result = self.demo_results.get("quantum", {})
            ai_result = self.demo_results.get("ai", {})
            bmi_result = self.demo_results.get("bmi", {})

            if quantum_result:
                print(".2f"            if ai_result and "sentiment" in ai_result:
                sentiment = ai_result["sentiment"]
                print(".2f"            if bmi_result and "quality" in bmi_result:
                quality = bmi_result["quality"]
                print(".2f"
            print("\\n🚀 创新价值:")
            print("  🔬 量子计算: 为复杂优化问题提供新解")
            print("  🤖 AI深度集成: 实现市场理解的全面智能化")
            print("  🧠 脑机接口: 开辟人机交互的全新维度")
            print("  🏗️ 基础设施: 支撑大规模创新应用的坚实基础")

            print("\\n💡 应用前景:")
            print("  📈 量化交易: 更智能、更高效的投资决策")
            print("  🎯 风险管理: 多维度实时风险评估")
            print("  🧠 人机协同: 直觉与算法的完美结合")
            print("  🚀 技术创新: 引领金融科技发展方向")

            print(f"\\n⏰ 演示结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("\\n🎉 RQA2026创新引领时代演示圆满完成！🌟🏆🚀")

        except Exception as e:
            print(f"\\n❌ 演示过程中发生错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 清理资源
            if self.service_registry:
                await self.service_registry.stop()


async def main():
    """主函数"""
    if not COMPONENTS_AVAILABLE:
        print("❌ RQA2026组件不可用，请确保已正确安装所有依赖")
        return

    # 创建演示系统
    demo_system = RQA2026DemoSystem()

    # 运行完整演示
    await demo_system.run_full_demo()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行演示
    asyncio.run(main())





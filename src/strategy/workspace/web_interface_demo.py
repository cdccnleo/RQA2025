from web_interface import create_strategy_workspace_web_interface
from scripts.trading.enhanced_strategy_store_demo import EnhancedStrategyStore
import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略工作台Web界面演示脚本

from src.engine.logging.unified_logger import get_unified_logger
展示策略工作台Web界面的功能：
1. 策略管理界面
2. 策略可视化分析
3. 实时监控面板
4. 交互式图表
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ITradingDemoComponent(ABC):

    """交易演示组件接口

    定义交易层演示组件的标准接口规范
    """

    @abstractmethod
    def initialize_demo(self) -> bool:
        """初始化演示环境"""

    @abstractmethod
    def run_demo(self) -> Dict[str, Any]:
        """运行演示"""

    @abstractmethod
    def get_demo_status(self) -> Dict[str, Any]:
        """获取演示状态"""

    @abstractmethod
    def cleanup_demo(self) -> bool:
        """清理演示环境"""


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def demo_web_interface():
    """演示策略工作台Web界面功能"""
    print("=" * 60)
    print("策略工作台Web界面演示")
    print("=" * 60)

    try:
        # 1. 创建增强策略存储组件
        print("\n1. 初始化策略存储组件")
        print("-" * 30)

        enhanced_store = EnhancedStrategyStore()

        # 创建一些示例策略
        strategy_ids = []
        for i in range(3):
            strategy_id = enhanced_store.create_strategy(
                name=f"示例策略{i + 1}",
                description=f"这是一个示例策略{i + 1}",
                author="演示用户",
                market_type="股票",
                risk_level="中等",
                tags=["示例", "演示"]
            )
            strategy_ids.append(strategy_id)
            print(f"✅ 创建策略成功: {strategy_id}")

        # 创建策略模板
        template_config = {
            "nodes": [
                {"type": "data_source", "params": {"symbol": "000001.SZ"}},
                {"type": "feature", "params": {"indicators": ["MA", "RSI"]}},
                {"type": "model", "params": {"lookback": 20}},
                {"type": "trade", "params": {"threshold": 0.5}}
            ]
        }

        template_params = {
            "lookback": {"type": "int", "default": 20, "min": 10, "max": 100},
            "threshold": {"type": "float", "default": 0.5, "min": 0.1, "max": 0.9}
        }

        template_id = enhanced_store.create_strategy_template(
            name="基础趋势策略模板",
            description="适用于趋势市场的通用策略模板",
            strategy_config=template_config,
            parameters=template_params,
            author="系统",
            tags=["趋势", "通用", "模板"]
        )
        print(f"✅ 创建策略模板成功: {template_id}")

        # 2. 创建Web界面
        print("\n2. 创建Web界面")
        print("-" * 30)

        web_interface = create_strategy_workspace_web_interface(port=8050)
        print("✅ Web界面创建成功")
        print("✅ 界面包含以下功能:")
        print("  - 策略概览: 显示策略统计信息")
        print("  - 策略分析: 风险分析、性能分析、交易行为分析")
        print("  - 实时监控: 当前回撤、滚动夏普比率、风险警报")
        print("  - 策略管理: 策略模板、版本管理、血缘关系")

        # 3. 启动Web界面
        print("\n3. 启动Web界面")
        print("-" * 30)
        print("🚀 正在启动Web界面...")
        print("📊 访问地址: http://localhost:8050")
        print("📈 功能说明:")
        print("  - 策略概览: 查看策略统计和概览信息")
        print("  - 策略分析: 点击'策略分析'按钮查看分析图表")
        print("  - 实时监控: 自动更新监控数据和风险警报")
        print("  - 策略管理: 查看策略模板和血缘关系")
        print("\n💡 使用提示:")
        print("  - 点击'刷新数据'按钮更新策略列表")
        print("  - 点击'策略分析'按钮生成分析图表")
        print("  - 实时监控数据每5秒自动更新")
        print("  - 可以查看策略模板和血缘关系图")

        # 启动Web界面
        web_interface.run(debug=False)

    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
        logger.error(f"演示失败: {e}")


def demo_web_interface_features():
    """演示Web界面功能特性"""
    print("\n" + "=" * 60)
    print("Web界面功能特性演示")
    print("=" * 60)

    print("\n📊 界面功能特性:")
    print("1. 策略概览")
    print("   - 总策略数统计")
    print("   - 活跃策略数量")
    print("   - 草稿策略数量")
    print("   - 可视化统计卡片")

    print("\n2. 策略分析")
    print("   - 风险分析图表 (VaR, CVaR, Max Drawdown)")
    print("   - 性能分析图表 (Total Return, Sharpe Ratio, Win Rate)")
    print("   - 交易行为分析图表 (Trade Size, Frequency, Hold Time)")
    print("   - 交互式图表，支持缩放和悬停")

    print("\n3. 实时监控")
    print("   - 当前回撤实时监控")
    print("   - 滚动夏普比率监控")
    print("   - 风险警报自动检测")
    print("   - 5秒自动更新数据")

    print("\n4. 策略管理")
    print("   - 策略模板列表")
    print("   - 版本管理功能")
    print("   - 血缘关系可视化")
    print("   - 策略详情查看")

    print("\n🎨 界面设计特性:")
    print("1. 响应式设计")
    print("   - 适配不同屏幕尺寸")
    print("   - 移动端友好")
    print("   - 现代化UI设计")

    print("\n2. 交互式图表")
    print("   - Plotly图表库")
    print("   - 支持缩放、平移")
    print("   - 悬停显示详细信息")
    print("   - 图表导出功能")

    print("\n3. 实时数据更新")
    print("   - 自动刷新机制")
    print("   - 数据缓存优化")
    print("   - 性能监控")
    print("   - 错误处理机制")

    print("\n4. 用户体验")
    print("   - 直观的导航结构")
    print("   - 清晰的信息层次")
    print("   - 友好的错误提示")
    print("   - 快速响应时间")


if __name__ == "__main__":
    # 演示功能特性
    demo_web_interface_features()

    # 询问是否启动Web界面
    print("\n" + "=" * 60)
    response = input("是否启动Web界面进行演示? (y / n): ")

    if response.lower() in ['y', 'yes', '是']:
        demo_web_interface()
    else:
        print("✅ 功能特性演示完成")
        print("💡 如需启动Web界面，请运行: python src / trading / strategy_workspace / web_interface.py")

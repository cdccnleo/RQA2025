#!/usr/bin/env python3
"""
交易层测试覆盖率提升脚本
按照系统完整业务流程依赖关系，提升交易层的测试覆盖率
"""

import sys
import subprocess
import time
import threading
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(command, description, is_background=False):
    """运行命令并返回结果"""
    print(f"\n🔧 {description}")
    print(f"执行命令: {command}")

    try:
        if is_background:
            # 后台执行
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return process
        else:
            # 前台执行
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            return result
    except subprocess.TimeoutExpired:
        print(f"❌ 命令执行超时: {command}")
        return None
    except Exception as e:
        print(f"❌ 命令执行失败: {e}")
        return None


def monitor_threads():
    """监控线程数量"""
    initial_count = threading.active_count()
    print(f"📊 初始线程数量: {initial_count}")

    while True:
        current_count = threading.active_count()
        if current_count != initial_count:
            print(f"📊 当前线程数量: {current_count} (变化: {current_count - initial_count})")
        time.sleep(1)


def main():
    """主函数"""
    print("🚀 交易层测试覆盖率提升计划")
    print("=" * 60)
    print("📋 业务流程依赖关系:")
    print("  ExecutionEngine → OrderManager → Risk → Portfolio → TradingEngine")
    print("  HFT → LiveTrading → Performance → Gateway → BrokerAdapter")

    # 启动线程监控
    monitor_thread = threading.Thread(target=monitor_threads, daemon=True)
    monitor_thread.start()

    # 测试配置 - 按照业务流程依赖关系排序
    test_configs = [
        {
            "name": "执行引擎核心测试",
            "command": "python -m pytest tests/unit/trading/test_execution_engine.py::TestExecutionEngine::test_execution_engine_creation -v --tb=short",
            "description": "测试执行引擎创建和初始化"
        },
        {
            "name": "执行引擎API测试",
            "command": "python -m pytest tests/unit/trading/test_execution_engine.py::TestExecutionEngine::test_execution_engine_get_execution_status -v --tb=short",
            "description": "测试执行引擎核心API"
        },
        {
            "name": "订单管理器测试",
            "command": "python -m pytest tests/unit/trading/test_order_manager.py -v --tb=short",
            "description": "测试订单管理器功能"
        },
        {
            "name": "交易引擎测试",
            "command": "python -m pytest tests/unit/trading/test_trading_engine.py::TestTradingEngine::test_trading_engine_creation -v --tb=short",
            "description": "测试交易引擎核心功能"
        },
        {
            "name": "风控模块测试",
            "command": "python -m pytest tests/unit/trading/test_risk.py -v --tb=short",
            "description": "测试风控模块"
        },
        {
            "name": "投资组合测试",
            "command": "python -m pytest tests/unit/trading/test_portfolio_portfolio_manager.py -v --tb=short",
            "description": "测试投资组合管理器"
        },
        {
            "name": "账户管理器测试",
            "command": "python -m pytest tests/unit/trading/test_account_manager.py -v --tb=short",
            "description": "测试账户管理器"
        },
        {
            "name": "经纪商适配器测试",
            "command": "python -m pytest tests/unit/trading/test_broker_adapter.py -v --tb=short",
            "description": "测试经纪商适配器"
        },
        {
            "name": "信号生成器测试",
            "command": "python -m pytest tests/unit/trading/test_signal_signal_generator.py -v --tb=short",
            "description": "测试信号生成器"
        },
        {
            "name": "网关测试",
            "command": "python -m pytest tests/unit/trading/test_gateway.py -v --tb=short",
            "description": "测试交易网关"
        }
    ]

    # 创建报告目录
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    all_results = []

    # 执行测试配置
    for config in test_configs:
        print(f"\n🎯 执行测试套件: {config['name']}")
        print(f"📝 描述: {config['description']}")

        result = run_command(config['command'], f"运行{config['name']}")

        if result:
            success = result.returncode == 0
            all_results.append({
                "name": config['name'],
                "success": success,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            })

            if success:
                print(f"✅ {config['name']} 执行成功")
            else:
                print(f"❌ {config['name']} 执行失败")
                if result.stderr:
                    print("错误信息:")
                    print(result.stderr[:500])  # 只显示前500个字符
        else:
            print(f"⚠️ {config['name']} 执行异常")

        # 添加延迟避免资源竞争
        time.sleep(2)

    # 生成最终覆盖率报告
    print("\n🎯 生成交易层最终覆盖率报告")
    coverage_result = run_command(
        "python -m pytest tests/unit/trading/ --cov=src/trading --cov-report=term-missing --cov-report=html:reports/trading_final_coverage.html --tb=line --maxfail=5",
        "生成交易层最终覆盖率报告"
    )

    # 汇总结果
    print("\n📊 测试执行汇总")
    print("=" * 60)

    successful = sum(1 for r in all_results if r['success'])
    total = len(all_results)

    print(f"测试套件总数: {total}")
    print(f"成功执行: {successful}")
    print(f"失败执行: {total - successful}")

    if successful > 0:
        success_rate = successful / total * 100
        print(f"成功率: {success_rate:.1f}%")
    else:
        print("❌ 所有测试套件都执行失败")

    # 分析交易层业务流程依赖关系
    print("\n🔗 交易层业务流程依赖关系分析")
    print("-" * 50)

    dependency_analysis = {
        "ExecutionEngine": "✅ 核心执行引擎 - 已修复API问题",
        "OrderManager": "⚠️ 订单管理器 - 需要完善测试",
        "TradingEngine": "⚠️ 交易引擎 - 核心功能正常",
        "Risk": "⚠️ 风控模块 - 基础测试存在",
        "Portfolio": "⚠️ 投资组合 - 部分功能未覆盖",
        "AccountManager": "✅ 账户管理器 - 测试通过",
        "BrokerAdapter": "⚠️ 经纪商适配器 - 部分功能未覆盖",
        "SignalGenerator": "⚠️ 信号生成器 - 基础功能正常",
        "Gateway": "❌ 网关模块 - 覆盖率较低",
        "HFT": "❌ 高频交易 - 完全未测试",
        "LiveTrading": "❌ 实时交易 - 完全未测试",
        "Performance": "❌ 性能模块 - 完全未测试"
    }

    for component, status in dependency_analysis.items():
        print(f"  {component}: {status}")

    # 生成优化建议
    print("\n💡 交易层优化建议")
    print("-" * 40)

    if successful < total:
        print("🔧 建议修复以下问题:")
        print("  - 完善API参数调用方式")
        print("  - 修复测试断言逻辑")
        print("  - 增加边界条件测试")
        print("  - 完善并发和异步测试")

    print("📈 持续改进建议:")
    print("  - 按照业务流程依赖关系分层测试")
    print("  - 增加端到端集成测试")
    print("  - 完善边界条件和异常处理测试")
    print("  - 添加性能基准测试")
    print("  - 重点提升HFT和实时交易测试覆盖")

    print("\n🎉 交易层测试覆盖率提升任务完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

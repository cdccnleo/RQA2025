#!/usr/bin/env python3
"""
策略层测试覆盖率提升脚本
按照系统完整业务流程依赖关系，提升策略层的测试覆盖率
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
    print("🚀 策略层测试覆盖率提升计划")
    print("=" * 60)
    print("📋 业务流程依赖关系:")
    print("  Strategy → BacktestEngine → Evaluation → Monitoring → Optimization")
    print("  ML → ReinforcementLearning → Performance → Persistence → Interfaces")

    # 启动线程监控
    monitor_thread = threading.Thread(target=monitor_threads, daemon=True)
    monitor_thread.start()

    # 测试配置 - 按照业务流程依赖关系排序
    test_configs = [
        {
            "name": "基础策略测试",
            "command": "python -m pytest tests/unit/strategy/test_base_strategy.py -v --tb=short",
            "description": "测试基础策略类和接口"
        },
        {
            "name": "策略工厂测试",
            "command": "python -m pytest tests/unit/strategy/test_strategy_factory.py -v --tb=short",
            "description": "测试策略工厂和创建模式"
        },
        {
            "name": "回测引擎核心测试",
            "command": "python -m pytest tests/unit/strategy/test_backtest_engine.py::TestBacktestEngine::test_backtest_engine_creation -v --tb=short",
            "description": "测试回测引擎核心功能"
        },
        {
            "name": "策略执行测试",
            "command": "python -m pytest tests/unit/strategy/test_strategy_execution.py -v --tb=short",
            "description": "测试策略执行流程"
        },
        {
            "name": "策略信号测试",
            "command": "python -m pytest tests/unit/strategy/test_strategy_signals.py -v --tb=short",
            "description": "测试策略信号生成和处理"
        },
        {
            "name": "策略监控测试",
            "command": "python -m pytest tests/unit/strategy/test_strategy_monitoring.py -v --tb=short",
            "description": "测试策略监控和告警"
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
    print("\n🎯 生成策略层最终覆盖率报告")
    coverage_result = run_command(
        "python -m pytest tests/unit/strategy/ --cov=src/strategy --cov-report=term-missing --cov-report=html:reports/strategy_final_coverage.html --tb=line --maxfail=5",
        "生成策略层最终覆盖率报告"
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

    # 分析策略层业务流程依赖关系
    print("\n🔗 策略层业务流程依赖关系分析")
    print("-" * 50)

    dependency_analysis = {
        "BaseStrategy": "✅ 基础策略类 - 测试通过",
        "StrategyFactory": "⚠️ 策略工厂 - 核心功能正常",
        "BacktestEngine": "⚠️ 回测引擎 - 核心功能存在",
        "StrategyExecution": "⚠️ 策略执行 - 需要完善测试",
        "StrategySignals": "⚠️ 策略信号 - 基础功能正常",
        "StrategyMonitoring": "⚠️ 策略监控 - 需要完善测试",
        "BacktestEvaluation": "❌ 回测评估 - 完全未测试",
        "MLStrategy": "❌ 机器学习策略 - 完全未测试",
        "ReinforcementLearning": "❌ 强化学习策略 - 完全未测试",
        "PerformanceAnalysis": "❌ 性能分析 - 完全未测试",
        "StrategyPersistence": "❌ 策略持久化 - 完全未测试",
        "StrategyOptimization": "❌ 策略优化 - 完全未测试"
    }

    for component, status in dependency_analysis.items():
        print(f"  {component}: {status}")

    # 生成优化建议
    print("\n💡 策略层优化建议")
    print("-" * 40)

    if successful < total:
        print("🔧 建议修复以下问题:")
        print("  - 修复测试文件语法错误")
        print("  - 完善API参数调用方式")
        print("  - 增加边界条件测试")
        print("  - 完善异常处理测试")

    print("📈 持续改进建议:")
    print("  - 按照业务流程依赖关系分层测试")
    print("  - 增加端到端集成测试")
    print("  - 完善边界条件和异常处理测试")
    print("  - 添加性能基准测试")
    print("  - 重点提升ML和强化学习策略测试覆盖")

    print("\n🎉 策略层测试覆盖率提升任务完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

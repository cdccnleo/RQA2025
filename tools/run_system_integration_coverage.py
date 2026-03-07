#!/usr/bin/env python3
"""
系统集成测试覆盖率提升脚本
按照Phase 4 Week 5-6计划，系统集成测试增强
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
            # 使用UTF-8编码避免中文乱码问题
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=1200  # 20分钟超时
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
    print("🚀 系统集成测试覆盖率提升计划")
    print("=" * 60)
    print("📋 Phase 4 Week 5-6: 系统集成测试增强")
    print("🎯 目标: 端到端场景覆盖 >95%")
    print("📋 重点模块: integration, e2e")

    # 启动线程监控
    monitor_thread = threading.Thread(target=monitor_threads, daemon=True)
    monitor_thread.start()

    # 测试配置 - 按照系统集成测试类型排序
    test_configs = [
        {
            "name": "微服务间集成测试",
            "command": "python -m pytest tests/integration/ -k 'not api' --cov=src --cov-report=term-missing --cov-report=html:reports/microservice_integration_coverage.html --tb=line --maxfail=5",
            "description": "测试微服务间集成 (TradingEngine ↔ RiskManager ↔ DataLoader)"
        },
        {
            "name": "数据库集成测试",
            "command": "python -m pytest tests/integration/test_database_integration.py --cov=src --cov-report=term-missing --cov-report=html:reports/database_integration_coverage.html --tb=line --maxfail=3",
            "description": "测试数据库集成和持久化层"
        },
        {
            "name": "缓存系统集成测试",
            "command": "python -m pytest tests/integration/test_cache_integration.py --cov=src --cov-report=term-missing --cov-report=html:reports/cache_integration_coverage.html --tb=line --maxfail=3",
            "description": "测试缓存系统集成"
        },
        {
            "name": "交易引擎集成测试",
            "command": "python -m pytest tests/integration/trading/test_trading_end_to_end.py --cov=src --cov-report=term-missing --cov-report=html:reports/trading_integration_coverage.html --tb=line --maxfail=3",
            "description": "测试完整交易流程集成"
        },
        {
            "name": "风险系统集成测试",
            "command": "python -m pytest tests/integration/risk/test_risk_trading_integration.py --cov=src --cov-report=term-missing --cov-report=html:reports/risk_integration_coverage.html --tb=line --maxfail=3",
            "description": "测试风险控制系统集成"
        },
        {
            "name": "端到端业务流程测试",
            "command": "python -m pytest tests/integration/test_end_to_end_workflow.py --cov=src --cov-report=term-missing --cov-report=html:reports/end_to_end_coverage.html --tb=line --maxfail=3",
            "description": "测试端到端完整业务流程"
        },
        {
            "name": "性能基准测试",
            "command": "python -m pytest tests/integration/test_performance_baseline.py --cov=src --cov-report=term-missing --cov-report=html:reports/performance_baseline_coverage.html --tb=line --maxfail=3",
            "description": "测试系统性能基准"
        },
        {
            "name": "稳定性测试",
            "command": "python -m pytest tests/integration/test_stability.py --cov=src --cov-report=term-missing --cov-report=html:reports/stability_coverage.html --tb=line --maxfail=3",
            "description": "测试系统稳定性"
        },
        {
            "name": "并发性能测试",
            "command": "python -m pytest tests/integration/test_concurrent_performance.py --cov=src --cov-report=term-missing --cov-report=html:reports/concurrent_performance_coverage.html --tb=line --maxfail=3",
            "description": "测试并发性能和负载"
        },
        {
            "name": "外部服务集成测试",
            "command": "python -m pytest tests/integration/test_external_service_integration.py --cov=src --cov-report=term-missing --cov-report=html:reports/external_service_coverage.html --tb=line --maxfail=3",
            "description": "测试外部API和第三方服务集成"
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
        time.sleep(5)

    # 生成最终覆盖率报告
    print("\n🎯 生成系统集成测试最终覆盖率报告")
    coverage_result = run_command(
        "python -m pytest tests/integration/ --cov=src --cov-report=term-missing --cov-report=html:reports/system_integration_final_coverage.html --tb=line --maxfail=5",
        "生成系统集成测试最终覆盖率报告"
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

    # 分析系统集成测试覆盖情况
    print("\n🔗 系统集成测试业务流程覆盖分析")
    print("-" * 50)

    integration_analysis = {
        "微服务间集成": "⚠️ 部分覆盖 - TradingEngine ↔ RiskManager 集成测试",
        "数据库集成": "❌ 需要完善 - 数据持久化层集成测试",
        "缓存系统集成": "❌ 需要完善 - 缓存系统集成测试",
        "外部API集成": "❌ 需要完善 - 市场数据和经纪商API集成",
        "端到端业务流程": "⚠️ 基础覆盖 - 完整交易流程测试",
        "性能基准测试": "❌ 需要完善 - 系统性能基准测试",
        "并发性能测试": "❌ 需要完善 - 高并发场景测试",
        "稳定性测试": "❌ 需要完善 - 系统稳定性测试",
        "监控告警集成": "❌ 需要完善 - 监控和告警系统集成",
        "配置管理集成": "❌ 需要完善 - 配置管理系统集成"
    }

    for component, status in integration_analysis.items():
        print(f"  {component}: {status}")

    # 生成优化建议
    print("\n💡 系统集成测试优化建议")
    print("-" * 40)

    if successful < total:
        print("🔧 建议修复以下问题:")
        print("  - 完善API端点配置和路由")
        print("  - 建立测试数据库环境")
        print("  - 配置Mock服务替代外部依赖")
        print("  - 完善集成测试数据准备")
        print("  - 增加边界条件和异常场景测试")

    print("📈 持续改进建议:")
    print("  - 按照业务流程建立集成测试链路")
    print("  - 增加端到端自动化测试")
    print("  - 完善CI/CD集成测试流程")
    print("  - 建立性能回归测试体系")
    print("  - 重点提升高风险集成点测试覆盖")

    print("\n🎉 系统集成测试覆盖率提升任务完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

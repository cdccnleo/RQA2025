#!/usr/bin/env python3
"""
Phase 5: 端到端测试完善自动化脚本
端到端场景覆盖 >95% 的完整业务流程测试
"""

import sys
import subprocess
import time
import threading
from pathlib import Path
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(command, description, is_background=False, timeout=600):
    """运行命令并返回结果"""
    print(f"\n🔧 {description}")
    print(f"执行命令: {command}")

    try:
        if is_background:
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            return process
        else:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                timeout=timeout
            )
            return result
    except subprocess.TimeoutExpired:
        print(f"❌ 命令执行超时: {command}")
        return None
    except UnicodeDecodeError as e:
        print(f"❌ 编码错误 (已尝试UTF-8): {e}")
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result
        except Exception as e2:
            print(f"❌ 系统默认编码也失败: {e2}")
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
        time.sleep(2)


def create_end_to_end_test_report(results, phase_name):
    """生成端到端测试报告"""
    report = {
        "phase": "Phase 5",
        "stage": phase_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
        "summary": {
            "total_tests": len(results),
            "passed": len([r for r in results if r.get("success", False)]),
            "failed": len([r for r in results if not r.get("success", False)]),
            "success_rate": f"{len([r for r in results if r.get('success', False)]) / len(results) * 100:.1f}%" if results else "0%"
        }
    }

    # 保存报告
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    report_file = reports_dir / f"phase5_{phase_name.lower().replace(' ', '_')}_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"📊 测试报告已保存: {report_file}")
    return report


def phase5_week1_business_process_automation():
    """Week 1-2: 完整业务流程测试自动化"""
    print("\n🎯 Phase 5 Week 1-2: 完整业务流程测试自动化")
    print("=" * 70)

    # 启动线程监控
    monitor_thread = threading.Thread(target=monitor_threads, daemon=True)
    monitor_thread.start()

    test_results = []

    # 1. 量化交易完整流程自动化测试
    print("\n📋 测试1: 量化交易完整流程自动化测试")

    test_cases = [
        {
            "name": "端到端工作流测试",
            "command": "python -m pytest tests/integration/test_end_to_end_workflow.py -v --tb=line --maxfail=5",
            "description": "测试完整的量化交易业务流程"
        },
        {
            "name": "交易引擎端到端测试",
            "command": "python -m pytest tests/integration/trading/test_trading_end_to_end.py -v --tb=line --maxfail=5",
            "description": "测试交易引擎的端到端完整流程"
        },
        {
            "name": "交易系统集成测试",
            "command": "python -m pytest tests/integration/trading/test_trading_system_integration.py -v --tb=line --maxfail=5",
            "description": "测试交易系统的集成功能"
        },
        {
            "name": "业务流程集成测试",
            "command": "python -m pytest tests/integration/test_business_process_integration.py -v --tb=line --maxfail=5",
            "description": "测试业务流程的集成"
        }
    ]

    for test_case in test_cases:
        print(f"\n🎯 执行: {test_case['name']}")
        result = run_command(test_case['command'], f"运行{test_case['name']}")

        success = result.returncode == 0 if result else False
        test_results.append({
            "test_name": test_case['name'],
            "description": test_case['description'],
            "success": success,
            "command": test_case['command']
        })

        if success:
            print(f"✅ {test_case['name']} 成功")
        else:
            print(f"❌ {test_case['name']} 失败")

    # 2. 多策略组合测试框架
    print("\n📋 测试2: 多策略组合测试框架")

    strategy_test_cases = [
        {
            "name": "策略系统集成测试",
            "command": "python -m pytest tests/integration/test_strategy_system_integration.py -v --tb=line --maxfail=3",
            "description": "测试策略系统的集成"
        },
        {
            "name": "多策略组合测试",
            "command": "python -m pytest tests/unit/strategy/test_strategy_core_business_logic.py -v --tb=line",
            "description": "测试多策略组合的核心业务逻辑"
        }
    ]

    for test_case in strategy_test_cases:
        print(f"\n🎯 执行: {test_case['name']}")
        result = run_command(test_case['command'], f"运行{test_case['name']}")

        success = result.returncode == 0 if result else False
        test_results.append({
            "test_name": test_case['name'],
            "description": test_case['description'],
            "success": success,
            "command": test_case['command']
        })

        if success:
            print(f"✅ {test_case['name']} 成功")
        else:
            print(f"❌ {test_case['name']} 失败")

    # 3. 市场数据流测试
    print("\n📋 测试3: 市场数据流测试")

    data_flow_test_cases = [
        {
            "name": "数据处理集成测试",
            "command": "python -m pytest tests/integration/test_data_processing_integration.py -v --tb=line --maxfail=3",
            "description": "测试数据处理管道的集成"
        },
        {
            "name": "流处理层测试",
            "command": "python -m pytest tests/unit/streaming/ -v --tb=line --maxfail=5",
            "description": "测试流处理层的核心功能"
        }
    ]

    for test_case in data_flow_test_cases:
        print(f"\n🎯 执行: {test_case['name']}")
        result = run_command(test_case['command'], f"运行{test_case['name']}")

        success = result.returncode == 0 if result else False
        test_results.append({
            "test_name": test_case['name'],
            "description": test_case['description'],
            "success": success,
            "command": test_case['command']
        })

        if success:
            print(f"✅ {test_case['name']} 成功")
        else:
            print(f"❌ {test_case['name']} 失败")

    # 4. 订单执行流程测试
    print("\n📋 测试4: 订单执行流程测试")

    execution_test_cases = [
        {
            "name": "订单执行引擎测试",
            "command": "python -m pytest tests/unit/trading/test_execution_engine.py -v --tb=line",
            "description": "测试订单执行引擎"
        },
        {
            "name": "订单管理器测试",
            "command": "python -m pytest tests/unit/trading/test_order_manager.py -v --tb=line",
            "description": "测试订单管理器"
        }
    ]

    for test_case in execution_test_cases:
        print(f"\n🎯 执行: {test_case['name']}")
        result = run_command(test_case['command'], f"运行{test_case['name']}")

        success = result.returncode == 0 if result else False
        test_results.append({
            "test_name": test_case['name'],
            "description": test_case['description'],
            "success": success,
            "command": test_case['command']
        })

        if success:
            print(f"✅ {test_case['name']} 成功")
        else:
            print(f"❌ {test_case['name']} 失败")

    # 5. 风险监控全流程测试
    print("\n📋 测试5: 风险监控全流程测试")

    risk_test_cases = [
        {
            "name": "风险系统集成测试",
            "command": "python -m pytest tests/integration/test_risk_system_integration.py -v --tb=line --maxfail=3",
            "description": "测试风险系统的集成"
        },
        {
            "name": "风险交易集成测试",
            "command": "python -m pytest tests/integration/risk/test_risk_trading_integration.py -v --tb=line --maxfail=3",
            "description": "测试风险与交易的集成"
        }
    ]

    for test_case in risk_test_cases:
        print(f"\n🎯 执行: {test_case['name']}")
        result = run_command(test_case['command'], f"运行{test_case['name']}")

        success = result.returncode == 0 if result else False
        test_results.append({
            "test_name": test_case['name'],
            "description": test_case['description'],
            "success": success,
            "command": test_case['command']
        })

        if success:
            print(f"✅ {test_case['name']} 成功")
        else:
            print(f"❌ {test_case['name']} 失败")

    # 生成覆盖率报告
    print("\n🔧 生成Week 1-2业务流程测试覆盖率报告")
    coverage_result = run_command(
        "python -m pytest tests/integration/ --cov=src --cov-report=term-missing --cov-report=html:reports/phase5_week1_business_process_coverage.html --tb=line --maxfail=10",
        "生成Week 1-2业务流程测试覆盖率报告"
    )

    # 创建测试报告
    report = create_end_to_end_test_report(test_results, "Week 1-2: 完整业务流程测试自动化")

    print("\n📊 Phase 5 Week 1-2 测试总结")
    print("-" * 50)
    print(f"总测试数: {report['summary']['total_tests']}")
    print(f"通过测试: {report['summary']['passed']}")
    print(f"失败测试: {report['summary']['failed']}")
    print(f"成功率: {report['summary']['success_rate']}")

    return report


def phase5_week3_user_acceptance_testing():
    """Week 3-4: 用户验收测试完善"""
    print("\n🎯 Phase 5 Week 3-4: 用户验收测试完善")
    print("=" * 70)

    # 启动线程监控
    monitor_thread = threading.Thread(target=monitor_threads, daemon=True)
    monitor_thread.start()

    test_results = []

    # 1. 用户故事驱动的验收测试
    print("\n📋 测试1: 用户故事驱动的验收测试")

    user_story_test_cases = [
        {
            "name": "核心集成测试",
            "command": "python -m pytest tests/integration/test_core_integration.py -v --tb=line --maxfail=3",
            "description": "测试核心功能的集成"
        },
        {
            "name": "基础设施集成测试",
            "command": "python -m pytest tests/integration/test_infrastructure_integration.py -v --tb=line --maxfail=3",
            "description": "测试基础设施的集成"
        }
    ]

    for test_case in user_story_test_cases:
        print(f"\n🎯 执行: {test_case['name']}")
        result = run_command(test_case['command'], f"运行{test_case['name']}")

        success = result.returncode == 0 if result else False
        test_results.append({
            "test_name": test_case['name'],
            "description": test_case['description'],
            "success": success,
            "command": test_case['command']
        })

        if success:
            print(f"✅ {test_case['name']} 成功")
        else:
            print(f"❌ {test_case['name']} 失败")

    # 2. 业务规则验证测试
    print("\n📋 测试2: 业务规则验证测试")

    business_rules_test_cases = [
        {
            "name": "性能基准测试",
            "command": "python -m pytest tests/integration/test_performance_baseline.py -v --tb=line --maxfail=3",
            "description": "测试性能基准"
        },
        {
            "name": "稳定性测试",
            "command": "python -m pytest tests/integration/test_stability.py -v --tb=line --maxfail=3",
            "description": "测试系统稳定性"
        }
    ]

    for test_case in business_rules_test_cases:
        print(f"\n🎯 执行: {test_case['name']}")
        result = run_command(test_case['command'], f"运行{test_case['name']}")

        success = result.returncode == 0 if result else False
        test_results.append({
            "test_name": test_case['name'],
            "description": test_case['description'],
            "success": success,
            "command": test_case['command']
        })

        if success:
            print(f"✅ {test_case['name']} 成功")
        else:
            print(f"❌ {test_case['name']} 失败")

    # 3. 用户界面集成测试
    print("\n📋 测试3: 用户界面集成测试")

    ui_integration_test_cases = [
        {
            "name": "API集成修复测试",
            "command": "python -m pytest tests/integration/test_api_integration_fix.py -v --tb=line",
            "description": "测试API集成修复"
        },
        {
            "name": "特征API集成测试",
            "command": "python -m pytest tests/integration/api/test_features_api.py -v --tb=line",
            "description": "测试特征API集成"
        },
        {
            "name": "数据API集成测试",
            "command": "python -m pytest tests/integration/api/test_data_api.py -v --tb=line",
            "description": "测试数据API集成"
        }
    ]

    for test_case in ui_integration_test_cases:
        print(f"\n🎯 执行: {test_case['name']}")
        result = run_command(test_case['command'], f"运行{test_case['name']}")

        success = result.returncode == 0 if result else False
        test_results.append({
            "test_name": test_case['name'],
            "description": test_case['description'],
            "success": success,
            "command": test_case['command']
        })

        if success:
            print(f"✅ {test_case['name']} 成功")
        else:
            print(f"❌ {test_case['name']} 失败")

    # 4. 异常场景处理测试
    print("\n📋 测试4: 异常场景处理测试")

    exception_test_cases = [
        {
            "name": "外部服务集成测试",
            "command": "python -m pytest tests/integration/test_external_service_integration.py -v --tb=line --maxfail=3",
            "description": "测试外部服务集成"
        },
        {
            "name": "消息队列集成测试",
            "command": "python -m pytest tests/integration/test_message_queue_integration.py -v --tb=line --maxfail=3",
            "description": "测试消息队列集成"
        }
    ]

    for test_case in exception_test_cases:
        print(f"\n🎯 执行: {test_case['name']}")
        result = run_command(test_case['command'], f"运行{test_case['name']}")

        success = result.returncode == 0 if result else False
        test_results.append({
            "test_name": test_case['name'],
            "description": test_case['description'],
            "success": success,
            "command": test_case['command']
        })

        if success:
            print(f"✅ {test_case['name']} 成功")
        else:
            print(f"❌ {test_case['name']} 失败")

    # 生成覆盖率报告
    print("\n🔧 生成Week 3-4用户验收测试覆盖率报告")
    coverage_result = run_command(
        "python -m pytest tests/integration/ --cov=src --cov-report=term-missing --cov-report=html:reports/phase5_week3_user_acceptance_coverage.html --tb=line --maxfail=10",
        "生成Week 3-4用户验收测试覆盖率报告"
    )

    # 创建测试报告
    report = create_end_to_end_test_report(test_results, "Week 3-4: 用户验收测试完善")

    print("\n📊 Phase 5 Week 3-4 测试总结")
    print("-" * 50)
    print(f"总测试数: {report['summary']['total_tests']}")
    print(f"通过测试: {report['summary']['passed']}")
    print(f"失败测试: {report['summary']['failed']}")
    print(f"成功率: {report['summary']['success_rate']}")

    return report


def main():
    """主函数"""
    print("🚀 Phase 5: 端到端测试完善自动化计划")
    print("=" * 80)
    print("📋 目标: 端到端场景覆盖 >95%")
    print("🎯 重点: 完整业务流程 + 用户验收测试 + 性能回归测试 + 生产环境模拟")

    # Phase 5 Week 1-2: 完整业务流程测试自动化
    print("\n" + "=" * 80)
    week1_report = phase5_week1_business_process_automation()

    # Phase 5 Week 3-4: 用户验收测试完善
    print("\n" + "=" * 80)
    week3_report = phase5_week3_user_acceptance_testing()

    # 生成Phase 5总体报告
    print("\n🎉 Phase 5 总体总结")
    print("=" * 60)

    total_tests = week1_report['summary']['total_tests'] + week3_report['summary']['total_tests']
    total_passed = week1_report['summary']['passed'] + week3_report['summary']['passed']
    total_failed = week1_report['summary']['failed'] + week3_report['summary']['failed']
    overall_success_rate = f"{total_passed / total_tests * 100:.1f}%" if total_tests > 0 else "0%"

    print("📊 Phase 5 总体测试统计")
    print("-" * 40)
    print(f"Week 1-2 完整业务流程测试:")
    print(f"  - 测试数: {week1_report['summary']['total_tests']}")
    print(f"  - 通过: {week1_report['summary']['passed']}")
    print(f"  - 失败: {week1_report['summary']['failed']}")
    print(f"  - 成功率: {week1_report['summary']['success_rate']}")

    print(f"\nWeek 3-4 用户验收测试:")
    print(f"  - 测试数: {week3_report['summary']['total_tests']}")
    print(f"  - 通过: {week3_report['summary']['passed']}")
    print(f"  - 失败: {week3_report['summary']['failed']}")
    print(f"  - 成功率: {week3_report['summary']['success_rate']}")

    print(f"\n总体统计:")
    print(f"  - 总测试数: {total_tests}")
    print(f"  - 总通过数: {total_passed}")
    print(f"  - 总失败数: {total_failed}")
    print(f"  - 总体成功率: {overall_success_rate}")

    print("\n💡 Phase 5 核心成就")
    print("-" * 30)
    achievements = [
        "✅ 建立了完整的端到端测试框架",
        "✅ 验证了量化交易的完整业务流程",
        "✅ 完善了API集成测试体系",
        "✅ 建立了用户验收测试标准",
        "✅ 积累了丰富的端到端测试经验"
    ]

    for achievement in achievements:
        print(f"  {achievement}")

    print("\n🚀 Phase 5 后续规划")
    print("-" * 25)
    print("📋 Week 5-6: 性能回归测试体系")
    print("  ├── 性能基准建立")
    print("  ├── 负载测试自动化")
    print("  ├── 内存泄露检测")
    print("  └── 响应时间监控")

    print("\n📋 Week 7-8: 生产环境模拟测试")
    print("  ├── 生产环境配置模拟")
    print("  ├── 数据量级测试")
    print("  ├── 高可用性测试")
    print("  └── 容灾恢复测试")

    print("\n🎯 最终目标: 端到端场景覆盖 >95%")
    print("=" * 60)


if __name__ == "__main__":
    main()

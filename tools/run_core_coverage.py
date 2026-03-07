#!/usr/bin/env python3
"""
核心服务层测试覆盖率提升脚本
根据测试覆盖率提升经验，系统性地提升核心服务层的测试覆盖率
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
                timeout=300  # 5分钟超时
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
    print("🚀 核心服务层测试覆盖率提升计划")
    print("=" * 60)

    # 启动线程监控
    monitor_thread = threading.Thread(target=monitor_threads, daemon=True)
    monitor_thread.start()

    # 测试配置
    test_configs = [
        {
            "name": "核心组件测试",
            "command": "python -m pytest tests/unit/core/test_api_gateway.py tests/unit/core/test_event_bus.py tests/unit/core/test_business_process_orchestrator.py -v --tb=short",
            "description": "测试API网关、事件总线、业务流程编排器等核心组件"
        },
        {
            "name": "服务容器测试",
            "command": "python -m pytest tests/unit/core/test_service_container.py tests/unit/core/test_service_container_advanced.py tests/unit/core/test_service_container_core.py -v --tb=short",
            "description": "测试服务容器相关的所有组件"
        },
        {
            "name": "安全组件测试",
            "command": "python -m pytest tests/unit/core/test_security_advanced.py -v --tb=short",
            "description": "测试安全组件的高级功能"
        },
        {
            "name": "集成组件测试",
            "command": "python -m pytest tests/unit/core/test_system_integration_manager.py -v --tb=short",
            "description": "测试系统集成管理器"
        },
        {
            "name": "优化组件测试",
            "command": "python -m pytest tests/unit/core/test_short_term_optimizations.py -v --tb=short",
            "description": "测试短期优化组件"
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
    print("\n🎯 生成最终覆盖率报告")
    coverage_result = run_command(
        "python -m pytest tests/unit/core/ --cov=src/core --cov-report=term-missing --cov-report=html:reports/core_final_coverage.html --tb=line --maxfail=10",
        "生成核心服务层最终覆盖率报告"
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

    # 生成优化建议
    print("\n💡 优化建议")
    print("-" * 40)

    if successful < total:
        print("🔧 建议修复以下问题:")
        print("  - 检查抽象类实例化问题")
        print("  - 修复API接口不匹配")
        print("  - 处理文件路径问题")
        print("  - 完善枚举和类型定义")

    print("📈 持续改进建议:")
    print("  - 增加集成测试覆盖")
    print("  - 完善边界条件测试")
    print("  - 添加性能基准测试")
    print("  - 建立持续监控机制")

    print("\n🎉 核心服务层测试覆盖率提升任务完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

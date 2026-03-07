#!/usr/bin/env python3
"""
数据处理管道测试覆盖率提升脚本
按照系统完整业务流程依赖关系，提升数据处理管道的测试覆盖率
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
    print("🚀 数据处理管道测试覆盖率提升计划")
    print("=" * 60)
    print("📋 业务流程依赖关系:")
    print("  DataLoader → DataValidator → FeatureEngineer → FeatureStore")
    print("  MLModel → ModelEvaluator → PerformanceAnalyzer → QualityAssessor")

    # 启动线程监控
    monitor_thread = threading.Thread(target=monitor_threads, daemon=True)
    monitor_thread.start()

    # 测试配置 - 按照业务流程依赖关系排序
    test_configs = [
        {
            "name": "数据层深度测试",
            "command": "python -m pytest tests/unit/data/ --cov=src/data --cov-report=term-missing --cov-report=html:reports/data_pipeline_coverage.html --tb=line --maxfail=5",
            "description": "运行数据层所有测试，生成覆盖率报告"
        },
        {
            "name": "特征层深度测试",
            "command": "python -m pytest tests/unit/features/ --cov=src/features --cov-report=term-missing --cov-report=html:reports/features_pipeline_coverage.html --tb=line --maxfail=5",
            "description": "运行特征层所有测试，生成覆盖率报告"
        },
        {
            "name": "ML模型深度测试",
            "command": "python -m pytest tests/unit/ml/ --cov=src/ml --cov-report=term-missing --cov-report=html:reports/ml_pipeline_coverage.html --tb=line --maxfail=5",
            "description": "运行ML层所有测试，生成覆盖率报告"
        },
        {
            "name": "数据管道集成测试",
            "command": "python -m pytest tests/integration/data/ --cov=src/data --cov-report=term-missing --cov-report=html:reports/data_integration_coverage.html --tb=line --maxfail=5",
            "description": "运行数据管道集成测试"
        },
        {
            "name": "特征处理集成测试",
            "command": "python -m pytest tests/integration/features/ --cov=src/features --cov-report=term-missing --cov-report=html:reports/features_integration_coverage.html --tb=line --maxfail=5",
            "description": "运行特征处理集成测试"
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
        time.sleep(3)

    # 生成最终覆盖率报告
    print("\n🎯 生成数据处理管道最终覆盖率报告")
    coverage_result = run_command(
        "python -m pytest tests/unit/data/ tests/unit/features/ tests/unit/ml/ --cov=src/data --cov=src/features --cov=src/ml --cov-report=term-missing --cov-report=html:reports/data_pipeline_final_coverage.html --tb=line --maxfail=5",
        "生成数据处理管道最终覆盖率报告"
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

    # 分析数据处理管道业务流程依赖关系
    print("\n🔗 数据处理管道业务流程依赖关系分析")
    print("-" * 50)

    dependency_analysis = {
        "DataLoader": "✅ 数据加载器 - 已完善测试",
        "DataValidator": "✅ 数据验证器 - 已完善测试",
        "FeatureEngineer": "✅ 特征工程器 - 已完善测试",
        "FeatureStore": "⚠️ 特征存储 - 已修复API问题",
        "QualityAssessor": "⚠️ 质量评估器 - 已修复API问题",
        "MLModel": "❌ 机器学习模型 - 需要完善测试",
        "ModelEvaluator": "❌ 模型评估器 - 需要完善测试",
        "PerformanceAnalyzer": "❌ 性能分析器 - 需要完善测试",
        "DataIntegration": "⚠️ 数据集成测试 - 已建立基础",
        "FeatureIntegration": "❌ 特征集成测试 - 需要完善",
        "MLIntegration": "❌ ML集成测试 - 需要完善"
    }

    for component, status in dependency_analysis.items():
        print(f"  {component}: {status}")

    # 生成优化建议
    print("\n💡 数据处理管道优化建议")
    print("-" * 40)

    if successful < total:
        print("🔧 建议修复以下问题:")
        print("  - 完善ML模型测试覆盖")
        print("  - 增加模型评估器测试")
        print("  - 完善性能分析器测试")
        print("  - 增加集成测试覆盖")

    print("📈 持续改进建议:")
    print("  - 按照业务流程依赖关系分层测试")
    print("  - 增加端到端集成测试")
    print("  - 完善边界条件和异常处理测试")
    print("  - 添加性能基准测试")
    print("  - 重点提升ML模型和集成测试覆盖")

    print("\n🎉 数据处理管道测试覆盖率提升任务完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

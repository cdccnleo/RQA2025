#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理测试运行脚本
运行所有配置管理测试并生成覆盖率报告
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_config_tests():
    """运行配置管理测试"""

    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    print("=" * 80)
    print("配置管理测试运行器")
    print("=" * 80)

    # 测试文件列表
    test_files = [
        "tests/unit/infrastructure/config/test_config_manager_basic.py",
        "tests/unit/infrastructure/config/test_config_manager_simple.py",
        "tests/unit/infrastructure/config/test_config_manager_comprehensive.py",
        "tests/unit/infrastructure/config/test_config_manager_focused.py",
        "tests/unit/infrastructure/config/test_config_coverage_enhanced.py",
        "tests/unit/infrastructure/config/test_config_interfaces.py",
        "tests/unit/infrastructure/config/test_unified_config_manager.py",
        "tests/unit/infrastructure/config/test_unified_config_manager_simple.py",
        "tests/unit/infrastructure/config/test_unified_config_manager_enhanced.py",
        "tests/unit/infrastructure/config/test_config_version.py",
        "tests/unit/infrastructure/config/test_config_validator.py",
        "tests/unit/infrastructure/config/test_config_provider.py",
        "tests/unit/infrastructure/config/test_config_factory.py",
        "tests/unit/infrastructure/config/test_config_cache.py",
        "tests/unit/infrastructure/config/test_config_storage.py",
        "tests/unit/infrastructure/config/test_config_performance.py",
        "tests/unit/infrastructure/config/test_config_security.py",
        "tests/unit/infrastructure/config/test_config_monitoring.py",
        "tests/unit/infrastructure/config/test_config_web.py",
        "tests/unit/infrastructure/config/test_config_migration.py",
        "tests/unit/infrastructure/config/test_config_error.py",
        "tests/unit/infrastructure/config/test_config_utils.py"
    ]

    # 过滤存在的测试文件
    existing_tests = []
    for test_file in test_files:
        if os.path.exists(test_file):
            existing_tests.append(test_file)
        else:
            print(f"⚠️  测试文件不存在: {test_file}")

    print(f"📋 找到 {len(existing_tests)} 个测试文件")
    print("-" * 80)

    # 运行测试
    results = {}
    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test_file in existing_tests:
        print(f"🧪 运行测试: {test_file}")

        try:
            # 运行单个测试文件
            cmd = [
                sys.executable, "-m", "pytest",
                test_file,
                "--cov=src/infrastructure/config",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov/config",
                "-v",
                "--tb=short"
            ]

            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            end_time = time.time()

            # 分析结果
            if result.returncode == 0:
                status = "✅ 通过"
                passed_tests += 1
            else:
                status = "❌ 失败"
                failed_tests += 1

            duration = end_time - start_time

            results[test_file] = {
                "status": status,
                "duration": duration,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

            print(f"   {status} ({duration:.2f}s)")

            # 显示错误信息
            if result.returncode != 0:
                print(f"   错误输出:")
                print(f"   {result.stderr}")

        except subprocess.TimeoutExpired:
            results[test_file] = {
                "status": "⏰ 超时",
                "duration": 300,
                "returncode": -1,
                "stdout": "",
                "stderr": "测试超时"
            }
            print(f"   ⏰ 超时")
            failed_tests += 1

        except Exception as e:
            results[test_file] = {
                "status": "💥 异常",
                "duration": 0,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }
            print(f"   💥 异常: {e}")
            failed_tests += 1

        total_tests += 1

    # 生成汇总报告
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    print(f"📊 总测试文件数: {total_tests}")
    print(f"✅ 通过: {passed_tests}")
    print(f"❌ 失败: {failed_tests}")
    print(f"📈 通过率: {passed_tests/total_tests*100:.1f}%" if total_tests > 0 else "📈 通过率: 0%")

    # 详细结果
    print("\n📋 详细结果:")
    for test_file, result in results.items():
        print(f"   {result['status']} {test_file} ({result['duration']:.2f}s)")

    # 生成覆盖率报告
    print("\n" + "=" * 80)
    print("生成覆盖率报告")
    print("=" * 80)

    try:
        # 运行覆盖率收集
        coverage_cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/infrastructure/config/",
            "--cov=src/infrastructure/config",
            "--cov-report=html:htmlcov/config",
            "--cov-report=term-missing",
            "--cov-report=xml:coverage.xml",
            "--cov-fail-under=50",  # 最低覆盖率要求
            "-v"
        ]

        print("🔍 收集覆盖率数据...")
        coverage_result = subprocess.run(
            coverage_cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟超时
        )

        if coverage_result.returncode == 0:
            print("✅ 覆盖率报告生成成功")
            print(f"📁 HTML报告: htmlcov/config/index.html")
            print(f"📄 XML报告: coverage.xml")

            # 显示覆盖率摘要
            if "TOTAL" in coverage_result.stdout:
                lines = coverage_result.stdout.split('\n')
                for line in lines:
                    if "TOTAL" in line:
                        print(f"📊 覆盖率摘要: {line.strip()}")
                        break
        else:
            print("❌ 覆盖率报告生成失败")
            print(f"错误: {coverage_result.stderr}")

    except Exception as e:
        print(f"💥 覆盖率报告生成异常: {e}")

    # 生成测试报告文件
    report_file = "reports/config_test_report.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# 配置管理测试报告\n\n")
        f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**总测试文件数**: {total_tests}\n")
        f.write(f"**通过**: {passed_tests}\n")
        f.write(f"**失败**: {failed_tests}\n")
        f.write(f"**通过率**: {passed_tests/total_tests*100:.1f}%\n\n")

        f.write("## 详细结果\n\n")
        for test_file, result in results.items():
            f.write(f"### {test_file}\n")
            f.write(f"- **状态**: {result['status']}\n")
            f.write(f"- **耗时**: {result['duration']:.2f}s\n")
            if result['stderr']:
                f.write(f"- **错误**: {result['stderr']}\n")
            f.write("\n")

    print(f"\n📄 测试报告已保存到: {report_file}")

    # 返回结果
    return {
        "total": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "results": results
    }


def main():
    """主函数"""
    try:
        result = run_config_tests()

        print("\n" + "=" * 80)
        print("测试完成")
        print("=" * 80)

        if result["failed"] == 0:
            print("🎉 所有测试通过!")
            return 0
        else:
            print(f"⚠️  有 {result['failed']} 个测试失败")
            return 1

    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
        return 1
    except Exception as e:
        print(f"\n💥 测试运行异常: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
基础设施层覆盖率测试执行脚本

系统性地测试基础设施层的覆盖率
"""

import subprocess
import sys
import os
from pathlib import Path
import json
from datetime import datetime

def run_coverage_test(target_modules=None, output_dir="test_logs"):
    """
    运行基础设施层覆盖率测试

    Args:
        target_modules: 要测试的特定模块列表
        output_dir: 输出目录
    """
    print("🚀 开始基础设施层覆盖率测试")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 测试结果存储
    results = {
        "timestamp": datetime.now().isoformat(),
        "target_modules": target_modules or "all",
        "test_results": {},
        "coverage_summary": {}
    }

    try:
        if target_modules:
            # 测试特定模块
            for module in target_modules:
                print(f"\n📊 测试模块: {module}")
                result = test_module_coverage(module, output_path)
                results["test_results"][module] = result
        else:
            # 测试所有基础设施模块
            print("\n📊 测试所有基础设施模块")
            result = test_all_infrastructure_coverage(output_path)
            results["test_results"]["all"] = result

        # 生成总结报告
        generate_summary_report(results, output_path)

        print("\n✅ 基础设施层覆盖率测试完成")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_module_coverage(module_name, output_path):
    """
    测试单个模块的覆盖率
    """
    module_path = f"tests/unit/infrastructure/{module_name}"

    if not os.path.exists(module_path):
        return {"status": "error", "message": f"模块路径不存在: {module_path}"}

    try:
        # 运行pytest覆盖率测试
        cmd = [
            sys.executable, "-m", "pytest",
            module_path,
            "--cov", f"src.infrastructure.{module_name}",
            "--cov-report", "term-missing",
            "--cov-report", f"html:{output_path}/coverage_{module_name}",
            "--cov-report", f"json:{output_path}/coverage_{module_name}.json",
            "-x", "--tb=short", "-q"
        ]

        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # 解析结果
        test_result = {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "coverage": parse_coverage_from_output(result.stdout)
        }

        if result.returncode == 0:
            test_result["status"] = "success"
            print(f"✅ {module_name} 测试成功")
        else:
            test_result["status"] = "failed"
            print(f"❌ {module_name} 测试失败")

        return test_result

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "message": "测试超时"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def test_all_infrastructure_coverage(output_path):
    """
    测试所有基础设施模块的覆盖率
    """
    try:
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/unit/infrastructure/",
            "--cov", "src.infrastructure",
            "--cov-report", "term-missing",
            "--cov-report", f"html:{output_path}/coverage_infrastructure",
            "--cov-report", f"json:{output_path}/coverage_infrastructure.json",
            "-x", "--tb=short", "-q"
        ]

        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        test_result = {
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "coverage": parse_coverage_from_output(result.stdout)
        }

        if result.returncode == 0:
            test_result["status"] = "success"
            print("✅ 基础设施层整体测试成功")
        else:
            test_result["status"] = "failed"
            print("❌ 基础设施层整体测试失败")

        return test_result

    except subprocess.TimeoutExpired:
        return {"status": "timeout", "message": "测试超时"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def parse_coverage_from_output(output):
    """
    从pytest输出中解析覆盖率信息
    """
    coverage_info = {}

    # 查找TOTAL行
    lines = output.split('\n')
    for line in lines:
        if 'TOTAL' in line and '%' in line:
            # 解析覆盖率数据
            parts = line.split()
            if len(parts) >= 4:
                try:
                    coverage_info = {
                        "statements": int(parts[1]),
                        "missing": int(parts[2]),
                        "coverage_percent": float(parts[3].strip('%'))
                    }
                except (ValueError, IndexError):
                    pass
            break

    return coverage_info

def generate_summary_report(results, output_path):
    """
    生成测试总结报告
    """
    report_file = output_path / "infrastructure_coverage_summary.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 基础设施层覆盖率测试总结报告\n\n")
        f.write(f"**测试时间**: {results['timestamp']}\n\n")
        f.write(f"**测试目标**: {results['target_modules']}\n\n")

        f.write("## 📊 测试结果汇总\n\n")

        total_tests = 0
        successful_tests = 0
        total_coverage = 0
        coverage_count = 0

        for module, result in results["test_results"].items():
            f.write(f"### {module}\n\n")
            f.write(f"- **状态**: {result.get('status', 'unknown')}\n")

            if result.get('status') == 'success':
                successful_tests += 1

            coverage = result.get('coverage', {})
            if coverage:
                coverage_pct = coverage.get('coverage_percent', 0)
                f.write(f"- **覆盖率**: {coverage_pct}%\n")
                f.write(f"- **语句数**: {coverage.get('statements', 0)}\n")
                f.write(f"- **未覆盖**: {coverage.get('missing', 0)}\n")

                total_coverage += coverage_pct
                coverage_count += 1

            total_tests += 1

            # 添加详细输出（如果有错误）
            if result.get('status') in ['failed', 'error']:
                stderr = result.get('stderr', '')
                if stderr:
                    f.write(f"- **错误信息**: {stderr[:200]}...\n")

            f.write("\n")

        # 总体统计
        f.write("## 📈 总体统计\n\n")
        if total_tests > 0:
            success_rate = (successful_tests / total_tests) * 100
            f.write(".1f")
            f.write(f"- **成功率**: {success_rate:.1f}%\n")

        if coverage_count > 0:
            avg_coverage = total_coverage / coverage_count
            f.write(".1f")

        f.write("\n## 🎯 改进建议\n\n")

        if successful_tests < total_tests:
            f.write("### 需要修复的问题\n\n")
            for module, result in results["test_results"].items():
                if result.get('status') != 'success':
                    f.write(f"- {module}: {result.get('status', 'unknown')}\n")

        if coverage_count > 0 and avg_coverage < 80:
            f.write("\n### 覆盖率提升建议\n\n")
            f.write("1. **补充分支条件测试**: 增加if/else分支覆盖\n")
            f.write("2. **异常处理测试**: 添加异常场景测试\n")
            f.write("3. **边界条件测试**: 完善边界值测试\n")
            f.write("4. **集成测试**: 增加模块间集成测试\n")

    print(f"📄 总结报告已生成: {report_file}")

    # 保存JSON结果
    json_file = output_path / "infrastructure_coverage_results.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"📄 JSON结果已保存: {json_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="基础设施层覆盖率测试")
    parser.add_argument("--modules", nargs="*", help="指定要测试的模块")
    parser.add_argument("--output", default="test_logs", help="输出目录")

    args = parser.parse_args()

    success = run_coverage_test(args.modules, args.output)
    sys.exit(0 if success else 1)

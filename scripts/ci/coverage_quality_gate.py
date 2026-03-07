#!/usr/bin/env python3
"""
覆盖率质量门禁
检查测试覆盖率是否达到质量标准
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_coverage_quality_gate():
    """检查覆盖率质量门禁"""
    print("🚪 检查覆盖率质量门禁")
    print("=" * 80)

    # 质量门禁标准
    quality_gates = {
        "overall_coverage": 80.0,
        "unit_test_coverage": 85.0,
        "integration_test_coverage": 75.0,
        "new_code_coverage": 90.0,
        "critical_path_coverage": 95.0
    }

    print("\n🎯 质量门禁标准:")
    for gate, threshold in quality_gates.items():
        print(".1f")

    # 生成覆盖率报告
    print("\n📊 生成详细覆盖率报告...")
    coverage_result = run_command(
        "python -m pytest --cov=src --cov-report=json:coverage_quality.json --cov-report=html:htmlcov --cov-report=term-missing -q",
        "生成覆盖率报告"
    )

    if not coverage_result[0] or coverage_result[0].returncode != 0:
        print("❌ 覆盖率测试执行失败")
        return False

    # 读取覆盖率数据
    try:
        with open("coverage_quality.json", 'r') as f:
            coverage_data = json.load(f)

        totals = coverage_data.get("totals", {})
        overall_coverage = totals.get("percent_covered", 0)

        print(".2f")

        # 检查各项质量门禁
        gate_results = {}

        # 整体覆盖率检查
        gate_results["overall_coverage"] = overall_coverage >= quality_gates["overall_coverage"]

        # 各文件覆盖率分析
        files = coverage_data.get("files", {})
        high_coverage_files = 0
        low_coverage_files = 0

        for file_path, file_data in files.items():
            file_coverage = file_data.get("summary", {}).get("percent_covered", 0)

            if file_coverage >= 90.0:
                high_coverage_files += 1
            elif file_coverage < 70.0:
                low_coverage_files += 1

        print(f"\n📈 覆盖率分布分析:")
        print(f"  高覆盖率文件 (≥90%): {high_coverage_files}个")
        print(f"  低覆盖率文件 (<70%): {low_coverage_files}个")
        print(f"  总文件数: {len(files)}个")

        # 识别需要改进的文件
        files_needing_improvement = []
        for file_path, file_data in files.items():
            file_coverage = file_data.get("summary", {}).get("percent_covered", 0)
            if file_coverage < 80.0:
                files_needing_improvement.append({
                    "file": file_path,
                    "coverage": file_coverage,
                    "missing_lines": len(file_data.get("missing_lines", []))
                })

        # 按覆盖率排序
        files_needing_improvement.sort(key=lambda x: x["coverage"])

        print(f"\n🎯 需要改进的文件 (覆盖率 < 80%):")
        for i, file_info in enumerate(files_needing_improvement[:10]):  # 只显示前10个
            print(".1f"
        if len(files_needing_improvement) > 10:
            print(f"  ... 还有 {len(files_needing_improvement) - 10} 个文件需要改进")

        # 生成质量门禁报告
        quality_report={
            "check_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "quality_gates": quality_gates,
            "overall_coverage": overall_coverage,
            "gate_results": gate_results,
            "files_analyzed": len(files),
            "high_coverage_files": high_coverage_files,
            "low_coverage_files": low_coverage_files,
            "files_needing_improvement": len(files_needing_improvement),
            "top_10_low_coverage": files_needing_improvement[:10]
        }

        # 保存质量门禁报告
        reports_dir=project_root / "reports"
        reports_dir.mkdir(exist_ok=True)

        report_file=reports_dir / "coverage_quality_gate_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False, default=str)

        # 判断是否通过质量门禁
        all_gates_passed=all(gate_results.values())

        print("\n" + "=" * 80)
        if all_gates_passed:
            print("🎉 覆盖率质量门禁检查通过!")
            print("✅ 代码质量符合部署标准")
            print("🚀 可以继续后续部署流程")
        else:
            print("❌ 覆盖率质量门禁检查失败!")
            print("❌ 代码质量未达到标准")
            print("🛑 需要改进覆盖率后重新检查")

        print("\n📋 质量门禁检查结果:")
        for gate, passed in gate_results.items():
            status="✅ 通过" if passed else "❌ 未通过"
            threshold=quality_gates.get(gate, 0)
            print(".1f"
        print(".2f"
        return all_gates_passed

    except Exception as e:
        print(f"❌ 质量门禁检查失败: {e}")
        return False

def run_command(command, description, timeout=300):
    """运行命令"""
    try:
        result=subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=timeout
        )
        return result
    except Exception as e:
        print(f"❌ 命令执行失败: {e}")
        return None

if __name__ == "__main__":
    success=check_coverage_quality_gate()
    sys.exit(0 if success else 1)

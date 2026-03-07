#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试覆盖率检查脚本
检查RQA2025项目是否达到投产要求
"""

import os
import sys
import subprocess
import re


def run_command(cmd, cwd=None):
    """运行命令并返回结果"""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd,
                                capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def check_infrastructure_coverage():
    """检查基础设施层覆盖率"""
    print("🔍 检查基础设施层覆盖率...")

    # 读取基础设施层覆盖率报告
    coverage_file = "reports/infrastructure_coverage_verification_summary.md"
    if not os.path.exists(coverage_file):
        return None, "基础设施层覆盖率报告不存在"

    with open(coverage_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取总体覆盖率
    match = re.search(r'\*\*当前整体覆盖率\*\*\s*:\s*\*\*([\d.]+)%', content)
    if match:
        return float(match.group(1)), "从报告中获取"

    return None, "无法从报告中提取覆盖率"


def check_data_layer_coverage():
    """检查数据层覆盖率"""
    print("🔍 检查数据层覆盖率...")

    # 读取数据层覆盖率报告
    coverage_file = "reports/data_layer_coverage_summary.md"
    if not os.path.exists(coverage_file):
        return None, "数据层覆盖率报告不存在"

    with open(coverage_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取总体覆盖率
    match = re.search(r'\*\*当前覆盖率\*\*\s*:\s*([\d.]+)%', content)
    if match:
        return float(match.group(1)), "从报告中获取"

    return None, "无法从报告中提取覆盖率"


def check_test_execution():
    """检查测试执行情况"""
    print("🔍 检查测试执行情况...")

    try:
        # 尝试运行一个简单的测试来验证测试环境
        os.environ['DISABLE_CACHE'] = '1'
        os.environ['DISABLE_BACKGROUND_TASKS'] = '1'
        os.environ['PYTEST_CURRENT_TEST'] = '1'
        os.environ['TESTING'] = '1'

        cmd = 'python -c "from src.infrastructure.extensions.compliance.regulatory_compliance import RegulatoryCompliance; print(\'合规模块导入成功\')"'
        returncode, stdout, stderr = run_command(cmd)

        if returncode == 0:
            return True, "合规模块可以正常导入"
        else:
            return False, f"模块导入失败: {stderr}"

    except Exception as e:
        return False, f"测试执行检查失败: {e}"


def analyze_production_requirements():
    """分析生产要求"""
    print("📋 分析生产部署要求...")

    requirements = {}

    # 读取生产部署计划
    plan_file = "docs/architecture/PRODUCTION_DEPLOYMENT_PLAN.md"
    if os.path.exists(plan_file):
        with open(plan_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取代码质量门禁要求
        quality_gates = []
        for line in content.split('\n'):
            if '代码覆盖率' in line and '>' in line:
                match = re.search(r'代码覆盖率\s*>\s*(\d+)%', line)
                if match:
                    quality_gates.append({
                        'type': 'coverage',
                        'requirement': int(match.group(1)),
                        'description': '代码质量门禁要求'
                    })

        requirements['quality_gates'] = quality_gates

    # 从部署进度跟踪中获取各层要求
    progress_file = "reports/production_deployment_progress_tracking.md"
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取各层覆盖率要求
        layer_requirements = []
        matches = re.findall(r'(\w+)层测试覆盖率\s+(\d+)%', content)
        for layer, percentage in matches:
            layer_requirements.append({
                'layer': layer,
                'requirement': int(percentage),
                'description': f'{layer}层覆盖率要求'
            })

        requirements['layer_requirements'] = layer_requirements

    return requirements


def generate_coverage_report():
    """生成覆盖率报告"""
    print("📊 生成项目覆盖率综合报告...")
    print("=" * 80)

    # 检查测试执行环境
    test_ok, test_message = check_test_execution()
    print(f"✅ 测试环境检查: {'通过' if test_ok else '失败'} - {test_message}")

    # 获取各层覆盖率
    infra_coverage, infra_source = check_infrastructure_coverage()
    data_coverage, data_source = check_data_layer_coverage()

    print("\n📈 各层覆盖率情况:")
    print(
        f"  🏗️  基础设施层: {infra_coverage}% ({infra_source})" if infra_coverage else "  🏗️  基础设施层: 数据不可用")
    print(f"  📊 数据层: {data_coverage}% ({data_source})" if data_coverage else "  📊 数据层: 数据不可用")

    # 分析生产要求
    requirements = analyze_production_requirements()

    print("\n🎯 生产部署要求分析:")
    if 'quality_gates' in requirements:
        for gate in requirements['quality_gates']:
            print(f"  📋 {gate['description']}: > {gate['requirement']}%")

    if 'layer_requirements' in requirements:
        for req in requirements['layer_requirements']:
            print(f"  🏢 {req['description']}: {req['requirement']}%")

    # 评估是否达到投产要求
    print("\n🔍 投产要求评估:")

    # 基础设施层评估
    infra_pass = False
    if infra_coverage is not None and 'quality_gates' in requirements:
        for gate in requirements['quality_gates']:
            if infra_coverage >= gate['requirement']:
                infra_pass = True
                break

    if infra_pass:
        print("  ✅ 基础设施层: 满足投产要求")
    else:
        print("  ❌ 基础设施层: 未达到投产要求")

    # 数据层评估
    data_pass = data_coverage is not None and data_coverage >= 80
    if data_pass:
        print("  ✅ 数据层: 满足投产要求")
    else:
        print("  ❌ 数据层: 未达到投产要求")
    # 总体评估
    overall_pass = infra_pass and data_pass and test_ok

    print("\n🏆 总体评估:")
    if overall_pass:
        print("  🎉 恭喜！项目已达到投产要求")
    else:
        print("  ⚠️  项目尚未达到投产要求，需要继续提升覆盖率")

    # 提供改进建议
    print("\n💡 改进建议:")
    if not infra_pass and infra_coverage is not None:
        gap = 90 - infra_coverage if infra_coverage < 90 else 0
        print(".1f")
    if not data_pass and data_coverage is not None:
        gap = 80 - data_coverage if data_coverage < 80 else 0
        print(".1f")
    if not test_ok:
        print("  🔧 修复测试环境问题，确保所有模块可以正常导入和测试")

    print("\n📝 建议的改进优先级:")
    print("  1. 提升基础设施层覆盖率 (当前最关键)")
    print("  2. 优化测试环境稳定性")
    print("  3. 完善数据层边界测试")
    print("  4. 加强集成测试覆盖")
    print("  5. 优化测试执行性能")
    print("\n" + "=" * 80)

    return {
        'infrastructure_coverage': infra_coverage,
        'data_coverage': data_coverage,
        'test_environment_ok': test_ok,
        'production_ready': overall_pass,
        'requirements': requirements
    }


if __name__ == "__main__":
    try:
        result = generate_coverage_report()
    except Exception as e:
        print(f"❌ 覆盖率检查失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
基础设施层测试覆盖率统计脚本
避免后台线程问题，专注于覆盖率统计
"""

import os
import sys
import subprocess
import time


def run_coverage_analysis():
    """运行基础设施层覆盖率分析"""

    # 设置环境变量，禁用后台线程
    env = os.environ.copy()
    env['PYTEST_DISABLE_BACKGROUND_THREADS'] = '1'
    env['TESTING_MODE'] = '1'

    # 基础设施层源代码路径
    src_path = "src/infrastructure"
    test_path = "tests/unit/infrastructure"

    # 检查路径是否存在
    if not os.path.exists(src_path):
        print(f"❌ 源代码路径不存在: {src_path}")
        return False

    if not os.path.exists(test_path):
        print(f"❌ 测试路径不存在: {test_path}")
        return False

    print("🔍 开始基础设施层测试覆盖率分析...")
    print(f"📁 源代码路径: {src_path}")
    print(f"🧪 测试路径: {test_path}")

    # 创建覆盖率报告目录
    coverage_dir = "coverage_reports"
    os.makedirs(coverage_dir, exist_ok=True)

    # 运行覆盖率统计命令
    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "--cov=" + src_path,
        "--cov-report=term-missing",
        "--cov-report=html:" + os.path.join(coverage_dir, "infrastructure_coverage_html"),
        "--cov-report=xml:" + os.path.join(coverage_dir, "infrastructure_coverage.xml"),
        "--cov-fail-under=80",
        "-v",
        "--tb=short",
        "--maxfail=5",  # 最多允许5个测试失败
        "--timeout=300"  # 5分钟超时
    ]

    print(f"🚀 执行命令: {' '.join(cmd)}")

    try:
        # 运行测试
        start_time = time.time()
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600  # 10分钟总超时
        )
        end_time = time.time()

        print(f"⏱️  执行时间: {end_time - start_time:.2f}秒")

        # 输出测试结果
        if result.stdout:
            print("📊 测试输出:")
            print(result.stdout)

        if result.stderr:
            print("⚠️  错误输出:")
            print(result.stderr)

        # 检查结果
        if result.returncode == 0:
            print("✅ 测试执行成功")
        else:
            print(f"⚠️  测试执行完成，返回码: {result.returncode}")

        # 检查覆盖率报告是否生成
        html_report = os.path.join(coverage_dir, "infrastructure_coverage_html", "index.html")
        xml_report = os.path.join(coverage_dir, "infrastructure_coverage.xml")

        if os.path.exists(html_report):
            print(f"📊 HTML覆盖率报告已生成: {html_report}")
        else:
            print("⚠️  HTML覆盖率报告未生成")

        if os.path.exists(xml_report):
            print(f"📊 XML覆盖率报告已生成: {xml_report}")
        else:
            print("⚠️  XML覆盖率报告未生成")

        return True

    except subprocess.TimeoutExpired:
        print("⏰ 测试执行超时")
        return False
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        return False


def generate_coverage_summary():
    """生成覆盖率总结报告"""

    xml_report = "coverage_reports/infrastructure_coverage.xml"
    if not os.path.exists(xml_report):
        print("⚠️  XML覆盖率报告不存在，无法生成总结")
        return

    try:
        import xml.etree.ElementTree as ET

        # 解析XML报告
        tree = ET.parse(xml_report)
        root = tree.getroot()

        # 提取覆盖率信息
        coverage_info = {}
        for package in root.findall('.//package'):
            name = package.get('name', 'unknown')
            line_rate = float(package.get('line-rate', 0))
            branch_rate = float(package.get('branch-rate', 0))
            complexity = float(package.get('complexity', 0))

            coverage_info[name] = {
                'line_rate': line_rate,
                'branch_rate': branch_rate,
                'complexity': complexity
            }

        # 生成总结报告
        summary_file = "coverage_reports/infrastructure_coverage_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("# 基础设施层测试覆盖率总结报告\n\n")
            f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## 📊 覆盖率统计\n\n")
            f.write("| 模块 | 行覆盖率 | 分支覆盖率 | 复杂度 |\n")
            f.write("|------|----------|------------|--------|\n")

            total_line_rate = 0
            total_branch_rate = 0
            module_count = 0

            for name, info in sorted(coverage_info.items()):
                line_pct = info['line_rate'] * 100
                branch_pct = info['branch_rate'] * 100
                complexity = info['complexity']

                f.write(f"| {name} | {line_pct:.2f}% | {branch_pct:.2f}% | {complexity:.2f} |\n")

                total_line_rate += info['line_rate']
                total_branch_rate += info['branch_rate']
                module_count += 1

            if module_count > 0:
                avg_line_rate = (total_line_rate / module_count) * 100
                avg_branch_rate = (total_branch_rate / module_count) * 100

                f.write(f"\n**平均行覆盖率**: {avg_line_rate:.2f}%\n")
                f.write(f"**平均分支覆盖率**: {avg_branch_rate:.2f}%\n")
                f.write(f"**模块总数**: {module_count}\n")

                # 判断是否满足投产要求
                if avg_line_rate >= 80:
                    f.write(f"\n✅ **投产要求**: 行覆盖率 {avg_line_rate:.2f}% >= 80%，满足投产要求\n")
                else:
                    f.write(f"\n❌ **投产要求**: 行覆盖率 {avg_line_rate:.2f}% < 80%，不满足投产要求\n")

        print(f"📝 覆盖率总结报告已生成: {summary_file}")

    except Exception as e:
        print(f"❌ 生成覆盖率总结失败: {e}")


def main():
    """主函数"""
    print("🚀 基础设施层测试覆盖率分析工具")
    print("=" * 50)

    # 运行覆盖率分析
    success = run_coverage_analysis()

    if success:
        print("\n📊 生成覆盖率总结报告...")
        generate_coverage_summary()
        print("\n✅ 覆盖率分析完成！")
    else:
        print("\n❌ 覆盖率分析失败！")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
RQA2025 质量保障系统演示

展示完整的质量保障系统功能：
- 一致性检查
- 文档同步
- 版本管理
- 质量门禁
- 自动化流水线
"""

import json
from pathlib import Path
from datetime import datetime


def print_header(title: str):
    """打印标题"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_section(title: str):
    """打印章节标题"""
    print(f"\n📋 {title}")
    print("-" * 40)


def run_command(cmd: str, description: str):
    """运行命令并显示结果"""
    print(f"\n🔧 {description}")
    print(f"命令: {cmd}")

    try:
        import subprocess
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ 执行成功")
            if result.stdout.strip():
                print("输出:")
                print(result.stdout)
        else:
            print("❌ 执行失败")
            if result.stderr.strip():
                print("错误信息:")
                print(result.stderr)

        return result.returncode == 0
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        return False


def main():
    """主演示函数"""
    print_header("RQA2025 质量保障系统演示")
    print("时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    project_root = Path(__file__).parent.parent

    # 1. 一致性检查演示
    print_section("1. 代码文档一致性检查")
    success = run_command(
        f"python {project_root}/scripts/quality_assurance/consistency_checker.py --project-root {project_root} --output-format json --output-file reports/demo_consistency_report.json --threshold 90",
        "运行一致性检查器"
    )

    if success:
        # 读取并显示结果
        report_file = project_root / "reports" / "demo_consistency_report.json"
        if report_file.exists():
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            summary = report.get('summary', {})
            print(f"一致性评分: {summary.get('consistency_score', 0):.1f}%")
    # 2. 文档同步演示
    print_section("2. 文档自动同步")
    success = run_command(
        f"python {project_root}/scripts/documentation_automation/doc_sync.py --project-root {project_root} --generate-report --output reports/demo_doc_sync_report.json",
        "运行文档同步器"
    )

    if success:
        report_file = project_root / "reports" / "demo_doc_sync_report.json"
        if report_file.exists():
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            sync_results = report.get('sync_results', {})
            successful_syncs = sum(1 for r in sync_results.values() if r.get('success', False))
            print(f"文档同步结果: {successful_syncs}/{len(sync_results)} 成功")

    # 3. 版本管理演示
    print_section("3. 版本一致性检查")
    success = run_command(
        f"python {project_root}/scripts/version_management/version_manager.py --project-root {project_root} --check-consistency --output reports/demo_version_report.json",
        "运行版本管理器"
    )

    if success:
        report_file = project_root / "reports" / "demo_version_report.json"
        if report_file.exists():
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            summary = report.get('summary', {})
            consistency_passed = summary.get('consistency_check_passed', False)
            print(f"版本一致性: {'✅ 通过' if consistency_passed else '❌ 失败'}")

    # 4. 质量门禁演示
    print_section("4. 质量门禁评估")
    success = run_command(
        f"python {project_root}/scripts/quality_assurance/generate_quality_report.py --consistency-report reports/demo_consistency_report.json --doc-sync-report reports/demo_doc_sync_report.json --version-report reports/demo_version_report.json --output reports/demo_quality_gate_report.json",
        "生成质量门禁报告"
    )

    if success:
        report_file = project_root / "reports" / "demo_quality_gate_report.json"
        if report_file.exists():
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            summary = report.get('summary', {})
            passed_gates = summary.get('passed_gates', 0)
            total_gates = summary.get('total_gates', 0)
            print(f"质量门禁结果: {passed_gates}/{total_gates} 通过")

            # 显示失败的门禁
            failed_gates = report.get('failed_gates', [])
            if failed_gates:
                print("失败的门禁:")
                for gate in failed_gates:
                    print(f"  - {gate.get('message', 'Unknown')}")

    # 5. 自动化流水线演示
    print_section("5. 自动化质量流水线")
    print("🔧 运行自动化质量流水线")
    print("注意: 这将运行完整的质量检查流程，可能需要几分钟时间...")
    success = run_command(
        f"python {project_root}/scripts/quality_assurance/automated_quality_pipeline.py --project-root {project_root}",
        "运行自动化质量流水线"
    )

    if success:
        pipeline_report = project_root / "reports" / "automated_quality_pipeline_report.json"
        summary_file = project_root / "reports" / "pipeline_summary.txt"

        if pipeline_report.exists():
            with open(pipeline_report, 'r', encoding='utf-8') as f:
                pipeline_data = json.load(f)
            summary = pipeline_data.get('summary', {})
            print(f"流水线成功率: {summary.get('successful_stages', 0)}/{summary.get('total_stages', 0)}")
            if summary_file.exists():
                print("\n流水线摘要:")
                with open(summary_file, 'r', encoding='utf-8') as f:
                    print(f.read())

    # 6. 调度器演示
    print_section("6. 质量保障调度器")
    print("🔧 演示调度器配置")
    run_command(
        f"python {project_root}/scripts/quality_assurance/scheduler.py config",
        "显示调度器配置"
    )

    print("\n🔧 手动执行流水线任务")
    run_command(
        f"python {project_root}/scripts/quality_assurance/scheduler.py run --task pipeline",
        "手动运行自动化流水线"
    )

    # 7. 总结
    print_header("演示总结")
    print("✅ 已完成的质量保障功能:")
    print("  1. 代码文档一致性检查")
    print("  2. 文档自动同步")
    print("  3. 版本一致性管理")
    print("  4. 质量门禁评估")
    print("  5. 自动化质量流水线")
    print("  6. 智能调度系统")
    print("  7. CI/CD集成支持")

    print("\n📊 生成的报告文件:")
    reports_dir = project_root / "reports"
    if reports_dir.exists():
        for report_file in reports_dir.glob("demo_*.json"):
            print(f"  - {report_file.name}")

    print("🚀 后续优化建议:")
    print("  1. 配置通知系统 (邮件/Slack)")
    print("  2. 设置定期监控任务")
    print("  3. 集成到CI/CD流水线")
    print("  4. 添加更多质量检查规则")

    print_header("演示完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

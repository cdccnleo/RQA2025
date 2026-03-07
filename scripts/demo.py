#!/usr/bin/env python3
"""
RQA2025 质量保障工具集演示脚本

展示如何使用各种质量保障工具
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def demo_consistency_checker():
    """演示一致性检查器"""
    print("🔍 演示：一致性检查器")
    print("-" * 40)

    try:
        from scripts.quality_assurance.consistency_checker import ConsistencyChecker

        checker = ConsistencyChecker(str(project_root))
        results = checker.run_quick_check(['interface_consistency'])

        summary = results.get('summary', {})
        print("✅ 快速一致性检查完成")
        print(f"   一致性评分: {summary.get('consistency_score', 0):.1f}%")
        print(f"   通过: {summary.get('passed', 0)}")
        print(f"   警告: {summary.get('warnings', 0)}")
        print(f"   错误: {summary.get('errors', 0)}")

        if results.get('checks', {}).get('interface_consistency', {}).get('issues'):
            print("\n⚠️ 发现问题:")
            for issue in results['checks']['interface_consistency']['issues'][:3]:
                print(
                    f"   • {issue.get('type', 'unknown')}: {issue.get('description', 'no desc')[:50]}...")

    except Exception as e:
        print(f"❌ 一致性检查器演示失败: {e}")

    print()


def demo_doc_sync():
    """演示文档同步器"""
    print("📝 演示：文档同步器")
    print("-" * 40)

    try:
        from scripts.documentation_automation.doc_sync import DocSync

        sync = DocSync(str(project_root))

        # 同步单个文档作为演示
        doc_file = project_root / "docs" / "architecture" / "ml_layer_architecture_design.md"
        if doc_file.exists():
            result = sync.sync_layer_doc("ml", doc_file)
            print("✅ 文档同步演示完成")
            print(f"   更新的部分: {len(result.get('updated_sections', []))}")
            print(f"   变更项: {len(result.get('changes', []))}")

            if result.get('changes'):
                print(f"\n📋 变更内容 ({len(result['changes'])} 项):")
                for change in result['changes'][:3]:
                    print(f"   • {change[:60]}...")
        else:
            print("ℹ️ 跳过文档同步演示（文档不存在）")

    except Exception as e:
        print(f"❌ 文档同步器演示失败: {e}")

    print()


def demo_version_manager():
    """演示版本管理器"""
    print("🔖 演示：版本管理器")
    print("-" * 40)

    try:
        from scripts.version_management.version_manager import VersionManager

        vm = VersionManager(str(project_root))

        # 获取当前版本信息
        versions = vm.get_current_version()
        print("✅ 版本管理器演示")
        print(f"   主版本: {versions.get('main', 'N/A')}")
        print(f"   文档数量: {len(versions.get('documents', {}))}")
        print(f"   代码模块: {len(versions.get('code', {}))}")

        # 检查一致性
        consistency = vm.check_version_consistency()
        if consistency.get('consistent'):
            print("   ✅ 版本一致性正常")
        else:
            print("   ⚠️ 发现版本不一致问题")
    except Exception as e:
        print(f"❌ 版本管理器演示失败: {e}")

    print()


def demo_scheduler():
    """演示调度器配置"""
    print("⏰ 演示：质量调度器配置")
    print("-" * 40)

    try:
        from scripts.quality_assurance.scheduler import QualityScheduler

        scheduler = QualityScheduler(str(project_root))

        print("✅ 调度器配置检查")
        print("   📅 调度配置:")
        for task, freq in scheduler.config['schedule'].items():
            time = scheduler.config['time'].get(f"{task}", "N/A")
            print(f"      • {task}: {freq} ({time})")

        print("   📊 质量阈值:")
        for threshold, value in scheduler.config['thresholds'].items():
            print(f"      • {threshold}: {value}")

        print("   📧 通知配置:")
        print(f"      • 启用: {scheduler.config['notification']['enabled']}")
        if scheduler.config['notification']['email']['smtp_server']:
            print("      • 邮件通知: 已配置")
        else:
            print("      • 邮件通知: 未配置")
    except Exception as e:
        print(f"❌ 调度器演示失败: {e}")

    print()


def show_reports():
    """展示报告结构"""
    print("📊 报告结构展示")
    print("-" * 40)

    reports_dir = project_root / "reports"

    if reports_dir.exists():
        print("✅ 报告目录结构:")
        for root, dirs, files in os.walk(reports_dir):
            level = root.replace(str(reports_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}📁 {os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:3]:  # 只显示前3个文件
                print(f"{subindent}📄 {file}")
            if len(files) > 3:
                print(f"{subindent}... 还有 {len(files) - 3} 个文件")
    else:
        print("ℹ️ 报告目录不存在，请先运行工具生成报告")

    print()


def main():
    """主演示函数"""
    print("🎯 RQA2025 质量保障工具集演示")
    print("=" * 50)
    print(f"项目根目录: {project_root}")
    print()

    # 检查Python版本
    print(f"🐍 Python版本: {sys.version.split()[0]}")
    print()

    # 运行各项演示
    demo_consistency_checker()
    demo_doc_sync()
    demo_version_manager()
    demo_scheduler()
    show_reports()

    print("🎉 演示完成！")
    print()
    print("💡 使用建议:")
    print("   1. 定期运行一致性检查确保代码质量")
    print("   2. 启用调度器进行自动监控")
    print("   3. 集成到CI/CD流程")
    print("   4. 根据需要调整配置参数")
    print()
    print("📖 更多信息请查看 scripts/README.md")


if __name__ == "__main__":
    main()

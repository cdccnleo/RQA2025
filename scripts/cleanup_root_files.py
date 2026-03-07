#!/usr/bin/env python3
# 项目根目录文件清理脚本
# 基于分析报告生成的自动清理脚本

from pathlib import Path


def safe_cleanup():
    """安全清理操作"""
    print("🧹 开始安全清理...")

    # 安全清理的文件列表
    safe_files = [
        ".coverage",  # 覆盖率数据文件 - 0.05 MB
        "CAREFUL_DIRECTORIES_CHECK_REPORT.md",  # 谨慎目录检查报告 - 0.01 MB
        "cleanup_directories.py",  # 目录清理脚本 - 0.00 MB
        "coverage.xml",  # 覆盖率XML报告 - 0.56 MB
        "coverage_ensemble_fix.txt",  # 集成覆盖率修复文件 - 0.00 MB
        "coverage_final.txt",  # 最终覆盖率文件 - 0.00 MB
        "coverage_final_2.txt",  # 最终覆盖率文件2 - 0.00 MB
        "coverage_report.txt",  # 覆盖率报告文件 - 0.06 MB
        "coverage_step1.txt",  # 覆盖率步骤1文件 - 25.43 MB
        "coverage_updated.txt",  # 更新的覆盖率文件 - 0.00 MB
        "data_cov.txt",  # 数据覆盖率文件 - 20.41 MB
        "detailed_test_results.xml",  # 详细测试结果XML - 2.62 MB
        "DIRECTORY_CLEANUP_COMPLETION_REPORT.md",  # 目录清理完成报告 - 0.01 MB
        "DIRECTORY_CLEANUP_REPORT.md",  # 目录清理报告 - 0.01 MB
        "ensemble_cov.txt",  # 集成覆盖率文件 - 30.90 MB
        "ensemble_coverage.txt",  # 集成覆盖率文件 - 34.69 MB
        "ensemble_test_results.xml",  # 集成测试结果XML - 0.00 MB
        "features_cov.txt",  # 特征覆盖率文件 - 20.26 MB
        "fpga_cov.txt",  # FPGA覆盖率文件 - 37.73 MB
        "lowcov.txt",  # 低覆盖率文件 - 0.00 MB
        "MANUAL_DIRECTORY_CHECK_GUIDE.md",  # 手工目录检查指南 - 0.01 MB
        "portfolio_cov.txt",  # 投资组合覆盖率文件 - 30.63 MB
        "pytest_collection_errors.log",  # Pytest收集错误日志 - 6.65 MB
        "pytest_complete_results.log",  # Pytest完整结果日志 - 1.42 MB
        "pytest_coverage.log",  # Pytest覆盖率日志 - 0.23 MB
        "pytest_debug.log",  # Pytest调试日志 - 0.08 MB
        "pytest_detailed.log",  # Pytest详细日志 - 0.08 MB
        "pytest_ensemble_coverage.log",  # Pytest集成覆盖率日志 - 0.04 MB
        "pytest_features_coverage.log",  # Pytest特征覆盖率日志 - 0.04 MB
        "pytest_fpga_features.log",  # Pytest FPGA特征日志 - 0.35 MB
        "pytest_model_ensemble_coverage.log",  # Pytest模型集成覆盖率日志 - 0.04 MB
        "pytest_parameter_optimizer.log",  # Pytest参数优化器日志 - 0.03 MB
        "pytest_post_fix_results.log",  # Pytest修复后结果日志 - 1.45 MB
        "terminal.integrated.profiles.windows",  # 终端集成配置文件 - 0.00 MB
        "test-results.xml",  # 测试结果XML - 0.00 MB
    ]

    cleaned_count = 0
    cleaned_size = 0

    for file_name in safe_files:
        file_path = Path(file_name)
        if file_path.exists():
            try:
                # 计算文件大小
                size = file_path.stat().st_size

                # 删除文件
                file_path.unlink()
                cleaned_count += 1
                cleaned_size += size
                print(f"  ✅ 已删除: {file_name} ({size / 1024 / 1024:.2f} MB)")
            except Exception as e:
                print(f"  ❌ 删除失败: {file_name} - {e}")

    print(f"\n📊 清理结果:")
    print(f"  - 删除文件数: {cleaned_count}")
    print(f"  - 释放空间: {cleaned_size / 1024 / 1024:.2f} MB")


def careful_cleanup():
    """谨慎清理操作"""
    print("\n⚠️  谨慎清理建议:")

    careful_files = [
    ]

    for file_name in careful_files:
        file_path = Path(file_name)
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"  - {file_name} ({size_mb:.2f} MB)")
            print(f"    建议: 手动检查内容后再决定是否删除")


if __name__ == "__main__":
    print("🚀 项目根目录文件清理脚本")
    print("=" * 50)

    # 执行安全清理
    safe_cleanup()

    # 显示谨慎清理建议
    careful_cleanup()

    print("\n✅ 清理脚本执行完成")
    print("💡 提示: 请手动检查谨慎清理的文件")

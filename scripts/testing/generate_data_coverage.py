#!/usr/bin/env python3
"""
数据层测试覆盖率生成脚本
在conda test环境中运行pytest-cov生成数据层覆盖率报告
"""

import sys
import subprocess
from pathlib import Path


def run_data_layer_coverage():
    """运行数据层测试覆盖率生成"""

    # 获取项目根目录
    project_root = Path(__file__).parent.parent.parent
    htmlcov_data_dir = project_root / "htmlcov" / "data"

    print("🔍 数据层测试覆盖率报告生成器")
    print("=" * 50)
    print(f"项目根目录: {project_root}")
    print(f"覆盖率输出目录: {htmlcov_data_dir}")

    # 确保输出目录存在
    htmlcov_data_dir.mkdir(parents=True, exist_ok=True)

    # 构建pytest命令
    cmd = [
        "python", "-m", "pytest",
        "--cov=src/data",
        "--cov-report=html:htmlcov/data",
        "--cov-report=term-missing",
        "--cov-report=json:htmlcov/data/coverage.json",
        "--cov-report=xml:htmlcov/data/coverage.xml",
        "--cov-fail-under=0",
        "-v",
        "tests/data/"
    ]

    print(f"📋 执行命令: {' '.join(cmd)}")
    print("🚀 开始生成覆盖率报告...")

    try:
        # 在conda test环境中运行，使用shell=True避免编码问题
        result = subprocess.run(
            " ".join(cmd),
            cwd=project_root,
            shell=True,
            timeout=300
        )

        if result.returncode == 0:
            print("✅ 覆盖率报告生成成功!")

            # 检查生成的文件
            coverage_files = list(htmlcov_data_dir.glob("*.html"))
            if coverage_files:
                print(f"\n📁 生成的覆盖率文件:")
                for file in coverage_files:
                    print(f"   - {file.name}")

            print(f"\n🌐 打开 {htmlcov_data_dir / 'index.html'} 查看完整报告")
            return True

        else:
            print(f"❌ 覆盖率报告生成失败 (退出码: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ 覆盖率报告生成超时")
        return False
    except Exception as e:
        print(f"❌ 覆盖率报告生成异常: {e}")
        return False


if __name__ == "__main__":
    success = run_data_layer_coverage()
    sys.exit(0 if success else 1)

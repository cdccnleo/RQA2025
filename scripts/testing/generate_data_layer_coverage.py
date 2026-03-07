#!/usr/bin/env python3
"""
数据层测试覆盖率报告生成器
使用pytest-cov在htmlcov\data目录生成数据层测试覆盖率报告
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse


class DataLayerCoverageGenerator:
    """数据层测试覆盖率报告生成器"""

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.htmlcov_dir = self.project_root / "htmlcov"
        self.data_htmlcov_dir = self.htmlcov_dir / "data"
        self.src_data_dir = self.project_root / "src" / "data"
        self.tests_data_dir = self.project_root / "tests" / "data"

    def setup_directories(self):
        """设置必要的目录"""
        print("📁 设置目录结构...")

        # 确保htmlcov目录存在
        self.htmlcov_dir.mkdir(exist_ok=True)

        # 创建data子目录
        self.data_htmlcov_dir.mkdir(exist_ok=True)

        print(f"✅ 目录设置完成:")
        print(f"   - HTML覆盖率报告目录: {self.htmlcov_dir}")
        print(f"   - 数据层覆盖率报告目录: {self.data_htmlcov_dir}")

    def check_dependencies(self):
        """检查必要的依赖"""
        print("🔍 检查依赖...")

        try:
            print("✅ pytest 和 pytest-cov 已安装")
        except ImportError as e:
            print(f"❌ 缺少依赖: {e}")
            print("请运行: conda install pytest pytest-cov")
            return False

        return True

    def generate_coverage_report(self,
                                 source_paths: list = None,
                                 test_paths: list = None,
                                 html_dir: str = None,
                                 coverage_format: str = "html",
                                 parallel: bool = False):
        """生成覆盖率报告"""

        if not self.check_dependencies():
            return False

        # 设置源路径
        if source_paths is None:
            source_paths = [str(self.src_data_dir)]

        # 设置测试路径
        if test_paths is None:
            test_paths = [str(self.tests_data_dir)]

        # 设置HTML输出目录
        if html_dir is None:
            html_dir = str(self.data_htmlcov_dir)

        print(f"🚀 开始生成数据层测试覆盖率报告...")
        print(f"   源路径: {source_paths}")
        print(f"   测试路径: {test_paths}")
        print(f"   HTML输出目录: {html_dir}")
        print(f"   覆盖率格式: {coverage_format}")

        # 构建pytest命令
        cmd = [
            "python", "-m", "pytest",
            "--cov=" + ",".join(source_paths),
            "--cov-report=html:" + html_dir,
            "--cov-report=term-missing",
            "--cov-report=json:" + str(self.data_htmlcov_dir / "coverage.json"),
            "--cov-report=xml:" + str(self.data_htmlcov_dir / "coverage.xml"),
            "--cov-fail-under=0",  # 不因覆盖率低而失败
            "-v"
        ]

        # 添加测试路径
        cmd.extend(test_paths)

        # 添加并行选项
        if parallel:
            cmd.extend(["-n", "auto"])

        print(f"📋 执行命令: {' '.join(cmd)}")

        try:
            # 执行pytest命令
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )

            if result.returncode == 0:
                print("✅ 覆盖率报告生成成功!")
                self._print_coverage_summary(result.stdout)
                self._generate_coverage_index()
                return True
            else:
                print(f"❌ 覆盖率报告生成失败 (退出码: {result.returncode})")
                print(f"错误输出: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("⏰ 覆盖率报告生成超时")
            return False
        except Exception as e:
            print(f"❌ 覆盖率报告生成异常: {e}")
            return False

    def _print_coverage_summary(self, output: str):
        """打印覆盖率摘要"""
        print("\n📊 覆盖率摘要:")

        # 查找覆盖率信息
        lines = output.split('\n')
        coverage_section = False

        for line in lines:
            if "---------- coverage: platform" in line:
                coverage_section = True
                print("   " + line)
            elif coverage_section and line.strip():
                if line.startswith("TOTAL"):
                    print("   " + line)
                    break
                elif "src/data" in line:
                    print("   " + line)

    def _generate_coverage_index(self):
        """生成覆盖率索引文件"""
        index_file = self.data_htmlcov_dir / "index.html"

        if not index_file.exists():
            print("⚠️  覆盖率索引文件未找到，可能需要手动生成")
            return

        print(f"📄 覆盖率索引文件: {index_file}")

        # 检查覆盖率文件
        coverage_files = list(self.data_htmlcov_dir.glob("*.html"))
        if coverage_files:
            print(f"📁 生成的覆盖率文件:")
            for file in coverage_files:
                print(f"   - {file.name}")

    def generate_detailed_coverage(self):
        """生成详细的覆盖率分析"""
        print("\n🔍 生成详细覆盖率分析...")

        # 检查源文件
        source_files = list(self.src_data_dir.rglob("*.py"))
        print(f"📁 发现 {len(source_files)} 个Python源文件:")

        for file in source_files[:10]:  # 只显示前10个
            print(f"   - {file.relative_to(self.project_root)}")

        if len(source_files) > 10:
            print(f"   ... 还有 {len(source_files) - 10} 个文件")

        # 检查测试文件
        test_files = list(self.tests_data_dir.rglob("*.py"))
        print(f"🧪 发现 {len(test_files)} 个测试文件:")

        for file in test_files[:10]:  # 只显示前10个
            print(f"   - {file.relative_to(self.project_root)}")

        if len(test_files) > 10:
            print(f"   ... 还有 {len(test_files) - 10} 个文件")

    def cleanup_old_reports(self):
        """清理旧的覆盖率报告"""
        print("\n🧹 清理旧的覆盖率报告...")

        # 备份旧的报告
        if self.data_htmlcov_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.htmlcov_dir / f"data_backup_{timestamp}"

            try:
                import shutil
                shutil.move(str(self.data_htmlcov_dir), str(backup_dir))
                print(f"✅ 旧报告已备份到: {backup_dir}")
            except Exception as e:
                print(f"⚠️  备份失败: {e}")

    def run(self,
            clean: bool = False,
            parallel: bool = False,
            source_paths: list = None,
            test_paths: list = None):
        """运行覆盖率报告生成"""

        print("=" * 60)
        print("🔍 数据层测试覆盖率报告生成器")
        print("=" * 60)

        # 设置目录
        self.setup_directories()

        # 清理旧报告
        if clean:
            self.cleanup_old_reports()
            self.setup_directories()

        # 生成覆盖率报告
        success = self.generate_coverage_report(
            source_paths=source_paths,
            test_paths=test_paths,
            parallel=parallel
        )

        if success:
            # 生成详细分析
            self.generate_detailed_coverage()

            print("\n" + "=" * 60)
            print("✅ 数据层测试覆盖率报告生成完成!")
            print(f"📁 报告位置: {self.data_htmlcov_dir}")
            print(f"🌐 打开 {self.data_htmlcov_dir / 'index.html'} 查看报告")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("❌ 数据层测试覆盖率报告生成失败!")
            print("=" * 60)

        return success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据层测试覆盖率报告生成器")

    parser.add_argument(
        "--clean",
        action="store_true",
        help="清理旧的覆盖率报告"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用并行测试执行"
    )

    parser.add_argument(
        "--source-paths",
        nargs="+",
        help="指定源文件路径"
    )

    parser.add_argument(
        "--test-paths",
        nargs="+",
        help="指定测试文件路径"
    )

    parser.add_argument(
        "--project-root",
        help="指定项目根目录"
    )

    args = parser.parse_args()

    # 创建生成器实例
    generator = DataLayerCoverageGenerator(project_root=args.project_root)

    # 运行覆盖率报告生成
    success = generator.run(
        clean=args.clean,
        parallel=args.parallel,
        source_paths=args.source_paths,
        test_paths=args.test_paths
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

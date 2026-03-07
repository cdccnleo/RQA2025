#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 缺失依赖库安装脚本
自动检测并安装测试中发现的缺失依赖库
"""

import subprocess
import sys
import importlib
from typing import List, Set


class DependencyInstaller:
    """依赖库安装器"""

    def __init__(self):
        self.missing_packages: Set[str] = set()
        self.installed_packages: Set[str] = set()

    def check_package(self, package_name: str, import_name: str = None) -> bool:
        """检查包是否已安装"""
        if import_name is None:
            import_name = package_name

        try:
            importlib.import_module(import_name)
            self.installed_packages.add(package_name)
            print(f"✅ {package_name} 已安装")
            return True
        except ImportError:
            self.missing_packages.add(package_name)
            print(f"❌ {package_name} 缺失")
            return False

    def check_core_dependencies(self):
        """检查核心依赖库"""
        print("\n=== 检查核心依赖库 ===")

        core_deps = [
            ("boto3", "boto3"),
            ("pymysql", "pymysql"),
            ("psycopg2", "psycopg2"),
            ("influxdb_client", "influxdb_client"),
            ("elasticsearch", "elasticsearch"),
            ("confluent_kafka", "confluent_kafka"),
            ("minio", "minio"),
            ("google.cloud", "google.cloud.storage"),
            ("azure.storage", "azure.storage.blob"),
        ]

        for package_name, import_name in core_deps:
            self.check_package(package_name, import_name)

    def check_optional_dependencies(self):
        """检查可选依赖库"""
        print("\n=== 检查可选依赖库 ===")

        optional_deps = [
            ("ta_lib", "talib"),
            ("yfinance", "yfinance"),
            ("ccxt", "ccxt"),
            ("quantlib", "QuantLib"),
            ("statsmodels", "statsmodels"),
            ("prophet", "prophet"),
            ("shap", "shap"),
            ("optuna", "optuna"),
            ("plotly", "plotly"),
            ("streamlit", "streamlit"),
            ("kubernetes", "kubernetes"),
            ("docker", "docker"),
        ]

        for package_name, import_name in optional_deps:
            self.check_package(package_name, import_name)

    def install_packages(self, packages: List[str]) -> bool:
        """安装包列表"""
        if not packages:
            print("没有需要安装的包")
            return True

        print(f"\n=== 安装缺失的依赖库: {', '.join(packages)} ===")

        try:
            # 使用pip安装
            cmd = [sys.executable, "-m", "pip", "install"] + packages
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print("✅ 依赖库安装成功！")
                return True
            else:
                print(f"❌ 安装失败: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ 安装超时")
            return False
        except Exception as e:
            print(f"❌ 安装过程中出错: {e}")
            return False

    def verify_installation(self) -> bool:
        """验证安装结果"""
        print("\n=== 验证安装结果 ===")

        success_count = 0
        for package in self.missing_packages:
            if package in ["google.cloud", "azure.storage"]:
                # 特殊处理复合包名
                if package == "google.cloud":
                    try:
                        importlib.import_module("google.cloud.storage")
                        print(f"✅ {package} 验证成功")
                        success_count += 1
                    except ImportError:
                        print(f"❌ {package} 验证失败")
                elif package == "azure.storage":
                    try:
                        importlib.import_module("azure.storage.blob")
                        print(f"✅ {package} 验证成功")
                        success_count += 1
                    except ImportError:
                        print(f"❌ {package} 验证失败")
            else:
                # 普通包名处理
                import_name = package.replace("_", "")
                try:
                    importlib.import_module(import_name)
                    print(f"✅ {package} 验证成功")
                    success_count += 1
                except ImportError:
                    print(f"❌ {package} 验证失败")

        total_missing = len(self.missing_packages)
        if success_count == total_missing:
            print(f"\n🎉 所有 {total_missing} 个缺失依赖库安装成功！")
            return True
        else:
            print(f"\n⚠️  {success_count}/{total_missing} 个依赖库安装成功")
            return False

    def generate_install_commands(self):
        """生成安装命令"""
        print("\n=== 生成安装命令 ===")

        if self.missing_packages:
            print("手动安装命令:")
            print(f"pip install {' '.join(self.missing_packages)}")

            # 分组安装命令
            core_packages = []
            optional_packages = []

            for package in self.missing_packages:
                if package in ["boto3", "pymysql", "psycopg2", "influxdb_client",
                               "elasticsearch", "confluent_kafka", "minio"]:
                    core_packages.append(package)
                else:
                    optional_packages.append(package)

            if core_packages:
                print(f"\n核心依赖安装命令:")
                print(f"pip install {' '.join(core_packages)}")

            if optional_packages:
                print(f"\n可选依赖安装命令:")
                print(f"pip install {' '.join(optional_packages)}")
        else:
            print("没有缺失的依赖库")

    def run(self):
        """主运行函数"""
        print("🚀 RQA2025 缺失依赖库检测和安装工具")
        print("=" * 50)

        # 检查核心依赖
        self.check_core_dependencies()

        # 检查可选依赖
        self.check_optional_dependencies()

        # 如果有缺失的包，尝试安装
        if self.missing_packages:
            packages_to_install = list(self.missing_packages)

            # 先尝试安装核心包
            core_packages = [p for p in packages_to_install
                             if p in ["boto3", "pymysql", "psycopg2", "influxdb_client",
                                      "elasticsearch", "confluent_kafka", "minio"]]

            if core_packages:
                print(f"\n尝试安装核心依赖库: {core_packages}")
                if self.install_packages(core_packages):
                    # 移除已成功安装的包
                    for package in core_packages:
                        if package in self.missing_packages:
                            self.missing_packages.remove(package)

            # 安装剩余的包
            remaining_packages = list(self.missing_packages)
            if remaining_packages:
                print(f"\n尝试安装剩余依赖库: {remaining_packages}")
                self.install_packages(remaining_packages)

            # 验证安装结果
            self.verify_installation()

        # 生成安装命令（无论是否安装成功，都提供手动安装命令）
        self.generate_install_commands()

        print("\n" + "=" * 50)
        if self.missing_packages:
            print(f"⚠️  仍有 {len(self.missing_packages)} 个依赖库缺失")
            print("请手动运行上述安装命令或检查网络连接")
        else:
            print("🎉 所有依赖库检查完成！")


def main():
    """主函数"""
    installer = DependencyInstaller()
    installer.run()


if __name__ == "__main__":
    main()

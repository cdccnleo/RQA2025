#!/usr/bin/env python3
"""
基础设施层测试用例重组织脚本

根据基础设施层各子模块及代码架构重新组织测试文件
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List, Set


class TestOrganizer:
    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.cache_tests = []
        self.config_tests = []
        self.error_tests = []
        self.health_tests = []
        self.logging_tests = []
        self.resource_tests = []
        self.monitoring_tests = []
        self.distributed_tests = []
        self.interfaces_tests = []
        self.utils_tests = []
        self.other_tests = []

    def categorize_test_files(self):
        """根据文件名模式分类测试文件"""
        test_files = list(self.test_dir.glob("*.py"))

        # 缓存相关测试文件模式
        cache_patterns = [
            r'.*cache.*', r'.*redis.*', r'.*memory.*', r'.*lru.*',
            r'.*unified_cache.*', r'.*multi_level.*', r'.*smart_cache.*'
        ]

        # 配置相关测试文件模式
        config_patterns = [
            r'.*config.*', r'.*configuration.*', r'.*registry.*'
        ]

        # 错误处理相关测试文件模式
        error_patterns = [
            r'.*error.*', r'.*exception.*', r'.*circuit_breaker.*',
            r'.*retry.*', r'.*boundary.*'
        ]

        # 健康检查相关测试文件模式
        health_patterns = [
            r'.*health.*', r'.*checker.*'
        ]

        # 日志相关测试文件模式
        logging_patterns = [
            r'.*log.*', r'.*logger.*', r'.*logging.*'
        ]

        # 资源管理相关测试文件模式
        resource_patterns = [
            r'.*resource.*', r'.*pool.*', r'.*quota.*',
            r'.*connection.*', r'.*concurrency.*'
        ]

        # 监控相关测试文件模式
        monitoring_patterns = [
            r'.*monitor.*', r'.*alert.*', r'.*metrics.*',
            r'.*performance.*', r'.*system_monitor.*'
        ]

        # 分布式相关测试文件模式
        distributed_patterns = [
            r'.*distributed.*'
        ]

        # 接口相关测试文件模式
        interfaces_patterns = [
            r'.*interfaces.*', r'.*base.*'
        ]

        # 工具相关测试文件模式
        utils_patterns = [
            r'.*utils.*', r'.*async.*', r'.*processor.*',
            r'.*file_system.*', r'.*micro.*'
        ]

        for test_file in test_files:
            if test_file.name == '__init__.py':
                continue

            filename = test_file.name.lower()

            # 检查是否匹配缓存模式
            if any(re.search(pattern, filename) for pattern in cache_patterns):
                self.cache_tests.append(test_file)
                continue

            # 检查是否匹配配置模式
            if any(re.search(pattern, filename) for pattern in config_patterns):
                self.config_tests.append(test_file)
                continue

            # 检查是否匹配错误处理模式
            if any(re.search(pattern, filename) for pattern in error_patterns):
                self.error_tests.append(test_file)
                continue

            # 检查是否匹配健康检查模式
            if any(re.search(pattern, filename) for pattern in health_patterns):
                self.health_tests.append(test_file)
                continue

            # 检查是否匹配日志模式
            if any(re.search(pattern, filename) for pattern in logging_patterns):
                self.logging_tests.append(test_file)
                continue

            # 检查是否匹配资源管理模式
            if any(re.search(pattern, filename) for pattern in resource_patterns):
                self.resource_tests.append(test_file)
                continue

            # 检查是否匹配监控模式
            if any(re.search(pattern, filename) for pattern in monitoring_patterns):
                self.monitoring_tests.append(test_file)
                continue

            # 检查是否匹配分布式模式
            if any(re.search(pattern, filename) for pattern in distributed_patterns):
                self.distributed_tests.append(test_file)
                continue

            # 检查是否匹配接口模式
            if any(re.search(pattern, filename) for pattern in interfaces_patterns):
                self.interfaces_tests.append(test_file)
                continue

            # 检查是否匹配工具模式
            if any(re.search(pattern, filename) for pattern in utils_patterns):
                self.utils_tests.append(test_file)
                continue

            # 其他文件
            self.other_tests.append(test_file)

    def move_test_files(self):
        """移动测试文件到对应的子模块目录"""
        print("开始移动测试文件...")

        # 移动缓存测试文件
        for test_file in self.cache_tests:
            target_dir = self.test_dir / "cache"
            self._move_file(test_file, target_dir)

        # 移动配置测试文件
        for test_file in self.config_tests:
            target_dir = self.test_dir / "config"
            self._move_file(test_file, target_dir)

        # 移动错误处理测试文件
        for test_file in self.error_tests:
            target_dir = self.test_dir / "error"
            self._move_file(test_file, target_dir)

        # 移动健康检查测试文件
        for test_file in self.health_tests:
            target_dir = self.test_dir / "health"
            self._move_file(test_file, target_dir)

        # 移动日志测试文件
        for test_file in self.logging_tests:
            target_dir = self.test_dir / "logging"
            self._move_file(test_file, target_dir)

        # 移动资源管理测试文件
        for test_file in self.resource_tests:
            target_dir = self.test_dir / "resource"
            self._move_file(test_file, target_dir)

        # 移动监控测试文件
        for test_file in self.monitoring_tests:
            target_dir = self.test_dir / "monitoring"
            self._move_file(test_file, target_dir)

        # 移动分布式测试文件
        for test_file in self.distributed_tests:
            target_dir = self.test_dir / "distributed"
            self._move_file(test_file, target_dir)

        # 移动接口测试文件
        for test_file in self.interfaces_tests:
            target_dir = self.test_dir / "interfaces"
            self._move_file(test_file, target_dir)

        # 移动工具测试文件
        for test_file in self.utils_tests:
            target_dir = self.test_dir / "utils"
            self._move_file(test_file, target_dir)

        # 移动其他测试文件
        for test_file in self.other_tests:
            target_dir = self.test_dir / "utils"  # 默认放到utils
            self._move_file(test_file, target_dir)

    def _move_file(self, source_file: Path, target_dir: Path):
        """移动单个文件"""
        target_file = target_dir / source_file.name
        try:
            shutil.move(str(source_file), str(target_file))
            print(f"✅ 移动: {source_file.name} -> {target_dir.name}/")
        except Exception as e:
            print(f"❌ 移动失败 {source_file.name}: {e}")

    def generate_report(self):
        """生成重组织报告"""
        print("\n" + "="*60)
        print("🏗️  基础设施层测试用例重组织报告")
        print("="*60)

        print(f"\n📊 分类统计:")
        print(f"   • 缓存测试: {len(self.cache_tests)} 个文件")
        print(f"   • 配置测试: {len(self.config_tests)} 个文件")
        print(f"   • 错误处理测试: {len(self.error_tests)} 个文件")
        print(f"   • 健康检查测试: {len(self.health_tests)} 个文件")
        print(f"   • 日志测试: {len(self.logging_tests)} 个文件")
        print(f"   • 资源管理测试: {len(self.resource_tests)} 个文件")
        print(f"   • 监控测试: {len(self.monitoring_tests)} 个文件")
        print(f"   • 分布式测试: {len(self.distributed_tests)} 个文件")
        print(f"   • 接口测试: {len(self.interfaces_tests)} 个文件")
        print(f"   • 工具测试: {len(self.utils_tests)} 个文件")
        print(f"   • 其他测试: {len(self.other_tests)} 个文件")

        total_moved = (len(self.cache_tests) + len(self.config_tests) + len(self.error_tests) +
                       len(self.health_tests) + len(self.logging_tests) + len(self.resource_tests) +
                       len(self.monitoring_tests) + len(self.distributed_tests) + len(self.interfaces_tests) +
                       len(self.utils_tests) + len(self.other_tests))

        print(f"\n🎯 总计移动: {total_moved} 个测试文件")

        print("
              📁 新目录结构: "        print("   tests/unit/infrastructure /")
        print("   ├── cache/           # 缓存系统测试")
        print("   ├── config/          # 配置管理测试")
        print("   ├── error/           # 错误处理测试")
        print("   ├── health/          # 健康检查测试")
        print("   ├── logging/         # 日志系统测试")
        print("   ├── resource/        # 资源管理测试")
        print("   ├── monitoring/      # 监控系统测试")
        print("   ├── distributed/     # 分布式系统测试")
        print("   ├── interfaces/      # 接口定义测试")
        print("   └── utils/           # 工具库测试")

        print("
              ✅ 重组织完成！"        print("   测试文件已按基础设施层子模块重新组织")
        print("   每个子模块的测试用例现在位于对应的目录中")

    def run(self):
        """执行重组织流程"""
        print("开始分析测试文件...")
        self.categorize_test_files()

        print("开始移动测试文件...")
        self.move_test_files()

        self.generate_report()


def main():
    test_dir = "tests/unit/infrastructure"
    organizer = TestOrganizer(test_dir)
    organizer.run()


if __name__ == "__main__":
    main()

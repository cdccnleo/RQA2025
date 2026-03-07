#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查测试覆盖率阈值
用于CI/CD流水线中的质量门禁
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional


class CoverageChecker:
    """覆盖率检查器"""

    def __init__(self, coverage_file: str):
        self.coverage_file = Path(coverage_file)
        self.coverage_data = {}

    def parse_coverage_xml(self) -> Dict[str, float]:
        """解析coverage XML文件"""
        if not self.coverage_file.exists():
            print(f"❌ Coverage file not found: {self.coverage_file}")
            return {}

        try:
            tree = ET.parse(self.coverage_file)
            root = tree.getroot()

            # 获取总覆盖率
            total_coverage = 0.0
            if 'line-rate' in root.attrib:
                total_coverage = float(root.attrib['line-rate']) * 100

            # 获取各模块覆盖率
            module_coverages = {}
            for package in root.findall('.//package'):
                package_name = package.get('name', '')
                if 'line-rate' in package.attrib:
                    coverage = float(package.attrib['line-rate']) * 100
                    module_coverages[package_name] = coverage

            self.coverage_data = {
                'total': total_coverage,
                'modules': module_coverages
            }

            return self.coverage_data

        except Exception as e:
            print(f"❌ Error parsing coverage XML: {e}")
            return {}

    def check_threshold(self, threshold: float, module: Optional[str] = None) -> bool:
        """检查覆盖率是否达到阈值"""
        if not self.coverage_data:
            self.parse_coverage_xml()

        if not self.coverage_data:
            return False

        if module:
            # 检查特定模块
            if module in self.coverage_data.get('modules', {}):
                coverage = self.coverage_data['modules'][module]
                return coverage >= threshold
            else:
                print(f"⚠️ Module '{module}' not found in coverage data")
                return False
        else:
            # 检查总覆盖率
            total_coverage = self.coverage_data.get('total', 0.0)
            return total_coverage >= threshold

    def print_report(self):
        """打印覆盖率报告"""
        if not self.coverage_data:
            return

        print("📊 Coverage Report")
        print("=" * 50)
        print(".1f")

        print("\n📦 Module Coverage:")
        for module, coverage in self.coverage_data.get('modules', {}).items():
            status = "✅" if coverage >= 75 else "⚠️" if coverage >= 50 else "❌"
            print(".1f")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Check test coverage threshold')
    parser.add_argument('--coverage-file', required=True, help='Coverage XML file path')
    parser.add_argument('--threshold', type=float, required=True,
                        help='Coverage threshold percentage')
    parser.add_argument('--module', help='Specific module to check (optional)')

    args = parser.parse_args()

    checker = CoverageChecker(args.coverage_file)
    coverage_data = checker.parse_coverage_xml()

    if not coverage_data:
        print("❌ Failed to parse coverage data")
        sys.exit(1)

    # 打印报告
    checker.print_report()

    # 检查阈值
    threshold_met = checker.check_threshold(args.threshold, args.module)

    if threshold_met:
        print(f"\n✅ Coverage threshold met: {args.threshold}%")
        if args.module:
            print(f"   Module: {args.module}")
        sys.exit(0)
    else:
        print(f"\n❌ Coverage threshold not met: {args.threshold}%")
        if args.module:
            print(f"   Module: {args.module}")

        # 显示实际覆盖率
        if args.module and args.module in coverage_data.get('modules', {}):
            actual = coverage_data['modules'][args.module]
        else:
            actual = coverage_data.get('total', 0.0)

        print(".1f")
        sys.exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成覆盖率徽章
用于在README中显示覆盖率状态
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional


class CoverageBadgeGenerator:
    """覆盖率徽章生成器"""

    def __init__(self, coverage_dir: str):
        self.coverage_dir = Path(coverage_dir)
        self.index_file = self.coverage_dir / "htmlcov" / "index.html"

    def parse_coverage_from_html(self) -> Optional[float]:
        """从HTML覆盖率报告中解析覆盖率"""
        if not self.index_file.exists():
            print(f"❌ Coverage HTML file not found: {self.index_file}")
            return None

        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找总覆盖率百分比
            # 通常在HTML中会有类似 "Total: XX.XX%" 的文本
            if "Total:" in content:
                # 查找百分比模式
                import re
                total_match = re.search(r'Total:\s*(\d+\.\d+)%', content)
                if total_match:
                    return float(total_match.group(1))

            # 备用方法：查找任何包含百分比的模式
            percent_match = re.search(r'(\d+\.\d+)%', content)
            if percent_match:
                return float(percent_match.group(1))

        except Exception as e:
            print(f"❌ Error parsing coverage HTML: {e}")

        return None

    def parse_coverage_from_xml(self) -> Optional[float]:
        """从XML覆盖率报告中解析覆盖率"""
        xml_file = self.coverage_dir / "coverage.xml"
        if not xml_file.exists():
            return None

        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_file)
            root = tree.getroot()

            if 'line-rate' in root.attrib:
                return float(root.attrib['line-rate']) * 100

        except Exception as e:
            print(f"❌ Error parsing coverage XML: {e}")

        return None

    def get_coverage_percentage(self) -> Optional[float]:
        """获取覆盖率百分比"""
        # 优先使用HTML报告
        coverage = self.parse_coverage_from_html()
        if coverage is not None:
            return coverage

        # 备用使用XML报告
        coverage = self.parse_coverage_from_xml()
        if coverage is not None:
            return coverage

        print("❌ Could not determine coverage percentage")
        return None

    def get_badge_color(self, coverage: float) -> str:
        """根据覆盖率获取徽章颜色"""
        if coverage >= 90:
            return "brightgreen"
        elif coverage >= 80:
            return "green"
        elif coverage >= 70:
            return "yellowgreen"
        elif coverage >= 60:
            return "yellow"
        elif coverage >= 50:
            return "orange"
        else:
            return "red"

    def generate_badge_data(self, coverage: float) -> Dict:
        """生成徽章数据"""
        color = self.get_badge_color(coverage)

        return {
            "schemaVersion": 1,
            "label": "Infrastructure Coverage",
            "message": ".1f",
            "color": color,
            "style": "flat"
        }

    def save_badge_data(self, output_file: str, coverage: float):
        """保存徽章数据到文件"""
        badge_data = self.generate_badge_data(coverage)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(badge_data, f, indent=2, ensure_ascii=False)

        print(f"✅ Coverage badge data saved to {output_path}")
        print(".1f")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Generate coverage badge')
    parser.add_argument('--coverage-dir', required=True, help='Coverage report directory')
    parser.add_argument('--output', required=True, help='Output JSON file for badge data')

    args = parser.parse_args()

    generator = CoverageBadgeGenerator(args.coverage_dir)
    coverage = generator.get_coverage_percentage()

    if coverage is None:
        print("❌ Failed to get coverage percentage")
        exit(1)

    generator.save_badge_data(args.output, coverage)
    print("🏷️ Badge generation complete!")
    print(f"   Coverage: {coverage: .1f} %")
    print(f"   Badge file: {args.output}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
更新README中的覆盖率状态
"""

import argparse
import json
import re
from pathlib import Path


class ReadmeUpdater:
    """README更新器"""

    def __init__(self, badge_file: str, readme_file: str):
        self.badge_file = Path(badge_file)
        self.readme_file = Path(readme_file)

    def load_badge_data(self) -> dict:
        """加载徽章数据"""
        if not self.badge_file.exists():
            raise FileNotFoundError(f"Badge file not found: {self.badge_file}")

        with open(self.badge_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def update_readme(self, badge_data: dict):
        """更新README文件"""
        if not self.readme_file.exists():
            raise FileNotFoundError(f"README file not found: {self.readme_file}")

        with open(self.readme_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找现有的覆盖率徽章
        badge_pattern = r'!\[Infrastructure Coverage\]\([^)]*\)'

        # 生成新的徽章URL
        coverage = badge_data['message']
        color = badge_data['color']
        badge_url = f"https://img.shields.io/badge/Infrastructure%20Coverage-{coverage}-{color}?style=flat"

        new_badge = f"![Infrastructure Coverage]({badge_url})"

        # 如果找到现有徽章，则替换
        if re.search(badge_pattern, content):
            content = re.sub(badge_pattern, new_badge, content)
            print("✅ Updated existing coverage badge")
        else:
            # 如果没有找到，则在合适的位置添加
            # 查找项目状态部分
            status_pattern = r'(## 📊 项目状态|## 🚀 快速开始|## 📋 功能特性)'

            # 在项目状态部分前添加覆盖率徽章
            replacement = f"## 📊 项目状态\n\n{new_badge}\n\n"

            if re.search(status_pattern, content):
                content = re.sub(status_pattern, replacement, content)
                print("✅ Added new coverage badge to project status section")
            else:
                # 如果找不到合适位置，在文件开头添加
                content = f"# RQA2025\n\n{new_badge}\n\n" + content
                print("✅ Added coverage badge at the beginning of README")

        # 查找并更新覆盖率统计信息
        coverage_stats_pattern = r'(基础设施层覆盖率|Infrastructure.*Coverage).*?(\d+\.\d+)%'

        if re.search(coverage_stats_pattern, content):
            # 提取新的覆盖率数值
            coverage_value = coverage.replace('%', '')

            # 更新覆盖率统计
            content = re.sub(
                coverage_stats_pattern,
                f'Infrastructure Layer Coverage: {coverage_value}%',
                content
            )
            print("✅ Updated coverage statistics in README")

        # 写入更新后的内容
        with open(self.readme_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✅ README updated successfully: {self.readme_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Update README with coverage status')
    parser.add_argument('--badge-file', required=True, help='Badge data JSON file')
    parser.add_argument('--readme-file', required=True, help='README file to update')

    args = parser.parse_args()

    updater = ReadmeUpdater(args.badge_file, args.readme_file)

    try:
        badge_data = updater.load_badge_data()
        print("📊 Badge data loaded:")
        print(f"   Label: {badge_data.get('label')}")
        print(f"   Message: {badge_data.get('message')}")
        print(f"   Color: {badge_data.get('color')}")

        updater.update_readme(badge_data)
        print("\n🎉 README update completed!")

    except Exception as e:
        print(f"❌ Error updating README: {e}")
        exit(1)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
自动化代码审查脚本
"""

from infrastructure_code_review import CodeReviewer


def run_automated_review():
    """运行自动化审查"""
    print("🔍 开始自动化代码审查...")

    reviewer = CodeReviewer()
    reviewer.run_review()

    print("✅ 自动化审查完成")
    return True


if __name__ == "__main__":
    run_automated_review()

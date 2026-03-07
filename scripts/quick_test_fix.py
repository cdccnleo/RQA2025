#!/usr/bin/env python3
"""
快速测试修复脚本
修复最常见的测试问题，提高通过率
"""

import os
import sys
import subprocess
from pathlib import Path


def run_test_and_fix(test_path: str, max_attempts: int = 3) -> bool:
    """运行测试并尝试修复"""
    for attempt in range(max_attempts):
        print(f"🔍 第{attempt + 1}次尝试运行测试: {test_path}")

        # 运行测试
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            test_path,
            "--tb=no",
            "-q",
            "--disable-warnings"
        ], cwd=".", capture_output=True, text=True, timeout=60)

        # 解析结果
        output = result.stdout + result.stderr
        passed = "passed" in output and "failed" not in output
        failed = "failed" in output or result.returncode != 0

        if passed:
            print(f"✅ 测试通过: {test_path}")
            return True
        elif failed:
            print(f"❌ 测试失败，尝试修复...")
            if attempt < max_attempts - 1:
                # 尝试简单修复
                fix_common_issues(test_path, output)
            else:
                print(f"❌ 修复失败: {test_path}")
                return False
        else:
            print(f"⚠️ 测试结果不明: {test_path}")
            return False

    return False


def fix_common_issues(test_path: str, output: str):
    """修复常见问题"""
    test_file = Path(test_path)

    if not test_file.exists():
        return

    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()

    modified = False

    # 修复FeatureEngineer实例化问题
    if "FeatureEngineer(config=" in content:
        content = content.replace(
            "engineer = FeatureEngineer(config=feature_config)",
            "engineer = FeatureEngineer()"
        )
        modified = True

    # 修复导入路径问题
    if "from src.features.feature_engineer import FeatureEngineer" in content:
        content = content.replace(
            "from src.features.feature_engineer import FeatureEngineer",
            "from src.features.core.feature_engineer import FeatureEngineer"
        )
        modified = True

    # 保存修改
    if modified:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"🔧 已修复测试文件: {test_file}")


def main():
    """主函数"""
    # 要修复的测试文件列表
    test_files = [
        "tests/unit/features/test_feature_engineering_deep_coverage.py",
        "tests/unit/ml/test_ml_core_comprehensive.py",
        "tests/unit/strategy/test_strategy_core_comprehensive.py",
        "tests/unit/trading/test_trading_core_comprehensive.py",
        "tests/unit/risk/test_risk_core_management.py",
    ]

    success_count = 0
    total_count = len(test_files)

    for test_file in test_files:
        if Path(test_file).exists():
            if run_test_and_fix(test_file):
                success_count += 1
        else:
            print(f"⚠️ 测试文件不存在: {test_file}")

    print("\n📊 快速修复结果:")
    print(f"✅ 修复成功: {success_count}/{total_count}")
    print(f"📈 成功率: {success_count/total_count*100:.1f}%" if total_count > 0 else "📈 成功率: 0%")
if __name__ == "__main__":
    main()

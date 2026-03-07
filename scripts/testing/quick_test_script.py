#!/usr/bin/env python3
"""
快速测试脚本 - 验证特征层修复效果
"""

import subprocess
import os


def run_quick_test():
    """运行快速测试"""
    print("=== 快速测试特征层修复效果 ===")

    # 设置环境变量
    os.environ['TESTING'] = 'true'
    os.environ['MOCK_EXTERNAL_DEPENDENCIES'] = 'true'

    # 测试1: SignalConfig
    print("\n1. 测试 SignalConfig...")
    try:
        result = subprocess.run([
            'conda', 'run', '-n', 'test', 'python', '-m', 'pytest',
            'tests/unit/features/test_signal_generator.py::TestSignalConfig::test_signal_config_defaults',
            '-v', '--tb=short'
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ SignalConfig 测试通过")
        else:
            print(f"❌ SignalConfig 测试失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ SignalConfig 测试超时")
    except Exception as e:
        print(f"❌ SignalConfig 测试异常: {e}")

    # 测试2: TechnicalProcessor
    print("\n2. 测试 TechnicalProcessor...")
    try:
        result = subprocess.run([
            'conda', 'run', '-n', 'test', 'python', '-m', 'pytest',
            'tests/unit/features/processors/test_technical.py::TestTechnicalProcessor::test_calculate_rsi_basic',
            '-v', '--tb=short'
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ TechnicalProcessor 测试通过")
        else:
            print(f"❌ TechnicalProcessor 测试失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ TechnicalProcessor 测试超时")
    except Exception as e:
        print(f"❌ TechnicalProcessor 测试异常: {e}")

    # 测试3: FeatureSelector
    print("\n3. 测试 FeatureSelector...")
    try:
        result = subprocess.run([
            'conda', 'run', '-n', 'test', 'python', '-m', 'pytest',
            'tests/unit/features/test_feature_selector.py::TestFeatureSelector::test_initialization',
            '-v', '--tb=short'
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ FeatureSelector 测试通过")
        else:
            print(f"❌ FeatureSelector 测试失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ FeatureSelector 测试超时")
    except Exception as e:
        print(f"❌ FeatureSelector 测试异常: {e}")

    # 测试4: FeatureConfig
    print("\n4. 测试 FeatureConfig...")
    try:
        result = subprocess.run([
            'conda', 'run', '-n', 'test', 'python', '-m', 'pytest',
            'tests/unit/features/test_feature_config.py::test_feature_config_init_and_validate',
            '-v', '--tb=short'
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ FeatureConfig 测试通过")
        else:
            print(f"❌ FeatureConfig 测试失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ FeatureConfig 测试超时")
    except Exception as e:
        print(f"❌ FeatureConfig 测试异常: {e}")

    print("\n=== 快速测试完成 ===")


if __name__ == "__main__":
    run_quick_test()

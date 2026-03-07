#!/usr/bin/env python3
"""
覆盖率配置验证脚本
检查pytest.ini、.coveragerc等配置文件是否正确设置
"""

import sys
import configparser
from pathlib import Path
import subprocess


def check_pytest_ini():
    """检查pytest.ini配置"""
    print("🔍 检查 pytest.ini 配置...")

    pytest_ini_path = Path("pytest.ini")
    if not pytest_ini_path.exists():
        print("❌ pytest.ini 文件不存在")
        return False

    config = configparser.ConfigParser()
    config.read(pytest_ini_path)

    if 'tool:pytest' not in config:
        print("❌ 缺少 [tool:pytest] 节")
        return False

    addopts = config.get('tool:pytest', 'addopts', fallback='')

    # 检查覆盖率配置
    coverage_checks = [
        '--cov=src',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov'
    ]

    missing_options = []
    for option in coverage_checks:
        if option not in addopts:
            missing_options.append(option)

    if missing_options:
        print(f"❌ 缺少覆盖率配置: {', '.join(missing_options)}")
        return False

    print("✅ pytest.ini 覆盖率配置正确")
    return True


def check_coveragerc():
    """检查.coveragerc配置"""
    print("🔍 检查 .coveragerc 配置...")

    coveragerc_path = Path(".coveragerc")
    if not coveragerc_path.exists():
        print("❌ .coveragerc 文件不存在")
        return False

    config = configparser.ConfigParser()
    config.read(coveragerc_path)

    # 检查必要的节
    required_sections = ['run', 'report']
    for section in required_sections:
        if section not in config:
            print(f"❌ 缺少 [{section}] 节")
            return False

    # 检查run节配置
    if 'source' not in config['run']:
        print("❌ [run] 节缺少 source 配置")
        return False

    if config['run']['source'] != 'src':
        print(f"❌ source 配置错误: {config['run']['source']}, 应为 'src'")
        return False

    # 检查report节配置
    if 'fail_under' not in config['report']:
        print("❌ [report] 节缺少 fail_under 配置")
        return False

    try:
        fail_under = float(config['report']['fail_under'])
        if fail_under < 0 or fail_under > 100:
            print(f"❌ fail_under 值无效: {fail_under}")
            return False
    except ValueError:
        print("❌ fail_under 值不是有效数字")
        return False

    print("✅ .coveragerc 配置正确")
    return True


def check_pytest_cov_installation():
    """检查pytest-cov是否正确安装"""
    print("🔍 检查 pytest-cov 安装...")

    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', '--help'
        ], capture_output=True, text=True, timeout=10)

        if '--cov' in result.stdout:
            print("✅ pytest-cov 插件已正确安装")
            return True
        else:
            print("❌ pytest-cov 插件未安装或未激活")
            return False

    except subprocess.TimeoutExpired:
        print("❌ pytest 命令执行超时")
        return False
    except Exception as e:
        print(f"❌ 检查 pytest-cov 时出错: {e}")
        return False


def check_coverage_files():
    """检查覆盖率相关文件"""
    print("🔍 检查覆盖率相关文件...")

    required_files = [
        'pytest.ini',
        '.coveragerc'
    ]

    optional_files = [
        'pytest_coverage.ini',
        'htmlcov/',
        'coverage.xml'
    ]

    missing_required = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_required.append(file_path)

    if missing_required:
        print(f"❌ 缺少必要文件: {', '.join(missing_required)}")
        return False

    print("✅ 覆盖率相关文件完整")

    # 检查可选文件
    existing_optional = []
    for file_path in optional_files:
        if Path(file_path).exists():
            existing_optional.append(file_path)

    if existing_optional:
        print(f"📁 发现可选文件: {', '.join(existing_optional)}")

    return True


def run_test_coverage():
    """运行测试覆盖率检查"""
    print("🔍 运行测试覆盖率检查...")

    try:
        # 运行一个简单的测试来验证覆盖率收集
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            'tests/unit/infrastructure/test_logging.py',  # 使用一个简单的测试文件
            '--cov=src/infrastructure/logging',
            '--cov-report=term-missing',
            '-v'
        ], capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print("✅ 测试覆盖率收集正常")

            # 检查输出中是否包含覆盖率信息
            if 'TOTAL' in result.stdout or 'coverage:' in result.stdout:
                print("✅ 覆盖率数据已收集")
                return True
            else:
                print("⚠️  测试通过但未收集到覆盖率数据")
                return False
        else:
            print(f"❌ 测试执行失败: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("❌ 测试执行超时")
        return False
    except Exception as e:
        print(f"❌ 运行测试覆盖率检查时出错: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("🧪 覆盖率配置验证工具")
    print("=" * 60)

    checks = [
        check_pytest_ini,
        check_coveragerc,
        check_pytest_cov_installation,
        check_coverage_files,
        run_test_coverage
    ]

    passed = 0
    total = len(checks)

    for check in checks:
        try:
            if check():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ 检查 {check.__name__} 时出错: {e}")
            print()

    print("=" * 60)
    print(f"📊 验证结果: {passed}/{total} 通过")

    if passed == total:
        print("🎉 所有覆盖率配置检查通过！")
        return 0
    else:
        print("⚠️  存在配置问题，请根据上述提示修复")
        return 1


if __name__ == "__main__":
    sys.exit(main())
